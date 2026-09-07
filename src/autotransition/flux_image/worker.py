from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .artifacts import FluxImageArtifactStore, utc_now
from .config import FluxImageConfig
from .contracts import FluxImageFailure, FluxImageJob, FluxImageRequest, FluxLoRARequest
from .observability import FluxImageEventLogger
from .framing import frame_avatar_image
from .runtime import FluxImageRuntime


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class FluxImageWorker:
    """One FLUX GPU job at a time with durable terminal state."""

    def __init__(self, config: FluxImageConfig, runtime: FluxImageRuntime | None = None):
        self.config = config
        self.store = FluxImageArtifactStore(config.artifact_root)
        self.runtime = runtime or FluxImageRuntime(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux-image-worker")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    async def submit(self, request: FluxImageRequest) -> FluxImageJob:
        request.validate(self.config)
        if request.external_job_id:
            try:
                existing = self.store.read_job(request.external_job_id)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                existing_request = existing.get("request") or {}
                if existing_request.get("payment_intent_id") != request.payment_intent_id:
                    raise HTTPException(status_code=409, detail="external job id is already bound to another payment intent")
                return self._job_from_dict(existing)
        job_id = request.external_job_id or uuid4().hex
        now = utc_now()
        job = FluxImageJob(id=job_id, status="queued", request=request.to_dict(), created_at=now, updated_at=now)
        self.store.create_job(job)
        with self._lock:
            self._futures[job.id] = self.executor.submit(self._run, request, job.id)
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        return self.store.read_job(job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _set_state(self, job_id: str, **fields: Any) -> dict[str, Any]:
        payload = self.store.read_job(job_id)
        payload.update(fields)
        payload["updated_at"] = utc_now()
        self.store._atomic_json(self.store.job_dir(job_id) / "job.json", payload)
        return payload

    def _event(self, job_id: str) -> FluxImageEventLogger:
        return FluxImageEventLogger(job_id, self.store.job_dir(job_id) / "events.jsonl")

    def _run(self, request: FluxImageRequest, job_id: str) -> None:
        logger = self._event(job_id)
        try:
            self.runtime.emit = logger.emit
            logger.emit("job_started", request=request.to_dict())
            self._set_state(job_id, status="running", attempt=1, stage="validate_request", progress=0.01)

            def progress(stage: str, fraction: float, message: str) -> None:
                value = max(0.0, min(1.0, float(fraction)))
                self._set_state(job_id, status="running", stage=stage, progress=value, message=message)
                logger.emit("progress", stage=stage, progress=value, message=message)

            attempt_dir = self.store.attempt_dir(job_id, 1)
            output_path = attempt_dir / "image.png"
            metadata_path = attempt_dir / "image-metadata.json"
            logger.emit("runtime_preflight_started")
            progress("load_model", 0.0, "Checking FLUX.2 Klein model assets")
            result = self.runtime.generate(request, output_path, progress)
            if not output_path.is_file() or output_path.stat().st_size < 100:
                raise RuntimeError("FLUX runtime returned an empty or invalid PNG")
            raw_output_path = attempt_dir / "model-output.png"
            output_path.replace(raw_output_path)
            progress(
                "frame_output",
                0.0,
                f"Framing avatar on the canonical {self.config.avatar_output_width}x{self.config.avatar_output_height} canvas",
            )
            framing = frame_avatar_image(
                raw_output_path,
                output_path,
                width=self.config.avatar_output_width,
                height=self.config.avatar_output_height,
                subject_scale=self.config.avatar_subject_scale,
            )
            logger.emit("avatar_framed", **framing.to_dict())
            progress("frame_output", 0.98, "Avatar framing complete")
            final_image = self.store.finalize_file(job_id, output_path, "image.png")
            metadata = {
                "schemaVersion": 1,
                "runtime": "flux-image",
                "modelRevision": "Flux2-Klein-4B",
                "request": request.to_dict(),
                "effective": {**result, "framing": framing.to_dict()},
                "output": {
                    "name": final_image.name,
                    "sizeBytes": final_image.stat().st_size,
                    "sha256": self.store.sha256(final_image),
                },
                "completedAt": utc_now(),
            }
            self.store.finalize_json(job_id, "image-metadata.json", metadata)
            image_artifact = self.store.artifact(job_id, "image.png", "image/png")
            metadata_artifact = self.store.artifact(job_id, "image-metadata.json", "application/json")
            self._set_state(
                job_id,
                status="succeeded",
                stage="finalizing",
                progress=1.0,
                artifacts=[image_artifact.__dict__, metadata_artifact.__dict__],
            )
            logger.emit("job_succeeded", artifacts=[image_artifact.name, metadata_artifact.name])
        except Exception as exc:
            logger.exception("job_failed", exc)
            current = self.store.read_job(job_id)
            failure = FluxImageFailure(
                code="flux_image_worker_failed",
                message=str(exc) or type(exc).__name__,
                stage=current.get("stage") or "validate_request",
                retryable=False,
                attempt=int(current.get("attempt") or 1),
                details={"errorType": type(exc).__name__, "traceback": traceback.format_exc()},
            )
            failure_summary = {
                "schemaVersion": 1,
                "runtime": "flux-image",
                "jobId": job_id,
                "request": request.to_dict(),
                "failure": failure.to_dict(),
                "eventLog": "events.jsonl",
                "createdAt": utc_now(),
            }
            self.store.finalize_json(job_id, "failure-summary.json", failure_summary)
            failure_artifact = self.store.artifact(job_id, "failure-summary.json", "application/json")
            event_artifact = self.store.artifact(job_id, "events.jsonl", "application/jsonl")
            self._set_state(
                job_id,
                status="failed",
                progress=1.0,
                failure=failure.to_dict(),
                failureCode=failure.code,
                refundRequired=True,
                refundReason="flux_image_generation_failed",
                artifacts=[failure_artifact.__dict__, event_artifact.__dict__],
            )
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
            logger.emit("job_thread_finished")

    @staticmethod
    def _job_from_dict(payload: dict[str, Any]) -> FluxImageJob:
        failure = payload.get("failure")
        return FluxImageJob(
            id=str(payload["id"]),
            status=payload.get("status", "queued"),
            request=payload.get("request") or {},
            stage=payload.get("stage"),
            progress=float(payload.get("progress") or 0),
            attempt=int(payload.get("attempt") or 0),
            artifacts=[],
            failure=FluxImageFailure(**failure) if isinstance(failure, dict) else None,
            created_at=str(payload.get("created_at") or payload.get("createdAt") or ""),
            updated_at=str(payload.get("updated_at") or payload.get("updatedAt") or ""),
        )


def create_flux_image_worker_app(config: FluxImageConfig | None = None) -> FastAPI:
    config = config or FluxImageConfig.from_env()
    store = FluxImageArtifactStore(config.artifact_root)
    interrupted = store.reconcile_interrupted_jobs()
    runtime = FluxImageRuntime(config)
    print(
        json.dumps(
            {
                "event": "flux_image_worker_starting",
                "runtime": "flux-image",
                "modelRevision": "Flux2-Klein-4B",
                "config": config.to_public_dict(),
                "preflight": runtime.preflight(),
                "interruptedJobs": interrupted,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )
    worker = FluxImageWorker(config, runtime=runtime)
    app = FastAPI(title="FLUX.2 Klein Image Worker")
    from .salad_adapter import install_salad_routes

    install_salad_routes(app, worker, config)
    app.state.flux_image_worker = worker

    @app.get("/health")
    async def health() -> dict[str, Any]:
        report = runtime.preflight()
        return {"ok": bool(report["ready"]), "runtime": "flux-image", "model": "Flux2-Klein-4B", "preflight": report}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        report = runtime.preflight()
        if not report["ready"]:
            raise HTTPException(status_code=503, detail=report)
        return {"ok": True, "runtime": "flux-image", "model": "Flux2-Klein-4B", "preflight": report}

    @app.get("/v1/worker/status")
    async def status() -> dict[str, Any]:
        report = runtime.preflight()
        with worker._lock:
            active = list(worker._futures)
        return {"runtime": "flux-image", "model": "Flux2-Klein-4B", "preflight": report, "activeJobs": active}

    @app.post("/v1/flux/jobs")
    async def submit(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = request_from_payload(payload)
            job = await worker.submit(request)
            return worker.get(job.id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/flux/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return worker.get(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def request_from_payload(payload: dict[str, Any]) -> FluxImageRequest:
    loras_value = payload.get("loras") or []
    if not isinstance(loras_value, list):
        raise ValueError("loras must be an array")
    loras = tuple(
        FluxLoRARequest(
            source_url=str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "local://adapter"),
            file_name=Path(str(item.get("fileName") or item.get("file_name") or "adapter.safetensors")).name,
            scale=float(item.get("scale", 1.0)),
            path=Path(str(item["path"])).expanduser() if item.get("path") else None,
        )
        for item in loras_value
        if isinstance(item, dict)
    )
    return FluxImageRequest(
        prompt=str(payload.get("prompt") or ""),
        negative_prompt=str(payload.get("negative_prompt") or payload.get("negativePrompt") or ""),
        width=int(payload.get("width", 960)),
        height=int(payload.get("height", 1664)),
        steps=int(payload.get("steps", 4)),
        true_cfg_scale=float(payload.get("true_cfg_scale", payload.get("trueCfgScale", 1.0))),
        seed=int(payload["seed"]) if payload.get("seed") is not None else None,
        loras=loras,
        external_job_id=str(payload.get("job_id") or payload.get("external_job_id") or "") or None,
        payment_intent_id=str(payload.get("payment_intent_id") or "") or None,
    )

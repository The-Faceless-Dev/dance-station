from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from .artifacts import LtxArtifactStore, utc_now
from .config import LtxVideoConfig
from .contracts import LtxArtifact, LtxFailure, LtxVideoJob, LtxVideoRequest
from .runtime import LtxRuntimeError, LtxVideoRuntime, create_runtime


TERMINAL_STATUSES = {"succeeded", "failed"}

# These weights reflect the expensive work in the actual pipeline. The value in
# job.json is a whole-job percentage; stageProgress remains available for
# diagnosing the exact sampler step or media operation.
STAGE_WEIGHTS = {
    "validate_request": 0.015,
    "preflight_memory": 0.025,
    "acquire_inputs": 0.03,
    "normalize_inputs": 0.03,
    "load_models": 0.12,
    "encode_prompt": 0.05,
    "stage_1_denoise": 0.39,
    "decode_stage_1": 0.01,
    "spatial_upscale": 0.06,
    "stage_2_refine": 0.18,
    "decode_video": 0.04,
    "decode_audio": 0.01,
    "mux_audio": 0.01,
    "encode_output": 0.015,
    "finalize": 0.005,
    "upload_artifacts": 0.01,
}
STAGE_OFFSETS: dict[str, float] = {}
_stage_offset = 0.0
for _stage_name, _stage_weight in STAGE_WEIGHTS.items():
    STAGE_OFFSETS[_stage_name] = _stage_offset
    _stage_offset += _stage_weight


class LtxVideoWorker:
    """Serialize LTX jobs and keep all request/output state on disk."""

    def __init__(self, config: LtxVideoConfig, runtime: LtxVideoRuntime | None = None):
        self.config = config
        self.store = LtxArtifactStore(config.artifact_root)
        self.runtime = runtime or create_runtime(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx-video-worker")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    async def submit(self, request: LtxVideoRequest) -> LtxVideoJob:
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
        readiness = self.runtime.preflight()
        if not readiness.get("ready"):
            raise HTTPException(status_code=503, detail={
                "message": "LTX video worker is not ready",
                "preflight": readiness,
            })
        job_id = request.external_job_id or uuid4().hex
        now = utc_now()
        job = LtxVideoJob(id=job_id, status="queued", request=request.to_dict(), created_at=now, updated_at=now)
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

    @staticmethod
    def _overall_progress(stage: str, fraction: float) -> float:
        weight = STAGE_WEIGHTS.get(stage)
        if weight is None:
            return max(0.0, min(1.0, fraction))
        return max(0.0, min(1.0, STAGE_OFFSETS[stage] + weight * max(0.0, min(1.0, fraction))))

    @staticmethod
    def _console_event(event: dict[str, Any]) -> None:
        # Avoid printing signed input URLs or the full request payload. The
        # complete event remains in the durable per-job JSONL artifact.
        print(json.dumps({"event": "ltx_worker", **event}, sort_keys=True, default=str), flush=True)

    def _run(self, request: LtxVideoRequest, job_id: str) -> None:
        started = time.monotonic()
        attempt_dir = self.store.attempt_dir(job_id, 1)
        self.store.write_event(job_id, {"event": "job_started", "request": request.to_dict()})
        self._console_event({"eventType": "job_started", "jobId": job_id, "attempt": 1})

        def progress(stage: str, fraction: float, message: str, details: dict[str, Any] | None = None) -> None:
            if time.monotonic() - started > self.config.job_timeout_seconds:
                raise TimeoutError(f"LTX video job exceeded the {self.config.job_timeout_seconds:.0f}s worker timeout")
            overall = self._overall_progress(stage, fraction)
            stage_details = {"stageProgress": fraction, "overallProgress": overall, **(details or {})}
            payload = self._set_state(
                job_id,
                status="running",
                stage=stage,
                progress=overall,
                stageProgress=fraction,
                message=message,
            )
            self.store.write_event(job_id, {"event": "progress", "stage": stage, "progress": overall, "stageProgress": fraction, "message": message, "details": stage_details})
            self._console_event({
                "eventType": "progress",
                "jobId": job_id,
                "stage": stage,
                "progress": round(overall, 6),
                "stageProgress": round(fraction, 6),
                "message": message,
                "details": stage_details,
            })
            if details:
                self.store.write_event(job_id, {"event": "stage_details", "stage": stage, "details": details})
            del payload

        try:
            self._set_state(job_id, status="running", attempt=1, stage="validate_request", progress=0.01, message="Validating LTX video request")
            progress("validate_request", 1.0, "LTX video request validated")
            runtime_result = self.runtime.generate(request, attempt_dir, progress)
            progress("upload_artifacts", 0.0, "Preparing durable LTX artifacts")
            self.store.finalize_file(job_id, runtime_result.video_path, runtime_result.video_path.name,)
            for intermediate in runtime_result.intermediate_paths:
                self.store.finalize_file(job_id, intermediate, intermediate.name)
            self.store.finalize_json(job_id, "generation-metadata.json", runtime_result.metadata)
            conditioning_manifest = attempt_dir / "conditioning-manifest.json"
            if conditioning_manifest.is_file():
                self.store.finalize_file(job_id, conditioning_manifest, conditioning_manifest.name)
            self.store.finalize_json(job_id, "request.json", request.to_dict())
            if runtime_result.audio_path and runtime_result.audio_path.is_file() and request.audio_mode == "generated":
                self.store.finalize_file(job_id, runtime_result.audio_path, "audio.wav")
            self.store.finalize_json(
                job_id,
                "memory-plan.json",
                runtime_result.metadata.get("memoryPlan", {}),
            )
            self.store.write_event(job_id, {"event": "artifacts_finalized", "output": runtime_result.video_path.name})
            self._console_event({"eventType": "artifacts_finalized", "jobId": job_id, "output": runtime_result.video_path.name})
            events_path = self.store.job_dir(job_id) / "events.jsonl"
            self.store.finalize_file(job_id, events_path, "events.jsonl")
            artifact_names: list[tuple[str, str, str]] = [
                (runtime_result.video_path.name, "video/mp4" if runtime_result.video_path.suffix == ".mp4" else "video/webm", "primary"),
                ("generation-metadata.json", "application/json", "metadata"),
                ("request.json", "application/json", "metadata"),
                ("memory-plan.json", "application/json", "diagnostic"),
                ("events.jsonl", "application/jsonl", "diagnostic"),
            ]
            artifact_names.extend((path.name, "video/mp4", "intermediate") for path in runtime_result.intermediate_paths if self.store.final_path(job_id, path.name).is_file())
            if self.store.final_path(job_id, "conditioning-manifest.json").is_file():
                artifact_names.append(("conditioning-manifest.json", "application/json", "metadata"))
            if request.audio_mode == "generated" and (self.store.final_path(job_id, "audio.wav")).is_file():
                artifact_names.append(("audio.wav", "audio/wav", "audio"))
            artifacts = [self.store.artifact(job_id, name, media_type, role).__dict__ for name, media_type, role in artifact_names]
            self.store.write_event(job_id, {"event": "job_succeeded", "elapsedSeconds": round(time.monotonic() - started, 3), "artifacts": [item["name"] for item in artifacts]})
            self._console_event({
                "eventType": "job_succeeded",
                "jobId": job_id,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "artifacts": [item["name"] for item in artifacts],
            })
            # Include the final success event in the durable event artifact.
            self.store.finalize_file(job_id, events_path, "events.jsonl")
            artifacts = [self.store.artifact(job_id, name, media_type, role).__dict__ for name, media_type, role in artifact_names]
            self._set_state(job_id, status="succeeded", stage="finalize", progress=1.0, message="LTX video generation completed", artifacts=artifacts)
        except Exception as exc:
            current = self.store.read_job(job_id)
            if isinstance(exc, LtxRuntimeError):
                code, stage, details = exc.code, exc.stage, exc.details
            else:
                code, stage, details = ("ltx_video_worker_failed", current.get("stage") or "validate_request", {})
            failure = LtxFailure(
                code=code,
                message=str(exc) or type(exc).__name__,
                stage=stage,
                retryable=False,
                attempt=int(current.get("attempt") or 1),
                details={"errorType": type(exc).__name__, "traceback": traceback.format_exc(), **details},
            )
            self.store.write_event(job_id, {"event": "job_failed", "failure": failure.to_dict()})
            self._console_event({
                "eventType": "job_failed",
                "jobId": job_id,
                "stage": stage,
                "errorType": type(exc).__name__,
                "code": code,
                "message": str(exc),
            })
            self.store.finalize_json(
                job_id,
                "failure-summary.json",
                {
                    "schemaVersion": 1,
                    "runtime": "ltx-video",
                    "jobId": job_id,
                    "request": request.to_dict(),
                    "failure": failure.to_dict(),
                    "eventLog": "events.jsonl",
                    "createdAt": utc_now(),
                },
            )
            events_path = self.store.job_dir(job_id) / "events.jsonl"
            if events_path.is_file():
                self.store.finalize_file(job_id, events_path, "events.jsonl")
            artifacts: list[dict[str, Any]] = []
            for name, media_type, role in (("failure-summary.json", "application/json", "diagnostic"), ("events.jsonl", "application/jsonl", "diagnostic")):
                if self.store.final_path(job_id, name).is_file():
                    artifacts.append(self.store.artifact(job_id, name, media_type, role).__dict__)
            self._set_state(
                job_id,
                status="failed",
                stage=stage,
                progress=float(current.get("progress") or 0.0),
                stageProgress=float(current.get("stageProgress") or 0.0),
                message=str(exc),
                failure=failure.to_dict(),
                artifacts=artifacts,
            )
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    @staticmethod
    def _job_from_dict(payload: dict[str, Any]) -> LtxVideoJob:
        artifacts = [LtxArtifact(**item) for item in payload.get("artifacts") or [] if isinstance(item, dict)]
        failure_value = payload.get("failure")
        failure = LtxFailure(**failure_value) if isinstance(failure_value, dict) else None
        return LtxVideoJob(
            id=str(payload["id"]),
            status=payload.get("status", "failed"),
            request=payload.get("request") or {},
            stage=payload.get("stage"),
            progress=float(payload.get("progress") or 0),
            attempt=int(payload.get("attempt") or 0),
            message=payload.get("message"),
            artifacts=artifacts,
            failure=failure,
            created_at=str(payload.get("created_at") or payload.get("createdAt") or ""),
            updated_at=str(payload.get("updated_at") or payload.get("updatedAt") or ""),
        )

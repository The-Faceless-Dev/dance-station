from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException
from PIL import Image

from .artifacts import QwenImageEditArtifactStore, utc_now
from .config import QwenImageEditConfig
from .contracts import QwenImageEditFailure, QwenImageEditRequest
from .lora import cleanup_loras, prepare_loras
from .observability import QwenImageEditEventLogger
from .references import cleanup_references, prepare_references
from .runtime import QwenImageEditDiffusersRuntime


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class QwenImageEditWorker:
    def __init__(self, config: QwenImageEditConfig, runtime: Any | None = None):
        self.config = config
        self.store = QwenImageEditArtifactStore(config.artifact_root)
        self.runtime = runtime or QwenImageEditDiffusersRuntime(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-image-edit-worker")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    async def submit(self, request: QwenImageEditRequest) -> dict[str, Any]:
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
                return existing
        job_id = request.external_job_id or uuid4().hex
        now = utc_now()
        job = {
            "id": job_id,
            "status": "queued",
            "request": request.to_dict(),
            "stage": None,
            "progress": 0.0,
            "attempt": 0,
            "artifacts": [],
            "failure": None,
            "created_at": now,
            "updated_at": now,
        }
        self.store.create_job(job)
        with self._lock:
            self._futures[job_id] = self.executor.submit(self._run, request, job_id)
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        return self.store.read_job(job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.runtime.close()

    def _set_state(self, job_id: str, **fields: Any) -> dict[str, Any]:
        payload = self.store.read_job(job_id)
        payload.update(fields)
        payload["updated_at"] = utc_now()
        self.store._atomic_json(self.store.job_dir(job_id) / "job.json", payload)
        return payload

    def _event(self, job_id: str) -> QwenImageEditEventLogger:
        return QwenImageEditEventLogger(job_id, self.store.job_dir(job_id) / "events.jsonl")

    def _run(self, request: QwenImageEditRequest, job_id: str) -> None:
        logger = self._event(job_id)
        reference_directory = self.store.temp_dir(job_id) / "references"
        lora_directory = self.store.temp_dir(job_id) / "loras"
        try:
            self.runtime.emit = logger.emit
            logger.emit("job_started", request=request.to_dict())
            self._set_state(job_id, status="running", attempt=1, stage="validate_request", progress=0.01)

            def progress(stage: str, fraction: float, message: str) -> None:
                value = max(0.0, min(1.0, float(fraction)))
                self._set_state(job_id, status="running", stage=stage, progress=value, message=message)
                logger.emit("progress", stage=stage, progress=value, message=message)

            progress("model_preflight", 0.0, "Checking the Qwen Image Edit 2511 Diffusers runtime and CUDA device")
            preflight = self.runtime.preflight()
            self.store.finalize_json(job_id, "preflight.json", preflight)
            logger.emit("model_preflight_finished", report=preflight)
            if not preflight.get("ready"):
                raise RuntimeError("Qwen Image Edit preflight failed: " + json.dumps(preflight, sort_keys=True))

            progress("prepare_references", 0.04, f"Preparing {len(request.references)} reference image(s)")
            references = prepare_references(request.references, reference_directory, self.config, logger.emit)
            self.store.finalize_json(job_id, "reference-metadata.json", [item.to_dict() for item in references])
            progress("prepare_loras", 0.07, f"Preparing {len(request.loras)} LoRA adapter(s)")
            prepared_loras = prepare_loras(request.loras, lora_directory, self.config, logger.emit)
            progress("load_model", 0.0, "Starting the persistent CUDA Qwen Image Edit runtime")
            attempt_dir = self.store.attempt_dir(job_id, 1)
            output_path = attempt_dir / "image.png"
            result = self.runtime.generate(request, references, prepared_loras, output_path, progress)
            if not output_path.is_file() or output_path.stat().st_size < 100:
                raise RuntimeError("Qwen Image Edit runtime returned an empty or invalid PNG")
            with Image.open(output_path) as image:
                if image.format != "PNG" or image.size != (request.width, request.height):
                    raise RuntimeError(f"generated PNG validation failed: format={image.format} size={image.size}")
            progress("validate_output", 0.92, "Validating generated PNG")
            final_image = self.store.finalize_file(job_id, output_path, "image.png")
            metadata = {
                "schemaVersion": 1,
                "runtime": "qwen-image-edit-2511",
                "modelRevision": preflight.get("modelRevision", self.config.model_name),
                "request": request.to_dict(),
                "effective": result,
                "preflight": preflight,
                "references": [item.to_dict() for item in references],
                "output": {
                    "name": final_image.name,
                    "sizeBytes": final_image.stat().st_size,
                    "sha256": self.store.sha256(final_image),
                },
                "completedAt": utc_now(),
            }
            self.store.finalize_json(job_id, "image-metadata.json", metadata)
            progress("finalizing", 0.98, "Finalizing durable Qwen Image Edit artifacts")
            logger.emit("job_succeeded", output=str(final_image))
            self.store.finalize_file(job_id, self.store.job_dir(job_id) / "events.jsonl", "events.jsonl")
            artifacts = [
                self.store.artifact(job_id, "image.png", "image/png"),
                self.store.artifact(job_id, "image-metadata.json", "application/json"),
                self.store.artifact(job_id, "reference-metadata.json", "application/json"),
                self.store.artifact(job_id, "preflight.json", "application/json"),
                self.store.artifact(job_id, "events.jsonl", "application/jsonl"),
            ]
            self._set_state(
                job_id,
                status="succeeded",
                stage="finalizing",
                progress=1.0,
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.exception("job_failed", exc)
            current = self.store.read_job(job_id)
            failure = QwenImageEditFailure(
                code="qwen_image_edit_worker_failed",
                message=str(exc) or type(exc).__name__,
                stage=current.get("stage") or "validate_request",
                retryable=False,
                attempt=int(current.get("attempt") or 1),
                details={"errorType": type(exc).__name__, "traceback": traceback.format_exc()},
            )
            self.store.finalize_json(job_id, "failure-summary.json", {
                "schemaVersion": 1,
                "runtime": "qwen-image-edit-2511",
                "jobId": job_id,
                "request": request.to_dict(),
                "failure": failure.to_dict(),
                "eventLog": "events.jsonl",
                "createdAt": utc_now(),
            })
            event_log = self.store.job_dir(job_id) / "events.jsonl"
            if event_log.is_file():
                self.store.finalize_file(job_id, event_log, "events.jsonl")
            artifacts = [self.store.artifact(job_id, "failure-summary.json", "application/json")]
            if (self.store.job_dir(job_id) / "final" / "preflight.json").is_file():
                artifacts.append(self.store.artifact(job_id, "preflight.json", "application/json"))
            if (self.store.job_dir(job_id) / "final" / "reference-metadata.json").is_file():
                artifacts.append(self.store.artifact(job_id, "reference-metadata.json", "application/json"))
            if (self.store.job_dir(job_id) / "final" / "events.jsonl").is_file():
                artifacts.append(self.store.artifact(job_id, "events.jsonl", "application/jsonl"))
            self._set_state(
                job_id,
                status="failed",
                progress=1.0,
                failure=failure.to_dict(),
                failureCode=failure.code,
                refundRequired=True,
                refundReason="qwen_image_edit_generation_failed",
                artifacts=artifacts,
            )
        finally:
            cleanup_references(reference_directory, logger.emit)
            cleanup_loras(lora_directory, logger.emit)
            logger.emit("job_thread_finished")
            event_log = self.store.job_dir(job_id) / "events.jsonl"
            if event_log.is_file():
                self.store.finalize_file(job_id, event_log, "events.jsonl")
            current = self.store.read_job(job_id)
            if current.get("status") in TERMINAL_STATUSES:
                refreshed_artifacts = []
                for item in current.get("artifacts") or []:
                    name = str(item.get("name") or "")
                    if name and (self.store.job_dir(job_id) / "final" / name).is_file():
                        refreshed_artifacts.append(self.store.artifact(job_id, name, item.get("media_type")))
                self._set_state(job_id, cleanupStatus="completed", cleanupAt=utc_now(), artifacts=refreshed_artifacts)
            with self._lock:
                self._futures.pop(job_id, None)


def _public_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    public_artifacts: list[dict[str, Any]] = []
    for artifact in payload.get("artifacts") or []:
        if not isinstance(artifact, dict) or not artifact.get("name"):
            continue
        public = {key: value for key, value in artifact.items() if key != "path"}
        public["downloadUrl"] = (
            f"/v1/qwen-image-edit/jobs/{quote(str(payload.get('id') or ''), safe='')}/artifacts/"
            f"{quote(str(artifact['name']), safe='')}"
        )
        public_artifacts.append(public)
    payload["artifacts"] = public_artifacts
    return payload

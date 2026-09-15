from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable
from uuid import uuid4

from fastapi import HTTPException

from .artifacts import Yue2ArtifactStore, utc_now
from .config import Yue2Config
from .contracts import Yue2Failure, Yue2Job, Yue2Request
from .observability import Yue2EventLogger
from .runtime import Yue2Runtime, Yue2RuntimeResult


class Yue2Worker:
    """A single-GPU, serialized YuE2 worker with durable job state."""

    def __init__(self, config: Yue2Config, runtime: Yue2Runtime | None = None):
        self.config = config
        self.store = Yue2ArtifactStore(config.artifact_root)
        self.runtime = runtime or Yue2Runtime(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yue2-worker")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    async def submit(self, request: Yue2Request) -> Yue2Job:
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
        job = Yue2Job(id=job_id, status="queued", request=request.to_dict(), created_at=now, updated_at=now)
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

    def _run(self, request: Yue2Request, job_id: str) -> None:
        logger = Yue2EventLogger(self.store.job_dir(job_id) / "events.jsonl", job_id)
        started = time.monotonic()
        runtime_result: Yue2RuntimeResult | None = None
        stage = "preflight"
        try:
            logger.emit("job_started", request=request.to_dict())
            self._set_state(job_id, status="running", attempt=1, stage="preflight", progress=0.01, message="Checking YuE2 runtime")
            preflight = self.runtime.preflight()
            self.store.finalize_json(job_id, "preflight.json", preflight)
            logger.emit("runtime_preflight", report=preflight)
            if not preflight.get("ready"):
                raise RuntimeError(f"YuE2 preflight failed: {preflight.get('reason') or preflight}")
            stage = "inference"
            self._set_state(job_id, status="running", stage="inference", progress=0.03, message="YuE2 runtime is ready")

            def progress(fraction: float, message: str) -> None:
                if time.monotonic() - started > self.config.job_timeout_seconds:
                    raise TimeoutError(f"YuE2 job exceeded the {self.config.job_timeout_seconds:.0f}s worker timeout")
                value = max(0.0, min(1.0, float(fraction)))
                self._set_state(job_id, status="running", stage="inference", progress=value, message=message)
                logger.emit("progress", stage="inference", progress=value, message=message)

            attempt_dir = self.store.attempt_dir(job_id, 1)
            runtime_result = self.runtime.generate(request, attempt_dir, progress)
            self.store.finalize_file(job_id, runtime_result.output_path, "audio.wav")
            self.store.finalize_json(job_id, "request.json", request.to_dict())
            self.store.finalize_json(job_id, "runtime-metadata.json", runtime_result.metadata)
            self.store.finalize_file(job_id, attempt_dir / "command.json", "command.json")
            self.store.finalize_file(job_id, attempt_dir / "audiocpp.log", "audiocpp.log")
            events_source = self.store.job_dir(job_id) / "events.jsonl"
            artifact_specs = [
                ("audio.wav", "audio/wav"),
                ("request.json", "application/json"),
                ("preflight.json", "application/json"),
                ("runtime-metadata.json", "application/json"),
                ("command.json", "application/json"),
                ("audiocpp.log", "text/plain"),
                ("events.jsonl", "application/jsonl"),
            ]
            logger.emit("job_succeeded", wallSeconds=time.monotonic() - started, artifacts=[item[0] for item in artifact_specs])
            self.store.finalize_file(job_id, events_source, "events.jsonl")
            artifacts = [self.store.artifact(job_id, name, media_type).__dict__ for name, media_type in artifact_specs]
            self._set_state(
                job_id,
                status="succeeded",
                stage="finalizing",
                progress=1.0,
                message="YuE2 music generation completed",
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.emit("job_failed", errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            failure = Yue2Failure(
                code="yue2_worker_timeout" if isinstance(exc, TimeoutError) else "yue2_worker_failed",
                message=str(exc),
                stage=stage,
                retryable=False,
                attempt=1,
            )
            self.store.finalize_json(job_id, "failure-summary.json", failure.to_dict())
            self.store.finalize_json(job_id, "request.json", request.to_dict())
            events_source = self.store.job_dir(job_id) / "events.jsonl"
            if events_source.is_file():
                self.store.finalize_file(job_id, events_source, "events.jsonl")
            artifact_specs = [("failure-summary.json", "application/json"), ("request.json", "application/json"), ("events.jsonl", "application/jsonl")]
            preflight_path = self.store.job_dir(job_id) / "final" / "preflight.json"
            if preflight_path.is_file():
                artifact_specs.append(("preflight.json", "application/json"))
            artifacts = [self.store.artifact(job_id, name, media_type).__dict__ for name, media_type in artifact_specs]
            self._set_state(
                job_id,
                status="failed",
                stage=failure.stage,
                progress=1.0,
                message=failure.message,
                failure=failure.to_dict(),
                failureCode=failure.code,
                refundRequired=True,
                refundReason="yue2_music_generation_failed",
                artifacts=artifacts,
            )
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    @staticmethod
    def _job_from_dict(payload: dict[str, Any]) -> Yue2Job:
        return Yue2Job(
            id=str(payload["id"]),
            status=payload.get("status", "failed"),
            request=payload.get("request") or {},
            stage=payload.get("stage"),
            progress=float(payload.get("progress") or 0),
            attempt=int(payload.get("attempt") or 0),
            message=payload.get("message"),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )

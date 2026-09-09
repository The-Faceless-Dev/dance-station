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

from .analysis import build_analysis
from .artifacts import MossMusicArtifactStore, utc_now
from .audio import acquire_audio, normalize_audio
from .config import MossMusicConfig
from .contracts import MossMusicFailure, MossMusicJob, MossMusicRequest
from .observability import MossMusicEventLogger
from .parser import parse_moss_response
from .runtime import MossRuntime, MossRuntimeResult, create_runtime
from .timeline import build_dense_timeline


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class MossMusicWorker:
    """One MOSS-Music analysis job at a time with durable terminal state."""

    def __init__(self, config: MossMusicConfig, runtime: MossRuntime | None = None):
        self.config = config
        self.store = MossMusicArtifactStore(config.artifact_root)
        self.runtime = runtime or create_runtime(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moss-music-worker")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    async def submit(self, request: MossMusicRequest) -> MossMusicJob:
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
        job = MossMusicJob(id=job_id, status="queued", request=request.to_dict(), created_at=now, updated_at=now)
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

    def _run(self, request: MossMusicRequest, job_id: str) -> None:
        logger = MossMusicEventLogger(self.store.job_dir(job_id) / "events.jsonl", job_id)
        runtime_result: MossRuntimeResult | None = None
        started_at = time.monotonic()
        try:
            logger.emit("job_started", request=request.to_dict())
            self._set_state(job_id, status="running", attempt=1, stage="validate_request", progress=0.01)

            def progress(stage: str, fraction: float, message: str) -> None:
                if time.monotonic() - started_at > self.config.job_timeout_seconds:
                    raise TimeoutError(f"MOSS-Music job exceeded the {self.config.job_timeout_seconds:.0f}s worker timeout")
                value = max(0.0, min(1.0, float(fraction)))
                self._set_state(job_id, status="running", stage=stage, progress=value, message=message)
                logger.emit("progress", stage=stage, progress=value, message=message)

            attempt_dir = self.store.attempt_dir(job_id, 1)
            progress("acquire_audio", 0.0, "Acquiring source audio")
            source_path = acquire_audio(
                request.audio,
                attempt_dir / "source",
                max_bytes=self.config.max_download_bytes,
            )
            logger.emit("audio_acquired", path=source_path.name, sizeBytes=source_path.stat().st_size)
            progress("acquire_audio", 1.0, "Source audio acquired")

            progress("normalize_audio", 0.0, "Normalizing audio to mono 16 kHz")
            audio = normalize_audio(
                source_path,
                attempt_dir / "audio",
                sample_rate=self.config.audio_sample_rate,
                max_duration_seconds=self.config.max_audio_seconds,
                progress=lambda message: logger.emit("audio_decode", message=message),
            )
            progress("normalize_audio", 1.0, f"Normalized {audio.duration_seconds:.3f}s of audio")
            logger.emit("audio_normalized", durationSeconds=audio.duration_seconds, sampleRate=audio.sample_rate)

            timeline: dict[str, Any] | None = None
            if request.include_dense_features:
                progress("build_timeline", 0.0, "Measuring dense visual-sync timeline")

                last_reported = 0

                def timeline_progress(completed: int, total: int) -> None:
                    nonlocal last_reported
                    fraction = completed / max(1, total)
                    report_interval = max(1, total // 100)
                    if completed == total or completed == 1 or completed - last_reported >= report_interval:
                        last_reported = completed
                        progress("build_timeline", fraction, f"Measured timeline cell {completed}/{total}")

                timeline = build_dense_timeline(
                    audio,
                    resolution_ms=request.event_resolution_ms,
                    max_cells=self.config.max_timeline_cells,
                    progress=timeline_progress,
                )
                logger.emit("timeline_built", cellCount=timeline["cell_count"], resolutionMs=request.event_resolution_ms)
            else:
                progress("build_timeline", 1.0, "Dense timeline disabled by request")

            if request.include_semantic_events:
                progress("load_backend", 0.0, "Checking MOSS-Music backend")
                backend_report = self.runtime.preflight()
                logger.emit("backend_preflight", report=backend_report)
                if backend_report.get("ready") is False:
                    raise RuntimeError(f"MOSS-Music backend is not ready: {backend_report}")
                progress("load_backend", 1.0, "MOSS-Music backend is ready")
                progress("moss_analysis", 0.0, "Submitting audio to MOSS-Music")
                runtime_result = self.runtime.analyze(
                    request,
                    audio,
                    lambda fraction, message: progress("moss_analysis", fraction, message),
                )
                self.store.finalize_json(job_id, "moss-response.json", runtime_result.response)
                raw_path = attempt_dir / "moss-raw.txt"
                raw_path.write_text(runtime_result.raw_text, encoding="utf-8")
                progress("parse_and_validate", 0.0, "Validating structured MOSS-Music response")
                semantic, raw_text = parse_moss_response(runtime_result.response)
                logger.emit("semantic_response_validated", rawCharacters=len(raw_text), eventCount=semantic["event_count"])
                progress("parse_and_validate", 1.0, "Structured MOSS-Music response validated")
            else:
                semantic = None
                runtime_result = MossRuntimeResult(response={}, raw_text="", metadata={"disabled": True})
                self.store.finalize_json(job_id, "moss-response.json", {"disabled": True})
                (attempt_dir / "moss-raw.txt").write_text("", encoding="utf-8")
                progress("load_backend", 1.0, "Semantic analysis disabled by request")
                progress("moss_analysis", 1.0, "Semantic analysis disabled by request")
                progress("parse_and_validate", 1.0, "No semantic response to validate")

            progress("merge_analysis", 0.0, "Combining measured and semantic analysis")
            analysis = build_analysis(request, audio, timeline, semantic, runtime_result.metadata)
            logger.emit(
                "analysis_merged",
                durationSeconds=analysis["source"]["duration_seconds"],
                denseCellCount=len(analysis["dense_features"]),
                semanticEventCount=len(analysis["events"]),
            )
            progress("merge_analysis", 1.0, "Analysis layers combined")

            progress("write_artifacts", 0.0, "Writing durable analysis artifacts")
            self.store.finalize_json(job_id, "analysis.json", analysis)
            self.store.finalize_json(
                job_id,
                "audio-metadata.json",
                {
                    "schema_version": 1,
                    "filename": request.audio.filename,
                    "duration_seconds": audio.duration_seconds,
                    "sample_rate": audio.sample_rate,
                    "channels": 1,
                    "normalized_filename": "normalized.wav",
                },
            )
            self.store.finalize_json(job_id, "request.json", request.to_dict())
            self.store.finalize_json(job_id, "moss-runtime.json", runtime_result.metadata)
            raw_attempt_path = attempt_dir / "moss-raw.txt"
            if raw_attempt_path.is_file():
                self.store.finalize_file(job_id, raw_attempt_path, "moss-raw.txt")
            events_path = self.store.job_dir(job_id) / "events.jsonl"
            if events_path.is_file():
                self.store.finalize_file(job_id, events_path, "events.jsonl")
            progress("write_artifacts", 1.0, "Analysis artifacts written")

            names = [
                ("analysis.json", "application/json"),
                ("audio-metadata.json", "application/json"),
                ("request.json", "application/json"),
                ("moss-response.json", "application/json"),
                ("moss-runtime.json", "application/json"),
                ("moss-raw.txt", "text/plain"),
                ("events.jsonl", "application/jsonl"),
            ]
            artifacts = [self.store.artifact(job_id, name, media_type).__dict__ for name, media_type in names]
            self._set_state(
                job_id,
                status="succeeded",
                stage="finalizing",
                progress=1.0,
                message="MOSS-Music analysis completed",
                artifacts=artifacts,
            )
            logger.emit("job_succeeded", artifacts=[artifact["name"] for artifact in artifacts])
        except Exception as exc:
            logger.exception("job_failed", exc)
            current = self.store.read_job(job_id)
            if runtime_result is not None:
                self.store.finalize_json(job_id, "moss-response.json", runtime_result.response)
                raw_attempt_path = self.store.attempt_dir(job_id, 1) / "moss-raw.txt"
                if raw_attempt_path.is_file():
                    self.store.finalize_file(job_id, raw_attempt_path, "moss-raw.txt")
            failure = MossMusicFailure(
                code="moss_music_worker_failed",
                message=str(exc) or type(exc).__name__,
                stage=current.get("stage") or "validate_request",
                retryable=False,
                attempt=int(current.get("attempt") or 1),
                details={"errorType": type(exc).__name__, "traceback": traceback.format_exc()},
            )
            failure_summary = {
                "schema_version": 1,
                "runtime": "moss-music",
                "job_id": job_id,
                "request": request.to_dict(),
                "failure": failure.to_dict(),
                "event_log": "events.jsonl",
                "created_at": utc_now(),
            }
            self.store.finalize_json(job_id, "failure-summary.json", failure_summary)
            event_log = self.store.job_dir(job_id) / "events.jsonl"
            if event_log.is_file():
                self.store.finalize_file(job_id, event_log, "events.jsonl")
            artifacts = [self.store.artifact(job_id, "failure-summary.json", "application/json").__dict__]
            artifacts.append(self.store.artifact(job_id, "events.jsonl", "application/jsonl").__dict__)
            raw_path = self.store.job_dir(job_id) / "final" / "moss-raw.txt"
            if raw_path.is_file():
                artifacts.append(self.store.artifact(job_id, "moss-raw.txt", "text/plain").__dict__)
            self._set_state(
                job_id,
                status="failed",
                progress=1.0,
                message=failure.message,
                failure=failure.to_dict(),
                failureCode=failure.code,
                refundRequired=True,
                refundReason="moss_music_analysis_failed",
                artifacts=artifacts,
            )
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
            logger.emit("job_thread_finished")

    @staticmethod
    def _job_from_dict(payload: dict[str, Any]) -> MossMusicJob:
        failure = payload.get("failure")
        return MossMusicJob(
            id=str(payload["id"]),
            status=payload.get("status", "queued"),
            request=payload.get("request") or {},
            stage=payload.get("stage"),
            progress=float(payload.get("progress") or 0),
            attempt=int(payload.get("attempt") or 0),
            message=payload.get("message"),
            artifacts=[],
            failure=MossMusicFailure(**failure) if isinstance(failure, dict) else None,
            created_at=str(payload.get("created_at") or payload.get("createdAt") or ""),
            updated_at=str(payload.get("updated_at") or payload.get("updatedAt") or ""),
        )

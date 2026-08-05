"""FastAPI worker service for paid avatar-generation jobs."""

from __future__ import annotations

import tempfile
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from autotransition.avatar.adapters import CommandImageGenerator, CommandMeshGenerator, CommandRigGenerator
from autotransition.avatar.artifacts import AvatarArtifactStore
from autotransition.avatar.contracts import AvatarJob, AvatarRequest
from autotransition.avatar.pipeline import AvatarPipeline
from autotransition.avatar.observability import emit_worker_event
from autotransition.avatar.resources import gpu_status
from autotransition.config import AvatarConfig


class AvatarWorker:
    """One GPU job at a time; job state and outputs are durable on disk."""

    def __init__(self, pipeline: AvatarPipeline):
        self.pipeline = pipeline
        self.store = pipeline.store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avatar-worker")
        self._cancel_events: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    async def submit(self, request: AvatarRequest, *, upload: UploadFile | None = None) -> AvatarJob:
        emit_worker_event(
            "api_job_submission_received",
            externalJobId=request.external_job_id,
            paymentIntentId=request.payment_intent_id,
            quality=request.quality,
            seed=request.seed,
            maxAttempts=request.max_attempts,
            hasReferenceUpload=upload is not None,
            descriptionCharacters=len(request.description),
        )
        if request.external_job_id:
            try:
                existing = self.pipeline.load_job(request.external_job_id)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                existing_request = existing.request
                if existing_request.get("payment_intent_id") != request.payment_intent_id:
                    emit_worker_event(
                        "api_job_idempotency_conflict",
                        externalJobId=request.external_job_id,
                        existingPaymentIntentId=existing_request.get("payment_intent_id"),
                        requestedPaymentIntentId=request.payment_intent_id,
                    )
                    raise HTTPException(status_code=409, detail="external job id is already bound to another payment intent")
                emit_worker_event("api_job_idempotency_replayed", jobId=existing.id, externalJobId=request.external_job_id)
                return existing
        temporary: Path | None = None
        if upload is not None:
            temporary = await self._receive_upload(upload, max_bytes=self.pipeline.config.max_image_bytes)
            request = replace(request, reference_image=temporary)
        try:
            job = self.pipeline.create_job(request, job_id=request.external_job_id or uuid4().hex)
            if temporary is not None:
                stored_reference = self.store.write_upload(job.id, temporary, upload.filename or "reference.png")
                request = replace(request, reference_image=stored_reference)
                job.request = request.to_dict()
                self.store.write_job(job)
            cancel_event = threading.Event()
            with self._lock:
                self._cancel_events[job.id] = cancel_event
                self._futures[job.id] = self.executor.submit(self._run, request, job.id, cancel_event, temporary)
            emit_worker_event("api_job_accepted", jobId=job.id, externalJobId=request.external_job_id, hasUpload=temporary is not None)
            return job
        except Exception as exc:
            emit_worker_event("api_job_submission_failed", errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
            emit_worker_event("api_job_cancel_requested", jobId=job_id)
        else:
            emit_worker_event("api_job_cancel_not_running", jobId=job_id)
        return self.store.read_job(job_id)

    def shutdown(self) -> None:
        emit_worker_event("worker_shutdown_requested", activeJobs=len(self._futures))
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _run(self, request: AvatarRequest, job_id: str, cancel_event: threading.Event, temporary: Path | None) -> None:
        emit_worker_event("worker_job_thread_started", jobId=job_id)
        try:
            self.pipeline.run(request, job_id=job_id, cancel_event=cancel_event)
        except Exception as exc:
            emit_worker_event("worker_job_thread_crashed", jobId=job_id, errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._futures.pop(job_id, None)
            emit_worker_event("worker_job_thread_finished", jobId=job_id, temporaryUploadCleaned=temporary is not None)

    @staticmethod
    async def _receive_upload(upload: UploadFile, *, max_bytes: int) -> Path:
        emit_worker_event(
            "reference_upload_started",
            filename=upload.filename,
            contentType=upload.content_type,
            maxBytes=max_bytes,
        )
        suffix = Path(upload.filename or "reference.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise HTTPException(status_code=400, detail="reference image must be PNG or JPEG")
        handle = tempfile.NamedTemporaryFile(prefix="avatar-upload-", suffix=suffix, delete=False)
        path = Path(handle.name)
        total = 0
        try:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    emit_worker_event("reference_upload_rejected_size", filename=upload.filename, bytesReceived=total, maxBytes=max_bytes)
                    raise HTTPException(status_code=413, detail="reference image exceeds the worker size limit")
                handle.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            handle.close()
        emit_worker_event("reference_upload_finished", filename=upload.filename, bytesReceived=total, path=path)
        return path


def create_avatar_worker_app(config: AvatarConfig | None = None) -> FastAPI:
    config = config or AvatarConfig.from_env()
    store = AvatarArtifactStore(config.artifact_root)
    interrupted = store.reconcile_interrupted_jobs()
    emit_worker_event(
        "worker_starting",
        artifactRoot=config.artifact_root,
        interruptedJobs=interrupted,
        maxAttempts=config.max_attempts,
        jobTimeoutSeconds=config.job_timeout_seconds,
        imageTimeoutSeconds=config.image_timeout_seconds,
        meshTimeoutSeconds=config.mesh_timeout_seconds,
        rigTimeoutSeconds=config.rig_timeout_seconds,
        gpuRequired=config.gpu_required,
        requireDeformationValidator=config.require_deformation_validator,
        imageModel=config.image_model_revision,
        meshModel=config.mesh_model_revision,
        rigModel=config.rig_model_revision,
        imageCommandConfigured=bool(config.image_command),
        meshCommandConfigured=bool(config.mesh_command),
        rigCommandConfigured=bool(config.rig_command),
        deformationValidatorConfigured=bool(config.deformation_validator_command),
    )
    pipeline = AvatarPipeline(
        config,
        image_generator=(
            CommandImageGenerator(config.image_command, timeout_seconds=config.image_timeout_seconds)
            if config.image_command
            else None
        ),
        mesh_generator=CommandMeshGenerator(config.mesh_command, timeout_seconds=config.mesh_timeout_seconds),
        rig_generator=CommandRigGenerator(config.rig_command, timeout_seconds=config.rig_timeout_seconds),
        store=store,
    )
    worker = AvatarWorker(pipeline)
    app = FastAPI(title="The Faceless Dancer Avatar Worker", version="0.1.0")
    app.state.avatar_worker = worker

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        missing = []
        if not config.image_command:
            missing.append("AVATAR_IMAGE_COMMAND")
        if not config.mesh_command:
            missing.append("AVATAR_MESH_COMMAND")
        if not config.rig_command:
            missing.append("AVATAR_RIG_COMMAND")
        if missing:
            emit_worker_event("worker_readiness_failed_configuration", missing=missing)
            raise HTTPException(status_code=503, detail={"message": "avatar worker is not configured", "missing": missing})
        gpu = gpu_status()
        if config.gpu_required and not gpu.get("available"):
            emit_worker_event("worker_readiness_failed_gpu", gpu=gpu)
            raise HTTPException(status_code=503, detail={"message": "avatar worker GPU is unavailable", "gpu": gpu})
        return {
            "status": "ready",
            "maxAttempts": config.max_attempts,
            "gpuRequired": config.gpu_required,
            "deformationValidation": "built-in" if config.require_deformation_validator else "structural-only",
            "externalDeformationValidator": bool(config.deformation_validator_command),
            "gpu": gpu,
        }

    @app.post("/v1/avatar/jobs", status_code=202)
    async def create_job(
        description: str = Form(...),
        quality: str = Form("runtime"),
        seed: int | None = Form(None),
        max_attempts: int = Form(3),
        external_job_id: str | None = Form(None),
        payment_intent_id: str | None = Form(None),
        reference_image: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        try:
            request = AvatarRequest(
                description=description,
                quality=quality,  # type: ignore[arg-type]
                seed=seed,
                max_attempts=max_attempts,
                external_job_id=external_job_id,
                payment_intent_id=payment_intent_id,
            )
            job = await worker.submit(request, upload=reference_image)
        except HTTPException:
            raise
        except ValueError as exc:
            emit_worker_event("api_create_job_validation_failed", error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            emit_worker_event("api_create_job_failed", errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise HTTPException(status_code=503, detail="avatar worker is temporarily unavailable") from exc
        return {"job": worker.store.read_job(job.id)}

    @app.get("/v1/avatar/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return {"job": store.read_job(job_id)}
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="avatar job was not found") from exc

    @app.post("/v1/avatar/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            worker.cancel(job_id)
            return {"job": store.read_job(job_id)}
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="avatar job was not found") from exc

    @app.get("/v1/avatar/jobs/{job_id}/artifacts/{artifact_name}")
    def get_artifact(job_id: str, artifact_name: str) -> FileResponse:
        try:
            payload = store.read_job(job_id)
            artifact = next(item for item in payload.get("artifacts", []) if item.get("name") == artifact_name)
            path = Path(artifact["path"]).resolve()
            final_dir = (store.job_dir(job_id) / "final").resolve()
            path.relative_to(final_dir)
        except (FileNotFoundError, StopIteration, KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="avatar artifact was not found") from exc
        return FileResponse(path, media_type=artifact.get("media_type", "application/octet-stream"), filename=path.name)

    return app

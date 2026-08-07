"""Image -> mesh -> rig -> validated runtime avatar pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from autotransition.avatar.adapters.base import AvatarAdapterError, ImageGenerator, MeshGenerator, RigGenerator
from autotransition.avatar.artifacts import AvatarArtifactStore, utc_now
from autotransition.avatar.contracts import AvatarArtifact, AvatarFailure, AvatarJob, AvatarRequest, AvatarResult
from autotransition.avatar.observability import AvatarEventLogger, current_event_logger, use_event_logger
from autotransition.avatar.prompts import HUMANOID_NEGATIVE_PROMPT, compose_avatar_prompt, compose_retry_prompt
from autotransition.avatar.resources import GpuLease, gpu_status, release_cuda_memory
from autotransition.avatar.validation import (
    AvatarValidationError,
    run_deformation_validator,
    validate_glb,
    validate_image,
    validate_manifest,
    read_glb_json,
)
from autotransition.config import AvatarConfig

ProgressCallback = Callable[[AvatarJob], None]


class AvatarPipeline:
    """Run one avatar request at a time and persist every state transition."""

    def __init__(
        self,
        config: AvatarConfig,
        *,
        image_generator: ImageGenerator | None,
        mesh_generator: MeshGenerator,
        rig_generator: RigGenerator,
        progress_callback: ProgressCallback | None = None,
        store: AvatarArtifactStore | None = None,
        gpu_lease: GpuLease | None = None,
    ):
        self.config = config
        self.store = store or AvatarArtifactStore(config.artifact_root)
        self.image_generator = image_generator
        self.mesh_generator = mesh_generator
        self.rig_generator = rig_generator
        self.progress_callback = progress_callback
        self.gpu_lease = gpu_lease or GpuLease()

    def create_job(self, request: AvatarRequest, *, job_id: str | None = None) -> AvatarJob:
        request.validate(
            max_description_characters=self.config.max_description_characters,
            max_attempts=self.config.max_attempts,
        )
        resolved_id = job_id or uuid4().hex
        now = utc_now()
        job = AvatarJob(
            id=resolved_id,
            status="queued",
            request=request.to_dict(),
            created_at=now,
            updated_at=now,
        )
        self.store.create_job(job)
        return job

    def run(self, request: AvatarRequest, *, job_id: str, cancel_event: threading.Event | None = None) -> AvatarResult:
        event_logger = AvatarEventLogger(self.store.event_log_path(job_id), job_id=job_id)
        with use_event_logger(event_logger):
            event_logger.emit(
                "job_run_started",
                description=request.description,
                quality=request.quality,
                seed=request.seed,
                maxAttempts=request.max_attempts,
                referenceImage=request.reference_image,
                externalJobId=request.external_job_id,
                paymentIntentId=request.payment_intent_id,
            )
            try:
                result = self._run(request, job_id=job_id, cancel_event=cancel_event)
            except Exception as exc:
                event_logger.exception("job_run_crashed", exc, stage="unhandled")
                raise
            event_logger.emit(
                "job_run_finished",
                status=result.status,
                refundRequired=result.refund_required,
                refundReason=result.refund_reason,
                failure=result.failure.to_dict() if result.failure else None,
            )
            return result

    def _run(self, request: AvatarRequest, *, job_id: str, cancel_event: threading.Event | None = None) -> AvatarResult:
        job = self.load_job(job_id)
        request.validate(
            max_description_characters=self.config.max_description_characters,
            max_attempts=self.config.max_attempts,
        )
        job.status = "running"
        self._update(job, stage="validate_request", progress=0.01)
        failure_history: list[AvatarFailure] = []

        active_logger = current_event_logger()
        if active_logger:
            active_logger.emit(
                "gpu_lease_wait_started",
                timeoutSeconds=self.config.job_timeout_seconds,
                gpu=gpu_status(),
            )
        if not self.gpu_lease.acquire(timeout=self.config.job_timeout_seconds):
            if active_logger:
                active_logger.emit("gpu_lease_acquire_failed", timeoutSeconds=self.config.job_timeout_seconds)
            failure = AvatarFailure(
                code="gpu_busy_timeout",
                message="avatar worker remained busy until the job timeout",
                stage="validate_request",
                retryable=False,
                attempt=0,
            )
            return self._finish_failed(job, failure, failure_history)
        if active_logger:
            active_logger.emit("gpu_lease_acquired", gpu=gpu_status())

        try:
            for attempt in range(1, request.max_attempts + 1):
                self._check_cancel(cancel_event, job, attempt)
                job.attempt = attempt
                prompt = compose_avatar_prompt(request.description)
                if failure_history:
                    prompt = compose_retry_prompt(request.description, [failure.code for failure in failure_history])
                if active_logger:
                    active_logger.emit(
                        "attempt_started",
                        attempt=attempt,
                        maxAttempts=request.max_attempts,
                        promptCharacters=len(prompt),
                        promptPreview=prompt[:500],
                        promptSha256=__import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
                        retryAfter=[failure.code for failure in failure_history],
                    )
                try:
                    artifacts, diagnostics = self._run_attempt(
                        request,
                        job,
                        attempt=attempt,
                        prompt=prompt,
                        cancel_event=cancel_event,
                    )
                    self._update(job, stage="finalizing", progress=0.98)
                    final_artifacts = self._finalize(job, request, attempt, artifacts, diagnostics, failure_history)
                    job.status = "succeeded"
                    job.progress = 1.0
                    job.stage = "finalizing"
                    job.artifacts = final_artifacts
                    job.failure = None
                    job.refund_required = False
                    job.refund_reason = None
                    self._save(job)
                    if active_logger:
                        active_logger.emit("attempt_succeeded", attempt=attempt, artifacts=final_artifacts)
                    return AvatarResult(
                        status="succeeded",
                        job_id=job.id,
                        artifacts=tuple(final_artifacts),
                        diagnostics={"attempts": attempt, **diagnostics},
                    )
                except _Cancelled:
                    if active_logger:
                        active_logger.emit("attempt_cancelled", attempt=attempt, stage=job.stage)
                    failure = AvatarFailure(
                        code="avatar_cancelled",
                        message="avatar generation was cancelled",
                        stage=job.stage or "validate_request",
                        retryable=False,
                        attempt=attempt,
                    )
                    return self._finish_cancelled(job, failure, failure_history)
                except (AvatarValidationError, AvatarAdapterError) as exc:
                    failure = self._failure_from_exception(exc, job, attempt)
                    if active_logger:
                        active_logger.exception(
                            "attempt_failed",
                            exc,
                            attempt=attempt,
                            stage=job.stage,
                            failure=failure.to_dict(),
                            willRetry=failure.retryable and attempt < request.max_attempts,
                        )
                    failure_history.append(failure)
                    self._record_attempt(job, failure)
                    self.store.write_attempt_failure(job.id, attempt, failure.to_dict())
                    self.store.remove_attempt_outputs(job.id, attempt)
                    release_cuda_memory()
                    if failure.retryable and attempt < request.max_attempts:
                        self._update(job, stage=job.stage, progress=min(0.12 + attempt * 0.04, 0.25))
                        continue
                    return self._finish_failed(job, self._exhausted_failure(failure, failure_history), failure_history)
                except Exception as exc:  # model wrappers can raise third-party exceptions
                    if active_logger:
                        active_logger.exception(
                            "attempt_crashed",
                            exc,
                            attempt=attempt,
                            stage=job.stage,
                            willRetry=attempt < request.max_attempts,
                        )
                    failure = AvatarFailure(
                        code="avatar_generation_failed",
                        message=str(exc),
                        stage=job.stage or "validate_request",
                        retryable=True,
                        attempt=attempt,
                    )
                    failure_history.append(failure)
                    self._record_attempt(job, failure)
                    self.store.write_attempt_failure(job.id, attempt, failure.to_dict())
                    self.store.remove_attempt_outputs(job.id, attempt)
                    release_cuda_memory()
                    if attempt < request.max_attempts:
                        self._update(job, stage=job.stage, progress=min(0.12 + attempt * 0.04, 0.25))
                        continue
                    return self._finish_failed(job, self._exhausted_failure(failure, failure_history), failure_history)
        finally:
            if active_logger:
                active_logger.emit("gpu_lease_releasing", gpu=gpu_status())
            self.gpu_lease.release()
            release_cuda_memory()
            if active_logger:
                active_logger.emit("gpu_lease_released", gpu=gpu_status())

        raise RuntimeError("avatar pipeline exited without a result")

    def load_job(self, job_id: str) -> AvatarJob:
        """Load a persisted job for API idempotency or status inspection."""

        return self._load_job(job_id)

    def _run_attempt(
        self,
        request: AvatarRequest,
        job: AvatarJob,
        *,
        attempt: int,
        prompt: str,
        cancel_event: threading.Event | None,
    ) -> tuple[dict[str, Path], dict[str, Any]]:
        attempt_dir = self.store.attempt_dir(job.id, attempt)
        event_logger = current_event_logger()
        source_suffix = request.reference_image.suffix.lower() if request.reference_image else ".png"
        source_image = attempt_dir / f"source-image{source_suffix}"
        if event_logger:
            event_logger.emit(
                "attempt_workspace_prepared",
                attempt=attempt,
                attemptDir=attempt_dir,
                sourceImage=source_image,
                referenceMode=request.reference_image is not None,
                imageGeneratorConfigured=self.image_generator is not None,
                meshGeneratorConfigured=bool(getattr(self.mesh_generator, "command", True)),
                rigGeneratorConfigured=bool(getattr(self.rig_generator, "command", True)),
            )
        if request.reference_image is not None and self.image_generator is None:
            self._update(job, stage="image_generation", progress=0.08)
            shutil.copy2(request.reference_image, source_image)
        else:
            if self.image_generator is None:
                raise AvatarAdapterError(
                    "image_generator_not_configured",
                    "no image generator is configured for text-only avatar requests",
                    retryable=False,
                )
            self._update(job, stage="image_generation", progress=0.08)
            self.image_generator.generate(
                prompt=prompt,
                negative_prompt=HUMANOID_NEGATIVE_PROMPT,
                output=source_image,
                seed=(request.seed + attempt - 1) if request.seed is not None else None,
                reference_image=request.reference_image,
                quality=request.quality,
            )
        if event_logger:
            event_logger.emit("image_generation_finished", attempt=attempt, output=file_details(source_image))
        self._check_cancel(cancel_event, job, attempt)
        self._update(job, stage="image_validation", progress=0.18)
        image_report = validate_image(
            source_image,
            min_width=self.config.min_image_width,
            min_height=self.config.min_image_height,
            max_bytes=self.config.max_image_bytes,
            max_pixels=self.config.max_reference_pixels,
        )
        if not image_report.ok:
            if request.reference_image is not None:
                raise AvatarValidationError("reference_image_invalid", "reference image failed validation", retryable=False)
            image_report.raise_if_invalid()
        if event_logger:
            event_logger.emit("image_validation_finished", attempt=attempt, ok=image_report.ok, report=image_report.details)

        self._check_cancel(cancel_event, job, attempt)
        mesh_dir = attempt_dir / "mesh"
        self._update(job, stage="mesh_generation", progress=0.28)
        mesh = self.mesh_generator.generate(image=source_image, output_dir=mesh_dir, quality=request.quality)
        if event_logger:
            event_logger.emit("mesh_generation_finished", attempt=attempt, output=file_details(mesh), outputDir=mesh_dir)
        self._update(job, stage="mesh_validation", progress=0.46)
        mesh_report = validate_glb(mesh, require_skin=False)
        mesh_report.raise_if_invalid()
        if event_logger:
            event_logger.emit("mesh_validation_finished", attempt=attempt, ok=mesh_report.ok, report=mesh_report.details)

        self._check_cancel(cancel_event, job, attempt)
        rig = attempt_dir / "avatar.glb"
        manifest = attempt_dir / "manifest.json"
        self._update(job, stage="rigging", progress=0.58)
        self.rig_generator.generate(mesh=mesh, output=rig, manifest=manifest, quality=request.quality)
        if event_logger:
            event_logger.emit(
                "rig_generation_finished",
                attempt=attempt,
                output=file_details(rig),
                manifest=file_details(manifest),
            )
        self._update(job, stage="rig_validation", progress=0.76)
        rig_report = validate_glb(rig)
        rig_report.raise_if_invalid()
        manifest_report = validate_manifest(manifest, read_glb_json(rig))
        manifest_report.raise_if_invalid()
        if event_logger:
            event_logger.emit(
                "rig_validation_finished",
                attempt=attempt,
                glbReport=rig_report.details,
                manifestReport=manifest_report.details,
            )

        self._check_cancel(cancel_event, job, attempt)
        self._update(job, stage="runtime_validation", progress=0.86)
        deformation_report = run_deformation_validator(
            _parse_validator_command(self.config.deformation_validator_command),
            glb=rig,
            manifest=manifest,
            output=attempt_dir / "deformation-report.json",
            timeout_seconds=self.config.rig_timeout_seconds,
            required=self.config.require_deformation_validator,
        )
        if event_logger:
            event_logger.emit(
                "runtime_validation_finished",
                attempt=attempt,
                report=deformation_report,
                output=file_details(attempt_dir / "deformation-report.json"),
            )
        diagnostics = {
            "attempt": attempt,
            "promptPolicyVersion": self.config.prompt_policy_version,
            "prompt": prompt,
            "seed": (request.seed + attempt - 1) if request.seed is not None else None,
            "models": {
                "image": self.config.image_model_revision,
                "mesh": self.config.mesh_model_revision,
                "rig": self.config.rig_model_revision,
            },
            "licenses": {
                "image": self.config.image_model_license,
                "mesh": self.config.mesh_model_license,
                "rig": self.config.rig_model_license,
            },
            "image": image_report.details,
            "mesh": mesh_report.details,
            "rig": rig_report.details,
            "manifest": manifest_report.details,
            "deformation": deformation_report,
        }
        (attempt_dir / "attempt.json").write_text(json.dumps(diagnostics, indent=2, default=str) + "\n", encoding="utf-8")
        return {"image": source_image, "mesh": mesh, "rig": rig, "manifest": manifest}, diagnostics

    def _finalize(
        self,
        job: AvatarJob,
        request: AvatarRequest,
        attempt: int,
        files: dict[str, Path],
        diagnostics: dict[str, Any],
        failures: list[AvatarFailure],
    ) -> list[AvatarArtifact]:
        event_logger = current_event_logger()
        if event_logger:
            event_logger.emit("finalization_started", attempt=attempt, files=files)
        image_suffix = files["image"].suffix.lower() or ".png"
        self.store.finalize_file(job.id, files["image"], f"source-image{image_suffix}")
        self.store.finalize_file(job.id, files["rig"], "avatar.glb")
        self.store.finalize_file(job.id, files["manifest"], "manifest.json")
        self.store.finalize_json(
            job.id,
            "diagnostics.json",
            {"successful_attempt": attempt, "request": request.to_dict(), "diagnostics": diagnostics, "failures": [failure.to_dict() for failure in failures]},
        )
        if request.retain_debug_artifacts or self.config.keep_debug_artifacts:
            debug_dir = self.store.job_dir(job.id) / "final" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(files["mesh"], debug_dir / "mesh.glb")
            shutil.copy2(files["rig"], debug_dir / "avatar.glb")
        artifacts = [
            self.store.artifact(job.id, f"source-image{image_suffix}", "image/*"),
            self.store.artifact(job.id, "avatar.glb", "model/gltf-binary"),
            self.store.artifact(job.id, "manifest.json", "application/json"),
            self.store.artifact(job.id, "diagnostics.json", "application/json"),
        ]
        if event_logger:
            event_logger.emit("finalization_finished", attempt=attempt, artifacts=artifacts)
        return artifacts

    def _finish_failed(self, job: AvatarJob, failure: AvatarFailure, history: list[AvatarFailure]) -> AvatarResult:
        job.status = "failed"
        job.progress = 1.0
        job.failure = failure
        job.refund_required = True
        job.refund_reason = (
            "avatar_output_validation_failed_after_retries"
            if failure.code == "avatar_validation_exhausted"
            else "avatar_generation_failed"
        )
        self._save(job)
        event_logger = current_event_logger()
        if event_logger:
            event_logger.emit(
                "job_failed_refund_required",
                failure=failure.to_dict(),
                failureHistory=[item.to_dict() for item in history],
                refundRequired=True,
                refundReason=job.refund_reason,
            )
        return AvatarResult(
            status="failed",
            job_id=job.id,
            failure=failure,
            refund_required=True,
            refund_reason=job.refund_reason,
            diagnostics={"attempts": [item.to_dict() for item in history]},
        )

    def _finish_cancelled(self, job: AvatarJob, failure: AvatarFailure, history: list[AvatarFailure]) -> AvatarResult:
        job.status = "cancelled"
        job.progress = 1.0
        job.failure = failure
        job.refund_required = True
        job.refund_reason = "avatar_generation_cancelled"
        self._save(job)
        event_logger = current_event_logger()
        if event_logger:
            event_logger.emit("job_cancelled_refund_required", failure=failure.to_dict(), refundReason=job.refund_reason)
        return AvatarResult(
            status="cancelled",
            job_id=job.id,
            failure=failure,
            refund_required=True,
            refund_reason=job.refund_reason,
            diagnostics={"attempts": [item.to_dict() for item in history]},
        )

    def _record_attempt(self, job: AvatarJob, failure: AvatarFailure) -> None:
        job.attempts.append(failure.to_dict())
        self._save(job)
        event_logger = current_event_logger()
        if event_logger:
            event_logger.emit("attempt_failure_recorded", attempt=failure.attempt, failure=failure.to_dict())

    def _update(self, job: AvatarJob, *, stage: str | None, progress: float) -> None:
        previous_stage = job.stage
        previous_progress = job.progress
        job.stage = stage  # type: ignore[assignment]
        job.progress = max(0.0, min(1.0, progress))
        self._save(job)
        event_logger = current_event_logger()
        if event_logger:
            event_logger.emit(
                "job_progress_updated",
                attempt=job.attempt,
                previousStage=previous_stage,
                stage=stage,
                previousProgress=previous_progress,
                progress=job.progress,
            )

    def _save(self, job: AvatarJob) -> None:
        self.store.write_job(job)
        if self.progress_callback:
            self.progress_callback(job)

    def _load_job(self, job_id: str) -> AvatarJob:
        payload = self.store.read_job(job_id)
        artifacts = tuple(AvatarArtifact(**artifact) for artifact in payload.get("artifacts", []))
        failure_payload = payload.get("failure")
        failure = AvatarFailure(**failure_payload) if failure_payload else None
        return AvatarJob(
            id=payload["id"],
            status=payload["status"],
            request=payload["request"],
            stage=payload.get("stage"),
            progress=payload.get("progress", 0.0),
            attempt=payload.get("attempt", 0),
            attempts=payload.get("attempts", []),
            artifacts=list(artifacts),
            failure=failure,
            refund_required=payload.get("refund_required", False),
            refund_reason=payload.get("refund_reason"),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
        )

    @staticmethod
    def _failure_from_exception(exc: AvatarValidationError | AvatarAdapterError, job: AvatarJob, attempt: int) -> AvatarFailure:
        return AvatarFailure(
            code=exc.code,
            message=str(exc),
            stage=job.stage or "validate_request",
            retryable=exc.retryable,
            attempt=attempt,
            details=exc.details,
        )

    @staticmethod
    def _exhausted_failure(last: AvatarFailure, history: list[AvatarFailure]) -> AvatarFailure:
        validation_codes = {
            "glb_invalid_header",
            "glb_missing_json",
            "glb_invalid_json",
            "glb_non_finite",
            "mesh_missing",
            "nodes_missing",
            "rig_missing",
            "rig_too_small",
            "skin_joints_invalid",
            "skin_weights_missing",
            "accessors_invalid",
            "manifest_invalid",
            "manifest_skeleton_invalid",
            "manifest_bones_missing",
            "manifest_roles_missing",
            "manifest_roles_invalid",
            "deformation_validation_failed",
        }
        if any(item.code in validation_codes or item.code.startswith("deformation_") or item.code.endswith("validation_failed") for item in history):
            return AvatarFailure(
                code="avatar_validation_exhausted",
                message="avatar output failed validation after the allowed retries",
                stage=last.stage,
                retryable=False,
                attempt=last.attempt,
                details={"failures": [item.to_dict() for item in history]},
            )
        return last

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None, job: AvatarJob, attempt: int) -> None:
        if cancel_event and cancel_event.is_set():
            event_logger = current_event_logger()
            if event_logger:
                event_logger.emit("cancellation_observed", attempt=attempt, stage=job.stage)
            job.stage = job.stage or "validate_request"
            raise _Cancelled()


class _Cancelled(Exception):
    pass


def _parse_validator_command(command: str | None) -> tuple[str, ...] | None:
    if not command:
        return None
    import shlex

    return tuple(shlex.split(command))


def file_details(path: Path) -> dict[str, Any]:
    """Describe an artifact without loading large model files into memory twice."""

    details: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
        details.update({"exists": True, "sizeBytes": stat.st_size})
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        details["sha256"] = digest.hexdigest()
    except OSError as exc:
        details.update({"exists": False, "error": str(exc)})
    return details

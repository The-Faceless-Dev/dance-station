"""Canonical skeleton re-skin jobs for existing avatar meshes."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.rig_generator import CommandReskinGenerator
from autotransition.avatar.artifacts import AvatarArtifactStore, utc_now
from autotransition.avatar.canonical_skeleton import validate_profile, write_skeleton_glb
from autotransition.avatar.contracts import AvatarArtifact, AvatarFailure, AvatarJob, AvatarReskinRequest, AvatarResult
from autotransition.avatar.observability import AvatarEventLogger, use_event_logger
from autotransition.avatar.pipeline import _parse_validator_command
from autotransition.avatar.resources import GpuLease, gpu_status, release_cuda_memory
from autotransition.avatar.validation import (
    AvatarValidationError,
    run_deformation_validator,
    validate_glb,
    validate_manifest,
    read_glb_json,
)
from autotransition.config import AvatarConfig

ProgressCallback = Callable[[AvatarJob], None]


class AvatarReskinPipeline:
    """Fit a canonical skeleton, calculate skin weights, and validate output."""

    def __init__(
        self,
        config: AvatarConfig,
        *,
        reskin_generator: CommandReskinGenerator,
        progress_callback: ProgressCallback | None = None,
        store: AvatarArtifactStore | None = None,
        gpu_lease: GpuLease | None = None,
    ):
        self.config = config
        self.reskin_generator = reskin_generator
        self.progress_callback = progress_callback
        self.store = store or AvatarArtifactStore(config.artifact_root)
        self.gpu_lease = gpu_lease or GpuLease()

    def create_job(self, request: AvatarReskinRequest, *, job_id: str | None = None) -> AvatarJob:
        request.validate(max_attempts=self.config.max_attempts)
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

    def run(
        self,
        request: AvatarReskinRequest,
        *,
        job_id: str,
        cancel_event: threading.Event | None = None,
    ) -> AvatarResult:
        logger = AvatarEventLogger(self.store.event_log_path(job_id), job_id=job_id)
        with use_event_logger(logger):
            try:
                return self._run(request, job_id=job_id, cancel_event=cancel_event)
            except Exception as exc:
                logger.exception("reskin_job_crashed", exc, stage="unhandled")
                raise

    def _run(self, request: AvatarReskinRequest, *, job_id: str, cancel_event: threading.Event | None) -> AvatarResult:
        job = self.load_job(job_id)
        request.validate(max_attempts=self.config.max_attempts)
        job.status = "running"
        self._update(job, stage="validate_request", progress=0.02)
        failures: list[AvatarFailure] = []
        if not self.gpu_lease.acquire(timeout=self.config.job_timeout_seconds):
            failure = AvatarFailure(
                code="gpu_busy_timeout",
                message="avatar re-skin worker remained busy until the job timeout",
                stage="validate_request",
                retryable=False,
                attempt=0,
            )
            return self._finish_failed(job, failure, failures)
        try:
            for attempt in range(1, self.config.max_attempts + 1):
                job.attempt = attempt
                try:
                    self._check_cancel(cancel_event, job)
                    files, diagnostics = self._run_attempt(request, job, attempt, cancel_event)
                    self._update(job, stage="finalizing", progress=0.98)
                    artifacts = self._finalize(job, request, files, diagnostics)
                    job.status = "succeeded"
                    job.progress = 1.0
                    job.stage = "finalizing"
                    job.artifacts = artifacts
                    job.failure = None
                    job.refund_required = False
                    job.refund_reason = None
                    self._save(job)
                    return AvatarResult(
                        status="succeeded",
                        job_id=job.id,
                        artifacts=tuple(artifacts),
                        diagnostics={"attempts": attempt, **diagnostics},
                    )
                except (AvatarValidationError, AvatarAdapterError) as exc:
                    failure = AvatarFailure(
                        code=exc.code,
                        message=str(exc),
                        stage=job.stage or "validate_request",
                        retryable=exc.retryable,
                        attempt=attempt,
                        details=getattr(exc, "details", {}),
                    )
                except _Cancelled:
                    failure = AvatarFailure(
                        code="avatar_cancelled",
                        message="avatar re-skin was cancelled",
                        stage=job.stage or "validate_request",
                        retryable=False,
                        attempt=attempt,
                    )
                    return self._finish_cancelled(job, failure)
                failures.append(failure)
                job.attempts.append(failure.to_dict())
                self.store.write_attempt_failure(job.id, attempt, failure.to_dict())
                self._save(job)
                if not failure.retryable or attempt >= self.config.max_attempts:
                    return self._finish_failed(job, failure, failures)
                release_cuda_memory()
        finally:
            self.gpu_lease.release()
            release_cuda_memory()
        raise RuntimeError("avatar re-skin exited without a result")

    def _run_attempt(
        self,
        request: AvatarReskinRequest,
        job: AvatarJob,
        attempt: int,
        cancel_event: threading.Event | None,
    ) -> tuple[dict[str, Path], dict[str, Any]]:
        attempt_dir = self.store.attempt_dir(job.id, attempt)
        self._update(job, stage="skeleton_fit", progress=0.20)
        profile = json.loads(request.profile.read_text(encoding="utf-8"))
        profile_errors = validate_profile(profile)
        if profile_errors:
            raise AvatarValidationError(
                "canonical_profile_invalid",
                "; ".join(profile_errors),
                details={"errors": profile_errors},
                retryable=False,
            )
        mesh_report = validate_glb(request.mesh, require_skin=False)
        mesh_report.raise_if_invalid()
        skeleton = attempt_dir / "canonical-skeleton.glb"
        write_skeleton_glb(request.mesh, profile, skeleton)
        skeleton_report = validate_glb(skeleton, require_skin=False)
        skeleton_report.raise_if_invalid()
        self._check_cancel(cancel_event, job)

        self._update(job, stage="reskinning", progress=0.45)
        output = attempt_dir / "avatar.glb"
        manifest = attempt_dir / "manifest.json"
        self.reskin_generator.generate(
            skeleton=skeleton,
            output=output,
            manifest=manifest,
            profile=request.profile,
            quality=request.quality,
        )
        self._check_cancel(cancel_event, job)
        self._update(job, stage="rig_validation", progress=0.74)
        rig_report = validate_glb(output)
        rig_report.raise_if_invalid()
        manifest_report = validate_manifest(manifest, read_glb_json(output))
        manifest_report.raise_if_invalid()
        self._update(job, stage="runtime_validation", progress=0.86)
        deformation_report = run_deformation_validator(
            _parse_validator_command(self.config.deformation_validator_command),
            glb=output,
            manifest=manifest,
            output=attempt_dir / "deformation-report.json",
            timeout_seconds=self.config.rig_timeout_seconds,
            required=self.config.require_deformation_validator,
        )
        diagnostics = {
            "jobType": "reskin",
            "attempt": attempt,
            "mesh": mesh_report.details,
            "skeleton": skeleton_report.details,
            "rig": rig_report.details,
            "manifest": manifest_report.details,
            "deformation": deformation_report,
            "profile": profile,
            "gpu": gpu_status(),
        }
        (attempt_dir / "attempt.json").write_text(json.dumps(diagnostics, indent=2, default=str) + "\n", encoding="utf-8")
        return {"mesh": request.mesh, "profile": request.profile, "skeleton": skeleton, "rig": output, "manifest": manifest}, diagnostics

    def _finalize(
        self,
        job: AvatarJob,
        request: AvatarReskinRequest,
        files: dict[str, Path],
        diagnostics: dict[str, Any],
    ) -> list[AvatarArtifact]:
        self.store.finalize_file(job.id, files["mesh"], "source-mesh.glb")
        self.store.finalize_file(job.id, files["profile"], "canonical-profile.json")
        self.store.finalize_file(job.id, files["skeleton"], "canonical-skeleton.glb")
        self.store.finalize_file(job.id, files["rig"], "avatar.glb")
        self.store.finalize_file(job.id, files["manifest"], "manifest.json")
        self.store.finalize_json(job.id, "diagnostics.json", {"request": request.to_dict(), "diagnostics": diagnostics})
        return [
            self.store.artifact(job.id, "source-mesh.glb", "model/gltf-binary"),
            self.store.artifact(job.id, "canonical-profile.json", "application/json"),
            self.store.artifact(job.id, "canonical-skeleton.glb", "model/gltf-binary"),
            self.store.artifact(job.id, "avatar.glb", "model/gltf-binary"),
            self.store.artifact(job.id, "manifest.json", "application/json"),
            self.store.artifact(job.id, "diagnostics.json", "application/json"),
        ]

    def _finish_failed(self, job: AvatarJob, failure: AvatarFailure, history: list[AvatarFailure]) -> AvatarResult:
        job.status = "failed"
        job.progress = 1.0
        job.failure = failure
        job.refund_required = True
        job.refund_reason = "avatar_reskin_failed"
        summary = {
            "schemaVersion": 1,
            "jobId": job.id,
            "jobType": "reskin",
            "failureCode": failure.code,
            "message": failure.message,
            "stage": failure.stage,
            "attempts": [item.to_dict() for item in history],
            "refundRequired": True,
            "createdAt": utc_now(),
        }
        self.store.finalize_json(job.id, "failure-summary.json", summary)
        summary_artifact = self.store.artifact(job.id, "failure-summary.json", "application/json")
        if summary_artifact.name not in {artifact.name for artifact in job.artifacts}:
            job.artifacts.append(summary_artifact)
        job.failure_summary = summary
        self._save(job)
        return AvatarResult(
            status="failed",
            job_id=job.id,
            artifacts=tuple(job.artifacts),
            failure=failure,
            refund_required=True,
            refund_reason=job.refund_reason,
            diagnostics=summary,
        )

    def _finish_cancelled(self, job: AvatarJob, failure: AvatarFailure) -> AvatarResult:
        job.status = "cancelled"
        job.progress = 1.0
        job.failure = failure
        job.refund_required = True
        job.refund_reason = "avatar_reskin_cancelled"
        self._save(job)
        return AvatarResult(status="cancelled", job_id=job.id, failure=failure, refund_required=True, refund_reason=job.refund_reason)

    def load_job(self, job_id: str) -> AvatarJob:
        payload = self.store.read_job(job_id)
        artifacts = [AvatarArtifact(**artifact) for artifact in payload.get("artifacts", [])]
        failure = AvatarFailure(**payload["failure"]) if payload.get("failure") else None
        return AvatarJob(
            id=payload["id"], status=payload["status"], request=payload["request"], stage=payload.get("stage"),
            progress=payload.get("progress", 0.0), attempt=payload.get("attempt", 0), attempts=payload.get("attempts", []),
            artifacts=artifacts, failure=failure, failure_summary=payload.get("failureSummary") or payload.get("failure_summary"),
            refund_required=payload.get("refundRequired", payload.get("refund_required", False)),
            refund_reason=payload.get("refundReason") or payload.get("refund_reason"),
            created_at=payload.get("created_at", ""), updated_at=payload.get("updated_at", ""),
        )

    def _update(self, job: AvatarJob, *, stage: str, progress: float) -> None:
        job.stage = stage  # type: ignore[assignment]
        job.progress = max(0.0, min(1.0, progress))
        self._save(job)

    def _save(self, job: AvatarJob) -> None:
        self.store.write_job(job)
        if self.progress_callback:
            self.progress_callback(job)

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None, job: AvatarJob) -> None:
        if cancel_event and cancel_event.is_set():
            raise _Cancelled()


class _Cancelled(Exception):
    pass

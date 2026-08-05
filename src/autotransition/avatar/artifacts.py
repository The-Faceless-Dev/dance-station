"""Durable job and artifact storage for the avatar worker."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autotransition.avatar.contracts import AvatarArtifact, AvatarJob


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AvatarArtifactStore:
    """Store job state on disk; only handles and locks stay in memory."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        if not job_id or any(part in {".", ".."} for part in Path(job_id).parts) or Path(job_id).name != job_id:
            raise ValueError("invalid avatar job id")
        return self.root / job_id

    def event_log_path(self, job_id: str) -> Path:
        """Return the append-only diagnostic log for one job."""

        return self.job_dir(job_id) / "events.jsonl"

    def create_job(self, job: AvatarJob) -> None:
        directory = self.job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "attempts").mkdir()
        (directory / "final").mkdir()
        self.write_job(job)

    def write_job(self, job: AvatarJob) -> None:
        directory = self.job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=True)
        job.updated_at = utc_now()
        self._atomic_json(directory / "job.json", job.to_dict())

    def read_job(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"avatar job was not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def reconcile_interrupted_jobs(self) -> list[str]:
        """Fail jobs that were active when the previous worker process died."""

        interrupted: list[str] = []
        for job_file in self.root.glob("*/job.json"):
            try:
                payload = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") not in {"queued", "running"}:
                continue
            job_id = str(payload.get("id", job_file.parent.name))
            failure = {
                "code": "avatar_worker_interrupted",
                "message": "avatar worker restarted before the job completed",
                "stage": payload.get("stage") or "validate_request",
                "retryable": False,
                "attempt": payload.get("attempt", 0),
                "details": {"reconciledAt": utc_now()},
            }
            payload.update(
                {
                    "status": "failed",
                    "progress": 1.0,
                    "failure": failure,
                    "refund_required": True,
                    "refund_reason": "avatar_worker_interrupted_before_completion",
                    "refundRequired": True,
                    "refundReason": "avatar_worker_interrupted_before_completion",
                    "failureCode": "avatar_worker_interrupted",
                    "updated_at": utc_now(),
                }
            )
            self._atomic_json(job_file, payload)
            interrupted.append(job_id)
        return interrupted

    def attempt_dir(self, job_id: str, attempt: int) -> Path:
        if attempt < 1:
            raise ValueError("attempt must be positive")
        path = self.job_dir(job_id) / "attempts" / f"attempt-{attempt}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_upload(self, job_id: str, source: Path, filename: str) -> Path:
        """Copy an upload into the durable request area with a safe suffix."""

        suffix = Path(filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("reference image must be PNG, JPEG, or WebP")
        destination = self.job_dir(job_id) / "request" / f"reference{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    def write_bytes(self, job_id: str, filename: str, data: bytes) -> Path:
        destination = self.job_dir(job_id) / "request" / Path(filename).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def finalize_file(self, job_id: str, source: Path, name: str) -> Path:
        destination = self.job_dir(job_id) / "final" / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def finalize_json(self, job_id: str, name: str, value: Any) -> Path:
        destination = self.job_dir(job_id) / "final" / Path(name).name
        self._atomic_json(destination, value)
        return destination

    def write_attempt_failure(self, job_id: str, attempt: int, failure: Any) -> Path:
        destination = self.attempt_dir(job_id, attempt) / "failure.json"
        self._atomic_json(destination, failure)
        return destination

    def remove_attempt_outputs(self, job_id: str, attempt: int) -> None:
        path = self.attempt_dir(job_id, attempt)
        for child in path.iterdir():
            if child.name in {
                "attempt.json",
                "failure.json",
                "source-image.png",
                "source-image.jpg",
                "source-image.jpeg",
                "source-image.webp",
            } or child.name.endswith(".stdout.log") or child.name.endswith(".stderr.log"):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    def artifact(self, job_id: str, name: str, media_type: str) -> AvatarArtifact:
        path = self.job_dir(job_id) / "final" / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return AvatarArtifact(
            name=path.name,
            path=str(path),
            media_type=media_type,
            size_bytes=path.stat().st_size,
            sha256=self.sha256(path),
        )

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

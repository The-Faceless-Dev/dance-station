from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import MuLaCoverArtifact, MuLaCoverJob


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MuLaCoverArtifactStore:
    """Durable job state and outputs; all paths remain inside artifact_root."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id or any(part in {".", ".."} for part in Path(job_id).parts):
            raise ValueError("invalid MuLaCover job id")
        return self.root / job_id

    def create_job(self, job: MuLaCoverJob) -> None:
        directory = self.job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "final").mkdir()
        (directory / "attempts").mkdir()
        self.write_job(job)

    def write_job(self, job: MuLaCoverJob) -> None:
        job.updated_at = utc_now()
        self._atomic_json(self.job_dir(job.id) / "job.json", job.to_dict())

    def read_job(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"MuLaCover job was not found: {job_id}")
        for attempt in range(8):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (PermissionError, OSError, json.JSONDecodeError):
                if attempt == 7:
                    raise
                time.sleep(0.02 * (attempt + 1))
        raise RuntimeError(f"could not read MuLaCover job state: {path}")

    def reconcile_interrupted_jobs(self) -> list[str]:
        interrupted: list[str] = []
        for path in self.root.glob("*/job.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") not in {"queued", "running"}:
                continue
            payload.update({
                "status": "failed", "progress": 1.0,
                "failure": {
                    "code": "mulacover_worker_interrupted",
                    "message": "MuLaCover worker restarted before the job completed",
                    "stage": payload.get("stage") or "startup", "retryable": False, "attempt": payload.get("attempt", 0),
                },
                "failureCode": "mulacover_worker_interrupted", "refundRequired": True,
                "refundReason": "mulacover_worker_interrupted_before_completion", "updated_at": utc_now(),
            })
            self._atomic_json(path, payload)
            interrupted.append(path.parent.name)
        return interrupted

    def attempt_dir(self, job_id: str, attempt: int) -> Path:
        path = self.job_dir(job_id) / "attempts" / f"attempt-{attempt}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def finalize_file(self, job_id: str, source: Path, name: str) -> Path:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = self.job_dir(job_id) / "final" / Path(name).name
        destination.write_bytes(source.read_bytes())
        return destination

    def finalize_json(self, job_id: str, name: str, value: Any) -> Path:
        destination = self.job_dir(job_id) / "final" / Path(name).name
        self._atomic_json(destination, value)
        return destination

    def artifact(self, job_id: str, name: str, media_type: str | None = None, role: str = "metadata") -> MuLaCoverArtifact:
        path = self.job_dir(job_id) / "final" / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(path)
        return MuLaCoverArtifact(
            name=path.name,
            path=str(path),
            media_type=media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            size_bytes=path.stat().st_size,
            sha256=self.sha256(path),
            role=role,
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
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            for attempt in range(8):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

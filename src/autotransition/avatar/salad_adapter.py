"""Salad queue adapter for the avatar-generation pipeline.

Salad delivers jobs to ``/process`` through its queue worker binary.  The
public avatar API remains available for local development, while this module
translates the shared launch-server job contract into ``AvatarRequest`` and
uploads the resulting artifacts through the callback contract.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request

from autotransition.config import AvatarConfig

from .contracts import AvatarRequest
from .worker import AvatarWorker


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
REFERENCE_ROLES = {"reference", "reference_image", "avatar_reference", "image"}


def _callback_payload(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    callback = payload.get("callback") or {}
    callback_url = str(callback.get("url") or "")
    complete_url = str(callback.get("complete_url") or "")
    progress_url = str(callback.get("progress_url") or "")
    callback_token = str(callback.get("token") or "")
    if not callback_url or not complete_url or not callback_token:
        raise ValueError("job is missing its callback contract")
    return callback_url, complete_url, progress_url, callback_token


def _job_id(payload: dict[str, Any]) -> str:
    value = str(payload.get("job_id") or "").strip()
    if not value or any(char in value for char in "\\/"):
        raise ValueError("job is missing a valid job_id")
    return value


def _reference_url(payload: dict[str, Any]) -> tuple[str | None, str]:
    for item in payload.get("inputs") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in REFERENCE_ROLES:
            continue
        source = str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "").strip()
        filename = Path(str(item.get("fileName") or item.get("file_name") or "reference.png")).name
        return source or None, filename
    parameters = payload.get("parameters") or {}
    if isinstance(parameters, dict):
        source = str(parameters.get("reference_image_url") or parameters.get("referenceImageUrl") or "").strip()
        return source or None, "reference.png"
    return None, "reference.png"


def _download_reference(url: str, filename: str, config: AvatarConfig) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("avatar reference image must use an HTTP(S) URL")
    suffix = Path(filename).suffix.lower() or Path(parsed.path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    handle = tempfile.NamedTemporaryFile(prefix="avatar-queue-reference-", suffix=suffix, delete=False)
    path = Path(handle.name)
    total = 0
    try:
        request = UrlRequest(url, headers={"Accept": "image/png,image/jpeg,image/webp,*/*"})
        with urlopen(request, timeout=120) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > config.max_image_bytes:
                raise ValueError("avatar reference image exceeds the worker size limit")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > config.max_image_bytes:
                    raise ValueError("avatar reference image exceeds the worker size limit")
                handle.write(chunk)
        if total == 0:
            raise ValueError("avatar reference image was empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        handle.close()


def _request_from_payload(payload: dict[str, Any], reference_image: Path | None) -> AvatarRequest:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("avatar job parameters must be an object")
    seed = parameters.get("seed")
    max_attempts = parameters.get("max_attempts", parameters.get("maxAttempts", 3))
    try:
        parsed_seed = int(seed) if seed is not None else None
        parsed_attempts = int(max_attempts)
    except (TypeError, ValueError) as exc:
        raise ValueError("avatar seed and max_attempts must be numeric") from exc
    return AvatarRequest(
        description=str(parameters.get("description") or parameters.get("prompt") or payload.get("description") or ""),
        reference_image=reference_image,
        quality=str(parameters.get("quality") or "runtime"),  # type: ignore[arg-type]
        seed=parsed_seed,
        max_attempts=parsed_attempts,
        external_job_id=_job_id(payload),
        payment_intent_id=str(parameters.get("payment_intent_id") or parameters.get("paymentIntentId") or "") or None,
    )


def _post_json(url: str, token: str, payload: Any, *, timeout: float = 120) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body)), "X-Job-Callback-Token": token},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"callback HTTP {exc.code}: {detail[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"callback failed to {url}: {exc}") from exc


def _artifact_role(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "preview"
    return "metadata"


def _upload_artifact(url: str, token: str, path: Path, artifact: dict[str, Any]) -> str:
    body = path.read_bytes()
    file_name = Path(str(artifact.get("name") or path.name)).name
    mime_type = str(artifact.get("media_type") or mimetypes.guess_type(file_name)[0] or "application/octet-stream")
    request = UrlRequest(
        url,
        data=body,
        headers={
            "Content-Type": mime_type,
            "Content-Length": str(len(body)),
            "X-Job-Callback-Token": token,
            "X-Artifact-Role": _artifact_role(file_name),
            "X-Artifact-Variant": "avatar-output",
            "X-Artifact-File-Name": file_name,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=1800) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"artifact callback HTTP {exc.code}: {detail[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"artifact callback failed to {url}: {exc}") from exc
    artifact_id = str(result.get("id") or result.get("artifactId") or "")
    if not artifact_id:
        raise RuntimeError("artifact callback returned no artifact id")
    return artifact_id


async def _report_progress(
    url: str,
    token: str,
    job_id: str,
    *,
    status: str,
    job: dict[str, Any],
    sequence: int,
    message: str | None = None,
) -> None:
    if not url:
        return
    progress = job.get("progress")
    completed_steps: int | None = None
    total_steps: int | None = None
    if progress is not None:
        total_steps = 100
        completed_steps = max(0, min(total_steps, round(float(progress) * total_steps)))
        progress = completed_steps / total_steps
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "runtime": "avatar",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": progress,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "attempt": job.get("attempt", 0),
        "message": message,
        "stage": job.get("stage"),
        "stage_index": None,
        "stage_count": None,
        "stage_kind": None,
        "task_id": None,
        "sequence": sequence,
        "updated_at": job.get("updated_at") or "",
    }
    try:
        await asyncio.to_thread(_post_json, url, token, payload, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "salad_progress_callback_failed", "job_id": job_id, "error": str(exc)}), flush=True)


async def _run_queue_job(payload: dict[str, Any], worker: AvatarWorker, config: AvatarConfig) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, callback_token = _callback_payload(payload)
    reference_path: Path | None = None
    try:
        reference_url, reference_name = _reference_url(payload)
        if reference_url:
            reference_path = await asyncio.to_thread(_download_reference, reference_url, reference_name, config)
        request = _request_from_payload(payload, reference_path)
        job = await worker.submit(request)
        sequence = 1
        await _report_progress(
            progress_url,
            callback_token,
            job_id,
            status="running",
            job=worker.store.read_job(job.id),
            sequence=sequence,
            message="Avatar worker accepted the job",
        )
        deadline = time.monotonic() + config.job_timeout_seconds + 60
        last_progress_key: tuple[Any, ...] | None = None
        while time.monotonic() < deadline:
            current = worker.store.read_job(job.id)
            key = (current.get("status"), current.get("stage"), current.get("progress"), current.get("attempt"), current.get("updated_at"))
            if key != last_progress_key:
                last_progress_key = key
                sequence += 1
                await _report_progress(progress_url, callback_token, job_id, status="running", job=current, sequence=sequence)
            if current.get("status") in TERMINAL_STATUSES:
                if current.get("status") != "succeeded":
                    message = str((current.get("failure") or {}).get("message") or "Avatar worker failed")
                    sequence += 1
                    await _report_progress(progress_url, callback_token, job_id, status="failed", job=current, sequence=sequence, message=message)
                    raise RuntimeError(message)
                artifact_ids = []
                for artifact in current.get("artifacts") or []:
                    path = Path(str(artifact.get("path") or ""))
                    if not path.is_file():
                        raise RuntimeError(f"avatar artifact is missing: {artifact.get('name')}")
                    artifact_ids.append(await asyncio.to_thread(_upload_artifact, callback_url, callback_token, path, artifact))
                await asyncio.to_thread(_post_json, complete_url, callback_token, {"artifactIds": artifact_ids})
                sequence += 1
                await _report_progress(progress_url, callback_token, job_id, status="succeeded", job=current, sequence=sequence, message="Avatar worker completed the job")
                return {"schema_version": 1, "status": "succeeded", "job_id": job_id, "artifact_ids": artifact_ids}
            await asyncio.sleep(2)
        raise TimeoutError("avatar job exceeded the worker timeout")
    finally:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)


def install_salad_routes(app: FastAPI, worker: AvatarWorker, config: AvatarConfig) -> FastAPI:
    """Attach the shared Salad `/process` contract to an avatar app."""

    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "avatar", "avatar-worker"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            return await _run_queue_job(payload, worker, config)
        except HTTPException:
            raise
        except Exception as exc:
            job_id = str(payload.get("job_id") or "unknown")
            print(json.dumps({"event": "salad_queue_job_failed", "job_id": job_id, "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app

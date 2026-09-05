"""Salad queue transport for the generative-dance worker."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import GenerativeDanceConfig
from .worker import GenerativeDanceWorker


TERMINAL = {"succeeded", "failed", "cancelled"}
MAX_COMPLETION_ARTIFACT_IDS = 32


def _callback(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    value = payload.get("callback") or {}
    callback_url = str(value.get("url") or "")
    complete_url = str(value.get("complete_url") or "")
    progress_url = str(value.get("progress_url") or "")
    token = str(value.get("token") or "")
    if not callback_url or not complete_url or not token:
        raise ValueError("job is missing its callback contract")
    return callback_url, complete_url, progress_url, token


def _post_json(url: str, token: str, payload: Any, *, timeout: float = 120) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body)), "X-Job-Callback-Token": token}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raise RuntimeError(f"callback HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"callback failed to {url}: {exc}") from exc


def _upload(url: str, token: str, artifact: dict[str, Any]) -> str:
    path = Path(str(artifact.get("path") or ""))
    if not path.is_file():
        raise RuntimeError(f"artifact is missing: {artifact.get('name')}")
    name = Path(str(artifact.get("name") or path.name)).name
    body = path.read_bytes()
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": str(artifact.get("mediaType") or mimetypes.guess_type(name)[0] or "application/octet-stream"),
            "Content-Length": str(len(body)),
            "X-Job-Callback-Token": token,
            "X-Artifact-Role": "preview" if name.endswith((".mp4", ".webm")) else "metadata",
            "X-Artifact-Variant": str(artifact.get("variant") or "generative-dance-output"),
            "X-Artifact-File-Name": name,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=1800) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"artifact callback HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"artifact callback failed: {exc}") from exc
    artifact_id = str(result.get("id") or result.get("artifactId") or "")
    if not artifact_id:
        raise RuntimeError("artifact callback returned no artifact id")
    return artifact_id


def _failure_url(complete_url: str) -> str:
    marker = "/complete"
    if not complete_url.endswith(marker):
        raise ValueError("generative dance callback complete_url must end with /complete")
    return f"{complete_url[:-len(marker)]}/fail"


def _completion_artifact_ids(uploaded: list[tuple[dict[str, Any], str]]) -> list[str]:
    """Keep callback completion within the launch-server schema limit.

    Every artifact is still uploaded and remains available through the job's
    artifact list. The completion payload only needs to reference the most
    useful subset, with primary outputs preferred so a large diagnostics bundle
    cannot prevent a successful job from reaching its terminal state.
    """

    ordered = sorted(
        uploaded,
        key=lambda item: (
            not bool(item[0].get("primary")),
            str(item[0].get("name") or ""),
        ),
    )
    return [artifact_id for _, artifact_id in ordered[:MAX_COMPLETION_ARTIFACT_IDS]]


async def _progress(url: str, token: str, job_id: str, job: dict[str, Any], sequence: int, *, status: str = "running", message: str | None = None) -> None:
    if not url:
        return
    progress = max(0.0, min(1.0, float(job.get("progress") or 0)))
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "runtime": "wan-animate",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": progress,
        "completed_steps": round(progress * 100),
        "total_steps": 100,
        "attempt": job.get("attempt", 0),
        "message": message or job.get("message"),
        "stage": job.get("stage"),
        "sequence": sequence,
        "updated_at": job.get("updated_at") or "",
    }
    try:
        await asyncio.to_thread(_post_json, url, token, payload, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "wan_progress_callback_failed", "jobId": job_id, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)


class SaladRequestError(RuntimeError):
    """An error that should be returned as a normal worker HTTP response."""

    status_code = 500


async def process_queue_job(payload: dict[str, Any], worker: GenerativeDanceWorker, config: GenerativeDanceConfig) -> dict[str, Any]:
    job_id = str(payload.get("job_id") or "")
    callback_url, complete_url, progress_url, token = _callback(payload)
    job = await worker.submit(payload)
    sequence = 1
    await _progress(progress_url, token, job_id, worker.get(job_id), sequence, message="Wan Animate worker accepted the job")
    last_key: tuple[Any, ...] | None = None
    while True:
        current = worker.get(job_id)
        key = (current.get("status"), current.get("stage"), current.get("progress"), current.get("updated_at"))
        if key != last_key:
            last_key = key
            sequence += 1
            await _progress(progress_url, token, job_id, current, sequence)
        if current.get("status") in TERMINAL:
            if current.get("status") != "succeeded":
                failure = current.get("failure") or {}
                message = f"[{failure.get('code', 'generative_dance_worker_failed')}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'Wan Animate failed')}"
                uploaded: list[tuple[dict[str, Any], str]] = []
                for artifact in current.get("artifacts") or []:
                    try:
                        artifact_id = await asyncio.to_thread(_upload, callback_url, token, artifact)
                        uploaded.append((artifact, artifact_id))
                    except Exception as exc:
                        print(json.dumps({"event": "wan_failure_artifact_upload_failed", "jobId": job_id, "name": artifact.get("name"), "error": str(exc)}), flush=True)
                artifact_ids = _completion_artifact_ids(uploaded)
                try:
                    await asyncio.to_thread(_post_json, _failure_url(complete_url), token, {"errorCode": failure.get("code", "generative_dance_worker_failed"), "errorMessage": message, "artifactIds": artifact_ids})
                except Exception as exc:
                    print(json.dumps({"event": "wan_failure_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence, status="failed", message=message)
                raise RuntimeError(message)
            uploaded = []
            for artifact in current.get("artifacts") or []:
                artifact_id = await asyncio.to_thread(_upload, callback_url, token, artifact)
                uploaded.append((artifact, artifact_id))
            artifact_ids = _completion_artifact_ids(uploaded)
            await asyncio.to_thread(_post_json, complete_url, token, {"artifactIds": artifact_ids})
            sequence += 1
            await _progress(progress_url, token, job_id, current, sequence, status="succeeded", message="Wan Animate worker completed the job")
            return {"schema_version": 1, "runtime": "wan-animate", "status": "succeeded", "job_id": job_id, "artifact_ids": artifact_ids}
        await asyncio.sleep(2)
    raise TimeoutError("Wan Animate job exceeded the worker timeout")

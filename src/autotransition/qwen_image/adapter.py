from __future__ import annotations

import asyncio
import json
import mimetypes
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .config import QwenImageConfig
from .worker import QwenImageWorker
from .contracts import request_from_payload


TERMINAL = {"succeeded", "failed", "cancelled"}


def _job_id(payload: dict[str, Any]) -> str:
    value = str(payload.get("job_id") or "").strip()
    if not value or any(char in value for char in "\\/"):
        raise ValueError("job is missing a valid job_id")
    return value


def _callback(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    value = payload.get("callback") or {}
    callback_url = str(value.get("url") or "")
    complete_url = str(value.get("complete_url") or value.get("completeUrl") or "")
    progress_url = str(value.get("progress_url") or value.get("progressUrl") or "")
    token = str(value.get("token") or "")
    if not callback_url or not complete_url or not token:
        raise ValueError("job is missing its callback contract")
    return callback_url, complete_url, progress_url, token


def _post_json(url: str, token: str, payload: Any, *, timeout: float = 120) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Content-Length": str(len(data)), "X-Job-Callback-Token": token},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"callback HTTP {exc.code}: {detail[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"callback failed to {url}: {exc}") from exc


def _artifact_role(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return "preview" if suffix in {".png", ".jpg", ".jpeg", ".webp"} else "diagnostic"


def _upload_artifact(url: str, token: str, path: Path, artifact: dict[str, Any]) -> str:
    name = Path(str(artifact.get("name") or path.name)).name
    body = path.read_bytes()
    request = UrlRequest(
        url,
        data=body,
        headers={
            "Content-Type": mimetypes.guess_type(name)[0] or "application/octet-stream",
            "Content-Length": str(len(body)),
            "X-Job-Callback-Token": token,
            "X-Artifact-Role": _artifact_role(name),
            "X-Artifact-Variant": "qwen-image-output",
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


async def _progress(url: str, token: str, job_id: str, job: dict[str, Any], sequence: int, message: str | None = None, status: str = "running") -> None:
    if not url:
        return
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "runtime": "qwen-image",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": max(0.0, min(1.0, float(job.get("progress") or 0))),
        "completed_steps": round(float(job.get("progress") or 0) * 100),
        "total_steps": 100,
        "attempt": job.get("attempt", 0),
        "message": message,
        "stage": job.get("stage"),
        "sequence": sequence,
        "updated_at": job.get("updated_at") or "",
    }
    try:
        await asyncio.to_thread(_post_json, url, token, payload, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "qwen_image_progress_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _fail_callback(complete_url: str, token: str, code: str, message: str) -> None:
    fail_url = f"{complete_url.rsplit('/complete', 1)[0]}/fail"
    try:
        await asyncio.to_thread(_post_json, fail_url, token, {"errorCode": code[:120], "errorMessage": message[:4000], "artifactIds": []}, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "qwen_image_failure_callback_failed", "error": str(exc)}), flush=True)


async def _run_queue_job(payload: dict[str, Any], worker: QwenImageWorker, config: QwenImageConfig) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, token = _callback(payload)
    failure_callback_sent = False
    try:
        await _progress(progress_url, token, job_id, {"stage": "accepted", "progress": 0, "attempt": 0}, 1, "Qwen image worker accepted the job")
        job = await worker.submit(request_from_payload(payload, config))
        sequence = 2
        deadline = time.monotonic() + config.job_timeout_seconds + 60
        last_key: tuple[Any, ...] | None = None
        while time.monotonic() < deadline:
            current = worker.get(job.id)
            key = (current.get("status"), current.get("stage"), current.get("progress"), current.get("updated_at"))
            if key != last_key:
                last_key = key
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence)
            if current.get("status") in TERMINAL:
                if current.get("status") != "succeeded":
                    failure = current.get("failure") or {}
                    code = str(failure.get("code") or "qwen_image_worker_failed")
                    message = f"[{code}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'Qwen image generation failed')}"
                    await _fail_callback(complete_url, token, code, message)
                    failure_callback_sent = True
                    sequence += 1
                    await _progress(progress_url, token, job_id, current, sequence, message, "failed")
                    raise RuntimeError(message)
                artifact_ids = []
                for artifact in current.get("artifacts") or []:
                    path = Path(str(artifact.get("path") or ""))
                    if not path.is_file():
                        raise RuntimeError(f"Qwen artifact is missing: {artifact.get('name')}")
                    artifact_ids.append(await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact))
                await asyncio.to_thread(_post_json, complete_url, token, {"artifactIds": artifact_ids})
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence, "Qwen image worker completed the job", "succeeded")
                return {"schema_version": 1, "runtime": "qwen-image", "status": "succeeded", "job_id": job_id, "artifact_ids": artifact_ids}
            await asyncio.sleep(config.poll_interval_seconds)
        raise TimeoutError(f"Qwen image job exceeded the {config.job_timeout_seconds:.0f}s worker timeout")
    except Exception as exc:
        if not failure_callback_sent:
            code = "qwen_image_worker_timeout" if isinstance(exc, TimeoutError) else "qwen_image_worker_failed"
            await _fail_callback(complete_url, token, code, f"[{code}] {exc}")
        raise


def install_process_routes(app: FastAPI, worker: QwenImageWorker, config: QwenImageConfig) -> FastAPI:
    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "qwen-image", "qwen-image-worker", "qwen_image"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            return await _run_queue_job(payload, worker, config)
        except HTTPException:
            raise
        except Exception as exc:
            job_id = str(payload.get("job_id") or "unknown")
            print(json.dumps({"event": "qwen_image_queue_job_failed", "jobId": job_id, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app

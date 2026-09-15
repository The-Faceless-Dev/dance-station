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

from .config import Yue2Config
from .contracts import Yue2Request
from .worker import Yue2Worker


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
    body = json.dumps(payload).encode("utf-8")
    request = UrlRequest(url, data=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body)), "X-Job-Callback-Token": token}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raise RuntimeError(f"callback HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"callback failed to {url}: {exc}") from exc


def _request_value(payload: dict[str, Any], key: str, *aliases: str) -> Any:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("YuE2 job parameters must be an object")
    for source in (parameters, payload):
        for candidate in (key, *aliases):
            if candidate in source:
                return source[candidate]
    return None


def request_from_payload(payload: dict[str, Any]) -> Yue2Request:
    lyrics = _request_value(payload, "lyrics", "text", "prompt")
    if lyrics is None:
        raise ValueError("YuE2 job requires lyrics or text")
    options = _request_value(payload, "request_options", "requestOptions") or {}
    if not isinstance(options, dict):
        raise ValueError("request_options must be an object")
    return Yue2Request(
        lyrics=str(lyrics),
        style=str(_request_value(payload, "style") or "") or None,
        cot=str(_request_value(payload, "cot") or "") or None,
        num_inference_steps=(int(_request_value(payload, "num_inference_steps", "numInferenceSteps")) if _request_value(payload, "num_inference_steps", "numInferenceSteps") is not None else None),
        seed=(int(_request_value(payload, "seed")) if _request_value(payload, "seed") is not None else None),
        duration_seconds=(float(_request_value(payload, "duration_seconds", "durationSeconds")) if _request_value(payload, "duration_seconds", "durationSeconds") is not None else None),
        guidance_scale=(float(_request_value(payload, "guidance_scale", "guidanceScale")) if _request_value(payload, "guidance_scale", "guidanceScale") is not None else None),
        temperature=(float(_request_value(payload, "temperature")) if _request_value(payload, "temperature") is not None else None),
        max_tokens=(int(_request_value(payload, "max_tokens", "maxTokens")) if _request_value(payload, "max_tokens", "maxTokens") is not None else None),
        request_options={str(key): value for key, value in options.items()},
        external_job_id=_job_id(payload),
        payment_intent_id=str(_request_value(payload, "payment_intent_id", "paymentIntentId") or "") or None,
    )


def _artifact_role(name: str) -> str:
    return "audio" if Path(name).name == "audio.wav" else "metadata"


def _upload_artifact(url: str, token: str, path: Path, artifact: dict[str, Any]) -> str:
    name = Path(str(artifact.get("name") or path.name)).name
    body = path.read_bytes()
    role = _artifact_role(name)
    headers = {
        "Content-Type": mimetypes.guess_type(name)[0] or "application/octet-stream",
        "Content-Length": str(len(body)),
        "X-Job-Callback-Token": token,
        "X-Artifact-Role": role,
        "X-Artifact-File-Name": name,
    }
    if role == "audio":
        headers["X-Artifact-Variant"] = "merged"
    request = UrlRequest(
        url,
        data=body,
        headers=headers,
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
    value = max(0.0, min(1.0, float(job.get("progress") or 0)))
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "runtime": "yue2-audio-cpp",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": value,
        "completed_steps": round(value * 100),
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
        print(json.dumps({"event": "yue2_progress_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _fail_callback(complete_url: str, token: str, job_id: str, code: str, message: str, artifact_ids: list[str] | None = None, artifacts: list[dict[str, Any]] | None = None) -> None:
    fail_url = f"{complete_url.rsplit('/complete', 1)[0]}/fail"
    try:
        await asyncio.to_thread(_post_json, fail_url, token, {"errorCode": code[:120], "errorMessage": message[:4000], "artifactIds": artifact_ids or [], "artifacts": artifacts or []}, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "yue2_failure_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _run_queue_job(payload: dict[str, Any], worker: Yue2Worker, config: Yue2Config) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, token = _callback(payload)
    failure_sent = False
    try:
        await _progress(progress_url, token, job_id, {"stage": "accepted", "progress": 0, "attempt": 0}, 1, "YuE2 worker accepted the job")
        job = await worker.submit(request_from_payload(payload))
        sequence = 2
        deadline = time.monotonic() + config.job_timeout_seconds + 60
        last_key: tuple[Any, ...] | None = None
        while time.monotonic() < deadline:
            current = worker.get(job.id)
            key = (current.get("status"), current.get("stage"), current.get("progress"), current.get("updated_at"), current.get("message"))
            if key != last_key:
                last_key = key
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence)
            if current.get("status") in TERMINAL:
                if current.get("status") != "succeeded":
                    failure = current.get("failure") or {}
                    message = f"[{failure.get('code', 'yue2_worker_failed')}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'YuE2 generation failed')}"
                    uploaded: list[str] = []
                    uploaded_meta: list[dict[str, Any]] = []
                    for artifact in current.get("artifacts") or []:
                        path = Path(str(artifact.get("path") or ""))
                        if path.is_file():
                            artifact_id = await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact)
                            uploaded.append(artifact_id)
                            uploaded_meta.append({"id": artifact_id, "artifactId": artifact_id, "name": path.name, "mediaType": artifact.get("media_type"), "sizeBytes": artifact.get("size_bytes"), "sha256": artifact.get("sha256")})
                    await _fail_callback(complete_url, token, job_id, failure.get("code", "yue2_worker_failed"), message, uploaded, uploaded_meta)
                    failure_sent = True
                    await _progress(progress_url, token, job_id, current, sequence + 1, message, "failed")
                    raise RuntimeError(message)
                artifact_ids = []
                uploaded_meta = []
                for artifact in current.get("artifacts") or []:
                    path = Path(str(artifact.get("path") or ""))
                    if not path.is_file():
                        raise RuntimeError(f"YuE2 artifact is missing: {artifact.get('name')}")
                    artifact_id = await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact)
                    artifact_ids.append(artifact_id)
                    uploaded_meta.append({"id": artifact_id, "artifactId": artifact_id, "name": path.name, "mediaType": artifact.get("media_type"), "sizeBytes": artifact.get("size_bytes"), "sha256": artifact.get("sha256")})
                await asyncio.to_thread(_post_json, complete_url, token, {"artifactIds": artifact_ids, "artifacts": uploaded_meta})
                await _progress(progress_url, token, job_id, current, sequence + 1, "YuE2 worker completed the job", "succeeded")
                return {"schema_version": 1, "runtime": "yue2-audio-cpp", "status": "succeeded", "job_id": job_id, "artifact_ids": artifact_ids, "artifacts": uploaded_meta}
            await asyncio.sleep(1)
        raise TimeoutError(f"YuE2 job exceeded the {config.job_timeout_seconds:.0f}s worker timeout")
    except Exception as exc:
        if not failure_sent:
            code = "yue2_worker_timeout" if isinstance(exc, TimeoutError) else "yue2_worker_failed"
            await _fail_callback(complete_url, token, job_id, code, f"[{code}] {exc}")
        raise


def install_callback_routes(app: FastAPI, worker: Yue2Worker, config: Yue2Config) -> FastAPI:
    background_tasks: set[asyncio.Task[Any]] = set()
    app.state.yue2_background_tasks = background_tasks

    def _background_task_done(task: asyncio.Task[Any]) -> None:
        background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            print(json.dumps({"event": "yue2_queue_job_cancelled", "jobId": task.get_name().removeprefix("yue2-job-")}), flush=True)
        except Exception as exc:
            print(json.dumps({"event": "yue2_queue_job_failed", "jobId": task.get_name().removeprefix("yue2-job-"), "errorType": type(exc).__name__, "error": str(exc)}), flush=True)

    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "yue2", "yue2-audio-cpp", "yue2-music", "yue2-worker"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            job_id = _job_id(payload)
            _callback(payload)
        except HTTPException:
            raise
        except Exception as exc:
            job_id = str(payload.get("job_id") or "unknown")
            print(json.dumps({"event": "yue2_queue_job_failed", "jobId": job_id, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        task = asyncio.create_task(_run_queue_job(payload, worker, config), name=f"yue2-job-{job_id}")
        background_tasks.add(task)
        task.add_done_callback(_background_task_done)
        print(json.dumps({"event": "yue2_queue_job_accepted", "jobId": job_id}), flush=True)
        return {"schema_version": 1, "runtime": "yue2-audio-cpp", "status": "accepted", "id": job_id, "job_id": job_id}

    return app

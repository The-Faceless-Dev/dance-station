from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .artifacts import QwenImageEditArtifactStore
from .config import QwenImageEditConfig
from .contracts import request_from_payload
from .runtime import QwenImageEditDiffusersRuntime
from .worker import QwenImageEditWorker, _public_job_payload


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
    return "preview" if suffix in {".png", ".jpg", ".jpeg", ".webp"} else "metadata"


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


async def _progress(
    url: str,
    token: str,
    job_id: str,
    job: dict[str, Any],
    sequence: int,
    message: str | None = None,
    status: str = "running",
) -> None:
    if not url:
        return
    progress = max(0.0, min(1.0, float(job.get("progress") or 0)))
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "runtime": "qwen-image-edit-2511",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": progress,
        "completed_steps": round(progress * 100),
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
        print(json.dumps({"event": "qwen_image_edit_progress_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _fail_callback(complete_url: str, token: str, code: str, message: str) -> None:
    fail_url = f"{complete_url.rsplit('/complete', 1)[0]}/fail"
    try:
        await asyncio.to_thread(
            _post_json,
            fail_url,
            token,
            {"errorCode": code[:120], "errorMessage": message[:4000], "artifactIds": []},
            timeout=10,
        )
    except Exception as exc:
        print(json.dumps({"event": "qwen_image_edit_failure_callback_failed", "error": str(exc)}), flush=True)


async def _run_queue_job(payload: dict[str, Any], worker: QwenImageEditWorker, config: QwenImageEditConfig) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, token = _callback(payload)
    failure_callback_sent = False
    try:
        await _progress(progress_url, token, job_id, {"stage": "accepted", "progress": 0, "attempt": 0}, 1, "Qwen Image Edit worker accepted the job")
        job = await worker.submit(request_from_payload(payload, config))
        sequence = 2
        deadline = time.monotonic() + config.job_timeout_seconds + 60
        last_key: tuple[Any, ...] | None = None
        while time.monotonic() < deadline:
            current = worker.get(job["id"])
            key = (current.get("status"), current.get("stage"), current.get("progress"), current.get("updated_at"))
            if key != last_key:
                last_key = key
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence)
            if current.get("status") in TERMINAL:
                if current.get("status") != "succeeded":
                    failure = current.get("failure") or {}
                    code = str(failure.get("code") or "qwen_image_edit_worker_failed")
                    message = f"[{code}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'Qwen Image Edit generation failed')}"
                    await _fail_callback(complete_url, token, code, message)
                    failure_callback_sent = True
                    sequence += 1
                    await _progress(progress_url, token, job_id, current, sequence, message, "failed")
                    raise RuntimeError(message)
                artifact_ids = []
                for artifact in current.get("artifacts") or []:
                    path = Path(str(artifact.get("path") or ""))
                    if not path.is_file():
                        raise RuntimeError(f"Qwen Image Edit artifact is missing: {artifact.get('name')}")
                    artifact_ids.append(await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact))
                await asyncio.to_thread(_post_json, complete_url, token, {"artifactIds": artifact_ids})
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence, "Qwen Image Edit worker completed the job", "succeeded")
                return {"schema_version": 1, "runtime": "qwen-image-edit-2511", "status": "succeeded", "job_id": job_id, "artifact_ids": artifact_ids}
            await asyncio.sleep(config.poll_interval_seconds)
        raise TimeoutError(f"Qwen Image Edit job exceeded the {config.job_timeout_seconds:.0f}s worker timeout")
    except Exception as exc:
        if not failure_callback_sent:
            code = "qwen_image_edit_worker_timeout" if isinstance(exc, TimeoutError) else "qwen_image_edit_worker_failed"
            await _fail_callback(complete_url, token, code, f"[{code}] {exc}")
        raise


def create_qwen_image_edit_worker_app(
    config: QwenImageEditConfig | None = None,
    runtime: Any | None = None,
) -> FastAPI:
    config = config or QwenImageEditConfig.from_env()
    runtime = runtime or QwenImageEditDiffusersRuntime(config)
    store = QwenImageEditArtifactStore(config.artifact_root)
    interrupted = store.reconcile_interrupted_jobs()
    print(
        json.dumps({
            "event": "qwen_image_edit_worker_starting",
            "runtime": "qwen-image-edit-2511",
            "config": config.to_public_dict(),
            "preflight": runtime.preflight(),
            "interruptedJobs": interrupted,
        }, sort_keys=True),
        flush=True,
    )
    worker = QwenImageEditWorker(config, runtime=runtime)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        worker.shutdown()

    app = FastAPI(title="Qwen Image Edit 2511 Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        report = runtime.preflight()
        return {"ok": bool(report["ready"]), "runtime": "qwen-image-edit-2511", "model": config.model_name, "preflight": report}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        report = runtime.preflight()
        if not report["ready"]:
            raise HTTPException(status_code=503, detail=report)
        return {"ok": True, "runtime": "qwen-image-edit-2511", "model": config.model_name, "preflight": report}

    @app.get("/v1/worker/status")
    async def status() -> dict[str, Any]:
        with worker._lock:
            active = list(worker._futures)
        return {"runtime": "qwen-image-edit-2511", "model": config.model_name, "preflight": runtime.preflight(), "activeJobs": active}

    @app.post("/v1/qwen-image-edit/jobs")
    async def submit(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            job = await worker.submit(request_from_payload(payload, config))
            return _public_job_payload(worker.get(job["id"]))
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/qwen-image-edit/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return _public_job_payload(worker.get(job_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/qwen-image-edit/jobs/{job_id}/artifacts/{artifact_name}")
    async def get_artifact(job_id: str, artifact_name: str):
        try:
            artifact = store.artifact(job_id, artifact_name)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Qwen Image Edit artifact was not found") from exc
        return FileResponse(artifact["path"], media_type=artifact["media_type"], filename=artifact["name"])

    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "qwen-image-edit-2511", "qwen-image-edit", "qwen_image_edit"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            return await _run_queue_job(payload, worker, config)
        except HTTPException:
            raise
        except Exception as exc:
            job_id = str(payload.get("job_id") or "unknown")
            print(json.dumps({"event": "qwen_image_edit_queue_job_failed", "jobId": job_id, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    app.state.qwen_image_edit_worker = worker
    return app


config = QwenImageEditConfig.from_env()
app = create_qwen_image_edit_worker_app(config)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("WORKER_HOST", "0.0.0.0"), port=int(os.getenv("WORKER_PORT", "8080")), log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))

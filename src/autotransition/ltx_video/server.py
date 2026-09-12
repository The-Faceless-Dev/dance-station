from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .artifacts import LtxArtifactStore
from .callback import install_callback_routes, request_from_payload
from .config import LtxVideoConfig
from .runtime import create_runtime
from .worker import LtxVideoWorker


def _public_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    job_id = str(payload.get("id") or "")
    artifacts = []
    for item in payload.get("artifacts") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        clean = {key: value for key, value in item.items() if key != "path"}
        clean["downloadUrl"] = f"/v1/ltx/jobs/{quote(job_id, safe='')}/artifacts/{quote(str(item['name']), safe='')}"
        artifacts.append(clean)
    payload["artifacts"] = artifacts
    return payload


def create_ltx_video_worker_app(config: LtxVideoConfig | None = None, runtime: Any | None = None) -> FastAPI:
    config = config or LtxVideoConfig.from_env()
    store = LtxArtifactStore(config.artifact_root)
    interrupted = store.reconcile_interrupted_jobs()
    runtime = runtime or create_runtime(config)
    print({"event": "ltx_video_worker_starting", "config": config.to_public_dict(), "interruptedJobs": interrupted}, flush=True)
    worker = LtxVideoWorker(config, runtime=runtime)
    app = FastAPI(title="LTX 2.5 Video Worker")
    install_callback_routes(app, worker, config)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        preflight = runtime.preflight()
        return {"ok": bool(preflight.get("ready")), "runtime": "ltx-video", "model": config.model_name, "preflight": preflight}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        preflight = runtime.preflight()
        if preflight.get("ready") is False:
            raise HTTPException(status_code=503, detail=preflight)
        return {"ok": True, "runtime": "ltx-video", "model": config.model_name, "preflight": preflight}

    @app.get("/v1/worker/status")
    async def status() -> dict[str, Any]:
        with worker._lock:
            active = list(worker._futures)
        residency = getattr(runtime, "residency_status", lambda: None)()
        return {"runtime": "ltx-video", "model": config.model_name, "preflight": runtime.preflight(), "residency": residency, "activeJobs": active}

    @app.post("/v1/worker/reset")
    async def reset_worker() -> dict[str, Any]:
        return {"ok": True, "runtime": "ltx-video", "reset": worker.reset_residency()}

    @app.post("/v1/ltx/jobs")
    async def submit(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            job = await worker.submit(request_from_payload({**payload, "job_id": payload.get("job_id") or "direct-" + os.urandom(8).hex()}))
            return _public_job_payload(worker.get(job.id))
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/ltx/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return _public_job_payload(worker.get(job_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/ltx/jobs/{job_id}/artifacts/{artifact_name}")
    async def get_artifact(job_id: str, artifact_name: str) -> FileResponse:
        try:
            artifact = store.artifact(job_id, artifact_name)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="LTX video artifact was not found") from exc
        return FileResponse(artifact.path, media_type=artifact.media_type or mimetypes.guess_type(artifact.name)[0] or "application/octet-stream", filename=artifact.name)

    app.state.ltx_video_worker = worker
    return app


config = LtxVideoConfig.from_env()
app = create_ltx_video_worker_app(config)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("WORKER_HOST", "0.0.0.0"), port=int(os.getenv("WORKER_PORT", "8080")), log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))

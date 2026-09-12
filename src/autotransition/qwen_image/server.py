from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import HTTPException
from fastapi.responses import FileResponse

from .adapter import install_process_routes
from .artifacts import QwenImageArtifactStore
from .config import QwenImageConfig
from .contracts import request_from_payload
from .runtime import QwenImageRuntime
from .worker import QwenImageWorker, _public_job_payload


def create_qwen_image_worker_app(config: QwenImageConfig | None = None, runtime: QwenImageRuntime | None = None):
    config = config or QwenImageConfig.from_env()
    runtime = runtime or QwenImageRuntime(config)
    store = QwenImageArtifactStore(config.artifact_root)
    interrupted = store.reconcile_interrupted_jobs()
    import json

    print(json.dumps({
        "event": "qwen_image_worker_starting",
        "runtime": "qwen-image",
        "config": config.to_public_dict(),
        "preflight": runtime.preflight(),
        "interruptedJobs": interrupted,
    }, sort_keys=True), flush=True)
    worker = QwenImageWorker(config, runtime=runtime)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        worker.shutdown()

    from fastapi import FastAPI

    app = FastAPI(title="Qwen-Image-2512 Worker", lifespan=lifespan)
    install_process_routes(app, worker, config)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        report = runtime.preflight()
        return {"ok": bool(report["ready"]), "runtime": "qwen-image", "model": config.model_name, "preflight": report}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        report = runtime.preflight()
        if not report["ready"]:
            raise HTTPException(status_code=503, detail=report)
        return {"ok": True, "runtime": "qwen-image", "model": config.model_name, "preflight": report}

    @app.get("/v1/worker/status")
    async def status() -> dict[str, Any]:
        with worker._lock:
            active = list(worker._futures)
        return {"runtime": "qwen-image", "model": config.model_name, "preflight": runtime.preflight(), "activeJobs": active}

    @app.post("/v1/qwen-image/jobs")
    async def submit(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            job = await worker.submit(request_from_payload(payload, config))
            return _public_job_payload(worker.get(job.id))
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/qwen-image/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return _public_job_payload(worker.get(job_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/qwen-image/jobs/{job_id}/artifacts/{artifact_name}")
    async def get_artifact(job_id: str, artifact_name: str):
        try:
            artifact = store.artifact(job_id, artifact_name)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Qwen image artifact was not found") from exc
        return FileResponse(artifact.path, media_type=artifact.media_type, filename=artifact.name)

    app.state.qwen_image_worker = worker

    return app


config = QwenImageConfig.from_env()
app = create_qwen_image_worker_app(config)


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("WORKER_HOST", "0.0.0.0"), port=int(os.getenv("WORKER_PORT", "8080")), log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))

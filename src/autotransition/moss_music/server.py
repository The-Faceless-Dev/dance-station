from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .callback import install_callback_routes
from .artifacts import MossMusicArtifactStore
from .config import MossMusicConfig
from .contracts import MossAudioInput, MossMusicRequest
from .runtime import create_runtime
from .worker import MossMusicWorker


def _request_from_http_payload(payload: dict[str, Any]) -> MossMusicRequest:
    audio = payload.get("audio") or {}
    if not isinstance(audio, dict):
        raise ValueError("audio must be an object")
    path = Path(str(audio["path"])).expanduser() if audio.get("path") else None
    return MossMusicRequest(
        audio=MossAudioInput(
            source_url=str(audio.get("source_url") or audio.get("sourceUrl") or audio.get("url") or ""),
            filename=Path(str(audio.get("filename") or audio.get("fileName") or "audio")).name,
            path=path,
        ),
        analysis_profile=str(payload.get("analysis_profile") or payload.get("analysisProfile") or "visual_sync_v1"),
        prompt=str(payload.get("prompt") or "") or None,
        event_resolution_ms=int(payload.get("event_resolution_ms", payload.get("eventResolutionMs", 80))),
        include_semantic_events=bool(payload.get("include_semantic_events", payload.get("includeSemanticEvents", True))),
        include_dense_features=bool(payload.get("include_dense_features", payload.get("includeDenseFeatures", True))),
        max_new_tokens=(
            int(raw_max_new_tokens)
            if (raw_max_new_tokens := payload.get("max_new_tokens", payload.get("maxNewTokens"))) is not None
            else None
        ),
        temperature=float(payload.get("temperature", 0.0)),
        external_job_id=str(payload.get("job_id") or payload.get("external_job_id") or "") or None,
        payment_intent_id=str(payload.get("payment_intent_id") or payload.get("paymentIntentId") or "") or None,
    )


def create_moss_music_worker_app(config: MossMusicConfig | None = None, runtime: Any | None = None) -> FastAPI:
    config = config or MossMusicConfig.from_env()
    store = MossMusicArtifactStore(config.artifact_root)
    interrupted = store.reconcile_interrupted_jobs()
    runtime = runtime or create_runtime(config)
    print(json.dumps({"event": "moss_music_worker_starting", "config": config.to_public_dict(), "interruptedJobs": interrupted}, sort_keys=True, default=str), flush=True)
    worker = MossMusicWorker(config, runtime=runtime)
    app = FastAPI(title="MOSS-Music Analysis Worker")
    install_callback_routes(app, worker, config)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        current = runtime.preflight()
        return {"ok": bool(current.get("ready")), "runtime": "moss-music", "model": config.model_name, "preflight": current}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        current = runtime.preflight()
        if current.get("ready") is False:
            raise HTTPException(status_code=503, detail=current)
        return {"ok": True, "runtime": "moss-music", "model": config.model_name, "preflight": current}

    @app.get("/v1/worker/status")
    async def status() -> dict[str, Any]:
        current = runtime.preflight()
        with worker._lock:
            active = list(worker._futures)
        return {"runtime": "moss-music", "model": config.model_name, "preflight": current, "activeJobs": active}

    @app.post("/v1/moss/jobs")
    async def submit(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            job = await worker.submit(_request_from_http_payload(payload))
            return worker.get(job.id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/moss/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return worker.get(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/moss/jobs/{job_id}/artifacts/{artifact_name}")
    async def get_artifact(job_id: str, artifact_name: str) -> FileResponse:
        try:
            artifact = store.artifact(job_id, artifact_name)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="MOSS-Music artifact was not found") from exc
        return FileResponse(
            artifact.path,
            media_type=artifact.media_type or mimetypes.guess_type(artifact.name)[0] or "application/octet-stream",
            filename=artifact.name,
        )

    app.state.moss_music_worker = worker
    return app


config = MossMusicConfig.from_env()
app = create_moss_music_worker_app(config)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("WORKER_HOST", "0.0.0.0"), port=int(os.getenv("WORKER_PORT", "8080")), log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))

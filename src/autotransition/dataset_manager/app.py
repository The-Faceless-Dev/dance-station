"""FastAPI application for the standalone melodic-vocal dataset manager."""

from __future__ import annotations

import hashlib
import mimetypes
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .app_service import _probe_upload
from .config import DatasetManagerConfig
from .service import DatasetManagerService


def main() -> None:
    """Run the standalone dataset manager using environment configuration."""

    import uvicorn

    config = DatasetManagerConfig.from_env()
    uvicorn.run(create_app(config), host=config.host, port=config.port)


def create_app(config: DatasetManagerConfig | None = None) -> FastAPI:
    service = DatasetManagerService(config)
    static_root = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        service.close()

    app = FastAPI(title="Melodic Dataset Manager", version="1.0", lifespan=lifespan)
    app.state.dataset_service = service
    app.mount("/static", StaticFiles(directory=static_root), name="dataset-static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/dataset/stats")
    def stats() -> dict[str, Any]:
        return service.repository.stats()

    @app.get("/api/dataset/providers")
    def providers() -> dict[str, Any]:
        return service.provider_status()

    @app.get("/api/dataset/songs")
    def songs(search: str = "", status: str = "") -> list[dict[str, Any]]:
        return service.repository.list_songs(search, status)

    @app.post("/api/dataset/songs")
    async def upload_song(
        file: UploadFile = File(...),
        title: str = Form(""),
        artist: str = Form(""),
        language: str = Form("en"),
        lyrics: str = Form(""),
        vocals_only: bool = Form(False),
    ) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(400, "An audio file is required.")
        temporary_path: Path | None = None
        size = 0
        digest = hashlib.sha256()
        try:
            with tempfile.NamedTemporaryFile(prefix="dataset-upload-", suffix=Path(file.filename).suffix, delete=False) as handle:
                temporary_path = Path(handle.name)
                while True:
                    block = await file.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > service.config.max_upload_bytes:
                        raise HTTPException(413, "The uploaded audio file is too large.")
                    digest.update(block)
                    handle.write(block)
            probe = _probe_upload(temporary_path)
            with temporary_path.open("rb") as source:
                song = service.upload_song(filename=file.filename, source=source, title=title, artist=artist, language=language, lyrics=lyrics, vocals_only=vocals_only, probe=probe, source_sha256=digest.hexdigest())
            return song
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)

    @app.get("/api/dataset/songs/{song_id}")
    def song(song_id: str) -> dict[str, Any]:
        value = service.get_song(song_id)
        if not value:
            raise HTTPException(404, "Song not found.")
        return value

    @app.post("/api/dataset/songs/{song_id}/extract")
    def extract(song_id: str) -> dict[str, Any]:
        try:
            return service.start_extraction(song_id)
        except KeyError as exc:
            raise HTTPException(404, "Song not found.") from exc
        except Exception as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.patch("/api/dataset/songs/{song_id}/draft")
    async def save_draft(song_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.update_draft(song_id, payload)
        except KeyError as exc:
            raise HTTPException(404, "Song not found.") from exc
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.patch("/api/dataset/songs/{song_id}/lyrics")
    async def update_lyrics(song_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.update_lyrics(song_id, str(payload.get("lyrics") or ""))
        except KeyError as exc:
            raise HTTPException(404, "Song not found.") from exc
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/dataset/songs/{song_id}/approve")
    def approve(song_id: str) -> dict[str, Any]:
        try:
            return service.approve(song_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/dataset/songs/{song_id}/audio")
    def audio(song_id: str, kind: str = Query("vocals", pattern="^(original|vocals|instrumental)$")) -> FileResponse:
        try:
            path = service.audio_path(song_id, kind)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "audio/wav", filename=path.name)

    @app.get("/api/dataset/songs/{song_id}/waveform")
    def waveform(song_id: str, kind: str = Query("vocals", pattern="^(original|vocals)$")) -> JSONResponse:
        try:
            path = service.waveform_path(song_id, kind)
            import numpy as np

            with np.load(path) as payload:
                return JSONResponse({"peaks": payload["peaks"].tolist(), "rms": payload["rms"].tolist(), "duration_seconds": float(payload["duration_seconds"]), "sample_rate": int(payload["sample_rate"])})
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/dataset/songs/{song_id}/log")
    def extraction_log(song_id: str) -> dict[str, Any]:
        value = service.repository.get_song(song_id)
        if not value or not value.get("active_run_id"):
            raise HTTPException(404, "No extraction run exists.")
        path = service.artifacts.run_dir(song_id, value["active_run_id"]) / "run.log"
        return {"path": service.artifacts.relative(path) if path.exists() else None, "text": path.read_text(encoding="utf-8") if path.exists() else ""}

    @app.get("/api/dataset/songs/{song_id}/extractions")
    def extractions(song_id: str) -> dict[str, Any]:
        value = service.repository.get_song(song_id)
        if not value:
            raise HTTPException(404, "Song not found.")
        return {"song_id": song_id, "runs": service.repository.list_runs(song_id)}

    @app.post("/api/dataset/exports")
    async def export_dataset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            song_ids = (payload or {}).get("song_ids")
            return service.export_dataset(song_ids if isinstance(song_ids, list) else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/dataset/exports/{export_id}/manifest")
    def export_manifest(export_id: str) -> FileResponse:
        path = service.artifacts.export_dir() / export_id / "manifest.json"
        if not path.exists():
            raise HTTPException(404, "Export not found.")
        return FileResponse(path, media_type="application/json", filename="manifest.json")

    return app

"""FastAPI routes for the local generative dance proof-of-concept client."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import PlacementTransform, SegmentPlacement
from autotransition.generative_dance.service import GenerativeDanceService


class PlacementRequest(BaseModel):
    translate_x: float = Field(0.0, ge=-2, le=2)
    translate_y: float = Field(0.0, ge=-2, le=2)
    scale: float = Field(1.0, ge=0.05, le=8)
    rotation_degrees: float = Field(0.0, ge=-180, le=180)


class RenderRequest(BaseModel):
    reference_id: str = Field(..., min_length=1, max_length=100)
    driver_id: str = Field(..., min_length=1, max_length=100)
    prompt: str | None = Field(None, max_length=4000)
    seed: int | None = None
    transparent: bool = True
    placement: PlacementRequest | None = None


class ComposeRequest(BaseModel):
    rendered_ids: list[str] = Field(..., min_length=1, max_length=12)


class DriverCompositionRequest(BaseModel):
    driver_ids: list[str] = Field(..., min_length=1, max_length=12)


def _save_upload(upload: UploadFile, destination: Path, *, max_bytes: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with destination.open("wb") as handle:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"upload exceeds {max_bytes} bytes")
            handle.write(chunk)
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return destination


def _error_response(exc: Exception) -> HTTPException:
    if isinstance(exc, AvatarAdapterError):
        status = 503 if exc.code.endswith("not_configured") else 500
        return HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc), "details": exc.details})
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def register_generative_dance_routes(
    app: FastAPI,
    *,
    static_dir: Path,
    config: GenerativeDanceConfig | None = None,
    ui_log: Any | None = None,
) -> GenerativeDanceService:
    service = GenerativeDanceService(config or GenerativeDanceConfig.from_env())

    def log(level: str, message: str) -> None:
        if ui_log is not None:
            ui_log.add(level, message)

    @app.get("/generative-dance")
    def generative_dance_index() -> FileResponse:
        return FileResponse(static_dir / "generative-dance.html")

    @app.get("/api/generative-dance/status")
    def generative_dance_status() -> dict[str, object]:
        return service.status()

    @app.get("/api/generative-dance/items")
    def generative_dance_items() -> dict[str, object]:
        references: list[dict[str, object]] = []
        drivers: list[dict[str, object]] = []
        renders: list[dict[str, object]] = []
        for manifest in sorted((service.store.root / "references").glob("*/reference.json")):
            try:
                references.append(service.public_reference(service.get_reference(manifest.parent.name)))
            except (OSError, KeyError, ValueError, TypeError):
                continue
        for manifest in sorted((service.store.root / "drivers").glob("*/driver.json")):
            try:
                drivers.append(service.public_driver(service.get_driver(manifest.parent.name)))
            except (OSError, KeyError, ValueError, TypeError):
                continue
        for manifest in sorted((service.store.root / "renders").glob("*/render.json")):
            try:
                renders.append(service.public_render(service.get_rendered_segment(manifest.parent.name)))
            except (OSError, KeyError, ValueError, TypeError):
                continue
        return {"references": references, "drivers": drivers, "renders": renders}

    @app.post("/api/generative-dance/references")
    def create_reference(
        description: str = Form(..., min_length=1, max_length=800),
        seed: int | None = Form(None),
        reference: UploadFile | None = File(None),
    ) -> dict[str, object]:
        upload_path: Path | None = None
        if reference is not None:
            suffix = Path(reference.filename or "reference.png").suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise HTTPException(status_code=400, detail="reference image must be PNG, JPEG, or WebP")
            upload_path = service.store.root / "incoming" / f"reference-{uuid4().hex[:12]}{suffix}"
            _save_upload(reference, upload_path, max_bytes=service.config.max_upload_bytes)
        try:
            result = service.create_reference(description=description, uploaded_image=upload_path, seed=seed)
        except Exception as exc:
            raise _error_response(exc) from exc
        log("info", f"Generative dance reference ready: {result.id}")
        return service.public_reference(result)

    @app.post("/api/generative-dance/drivers")
    def create_driver(
        label: str = Form("Dance driver", min_length=1, max_length=160),
        video: UploadFile = File(...),
    ) -> dict[str, object]:
        suffix = Path(video.filename or "driver.mp4").suffix.lower()
        if suffix not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
            raise HTTPException(status_code=400, detail="dance driver must be an MP4, WebM, MOV, MKV, or AVI video")
        incoming = service.store.root / "incoming" / f"driver-{uuid4().hex[:12]}{suffix}"
        _save_upload(video, incoming, max_bytes=service.config.max_upload_bytes)
        try:
            result = service.create_driver(source=incoming, label=label)
        except Exception as exc:
            raise _error_response(exc) from exc
        log("info", f"Generative dance driver normalized: {result.id}")
        return service.public_driver(result)

    @app.post("/api/generative-dance/render")
    def render_segment(request: RenderRequest) -> dict[str, object]:
        try:
            result = service.render(
                reference_id=request.reference_id,
                driver_id=request.driver_id,
                prompt=request.prompt,
                seed=request.seed,
                transparent=request.transparent,
                placement=(
                    SegmentPlacement(
                        segment_id="request",
                        source_driver_id=request.driver_id,
                        transform=PlacementTransform(
                            translate_x=request.placement.translate_x,
                            translate_y=request.placement.translate_y,
                            scale=request.placement.scale,
                            rotation_degrees=request.placement.rotation_degrees,
                        ),
                    )
                    if request.placement
                    else None
                ),
            )
        except Exception as exc:
            log("error", f"Wan Animate render failed: {exc}")
            raise _error_response(exc) from exc
        log("info", f"Wan Animate segment ready: {result.id}")
        return service.public_render(result)

    @app.post("/api/generative-dance/compose")
    def compose_segments(request: ComposeRequest) -> dict[str, object]:
        try:
            result = service.compose(rendered_ids=request.rendered_ids)
        except Exception as exc:
            log("error", f"Generative dance composition failed: {exc}")
            raise _error_response(exc) from exc
        log("info", f"Generative dance composition ready: {result.id}")
        return service.public_composition(result)

    @app.post("/api/generative-dance/driver-composition")
    def plan_driver_composition(request: DriverCompositionRequest) -> dict[str, object]:
        try:
            result = service.plan_driver_composition(driver_ids=request.driver_ids)
        except Exception as exc:
            log("error", f"Driver composition planning failed: {exc}")
            raise _error_response(exc) from exc
        log("info", f"Driver composition plan ready: {result['id']}")
        return result

    @app.get("/api/generative-dance/files")
    def get_generative_dance_file(path: str = Query(..., min_length=1)) -> FileResponse:
        try:
            artifact = service.store.resolve_relative(path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not artifact.is_file():
            raise HTTPException(status_code=404, detail="generative dance artifact was not found")
        return FileResponse(artifact, media_type=mimetypes.guess_type(artifact.name)[0])

    return service

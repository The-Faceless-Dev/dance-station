"""Motion-driver preparation and boundary metadata extraction."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import BoundaryState, DanceDriver
from autotransition.generative_dance.video import VideoProbe, calculate_normalization_transform, normalize_video, probe_video


def _read_sidecar(source: Path) -> dict[str, Any]:
    candidates = [source.with_suffix(".json"), source.with_name(f"{source.stem}.motion.json")]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _boundary(payload: dict[str, Any], key: str, *, time_seconds: float, config: GenerativeDanceConfig) -> BoundaryState:
    value = payload.get(key)
    if not isinstance(value, dict):
        return BoundaryState(
            time_seconds=time_seconds,
            anchor=(config.canvas.anchor_x, config.canvas.anchor_y),
            subject_bounds=(config.canvas.subject_margin, config.canvas.subject_margin, 1 - config.canvas.subject_margin, 1 - config.canvas.subject_margin),
            foot_floor=config.canvas.floor_y,
            confidence=0.0,
            source="default",
        )
    anchor_value = value.get("anchor") or [config.canvas.anchor_x, config.canvas.anchor_y]
    bounds_value = value.get("subjectBounds") or [config.canvas.subject_margin, config.canvas.subject_margin, 1 - config.canvas.subject_margin, 1 - config.canvas.subject_margin]
    try:
        anchor = (float(anchor_value[0]), float(anchor_value[1]))
        bounds = tuple(float(item) for item in bounds_value[:4])
        if len(bounds) != 4:
            raise ValueError
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key} boundary metadata in motion sidecar") from exc
    return BoundaryState(
        time_seconds=float(value.get("timeSeconds", time_seconds)),
        anchor=anchor,
        subject_bounds=bounds,  # type: ignore[arg-type]
        foot_floor=(
            float(value.get("footFloor", value.get("foot_floor")))
            if value.get("footFloor", value.get("foot_floor")) is not None
            else config.canvas.floor_y
        ),
        pose_signature=str(value["poseSignature"]) if value.get("poseSignature") else None,
        confidence=float(value.get("confidence", 0.0)),
        source=str(value.get("source", "sidecar")),
    )


def prepare_driver(
    source: Path,
    *,
    driver_id: str,
    label: str,
    config: GenerativeDanceConfig,
    store: ArtifactStore,
    output_fps: int | None = None,
) -> DanceDriver:
    if source.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
        raise ValueError("dance driver must be an MP4, WebM, MOV, MKV, or AVI video")
    directory = store.create_id_dir("drivers", driver_id)
    normalized = directory / "normalized-driver.mp4"
    source_metadata = _read_sidecar(source)
    default_bounds = (0.0, 0.0, 1.0, 1.0)
    bounds_value = source_metadata.get("subjectBounds") or source_metadata.get("subject_bounds") or default_bounds
    try:
        subject_bounds = tuple(float(item) for item in bounds_value[:4])
        if len(subject_bounds) != 4:
            raise ValueError
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("motion sidecar subjectBounds must contain four normalized values") from exc
    anchor_value = source_metadata.get("anchor") or [config.canvas.anchor_x, config.canvas.anchor_y]
    try:
        anchor = (float(anchor_value[0]), float(anchor_value[1]))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("motion sidecar anchor must contain two normalized values") from exc
    source_probe = probe_video(source)
    normalized_fps = output_fps or config.canvas.fps
    if normalized_fps < 1 or normalized_fps > 120:
        raise ValueError("driver output frame rate must be between 1 and 120")
    driver_canvas = replace(config.canvas, fps=int(normalized_fps))
    driver_config = replace(config, canvas=driver_canvas)
    normalization = calculate_normalization_transform(
        source_probe,
        width=config.canvas.width,
        height=config.canvas.height,
        anchor=anchor,
        subject_bounds=subject_bounds,  # type: ignore[arg-type]
        subject_margin=config.canvas.subject_margin,
    )
    probe: VideoProbe = normalize_video(
        source,
        normalized,
        width=config.canvas.width,
        height=config.canvas.height,
        fps=driver_canvas.fps,
        pixel_aspect_ratio=driver_canvas.pixel_aspect_ratio,
        anchor=anchor,
        subject_bounds=subject_bounds,  # type: ignore[arg-type]
        subject_margin=config.canvas.subject_margin,
    )
    start = _boundary(source_metadata, "startBoundary", time_seconds=0.0, config=driver_config)
    end = _boundary(source_metadata, "endBoundary", time_seconds=probe.duration_seconds, config=driver_config)
    metadata_path = directory / "driver.json"
    driver = DanceDriver(
        id=driver_id,
        label=label.strip() or driver_id,
        source_video=source,
        normalized_video=normalized,
        duration_seconds=probe.duration_seconds,
        canvas=driver_canvas,
        start_boundary=start,
        end_boundary=end,
        metadata_path=metadata_path,
        source_metadata={"probe": probe.to_dict(), "sidecar": source_metadata, "normalization": normalization},
    )
    store.write_json(metadata_path, driver.to_dict())
    return driver


def copy_source_video(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination

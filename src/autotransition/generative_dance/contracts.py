"""Stable data contracts for the generative avatar dance pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class CanvasContract:
    """The coordinate and timing contract shared by every driver and render."""

    width: int = 480
    height: int = 832
    fps: int = 24
    anchor_x: float = 0.5
    anchor_y: float = 0.58
    anchor_semantic: str = "pelvis"
    floor_y: float = 0.94
    pixel_aspect_ratio: float = 1.0
    subject_margin: float = 0.12

    def validate(self) -> None:
        if self.width < 64 or self.height < 64:
            raise ValueError("canvas dimensions must be at least 64 pixels")
        if self.fps < 1 or self.fps > 120:
            raise ValueError("canvas fps must be between 1 and 120")
        if not 0 <= self.anchor_x <= 1 or not 0 <= self.anchor_y <= 1:
            raise ValueError("canvas anchor must be normalized between 0 and 1")
        if self.anchor_semantic not in {"pelvis", "root", "custom"}:
            raise ValueError("canvas anchor semantic must be pelvis, root, or custom")
        if not 0 <= self.floor_y <= 1:
            raise ValueError("canvas floor must be normalized between 0 and 1")
        if self.pixel_aspect_ratio <= 0:
            raise ValueError("canvas pixel aspect ratio must be positive")
        if not 0 <= self.subject_margin < 0.5:
            raise ValueError("subject margin must be between 0 and 0.5")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryState:
    """A compact description of a driver's start or end continuity window."""

    time_seconds: float
    anchor: tuple[float, float]
    subject_bounds: tuple[float, float, float, float]
    foot_floor: float | None = None
    pose_signature: str | None = None
    confidence: float = 0.0
    source: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class PlacementTransform:
    """A deterministic transform in normalized canonical-canvas coordinates.

    Translation is relative to the untransformed canvas center. Scale and
    rotation are applied around that same center, so the worker never has to
    infer a new position from generated pixels.
    """

    translate_x: float = 0.0
    translate_y: float = 0.0
    scale: float = 1.0
    rotation_degrees: float = 0.0

    def validate(self) -> None:
        if not -2 <= self.translate_x <= 2 or not -2 <= self.translate_y <= 2:
            raise ValueError("placement translation must be between -2 and 2 canvas widths/heights")
        if not 0.05 <= self.scale <= 8:
            raise ValueError("placement scale must be between 0.05 and 8")
        if not -180 <= self.rotation_degrees <= 180:
            raise ValueError("placement rotation must be between -180 and 180 degrees")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class SegmentPlacement:
    """Saved composition placement authored by the dance assembly client."""

    segment_id: str
    source_driver_id: str
    timeline_start_seconds: float = 0.0
    source_start_seconds: float = 0.0
    source_end_seconds: float | None = None
    transform: PlacementTransform = field(default_factory=PlacementTransform)
    anchor_path: tuple[tuple[float, float], ...] = ()
    start_boundary: BoundaryState | None = None
    end_boundary: BoundaryState | None = None

    def validate(self) -> None:
        if not self.segment_id or not self.source_driver_id:
            raise ValueError("segment placement requires segment and driver ids")
        if self.timeline_start_seconds < 0 or self.source_start_seconds < 0:
            raise ValueError("segment placement times cannot be negative")
        if self.source_end_seconds is not None and self.source_end_seconds <= self.source_start_seconds:
            raise ValueError("segment source end must be after its source start")
        self.transform.validate()
        for point in self.anchor_path:
            if len(point) != 2 or not all(0 <= value <= 1 for value in point):
                raise ValueError("segment anchor path points must be normalized x/y pairs")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = _jsonable(asdict(self))
        payload["transform"] = self.transform.to_dict()
        if self.start_boundary:
            payload["start_boundary"] = self.start_boundary.to_dict()
        if self.end_boundary:
            payload["end_boundary"] = self.end_boundary.to_dict()
        return payload


@dataclass(frozen=True)
class AvatarReference:
    id: str
    description: str
    prompt: str
    source_image: Path
    normalized_image: Path
    matte_image: Path | None
    canvas: CanvasContract
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DanceDriver:
    id: str
    label: str
    source_video: Path
    normalized_video: Path
    duration_seconds: float
    canvas: CanvasContract
    start_boundary: BoundaryState
    end_boundary: BoundaryState
    metadata_path: Path
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload["start_boundary"] = self.start_boundary.to_dict()
        payload["end_boundary"] = self.end_boundary.to_dict()
        return payload


@dataclass(frozen=True)
class RenderedSegment:
    id: str
    driver_id: str
    reference_id: str
    output_video: Path
    duration_seconds: float
    canvas: CanvasContract
    metadata_path: Path
    model_revision: str
    prompt: str
    placement: SegmentPlacement | None = None
    matte_video: Path | None = None
    transparent_source_video: Path | None = None
    transparent_placed_video: Path | None = None
    transparent_video: Path | None = None
    transparent_preview_video: Path | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CompositionSegment:
    rendered_segment_id: str
    order: int
    transition_after_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DanceComposition:
    id: str
    segments: tuple[CompositionSegment, ...]
    output_video: Path | None
    canvas: CanvasContract
    metadata_path: Path
    transition_strategy: str = "direct-driver-stitch"
    matte_video: Path | None = None
    transparent_video: Path | None = None
    transparent_preview_video: Path | None = None
    anchor_synchronization: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload

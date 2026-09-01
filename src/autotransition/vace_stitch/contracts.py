"""Validated contracts for one complete VACE dance-stitch job."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _value(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


@dataclass(frozen=True)
class StitchSegment:
    id: str
    input_id: str
    source_start_seconds: float = 0.0
    source_end_seconds: float | None = None
    timeline_start_seconds: float | None = None
    timeline_end_seconds: float | None = None
    label: str = ""

    @classmethod
    def from_payload(cls, value: dict[str, Any], index: int) -> "StitchSegment":
        segment_id = str(_value(value, "id", "segmentId", default=f"segment-{index + 1}")).strip()
        input_id = str(_value(value, "inputId", "input_id", "sourceId", "source_id", default="")).strip()
        if not segment_id or not input_id:
            raise ValueError(f"sequence segment {index + 1} requires id and inputId")
        start = float(_value(value, "sourceStartSeconds", "source_start_seconds", default=0.0))
        raw_end = _value(value, "sourceEndSeconds", "source_end_seconds")
        raw_timeline_start = _value(value, "timelineStartSeconds", "timeline_start_seconds")
        raw_timeline_end = _value(value, "timelineEndSeconds", "timeline_end_seconds")
        segment = cls(
            id=segment_id,
            input_id=input_id,
            source_start_seconds=start,
            source_end_seconds=float(raw_end) if raw_end is not None else None,
            timeline_start_seconds=float(raw_timeline_start) if raw_timeline_start is not None else None,
            timeline_end_seconds=float(raw_timeline_end) if raw_timeline_end is not None else None,
            label=str(_value(value, "title", "label", default="")).strip(),
        )
        segment.validate()
        return segment

    def validate(self) -> None:
        if not self.id or not self.input_id:
            raise ValueError("sequence segments require id and inputId")
        if self.source_start_seconds < 0:
            raise ValueError(f"segment {self.id} source start cannot be negative")
        if self.source_end_seconds is not None and self.source_end_seconds <= self.source_start_seconds:
            raise ValueError(f"segment {self.id} source end must be after source start")
        if self.timeline_start_seconds is not None and self.timeline_start_seconds < 0:
            raise ValueError(f"segment {self.id} timeline start cannot be negative")
        if self.timeline_end_seconds is not None and self.timeline_start_seconds is not None:
            if self.timeline_end_seconds <= self.timeline_start_seconds:
                raise ValueError(f"segment {self.id} timeline end must be after timeline start")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BridgeSpec:
    id: str
    before_segment_id: str
    after_segment_id: str
    duration_seconds: float
    prompt: str | None = None
    context_before_seconds: float | None = None
    context_after_seconds: float | None = None
    enabled: bool = True
    loop: bool = False

    @classmethod
    def from_payload(
        cls,
        value: dict[str, Any],
        *,
        index: int,
        before: StitchSegment,
        after: StitchSegment,
        default_duration: float,
        loop: bool = False,
    ) -> "BridgeSpec":
        bridge_id = str(_value(value, "id", "bridgeId", "transitionId", default=f"bridge-{index + 1}")).strip()
        duration = float(_value(value, "durationSeconds", "duration_seconds", "lengthSeconds", default=default_duration))
        raw_before = _value(value, "contextBeforeSeconds", "context_before_seconds")
        raw_after = _value(value, "contextAfterSeconds", "context_after_seconds")
        spec = cls(
            id=bridge_id,
            before_segment_id=before.id,
            after_segment_id=after.id,
            duration_seconds=duration,
            prompt=(str(_value(value, "prompt", "bridgePrompt", "transitionPrompt", default="")).strip() or None),
            context_before_seconds=float(raw_before) if raw_before is not None else None,
            context_after_seconds=float(raw_after) if raw_after is not None else None,
            enabled=bool(_value(value, "enabled", default=True)),
            loop=loop,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.id or not self.before_segment_id or not self.after_segment_id:
            raise ValueError("bridge requires id and both segment references")
        if self.duration_seconds <= 0:
            raise ValueError(f"bridge {self.id} duration must be positive")
        for name, value in (
            ("context before", self.context_before_seconds),
            ("context after", self.context_after_seconds),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"bridge {self.id} {name} duration must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StitchSequence:
    segments: tuple[StitchSegment, ...]
    bridges: tuple[BridgeSpec, ...]
    loop_bridge: BridgeSpec | None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    background: str | None = None

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any], *, default_duration: float) -> "StitchSequence":
        raw_sequence = parameters.get("sequence")
        if not isinstance(raw_sequence, dict):
            raise ValueError("VACE stitch job parameters.sequence must be an object")
        raw_segments = _value(raw_sequence, "segments", "clips")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("VACE stitch sequence must contain at least one segment")
        segments = tuple(
            StitchSegment.from_payload(value, index)
            for index, value in enumerate(raw_segments)
            if isinstance(value, dict)
        )
        if len(segments) != len(raw_segments):
            raise ValueError("VACE stitch sequence segments must be objects")
        if len({segment.id for segment in segments}) != len(segments):
            raise ValueError("VACE stitch sequence segment ids must be unique")

        raw_bridges = _value(raw_sequence, "bridges", "stitches", "transitions", default=[])
        if raw_bridges is None:
            raw_bridges = []
        if not isinstance(raw_bridges, list):
            raise ValueError("VACE stitch sequence bridges must be an array")
        bridges: list[BridgeSpec] = []
        for index in range(max(0, len(segments) - 1)):
            before, after = segments[index], segments[index + 1]
            supplied = raw_bridges[index] if index < len(raw_bridges) and isinstance(raw_bridges[index], dict) else {}
            bridges.append(
                BridgeSpec.from_payload(
                    supplied,
                    index=index,
                    before=before,
                    after=after,
                    default_duration=default_duration,
                )
            )

        raw_loop = _value(raw_sequence, "loop", "loopBridge", "loop_bridge", default={})
        if raw_loop is None:
            raw_loop = {}
        if isinstance(raw_loop, bool):
            raw_loop = {"enabled": raw_loop}
        if not isinstance(raw_loop, dict):
            raise ValueError("VACE stitch loop must be an object or boolean")
        loop_enabled = bool(_value(raw_loop, "enabled", default=True))
        loop_bridge = None
        if loop_enabled and segments:
            loop_bridge = BridgeSpec.from_payload(
                raw_loop,
                index=len(bridges),
                before=segments[-1],
                after=segments[0],
                default_duration=default_duration,
                loop=True,
            )
        raw_canvas = _value(raw_sequence, "canvas", default={})
        canvas = raw_canvas if isinstance(raw_canvas, dict) else {}
        raw_fps = _value(raw_sequence, "fps", "outputFps", "output_fps")
        raw_width = _value(canvas, "width", default=_value(raw_sequence, "width"))
        raw_height = _value(canvas, "height", default=_value(raw_sequence, "height"))
        background = _value(raw_sequence, "background", "temporaryBackground", "temporary_background")
        return cls(
            segments=segments,
            bridges=tuple(bridges),
            loop_bridge=loop_bridge,
            fps=int(raw_fps) if raw_fps is not None else None,
            width=int(raw_width) if raw_width is not None else None,
            height=int(raw_height) if raw_height is not None else None,
            background=str(background) if background else None,
        )

    def validate(self) -> None:
        if not self.segments:
            raise ValueError("VACE stitch sequence cannot be empty")
        for segment in self.segments:
            segment.validate()
        for bridge in self.bridges:
            bridge.validate()
        if self.loop_bridge:
            self.loop_bridge.validate()
        if self.fps is not None and not 1 <= self.fps <= 120:
            raise ValueError("VACE stitch output FPS must be between 1 and 120")
        if self.width is not None and self.width < 64:
            raise ValueError("VACE stitch output width must be at least 64")
        if self.height is not None and self.height < 64:
            raise ValueError("VACE stitch output height must be at least 64")

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "bridges": [bridge.to_dict() for bridge in self.bridges],
            "loopBridge": self.loop_bridge.to_dict() if self.loop_bridge else None,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "background": self.background,
        }


@dataclass(frozen=True)
class BridgeResult:
    bridge: BridgeSpec
    prompt: str
    requested_gap_frames: int
    actual_gap_frames: int
    context_before_frames: int
    context_after_frames: int
    source_video: str
    source_mask: str
    full_model_output: str
    generated_video: str
    alpha_video: str | None = None
    metadata_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bridge"] = self.bridge.to_dict()
        return payload

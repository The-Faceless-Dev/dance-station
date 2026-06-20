"""Planning for full-song source selection workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from autotransition.config import TransitionConfig

GenerationRegion = Literal["extend", "repaint_existing"]


@dataclass(frozen=True)
class SourceSelectionRequest:
    source_path: Path
    source_duration_seconds: float
    continuation_point_seconds: float
    caption: str
    config: TransitionConfig
    transition_id: str | None = None
    generation_region: GenerationRegion = "extend"
    ace_step_settings: dict[str, object] | None = None


@dataclass(frozen=True)
class SourceSelectionPlan:
    transition_id: str
    source_path: Path
    source_extension: str
    source_format: str
    source_duration_seconds: float
    continuation_point_seconds: float
    tail_start_seconds: float
    tail_end_seconds: float
    scaffold_path: Path
    metadata_path: Path
    caption: str
    context_seconds: float
    repaint_overlap_seconds: float
    new_section_seconds: float
    requested_continuation_seconds: float
    effective_continuation_seconds: float | None
    repainting_start_seconds: float
    repainting_end_seconds: float
    audio_format: str
    bpm_hint: float | None
    key_hint: str | None
    seed: int | None
    generation_region: GenerationRegion = "extend"
    ace_step_settings: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["scaffold_path"] = str(self.scaffold_path)
        data["metadata_path"] = str(self.metadata_path)
        return data


def create_source_selection_plan(request: SourceSelectionRequest) -> SourceSelectionPlan:
    from autotransition.audio.formats import source_extension, source_format_label

    config = request.config
    tail_seconds = config.tail_seconds
    if request.source_duration_seconds <= 0:
        raise ValueError("source_duration_seconds must be greater than 0")
    if request.continuation_point_seconds <= 0:
        raise ValueError("continuation_point_seconds must be greater than 0")
    if request.continuation_point_seconds > request.source_duration_seconds:
        raise ValueError("continuation point cannot be after the source duration")
    if request.continuation_point_seconds < tail_seconds:
        raise ValueError(
            "continuation point is too early for the requested context and repaint overlap "
            f"({tail_seconds:.2f}s required)"
        )
    if request.generation_region == "repaint_existing":
        repaint_end = request.continuation_point_seconds + config.new_section_seconds
        if repaint_end > request.source_duration_seconds:
            available = request.source_duration_seconds - request.continuation_point_seconds
            raise ValueError(
                "not enough source audio after the continuation point to repaint "
                f"{config.new_section_seconds:.2f}s ({available:.2f}s available)"
            )
    elif request.generation_region != "extend":
        raise ValueError(f"Unknown generation region: {request.generation_region}")
    repainting_end_seconds = tail_seconds + config.new_section_seconds

    transition_id = request.transition_id or f"selection-{uuid4().hex[:12]}"
    output_dir = config.output.scaffold_dir / transition_id
    return SourceSelectionPlan(
        transition_id=transition_id,
        source_path=request.source_path,
        source_extension=source_extension(request.source_path),
        source_format=source_format_label(request.source_path),
        source_duration_seconds=request.source_duration_seconds,
        continuation_point_seconds=request.continuation_point_seconds,
        tail_start_seconds=request.continuation_point_seconds - tail_seconds,
        tail_end_seconds=request.continuation_point_seconds,
        scaffold_path=output_dir / f"scaffold.{config.output.audio_format}",
        metadata_path=output_dir / "metadata.json",
        caption=request.caption,
        context_seconds=config.context_seconds,
        repaint_overlap_seconds=config.repaint_overlap_seconds,
        new_section_seconds=config.new_section_seconds,
        requested_continuation_seconds=config.new_section_seconds,
        effective_continuation_seconds=None,
        repainting_start_seconds=config.repainting_start_seconds,
        repainting_end_seconds=repainting_end_seconds,
        audio_format=config.output.audio_format,
        bpm_hint=config.bpm_hint,
        key_hint=config.key_hint,
        seed=config.seed,
        generation_region=request.generation_region,
        ace_step_settings=request.ace_step_settings or {},
    )

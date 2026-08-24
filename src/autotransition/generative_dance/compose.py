"""Driver composition and rendered segment stitching."""

from __future__ import annotations

from pathlib import Path

from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import (
    BoundaryState,
    CompositionSegment,
    DanceComposition,
    DanceDriver,
    PlacementTransform,
    RenderedSegment,
)
from autotransition.generative_dance.video import (
    apply_placement,
    calculate_boundary_sync_transform,
    encode_transparent_video,
    make_transparent_preview,
    stitch_transparent_videos,
    stitch_videos,
    transform_boundary,
)


def _canonical_boundary(config: GenerativeDanceConfig) -> BoundaryState:
    return BoundaryState(
        time_seconds=0.0,
        anchor=(config.canvas.anchor_x, config.canvas.anchor_y),
        subject_bounds=(
            config.canvas.subject_margin,
            config.canvas.subject_margin,
            1 - config.canvas.subject_margin,
            1 - config.canvas.subject_margin,
        ),
        foot_floor=config.canvas.floor_y,
        confidence=1.0,
        source="canonical-canvas",
    )


def build_composition(
    *,
    composition_id: str,
    rendered_segments: list[RenderedSegment],
    drivers: dict[str, DanceDriver] | None = None,
    config: GenerativeDanceConfig,
    store: ArtifactStore,
) -> DanceComposition:
    if not rendered_segments:
        raise ValueError("at least one rendered segment is required")
    for segment in rendered_segments:
        if segment.canvas != config.canvas:
            raise ValueError(f"rendered segment {segment.id} does not match the configured canvas")
    directory = store.create_id_dir("compositions", composition_id)
    output = directory / "composition.mp4"
    stitch_videos(
        [segment.output_video for segment in rendered_segments],
        output,
        width=config.canvas.width,
        height=config.canvas.height,
        fps=config.canvas.fps,
    )
    transparent_output = None
    transparent_preview_output = None
    anchor_reports: list[dict[str, object]] = []
    transparent_segments = [segment.transparent_source_video or segment.transparent_video for segment in rendered_segments]
    if any(path is not None for path in transparent_segments) and not all(path is not None for path in transparent_segments):
        raise ValueError("transparent composition requires every rendered segment to have an alpha video")
    if all(path is not None for path in transparent_segments):
        synchronized_segments: list[Path] = []
        previous_end: BoundaryState | None = None
        driver_map = drivers or {}
        for index, segment in enumerate(rendered_segments):
            driver = driver_map.get(segment.driver_id)
            source_video = segment.transparent_source_video or segment.transparent_placed_video or segment.transparent_video
            if driver is None or source_video is None:
                synchronized_segments.append(source_video)  # type: ignore[arg-type]
                previous_end = None
                anchor_reports.append(
                    {
                        "segmentId": segment.id,
                        "mode": "clip-boundary",
                        "applied": False,
                        "reason": "driver-metadata-unavailable",
                    }
                )
                continue
            authored_transform = segment.placement.transform if segment.placement else PlacementTransform()
            source_start = transform_boundary(driver.start_boundary, authored_transform, canvas=config.canvas)
            source_end = transform_boundary(driver.end_boundary, authored_transform, canvas=config.canvas)
            target = previous_end or _canonical_boundary(config)
            sync_transform, report = calculate_boundary_sync_transform(source_start, target)
            report = {"segmentId": segment.id, **report}
            if any(abs(value) > 0.00001 for value in (sync_transform.translate_x, sync_transform.translate_y)):
                synchronized_path = directory / f"segment-{index + 1:02d}-boundary-synchronized.mov"
                apply_placement(
                    source_video,
                    synchronized_path,
                    width=config.canvas.width,
                    height=config.canvas.height,
                    fps=config.canvas.fps,
                    pixel_aspect_ratio=config.canvas.pixel_aspect_ratio,
                    placement=sync_transform,
                    codec="prores_ks",
                    crf=config.transparent_crf,
                )
                synchronized_segments.append(synchronized_path)
                report["applied"] = True
            else:
                synchronized_segments.append(source_video)
                report["applied"] = False
            previous_end = transform_boundary(source_end, sync_transform, canvas=config.canvas)
            report["finalBoundary"] = previous_end.to_dict()
            anchor_reports.append(report)
        transparent_internal_output = directory / "composition-transparent.mov"
        stitch_transparent_videos(
            synchronized_segments,
            transparent_internal_output,
            width=config.canvas.width,
            height=config.canvas.height,
            fps=config.canvas.fps,
            pixel_aspect_ratio=config.canvas.pixel_aspect_ratio,
            codec="prores_ks",
            crf=config.transparent_crf,
        )
        transparent_output = directory / "composition-transparent.webm"
        encode_transparent_video(
            transparent_internal_output,
            transparent_output,
            codec=config.transparent_codec,
            crf=config.transparent_crf,
        )
        transparent_preview_output = directory / "composition-transparent-preview.mp4"
        make_transparent_preview(
            transparent_internal_output,
            transparent_preview_output,
            width=config.canvas.width,
            height=config.canvas.height,
            fps=config.canvas.fps,
            pixel_aspect_ratio=config.canvas.pixel_aspect_ratio,
        )
    metadata_path = directory / "composition.json"
    composition = DanceComposition(
        id=composition_id,
        segments=tuple(CompositionSegment(rendered_segment_id=segment.id, order=index) for index, segment in enumerate(rendered_segments)),
        output_video=output,
        canvas=config.canvas,
        metadata_path=metadata_path,
        transparent_video=transparent_output,
        transparent_preview_video=transparent_preview_output,
        anchor_synchronization=tuple(anchor_reports),
    )
    store.write_json(metadata_path, composition.to_dict())
    return composition


def build_driver_composition_plan(
    *,
    composition_id: str,
    drivers: list[DanceDriver],
    config: GenerativeDanceConfig,
    store: ArtifactStore,
) -> dict[str, object]:
    """Create an inspectable driver-level chain before model inference."""

    if not drivers:
        raise ValueError("at least one dance driver is required")
    for driver in drivers:
        if driver.canvas != config.canvas:
            raise ValueError(f"driver {driver.id} does not match the configured canvas")
    transitions: list[dict[str, object]] = []
    for previous, current in zip(drivers, drivers[1:]):
        anchor_delta = [
            round(current.start_boundary.anchor[index] - previous.end_boundary.anchor[index], 4)
            for index in range(2)
        ]
        bounds_delta = [
            round(current.start_boundary.subject_bounds[index] - previous.end_boundary.subject_bounds[index], 4)
            for index in range(4)
        ]
        signatures_match = (
            previous.end_boundary.pose_signature is not None
            and current.start_boundary.pose_signature is not None
            and previous.end_boundary.pose_signature == current.start_boundary.pose_signature
        )
        transitions.append(
            {
                "fromDriverId": previous.id,
                "toDriverId": current.id,
                "recommendedTransitionSeconds": 0.5,
                "anchorDelta": anchor_delta,
                "subjectBoundsDelta": bounds_delta,
                "boundaryConfidence": min(previous.end_boundary.confidence, current.start_boundary.confidence),
                "poseSignaturesMatch": signatures_match,
                "strategy": "direct-driver-stitch" if signatures_match else "needs-transition-review",
            }
        )
    directory = store.create_id_dir("driver-compositions", composition_id)
    metadata_path = directory / "driver-composition.json"
    payload: dict[str, object] = {
        "id": composition_id,
        "driverIds": [driver.id for driver in drivers],
        "canvas": config.canvas.to_dict(),
        "transitions": transitions,
        "status": "planned",
        "metadataPath": store.relative(metadata_path),
    }
    store.write_json(metadata_path, payload)
    return payload

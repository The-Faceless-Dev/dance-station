"""Application service used by the local generative dance test client."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.compose import build_composition, build_driver_composition_plan
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import (
    AvatarReference,
    BoundaryState,
    CanvasContract,
    DanceDriver,
    PlacementTransform,
    RenderedSegment,
    SegmentPlacement,
)
from autotransition.generative_dance.driver import prepare_driver
from autotransition.generative_dance.image_pipeline import ReferenceImagePipeline
from autotransition.generative_dance.matting import BiRefNetMattingAdapter
from autotransition.generative_dance.video import (
    apply_placement,
    calculate_boundary_sync_transform,
    make_transparent_preview,
    encode_transparent_video,
    resolve_ffmpeg,
    resolve_ffprobe,
    probe_video,
    transform_boundary,
)
from autotransition.generative_dance.wan_animate import WanAnimate2LiteAdapter


def _placement_from_payload(payload: dict[str, object], *, segment_id: str, driver_id: str) -> SegmentPlacement:
    raw = payload.get("placement")
    if not isinstance(raw, dict):
        return SegmentPlacement(segment_id=segment_id, source_driver_id=driver_id)
    transform_raw = raw.get("transform")
    transform = transform_raw if isinstance(transform_raw, dict) else {}
    anchor_path_raw = raw.get("anchor_path") or raw.get("anchorPath") or []
    anchor_path = tuple(
        (float(point[0]), float(point[1]))
        for point in anchor_path_raw
        if isinstance(point, (list, tuple)) and len(point) == 2
    )
    placement = SegmentPlacement(
        segment_id=str(raw.get("segment_id") or raw.get("segmentId") or segment_id),
        source_driver_id=str(raw.get("source_driver_id") or raw.get("sourceDriverId") or driver_id),
        timeline_start_seconds=float(raw.get("timeline_start_seconds") or raw.get("timelineStartSeconds") or 0),
        source_start_seconds=float(raw.get("source_start_seconds") or raw.get("sourceStartSeconds") or 0),
        source_end_seconds=(
            float(raw.get("source_end_seconds") or raw.get("sourceEndSeconds"))
            if raw.get("source_end_seconds") is not None or raw.get("sourceEndSeconds") is not None
            else None
        ),
        transform=PlacementTransform(
            translate_x=float(transform.get("translate_x") or transform.get("translateX") or 0),
            translate_y=float(transform.get("translate_y") or transform.get("translateY") or 0),
            scale=float(transform.get("scale") or 1),
            rotation_degrees=float(transform.get("rotation_degrees") or transform.get("rotationDegrees") or 0),
        ),
        anchor_path=anchor_path,
    )
    placement.validate()
    return placement


class GenerativeDanceService:
    def __init__(self, config: GenerativeDanceConfig):
        config.validate()
        self.config = config
        self.store = ArtifactStore(config.artifact_root)
        self.references = ReferenceImagePipeline(config, self.store)
        self.wan = WanAnimate2LiteAdapter(config)
        self.matte = BiRefNetMattingAdapter(config)

    def status(self) -> dict[str, object]:
        config_ready = self.config.wan_config_file is None or self.config.wan_config_file.is_file()
        return {
            **self.config.to_public_dict(),
            "ffmpeg": resolve_ffmpeg(),
            "ffprobe": resolve_ffprobe(),
            "wanReady": bool(self.wan.configured and config_ready),
            "birefnetReady": self.matte.configured,
            "transparentOutputReady": bool(self.matte.configured and resolve_ffmpeg() and resolve_ffprobe()),
        }

    def create_reference(
        self,
        *,
        description: str,
        uploaded_image: Path | None = None,
        seed: int | None = None,
        inference_steps: int | None = None,
        text_length: int | None = None,
    ) -> AvatarReference:
        reference_id = f"reference-{uuid4().hex[:12]}"
        return self.references.create(
            reference_id=reference_id,
            description=description,
            uploaded_image=uploaded_image,
            seed=seed,
        )

    def create_driver(
        self,
        *,
        source: Path,
        label: str,
        preserve_source_fps: bool = False,
        output_fps: int | None = None,
    ) -> DanceDriver:
        driver_id = f"driver-{uuid4().hex[:12]}"
        normalized_fps = output_fps
        if normalized_fps is None and preserve_source_fps:
            source_probe = probe_video(source)
            if source_probe.fps > 0:
                normalized_fps = max(1, min(120, int(round(source_probe.fps))))
        return prepare_driver(
            source,
            driver_id=driver_id,
            label=label,
            config=self.config,
            store=self.store,
            output_fps=normalized_fps,
        )

    def render(
        self,
        *,
        reference_id: str,
        driver_id: str,
        prompt: str | None = None,
        seed: int | None = None,
        inference_steps: int | None = None,
        text_length: int | None = None,
        placement: SegmentPlacement | None = None,
        transparent: bool = True,
        continuation_frame: Path | None = None,
    ) -> RenderedSegment:
        reference = self.get_reference(reference_id)
        driver = self.get_driver(driver_id)
        segment_id = f"segment-{uuid4().hex[:12]}"
        output_dir = self.store.create_id_dir("renders", segment_id)
        placement = placement or SegmentPlacement(
            segment_id=segment_id,
            source_driver_id=driver.id,
        )
        if placement.segment_id != segment_id:
            placement = SegmentPlacement(
                segment_id=segment_id,
                source_driver_id=placement.source_driver_id,
                timeline_start_seconds=placement.timeline_start_seconds,
                source_start_seconds=placement.source_start_seconds,
                source_end_seconds=placement.source_end_seconds,
                transform=placement.transform,
                anchor_path=placement.anchor_path,
                start_boundary=placement.start_boundary,
                end_boundary=placement.end_boundary,
            )
        placement.validate()
        result = self.wan.render(
            segment_id=segment_id,
            reference=reference,
            driver=driver,
            output_dir=output_dir,
            prompt=prompt,
            seed=seed,
            inference_steps=inference_steps,
            text_length=text_length,
            continuation_frame=continuation_frame,
        )
        matte_video = None
        transparent_source_video = None
        transparent_video = None
        transparent_placed_video = None
        transparent_preview_video = None
        matte_metadata = None
        transparent_probe = None
        anchor_sync_report: dict[str, object] | None = None
        if transparent:
            matte = self.matte.process(input_video=result.output_video, output_dir=output_dir / "matte")
            matte_video = matte.output_video
            matte_metadata = matte.metadata_path
            transparent_source_video = output_dir / "transparent-placed.mov"
            apply_placement(
                matte.output_video,
                transparent_source_video,
                width=self.config.canvas.width,
                height=self.config.canvas.height,
                fps=driver.canvas.fps,
                pixel_aspect_ratio=self.config.canvas.pixel_aspect_ratio,
                placement=placement.transform,
                codec="prores_ks",
                crf=self.config.transparent_crf,
            )
            if self.config.anchor_sync_enabled:
                source_boundary = transform_boundary(driver.start_boundary, placement.transform, canvas=self.config.canvas)
                target_boundary = BoundaryState(
                    time_seconds=0.0,
                    anchor=(self.config.canvas.anchor_x, self.config.canvas.anchor_y),
                    subject_bounds=(
                        self.config.canvas.subject_margin,
                        self.config.canvas.subject_margin,
                        1 - self.config.canvas.subject_margin,
                        1 - self.config.canvas.subject_margin,
                    ),
                    foot_floor=self.config.canvas.floor_y,
                    confidence=1.0,
                    source="canonical-canvas",
                )
                sync_transform, anchor_sync_report = calculate_boundary_sync_transform(source_boundary, target_boundary)
                if any(
                    abs(value) > 0.00001
                    for value in (sync_transform.translate_x, sync_transform.translate_y)
                ):
                    synchronized_video = output_dir / "transparent-synchronized.mov"
                    apply_placement(
                        transparent_source_video,
                        synchronized_video,
                        width=self.config.canvas.width,
                        height=self.config.canvas.height,
                        fps=driver.canvas.fps,
                        pixel_aspect_ratio=self.config.canvas.pixel_aspect_ratio,
                        placement=sync_transform,
                        codec="prores_ks",
                        crf=self.config.transparent_crf,
                    )
                    transparent_source_video = synchronized_video
                else:
                    anchor_sync_report["applied"] = False
            if anchor_sync_report is None:
                anchor_sync_report = {"mode": "clip-boundary", "enabled": False, "applied": False}
            transparent_placed_video = output_dir / "transparent-placed.webm"
            transparent_probe = encode_transparent_video(
                transparent_source_video,
                transparent_placed_video,
                codec=self.config.transparent_codec,
                crf=self.config.transparent_crf,
            )
            transparent_video = transparent_placed_video
            transparent_preview_video = output_dir / "transparent-preview.mp4"
            make_transparent_preview(
                transparent_source_video,
                transparent_preview_video,
                width=self.config.canvas.width,
                height=self.config.canvas.height,
                fps=driver.canvas.fps,
                pixel_aspect_ratio=self.config.canvas.pixel_aspect_ratio,
            )
            if not self.config.retain_matte_artifacts:
                matte_video.unlink(missing_ok=True)
                matte_metadata.unlink(missing_ok=True)
                matte_video = None
                matte_metadata = None
        payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        payload["outputVideo"] = self.store.relative(result.output_video)
        payload["rgbVideo"] = self.store.relative(result.output_video)
        payload["metadataPath"] = self.store.relative(result.metadata_path)
        payload["placement"] = placement.to_dict()
        payload["transparentRequested"] = transparent
        payload["matteVideo"] = self.store.relative(matte_video) if matte_video else None
        payload["matteMetadataPath"] = self.store.relative(matte_metadata) if matte_metadata else None
        payload["transparentVideo"] = self.store.relative(transparent_video) if transparent_video else None
        payload["transparentSourceVideo"] = self.store.relative(transparent_source_video) if transparent_source_video else None
        payload["transparentPlacedVideo"] = self.store.relative(transparent_placed_video) if transparent_placed_video else None
        payload["transparentPreviewVideo"] = self.store.relative(transparent_preview_video) if transparent_preview_video else None
        payload["transparentProbe"] = transparent_probe.to_dict() if transparent_probe else None
        payload["anchorSynchronization"] = anchor_sync_report
        payload["positionStabilization"] = {
            "enabled": False,
            "replacedBy": "anchorSynchronization",
        }
        self.store.write_json(result.metadata_path, payload)
        return RenderedSegment(
            id=result.id,
            driver_id=result.driver_id,
            reference_id=result.reference_id,
            output_video=result.output_video,
            duration_seconds=result.duration_seconds,
            canvas=result.canvas,
            metadata_path=result.metadata_path,
            model_revision=result.model_revision,
            prompt=result.prompt,
            placement=placement,
            matte_video=matte_video,
            transparent_placed_video=transparent_placed_video,
            transparent_source_video=transparent_source_video,
            transparent_video=transparent_video,
            transparent_preview_video=transparent_preview_video,
        )

    def compose(self, *, rendered_ids: list[str]) -> object:
        segments = [self.get_rendered_segment(item_id) for item_id in rendered_ids]
        composition_id = f"composition-{uuid4().hex[:12]}"
        drivers: dict[str, DanceDriver] = {}
        for segment in segments:
            try:
                drivers[segment.driver_id] = self.get_driver(segment.driver_id)
            except FileNotFoundError:
                # Existing/manual renders may not have a local driver manifest;
                # they remain composable without boundary synchronization.
                continue
        return build_composition(
            composition_id=composition_id,
            rendered_segments=segments,
            drivers=drivers,
            config=self.config,
            store=self.store,
        )

    def plan_driver_composition(self, *, driver_ids: list[str]) -> dict[str, object]:
        drivers = [self.get_driver(driver_id) for driver_id in driver_ids]
        return build_driver_composition_plan(
            composition_id=f"driver-composition-{uuid4().hex[:12]}",
            drivers=drivers,
            config=self.config,
            store=self.store,
        )

    def get_reference(self, reference_id: str) -> AvatarReference:
        path = self.store.create_id_dir("references", reference_id) / "reference.json"
        if not path.is_file():
            raise FileNotFoundError(f"reference was not found: {reference_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AvatarReference(
            id=str(payload["id"]),
            description=str(payload["description"]),
            prompt=str(payload["prompt"]),
            source_image=Path(payload["source_image"]),
            normalized_image=Path(payload["normalized_image"]),
            matte_image=Path(payload["matte_image"]) if payload.get("matte_image") else None,
            canvas=CanvasContract(**payload["canvas"]),
            metadata_path=path,
        )

    def get_driver(self, driver_id: str) -> DanceDriver:
        path = self.store.create_id_dir("drivers", driver_id) / "driver.json"
        if not path.is_file():
            raise FileNotFoundError(f"driver was not found: {driver_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DanceDriver(
            id=str(payload["id"]),
            label=str(payload["label"]),
            source_video=Path(payload["source_video"]),
            normalized_video=Path(payload["normalized_video"]),
            duration_seconds=float(payload["duration_seconds"]),
            canvas=CanvasContract(**payload["canvas"]),
            start_boundary=BoundaryState(
                time_seconds=float(payload["start_boundary"]["time_seconds"]),
                anchor=tuple(payload["start_boundary"]["anchor"]),
                subject_bounds=tuple(payload["start_boundary"]["subject_bounds"]),
                foot_floor=(
                    float(payload["start_boundary"]["foot_floor"])
                    if payload["start_boundary"].get("foot_floor") is not None
                    else self.config.canvas.floor_y
                ),
                pose_signature=payload["start_boundary"].get("pose_signature"),
                confidence=float(payload["start_boundary"].get("confidence", 0.0)),
                source=str(payload["start_boundary"].get("source", "default")),
            ),
            end_boundary=BoundaryState(
                time_seconds=float(payload["end_boundary"]["time_seconds"]),
                anchor=tuple(payload["end_boundary"]["anchor"]),
                subject_bounds=tuple(payload["end_boundary"]["subject_bounds"]),
                foot_floor=(
                    float(payload["end_boundary"]["foot_floor"])
                    if payload["end_boundary"].get("foot_floor") is not None
                    else self.config.canvas.floor_y
                ),
                pose_signature=payload["end_boundary"].get("pose_signature"),
                confidence=float(payload["end_boundary"].get("confidence", 0.0)),
                source=str(payload["end_boundary"].get("source", "default")),
            ),
            metadata_path=path,
            source_metadata=payload.get("source_metadata") or {},
        )

    def get_rendered_segment(self, segment_id: str) -> RenderedSegment:
        path = self.store.create_id_dir("renders", segment_id) / "render.json"
        if not path.is_file():
            raise FileNotFoundError(f"rendered segment was not found: {segment_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        output_path = self.store.resolve_relative(str(payload["outputVideo"]))
        return RenderedSegment(
            id=str(payload["segmentId"]),
            driver_id=str(payload["driverId"]),
            reference_id=str(payload["referenceId"]),
            output_video=output_path,
            duration_seconds=float(payload["probe"]["durationSeconds"]),
            canvas=self.config.canvas,
            metadata_path=path,
            model_revision=str(payload["modelRevision"]),
            prompt=str(payload["prompt"]),
            placement=_placement_from_payload(payload, segment_id=str(payload["segmentId"]), driver_id=str(payload["driverId"])),
            matte_video=self.store.resolve_relative(str(payload["matteVideo"])) if payload.get("matteVideo") else None,
            transparent_source_video=(
                self.store.resolve_relative(str(payload["transparentSourceVideo"]))
                if payload.get("transparentSourceVideo")
                else None
            ),
            transparent_placed_video=(
                self.store.resolve_relative(str(payload["transparentPlacedVideo"]))
                if payload.get("transparentPlacedVideo")
                else None
            ),
            transparent_video=self.store.resolve_relative(str(payload["transparentVideo"])) if payload.get("transparentVideo") else None,
            transparent_preview_video=(
                self.store.resolve_relative(str(payload["transparentPreviewVideo"]))
                if payload.get("transparentPreviewVideo")
                else None
            ),
        )

    def list_items(self, category: str, manifest_name: str) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for path in sorted((self.store.root / category).glob(f"*/{manifest_name}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload["metadataPath"] = self.store.relative(path)
            results.append(payload)
        return results

    def public_reference(self, reference: AvatarReference) -> dict[str, object]:
        return {
            "id": reference.id,
            "description": reference.description,
            "prompt": reference.prompt,
            "sourceImage": self.store.relative(reference.source_image),
            "normalizedImage": self.store.relative(reference.normalized_image),
            "matteImage": self.store.relative(reference.matte_image) if reference.matte_image else None,
            "metadataPath": self.store.relative(reference.metadata_path),
            "canvas": reference.canvas.to_dict(),
        }

    def public_driver(self, driver: DanceDriver) -> dict[str, object]:
        return {
            "id": driver.id,
            "label": driver.label,
            "sourceVideo": self.store.relative(driver.source_video),
            "normalizedVideo": self.store.relative(driver.normalized_video),
            "durationSeconds": driver.duration_seconds,
            "metadataPath": self.store.relative(driver.metadata_path),
            "startBoundary": driver.start_boundary.to_dict(),
            "endBoundary": driver.end_boundary.to_dict(),
            "sourceMetadata": driver.source_metadata,
            "canvas": driver.canvas.to_dict(),
        }

    def public_render(self, render: RenderedSegment) -> dict[str, object]:
        return {
            "id": render.id,
            "driverId": render.driver_id,
            "referenceId": render.reference_id,
            "outputVideo": self.store.relative(render.output_video),
            "rgbVideo": self.store.relative(render.output_video),
            "metadataPath": self.store.relative(render.metadata_path),
            "durationSeconds": render.duration_seconds,
            "modelRevision": render.model_revision,
            "prompt": render.prompt,
            "canvas": render.canvas.to_dict(),
            "placement": render.placement.to_dict() if render.placement else None,
            "matteVideo": self.store.relative(render.matte_video) if render.matte_video else None,
            "transparentPlacedVideo": (
                self.store.relative(render.transparent_placed_video) if render.transparent_placed_video else None
            ),
            "transparentVideo": self.store.relative(render.transparent_video) if render.transparent_video else None,
            "transparentPreviewVideo": (
                self.store.relative(render.transparent_preview_video) if render.transparent_preview_video else None
            ),
        }

    def public_composition(self, composition: object) -> dict[str, object]:
        payload = composition.to_dict()
        if composition.output_video:
            payload["output_video"] = self.store.relative(composition.output_video)
            payload["outputVideo"] = self.store.relative(composition.output_video)
        if composition.transparent_video:
            payload["transparent_video"] = self.store.relative(composition.transparent_video)
            payload["transparentVideo"] = self.store.relative(composition.transparent_video)
        if composition.transparent_preview_video:
            payload["transparent_preview_video"] = self.store.relative(composition.transparent_preview_video)
            payload["transparentPreviewVideo"] = self.store.relative(composition.transparent_preview_video)
        payload["metadataPath"] = self.store.relative(composition.metadata_path)
        return payload

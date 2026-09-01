"""VACE boundary generation for the unified Wan Animate sequence worker."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Callable

from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.video import probe_video

from autotransition.vace_stitch.contracts import BridgeResult, BridgeSpec, StitchSegment
from autotransition.vace_stitch.runtime import VaceRuntime
from autotransition.vace_stitch.video import (
    extract_generated_gap,
    frame_count,
    prepare_firstlastclip,
)
from autotransition.vace_stitch.config import VaceStitchConfig


class UnifiedVaceError(RuntimeError):
    """Raised when a unified sequence cannot produce a VACE boundary."""


def _value(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


class VaceBridgeComposer:
    """Generate only the inter-clip and loop portions of an Animate sequence."""

    def __init__(
        self,
        config: VaceStitchConfig,
        store: ArtifactStore,
        *,
        event: Callable[..., None],
        matte: Any = None,
    ) -> None:
        config.validate()
        self.config = config
        self.store = store
        self.event = event
        self.runtime = VaceRuntime(config)
        self.matte = matte

    @staticmethod
    def _segment(item: dict[str, Any]) -> StitchSegment:
        return StitchSegment(
            id=str(item["segmentId"]),
            input_id=str(item.get("inputId") or item["segmentId"]),
            source_start_seconds=0.0,
            source_end_seconds=float(item["result"].duration_seconds),
            timeline_start_seconds=float(item["timelineStartSeconds"]),
            timeline_end_seconds=float(item["timelineEndSeconds"]),
            label=str(item.get("segmentId") or ""),
        )

    def _supplied_bridge(
        self,
        sequence: dict[str, Any],
        index: int,
        before_id: str,
        after_id: str,
    ) -> dict[str, Any]:
        raw_bridges = sequence.get("bridges") or sequence.get("stitches") or sequence.get("transitions") or []
        if not isinstance(raw_bridges, list):
            raise UnifiedVaceError("sequence bridges must be an array")
        for candidate in raw_bridges:
            if not isinstance(candidate, dict):
                continue
            supplied_before = str(_value(candidate, "beforeSegmentId", "before_segment_id", default=""))
            supplied_after = str(_value(candidate, "afterSegmentId", "after_segment_id", default=""))
            if supplied_before == before_id and supplied_after == after_id:
                return candidate
        if index < len(raw_bridges) and isinstance(raw_bridges[index], dict):
            return raw_bridges[index]
        return {}

    def plan(self, rendered: list[dict[str, Any]], sequence: dict[str, Any]) -> tuple[list[BridgeSpec], BridgeSpec | None]:
        if len(rendered) < 2:
            return [], None
        segments = [self._segment(item) for item in rendered]
        bridges: list[BridgeSpec] = []
        for index, (before, after) in enumerate(zip(segments, segments[1:])):
            gap = max(
                0.0,
                float(after.timeline_start_seconds or 0.0)
                - float(before.timeline_end_seconds or 0.0),
            )
            default_duration = gap if gap >= self.config.min_gap_seconds else self.config.default_gap_seconds
            supplied = self._supplied_bridge(sequence, index, before.id, after.id)
            bridges.append(
                BridgeSpec.from_payload(
                    supplied,
                    index=index,
                    before=before,
                    after=after,
                    default_duration=default_duration,
                )
            )

        raw_loop = _value(sequence, "loop", "loopBridge", "loop_bridge", default={})
        if raw_loop is None:
            raw_loop = {}
        if isinstance(raw_loop, bool):
            raw_loop = {"enabled": raw_loop}
        if not isinstance(raw_loop, dict):
            raise UnifiedVaceError("sequence loop must be an object or boolean")
        enabled = bool(_value(raw_loop, "enabled", default=self.config.loop_enabled))
        if not enabled:
            return bridges, None
        return bridges, BridgeSpec.from_payload(
            raw_loop,
            index=len(bridges),
            before=segments[-1],
            after=segments[0],
            default_duration=float(
                _value(raw_loop, "durationSeconds", "duration_seconds", "lengthSeconds", default=self.config.default_gap_seconds)
            ),
            loop=True,
        )

    def _resolve_prompt(self, parameters: dict[str, Any], bridge: BridgeSpec) -> str:
        candidates = (
            parameters.get("loop_prompt"),
            parameters.get("loopPrompt"),
            self.config.default_loop_prompt,
        ) if bridge.loop else (
            parameters.get("bridge_prompt"),
            parameters.get("bridgePrompt"),
            parameters.get("prompt"),
            self.config.default_prompt,
        )
        for candidate in candidates:
            value = str(candidate or "").strip()
            if value:
                return value
        raise UnifiedVaceError(f"VACE prompt is empty for bridge {bridge.id}")

    def _resolve_seed(self, parameters: dict[str, Any]) -> int:
        raw = _value(parameters, "vace_seed", "vaceSeed")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return secrets.randbits(31)
        value = int(raw)
        if value < 0 or value > 0x7FFFFFFF:
            raise ValueError("VACE seed must be between 0 and 2147483647")
        return value

    def _run_bridge(
        self,
        *,
        job_id: str,
        bridge: BridgeSpec,
        before: dict[str, Any],
        after: dict[str, Any],
        parameters: dict[str, Any],
        output_dir: Path,
        output_width: int,
        output_height: int,
        output_fps: int,
        bridge_index: int,
        job_seed: int,
        transparent: bool,
    ) -> tuple[dict[str, Any], BridgeResult]:
        if not self.config.min_gap_seconds <= bridge.duration_seconds <= self.config.max_gap_seconds:
            raise ValueError(
                f"VACE bridge {bridge.id} duration must be between {self.config.min_gap_seconds:g} "
                f"and {self.config.max_gap_seconds:g} seconds"
            )
        prompt = self._resolve_prompt(parameters, bridge)
        bridge_seed = (job_seed + bridge_index) % (2**31)
        requested_gap_frames = max(1, round(bridge.duration_seconds * self.config.model_fps))
        model_width, model_height = (
            (1280, 720) if self.config.model_size == "720p" else (832, 480)
        )
        prepared = prepare_firstlastclip(
            before["result"].output_video,
            after["result"].output_video,
            output_dir / "prepared",
            width=model_width,
            height=model_height,
            model_fps=self.config.model_fps,
            gap_frames=requested_gap_frames,
            context_before_seconds=bridge.context_before_seconds or self.config.default_context_before_seconds,
            context_after_seconds=bridge.context_after_seconds or self.config.default_context_after_seconds,
            max_window_frames=self.config.max_window_frames,
            background=self.config.temporary_background,
        )
        self.event(
            job_id,
            "vace_window_prepared",
            bridgeId=bridge.id,
            loop=bridge.loop,
            prompt=prompt,
            seed=bridge_seed,
            requestedGapFrames=requested_gap_frames,
            actualGapFrames=prepared.gap_frames,
            contextBeforeFrames=prepared.tail_frames,
            contextAfterFrames=prepared.head_frames,
            totalFrames=prepared.total_frames,
            modelName=self.config.model_name,
            modelSize=self.config.model_size,
        )
        model_output = self.runtime.generate(
            source_video=prepared.source_video,
            source_mask=prepared.source_mask,
            output_dir=output_dir / "model",
            prompt=prompt,
            frame_num=prepared.total_frames,
            seed=bridge_seed,
            sample_steps=int(_value(parameters, "vace_steps", "vaceSteps", "steps", default=self.config.sample_steps)),
            sample_shift=float(_value(parameters, "vace_shift", "vaceShift", "shift", default=self.config.sample_shift)),
            guide_scale=float(_value(parameters, "vace_guidance", "vaceGuidance", "guidance", default=self.config.guide_scale)),
            model_name=str(_value(parameters, "vace_model_name", "vaceModelName", default=self.config.model_name)),
            model_size=str(_value(parameters, "vace_model_size", "vaceModelSize", default=self.config.model_size)),
        )
        bridge_rgb = output_dir / "generated-gap.mp4"
        extract_generated_gap(
            model_output,
            bridge_rgb,
            prepared=prepared,
            output_width=output_width,
            output_height=output_height,
            output_fps=output_fps,
            model_fps=self.config.model_fps,
            background=self.config.temporary_background,
        )
        alpha: Path | None = None
        if transparent:
            if self.matte is None or not self.matte.configured:
                raise UnifiedVaceError(
                    "transparent VACE output requested but the existing BiRefNet matte runtime is not configured"
                )
            alpha = self.matte.process(input_video=bridge_rgb, output_dir=output_dir / "matte").output_video
        probe = probe_video(bridge_rgb)
        metadata = {
            "schemaVersion": 1,
            "runtime": "wan-animate-vace",
            "bridge": bridge.to_dict(),
            "prompt": prompt,
            "seed": bridge_seed,
            "requestedGapFrames": requested_gap_frames,
            "actualGapFrames": prepared.gap_frames,
            "modelFps": self.config.model_fps,
            "outputFps": output_fps,
            "totalWindowFrames": prepared.total_frames,
            "sourceVideo": str(prepared.source_video),
            "sourceMask": str(prepared.source_mask),
            "modelOutput": str(model_output),
            "generatedGap": str(bridge_rgb),
            "alphaVideo": str(alpha) if alpha else None,
            "rgbProbe": probe.to_dict(),
        }
        metadata_path = output_dir / "bridge.json"
        self.store.write_json(metadata_path, metadata)
        result = BridgeResult(
            bridge=bridge,
            prompt=prompt,
            requested_gap_frames=requested_gap_frames,
            actual_gap_frames=prepared.gap_frames,
            context_before_frames=prepared.tail_frames,
            context_after_frames=prepared.head_frames,
            source_video=str(prepared.source_video),
            source_mask=str(prepared.source_mask),
            full_model_output=str(model_output),
            generated_video=str(bridge_rgb),
            alpha_video=str(alpha) if alpha else None,
            metadata_path=str(metadata_path),
        )
        return {"rgb": bridge_rgb, "alpha": alpha, "metadata": metadata}, result

    def run(
        self,
        *,
        job_id: str,
        rendered: list[dict[str, Any]],
        sequence: dict[str, Any],
        parameters: dict[str, Any],
        job_dir: Path,
        output_width: int,
        output_height: int,
        output_fps: int,
        transparent: bool,
        job_seed: int | None = None,
    ) -> tuple[dict[str, dict[str, Any]], list[BridgeResult], list[BridgeSpec], BridgeSpec | None]:
        bridges, loop_bridge = self.plan(rendered, sequence)
        if not bridges and loop_bridge is None:
            return {}, [], bridges, loop_bridge
        seed = self._resolve_seed(parameters) if job_seed is None else job_seed
        all_bridges = [*bridges, *( [loop_bridge] if loop_bridge else [])]
        parts: dict[str, dict[str, Any]] = {}
        results: list[BridgeResult] = []
        by_id = {str(item["segmentId"]): item for item in rendered}
        for index, bridge in enumerate(all_bridges):
            before = by_id[bridge.before_segment_id]
            after = by_id[bridge.after_segment_id]
            output_dir = job_dir / "vace-bridges" / f"{index + 1:03d}-{bridge.id}"
            part, result = self._run_bridge(
                job_id=job_id,
                bridge=bridge,
                before=before,
                after=after,
                parameters=parameters,
                output_dir=output_dir,
                output_width=output_width,
                output_height=output_height,
                output_fps=output_fps,
                bridge_index=index,
                job_seed=seed,
                transparent=transparent,
            )
            parts[bridge.id] = part
            results.append(result)
        return parts, results, bridges, loop_bridge

"""Durable single-GPU worker for complete Wan2.1 VACE stitch jobs."""

from __future__ import annotations

import json
import re
import secrets
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.matting import BiRefNetMattingAdapter
from autotransition.generative_dance.video import (
    encode_transparent_video,
    make_transparent_preview,
    probe_video,
    stitch_transparent_videos,
    stitch_videos,
    transform_alpha_video,
)

from .config import VaceStitchConfig
from .contracts import BridgeResult, BridgeSpec, StitchSegment, StitchSequence
from .diagnostics import analyze_video_seams, part_boundaries
from .enhancement import VaceVideoStage
from .runtime import VaceRuntime
from .video import (
    VaceVideoError,
    extract_generated_gap,
    extract_time_range,
    frame_count,
    native_vace_canvas,
    normalize_canvas,
    prepare_firstlastclip,
)


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return token or fallback


class VaceStitchWorker:
    """Process one full dance sequence at a time and retain diagnostics."""

    def __init__(self, config: VaceStitchConfig, runtime: VaceRuntime | None = None):
        config.validate()
        self.config = config
        self.store = ArtifactStore(config.artifact_root)
        self.runtime = runtime or VaceRuntime(config)
        self.matte = self._build_matte_adapter()
        self.enhancement = VaceVideoStage(config, stage="enhancement")
        self.motion_interpolation = VaceVideoStage(config, stage="motion-interpolation")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vace-stitch-worker")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    def _build_matte_adapter(self) -> BiRefNetMattingAdapter | None:
        if not self.config.matte_command and not self.config.matte_checkpoint:
            return None
        matte_config = GenerativeDanceConfig(
            artifact_root=self.config.artifact_root,
            matte_backend=self.config.matte_backend,
            matte_command=self.config.matte_command,
            matte_cwd=self.config.matte_cwd,
            matte_model=self.config.matte_model,
            matte_checkpoint=self.config.matte_checkpoint,
            matte_python=self.config.matte_python,
            matte_device=self.config.matte_device,
            matte_compute_dtype=self.config.matte_dtype,
            matte_batch_size=self.config.matte_batch_size,
            matte_input_size=self.config.matte_input_size,
            transparent_crf=self.config.transparent_crf,
        )
        return BiRefNetMattingAdapter(matte_config)

    def job_dir(self, job_id: str) -> Path:
        return self.store.create_id_dir("jobs", job_id)

    def _write(self, job_id: str, payload: dict[str, Any]) -> None:
        self.store.write_json(self.job_dir(job_id) / "job.json", payload)

    def get(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"VACE stitch job was not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _event(self, job_id: str, event: str, **details: Any) -> None:
        payload = {"timestamp": _now(), "event": event, "jobId": job_id, **details}
        with (self.job_dir(job_id) / "events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        print(json.dumps(payload, ensure_ascii=True, default=str), flush=True)

    def _state(self, job_id: str, **fields: Any) -> dict[str, Any]:
        payload = self.get(job_id)
        payload.update(fields)
        payload["updated_at"] = _now()
        self._write(job_id, payload)
        return payload

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id or any(char in job_id for char in "\\/"):
            raise ValueError("VACE stitch job is missing a valid job_id")
        try:
            return self.get(job_id)
        except FileNotFoundError:
            pass
        now = _now()
        job = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "attempt": 0,
            "request": payload,
            "created_at": now,
            "updated_at": now,
        }
        self._write(job_id, job)
        self._event(job_id, "job_accepted", runtime="wan-vace-stitch", parameters=payload.get("parameters") or {})
        with self._lock:
            self._futures[job_id] = self.executor.submit(self._run, job_id, payload)
        return job

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _progress(self, job_id: str, stage: str, progress: float, message: str) -> None:
        value = max(0.0, min(1.0, float(progress)))
        self._state(job_id, status="running", stage=stage, progress=value, message=message)
        self._event(job_id, "progress", stage=stage, progress=value, message=message)

    def _input_records(self, payload: dict[str, Any]) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for index, item in enumerate(payload.get("inputs") or []):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"source", "source_clip", "sequence_clip", "dance", "driver", "video"}:
                continue
            input_id = str(item.get("id") or item.get("inputId") or f"clip-{index + 1}").strip()
            url = str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "").strip()
            name = Path(str(item.get("fileName") or item.get("file_name") or f"{input_id}.mp4")).name
            if input_id and url:
                records[input_id] = {"url": url, "name": name}
        return records

    def _download(self, job_id: str, input_id: str, url: str, name: str) -> Path:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"source clip {input_id} must use an HTTP(S) URL")
        suffix = Path(name).suffix.lower() or Path(parsed.path).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError(f"source clip {input_id} must be a video file")
        target = self.job_dir(job_id) / "inputs" / f"{_safe_token(input_id, 'clip')}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        self._event(job_id, "download_started", inputId=input_id, url=url, destination=str(target.relative_to(self.job_dir(job_id))))
        request = Request(url, headers={"Accept": "video/*,application/octet-stream,*/*"})
        with urlopen(request, timeout=300) as response, target.open("wb") as handle:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > self.config.max_upload_bytes:
                raise ValueError(f"source clip {input_id} exceeds the worker size limit")
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.config.max_upload_bytes:
                    raise ValueError(f"source clip {input_id} exceeds the worker size limit")
                handle.write(chunk)
        if total == 0:
            raise ValueError(f"source clip {input_id} was empty")
        self._event(job_id, "download_complete", inputId=input_id, bytes=total, probe=probe_video(target).to_dict())
        return target

    def _resolve_prompt(self, parameters: dict[str, Any], bridge: BridgeSpec) -> str:
        if bridge.prompt:
            return bridge.prompt
        if bridge.loop:
            candidates = (
                parameters.get("loop_prompt"),
                parameters.get("loopPrompt"),
                self.config.default_loop_prompt,
            )
        else:
            candidates = (
                parameters.get("bridge_prompt"),
                parameters.get("bridgePrompt"),
                parameters.get("prompt"),
                self.config.default_prompt,
            )
        for candidate in candidates:
            value = str(candidate or "").strip()
            if value:
                return value
        raise ValueError("VACE stitch bridge prompt cannot be empty")

    @staticmethod
    def _resolve_seed(parameters: dict[str, Any]) -> tuple[int, bool]:
        raw_seed = parameters.get("seed")
        if raw_seed is None or (isinstance(raw_seed, str) and not raw_seed.strip()):
            return secrets.randbits(31), False
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("VACE stitch seed must be an integer when provided") from exc
        if seed < 0 or seed > 0x7FFFFFFF:
            raise ValueError("VACE stitch seed must be between 0 and 2147483647")
        return seed, True

    def _validate_gap(self, bridge: BridgeSpec) -> None:
        if not self.config.min_gap_seconds <= bridge.duration_seconds <= self.config.max_gap_seconds:
            raise ValueError(
                f"bridge {bridge.id} duration must be between {self.config.min_gap_seconds:g} and "
                f"{self.config.max_gap_seconds:g} seconds"
            )

    def _prepare_segment(
        self,
        source: Path,
        segment: StitchSegment,
        output_dir: Path,
        *,
        output_width: int,
        output_height: int,
        output_fps: int,
        transparent: bool,
    ) -> dict[str, Any]:
        source_probe = probe_video(source)
        source_start = segment.source_start_seconds
        source_end = segment.source_end_seconds if segment.source_end_seconds is not None else source_probe.duration_seconds
        if source_start >= source_probe.duration_seconds:
            raise ValueError(f"segment {segment.id} starts after source clip {segment.input_id} ends")
        source_end = min(source_end, source_probe.duration_seconds)
        if source_end <= source_start:
            raise ValueError(f"segment {segment.id} has no usable source frames")
        raw_rgb = output_dir / "source-range.mp4"
        extract_time_range(source, raw_rgb, start_seconds=source_start, end_seconds=source_end, fps=output_fps)
        rgb = output_dir / "source.mp4"
        normalize_canvas(
            raw_rgb,
            rgb,
            width=output_width,
            height=output_height,
            fps=output_fps,
            background=self.config.temporary_background,
        )
        alpha: Path | None = None
        if transparent:
            alpha_candidate = output_dir / "source-alpha-raw.mov"
            if source_probe.has_alpha:
                extract_time_range(
                    source,
                    alpha_candidate,
                    start_seconds=source_start,
                    end_seconds=source_end,
                    fps=output_fps,
                    preserve_alpha=True,
                )
                alpha = output_dir / "source-alpha.mov"
                normalize_canvas(
                    alpha_candidate,
                    alpha,
                    width=output_width,
                    height=output_height,
                    fps=output_fps,
                    preserve_alpha=True,
                )
            elif self.matte is not None:
                alpha = self.matte.process(input_video=rgb, output_dir=output_dir / "source-matte").output_video
            else:
                raise AvatarAdapterError(
                    "vace_source_alpha_unavailable",
                    f"segment {segment.id} is not alpha video and no matting runtime is configured",
                    retryable=False,
                )
        return {
            "segment": segment,
            "source": rgb,
            "alpha": alpha,
            "sourceProbe": source_probe.to_dict(),
            "durationSeconds": probe_video(rgb).duration_seconds,
            "sourceStartSeconds": source_start,
            "sourceEndSeconds": source_end,
        }

    def _run_bridge(
        self,
        job_id: str,
        bridge: BridgeSpec,
        before: dict[str, Any],
        after: dict[str, Any],
        parameters: dict[str, Any],
        output_dir: Path,
        *,
        output_width: int,
        output_height: int,
        output_fps: int,
        model_width: int,
        model_height: int,
        transparent: bool,
        bridge_index: int,
        job_seed: int,
    ) -> tuple[dict[str, Any], BridgeResult]:
        self._validate_gap(bridge)
        prompt = self._resolve_prompt(parameters, bridge)
        bridge_seed = job_seed + bridge_index
        requested_gap_frames = max(1, int(round(bridge.duration_seconds * self.config.model_fps)))
        before_seconds = bridge.context_before_seconds or self.config.default_context_before_seconds
        after_seconds = bridge.context_after_seconds or self.config.default_context_after_seconds
        prepared_dir = output_dir / "prepared"
        prepared = prepare_firstlastclip(
            before["source"],
            after["source"],
            prepared_dir,
            width=model_width,
            height=model_height,
            model_fps=self.config.model_fps,
            gap_frames=requested_gap_frames,
            context_before_seconds=before_seconds,
            context_after_seconds=after_seconds,
            max_window_frames=self.config.max_window_frames,
            background=self.config.temporary_background,
        )
        self._event(
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
            sourceVideo=str(prepared.source_video),
            sourceMask=str(prepared.source_mask),
        )
        model_dir = output_dir / "model"
        model_output = self.runtime.generate(
            source_video=prepared.source_video,
            source_mask=prepared.source_mask,
            output_dir=model_dir,
            prompt=prompt,
            frame_num=prepared.total_frames,
            seed=bridge_seed,
            sample_steps=int(parameters.get("steps", parameters.get("sample_steps", self.config.sample_steps))),
            sample_shift=float(parameters.get("shift", parameters.get("sample_shift", self.config.sample_shift))),
            guide_scale=float(parameters.get("guidance", parameters.get("guide_scale", self.config.guide_scale))),
            model_name=str(parameters.get("model_name") or parameters.get("modelName") or self.config.model_name),
            model_size=str(parameters.get("model_size") or parameters.get("modelSize") or self.config.model_size),
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
        bridge_probe = probe_video(bridge_rgb)
        alpha: Path | None = None
        if transparent:
            if self.matte is None:
                raise AvatarAdapterError(
                    "vace_bridge_alpha_unavailable",
                    "VACE generated a bridge but no matting runtime is configured for transparent output",
                    retryable=False,
                )
            alpha = self.matte.process(input_video=bridge_rgb, output_dir=output_dir / "matte").output_video
        metadata = {
            "schemaVersion": 1,
            "runtime": "wan-vace-stitch",
            "bridge": bridge.to_dict(),
            "prompt": prompt,
            "seed": bridge_seed,
            "requestedGapFrames": requested_gap_frames,
            "actualGapFrames": prepared.gap_frames,
            "modelFps": self.config.model_fps,
            "outputFps": output_fps,
            "generatedGapOutputFrames": frame_count(bridge_probe, output_fps),
            "generatedGapDurationSeconds": bridge_probe.duration_seconds,
            "contextBeforeFrames": prepared.tail_frames,
            "contextAfterFrames": prepared.head_frames,
            "totalWindowFrames": prepared.total_frames,
            "prepared": {
                "sourceVideo": str(prepared.source_video),
                "sourceMask": str(prepared.source_mask),
            },
            "modelOutput": str(model_output),
            "generatedGap": str(bridge_rgb),
            "alphaVideo": str(alpha) if alpha else None,
            "rgbProbe": bridge_probe.to_dict(),
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

    def _artifacts(self, job_id: str, *, primary_names: set[str] | None = None) -> list[dict[str, Any]]:
        root = self.job_dir(job_id)
        primary_names = primary_names or {"dance-stitch.mp4", "dance-stitch.webm", "dance-stitch-preview.mp4"}
        allowed_suffixes = {".mp4", ".webm", ".mov", ".json", ".jsonl", ".log", ".txt"}
        artifacts: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            relative_parts = path.relative_to(root).parts
            is_final = "final" in relative_parts
            is_diagnostic = path.name in {"job-result.json", "failure-summary.json", "events.jsonl", "runtime-diagnostics.json"} or path.suffix.lower() in {".log", ".txt"}
            is_bridge_metadata = path.name == "bridge.json"
            if not (is_final or is_diagnostic or is_bridge_metadata):
                continue
            suffix = path.suffix.lower()
            media_type = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
                ".json": "application/json",
                ".jsonl": "application/jsonl",
                ".log": "text/plain",
                ".txt": "text/plain",
            }.get(suffix, "application/octet-stream")
            artifacts.append(
                {
                    "path": str(path),
                    "name": (
                        "__".join(relative_parts)
                        if is_bridge_metadata or "bridges" in relative_parts
                        else path.name
                    ),
                    "relativePath": self.store.relative(path),
                    "sizeBytes": path.stat().st_size,
                    "mediaType": media_type,
                    "variant": "generative-dance-vace-stitch-final" if is_final else "generative-dance-vace-stitch-diagnostics",
                    "primary": is_final and path.name in primary_names,
                }
            )
        return artifacts

    def _run(self, job_id: str, payload: dict[str, Any]) -> None:
        try:
            self._state(job_id, status="running", attempt=1, stage="validate_inputs", progress=0.01)
            self._event(job_id, "stage_started", stage="validate_inputs", runtime="wan-vace-stitch")
            parameters = payload.get("parameters") or {}
            if not isinstance(parameters, dict):
                raise ValueError("VACE stitch job parameters must be an object")
            job_seed, seed_was_provided = self._resolve_seed(parameters)
            self._state(job_id, seed=job_seed, seedProvided=seed_was_provided)
            self._event(job_id, "vace_seed_resolved", seed=job_seed, provided=seed_was_provided)
            sequence = StitchSequence.from_parameters(parameters, default_duration=self.config.default_gap_seconds)
            sequence.validate()
            output_fps = sequence.fps or self.config.output_fps
            requested_width = sequence.width or self.config.output_width
            requested_height = sequence.height or self.config.output_height
            transparent = bool(parameters.get("transparent", self.config.transparent_default))
            model_size = str(parameters.get("model_size") or parameters.get("modelSize") or self.config.model_size)
            if model_size not in {"480p", "720p"}:
                raise ValueError("VACE stitch model size must be 480p or 720p")
            output_width, output_height = native_vace_canvas(
                model_size,
                portrait=requested_height > requested_width,
            )
            self._event(
                job_id,
                "vace_canvas_resolved",
                modelSize=model_size,
                requestedWidth=requested_width,
                requestedHeight=requested_height,
                resolvedWidth=output_width,
                resolvedHeight=output_height,
                orientation="portrait" if requested_height > requested_width else "landscape",
            )
            model_width, model_height = output_width, output_height
            records = self._input_records(payload)
            missing = [segment.input_id for segment in sequence.segments if segment.input_id not in records]
            if missing:
                raise ValueError("VACE stitch sequence references missing inputs: " + ", ".join(missing))
            if not self.runtime.configured:
                raise AvatarAdapterError(
                    "vace_runtime_not_configured",
                    "Wan2.1 VACE runtime is not configured",
                    retryable=False,
                )
            downloaded: dict[str, Path] = {}
            for input_id, record in records.items():
                downloaded[input_id] = self._download(job_id, input_id, record["url"], record["name"])
            self._progress(job_id, "prepare_segments", 0.08, f"Preparing {len(sequence.segments)} source clips")
            sequence_dir = self.job_dir(job_id) / "sequence"
            prepared_segments: list[dict[str, Any]] = []
            for index, segment in enumerate(sequence.segments):
                segment_dir = sequence_dir / f"segment-{index + 1:03d}-{_safe_token(segment.id, 'segment')}"
                prepared = self._prepare_segment(
                    downloaded[segment.input_id],
                    segment,
                    segment_dir,
                    output_width=output_width,
                    output_height=output_height,
                    output_fps=output_fps,
                    transparent=transparent,
                )
                prepared_segments.append(prepared)
                self._progress(
                    job_id,
                    "prepare_segments",
                    0.08 + 0.12 * ((index + 1) / len(sequence.segments)),
                    f"Prepared source clip {index + 1} of {len(sequence.segments)}",
                )
            bridge_specs = [bridge for bridge in sequence.bridges if bridge.enabled]
            if sequence.loop_bridge and sequence.loop_bridge.enabled:
                bridge_specs.append(sequence.loop_bridge)
            bridge_results: list[BridgeResult] = []
            bridge_parts: dict[str, dict[str, Any]] = {}
            for index, bridge in enumerate(bridge_specs):
                before_index = next(
                    (item for item, prepared in enumerate(prepared_segments) if prepared["segment"].id == bridge.before_segment_id),
                    None,
                )
                after_index = next(
                    (item for item, prepared in enumerate(prepared_segments) if prepared["segment"].id == bridge.after_segment_id),
                    None,
                )
                if before_index is None or after_index is None:
                    raise ValueError(f"bridge {bridge.id} references an unknown segment")
                bridge_dir = self.job_dir(job_id) / "bridges" / f"{index + 1:03d}-{_safe_token(bridge.id, 'bridge')}"
                self._progress(
                    job_id,
                    "vace_bridge",
                    0.20 + 0.65 * (index / max(1, len(bridge_specs))),
                    f"Generating transition {index + 1} of {len(bridge_specs)}",
                )
                part, result = self._run_bridge(
                    job_id,
                    bridge,
                    prepared_segments[before_index],
                    prepared_segments[after_index],
                    parameters,
                    bridge_dir,
                    output_width=output_width,
                    output_height=output_height,
                    output_fps=output_fps,
                    model_width=model_width,
                    model_height=model_height,
                    transparent=transparent,
                    bridge_index=index,
                    job_seed=job_seed,
                )
                bridge_parts[bridge.id] = part
                bridge_results.append(result)
                self._progress(
                    job_id,
                    "vace_bridge",
                    0.20 + 0.65 * ((index + 1) / max(1, len(bridge_specs))),
                    f"Completed transition {index + 1} of {len(bridge_specs)}",
                )

            self._progress(job_id, "compose", 0.90, "Assembling source clips and generated transitions")
            final_dir = self.job_dir(job_id) / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            rgb_parts: list[Path] = []
            alpha_parts: list[Path] = []
            normal_bridges_by_before: dict[str, BridgeSpec] = {}
            for bridge in sequence.bridges:
                if not bridge.enabled:
                    continue
                if bridge.before_segment_id in normal_bridges_by_before:
                    raise ValueError(
                        f"multiple enabled bridges start at segment {bridge.before_segment_id}"
                    )
                normal_bridges_by_before[bridge.before_segment_id] = bridge
            timeline_inputs: list[dict[str, Any]] = []
            used_bridge_ids: list[str] = []

            def append_timeline_part(kind: str, part_id: str, path: Path, *, loop: bool = False) -> None:
                probe = probe_video(path)
                frames = frame_count(probe, output_fps)
                timeline_inputs.append(
                    {
                        "kind": kind,
                        "id": part_id,
                        "loop": loop,
                        "path": str(path),
                        "probe": probe.to_dict(),
                        "frameCount": frames,
                        "durationSeconds": frames / float(output_fps),
                    }
                )
                rgb_parts.append(path)

            for index, prepared in enumerate(prepared_segments):
                segment_id = prepared["segment"].id
                append_timeline_part("segment", segment_id, prepared["source"])
                if transparent:
                    if prepared["alpha"] is None:
                        raise RuntimeError(f"segment {segment_id} has no alpha output")
                    alpha_parts.append(prepared["alpha"])
                bridge = normal_bridges_by_before.get(segment_id)
                if bridge is not None:
                    part = bridge_parts[bridge.id]
                    append_timeline_part("bridge", bridge.id, part["rgb"], loop=bridge.loop)
                    used_bridge_ids.append(bridge.id)
                    if transparent:
                        if part["alpha"] is None:
                            raise RuntimeError(f"bridge {bridge.id} has no alpha output")
                        alpha_parts.append(part["alpha"])
            if sequence.loop_bridge and sequence.loop_bridge.enabled:
                loop_part = bridge_parts[sequence.loop_bridge.id]
                append_timeline_part("bridge", sequence.loop_bridge.id, loop_part["rgb"], loop=True)
                used_bridge_ids.append(sequence.loop_bridge.id)
                if transparent:
                    if loop_part["alpha"] is None:
                        raise RuntimeError("loop bridge has no alpha output")
                    alpha_parts.append(loop_part["alpha"])
            expected_bridge_ids = [bridge.id for bridge in bridge_specs]
            if sorted(used_bridge_ids) != sorted(expected_bridge_ids) or len(used_bridge_ids) != len(set(used_bridge_ids)):
                raise VaceVideoError(
                    "VACE timeline did not include every generated bridge exactly once: "
                    f"expected={expected_bridge_ids} used={used_bridge_ids}"
                )
            timeline_path = final_dir / "timeline-inputs.json"
            self.store.write_json(
                timeline_path,
                {
                    "schemaVersion": 1,
                    "outputFps": output_fps,
                    "outputWidth": output_width,
                    "outputHeight": output_height,
                    "inputs": timeline_inputs,
                    "expectedBridgeIds": expected_bridge_ids,
                    "usedBridgeIds": used_bridge_ids,
                },
            )
            final_rgb = final_dir / "dance-stitch.mp4"
            part_frame_counts = [item["frameCount"] for item in timeline_inputs]
            expected_frame_count = sum(part_frame_counts)
            final_probe = stitch_videos(
                rgb_parts,
                final_rgb,
                width=output_width,
                height=output_height,
                fps=output_fps,
            )
            actual_frame_count = frame_count(final_probe, output_fps)
            if actual_frame_count != expected_frame_count:
                raise VaceVideoError(
                    "VACE final composition changed the timeline frame count: "
                    f"expected={expected_frame_count} actual={actual_frame_count}"
                )
            seam_boundaries = part_boundaries(part_frame_counts)
            seam_reports = analyze_video_seams(final_rgb, seam_boundaries)

            quality_rgb = final_rgb
            stage_results: list[dict[str, Any]] = []
            quality_fps = float(output_fps)
            for stage in (self.enhancement, self.motion_interpolation):
                if not stage.enabled:
                    continue
                self._event(job_id, "vace_video_stage_started", stage=stage.stage, input=str(quality_rgb))
                stage_result = stage.process(
                    input_video=quality_rgb,
                    output_dir=final_dir / stage.stage,
                    width=output_width,
                    height=output_height,
                    fps=output_fps,
                )
                quality_rgb = stage_result.output_video
                quality_fps = float(stage_result.probe.fps) if stage_result.probe is not None else quality_fps
                stage_results.append(stage_result.to_dict())
                self._event(job_id, "vace_video_stage_completed", stage=stage.stage, output=str(quality_rgb))
            quality_probe = final_probe if quality_rgb == final_rgb else probe_video(quality_rgb)
            expected_duration = expected_frame_count / float(output_fps)
            # ffmpeg reports a stream duration using container timestamps. A
            # stage such as RIFE intentionally emits N + (N - 1) frames when
            # doubling cadence, so the reported duration can differ from the
            # source's N/fps convention by one output frame without changing
            # the playable timeline. Allow two cadence frames for timestamp
            # rounding, while still rejecting a real stage-length change.
            duration_tolerance = 2.0 / max(quality_fps, 1.0)
            if abs(quality_probe.duration_seconds - expected_duration) > duration_tolerance:
                raise VaceVideoError(
                    "VACE video stage changed the timeline duration: "
                    f"expected={expected_duration:.6f}s actual={quality_probe.duration_seconds:.6f}s"
                )
            quality_frame_count = frame_count(quality_probe, max(1, round(quality_fps)))
            quality_part_frame_counts = [
                max(1, round(part_count * quality_fps / float(output_fps)))
                for part_count in part_frame_counts
            ]
            quality_seams = (
                analyze_video_seams(quality_rgb, part_boundaries(quality_part_frame_counts))
                if quality_rgb != final_rgb
                else seam_reports
            )
            final_alpha: Path | None = None
            final_webm: Path | None = None
            final_preview: Path | None = None
            if transparent:
                if len(alpha_parts) != len(rgb_parts):
                    raise RuntimeError("transparent stitch part count does not match RGB part count")
                final_alpha = final_dir / "dance-stitch-alpha.mov"
                stitch_transparent_videos(
                    alpha_parts,
                    final_alpha,
                    width=output_width,
                    height=output_height,
                    fps=output_fps,
                    codec="prores_ks",
                    crf=self.config.transparent_crf,
                )
                if quality_rgb != final_rgb:
                    self._event(
                        job_id,
                        "vace_alpha_timeline_started",
                        source=str(final_alpha),
                        target=str(quality_rgb),
                        targetFps=round(quality_fps),
                        targetFrames=quality_frame_count,
                    )
                    final_alpha = transform_alpha_video(
                        final_alpha,
                        final_dir / "quality-alpha" / "alpha.mov",
                        width=quality_probe.width,
                        height=quality_probe.height,
                        fps=max(1, round(quality_fps)),
                        frame_count=quality_frame_count,
                        crf=self.config.transparent_crf,
                    ).path
                    self._event(
                        job_id,
                        "vace_alpha_timeline_completed",
                        output=str(final_alpha),
                    )
                final_webm = final_dir / "dance-stitch.webm"
                encode_transparent_video(final_alpha, final_webm, codec=self.config.transparent_codec, crf=self.config.transparent_crf)
                final_preview = final_dir / "dance-stitch-preview.mp4"
                make_transparent_preview(
                    final_alpha,
                    final_preview,
                    width=quality_probe.width,
                    height=quality_probe.height,
                    fps=max(1, round(quality_fps)),
                )
            result_metadata = {
                "schemaVersion": 1,
                "runtime": "wan-vace-stitch",
                "modelName": self.config.model_name,
                "modelSize": model_size,
                "canvas": {
                    "requestedWidth": requested_width,
                    "requestedHeight": requested_height,
                    "resolvedWidth": output_width,
                    "resolvedHeight": output_height,
                    "orientation": "portrait" if requested_height > requested_width else "landscape",
                },
                "singleParentJob": True,
                "seed": job_seed,
                "seedProvided": seed_was_provided,
                "promptDefaults": {
                    "bridge": self.config.default_prompt,
                    "loop": self.config.default_loop_prompt,
                },
                "sequence": sequence.to_dict(),
                "output": {
                    "rgb": str(quality_rgb),
                    "rgbMaster": str(final_rgb),
                    "alpha": str(final_webm) if final_webm else None,
                    "preview": str(final_preview) if final_preview else None,
                    "fps": quality_probe.fps,
                    "width": quality_probe.width,
                    "height": quality_probe.height,
                },
                "timeline": {
                    "partFrameCounts": part_frame_counts,
                    "expectedFrameCount": expected_frame_count,
                    "actualFrameCount": actual_frame_count,
                    "qualityFrameCount": quality_frame_count,
                    "qualityPartFrameCounts": quality_part_frame_counts,
                    "seamBoundaries": seam_boundaries,
                    "seams": seam_reports,
                    "qualitySeams": quality_seams,
                    "outputProbe": final_probe.to_dict(),
                    "qualityProbe": quality_probe.to_dict(),
                },
                "videoStages": stage_results,
                "segments": [
                    {
                        "segment": prepared["segment"].to_dict(),
                        "source": str(prepared["source"]),
                        "alpha": str(prepared["alpha"]) if prepared["alpha"] else None,
                        "durationSeconds": prepared["durationSeconds"],
                    }
                    for prepared in prepared_segments
                ],
                "bridges": [result.to_dict() for result in bridge_results],
                "completedAt": _now(),
            }
            metadata_path = final_dir / "job-result.json"
            self.store.write_json(metadata_path, result_metadata)
            self._progress(job_id, "finalize_artifacts", 0.97, "Validating final VACE stitch artifacts")
            primary_names = {"dance-stitch.mp4", "dance-stitch.webm", "dance-stitch-preview.mp4", Path(quality_rgb).name}
            artifacts = self._artifacts(job_id, primary_names=primary_names)
            if not artifacts:
                raise RuntimeError("VACE stitch completed without final artifacts")
            self._state(
                job_id,
                status="succeeded",
                stage="finalizing",
                progress=1.0,
                result=result_metadata,
                artifacts=artifacts,
            )
            self._event(job_id, "job_succeeded", artifactCount=len(artifacts), artifactNames=[item["name"] for item in artifacts])
        except Exception as exc:
            current = self.get(job_id)
            details = getattr(exc, "details", {})
            if not isinstance(details, dict):
                details = {}
            failure = {
                "code": getattr(exc, "code", "vace_stitch_worker_failed"),
                "message": str(exc) or type(exc).__name__,
                "stage": current.get("stage") or "validate_inputs",
                "attempt": int(current.get("attempt") or 1),
                "errorType": type(exc).__name__,
                "details": details,
                "traceback": traceback.format_exc(),
            }
            failure_path = self.job_dir(job_id) / "failure-summary.json"
            self.store.write_json(
                failure_path,
                {
                    "schemaVersion": 1,
                    "runtime": "wan-vace-stitch",
                    "jobId": job_id,
                    "request": payload,
                    "failure": failure,
                    "eventLog": "events.jsonl",
                    "createdAt": _now(),
                },
            )
            artifacts = self._artifacts(job_id)
            self._state(
                job_id,
                status="failed",
                stage=failure["stage"],
                progress=1.0,
                failure=failure,
                refundRequired=True,
                refundReason="wan_vace_stitch_failed",
                artifacts=artifacts,
            )
            self._event(job_id, "job_failed", failure=failure)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
            self._event(job_id, "job_thread_finished")


def create_vace_stitch_worker(config: VaceStitchConfig) -> VaceStitchWorker:
    return VaceStitchWorker(config)

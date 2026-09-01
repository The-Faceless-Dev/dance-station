"""Durable single-GPU worker for full-length Wan Animate jobs."""

from __future__ import annotations

import json
import secrets
import re
import shutil
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import ArtifactStore
from .config import GenerativeDanceConfig
from .contracts import PlacementTransform, SegmentPlacement
from .identity import audit_segment_identity
from .service import GenerativeDanceService, _placement_from_payload
from .video import (
    encode_transparent_video,
    extract_video_range,
    make_blank_video,
    make_transparent_preview,
    probe_video,
    stitch_transparent_videos,
    stitch_videos,
    transform_alpha_video,
)
from .vace_bridge import VaceBridgeComposer
from autotransition.vace_stitch.config import VaceStitchConfig
from autotransition.vace_stitch.enhancement import VaceVideoStage
from autotransition.vace_stitch.video import frame_count as vace_frame_count


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sequence_segments_are_adjacent(
    timeline_start: float,
    previous_timeline_end: float | None,
    *,
    fps: int,
) -> bool:
    """Return whether Wan context can cross a sequence boundary."""

    if previous_timeline_end is None:
        return False
    return abs(timeline_start - previous_timeline_end) <= (1.0 / fps + 0.01)


def _resolve_identity_seed(parameters: dict[str, Any]) -> int:
    """Resolve one seed for the whole request, rather than one per segment."""

    configured = parameters.get("seed")
    if configured is not None:
        return int(configured) % (2**32)
    return secrets.randbelow(2**32)


def _resolve_reference_strength(
    config: GenerativeDanceConfig,
    parameters: dict[str, Any],
) -> float:
    configured = parameters.get("reference_strength", parameters.get("referenceStrength"))
    value = config.wan_reference_strength if configured is None else float(configured)
    if not 0 < value <= 5:
        raise ValueError("Wan-Animate-2 reference strength must be greater than 0 and at most 5")
    return value


class GenerativeDanceWorker:
    """Run one heavyweight video job at a time and retain all diagnostics."""

    def __init__(self, config: GenerativeDanceConfig, service: GenerativeDanceService | None = None):
        self.config = config
        self.store = ArtifactStore(config.artifact_root)
        self.service = service or GenerativeDanceService(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan-animate-worker")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        return self.store.create_id_dir("jobs", job_id)

    def _write(self, job_id: str, payload: dict[str, Any]) -> None:
        path = self.job_dir(job_id) / "job.json"
        self.store.write_json(path, payload)

    def get(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"generative dance job was not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _event(self, job_id: str, event: str, **details: Any) -> None:
        payload = {"timestamp": _now(), "event": event, "jobId": job_id, **details}
        path = self.job_dir(job_id) / "events.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
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
            raise ValueError("job is missing a valid job_id")
        try:
            existing = self.get(job_id)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            return existing
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
        self._event(job_id, "job_accepted", runtime=payload.get("runtime"), parameters=payload.get("parameters") or {})
        with self._lock:
            self._futures[job_id] = self.executor.submit(self._run, job_id, payload)
        return job

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _progress(self, job_id: str, stage: str, progress: float, message: str) -> None:
        value = max(0.0, min(1.0, float(progress)))
        self._state(job_id, status="running", stage=stage, progress=value, message=message)
        self._event(job_id, "progress", stage=stage, progress=value, message=message)

    @staticmethod
    def _input(payload: dict[str, Any], roles: set[str], default_name: str) -> tuple[str | None, str]:
        for item in payload.get("inputs") or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in roles:
                continue
            url = str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "").strip()
            name = Path(str(item.get("fileName") or item.get("file_name") or default_name)).name
            return url or None, name
        parameters = payload.get("parameters") or {}
        if isinstance(parameters, dict):
            for key in ("reference_image_url", "referenceImageUrl") if "reference" in roles else ("driver_video_url", "driverVideoUrl"):
                value = str(parameters.get(key) or "").strip()
                if value:
                    return value, default_name
        return None, default_name

    @staticmethod
    def _input_records(payload: dict[str, Any], role: str) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for item in payload.get("inputs") or []:
            if not isinstance(item, dict) or str(item.get("role") or "").strip().lower() != role:
                continue
            input_id = str(item.get("id") or role).strip()
            url = str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "").strip()
            name = Path(str(item.get("fileName") or item.get("file_name") or f"{role}.bin")).name
            if input_id and url:
                records[input_id] = {"url": url, "name": name}
        return records

    @staticmethod
    def _safe_token(value: str, fallback: str) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
        return token or fallback

    def _download(self, job_id: str, url: str, name: str, *, prefix: str, suffixes: set[str]) -> Path:
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{prefix} input must use an HTTP(S) URL")
        suffix = Path(name).suffix.lower() or Path(parsed.path).suffix.lower()
        if suffix not in suffixes:
            raise ValueError(f"{prefix} input must use one of: {', '.join(sorted(suffixes))}")
        target = self.job_dir(job_id) / f"{self._safe_token(prefix, 'input')}{suffix}"
        total = 0
        self._event(job_id, "download_started", input=prefix, url=url, destination=target.name)
        request = Request(url, headers={"Accept": "*/*"})
        with urlopen(request, timeout=300) as response, target.open("wb") as handle:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > self.config.max_upload_bytes:
                raise ValueError(f"{prefix} input exceeds the worker size limit")
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.config.max_upload_bytes:
                    raise ValueError(f"{prefix} input exceeds the worker size limit")
                handle.write(chunk)
        if total == 0:
            raise ValueError(f"{prefix} input was empty")
        self._event(job_id, "download_complete", input=prefix, bytes=total)
        return target

    def _artifacts(self, job_id: str, render_dir: Path) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        job_root = self.job_dir(job_id)
        # These are request inputs staged at the job root and removed in the
        # worker's finally block. They must not be advertised for callback
        # upload, or Salad will retry a job after inference already succeeded.
        transient_inputs = {job_root / "reference.png", job_root / "reference.jpg", job_root / "reference.jpeg", job_root / "reference.webp"}
        transient_inputs.update(job_root.glob("driver-*"))
        for path in sorted(render_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path in transient_inputs:
                continue
            relative = self.store.relative(path)
            suffix = path.suffix.lower()
            if suffix in {".mp4", ".webm", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".json", ".log", ".txt"}:
                primary = "00-final" in path.parts or path.name.startswith("generative-dance-output")
                artifacts.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "relativePath": relative,
                        "sizeBytes": path.stat().st_size,
                        "mediaType": {
                            ".mp4": "video/mp4",
                            ".webm": "video/webm",
                            ".png": "image/png",
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".json": "application/json",
                            ".log": "text/plain",
                            ".txt": "text/plain",
                        }.get(suffix, "application/octet-stream"),
                        "variant": "generative-dance-final" if primary else "generative-dance-diagnostics",
                        "primary": primary,
                    }
                )
        return artifacts

    def _run_sequence(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        parameters: dict[str, Any],
        reference_id: str,
        identity_seed: int | None = None,
        reference_strength: float | None = None,
    ) -> tuple[dict[str, Any], Path]:
        """Render the client sequence and assemble one final timeline."""

        sequence = parameters.get("sequence")
        if not isinstance(sequence, dict):
            raise ValueError("generative dance sequence must be an object")
        raw_segments = sequence.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("generative dance sequence must contain at least one segment")
        sequence_fps = int(round(float(sequence.get("fps") or self.config.canvas.fps)))
        if sequence_fps < 1 or sequence_fps > 120:
            raise ValueError("generative dance sequence FPS must be between 1 and 120")
        if identity_seed is None:
            identity_seed = _resolve_identity_seed(parameters)
        if reference_strength is None:
            reference_strength = _resolve_reference_strength(self.config, parameters)

        driver_inputs = self._input_records(payload, "driver")
        if not driver_inputs:
            raise ValueError("Wan Animate sequence has no driver inputs")
        downloaded: dict[str, Path] = {}
        for input_id, record in driver_inputs.items():
            downloaded[input_id] = self._download(
                job_id,
                record["url"],
                record["name"],
                prefix=f"driver-{input_id}",
                suffixes={".mp4", ".webm", ".mov", ".mkv", ".avi"},
            )
        self._event(job_id, "sequence_inputs_ready", driverCount=len(downloaded), fps=sequence_fps)

        sequence_dir = self.job_dir(job_id) / "sequence"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        reference = self.service.get_reference(reference_id) if self.config.identity_audit_enabled else None
        rendered: list[dict[str, Any]] = []
        ordered_segments = sorted(
            (segment for segment in raw_segments if isinstance(segment, dict)),
            key=lambda segment: float(segment.get("timelineStartSeconds") or 0.0),
        )
        previous_timeline_end: float | None = None
        tolerance = 1.0 / sequence_fps + 0.01
        total = len(ordered_segments)
        for index, segment in enumerate(ordered_segments):
            segment_id = str(segment.get("id") or f"segment-{index + 1}")
            input_id = str(segment.get("inputId") or "")
            source = downloaded.get(input_id)
            if source is None:
                raise ValueError(f"sequence segment {segment_id} references an unknown driver input: {input_id}")
            source_start = float(segment.get("sourceStartSeconds") or 0.0)
            source_end = float(segment.get("sourceEndSeconds") or 0.0)
            timeline_start = float(segment.get("timelineStartSeconds") or 0.0)
            timeline_end = float(segment.get("timelineEndSeconds") or timeline_start + (source_end - source_start))
            if source_end <= source_start or timeline_end <= timeline_start:
                raise ValueError(f"sequence segment {segment_id} has an invalid time range")
            if timeline_start < 0:
                raise ValueError(f"sequence segment {segment_id} starts before zero")

            segment_dir = sequence_dir / f"{index + 1:03d}-{self._safe_token(segment_id, 'segment')}"
            segment_source = segment_dir / "source.mp4"
            source_probe = probe_video(source)
            if source_start >= source_probe.duration_seconds:
                raise ValueError(f"sequence segment {segment_id} starts after its source video ends")
            bounded_end = min(source_end, source_probe.duration_seconds)
            extract_video_range(
                source,
                segment_source,
                start_seconds=source_start,
                end_seconds=bounded_end,
                output_fps=sequence_fps,
            )
            segment_sidecar = segment_source.with_suffix(".json")
            segment_sidecar.write_text(
                json.dumps(
                    {
                        "anchor": segment.get("anchor"),
                        "subjectBounds": segment.get("subjectBounds"),
                        "startBoundary": segment.get("startBoundary"),
                        "endBoundary": segment.get("endBoundary"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            driver = self.service.create_driver(
                source=segment_source,
                label=str(segment.get("title") or segment_id),
                output_fps=sequence_fps,
            )
            placement_payload = dict(segment)
            placement_payload["placement"] = {
                **(segment.get("placement") if isinstance(segment.get("placement"), dict) else {}),
                "segmentId": segment_id,
                "sourceDriverId": driver.id,
                "timelineStartSeconds": timeline_start,
                "sourceStartSeconds": 0.0,
                "sourceEndSeconds": driver.duration_seconds,
            }
            placement = _placement_from_payload(
                placement_payload,
                segment_id=segment_id,
                driver_id=driver.id,
            )
            adjacent = _sequence_segments_are_adjacent(timeline_start, previous_timeline_end, fps=sequence_fps)
            self._progress(
                job_id,
                "wan_render",
                0.18 + (0.70 * index / max(1, total)),
                f"Rendering dance segment {index + 1} of {total}",
            )
            base_seed = identity_seed
            audit_enabled = self.config.identity_audit_enabled and index > 0
            audit_reports: list[dict[str, Any]] = []
            result = None
            for attempt in range(self.config.identity_audit_max_retries + 1):
                effective_seed = base_seed + attempt
                render_seed = effective_seed
                self._event(
                    job_id,
                    "wan_segment_attempt",
                    segmentId=segment_id,
                    segmentIndex=index,
                    attempt=attempt,
                    seed=render_seed,
                    seedScope="job",
                    referenceStrength=reference_strength,
                    crossSegmentContinuationFrames=0,
                )
                result = self.service.render(
                    reference_id=reference_id,
                    driver_id=driver.id,
                    prompt=str(segment.get("prompt") or parameters.get("prompt") or "a person performing a full-body dance"),
                    seed=render_seed,
                    inference_steps=int(parameters.get("steps", parameters.get("num_inference_steps", self.config.wan_inference_steps))),
                    text_length=int(parameters.get("text_length", parameters.get("textLength", self.config.wan_text_length))),
                    reference_strength=reference_strength,
                    placement=placement,
                    transparent=bool(parameters.get("transparent", True)),
                )
                if not audit_enabled:
                    audit_reports.append(
                        {
                            "enabled": False,
                            "passed": True,
                            "reason": "first-source-segment" if index == 0 else "disabled",
                            "attempt": attempt,
                            "seed": render_seed,
                            "referenceStrength": reference_strength,
                        }
                    )
                    break
                if reference is None:
                    raise RuntimeError("identity audit reference was not loaded")
                audit = audit_segment_identity(
                    self.config,
                    reference_image=reference.normalized_image,
                    render_video=result.output_video,
                    output_dir=sequence_dir / f"{index + 1:03d}-{self._safe_token(segment_id, 'segment')}" / f"identity-attempt-{attempt + 1:02d}",
                    segment_id=segment_id,
                    seed=effective_seed,
                    attempt=attempt,
                )
                audit_reports.append(audit)
                self._event(job_id, "wan_identity_audit", segmentId=segment_id, **audit)
                if audit["passed"]:
                    break
                if attempt < self.config.identity_audit_max_retries:
                    self._event(
                        job_id,
                        "wan_identity_retry",
                        segmentId=segment_id,
                        attempt=attempt,
                        nextAttempt=attempt + 1,
                        score=audit["score"],
                        threshold=audit["threshold"],
                    )
            if result is None:
                raise RuntimeError(f"Wan Animate produced no result for segment {segment_id}")
            if self.config.identity_audit_enabled and not audit_reports[-1].get("passed", False):
                raise AvatarAdapterError(
                    "wan_identity_audit_failed",
                    f"Wan Animate identity audit failed for segment {segment_id} after {len(audit_reports)} attempt(s)",
                    retryable=False,
                    details={"segmentId": segment_id, "audits": audit_reports},
                )
            previous_timeline_end = timeline_end
            rendered.append(
                {
                    "index": index,
                    "segmentId": segment_id,
                    "inputId": input_id,
                    "timelineStartSeconds": timeline_start,
                    "timelineEndSeconds": timeline_end,
                    "adjacentToPrevious": adjacent,
                    "continuity": {
                        "policy": "within-source-segment-only",
                        "crossSegmentContinuationUsed": False,
                        "crossSegmentContinuationFrames": 0,
                        "withinSegmentContextFrames": self.config.wan_temporal_context_frames,
                        "identityAudit": audit_reports,
                    },
                    "driver": driver,
                    "result": result,
                }
            )

        final_dir = self.job_dir(job_id) / "00-final"
        final_dir.mkdir(parents=True, exist_ok=True)
        rgb_inputs: list[Path] = []
        alpha_inputs: list[Path] = []
        has_alpha = bool(parameters.get("transparent", True))
        vace_config = VaceStitchConfig.from_env()
        vace_parts: dict[str, dict[str, Any]] = {}
        vace_results: list[Any] = []
        vace_bridges: list[Any] = []
        vace_loop = None
        if vace_config.enabled and len(rendered) > 1:
            self._progress(job_id, "vace_bridge", 0.80, "Generating VACE transitions between dance clips")
            composer = VaceBridgeComposer(
                vace_config,
                self.store,
                event=self._event,
                matte=self.service.matte,
            )
            vace_parts, vace_results, vace_bridges, vace_loop = composer.run(
                job_id=job_id,
                rendered=rendered,
                sequence=sequence,
                parameters=parameters,
                job_dir=self.job_dir(job_id),
                output_width=self.config.canvas.width,
                output_height=self.config.canvas.height,
                output_fps=sequence_fps,
                transparent=has_alpha,
                job_seed=None,
            )
            self._event(
                job_id,
                "vace_sequence_completed",
                bridgeCount=len(vace_results),
                loopEnabled=bool(vace_loop),
                modelName=vace_config.model_name,
                modelSize=vace_config.model_size,
            )

        for index, item in enumerate(rendered):
            if vace_parts and index > 0:
                bridge = vace_bridges[index - 1]
                part = vace_parts[bridge.id]
                rgb_inputs.append(part["rgb"])
                if has_alpha:
                    if part["alpha"] is None:
                        raise RuntimeError(f"VACE bridge {bridge.id} has no alpha output")
                    alpha_inputs.append(part["alpha"])
            elif not vace_parts:
                timeline_start = float(item["timelineStartSeconds"])
                cursor = max(
                    (float(previous["timelineEndSeconds"]) for previous in rendered[:index]),
                    default=0.0,
                )
                if timeline_start > cursor + tolerance:
                    gap = timeline_start - cursor
                    rgb_inputs.append(
                        make_blank_video(
                            final_dir / f"gap-{len(rgb_inputs):03d}.mp4",
                            width=self.config.canvas.width,
                            height=self.config.canvas.height,
                            fps=sequence_fps,
                            duration_seconds=gap,
                        ).path
                    )
                    if has_alpha:
                        alpha_inputs.append(
                            make_blank_video(
                                final_dir / f"gap-{len(alpha_inputs):03d}.mov",
                                width=self.config.canvas.width,
                                height=self.config.canvas.height,
                                fps=sequence_fps,
                                duration_seconds=gap,
                                transparent=True,
                                crf=self.config.transparent_crf,
                            ).path
                        )
            result = item["result"]
            rgb_inputs.append(result.output_video)
            alpha_path = result.transparent_source_video or result.transparent_video
            if has_alpha and alpha_path is not None:
                alpha_inputs.append(alpha_path)
        if vace_loop is not None:
            loop_part = vace_parts[vace_loop.id]
            rgb_inputs.append(loop_part["rgb"])
            if has_alpha:
                if loop_part["alpha"] is None:
                    raise RuntimeError("VACE loop bridge has no alpha output")
                alpha_inputs.append(loop_part["alpha"])
        if not rgb_inputs:
            raise RuntimeError("sequence rendering produced no output videos")
        final_rgb = final_dir / "generative-dance-output.mp4"
        final_probe = stitch_videos(
            rgb_inputs,
            final_rgb,
            width=self.config.canvas.width,
            height=self.config.canvas.height,
            fps=sequence_fps,
        )

        # Postprocessing is deliberately after the complete RGB timeline is
        # composed. The alpha timeline is transformed to the resulting frame
        # cadence below so transparency stays frame-aligned.
        quality_rgb = final_rgb
        quality_probe = final_probe
        video_stages: list[dict[str, Any]] = []
        for stage in (
            VaceVideoStage(vace_config, stage="enhancement"),
            VaceVideoStage(vace_config, stage="motion-interpolation"),
        ):
            if not stage.enabled:
                continue
            self._event(job_id, "vace_video_stage_started", stage=stage.stage, input=str(quality_rgb))
            stage_result = stage.process(
                input_video=quality_rgb,
                output_dir=final_dir / stage.stage,
                width=self.config.canvas.width,
                height=self.config.canvas.height,
                fps=sequence_fps,
            )
            quality_rgb = stage_result.output_video
            quality_probe = stage_result.probe or probe_video(quality_rgb)
            video_stages.append(stage_result.to_dict())
            self._event(job_id, "vace_video_stage_completed", stage=stage.stage, output=str(quality_rgb))

        final_alpha = None
        final_webm = None
        final_preview = None
        if has_alpha and alpha_inputs and len(alpha_inputs) == len(rgb_inputs):
            final_alpha = final_dir / "generative-dance-output-alpha.mov"
            stitch_transparent_videos(
                alpha_inputs,
                final_alpha,
                width=self.config.canvas.width,
                height=self.config.canvas.height,
                fps=sequence_fps,
                codec="prores_ks",
                crf=self.config.transparent_crf,
            )
            if quality_rgb != final_rgb:
                final_alpha = transform_alpha_video(
                    final_alpha,
                    final_dir / "postprocessed-alpha" / "generative-dance-output-alpha.mov",
                    width=quality_probe.width,
                    height=quality_probe.height,
                    fps=max(1, round(quality_probe.fps)),
                    frame_count=vace_frame_count(
                        quality_probe,
                        max(1, round(quality_probe.fps)),
                    ),
                    crf=self.config.transparent_crf,
                ).path
            final_webm = final_dir / "generative-dance-output.webm"
            encode_transparent_video(final_alpha, final_webm, codec=self.config.transparent_codec, crf=self.config.transparent_crf)
            final_preview = final_dir / "generative-dance-output-preview.mp4"
            make_transparent_preview(final_alpha, final_preview, width=quality_probe.width, height=quality_probe.height, fps=max(1, round(quality_probe.fps)), pixel_aspect_ratio=self.config.canvas.pixel_aspect_ratio)
        result_metadata = {
            "schemaVersion": 2,
            "runtime": "wan-animate",
            "continuityMode": (
                "multi-frame-carry"
                if self.config.wan_temporal_context_frames > 0
                else "disabled"
            ),
            "temporalWindow": self.config.wan_temporal_window,
            "temporalContextFrames": self.config.wan_temporal_context_frames,
            "identitySeed": identity_seed,
            "identitySeedScope": "job",
            "referenceStrength": reference_strength,
            "continuityPolicy": "within-source-segment-only-with-vace-boundaries" if vace_parts else "within-source-segment-only",
            "crossSegmentContinuation": False,
            "vace": {
                "enabled": bool(vace_parts),
                "modelName": vace_config.model_name,
                "modelSize": vace_config.model_size,
                "bridges": [result.to_dict() for result in vace_results],
                "loopEnabled": bool(vace_loop),
                "config": vace_config.to_public_dict(),
            },
            "videoStages": video_stages,
            "sequence": sequence,
            "segments": [
                {
                    "index": item["index"],
                    "segmentId": item["segmentId"],
                    "inputId": item["inputId"],
                    "timelineStartSeconds": item["timelineStartSeconds"],
                    "timelineEndSeconds": item["timelineEndSeconds"],
                    "adjacentToPrevious": item["adjacentToPrevious"],
                    "continuity": item["continuity"],
                    "driver": item["driver"].to_dict(),
                    "render": item["result"].to_dict(),
                }
                for item in rendered
            ],
            "finalOutput": str(quality_rgb),
            "finalMasterOutput": str(final_rgb),
            "finalTransparentOutput": str(final_webm) if final_webm else None,
            "finalOutputProbe": quality_probe.to_dict(),
        }
        metadata_path = final_dir / "job-result.json"
        self.store.write_json(metadata_path, result_metadata)
        return result_metadata, final_rgb

    def _run(self, job_id: str, payload: dict[str, Any]) -> None:
        reference_path: Path | None = None
        driver_path: Path | None = None
        try:
            self._state(job_id, status="running", attempt=1, stage="validate_inputs", progress=0.01)
            self._event(job_id, "stage_started", stage="validate_inputs")
            reference_url, reference_name = self._input(payload, {"reference", "reference_image", "reference-image", "avatar_reference", "avatar-reference", "image"}, "reference.png")
            driver_url, driver_name = self._input(payload, {"driver", "driver_video", "dance", "motion", "video"}, "driver.mp4")
            parameters = payload.get("parameters") or {}
            if not isinstance(parameters, dict):
                raise ValueError("generative dance job parameters must be an object")
            sequence = parameters.get("sequence")
            has_sequence = isinstance(sequence, dict) and isinstance(sequence.get("segments"), list) and bool(sequence.get("segments"))
            if not reference_url:
                raise ValueError("Wan Animate jobs require an avatar reference input")
            if not driver_url and not has_sequence:
                raise ValueError("Wan Animate jobs require a driver video input")
            identity_seed = _resolve_identity_seed(parameters)
            reference_strength = _resolve_reference_strength(self.config, parameters)
            self._event(
                job_id,
                "identity_policy_resolved",
                seed=identity_seed,
                seedScope="job",
                referenceStrength=reference_strength,
                referencePolicy="immutable-original-reference-per-window-and-segment",
            )
            reference_path = self._download(job_id, reference_url, reference_name, prefix="reference", suffixes={".png", ".jpg", ".jpeg", ".webp"})
            self._event(job_id, "reference_input_validated", reference=str(reference_path))
            description = str(parameters.get("description") or parameters.get("prompt") or "uploaded character reference")
            self._progress(job_id, "prepare_reference", 0.08, "Normalizing the reference image")
            reference = self.service.create_reference(description=description, uploaded_image=reference_path)
            if has_sequence:
                self._event(job_id, "sequence_detected", segmentCount=len(sequence["segments"]))
                sequence_result, final_output = self._run_sequence(
                    job_id=job_id,
                    payload=payload,
                    parameters=parameters,
                    reference_id=reference.id,
                    identity_seed=identity_seed,
                    reference_strength=reference_strength,
                )
                output_root = final_output.parent.parent
                self._progress(job_id, "finalize_artifacts", 0.95, "Validating the assembled sequence outputs")
                artifacts = self._artifacts(job_id, output_root)
                if not artifacts:
                    raise RuntimeError("Wan Animate sequence completed without output artifacts")
                metadata = {
                    "schemaVersion": 2,
                    "runtime": "wan-animate",
                    "modelRevision": self.config.wan_model_revision,
                    "request": payload,
                    "result": {
                        **sequence_result,
                        "reference": reference.to_dict(),
                    },
                    "artifacts": artifacts,
                    "completedAt": _now(),
                }
                metadata_path = final_output.parent / "job-result.json"
                self.store.write_json(metadata_path, metadata)
                artifacts = self._artifacts(job_id, output_root)
            else:
                driver_path = self._download(job_id, driver_url or "", driver_name, prefix="driver", suffixes={".mp4", ".webm", ".mov", ".mkv", ".avi"})
                driver_probe = probe_video(driver_path)
                self._event(job_id, "inputs_validated", reference=str(reference_path), driver=str(driver_path), driverProbe=driver_probe.to_dict())
                self._progress(job_id, "prepare_driver", 0.14, "Preparing the full-rate motion driver")
                source_fps = int(round(driver_probe.fps)) if driver_probe.fps > 0 else self.config.canvas.fps
                driver = self.service.create_driver(source=driver_path, label=str(parameters.get("driver_label") or "dance driver"), preserve_source_fps=True)
                self._event(job_id, "driver_ready", driver=driver.to_dict(), sourceFps=source_fps)
                self._progress(job_id, "wan_render", 0.18, "Running Wan Animate over the complete driver")
                placement = _placement_from_payload(
                    payload,
                    segment_id=f"segment-{uuid4().hex[:12]}",
                    driver_id=driver.id,
                )
                result = self.service.render(
                    reference_id=reference.id,
                    driver_id=driver.id,
                    prompt=str(parameters.get("prompt") or reference.prompt),
                    seed=identity_seed,
                    inference_steps=int(parameters.get("steps", parameters.get("num_inference_steps", self.config.wan_inference_steps))),
                    text_length=int(parameters.get("text_length", parameters.get("textLength", self.config.wan_text_length))),
                    reference_strength=reference_strength,
                    placement=placement,
                    transparent=bool(parameters.get("transparent", True)),
                )
                self._progress(job_id, "finalize_artifacts", 0.95, "Validating placed and transparent outputs")
                artifacts = self._artifacts(job_id, result.output_video.parent)
                if not artifacts:
                    raise RuntimeError("Wan Animate completed without output artifacts")
                metadata = {
                    "schemaVersion": 1,
                    "runtime": "wan-animate",
                    "modelRevision": self.config.wan_model_revision,
                    "request": payload,
                    "result": {
                        "renderedSegment": result.to_dict(),
                        "driver": driver.to_dict(),
                        "reference": reference.to_dict(),
                        "identitySeed": identity_seed,
                        "identitySeedScope": "job",
                        "referenceStrength": reference_strength,
                    },
                    "artifacts": artifacts,
                    "completedAt": _now(),
                }
                metadata_path = result.output_video.parent / "job-result.json"
                self.store.write_json(metadata_path, metadata)
                artifacts = self._artifacts(job_id, result.output_video.parent)
            events_path = self.job_dir(job_id) / "events.jsonl"
            if events_path.is_file():
                artifacts.append({"path": str(events_path), "name": events_path.name, "sizeBytes": events_path.stat().st_size, "mediaType": "application/jsonl", "variant": "generative-dance-diagnostics"})
            self._state(job_id, status="succeeded", stage="finalizing", progress=1.0, artifacts=artifacts, result=metadata)
            self._event(job_id, "job_succeeded", artifactCount=len(artifacts), artifactNames=[item["name"] for item in artifacts])
        except Exception as exc:
            current = self.get(job_id)
            details = getattr(exc, "details", {})
            if not isinstance(details, dict):
                details = {}
            failure = {
                "code": "generative_dance_worker_failed",
                "message": str(exc) or type(exc).__name__,
                "stage": current.get("stage") or "validate_inputs",
                "attempt": int(current.get("attempt") or 1),
                "errorType": type(exc).__name__,
                "details": details,
                "traceback": traceback.format_exc(),
            }
            failure_path = self.job_dir(job_id) / "failure-summary.json"
            artifacts = [{"path": str(failure_path), "name": failure_path.name, "sizeBytes": 0, "mediaType": "application/json", "variant": "generative-dance-failure"}]
            diagnostic_names: list[str] = []
            for key, value in (("stdoutLog", details.get("stdoutLog")), ("stderrLog", details.get("stderrLog"))):
                if not value:
                    continue
                source = Path(str(value))
                if not source.is_file():
                    continue
                target = self.job_dir(job_id) / f"wan-animate-2-lite.{key.removesuffix('Log')}.log"
                try:
                    shutil.copy2(source, target)
                except OSError:
                    continue
                diagnostic_names.append(target.name)
                artifacts.append({
                    "path": str(target),
                    "name": target.name,
                    "sizeBytes": target.stat().st_size,
                    "mediaType": "text/plain",
                    "variant": "generative-dance-failure-diagnostics",
                })
            if diagnostic_names:
                failure["details"] = {**details, "diagnosticArtifacts": diagnostic_names}
            self.store.write_json(failure_path, {"schemaVersion": 1, "runtime": "wan-animate", "jobId": job_id, "request": payload, "failure": failure, "eventLog": "events.jsonl", "createdAt": _now()})
            artifacts[0]["sizeBytes"] = failure_path.stat().st_size
            events = self.job_dir(job_id) / "events.jsonl"
            if events.is_file():
                artifacts.append({"path": str(events), "name": events.name, "sizeBytes": events.stat().st_size, "mediaType": "application/jsonl", "variant": "generative-dance-failure"})
            self._state(job_id, status="failed", stage=failure["stage"], progress=1.0, failure=failure, refundRequired=True, refundReason="generative_dance_generation_failed", artifacts=artifacts)
            self._event(job_id, "job_failed", failure=failure)
        finally:
            if reference_path:
                reference_path.unlink(missing_ok=True)
            if driver_path:
                driver_path.unlink(missing_ok=True)
            for downloaded_driver in self.job_dir(job_id).glob("driver-*"):
                downloaded_driver.unlink(missing_ok=True)
            with self._lock:
                self._futures.pop(job_id, None)
            self._event(job_id, "job_thread_finished")


def create_generative_dance_worker(config: GenerativeDanceConfig) -> GenerativeDanceWorker:
    return GenerativeDanceWorker(config)

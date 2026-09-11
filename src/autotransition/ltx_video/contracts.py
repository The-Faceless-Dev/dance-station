from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


def _safe_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


LtxJobStatus = Literal["queued", "running", "succeeded", "failed"]
LtxStage = Literal[
    "validate_request", "acquire_inputs", "normalize_inputs", "preflight_memory", "load_models",
    "encode_prompt", "stage_1_denoise", "spatial_upscale", "stage_2_refine", "decode_video",
    "decode_stage_1", "decode_audio", "mux_audio", "encode_output", "finalize", "upload_artifacts",
]


@dataclass(frozen=True)
class LtxConditioningImage:
    source_url: str = ""
    filename: str = "conditioning.png"
    path: Path | None = None
    frame_index: int = 0
    strength: float = 1.0
    mode: Literal["replace", "guide"] = "guide"
    crf: int | None = None

    def validate(self, config: Any) -> None:
        if self.path is None:
            parsed = urlsplit(self.source_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("conditioning image requires an HTTP(S) source_url or an allowed local path")
        else:
            if not config.allow_local_inputs:
                raise ValueError("local conditioning inputs are disabled for this worker")
            if not self.path.is_file():
                raise ValueError(f"conditioning image was not found: {self.path}")
        if self.frame_index < 0:
            raise ValueError("conditioning frame_index cannot be negative")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("conditioning strength must be finite and non-negative")
        if self.mode not in {"replace", "guide"}:
            raise ValueError("conditioning mode must be replace or guide")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceUrl": _safe_url(self.source_url),
            "filename": Path(self.filename).name,
            "path": str(self.path) if self.path else None,
            "frameIndex": self.frame_index,
            "strength": self.strength,
            "mode": self.mode,
            "crf": self.crf,
        }


@dataclass(frozen=True)
class LtxVideoRequest:
    prompt: str
    negative_prompt: str | None = None
    aspect_ratio: str = "16:9"
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    duration_seconds: float | None = None
    frame_rate: float = 24.0
    stage_1_steps: int | None = None
    stage_2_steps: int | None = None
    seed: int | None = None
    conditioning_images: tuple[LtxConditioningImage, ...] = ()
    audio_mode: Literal["off", "generated", "source"] = "off"
    source_audio_url: str = ""
    source_audio_path: Path | None = None
    loras: tuple[dict[str, Any], ...] = ()
    output_format: Literal["mp4", "webm"] = "mp4"
    retain_intermediates: bool | None = None
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def resolve_dimensions(self, config: Any) -> tuple[int, int]:
        if self.width is None and self.height is None:
            try:
                width, height = config.aspect_presets[self.aspect_ratio]
            except KeyError as exc:
                raise ValueError(f"unsupported aspect_ratio: {self.aspect_ratio}") from exc
        elif self.width is None or self.height is None:
            raise ValueError("width and height must be provided together")
        else:
            width, height = self.width, self.height
        if not config.allow_custom_dimensions and self.aspect_ratio not in config.aspect_presets:
            raise ValueError("custom dimensions are disabled")
        if width < 32 or height < 32 or width > config.max_width or height > config.max_height:
            raise ValueError(f"dimensions must be within 32..{config.max_width}x{config.max_height}")
        if width % 64 or height % 64:
            raise ValueError("width and height must be divisible by 64 for the LTX 2.5 two-stage pipeline")
        return width, height

    def resolved_aspect_ratio(self, config: Any) -> str:
        width, height = self.resolve_dimensions(config)
        if self.width is None and self.height is None:
            return self.aspect_ratio
        for name, dimensions in config.aspect_presets.items():
            if dimensions == (width, height):
                return name
        return "custom"

    def resolve_frames(self, config: Any) -> int:
        if self.num_frames is not None and self.duration_seconds is not None:
            raise ValueError("set either num_frames or duration_seconds, not both")
        if self.num_frames is None and self.duration_seconds is None:
            raise ValueError("num_frames or duration_seconds is required")
        if self.num_frames is not None:
            frames = self.num_frames
        else:
            if self.duration_seconds is None or not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
                raise ValueError("duration_seconds must be positive")
            if self.frame_rate <= 0 or not math.isfinite(self.frame_rate):
                raise ValueError("frame_rate must be positive and finite")
            frames = round(self.duration_seconds * self.frame_rate)
        if frames < 9:
            raise ValueError("LTX requires at least 9 frames")
        # LTX causal temporal compression works on 8*K+1 frame counts.
        frames = 8 * round((frames - 1) / 8) + 1
        if self.frame_rate <= 0 or not math.isfinite(self.frame_rate):
            raise ValueError("frame_rate must be positive and finite")
        if config.max_frames and frames > config.max_frames:
            raise ValueError(f"resolved frame count {frames} exceeds configured max_frames {config.max_frames}")
        if config.max_duration_seconds and frames / self.frame_rate > config.max_duration_seconds:
            raise ValueError("resolved duration exceeds configured max_duration_seconds")
        return frames

    def validate(self, config: Any) -> None:
        config.validate()
        if not self.prompt.strip() or len(self.prompt) > 16000:
            raise ValueError("prompt must be between 1 and 16000 characters")
        if self.negative_prompt is not None and len(self.negative_prompt) > 16000:
            raise ValueError("negative_prompt must be 16000 characters or fewer")
        self.resolve_dimensions(config)
        self.resolve_frames(config)
        if self.stage_1_steps is not None and self.stage_1_steps != 8:
            raise ValueError("the LTX 2.5 distilled pipeline uses its fixed 8-step stage 1 schedule")
        if self.stage_2_steps is not None and self.stage_2_steps != 4:
            raise ValueError("the LTX 2.5 distilled pipeline uses its fixed 4-step stage 2 schedule")
        if self.audio_mode not in {"off", "generated", "source"}:
            raise ValueError("audio_mode must be off, generated, or source")
        if self.audio_mode == "source" and self.source_audio_path is None and not self.source_audio_url:
            raise ValueError("source audio mode requires source_audio_url or an allowed local path")
        if self.audio_mode == "generated" and config.audio_vae_path is None:
            raise ValueError("generated audio requires LTX_VIDEO_AUDIO_VAE_PATH")
        if len(self.conditioning_images) > config.max_conditioning_images:
            raise ValueError(f"at most {config.max_conditioning_images} conditioning images are supported")
        for item in self.conditioning_images:
            item.validate(config)
        if self.output_format not in {"mp4", "webm"}:
            raise ValueError("output_format must be mp4 or webm")
        if self.seed is not None and not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.external_job_id and any(char in self.external_job_id for char in "\\/"):
            raise ValueError("external_job_id cannot contain path separators")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conditioning_images"] = [item.to_dict() for item in self.conditioning_images]
        for key in ("source_audio_path",):
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        payload["source_audio_url"] = _safe_url(self.source_audio_url)
        return payload


@dataclass(frozen=True)
class LtxArtifact:
    name: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str
    role: str = "output"


@dataclass(frozen=True)
class LtxFailure:
    code: str
    message: str
    stage: str
    retryable: bool = False
    attempt: int = 1
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LtxVideoJob:
    id: str
    status: LtxJobStatus
    request: dict[str, Any]
    stage: LtxStage | str | None = None
    progress: float = 0.0
    attempt: int = 0
    message: str | None = None
    artifacts: list[LtxArtifact] = field(default_factory=list)
    failure: LtxFailure | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(item) for item in self.artifacts]
        payload["failure"] = self.failure.to_dict() if self.failure else None
        payload["failureCode"] = self.failure.code if self.failure else None
        payload["refundRequired"] = bool(self.failure)
        payload["refundReason"] = "ltx_video_generation_failed" if self.failure else None
        return payload

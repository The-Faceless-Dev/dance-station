"""Environment-backed configuration for the VACE stitch worker."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser() if value else None


def _optional_number(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip().lower() in {"", "none", "null", "off", "disabled"}:
        return None
    return float(value)


@dataclass(frozen=True)
class VaceStitchConfig:
    """Model, timeline, artifact, and callback settings for one GPU worker."""

    artifact_root: Path = Path("data/vace-stitch")
    enabled: bool = False
    runtime_backend: str = "native"
    runtime_command: str | None = None
    runtime_cwd: Path | None = None
    python_executable: str | None = None
    lightx2v_source_root: Path | None = None
    lightx2v_config: Path | None = None
    lightx2v_lora: Path | None = None
    lightx2v_lora_strength: float = 1.0
    lightx2v_steps: int = 4
    lightx2v_attention_backend: str = "flash_attn2"
    source_root: Path | None = None
    checkpoint_dir: Path | None = None
    checkpoint_file: Path | None = None
    model_name: str = "vace-1.3B"
    model_size: str = "480p"
    model_fps: int = 16
    output_fps: int = 24
    output_width: int = 832
    output_height: int = 480
    default_prompt: str = "the character continues dancing"
    default_loop_prompt: str = "the character continues dancing and returns smoothly to the starting motion"
    default_gap_seconds: float = 2.0
    loop_enabled: bool = True
    min_gap_seconds: float = 0.25
    max_gap_seconds: float = 20.0
    default_context_before_seconds: float = 1.0
    default_context_after_seconds: float = 1.0
    max_window_frames: int = 81
    sample_steps: int = 50
    sample_shift: float = 16.0
    guide_scale: float = 5.0
    context_scale: float = 1.0
    sample_solver: str = "unipc"
    offload_model: bool = True
    t5_cpu: bool = True
    attention_backend: str = "auto"
    tf32: bool = False
    temporary_background: str = "0x7f7f7f"
    transparent_default: bool = True
    matte_backend: str = "command"
    matte_command: str | None = None
    matte_cwd: Path | None = None
    matte_model: str = "ZhengPeng7/BiRefNet-matting"
    matte_checkpoint: Path | None = None
    matte_python: str | None = None
    matte_device: str = "cuda"
    matte_dtype: str = "float16"
    matte_batch_size: int = 2
    matte_input_size: int = 1024
    transparent_codec: str = "libvpx-vp9"
    transparent_crf: int = 24
    enhancement_enabled: bool = False
    enhancement_backend: str = "command"
    enhancement_command: str | None = None
    enhancement_cwd: Path | None = None
    enhancement_scale: float = 1.0
    enhancement_target_fps: int = 0
    enhancement_tile_size: int = 0
    enhancement_fp16: bool = True
    motion_interpolation_enabled: bool = False
    motion_interpolation_backend: str = "command"
    motion_interpolation_command: str | None = None
    motion_interpolation_cwd: Path | None = None
    motion_interpolation_target_fps: int = 0
    job_timeout_seconds: float = 7200.0
    # VACE inference may legitimately run for hours on a long sequence. None
    # means the model process runs until completion or an actual process error.
    runtime_timeout_seconds: float | None = None
    max_upload_bytes: int = 2_147_483_648
    keep_intermediate: bool = True

    @classmethod
    def from_env(cls) -> "VaceStitchConfig":
        def integer(name: str, default: int) -> int:
            value = os.getenv(name)
            return default if value is None else int(value)

        def number(name: str, default: float) -> float:
            value = os.getenv(name)
            return default if value is None else float(value)

        return cls(
            artifact_root=Path(os.getenv("VACE_STITCH_ARTIFACT_ROOT", "data/vace-stitch")),
            enabled=_bool("VACE_STITCH_ENABLED", False),
            runtime_backend=os.getenv("VACE_STITCH_BACKEND", "native"),
            runtime_command=os.getenv("VACE_STITCH_COMMAND"),
            runtime_cwd=_path("VACE_STITCH_CWD"),
            python_executable=os.getenv("VACE_STITCH_PYTHON"),
            lightx2v_source_root=_path("VACE_STITCH_LIGHTX2V_SOURCE_ROOT"),
            lightx2v_config=_path("VACE_STITCH_LIGHTX2V_CONFIG"),
            lightx2v_lora=_path("VACE_STITCH_LIGHTX2V_LORA"),
            lightx2v_lora_strength=number("VACE_STITCH_LIGHTX2V_LORA_STRENGTH", 1.0),
            lightx2v_steps=integer("VACE_STITCH_LIGHTX2V_STEPS", 4),
            lightx2v_attention_backend=os.getenv("VACE_STITCH_LIGHTX2V_ATTENTION", "flash_attn2").lower(),
            source_root=_path("VACE_STITCH_SOURCE_ROOT"),
            checkpoint_dir=_path("VACE_STITCH_CHECKPOINT_DIR"),
            checkpoint_file=_path("VACE_STITCH_CHECKPOINT_FILE"),
            model_name=os.getenv("VACE_STITCH_MODEL_NAME", "vace-1.3B"),
            model_size=os.getenv("VACE_STITCH_MODEL_SIZE", "480p"),
            model_fps=integer("VACE_STITCH_MODEL_FPS", 16),
            output_fps=integer("VACE_STITCH_OUTPUT_FPS", 24),
            output_width=integer("VACE_STITCH_OUTPUT_WIDTH", 832),
            output_height=integer("VACE_STITCH_OUTPUT_HEIGHT", 480),
            default_prompt=os.getenv("VACE_STITCH_DEFAULT_PROMPT", "the character continues dancing"),
            default_loop_prompt=os.getenv(
                "VACE_STITCH_DEFAULT_LOOP_PROMPT",
                "the character continues dancing and returns smoothly to the starting motion",
            ),
            default_gap_seconds=number("VACE_STITCH_DEFAULT_GAP_SECONDS", 2.0),
            loop_enabled=_bool("VACE_STITCH_LOOP_ENABLED", True),
            min_gap_seconds=number("VACE_STITCH_MIN_GAP_SECONDS", 0.25),
            max_gap_seconds=number("VACE_STITCH_MAX_GAP_SECONDS", 20.0),
            default_context_before_seconds=number("VACE_STITCH_CONTEXT_BEFORE_SECONDS", 1.0),
            default_context_after_seconds=number("VACE_STITCH_CONTEXT_AFTER_SECONDS", 1.0),
            max_window_frames=integer("VACE_STITCH_MAX_WINDOW_FRAMES", 81),
            sample_steps=integer("VACE_STITCH_SAMPLE_STEPS", 50),
            sample_shift=number("VACE_STITCH_SAMPLE_SHIFT", 16.0),
            guide_scale=number("VACE_STITCH_GUIDE_SCALE", 5.0),
            context_scale=number("VACE_STITCH_CONTEXT_SCALE", 1.0),
            sample_solver=os.getenv("VACE_STITCH_SAMPLE_SOLVER", "unipc"),
            offload_model=_bool("VACE_STITCH_OFFLOAD_MODEL", True),
            t5_cpu=_bool("VACE_STITCH_T5_CPU", True),
            attention_backend=os.getenv("VACE_STITCH_ATTENTION_BACKEND", "auto").lower(),
            tf32=_bool("VACE_STITCH_TF32", False),
            temporary_background=os.getenv("VACE_STITCH_TEMPORARY_BACKGROUND", "0x7f7f7f"),
            transparent_default=_bool("VACE_STITCH_TRANSPARENT_DEFAULT", True),
            matte_backend=os.getenv("VACE_STITCH_MATTE_BACKEND", "command"),
            matte_command=os.getenv("VACE_STITCH_MATTE_COMMAND"),
            matte_cwd=_path("VACE_STITCH_MATTE_CWD"),
            matte_model=os.getenv("VACE_STITCH_MATTE_MODEL", "ZhengPeng7/BiRefNet-matting"),
            matte_checkpoint=_path("VACE_STITCH_MATTE_CHECKPOINT"),
            matte_python=os.getenv("VACE_STITCH_MATTE_PYTHON"),
            matte_device=os.getenv("VACE_STITCH_MATTE_DEVICE", "cuda"),
            matte_dtype=os.getenv("VACE_STITCH_MATTE_DTYPE", "float16"),
            matte_batch_size=integer("VACE_STITCH_MATTE_BATCH_SIZE", 2),
            matte_input_size=integer("VACE_STITCH_MATTE_INPUT_SIZE", 1024),
            transparent_codec=os.getenv("VACE_STITCH_TRANSPARENT_CODEC", "libvpx-vp9"),
            transparent_crf=integer("VACE_STITCH_TRANSPARENT_CRF", 24),
            enhancement_enabled=_bool("VACE_STITCH_ENHANCEMENT_ENABLED", False),
            enhancement_backend=os.getenv("VACE_STITCH_ENHANCEMENT_BACKEND", "command"),
            enhancement_command=os.getenv("VACE_STITCH_ENHANCEMENT_COMMAND"),
            enhancement_cwd=_path("VACE_STITCH_ENHANCEMENT_CWD"),
            enhancement_scale=number("VACE_STITCH_ENHANCEMENT_SCALE", 1.0),
            enhancement_target_fps=integer("VACE_STITCH_ENHANCEMENT_TARGET_FPS", 0),
            enhancement_tile_size=integer("VACE_STITCH_ENHANCEMENT_TILE_SIZE", 0),
            enhancement_fp16=_bool("VACE_STITCH_ENHANCEMENT_FP16", True),
            motion_interpolation_enabled=_bool("VACE_STITCH_MOTION_INTERPOLATION_ENABLED", False),
            motion_interpolation_backend=os.getenv("VACE_STITCH_MOTION_INTERPOLATION_BACKEND", "command"),
            motion_interpolation_command=os.getenv("VACE_STITCH_MOTION_INTERPOLATION_COMMAND"),
            motion_interpolation_cwd=_path("VACE_STITCH_MOTION_INTERPOLATION_CWD"),
            motion_interpolation_target_fps=integer("VACE_STITCH_MOTION_INTERPOLATION_TARGET_FPS", 0),
            job_timeout_seconds=number(
                "VACE_STITCH_STAGE_TIMEOUT_SECONDS",
                number("VACE_STITCH_JOB_TIMEOUT_SECONDS", 7200.0),
            ),
            runtime_timeout_seconds=_optional_number("VACE_STITCH_RUNTIME_TIMEOUT_SECONDS"),
            max_upload_bytes=integer("VACE_STITCH_MAX_UPLOAD_BYTES", 2_147_483_648),
            keep_intermediate=_bool("VACE_STITCH_KEEP_INTERMEDIATE", True),
        )

    def validate(self) -> None:
        if self.runtime_backend not in {"command", "native", "lightx2v"}:
            raise ValueError("VACE stitch backend must be command, native, or lightx2v")
        if self.model_size not in {"480p", "720p"}:
            raise ValueError("VACE stitch model size must be 480p or 720p")
        if self.model_fps < 1 or self.model_fps > 60:
            raise ValueError("VACE stitch model FPS must be between 1 and 60")
        if self.output_fps < 1 or self.output_fps > 120:
            raise ValueError("VACE stitch output FPS must be between 1 and 120")
        if self.output_width < 64 or self.output_height < 64:
            raise ValueError("VACE stitch output dimensions must be at least 64 pixels")
        if self.default_gap_seconds <= 0 or self.min_gap_seconds <= 0:
            raise ValueError("VACE stitch gap durations must be positive")
        if self.min_gap_seconds > self.default_gap_seconds or self.default_gap_seconds > self.max_gap_seconds:
            raise ValueError("VACE stitch gap duration defaults must be within min/max bounds")
        if self.default_context_before_seconds <= 0 or self.default_context_after_seconds <= 0:
            raise ValueError("VACE stitch context durations must be positive")
        if self.max_window_frames < 5 or (self.max_window_frames - 1) % 4:
            raise ValueError("VACE stitch max window frames must be 4n+1 and at least 5")
        if self.sample_steps < 1 or self.sample_shift <= 0 or self.guide_scale < 0 or self.context_scale < 0:
            raise ValueError("VACE stitch sampling settings are invalid")
        if self.lightx2v_steps < 1 or self.lightx2v_lora_strength < 0:
            raise ValueError("LightX2V VACE sampling settings are invalid")
        if self.lightx2v_attention_backend not in {"flash_attn2", "flash_attn3"}:
            raise ValueError("LightX2V VACE attention must be flash_attn2 or flash_attn3")
        if self.sample_solver not in {"unipc", "dpm++"}:
            raise ValueError("VACE stitch sample solver must be unipc or dpm++")
        if self.attention_backend not in {"auto", "flash_attention_2"}:
            raise ValueError("VACE stitch attention backend must be auto or flash_attention_2")
        if self.matte_backend not in {"command", "native"}:
            raise ValueError("VACE stitch matte backend must be command or native")
        if self.matte_batch_size < 1 or self.matte_input_size < 64:
            raise ValueError("VACE stitch matte settings are invalid")
        if self.transparent_crf < 0 or self.transparent_crf > 63:
            raise ValueError("VACE stitch transparent CRF must be between 0 and 63")
        if not self.enhancement_backend.strip() or not self.motion_interpolation_backend.strip():
            raise ValueError("VACE video stage backends must not be empty")
        if self.enhancement_scale < 1 or self.enhancement_scale > 8:
            raise ValueError("VACE enhancement scale must be between 1 and 8")
        if self.enhancement_target_fps < 0 or self.enhancement_target_fps > 120:
            raise ValueError("VACE enhancement target FPS must be between 0 and 120")
        if self.enhancement_tile_size < 0:
            raise ValueError("VACE enhancement tile size must be non-negative")
        if self.motion_interpolation_target_fps < 0 or self.motion_interpolation_target_fps > 120:
            raise ValueError("VACE motion interpolation target FPS must be between 0 and 120")
        if self.job_timeout_seconds <= 0 or self.max_upload_bytes < 1:
            raise ValueError("VACE stitch stage timeout and upload limit must be positive")
        if self.runtime_timeout_seconds is not None and self.runtime_timeout_seconds <= 0:
            raise ValueError("VACE runtime timeout must be positive when configured")
        if self.checkpoint_file is not None and self.checkpoint_file.suffix.lower() not in {".safetensors", ".ckpt", ".bin"}:
            raise ValueError("VACE checkpoint file must be a safetensors, ckpt, or bin file")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "artifactRoot": str(self.artifact_root),
            "enabled": self.enabled,
            "runtimeBackend": self.runtime_backend,
            "runtimeCommandConfigured": bool(self.runtime_command),
            "runtimeCwd": str(self.runtime_cwd) if self.runtime_cwd else None,
            "lightx2vSourceRoot": str(self.lightx2v_source_root) if self.lightx2v_source_root else None,
            "lightx2vConfig": str(self.lightx2v_config) if self.lightx2v_config else None,
            "lightx2vLora": str(self.lightx2v_lora) if self.lightx2v_lora else None,
            "lightx2vLoraStrength": self.lightx2v_lora_strength,
            "lightx2vSteps": self.lightx2v_steps,
            "lightx2vAttentionBackend": self.lightx2v_attention_backend,
            "sourceRoot": str(self.source_root) if self.source_root else None,
            "checkpointDirConfigured": bool(self.checkpoint_dir),
            "checkpointFile": str(self.checkpoint_file) if self.checkpoint_file else None,
            "checkpointFileConfigured": bool(self.checkpoint_file),
            "modelName": self.model_name,
            "modelSize": self.model_size,
            "modelFps": self.model_fps,
            "outputFps": self.output_fps,
            "outputWidth": self.output_width,
            "outputHeight": self.output_height,
            "defaultPrompt": self.default_prompt,
            "defaultLoopPrompt": self.default_loop_prompt,
            "defaultGapSeconds": self.default_gap_seconds,
            "loopEnabled": self.loop_enabled,
            "minGapSeconds": self.min_gap_seconds,
            "maxGapSeconds": self.max_gap_seconds,
            "defaultContextBeforeSeconds": self.default_context_before_seconds,
            "defaultContextAfterSeconds": self.default_context_after_seconds,
            "maxWindowFrames": self.max_window_frames,
            "sampleSteps": self.sample_steps,
            "sampleShift": self.sample_shift,
            "guideScale": self.guide_scale,
            "contextScale": self.context_scale,
            "sampleSolver": self.sample_solver,
            "offloadModel": self.offload_model,
            "t5Cpu": self.t5_cpu,
            "attentionBackend": self.attention_backend,
            "tf32": self.tf32,
            "temporaryBackground": self.temporary_background,
            "transparentDefault": self.transparent_default,
            "matteBackend": self.matte_backend,
            "matteCommandConfigured": bool(self.matte_command),
            "matteModel": self.matte_model,
            "matteCheckpointConfigured": bool(self.matte_checkpoint),
            "matteDevice": self.matte_device,
            "matteDtype": self.matte_dtype,
            "matteBatchSize": self.matte_batch_size,
            "matteInputSize": self.matte_input_size,
            "transparentCodec": self.transparent_codec,
            "transparentCrf": self.transparent_crf,
            "enhancementEnabled": self.enhancement_enabled,
            "enhancementBackend": self.enhancement_backend,
            "enhancementCommandConfigured": bool(self.enhancement_command),
            "enhancementScale": self.enhancement_scale,
            "enhancementTargetFps": self.enhancement_target_fps,
            "enhancementTileSize": self.enhancement_tile_size,
            "enhancementFp16": self.enhancement_fp16,
            "motionInterpolationEnabled": self.motion_interpolation_enabled,
            "motionInterpolationBackend": self.motion_interpolation_backend,
            "motionInterpolationCommandConfigured": bool(self.motion_interpolation_command),
            "motionInterpolationTargetFps": self.motion_interpolation_target_fps,
            "stageTimeoutSeconds": self.job_timeout_seconds,
            "runtimeTimeoutSeconds": self.runtime_timeout_seconds,
            "maxUploadBytes": self.max_upload_bytes,
            "keepIntermediate": self.keep_intermediate,
        }

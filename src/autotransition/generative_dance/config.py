"""Configuration for the local generative dance proof of concept."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autotransition.generative_dance.contracts import CanvasContract


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _optional_number(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip().lower() in {"", "none", "null", "off", "disabled"}:
        return None
    return float(value)


@dataclass(frozen=True)
class GenerativeDanceConfig:
    """Paths, resource limits, and external model command boundaries."""

    artifact_root: Path = Path("data/generative-dance")
    image_command: str | None = None
    image_cwd: Path | None = None
    matte_command: str | None = None
    matte_cwd: Path | None = None
    matte_backend: str = "command"
    matte_model: str = "ZhengPeng7/BiRefNet-matting"
    matte_checkpoint: Path | None = None
    matte_device: str = "cuda"
    matte_compute_dtype: str = "float16"
    matte_python: str | None = None
    matte_batch_size: int = 4
    matte_input_size: int = 1024
    matte_alpha_threshold: float = 0.0
    matte_edge_feather: float = 0.0
    matte_despill_strength: float = 1.0
    matte_background_key_distance: float = 80.0
    matte_background_key_strength: float = 1.0
    anchor_sync_enabled: bool = True
    # Retained for compatibility with older local configurations. The render
    # path no longer uses silhouette-median stabilization.
    stabilize_position: bool = True
    stabilize_position_threshold_px: float = 12.0
    stabilize_position_strength: float = 0.85
    stabilize_position_window: int = 5
    transparent_codec: str = "libvpx-vp9"
    transparent_crf: int = 30
    retain_matte_artifacts: bool = True
    wan_command: str | None = None
    wan_cwd: Path | None = None
    wan_backend: str = "command"
    wan_model_revision: str = "Wan-Animate-2-Lite"
    wan_checkpoint_format: str = "gguf"
    wan_config_file: Path | None = None
    wan_transformer_checkpoint: Path | None = None
    wan_official_source: Path | None = None
    wan_t5_checkpoint: Path | None = None
    wan_t5_tokenizer: Path | None = None
    wan_clip_checkpoint: Path | None = None
    wan_clip_tokenizer: Path | None = None
    wan_vae_checkpoint: Path | None = None
    wan_lightx2v_enabled: bool = False
    wan_lightx2v_checkpoint: Path | None = None
    wan_lightx2v_strength: float = 1.0
    wan_device: str = "cuda"
    wan_compute_dtype: str = "bfloat16"
    wan_python: str | None = None
    wan_inference_steps: int = 10
    wan_min_inference_steps: int = 10
    wan_guidance_scale: float = 1.0
    wan_text_length: int = 256
    wan_temporal_window: int = 81
    # Wan Animate's documented temporal-guidance choices are 1 or 5. The
    # context count also controls source overlap and decoded-frame trimming.
    wan_temporal_context_frames: int = 5
    # Scales the reference-image value tokens during the generation pass.
    # 1.0 preserves the original behavior; production can opt into a stronger
    # identity anchor without changing the request contract.
    wan_reference_strength: float = 1.0
    # Wan renders can legitimately run for hours on long sequences. None means
    # completion is governed by the process and worker state, not wall time.
    wan_render_timeout_seconds: float | None = None
    identity_audit_enabled: bool = False
    identity_audit_command: str | None = None
    identity_audit_cwd: Path | None = None
    identity_audit_threshold: float = 0.8
    identity_audit_max_retries: int = 1
    job_timeout_seconds: float = 1800.0
    max_upload_bytes: int = 1_000_000_000
    keep_failed_artifacts: bool = True
    canvas: CanvasContract = CanvasContract()

    @classmethod
    def from_env(cls) -> "GenerativeDanceConfig":
        def integer(name: str, default: int) -> int:
            value = os.getenv(name)
            return default if value is None else int(value)

        def number(name: str, default: float) -> float:
            value = os.getenv(name)
            return default if value is None else float(value)

        canvas = CanvasContract(
            width=integer("GENERATIVE_DANCE_CANVAS_WIDTH", 480),
            height=integer("GENERATIVE_DANCE_CANVAS_HEIGHT", 832),
            fps=integer("GENERATIVE_DANCE_FPS", 24),
            anchor_x=number("GENERATIVE_DANCE_ANCHOR_X", 0.5),
            anchor_y=number("GENERATIVE_DANCE_ANCHOR_Y", 0.58),
            subject_margin=number("GENERATIVE_DANCE_SUBJECT_MARGIN", 0.12),
        )
        return cls(
            artifact_root=Path(os.getenv("GENERATIVE_DANCE_ARTIFACT_ROOT", "data/generative-dance")),
            image_command=os.getenv("GENERATIVE_DANCE_IMAGE_COMMAND") or os.getenv("AVATAR_IMAGE_COMMAND"),
            image_cwd=Path(os.environ["GENERATIVE_DANCE_IMAGE_CWD"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_IMAGE_CWD")
            else None,
            matte_command=os.getenv("GENERATIVE_DANCE_MATTE_COMMAND"),
            matte_cwd=Path(os.environ["GENERATIVE_DANCE_MATTE_CWD"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_MATTE_CWD")
            else None,
            matte_backend=os.getenv("GENERATIVE_DANCE_MATTE_BACKEND", "command"),
            matte_model=os.getenv("GENERATIVE_DANCE_MATTE_MODEL", "ZhengPeng7/BiRefNet-matting"),
            matte_checkpoint=Path(os.environ["GENERATIVE_DANCE_MATTE_CHECKPOINT"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_MATTE_CHECKPOINT")
            else None,
            matte_device=os.getenv("GENERATIVE_DANCE_MATTE_DEVICE", "cuda"),
            matte_compute_dtype=os.getenv("GENERATIVE_DANCE_MATTE_DTYPE", "float16"),
            matte_python=os.getenv("GENERATIVE_DANCE_MATTE_PYTHON"),
            matte_batch_size=integer("GENERATIVE_DANCE_MATTE_BATCH_SIZE", 4),
            matte_input_size=integer("GENERATIVE_DANCE_MATTE_INPUT_SIZE", 1024),
            matte_alpha_threshold=number("GENERATIVE_DANCE_MATTE_ALPHA_THRESHOLD", 0.0),
            matte_edge_feather=number("GENERATIVE_DANCE_MATTE_EDGE_FEATHER", 0.0),
            matte_despill_strength=number("GENERATIVE_DANCE_MATTE_DESPILL_STRENGTH", 1.0),
            matte_background_key_distance=number("GENERATIVE_DANCE_MATTE_BACKGROUND_KEY_DISTANCE", 80.0),
            matte_background_key_strength=number("GENERATIVE_DANCE_MATTE_BACKGROUND_KEY_STRENGTH", 1.0),
            anchor_sync_enabled=_bool("GENERATIVE_DANCE_ANCHOR_SYNC", True),
            stabilize_position=_bool("GENERATIVE_DANCE_STABILIZE_POSITION", True),
            stabilize_position_threshold_px=number("GENERATIVE_DANCE_STABILIZE_POSITION_THRESHOLD_PX", 12.0),
            stabilize_position_strength=number("GENERATIVE_DANCE_STABILIZE_POSITION_STRENGTH", 0.85),
            stabilize_position_window=integer("GENERATIVE_DANCE_STABILIZE_POSITION_WINDOW", 5),
            transparent_codec=os.getenv("GENERATIVE_DANCE_TRANSPARENT_CODEC", "libvpx-vp9"),
            transparent_crf=integer("GENERATIVE_DANCE_TRANSPARENT_CRF", 30),
            retain_matte_artifacts=_bool("GENERATIVE_DANCE_RETAIN_MATTE_ARTIFACTS", True),
            wan_command=os.getenv("GENERATIVE_DANCE_WAN_COMMAND"),
            wan_cwd=Path(os.environ["GENERATIVE_DANCE_WAN_CWD"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_CWD")
            else None,
            wan_backend=os.getenv("GENERATIVE_DANCE_WAN_BACKEND", "command"),
            wan_model_revision=os.getenv("GENERATIVE_DANCE_WAN_MODEL", "Wan-Animate-2-Lite"),
            wan_checkpoint_format=os.getenv("GENERATIVE_DANCE_WAN_CHECKPOINT_FORMAT", "gguf").strip().lower(),
            wan_config_file=Path(os.environ["GENERATIVE_DANCE_WAN_CONFIG"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_CONFIG")
            else None,
            wan_transformer_checkpoint=Path(os.environ["GENERATIVE_DANCE_WAN_TRANSFORMER"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_TRANSFORMER")
            else None,
            wan_official_source=Path(os.environ["GENERATIVE_DANCE_WAN_SOURCE"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_SOURCE")
            else None,
            wan_t5_checkpoint=Path(os.environ["GENERATIVE_DANCE_WAN_T5_CHECKPOINT"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_T5_CHECKPOINT")
            else None,
            wan_t5_tokenizer=Path(os.environ["GENERATIVE_DANCE_WAN_T5_TOKENIZER"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_T5_TOKENIZER")
            else None,
            wan_clip_checkpoint=Path(os.environ["GENERATIVE_DANCE_WAN_CLIP_CHECKPOINT"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_CLIP_CHECKPOINT")
            else None,
            wan_clip_tokenizer=Path(os.environ["GENERATIVE_DANCE_WAN_CLIP_TOKENIZER"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_CLIP_TOKENIZER")
            else None,
            wan_vae_checkpoint=Path(os.environ["GENERATIVE_DANCE_WAN_VAE_CHECKPOINT"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_VAE_CHECKPOINT")
            else None,
            wan_lightx2v_enabled=_bool("GENERATIVE_DANCE_WAN_LIGHTX2V_ENABLED", False),
            wan_lightx2v_checkpoint=Path(os.environ["GENERATIVE_DANCE_WAN_LIGHTX2V_CHECKPOINT"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_WAN_LIGHTX2V_CHECKPOINT")
            else None,
            wan_lightx2v_strength=number("GENERATIVE_DANCE_WAN_LIGHTX2V_STRENGTH", 1.0),
            wan_device=os.getenv("GENERATIVE_DANCE_WAN_DEVICE", "cuda"),
            wan_compute_dtype=os.getenv("GENERATIVE_DANCE_WAN_DTYPE", "bfloat16"),
            wan_python=os.getenv("GENERATIVE_DANCE_WAN_PYTHON"),
            wan_inference_steps=integer("GENERATIVE_DANCE_WAN_STEPS", 10),
            wan_min_inference_steps=integer("GENERATIVE_DANCE_WAN_MIN_STEPS", 10),
            wan_guidance_scale=number("GENERATIVE_DANCE_WAN_GUIDANCE_SCALE", 1.0),
            wan_text_length=integer("GENERATIVE_DANCE_WAN_TEXT_LENGTH", 256),
            wan_temporal_window=integer("GENERATIVE_DANCE_WAN_TEMPORAL_WINDOW", 81),
            wan_temporal_context_frames=integer("GENERATIVE_DANCE_WAN_TEMPORAL_CONTEXT_FRAMES", 5),
            wan_reference_strength=number("GENERATIVE_DANCE_WAN_REFERENCE_STRENGTH", 1.0),
            identity_audit_enabled=_bool("GENERATIVE_DANCE_IDENTITY_AUDIT_ENABLED", False),
            identity_audit_command=os.getenv("GENERATIVE_DANCE_IDENTITY_AUDIT_COMMAND"),
            identity_audit_cwd=Path(os.environ["GENERATIVE_DANCE_IDENTITY_AUDIT_CWD"]).expanduser()
            if os.getenv("GENERATIVE_DANCE_IDENTITY_AUDIT_CWD")
            else None,
            identity_audit_threshold=number("GENERATIVE_DANCE_IDENTITY_AUDIT_THRESHOLD", 0.8),
            identity_audit_max_retries=integer("GENERATIVE_DANCE_IDENTITY_AUDIT_MAX_RETRIES", 1),
            job_timeout_seconds=number(
                "GENERATIVE_DANCE_STAGE_TIMEOUT_SECONDS",
                number("GENERATIVE_DANCE_JOB_TIMEOUT_SECONDS", 1800.0),
            ),
            wan_render_timeout_seconds=_optional_number("GENERATIVE_DANCE_WAN_RENDER_TIMEOUT_SECONDS"),
            max_upload_bytes=integer("GENERATIVE_DANCE_MAX_UPLOAD_BYTES", 1_000_000_000),
            keep_failed_artifacts=_bool("GENERATIVE_DANCE_KEEP_FAILED_ARTIFACTS", True),
            canvas=canvas,
        )

    def validate(self) -> None:
        self.canvas.validate()
        if self.job_timeout_seconds <= 0:
            raise ValueError("generative dance stage timeout must be positive")
        if self.wan_render_timeout_seconds is not None and self.wan_render_timeout_seconds <= 0:
            raise ValueError("Wan render timeout must be positive when configured")
        if self.max_upload_bytes < 1:
            raise ValueError("generative dance max upload size must be positive")
        if self.wan_backend not in {"command", "native"}:
            raise ValueError("generative dance Wan backend must be command or native")
        if self.wan_checkpoint_format not in {"gguf", "int8_convrot"}:
            raise ValueError(
                "generative dance Wan checkpoint format must be gguf or int8_convrot"
            )
        if self.wan_lightx2v_enabled and self.wan_lightx2v_checkpoint is None:
            raise ValueError(
                "LightX2V is enabled but GENERATIVE_DANCE_WAN_LIGHTX2V_CHECKPOINT is missing"
            )
        if not 0 < self.wan_lightx2v_strength <= 2:
            raise ValueError("LightX2V strength must be greater than 0 and at most 2")
        if self.matte_backend not in {"command", "native"}:
            raise ValueError("generative dance matte backend must be command or native")
        if self.matte_batch_size < 1:
            raise ValueError("generative dance matte batch size must be positive")
        if self.wan_text_length < 32 or self.wan_text_length > 512:
            raise ValueError("generative dance Wan text length must be between 32 and 512")
        if self.wan_temporal_window < 2:
            raise ValueError("generative dance Wan temporal window must be at least 2")
        if self.wan_temporal_context_frames not in {0, 1, 5}:
            raise ValueError(
                "generative dance Wan temporal context must be 0, 1, or 5 frames"
            )
        if self.wan_temporal_context_frames >= self.wan_temporal_window:
            raise ValueError(
                "generative dance Wan temporal context must be smaller than the temporal window"
            )
        if not 0 < self.wan_reference_strength <= 5:
            raise ValueError(
                "generative dance Wan reference strength must be greater than 0 and at most 5"
            )
        if not 0 <= self.identity_audit_threshold <= 1:
            raise ValueError("generative dance identity audit threshold must be between 0 and 1")
        if self.identity_audit_max_retries < 0 or self.identity_audit_max_retries > 2:
            raise ValueError("generative dance identity audit retries must be between 0 and 2")
        if self.wan_inference_steps < 1:
            raise ValueError("generative dance Wan inference steps must be positive")
        if self.wan_min_inference_steps < 1:
            raise ValueError("generative dance Wan minimum inference steps must be positive")
        if self.wan_inference_steps < self.wan_min_inference_steps:
            raise ValueError(
                "generative dance Wan default inference steps cannot be below the configured minimum"
            )
        if self.matte_input_size < 64:
            raise ValueError("generative dance matte input size must be at least 64")
        if not 0 <= self.matte_alpha_threshold <= 1:
            raise ValueError("generative dance matte alpha threshold must be between 0 and 1")
        if not 0 <= self.matte_edge_feather <= 1:
            raise ValueError("generative dance matte edge feather must be between 0 and 1")
        if not 0 <= self.matte_despill_strength <= 1:
            raise ValueError("generative dance matte despill strength must be between 0 and 1")
        if not 0 < self.matte_background_key_distance <= 441.7:
            raise ValueError("generative dance background key distance must be greater than 0 and at most 441.7")
        if not 0 <= self.matte_background_key_strength <= 1:
            raise ValueError("generative dance background key strength must be between 0 and 1")
        if self.stabilize_position_threshold_px < 0:
            raise ValueError("generative dance position stabilization threshold must be non-negative")
        if not 0 <= self.stabilize_position_strength <= 1:
            raise ValueError("generative dance position stabilization strength must be between 0 and 1")
        if self.stabilize_position_window < 3 or self.stabilize_position_window % 2 == 0:
            raise ValueError("generative dance position stabilization window must be an odd number of at least 3")
        if not 0 <= self.transparent_crf <= 63:
            raise ValueError("generative dance transparent CRF must be between 0 and 63")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "artifactRoot": str(self.artifact_root),
            "canvas": asdict(self.canvas),
            "wanModelRevision": self.wan_model_revision,
            "imageCommandConfigured": bool(self.image_command),
            "matteCommandConfigured": bool(self.matte_command),
            "matteBackend": self.matte_backend,
            "matteModel": self.matte_model,
            "matteCheckpointConfigured": bool(self.matte_checkpoint),
            "matteDevice": self.matte_device,
            "matteComputeDtype": self.matte_compute_dtype,
            "mattePythonConfigured": bool(self.matte_python),
            "matteBatchSize": self.matte_batch_size,
            "matteInputSize": self.matte_input_size,
            "matteAlphaThreshold": self.matte_alpha_threshold,
            "matteEdgeFeather": self.matte_edge_feather,
            "matteDespillStrength": self.matte_despill_strength,
            "matteBackgroundKeyDistance": self.matte_background_key_distance,
            "matteBackgroundKeyStrength": self.matte_background_key_strength,
            "anchorSyncEnabled": self.anchor_sync_enabled,
            "stabilizePosition": self.stabilize_position,
            "stabilizePositionThresholdPx": self.stabilize_position_threshold_px,
            "stabilizePositionStrength": self.stabilize_position_strength,
            "stabilizePositionWindow": self.stabilize_position_window,
            "transparentCodec": self.transparent_codec,
            "transparentCrf": self.transparent_crf,
            "retainMatteArtifacts": self.retain_matte_artifacts,
            "wanCommandConfigured": bool(self.wan_command),
            "wanBackend": self.wan_backend,
            "wanCheckpointFormat": self.wan_checkpoint_format,
            "wanConfigFile": str(self.wan_config_file) if self.wan_config_file else None,
            "wanTransformerConfigured": bool(self.wan_transformer_checkpoint),
            "wanOfficialSourceConfigured": bool(self.wan_official_source),
            "wanCompanionsConfigured": all(
                (
                    self.wan_t5_checkpoint,
                    self.wan_t5_tokenizer,
                    self.wan_clip_checkpoint,
                    self.wan_clip_tokenizer,
                    self.wan_vae_checkpoint,
                )
            ),
            "wanLightX2VEnabled": self.wan_lightx2v_enabled,
            "wanLightX2VCheckpoint": str(self.wan_lightx2v_checkpoint) if self.wan_lightx2v_checkpoint else None,
            "wanLightX2VStrength": self.wan_lightx2v_strength,
            "wanDevice": self.wan_device,
            "wanComputeDtype": self.wan_compute_dtype,
            "wanPythonConfigured": bool(self.wan_python),
            "wanInferenceSteps": self.wan_inference_steps,
            "wanGuidanceScale": self.wan_guidance_scale,
            "wanTextLength": self.wan_text_length,
            "wanTemporalWindow": self.wan_temporal_window,
            "wanTemporalContextFrames": self.wan_temporal_context_frames,
            "wanReferenceStrength": self.wan_reference_strength,
            "wanRenderTimeoutSeconds": self.wan_render_timeout_seconds,
            "identityAuditEnabled": self.identity_audit_enabled,
            "identityAuditCommandConfigured": bool(self.identity_audit_command),
            "identityAuditCwd": str(self.identity_audit_cwd) if self.identity_audit_cwd else None,
            "identityAuditThreshold": self.identity_audit_threshold,
            "identityAuditMaxRetries": self.identity_audit_max_retries,
            "stageTimeoutSeconds": self.job_timeout_seconds,
            "maxUploadBytes": self.max_upload_bytes,
            "keepFailedArtifacts": self.keep_failed_artifacts,
        }

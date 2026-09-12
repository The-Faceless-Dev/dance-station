from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class LtxVideoConfig:
    """Runtime and operational policy for one serialized LTX worker."""

    artifact_root: Path = Path("data/ltx-video-jobs")
    model_root: Path = Path("models/ltx-2.5")
    transformer_path: Path = Path("models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors")
    text_encoder_path: Path = Path("models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    video_vae_path: Path = Path("models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    spatial_upsampler_path: Path = Path("models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
    # Included in the image for generated-audio requests, but never loaded by the
    # video-only path.
    audio_vae_path: Path | None = Path("models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors")
    model_name: str = "LTX-2.5-22B-Distilled-NVFP4"
    pipeline: str = "distilled"
    quantization: str = "nvfp4-prequant"
    device: str = "cuda"
    dtype: str = "bfloat16"
    offload_mode: str = "none"
    gpu_required: bool = True
    allow_local_inputs: bool = False
    allow_custom_dimensions: bool = True
    reserve_vram_gb: float = 1.5
    # ConvVAE decode is temporally tiled so long clips do not retain a full
    # clip's intermediate feature maps. Both values are in output video frames.
    vae_temporal_tile_frames: int = 40
    vae_temporal_overlap_frames: int = 16
    residency_reset_threshold_gb: float = 2.0
    max_width: int = 2048
    max_height: int = 2048
    max_frames: int = 0
    max_duration_seconds: float = 0.0
    max_conditioning_images: int = 9
    max_download_bytes: int = 2 * 1024 * 1024 * 1024
    request_timeout_seconds: float = 3600.0
    job_timeout_seconds: float = 3900.0
    keep_intermediates: bool = False
    compile_transformer: bool = False
    use_prompt_enhancement: bool = False
    enforce_nvfp4: bool = True
    enforce_fast_attention: bool = True
    aspect_presets: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "1:1": (768, 768),
        }
    )

    @classmethod
    def from_env(cls) -> "LtxVideoConfig":
        root = _path("LTX_VIDEO_MODEL_ROOT", "models/ltx-2.5")
        return cls(
            artifact_root=_path("LTX_VIDEO_ARTIFACT_ROOT", "data/ltx-video-jobs"),
            model_root=root,
            transformer_path=_path(
                "LTX_VIDEO_TRANSFORMER_PATH",
                str(root / "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"),
            ),
            text_encoder_path=_path(
                "LTX_VIDEO_TEXT_ENCODER_PATH",
                str(root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
            ),
            video_vae_path=_path(
                "LTX_VIDEO_VAE_PATH",
                str(root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            ),
            spatial_upsampler_path=_path(
                "LTX_VIDEO_SPATIAL_UPSAMPLER_PATH",
                str(root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
            ),
            audio_vae_path=_path(
                "LTX_VIDEO_AUDIO_VAE_PATH",
                str(root / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
            ),
            model_name=os.getenv("LTX_VIDEO_MODEL_NAME", cls.model_name),
            pipeline=os.getenv("LTX_VIDEO_PIPELINE", cls.pipeline).strip().lower(),
            quantization=os.getenv("LTX_VIDEO_QUANTIZATION", cls.quantization).strip().lower(),
            device=os.getenv("LTX_VIDEO_DEVICE", cls.device),
            dtype=os.getenv("LTX_VIDEO_DTYPE", cls.dtype),
            offload_mode=os.getenv("LTX_VIDEO_OFFLOAD_MODE", cls.offload_mode).strip().lower(),
            gpu_required=_bool("LTX_VIDEO_GPU_REQUIRED", cls.gpu_required),
            allow_local_inputs=_bool("LTX_VIDEO_ALLOW_LOCAL_INPUTS", cls.allow_local_inputs),
            allow_custom_dimensions=_bool("LTX_VIDEO_ALLOW_CUSTOM_DIMENSIONS", cls.allow_custom_dimensions),
            reserve_vram_gb=float(os.getenv("LTX_VIDEO_RESERVE_VRAM_GB", str(cls.reserve_vram_gb))),
            vae_temporal_tile_frames=int(os.getenv("LTX_VIDEO_VAE_TEMPORAL_TILE_FRAMES", str(cls.vae_temporal_tile_frames))),
            vae_temporal_overlap_frames=int(os.getenv("LTX_VIDEO_VAE_TEMPORAL_OVERLAP_FRAMES", str(cls.vae_temporal_overlap_frames))),
            residency_reset_threshold_gb=float(os.getenv("LTX_VIDEO_RESIDENCY_RESET_THRESHOLD_GB", str(cls.residency_reset_threshold_gb))),
            max_width=int(os.getenv("LTX_VIDEO_MAX_WIDTH", str(cls.max_width))),
            max_height=int(os.getenv("LTX_VIDEO_MAX_HEIGHT", str(cls.max_height))),
            max_frames=int(os.getenv("LTX_VIDEO_MAX_FRAMES", str(cls.max_frames))),
            max_duration_seconds=float(os.getenv("LTX_VIDEO_MAX_DURATION_SECONDS", str(cls.max_duration_seconds))),
            max_conditioning_images=int(os.getenv("LTX_VIDEO_MAX_CONDITIONING_IMAGES", str(cls.max_conditioning_images))),
            max_download_bytes=int(os.getenv("LTX_VIDEO_MAX_DOWNLOAD_BYTES", str(cls.max_download_bytes))),
            request_timeout_seconds=float(os.getenv("LTX_VIDEO_REQUEST_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))),
            job_timeout_seconds=float(os.getenv("LTX_VIDEO_JOB_TIMEOUT_SECONDS", str(cls.job_timeout_seconds))),
            keep_intermediates=_bool("LTX_VIDEO_KEEP_INTERMEDIATES", cls.keep_intermediates),
            compile_transformer=_bool("LTX_VIDEO_COMPILE_TRANSFORMER", cls.compile_transformer),
            use_prompt_enhancement=_bool("LTX_VIDEO_USE_PROMPT_ENHANCEMENT", cls.use_prompt_enhancement),
            enforce_nvfp4=_bool("LTX_VIDEO_ENFORCE_NVFP4", cls.enforce_nvfp4),
            enforce_fast_attention=_bool("LTX_VIDEO_ENFORCE_FAST_ATTENTION", cls.enforce_fast_attention),
        )

    def validate(self) -> None:
        if self.pipeline not in {"distilled"}:
            raise ValueError("LTX_VIDEO_PIPELINE must be distilled until other LTX 2.5 pipelines are implemented")
        if self.quantization not in {"nvfp4-prequant", "nvfp4-cast", "fp8-cast", "none"}:
            raise ValueError("LTX_VIDEO_QUANTIZATION is unsupported")
        if self.enforce_nvfp4 and self.quantization != "nvfp4-prequant":
            raise ValueError("LTX 2.5 production mode requires the pre-quantized NVFP4 transformer")
        if self.device != "cuda" and not self.device.startswith("cuda:"):
            if self.gpu_required:
                raise ValueError("LTX video production mode requires a CUDA device")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("LTX_VIDEO_DTYPE must be bfloat16, float16, or float32")
        if self.offload_mode not in {"none", "cpu", "disk"}:
            raise ValueError("LTX_VIDEO_OFFLOAD_MODE must be none, cpu, or disk")
        if self.reserve_vram_gb < 0 or self.max_width < 32 or self.max_height < 32:
            raise ValueError("LTX video memory and dimension limits are invalid")
        if self.residency_reset_threshold_gb < 0:
            raise ValueError("LTX_VIDEO_RESIDENCY_RESET_THRESHOLD_GB cannot be negative")
        if (
            self.vae_temporal_tile_frames < 8
            or self.vae_temporal_tile_frames % 8 != 0
            or self.vae_temporal_overlap_frames < 0
            or self.vae_temporal_overlap_frames % 8 != 0
            or self.vae_temporal_overlap_frames >= self.vae_temporal_tile_frames
        ):
            raise ValueError(
                "LTX VAE temporal tiling must use frame counts divisible by 8 with "
                "0 <= overlap < tile size"
            )
        if self.max_frames < 0 or self.max_duration_seconds < 0:
            raise ValueError("LTX video limits cannot be negative")
        if self.max_conditioning_images < 0 or self.max_download_bytes <= 0:
            raise ValueError("LTX video input limits are invalid")
        if self.request_timeout_seconds <= 0 or self.job_timeout_seconds <= 0:
            raise ValueError("LTX video timeouts must be positive")
        for name, path in {
            "transformer_path": self.transformer_path,
            "text_encoder_path": self.text_encoder_path,
            "video_vae_path": self.video_vae_path,
            "spatial_upsampler_path": self.spatial_upsampler_path,
        }.items():
            if not str(path).strip():
                raise ValueError(f"{name} must be configured")

    def preflight(self) -> dict[str, object]:
        self.validate()
        paths = {
            "transformer": self.transformer_path,
            "textEncoder": self.text_encoder_path,
            "videoVae": self.video_vae_path,
            "spatialUpsampler": self.spatial_upsampler_path,
        }
        if self.audio_vae_path:
            paths["audioVae"] = self.audio_vae_path
        missing = [name for name, path in paths.items() if not path.is_file()]
        cuda = None
        try:
            import torch

            cuda = {"available": bool(torch.cuda.is_available()), "deviceCount": torch.cuda.device_count()}
            if cuda["available"]:
                device = torch.device(self.device)
                props = torch.cuda.get_device_properties(device)
                cuda.update({"name": props.name, "totalBytes": props.total_memory, "capability": f"{props.major}.{props.minor}"})
        except Exception as exc:
            cuda = {"available": False, "errorType": type(exc).__name__, "error": str(exc)}
        ready = not missing and (not self.gpu_required or bool(cuda and cuda.get("available")))
        return {
            "runtime": "ltx-video",
            "model": self.model_name,
            "pipeline": self.pipeline,
            "quantization": self.quantization,
            "device": self.device,
            "offloadMode": self.offload_mode,
            "audioVaeConfigured": bool(self.audio_vae_path),
            "enforceFastAttention": self.enforce_fast_attention,
            "paths": {name: str(path) for name, path in paths.items()},
            "missing": missing,
            "cuda": cuda,
            "ready": ready,
            "reason": (f"missing model components: {', '.join(missing)}" if missing else None),
        }

    def to_public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in {"artifact_root", "model_root", "transformer_path", "text_encoder_path", "video_vae_path", "spatial_upsampler_path", "audio_vae_path"}:
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        return payload

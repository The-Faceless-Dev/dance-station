from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path | None = None) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class FluxImageConfig:
    """Configuration for the local-only FLUX.2 Klein 4B worker."""

    artifact_root: Path = Path("data/flux-image-jobs")
    model_root: Path = Path("models/flux2-klein-4b")
    model_path: Path | None = None
    text_encoder: Path | None = None
    vae: Path | None = None
    model_name: str = "flux.2-klein-4b"
    device: str = "cuda"
    dtype: str = "bfloat16"
    # Keep the encoder and transformer off the GPU until each phase needs them.
    # The local 3080 path proved this is required for the unquantized Klein
    # weights on a 24 GB production GPU once denoising activations are included.
    cpu_offload: bool = True
    max_prompt_characters: int = 8000
    max_steps: int = 4
    max_true_cfg_scale: float = 12.0
    max_loras: int = 4
    max_lora_scale: float = 3.0
    max_lora_bytes: int = 512 * 1024 * 1024
    avatar_output_width: int = 480
    avatar_output_height: int = 832
    avatar_subject_scale: float = 0.70
    job_timeout_seconds: float = 3600.0
    gpu_required: bool = True
    keep_failed_artifacts: bool = True
    supported_resolutions: tuple[tuple[int, int], ...] = (
        (1328, 1328),
        (1664, 928),
        (928, 1664),
        (1472, 1104),
        (1104, 1472),
        (1584, 1056),
        (1056, 1584),
    )

    @classmethod
    def from_env(cls) -> "FluxImageConfig":
        root = Path(os.getenv("FLUX_IMAGE_MODEL_ROOT", "models/flux2-klein-4b")).expanduser()
        return cls(
            artifact_root=Path(os.getenv("FLUX_IMAGE_ARTIFACT_ROOT", "data/flux-image-jobs")).expanduser(),
            model_root=root,
            model_path=_path("FLUX_IMAGE_MODEL_PATH", root / "flux-2-klein-4b.safetensors"),
            text_encoder=_path("FLUX_IMAGE_TEXT_ENCODER_PATH", root / "text_encoder"),
            vae=_path("FLUX_IMAGE_VAE_PATH", root / "flux_vae.safetensors"),
            model_name=os.getenv("FLUX_IMAGE_MODEL_NAME", "flux.2-klein-4b"),
            device=os.getenv("FLUX_IMAGE_DEVICE", "cuda"),
            dtype=os.getenv("FLUX_IMAGE_DTYPE", "bfloat16"),
            cpu_offload=_bool("FLUX_IMAGE_CPU_OFFLOAD", True),
            max_prompt_characters=int(os.getenv("FLUX_IMAGE_MAX_PROMPT_CHARACTERS", "8000")),
            max_steps=int(os.getenv("FLUX_IMAGE_MAX_STEPS", "4")),
            max_true_cfg_scale=float(os.getenv("FLUX_IMAGE_MAX_TRUE_CFG_SCALE", "12")),
            max_loras=int(os.getenv("FLUX_IMAGE_MAX_LORAS", "4")),
            max_lora_scale=float(os.getenv("FLUX_IMAGE_MAX_LORA_SCALE", "3")),
            max_lora_bytes=int(os.getenv("FLUX_IMAGE_MAX_LORA_BYTES", str(512 * 1024 * 1024))),
            avatar_output_width=int(os.getenv("FLUX_IMAGE_AVATAR_WIDTH", "480")),
            avatar_output_height=int(os.getenv("FLUX_IMAGE_AVATAR_HEIGHT", "832")),
            avatar_subject_scale=float(os.getenv("FLUX_IMAGE_AVATAR_SUBJECT_SCALE", "0.70")),
            job_timeout_seconds=float(os.getenv("FLUX_IMAGE_JOB_TIMEOUT_SECONDS", "3600")),
            gpu_required=_bool("FLUX_IMAGE_GPU_REQUIRED", True),
            keep_failed_artifacts=_bool("FLUX_IMAGE_KEEP_FAILED_ARTIFACTS", True),
        )

    def resolved_paths(self) -> dict[str, Path | None]:
        return {
            "model_path": self.model_path,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
        }

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ValueError("Flux max steps must be positive")
        if self.max_loras < 0:
            raise ValueError("Flux max LoRAs cannot be negative")
        if self.job_timeout_seconds <= 0:
            raise ValueError("Flux job timeout must be positive")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Flux dtype must be bfloat16, float16, or float32")
        if self.device != "cpu" and self.gpu_required and not self.device.startswith("cuda"):
            raise ValueError("Flux GPU-required mode needs a CUDA device")
        if self.max_steps > 4:
            raise ValueError("Distilled FLUX.2 Klein 4B supports at most 4 denoising steps")
        if self.avatar_output_width < 1 or self.avatar_output_height < 1:
            raise ValueError("Flux avatar output dimensions must be positive")
        if not 0.1 <= self.avatar_subject_scale <= 1.0:
            raise ValueError("Flux avatar subject scale must be between 0.1 and 1.0")

    def preflight(self) -> dict[str, Any]:
        self.validate()
        paths = self.resolved_paths()
        required = {name: str(path) if path else None for name, path in paths.items()}
        missing = [name for name, path in paths.items() if path is None or not path.exists()]
        return {
            "modelRevision": "Flux2-Klein-4B",
            "modelFamily": "FLUX.2 Klein",
            "quantization": "bf16 safetensors",
            "required": required,
            "missing": missing,
            "ready": not missing,
            "device": self.device,
            "dtype": self.dtype,
            "cpuOffload": self.cpu_offload,
        }

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_root"] = str(self.artifact_root)
        payload["model_root"] = str(self.model_root)
        for key, path in self.resolved_paths().items():
            payload[key] = str(path) if path else None
        payload["supported_resolutions"] = [f"{width}x{height}" for width, height in self.supported_resolutions]
        return payload

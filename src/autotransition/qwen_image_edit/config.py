from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass(frozen=True)
class QwenImageEditConfig:
    artifact_root: Path = Path("data/qwen-image-edit-jobs")
    model_root: Path = Path("models/qwen-image-edit-2511")
    transformer: Path = Path("models/qwen-image-edit-2511/qwen-image-edit-2511-Q8_0.gguf")
    runtime_backend: str = "diffusers"
    diffusers_model_id: str = "Qwen/Qwen-Image-Edit-2511"
    diffusers_revision: str = "main"
    diffusers_cache: Path = Path("data/qwen-image-edit-huggingface")
    diffusers_dtype: str = "bfloat16"
    diffusers_compute_dtype: str = "bfloat16"
    diffusers_cpu_offload: bool = True
    diffusers_attention_backend: str = "sdpa"
    diffusers_compile: bool = False
    model_name: str = "Qwen-Image-Edit-2511-Q8_0"
    default_width: int = 1328
    default_height: int = 1328
    default_steps: int = 20
    default_cfg_scale: float = 2.5
    max_steps: int = 50
    max_cfg_scale: float = 20.0
    min_dimension: int = 256
    max_dimension: int = 2048
    dimension_alignment: int = 16
    max_pixels: int = 4_000_000
    max_prompt_characters: int = 16_000
    max_references: int = 8
    max_reference_bytes: int = 64 * 1024 * 1024
    max_reference_pixels: int = 16_000_000
    max_loras: int = 8
    max_lora_scale: float = 4.0
    max_lora_bytes: int = 512 * 1024 * 1024
    allow_local_references: bool = False
    allow_local_loras: bool = False
    artifact_root_env: str = "QWEN_IMAGE_EDIT_ARTIFACT_ROOT"
    lora_root: Path = Path("data/qwen-image-edit-loras")
    job_timeout_seconds: float = 3600.0
    startup_timeout_seconds: float = 300.0
    poll_interval_seconds: float = 2.0
    gpu_required: bool = True
    keep_failed_artifacts: bool = True

    @classmethod
    def from_env(cls) -> "QwenImageEditConfig":
        root = _path("QWEN_IMAGE_EDIT_MODEL_ROOT", Path("models/qwen-image-edit-2511"))
        return cls(
            artifact_root=_path("QWEN_IMAGE_EDIT_ARTIFACT_ROOT", Path("data/qwen-image-edit-jobs")),
            model_root=root,
            transformer=_path("QWEN_IMAGE_EDIT_TRANSFORMER", root / "qwen-image-edit-2511-Q8_0.gguf"),
            runtime_backend=os.getenv("QWEN_IMAGE_EDIT_RUNTIME", "diffusers"),
            diffusers_model_id=os.getenv("QWEN_IMAGE_EDIT_DIFFUSERS_MODEL_ID", "Qwen/Qwen-Image-Edit-2511"),
            diffusers_revision=os.getenv("QWEN_IMAGE_EDIT_DIFFUSERS_REVISION", "main"),
            diffusers_cache=_path(
                "QWEN_IMAGE_EDIT_DIFFUSERS_CACHE",
                Path(os.getenv("HF_HOME", "data/qwen-image-edit-huggingface")),
            ),
            diffusers_dtype=os.getenv("QWEN_IMAGE_EDIT_DIFFUSERS_DTYPE", "bfloat16"),
            diffusers_compute_dtype=os.getenv("QWEN_IMAGE_EDIT_DIFFUSERS_COMPUTE_DTYPE", "bfloat16"),
            diffusers_cpu_offload=_bool("QWEN_IMAGE_EDIT_DIFFUSERS_CPU_OFFLOAD", True),
            diffusers_attention_backend=os.getenv("QWEN_IMAGE_EDIT_DIFFUSERS_ATTENTION_BACKEND", "sdpa"),
            diffusers_compile=_bool("QWEN_IMAGE_EDIT_DIFFUSERS_COMPILE", False),
            model_name=os.getenv("QWEN_IMAGE_EDIT_MODEL_NAME", "Qwen-Image-Edit-2511-Q8_0"),
            default_width=int(os.getenv("QWEN_IMAGE_EDIT_DEFAULT_WIDTH", "1328")),
            default_height=int(os.getenv("QWEN_IMAGE_EDIT_DEFAULT_HEIGHT", "1328")),
            default_steps=int(os.getenv("QWEN_IMAGE_EDIT_DEFAULT_STEPS", "20")),
            default_cfg_scale=float(os.getenv("QWEN_IMAGE_EDIT_DEFAULT_CFG_SCALE", "2.5")),
            max_steps=int(os.getenv("QWEN_IMAGE_EDIT_MAX_STEPS", "50")),
            max_cfg_scale=float(os.getenv("QWEN_IMAGE_EDIT_MAX_CFG_SCALE", "20")),
            min_dimension=int(os.getenv("QWEN_IMAGE_EDIT_MIN_DIMENSION", "256")),
            max_dimension=int(os.getenv("QWEN_IMAGE_EDIT_MAX_DIMENSION", "2048")),
            dimension_alignment=int(os.getenv("QWEN_IMAGE_EDIT_DIMENSION_ALIGNMENT", "16")),
            max_pixels=int(os.getenv("QWEN_IMAGE_EDIT_MAX_PIXELS", "4000000")),
            max_prompt_characters=int(os.getenv("QWEN_IMAGE_EDIT_MAX_PROMPT_CHARACTERS", "16000")),
            max_references=int(os.getenv("QWEN_IMAGE_EDIT_MAX_REFERENCES", "8")),
            max_reference_bytes=int(os.getenv("QWEN_IMAGE_EDIT_MAX_REFERENCE_BYTES", str(64 * 1024 * 1024))),
            max_reference_pixels=int(os.getenv("QWEN_IMAGE_EDIT_MAX_REFERENCE_PIXELS", "16000000")),
            max_loras=int(os.getenv("QWEN_IMAGE_EDIT_MAX_LORAS", "8")),
            max_lora_scale=float(os.getenv("QWEN_IMAGE_EDIT_MAX_LORA_SCALE", "4")),
            max_lora_bytes=int(os.getenv("QWEN_IMAGE_EDIT_MAX_LORA_BYTES", str(512 * 1024 * 1024))),
            allow_local_references=_bool("QWEN_IMAGE_EDIT_ALLOW_LOCAL_REFERENCES", False),
            allow_local_loras=_bool("QWEN_IMAGE_EDIT_ALLOW_LOCAL_LORAS", False),
            lora_root=_path("QWEN_IMAGE_EDIT_LORA_ROOT", Path("data/qwen-image-edit-loras")),
            job_timeout_seconds=float(os.getenv("QWEN_IMAGE_EDIT_JOB_TIMEOUT_SECONDS", "3600")),
            startup_timeout_seconds=float(os.getenv("QWEN_IMAGE_EDIT_STARTUP_TIMEOUT_SECONDS", "300")),
            poll_interval_seconds=float(os.getenv("QWEN_IMAGE_EDIT_POLL_INTERVAL_SECONDS", "2")),
            gpu_required=_bool("QWEN_IMAGE_EDIT_GPU_REQUIRED", True),
            keep_failed_artifacts=_bool("QWEN_IMAGE_EDIT_KEEP_FAILED_ARTIFACTS", True),
        )

    def validate(self) -> None:
        if self.runtime_backend != "diffusers":
            raise ValueError("Qwen Image Edit 2511 requires the Diffusers runtime")
        if self.default_steps < 1 or self.default_steps > self.max_steps:
            raise ValueError("default edit steps must be within the configured step range")
        if self.dimension_alignment < 1:
            raise ValueError("dimension alignment must be positive")
        if self.max_references < 1:
            raise ValueError("max references must be positive")
        if self.max_loras < 0:
            raise ValueError("max LoRAs cannot be negative")

    def preflight(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "ready": True,
            "runtime": self.runtime_backend,
            "modelRevision": self.diffusers_model_id,
            "transformerFile": str(self.transformer),
            "transformerFormat": "GGUF",
            "transformerQuantization": "Q8_0",
            "diffusersModel": self.diffusers_model_id,
            "diffusersRevision": self.diffusers_revision,
            "gpuRequired": self.gpu_required,
            "missing": [],
            "diagnostics": [],
        }
        try:
            self.validate()
        except ValueError as exc:
            report["ready"] = False
            report["diagnostics"].append(str(exc))
        if not self.transformer.is_file():
            report["ready"] = False
            report["missing"].append("transformer")
            report["diagnostics"].append(f"Q8 transformer was not found: {self.transformer}")
        elif "q8_0" not in self.transformer.name.lower():
            report["ready"] = False
            report["diagnostics"].append("configured transformer filename does not identify Q8_0")
        return report

    def to_public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
        return values

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass(frozen=True)
class QwenImageConfig:
    """Configuration for the Q8 Qwen-Image worker and its native server."""

    artifact_root: Path = Path("data/qwen-image-jobs")
    model_root: Path = Path("models/qwen-image-2512")
    diffusion_model: Path = Path("models/qwen-image-2512/qwen-image-2512-Q8_0.gguf")
    text_encoder: Path = Path("models/qwen-image-2512/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf")
    vae: Path = Path("models/qwen-image-2512/qwen_image_vae.safetensors")
    runtime_binary: Path = Path("/usr/local/bin/sd-server")
    runtime_host: str = "127.0.0.1"
    runtime_port: int = 1234
    device_backend: str = "cuda"
    model_name: str = "Qwen-Image-2512-Q8_0"
    sampler: str = "euler"
    scheduler: str = "discrete"
    flow_shift: float = 3.0
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
    max_loras: int = 8
    max_lora_scale: float = 4.0
    max_lora_bytes: int = 512 * 1024 * 1024
    allow_local_loras: bool = False
    cpu_offload: bool = True
    flash_attention: bool = True
    mmap: bool = True
    lora_apply_mode: str = "at_runtime"
    job_timeout_seconds: float = 3600.0
    startup_timeout_seconds: float = 300.0
    poll_interval_seconds: float = 2.0
    gpu_required: bool = True
    require_q8_transformer: bool = True
    require_abliterated_encoder: bool = True
    keep_failed_artifacts: bool = True
    lora_root: Path = Path("/var/lib/autotransition/qwen-image-loras")

    @classmethod
    def from_env(cls) -> "QwenImageConfig":
        root = _path("QWEN_IMAGE_MODEL_ROOT", Path("models/qwen-image-2512"))
        return cls(
            artifact_root=_path("QWEN_IMAGE_ARTIFACT_ROOT", Path("data/qwen-image-jobs")),
            model_root=root,
            diffusion_model=_path("QWEN_IMAGE_DIFFUSION_MODEL", root / "qwen-image-2512-Q8_0.gguf"),
            text_encoder=_path(
                "QWEN_IMAGE_TEXT_ENCODER",
                root / "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
            ),
            vae=_path("QWEN_IMAGE_VAE", root / "qwen_image_vae.safetensors"),
            runtime_binary=_path("QWEN_IMAGE_RUNTIME_BINARY", Path("/usr/local/bin/sd-server")),
            runtime_host=os.getenv("QWEN_IMAGE_RUNTIME_HOST", "127.0.0.1"),
            runtime_port=int(os.getenv("QWEN_IMAGE_RUNTIME_PORT", "1234")),
            device_backend=os.getenv("QWEN_IMAGE_BACKEND", "cuda"),
            model_name=os.getenv("QWEN_IMAGE_MODEL_NAME", "Qwen-Image-2512-Q8_0"),
            sampler=os.getenv("QWEN_IMAGE_SAMPLER", "euler"),
            scheduler=os.getenv("QWEN_IMAGE_SCHEDULER", "discrete"),
            flow_shift=float(os.getenv("QWEN_IMAGE_FLOW_SHIFT", "3")),
            default_width=int(os.getenv("QWEN_IMAGE_DEFAULT_WIDTH", "1328")),
            default_height=int(os.getenv("QWEN_IMAGE_DEFAULT_HEIGHT", "1328")),
            default_steps=int(os.getenv("QWEN_IMAGE_DEFAULT_STEPS", "20")),
            default_cfg_scale=float(os.getenv("QWEN_IMAGE_DEFAULT_CFG_SCALE", "2.5")),
            max_steps=int(os.getenv("QWEN_IMAGE_MAX_STEPS", "50")),
            max_cfg_scale=float(os.getenv("QWEN_IMAGE_MAX_CFG_SCALE", "20")),
            min_dimension=int(os.getenv("QWEN_IMAGE_MIN_DIMENSION", "256")),
            max_dimension=int(os.getenv("QWEN_IMAGE_MAX_DIMENSION", "2048")),
            dimension_alignment=int(os.getenv("QWEN_IMAGE_DIMENSION_ALIGNMENT", "16")),
            max_pixels=int(os.getenv("QWEN_IMAGE_MAX_PIXELS", "4000000")),
            max_prompt_characters=int(os.getenv("QWEN_IMAGE_MAX_PROMPT_CHARACTERS", "16000")),
            max_loras=int(os.getenv("QWEN_IMAGE_MAX_LORAS", "8")),
            max_lora_scale=float(os.getenv("QWEN_IMAGE_MAX_LORA_SCALE", "4")),
            max_lora_bytes=int(os.getenv("QWEN_IMAGE_MAX_LORA_BYTES", str(512 * 1024 * 1024))),
            allow_local_loras=_bool("QWEN_IMAGE_ALLOW_LOCAL_LORAS", False),
            cpu_offload=_bool("QWEN_IMAGE_CPU_OFFLOAD", True),
            flash_attention=_bool("QWEN_IMAGE_FLASH_ATTENTION", True),
            mmap=_bool("QWEN_IMAGE_MMAP", True),
            lora_apply_mode=os.getenv("QWEN_IMAGE_LORA_APPLY_MODE", "at_runtime"),
            job_timeout_seconds=float(os.getenv("QWEN_IMAGE_JOB_TIMEOUT_SECONDS", "3600")),
            startup_timeout_seconds=float(os.getenv("QWEN_IMAGE_STARTUP_TIMEOUT_SECONDS", "300")),
            poll_interval_seconds=float(os.getenv("QWEN_IMAGE_POLL_INTERVAL_SECONDS", "2")),
            gpu_required=_bool("QWEN_IMAGE_GPU_REQUIRED", True),
            require_q8_transformer=_bool("QWEN_IMAGE_REQUIRE_Q8_TRANSFORMER", True),
            require_abliterated_encoder=_bool("QWEN_IMAGE_REQUIRE_ABLITERATED_ENCODER", True),
            keep_failed_artifacts=_bool("QWEN_IMAGE_KEEP_FAILED_ARTIFACTS", True),
            lora_root=_path("QWEN_IMAGE_LORA_ROOT", Path("/var/lib/autotransition/qwen-image-loras")),
        )

    def validate(self) -> None:
        if self.device_backend != "cuda" and self.gpu_required:
            raise ValueError("Qwen image worker requires the CUDA backend")
        if self.runtime_port < 1 or self.runtime_port > 65535:
            raise ValueError("Qwen image runtime port is invalid")
        if self.default_steps < 1 or self.max_steps < self.default_steps:
            raise ValueError("Qwen image step defaults are invalid")
        if self.max_steps > 50:
            raise ValueError("Qwen image worker steps cannot exceed 50")
        if self.default_cfg_scale <= 0 or self.max_cfg_scale < self.default_cfg_scale:
            raise ValueError("Qwen image CFG defaults are invalid")
        if self.min_dimension < 1 or self.max_dimension < self.min_dimension:
            raise ValueError("Qwen image dimension bounds are invalid")
        if self.dimension_alignment < 1 or self.max_pixels < self.min_dimension**2:
            raise ValueError("Qwen image dimension constraints are invalid")
        if self.job_timeout_seconds <= 0 or self.startup_timeout_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("Qwen image timeouts must be positive")
        if self.max_loras < 0 or self.max_lora_scale < 0 or self.max_lora_bytes < 1:
            raise ValueError("Qwen image LoRA limits are invalid")
        if self.lora_apply_mode not in {"at_runtime", "immediately"}:
            raise ValueError("Qwen image LoRA apply mode must be at_runtime or immediately")

    def preflight(self) -> dict[str, Any]:
        self.validate()
        files = {
            "runtimeBinary": self.runtime_binary,
            "diffusionModel": self.diffusion_model,
            "textEncoder": self.text_encoder,
            "vae": self.vae,
        }
        missing = [name for name, path in files.items() if not path.is_file()]
        diagnostics: list[str] = []
        diffusion_name = self.diffusion_model.name.lower()
        encoder_name = self.text_encoder.name.lower()
        if self.require_q8_transformer and "q8_0" not in diffusion_name:
            diagnostics.append("diffusionModel must be the Q8_0 Qwen-Image-2512 transformer")
        if self.require_abliterated_encoder and not any(token in encoder_name for token in ("ablit", "heretic")):
            diagnostics.append("textEncoder must be an abliterated/heretic Qwen2.5-VL encoder")
        if self.require_abliterated_encoder and not any(token in encoder_name for token in ("qwen2.5-vl", "qwen2_5_vl", "qwen2-5-vl")):
            diagnostics.append("textEncoder filename does not identify a Qwen2.5-VL encoder")
        if self.require_abliterated_encoder and "q8_0" not in encoder_name:
            diagnostics.append("textEncoder must use the configured Q8_0 GGUF profile")
        gpu = self._gpu_preflight()
        if self.gpu_required and not gpu["ready"]:
            diagnostics.append(gpu["error"])
        return {
            "ready": not missing and not diagnostics,
            "runtime": "stable-diffusion.cpp",
            "modelRevision": self.model_name,
            "transformerQuantization": "Q8_0",
            "textEncoderProfile": "Qwen2.5-VL-7B-Instruct-abliterated",
            "required": {name: str(path) for name, path in files.items()},
            "missing": missing,
            "diagnostics": diagnostics,
            "gpu": gpu,
            "backend": self.device_backend,
            "flashAttention": self.flash_attention,
            "cpuOffload": self.cpu_offload,
            "mmap": self.mmap,
            "loraApplyMode": self.lora_apply_mode,
        }

    def _gpu_preflight(self) -> dict[str, Any]:
        if self.device_backend != "cuda":
            return {"ready": False, "backend": self.device_backend, "error": "CUDA backend is required"}
        executable = shutil.which("nvidia-smi")
        if executable:
            try:
                result = subprocess.run(
                    [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return {"ready": True, "source": "nvidia-smi", "devices": [line.strip() for line in result.stdout.splitlines() if line.strip()]}
                return {"ready": False, "source": "nvidia-smi", "error": f"nvidia-smi failed with exit code {result.returncode}"}
            except (OSError, subprocess.SubprocessError) as exc:
                return {"ready": False, "source": "nvidia-smi", "error": f"CUDA GPU probe failed: {exc}"}
        if os.name != "nt" and Path("/dev/nvidiactl").exists():
            return {"ready": True, "source": "/dev/nvidiactl"}
        return {"ready": False, "error": "CUDA GPU is unavailable; nvidia-smi or /dev/nvidiactl was not found"}

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

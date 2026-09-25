from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MuLaCoverConfig:
    artifact_root: Path = Path("data/mulacover-jobs")
    model_root: Path = Path("models/mulacover")
    model_name: str = "MuLaCover"
    device: str = "cuda:0"
    mulacover_dtype: str = "bfloat16"
    gpu_required: bool = True
    allow_local_inputs: bool = False
    max_download_bytes: int = 2 * 1024 * 1024 * 1024
    min_duration_seconds: float = 1.0
    default_duration_seconds: float = 30.0
    max_duration_seconds: float = 300.0
    max_lyrics_characters: int = 100000
    max_tags_characters: int = 8192
    max_top_k: int = 8192
    minimum_free_vram_gb: float = 20.0
    recommended_vram_gb: float = 24.0
    request_timeout_seconds: float = 3600.0
    job_timeout_seconds: float = 3900.0
    startup_timeout_seconds: float = 1800.0
    keep_intermediates: bool = True

    @classmethod
    def from_env(cls) -> "MuLaCoverConfig":
        return cls(
            artifact_root=Path(os.getenv("MULACOVER_ARTIFACT_ROOT", "data/mulacover-jobs")).expanduser(),
            model_root=Path(os.getenv("MULACOVER_MODEL_ROOT", "models/mulacover")).expanduser(),
            model_name=os.getenv("MULACOVER_MODEL_NAME", cls.model_name),
            device=os.getenv("MULACOVER_DEVICE", cls.device),
            mulacover_dtype=os.getenv("MULACOVER_DTYPE", cls.mulacover_dtype),
            gpu_required=_bool("MULACOVER_GPU_REQUIRED", cls.gpu_required),
            allow_local_inputs=_bool("MULACOVER_ALLOW_LOCAL_INPUTS", cls.allow_local_inputs),
            max_download_bytes=int(os.getenv("MULACOVER_MAX_DOWNLOAD_BYTES", str(cls.max_download_bytes))),
            min_duration_seconds=float(os.getenv("MULACOVER_MIN_DURATION_SECONDS", str(cls.min_duration_seconds))),
            default_duration_seconds=float(os.getenv("MULACOVER_DEFAULT_DURATION_SECONDS", str(cls.default_duration_seconds))),
            max_duration_seconds=float(os.getenv("MULACOVER_MAX_DURATION_SECONDS", str(cls.max_duration_seconds))),
            max_lyrics_characters=int(os.getenv("MULACOVER_MAX_LYRICS_CHARACTERS", str(cls.max_lyrics_characters))),
            max_tags_characters=int(os.getenv("MULACOVER_MAX_TAGS_CHARACTERS", str(cls.max_tags_characters))),
            max_top_k=int(os.getenv("MULACOVER_MAX_TOP_K", str(cls.max_top_k))),
            minimum_free_vram_gb=float(os.getenv("MULACOVER_MINIMUM_FREE_VRAM_GB", str(cls.minimum_free_vram_gb))),
            recommended_vram_gb=float(os.getenv("MULACOVER_RECOMMENDED_VRAM_GB", str(cls.recommended_vram_gb))),
            request_timeout_seconds=float(os.getenv("MULACOVER_REQUEST_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))),
            job_timeout_seconds=float(os.getenv("MULACOVER_JOB_TIMEOUT_SECONDS", str(cls.job_timeout_seconds))),
            startup_timeout_seconds=float(os.getenv("MULACOVER_STARTUP_TIMEOUT_SECONDS", str(cls.startup_timeout_seconds))),
            keep_intermediates=_bool("MULACOVER_KEEP_INTERMEDIATES", cls.keep_intermediates),
        )

    def validate(self) -> None:
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("MULACOVER_DEVICE must be cuda:0 or cpu")
        if self.gpu_required and self.device == "cpu":
            raise ValueError("MULACOVER_GPU_REQUIRED forbids CPU inference")
        if self.mulacover_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("MULACOVER_DTYPE must be bfloat16, float16, or float32")
        if self.max_download_bytes <= 0 or self.max_lyrics_characters < 1 or self.max_tags_characters < 1:
            raise ValueError("MuLaCover input limits must be positive")
        if not math.isfinite(self.min_duration_seconds) or not 0 < self.min_duration_seconds <= self.max_duration_seconds:
            raise ValueError("MuLaCover duration bounds are invalid")
        if not self.min_duration_seconds <= self.default_duration_seconds <= self.max_duration_seconds:
            raise ValueError("MULACOVER_DEFAULT_DURATION_SECONDS is outside the configured bounds")
        if not math.isfinite(self.minimum_free_vram_gb) or self.minimum_free_vram_gb <= 0:
            raise ValueError("MULACOVER_MINIMUM_FREE_VRAM_GB must be positive")
        if self.recommended_vram_gb < self.minimum_free_vram_gb:
            raise ValueError("recommended VRAM cannot be below minimum VRAM")
        if self.request_timeout_seconds <= 0 or self.job_timeout_seconds <= 0 or self.startup_timeout_seconds <= 0:
            raise ValueError("MuLaCover timeouts must be positive")

    def _gpu_report(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"available": False, "errorType": type(exc).__name__, "error": str(exc)}
        if result.returncode != 0 or not result.stdout.strip():
            return {"available": False, "error": (result.stderr or result.stdout).strip()[-1000:]}
        devices = []
        for line in result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) >= 5:
                try:
                    total = float(parts[2])
                    free = float(parts[3])
                except ValueError:
                    continue
                devices.append({"index": parts[0], "name": parts[1], "totalMiB": total, "freeMiB": free, "driver": parts[4]})
        return {"available": bool(devices), "devices": devices}

    def preflight(self) -> dict[str, Any]:
        self.validate()
        gpu = self._gpu_report()
        required = [
            "MuLaCover/config.json", "MuLaCover/gen_config.json", "MuLaCover/model.safetensors.index.json",
            "HeartCodec-oss/config.json", "HeartCodec-oss/model.safetensors.index.json",
            "Qwen3-Embedding-0.6B/config.json", "Qwen3-Embedding-0.6B/model.safetensors",
        ]
        missing = [item for item in required if not (self.model_root / item).is_file()]
        if not (self.model_root / "SymbolicTranscriptor/yourmt3/last.ckpt").is_file():
            missing.append("SymbolicTranscriptor/yourmt3/last.ckpt")
        chord_files = list((self.model_root / "SymbolicTranscriptor/chord").glob("*.best.sdict")) if (self.model_root / "SymbolicTranscriptor/chord").is_dir() else []
        if len(chord_files) < 5:
            missing.append("SymbolicTranscriptor/chord/*.best.sdict (5 files)")
        free_vram = None
        if gpu.get("devices"):
            index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            for item in gpu["devices"]:
                if int(item["index"]) == index:
                    free_vram = float(item["freeMiB"]) / 1024
                    break
        if self.gpu_required and not gpu.get("available"):
            missing.append("CUDA GPU")
        if self.gpu_required and free_vram is not None and free_vram < self.minimum_free_vram_gb:
            missing.append(f"{self.minimum_free_vram_gb:.1f} GiB free VRAM (found {free_vram:.1f} GiB)")
        return {
            "runtime": "mulacover",
            "model": self.model_name,
            "modelRoot": str(self.model_root.resolve()),
            "device": self.device,
            "dtype": self.mulacover_dtype,
            "lazyLoad": True,
            "gpuRequired": self.gpu_required,
            "minimumFreeVramGiB": self.minimum_free_vram_gb,
            "recommendedVramGiB": self.recommended_vram_gb,
            "gpu": gpu,
            "freeVramGiB": free_vram,
            "modelPresent": not missing,
            "missing": missing,
            "ready": not missing,
            "reason": f"missing required components: {', '.join(missing)}" if missing else None,
            "source": "HeartMuLa/MuLaCover",
        }

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_root"] = str(self.artifact_root)
        payload["model_root"] = str(self.model_root)
        return payload

    def resolve_cli(self) -> str | None:
        return shutil.which("mulacover")

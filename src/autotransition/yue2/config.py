from __future__ import annotations

import os
import shutil
import subprocess
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class Yue2Config:
    """Configuration for one serialized native YuE2 GGUF worker."""

    artifact_root: Path = Path("data/yue2-jobs")
    model_root: Path = Path("models/yue2-3b-gguf")
    cli_path: str = "audiocpp_cli"
    model_file: str = "yue2-3b-q8_0.gguf"
    vae_file: str = "yue2-vae-f16.gguf"
    sidecars: tuple[str, ...] = (
        "sidecars/yue2-model-config.json",
        "sidecars/yue2-generation-config.json",
        "sidecars/yue2-qwen.tiktoken",
        "sidecars/yue2-vae-config.json",
    )
    model_name: str = "YuE2-3B-Q8_0-F16-VAE"
    audio_cpp_revision: str = "dev"
    backend: str = "cuda"
    device: str = "0"
    threads: int = 8
    gpu_required: bool = True
    default_style: str = "English, modern pop, clear lead vocal, polished full-band production"
    default_cot: str = "off"
    default_inference_steps: int = 8
    semantic_tokens_per_second: float = 25.0
    min_inference_steps: int = 1
    max_inference_steps: int = 64
    max_lyrics_characters: int = 100000
    max_style_characters: int = 4000
    max_request_options: int = 32
    request_timeout_seconds: float = 3600.0
    job_timeout_seconds: float = 3900.0
    keep_intermediates: bool = True

    @classmethod
    def from_env(cls) -> "Yue2Config":
        root = _path("YUE2_MODEL_ROOT", "models/yue2-3b-gguf")
        return cls(
            artifact_root=_path("YUE2_ARTIFACT_ROOT", "data/yue2-jobs"),
            model_root=root,
            cli_path=os.getenv("YUE2_AUDIOCPP_CLI", "audiocpp_cli"),
            model_file=os.getenv("YUE2_MODEL_FILE", cls.model_file),
            vae_file=os.getenv("YUE2_VAE_FILE", cls.vae_file),
            model_name=os.getenv("YUE2_MODEL_NAME", cls.model_name),
            audio_cpp_revision=os.getenv("YUE2_AUDIOCPP_REVISION", cls.audio_cpp_revision),
            backend=os.getenv("YUE2_BACKEND", cls.backend).strip().lower(),
            device=os.getenv("YUE2_DEVICE", cls.device),
            threads=int(os.getenv("YUE2_THREADS", str(cls.threads))),
            gpu_required=_bool("YUE2_GPU_REQUIRED", cls.gpu_required),
            default_style=os.getenv("YUE2_DEFAULT_STYLE", cls.default_style),
            default_cot=os.getenv("YUE2_DEFAULT_COT", cls.default_cot).strip().lower(),
            default_inference_steps=int(os.getenv("YUE2_DEFAULT_INFERENCE_STEPS", str(cls.default_inference_steps))),
            semantic_tokens_per_second=float(os.getenv("YUE2_SEMANTIC_TOKENS_PER_SECOND", str(cls.semantic_tokens_per_second))),
            min_inference_steps=int(os.getenv("YUE2_MIN_INFERENCE_STEPS", str(cls.min_inference_steps))),
            max_inference_steps=int(os.getenv("YUE2_MAX_INFERENCE_STEPS", str(cls.max_inference_steps))),
            max_lyrics_characters=int(os.getenv("YUE2_MAX_LYRICS_CHARACTERS", str(cls.max_lyrics_characters))),
            max_style_characters=int(os.getenv("YUE2_MAX_STYLE_CHARACTERS", str(cls.max_style_characters))),
            max_request_options=int(os.getenv("YUE2_MAX_REQUEST_OPTIONS", str(cls.max_request_options))),
            request_timeout_seconds=float(os.getenv("YUE2_REQUEST_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))),
            job_timeout_seconds=float(os.getenv("YUE2_JOB_TIMEOUT_SECONDS", str(cls.job_timeout_seconds))),
            keep_intermediates=_bool("YUE2_KEEP_INTERMEDIATES", cls.keep_intermediates),
        )

    @property
    def model_path(self) -> Path:
        return self.model_root / self.model_file

    @property
    def vae_path(self) -> Path:
        return self.model_root / self.vae_file

    def resolve_cli(self) -> str | None:
        configured = Path(self.cli_path).expanduser()
        if configured.is_file():
            return str(configured.resolve())
        return shutil.which(self.cli_path)

    def validate(self) -> None:
        if self.backend != "cuda":
            raise ValueError("YUE2_BACKEND must be cuda; CPU fallback is intentionally disabled")
        if self.threads < 1:
            raise ValueError("YUE2_THREADS must be positive")
        if self.default_cot not in {"off", "melody", "full"}:
            raise ValueError("YUE2_DEFAULT_COT must be off, melody, or full")
        if self.min_inference_steps < 1 or self.max_inference_steps < self.min_inference_steps:
            raise ValueError("YuE2 inference step bounds are invalid")
        if not self.min_inference_steps <= self.default_inference_steps <= self.max_inference_steps:
            raise ValueError("YUE2_DEFAULT_INFERENCE_STEPS is outside its configured bounds")
        if self.max_lyrics_characters < 1 or self.max_style_characters < 1:
            raise ValueError("YuE2 text limits must be positive")
        if not self.default_style.strip():
            raise ValueError("YUE2_DEFAULT_STYLE must be non-empty")
        if not math.isfinite(self.semantic_tokens_per_second) or self.semantic_tokens_per_second <= 0:
            raise ValueError("YUE2_SEMANTIC_TOKENS_PER_SECOND must be positive")
        if self.max_request_options < 0 or self.request_timeout_seconds <= 0 or self.job_timeout_seconds <= 0:
            raise ValueError("YuE2 request limits and timeouts must be positive")
        if not self.model_file or Path(self.model_file).name != self.model_file:
            raise ValueError("YUE2_MODEL_FILE must be a filename inside YUE2_MODEL_ROOT")
        if not self.vae_file or Path(self.vae_file).name != self.vae_file:
            raise ValueError("YUE2_VAE_FILE must be a filename inside YUE2_MODEL_ROOT")

    def _gpu_report(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"available": False, "errorType": type(exc).__name__, "error": str(exc)}
        if result.returncode != 0 or not result.stdout.strip():
            return {"available": False, "error": (result.stderr or result.stdout).strip()[-1000:]}
        rows = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) == 3:
                rows.append({"name": parts[0], "totalMiB": parts[1], "freeMiB": parts[2]})
        return {"available": bool(rows), "devices": rows}

    def preflight(self) -> dict[str, Any]:
        self.validate()
        cli = self.resolve_cli()
        report: dict[str, Any] = {
            "runtime": "yue2-audio-cpp",
            "model": self.model_name,
            "audioCppRevision": self.audio_cpp_revision,
            "backend": self.backend,
            "device": self.device,
            "threads": self.threads,
            "profile": {"main": self.model_file, "vae": self.vae_file},
            "paths": {
                "cli": cli or self.cli_path,
                "modelRoot": str(self.model_root.resolve()),
                "mainModel": str(self.model_path.resolve()),
                "vae": str(self.vae_path.resolve()),
            },
            "cliPresent": bool(cli),
            "mainModelPresent": self.model_path.is_file(),
            "vaePresent": self.vae_path.is_file(),
            "sidecars": {name: (self.model_root / name).is_file() for name in self.sidecars},
            "gpu": self._gpu_report(),
        }
        missing = []
        if not cli:
            missing.append("audiocpp_cli")
        if not self.model_path.is_file():
            missing.append(self.model_file)
        if not self.vae_path.is_file():
            missing.append(self.vae_file)
        missing.extend(name for name in self.sidecars if not (self.model_root / name).is_file())
        if self.gpu_required and not report["gpu"].get("available"):
            missing.append("CUDA GPU")
        report["missing"] = missing
        report["ready"] = not missing
        report["reason"] = f"missing required components: {', '.join(missing)}" if missing else None
        return report

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_root"] = str(self.artifact_root)
        payload["model_root"] = str(self.model_root)
        return payload

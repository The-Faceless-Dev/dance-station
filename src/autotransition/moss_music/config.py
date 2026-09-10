from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MossMusicConfig:
    """Configuration for one serialized MOSS-Music GPU worker."""

    artifact_root: Path = Path("data/moss-music-jobs")
    model_root: Path = Path("models/moss-music-8b-instruct")
    model_name: str = "MOSS-Music-8B-Instruct"
    backend: str = "sglang"
    sglang_url: str = "http://127.0.0.1:30000"
    sglang_endpoint: str = "/generate"
    sglang_health_path: str = "/health"
    device: str = "cuda"
    dtype: str = "bfloat16"
    gpu_required: bool = True
    allow_local_audio_paths: bool = False
    check_backend_on_preflight: bool = True
    audio_sample_rate: int = 16000
    event_resolution_ms: int = 80
    min_event_resolution_ms: int = 40
    max_event_resolution_ms: int = 500
    max_audio_seconds: float = 1800.0
    max_download_bytes: int = 1024 * 1024 * 1024
    max_prompt_characters: int = 16000
    default_temperature: float = 0.0
    request_timeout_seconds: float = 3600.0
    job_timeout_seconds: float = 3900.0
    max_timeline_cells: int = 100000
    keep_failed_artifacts: bool = True

    @classmethod
    def from_env(cls) -> "MossMusicConfig":
        model_root = Path(os.getenv("MOSS_MUSIC_MODEL_ROOT", "models/moss-music-8b-instruct")).expanduser()
        return cls(
            artifact_root=Path(os.getenv("MOSS_MUSIC_ARTIFACT_ROOT", "data/moss-music-jobs")).expanduser(),
            model_root=model_root,
            model_name=os.getenv("MOSS_MUSIC_MODEL_NAME", "MOSS-Music-8B-Instruct"),
            backend=os.getenv("MOSS_MUSIC_BACKEND", "sglang").strip().lower(),
            sglang_url=os.getenv("MOSS_MUSIC_SGLANG_URL", "http://127.0.0.1:30000").rstrip("/"),
            sglang_endpoint=os.getenv("MOSS_MUSIC_SGLANG_ENDPOINT", "/generate"),
            sglang_health_path=os.getenv("MOSS_MUSIC_SGLANG_HEALTH_PATH", "/health"),
            device=os.getenv("MOSS_MUSIC_DEVICE", "cuda"),
            dtype=os.getenv("MOSS_MUSIC_DTYPE", "bfloat16"),
            gpu_required=_bool("MOSS_MUSIC_GPU_REQUIRED", True),
            allow_local_audio_paths=_bool("MOSS_MUSIC_ALLOW_LOCAL_AUDIO_PATHS", False),
            check_backend_on_preflight=_bool("MOSS_MUSIC_CHECK_BACKEND_ON_PREFLIGHT", True),
            audio_sample_rate=int(os.getenv("MOSS_MUSIC_AUDIO_SAMPLE_RATE", "16000")),
            event_resolution_ms=int(os.getenv("MOSS_MUSIC_EVENT_RESOLUTION_MS", "80")),
            min_event_resolution_ms=int(os.getenv("MOSS_MUSIC_MIN_EVENT_RESOLUTION_MS", "40")),
            max_event_resolution_ms=int(os.getenv("MOSS_MUSIC_MAX_EVENT_RESOLUTION_MS", "500")),
            max_audio_seconds=float(os.getenv("MOSS_MUSIC_MAX_AUDIO_SECONDS", "1800")),
            max_download_bytes=int(os.getenv("MOSS_MUSIC_MAX_DOWNLOAD_BYTES", str(1024 * 1024 * 1024))),
            max_prompt_characters=int(os.getenv("MOSS_MUSIC_MAX_PROMPT_CHARACTERS", "16000")),
            default_temperature=float(os.getenv("MOSS_MUSIC_DEFAULT_TEMPERATURE", "0")),
            request_timeout_seconds=float(os.getenv("MOSS_MUSIC_REQUEST_TIMEOUT_SECONDS", "3600")),
            job_timeout_seconds=float(os.getenv("MOSS_MUSIC_JOB_TIMEOUT_SECONDS", "3900")),
            max_timeline_cells=int(os.getenv("MOSS_MUSIC_MAX_TIMELINE_CELLS", "100000")),
            keep_failed_artifacts=_bool("MOSS_MUSIC_KEEP_FAILED_ARTIFACTS", True),
        )

    def validate(self) -> None:
        if self.backend not in {"sglang", "mock"}:
            raise ValueError("MOSS_MUSIC_BACKEND must be sglang or mock")
        if self.audio_sample_rate != 16000:
            raise ValueError("MOSS-Music requires a 16000 Hz analysis input")
        if self.min_event_resolution_ms < 1 or self.max_event_resolution_ms < self.min_event_resolution_ms:
            raise ValueError("MOSS event resolution bounds are invalid")
        if not self.min_event_resolution_ms <= self.event_resolution_ms <= self.max_event_resolution_ms:
            raise ValueError("MOSS event resolution is outside its configured bounds")
        if self.max_audio_seconds <= 0 or self.max_download_bytes <= 0:
            raise ValueError("MOSS audio limits must be positive")
        if self.max_prompt_characters < 1:
            raise ValueError("MOSS prompt limit must be positive")
        if self.request_timeout_seconds <= 0 or self.job_timeout_seconds <= 0:
            raise ValueError("MOSS timeouts must be positive")
        if self.max_timeline_cells < 1:
            raise ValueError("MOSS timeline cell limit must be positive")
        if self.gpu_required and self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("MOSS GPU-required mode needs a CUDA device")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("MOSS dtype must be bfloat16, float16, or float32")

    def preflight(self) -> dict[str, object]:
        self.validate()
        model_path = self.model_root.expanduser()
        return {
            "runtime": "moss-music",
            "model": self.model_name,
            "modelRoot": str(model_path),
            "modelPresent": model_path.exists(),
            "backend": self.backend,
            "sglangUrl": self.sglang_url,
            "device": self.device,
            "dtype": self.dtype,
            "gpuRequired": self.gpu_required,
            "allowLocalAudioPaths": self.allow_local_audio_paths,
            "eventResolutionMs": self.event_resolution_ms,
            "audioSampleRate": self.audio_sample_rate,
        }

    def to_public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["artifact_root"] = str(self.artifact_root)
        payload["model_root"] = str(self.model_root)
        return payload

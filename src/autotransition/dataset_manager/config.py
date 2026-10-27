"""Configuration for the standalone dataset manager service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class DatasetManagerConfig:
    storage_root: Path = Path("data/vevo2-dataset-manager")
    model_root: Path = Path("models/vevo2-dataset-manager")
    host: str = "127.0.0.1"
    port: int = 7877
    max_upload_bytes: int = 1_000_000_000
    max_workers: int = 1
    separation_timeout_seconds: float = 900.0
    sample_rate: int = 24_000
    language: str = "en"
    asr_model: str = "small.en"
    asr_device: str = "auto"
    asr_compute_type: str = "int8"
    separator_primary_model: str = "vocals_mel_band_roformer.ckpt"
    # audio-separator's published Demucs v4 profile is named htdemucs_ft.yaml;
    # this is the concrete file for the documented htdemucs_ft model.
    separator_fallback_model: str = "htdemucs_ft.yaml"
    separator_runtime_dir: Path = Path("runtimes/Source-Separation")
    sofa_command: str | None = None
    sofa_script: Path | None = None
    sofa_checkpoint: Path | None = None
    sofa_dictionary: Path | None = None
    sofa_python: str | None = None
    nfa_script: Path | None = None
    nfa_model: str = "stt_en_fastconformer_hybrid_large_pc"
    nfa_device: str = "cuda"
    nfa_python: str | None = None
    rmvpe_checkpoint: Path | None = None
    rmvpe_device: str = "cuda:0"

    @classmethod
    def from_env(cls) -> "DatasetManagerConfig":
        def integer(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def optional_path(name: str) -> Path | None:
            value = os.getenv(name)
            return Path(value).expanduser().resolve() if value else None

        def existing_default(value: str) -> Path | None:
            path = Path(value).expanduser().resolve()
            return path if path.exists() else None

        return cls(
            storage_root=_path("DATASET_MANAGER_ROOT", "data/vevo2-dataset-manager"),
            model_root=_path("DATASET_MANAGER_MODEL_ROOT", "models/vevo2-dataset-manager"),
            host=os.getenv("DATASET_MANAGER_HOST", "127.0.0.1"),
            port=integer("DATASET_MANAGER_PORT", 7877),
            max_upload_bytes=integer("DATASET_MANAGER_MAX_UPLOAD_BYTES", 1_000_000_000),
            max_workers=max(1, integer("DATASET_MANAGER_MAX_WORKERS", 1)),
            separation_timeout_seconds=float(os.getenv("DATASET_MANAGER_SEPARATION_TIMEOUT_SECONDS", "900")),
            sample_rate=integer("DATASET_MANAGER_SAMPLE_RATE", 24_000),
            language=os.getenv("DATASET_MANAGER_LANGUAGE", "en"),
            asr_model=os.getenv("DATASET_MANAGER_ASR_MODEL", "small.en"),
            asr_device=os.getenv("DATASET_MANAGER_ASR_DEVICE", "auto"),
            asr_compute_type=os.getenv("DATASET_MANAGER_ASR_COMPUTE_TYPE", "int8"),
            separator_primary_model=os.getenv(
                "DATASET_MANAGER_SEPARATOR_PRIMARY_MODEL", "vocals_mel_band_roformer.ckpt"
            ),
            separator_fallback_model=os.getenv(
                "DATASET_MANAGER_SEPARATOR_FALLBACK_MODEL", "htdemucs_ft.yaml"
            ),
            separator_runtime_dir=_path(
                "DATASET_MANAGER_SEPARATOR_RUNTIME", "runtimes/Source-Separation"
            ),
            sofa_command=os.getenv("DATASET_MANAGER_SOFA_COMMAND") or None,
            sofa_script=optional_path("DATASET_MANAGER_SOFA_SCRIPT"),
            sofa_checkpoint=optional_path("DATASET_MANAGER_SOFA_CHECKPOINT"),
            sofa_dictionary=optional_path("DATASET_MANAGER_SOFA_DICTIONARY"),
            sofa_python=os.getenv("DATASET_MANAGER_SOFA_PYTHON") or None,
            nfa_script=optional_path("DATASET_MANAGER_NFA_SCRIPT") or existing_default("data/runtime/NeMo-source/tools/nemo_forced_aligner/align.py"),
            nfa_model=os.getenv(
                "DATASET_MANAGER_NFA_MODEL", "stt_en_fastconformer_hybrid_large_pc"
            ),
            nfa_device=os.getenv("DATASET_MANAGER_NFA_DEVICE", "cuda"),
            nfa_python=os.getenv("DATASET_MANAGER_NFA_PYTHON") or str(existing_default("data/runtime/SoulX-Singer-preprocess-venv/Scripts/python.exe") or existing_default("data/runtime/Vevo2-venv-system/Scripts/python.exe") or "") or None,
            rmvpe_checkpoint=optional_path("DATASET_MANAGER_RMVPE_CHECKPOINT") or existing_default("data/runtime/YingMusic-RMVPE-checkpoints/rmvpe.pt"),
            rmvpe_device=os.getenv("DATASET_MANAGER_RMVPE_DEVICE", "cuda:0"),
        )

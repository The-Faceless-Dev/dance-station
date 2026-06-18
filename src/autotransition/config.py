"""Central configuration objects for transition workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputConfig:
    """Where generated pipeline artifacts should be written."""

    root_dir: Path = Path("data")
    scaffold_dir: Path = Path("data/scaffolds")
    generated_dir: Path = Path("data/generated")
    export_dir: Path = Path("data/exports")
    audio_format: str = "wav"


@dataclass(frozen=True)
class ModelConfig:
    """Model storage and runtime defaults."""

    models_dir: Path = Path("models")
    auto_install: bool = True
    device: str = "auto"
    use_cpu_offload: bool = False
    use_quantization: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    """External runtime configuration."""

    ace_step_dir: Path = Path("runtimes/ACE-Step-1.5")
    api_host: str = "127.0.0.1"
    api_port: int = 8001
    api_timeout_seconds: float = 10.0
    api_startup_timeout_seconds: float = 600.0
    poll_interval_seconds: float = 2.0
    generation_timeout_seconds: float = 1800.0

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"


@dataclass(frozen=True)
class TransitionConfig:
    """User-tunable transition settings."""

    context_seconds: float = 16.0
    repaint_overlap_seconds: float = 4.0
    new_section_seconds: float = 32.0
    output: OutputConfig = OutputConfig()
    model: ModelConfig = ModelConfig()
    candidate_count: int = 2
    seed: int | None = None
    bpm_hint: float | None = None
    key_hint: str | None = None

    @property
    def tail_seconds(self) -> float:
        return self.context_seconds + self.repaint_overlap_seconds

    @property
    def scaffold_seconds(self) -> float:
        return self.tail_seconds + self.new_section_seconds

    @property
    def repainting_start_seconds(self) -> float:
        return self.context_seconds


DEFAULT_CONFIG = TransitionConfig()

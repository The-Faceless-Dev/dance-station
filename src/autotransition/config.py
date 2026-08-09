"""Central configuration objects for transition workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


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
    side_step_dir: Path = Path("runtimes/Side-Step")
    source_separation_dir: Path = Path("runtimes/Source-Separation")
    rvc_dir: Path = Path("runtimes/Seed-VC")
    tango_flux_dir: Path = Path("runtimes/TangoFlux")
    rvc_host: str = "127.0.0.1"
    rvc_port: int = 7898
    rvc_timeout_seconds: float = 240.0
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
    # Backward-compatible name. In the ACE-Step request this is the number of
    # source seconds before the continuation point included in the repaint range.
    repaint_overlap_seconds: float = 2.0
    new_section_seconds: float = 32.0
    output: OutputConfig = OutputConfig()
    model: ModelConfig = ModelConfig()
    candidate_count: int = 2
    seed: int | None = None
    bpm_hint: float | None = None
    key_hint: str | None = None

    @property
    def tail_seconds(self) -> float:
        return self.context_seconds

    @property
    def repaint_margin_seconds(self) -> float:
        return self.repaint_overlap_seconds

    @property
    def scaffold_seconds(self) -> float:
        return self.tail_seconds + self.new_section_seconds

    @property
    def repainting_start_seconds(self) -> float:
        return max(0.0, self.context_seconds - self.repaint_margin_seconds)


@dataclass(frozen=True)
class AvatarConfig:
    """Runtime configuration for the avatar-generation worker.

    Model runtimes remain external to this package. Each adapter is invoked
    through an argv template so model repositories and large checkpoints can
    be mounted independently in local and production containers.
    """

    artifact_root: Path = Path("data/avatar-jobs")
    max_attempts: int = 3
    max_description_characters: int = 800
    max_prompt_characters: int = 4000
    job_timeout_seconds: float = 1800.0
    image_timeout_seconds: float = 600.0
    mesh_timeout_seconds: float = 900.0
    rig_timeout_seconds: float = 900.0
    max_image_bytes: int = 20 * 1024 * 1024
    max_reference_pixels: int = 16_000_000
    min_image_width: int = 256
    min_image_height: int = 256
    gpu_required: bool = True
    # The built-in validator is dependency-free and should be enabled in the
    # production profile. Unit tests can explicitly disable it when using
    # intentionally minimal fake GLBs.
    require_deformation_validator: bool = True
    prompt_policy_version: str = "bipedal-source-v1"
    image_model_revision: str = "flux.2-klein-4b"
    mesh_model_revision: str = "stable-fast-3d-configured"
    rig_model_revision: str = "skintokens-tokenrig-configured"
    image_model_license: str = "Apache-2.0"
    mesh_model_license: str = "configured-runtime-license"
    rig_model_license: str = "configured-runtime-license"
    image_command: str | None = None
    mesh_command: str | None = None
    rig_command: str | None = None
    reskin_command: str | None = None
    deformation_validator_command: str | None = None
    keep_debug_artifacts: bool = False

    @classmethod
    def from_env(cls) -> "AvatarConfig":
        """Load deployment-specific paths and command templates from env."""

        def integer(name: str, default: int) -> int:
            value = os.getenv(name)
            return default if value is None else int(value)

        def number(name: str, default: float) -> float:
            value = os.getenv(name)
            return default if value is None else float(value)

        return cls(
            artifact_root=Path(os.getenv("AVATAR_ARTIFACT_ROOT", "data/avatar-jobs")),
            max_attempts=min(3, max(1, integer("AVATAR_MAX_ATTEMPTS", 3))),
            max_description_characters=integer("AVATAR_MAX_DESCRIPTION_CHARACTERS", 800),
            max_prompt_characters=integer("AVATAR_MAX_PROMPT_CHARACTERS", 4000),
            job_timeout_seconds=number("AVATAR_JOB_TIMEOUT_SECONDS", 1800.0),
            image_timeout_seconds=number("AVATAR_IMAGE_TIMEOUT_SECONDS", 600.0),
            mesh_timeout_seconds=number("AVATAR_MESH_TIMEOUT_SECONDS", 900.0),
            rig_timeout_seconds=number("AVATAR_RIG_TIMEOUT_SECONDS", 900.0),
            max_image_bytes=integer("AVATAR_MAX_IMAGE_BYTES", 20 * 1024 * 1024),
            max_reference_pixels=integer("AVATAR_MAX_REFERENCE_PIXELS", 16_000_000),
            min_image_width=integer("AVATAR_MIN_IMAGE_WIDTH", 256),
            min_image_height=integer("AVATAR_MIN_IMAGE_HEIGHT", 256),
            gpu_required=os.getenv("AVATAR_GPU_REQUIRED", "1").lower() not in {"0", "false", "no"},
            require_deformation_validator=os.getenv("AVATAR_REQUIRE_DEFORMATION_VALIDATOR", "1").lower() in {"1", "true", "yes"},
            prompt_policy_version=os.getenv("AVATAR_PROMPT_POLICY_VERSION", "bipedal-source-v1"),
            image_model_revision=os.getenv("AVATAR_IMAGE_MODEL_REVISION", "flux.2-klein-4b"),
            mesh_model_revision=os.getenv("AVATAR_MESH_MODEL_REVISION", "stable-fast-3d-configured"),
            rig_model_revision=os.getenv("AVATAR_RIG_MODEL_REVISION", "skintokens-tokenrig-configured"),
            image_model_license=os.getenv("AVATAR_IMAGE_MODEL_LICENSE", "Apache-2.0"),
            mesh_model_license=os.getenv("AVATAR_MESH_MODEL_LICENSE", "configured-runtime-license"),
            rig_model_license=os.getenv("AVATAR_RIG_MODEL_LICENSE", "configured-runtime-license"),
            image_command=os.getenv("AVATAR_IMAGE_COMMAND") or os.getenv("FLUX2_KLEIN_COMMAND"),
            mesh_command=os.getenv("AVATAR_MESH_COMMAND"),
            rig_command=os.getenv("AVATAR_RIG_COMMAND"),
            reskin_command=os.getenv("AVATAR_RESKIN_COMMAND"),
            deformation_validator_command=os.getenv("AVATAR_DEFORMATION_VALIDATOR_COMMAND"),
            keep_debug_artifacts=os.getenv("AVATAR_KEEP_DEBUG_ARTIFACTS", "0").lower() in {"1", "true", "yes"},
        )


DEFAULT_CONFIG = TransitionConfig()

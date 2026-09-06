from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


FluxImageJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
FluxImageStage = Literal[
    "validate_request",
    "load_model",
    "load_loras",
    "encode_prompt",
    "denoise",
    "decode",
    "frame_output",
    "validate_output",
    "finalizing",
]


def _safe_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


@dataclass(frozen=True)
class FluxLoRARequest:
    """A caller-supplied adapter retained for the worker request contract."""

    source_url: str
    file_name: str = "adapter.safetensors"
    scale: float = 1.0
    path: Path | None = None

    def validate(self, *, max_scale: float, max_name_characters: int) -> None:
        if self.path is None:
            parts = urlsplit(self.source_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("LoRA source_url must use an HTTP(S) URL")
        if not self.file_name.strip() or len(self.file_name) > max_name_characters:
            raise ValueError("LoRA file_name is invalid")
        if Path(self.file_name).suffix.lower() != ".safetensors":
            raise ValueError("LoRA files must use the .safetensors format")
        if self.scale < 0 or self.scale > max_scale:
            raise ValueError(f"LoRA scale must be between 0 and {max_scale}")
        if self.path is not None and not self.path.is_file():
            raise ValueError(f"LoRA file was not found: {self.path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceUrl": _safe_url(self.source_url),
            "fileName": Path(self.file_name).name,
            "scale": self.scale,
            "path": str(self.path) if self.path else None,
        }


@dataclass(frozen=True)
class FluxImageRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1328
    height: int = 1328
    steps: int = 4
    true_cfg_scale: float = 1.0
    seed: int | None = None
    loras: tuple[FluxLoRARequest, ...] = ()
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def validate(self, config: Any) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if len(self.prompt) > config.max_prompt_characters:
            raise ValueError(f"prompt must be {config.max_prompt_characters} characters or fewer")
        if len(self.negative_prompt) > config.max_prompt_characters:
            raise ValueError(f"negative_prompt must be {config.max_prompt_characters} characters or fewer")
        if (self.width, self.height) not in config.supported_resolutions:
            supported = ", ".join(f"{width}x{height}" for width, height in config.supported_resolutions)
            raise ValueError(f"resolution must be one of: {supported}")
        if self.steps < 1 or self.steps > config.max_steps:
            raise ValueError(f"steps must be between 1 and {config.max_steps}")
        if self.true_cfg_scale < 1 or self.true_cfg_scale > config.max_true_cfg_scale:
            raise ValueError(f"true_cfg_scale must be between 1 and {config.max_true_cfg_scale}")
        if len(self.loras) > config.max_loras:
            raise ValueError(f"at most {config.max_loras} LoRAs may be supplied")
        for lora in self.loras:
            lora.validate(max_scale=config.max_lora_scale, max_name_characters=240)
        for name, value in (("external_job_id", self.external_job_id), ("payment_intent_id", self.payment_intent_id)):
            if value is not None and (not value.strip() or len(value) > 240):
                raise ValueError(f"{name} must be a non-empty value of 240 characters or fewer")
        if self.external_job_id and any(char in self.external_job_id for char in "\\/"):
            raise ValueError("external_job_id cannot contain path separators")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loras"] = [lora.to_dict() for lora in self.loras]
        payload["external_job_id"] = self.external_job_id
        payload["payment_intent_id"] = self.payment_intent_id
        return payload


@dataclass(frozen=True)
class FluxImageArtifact:
    name: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class FluxImageFailure:
    code: str
    message: str
    stage: FluxImageStage | str
    retryable: bool
    attempt: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FluxImageJob:
    id: str
    status: FluxImageJobStatus
    request: dict[str, Any]
    stage: FluxImageStage | None = None
    progress: float = 0.0
    attempt: int = 0
    artifacts: list[FluxImageArtifact] = field(default_factory=list)
    failure: FluxImageFailure | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        payload["failure"] = self.failure.to_dict() if self.failure else None
        payload["failureSummary"] = {
            "failureCode": self.failure.code,
            "stage": self.failure.stage,
            "message": self.failure.message,
        } if self.failure else None
        payload["failureCode"] = self.failure.code if self.failure else None
        payload["refundRequired"] = bool(self.failure)
        payload["refundReason"] = "flux_image_generation_failed" if self.failure else None
        return payload

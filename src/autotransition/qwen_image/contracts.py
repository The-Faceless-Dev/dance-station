from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


QwenImageJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
QwenImageStage = Literal[
    "validate_request",
    "model_preflight",
    "prepare_loras",
    "load_model",
    "encode_prompt",
    "denoise",
    "decode",
    "validate_output",
    "finalizing",
    "cleanup",
]


def _safe_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


@dataclass(frozen=True)
class QwenLoRARequest:
    source_url: str = ""
    file_name: str = "adapter.safetensors"
    scale: float = 1.0
    is_high_noise: bool = False
    path: Path | None = None

    def validate(self, config: Any) -> None:
        if self.path is None:
            parts = urlsplit(self.source_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("LoRA source_url must use an HTTP(S) URL")
        elif not config.allow_local_loras:
            raise ValueError("local LoRA paths are disabled for this worker")
        if not self.file_name.strip() or len(self.file_name) > 240:
            raise ValueError("LoRA file_name is invalid")
        if Path(self.file_name).suffix.lower() != ".safetensors":
            raise ValueError("LoRA files must use the .safetensors format")
        if self.scale < 0 or self.scale > config.max_lora_scale:
            raise ValueError(f"LoRA scale must be between 0 and {config.max_lora_scale}")
        if self.path is not None:
            if not self.path.is_file():
                raise ValueError(f"LoRA file was not found: {self.path}")
            if self.path.stat().st_size > config.max_lora_bytes:
                raise ValueError("LoRA exceeds the worker size limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceUrl": _safe_url(self.source_url) if self.source_url else "",
            "fileName": Path(self.file_name).name,
            "scale": self.scale,
            "isHighNoise": self.is_high_noise,
            "path": str(self.path) if self.path else None,
        }


@dataclass(frozen=True)
class QwenImageRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1328
    height: int = 1328
    steps: int = 20
    cfg_scale: float = 2.5
    seed: int | None = None
    loras: tuple[QwenLoRARequest, ...] = ()
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def validate(self, config: Any) -> None:
        config.validate()
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if len(self.prompt) > config.max_prompt_characters:
            raise ValueError(f"prompt must be {config.max_prompt_characters} characters or fewer")
        if len(self.negative_prompt) > config.max_prompt_characters:
            raise ValueError(f"negative_prompt must be {config.max_prompt_characters} characters or fewer")
        for name, value in (("width", self.width), ("height", self.height)):
            if value < config.min_dimension or value > config.max_dimension:
                raise ValueError(f"{name} must be between {config.min_dimension} and {config.max_dimension}")
            if value % config.dimension_alignment:
                raise ValueError(f"{name} must be aligned to {config.dimension_alignment} pixels")
        if self.width * self.height > config.max_pixels:
            raise ValueError(f"image area cannot exceed {config.max_pixels} pixels")
        if self.steps < 1 or self.steps > config.max_steps:
            raise ValueError(f"steps must be between 1 and {config.max_steps}")
        if self.cfg_scale <= 0 or self.cfg_scale > config.max_cfg_scale:
            raise ValueError(f"cfg_scale must be between 0 and {config.max_cfg_scale}")
        if len(self.loras) > config.max_loras:
            raise ValueError(f"at most {config.max_loras} LoRAs may be supplied")
        for lora in self.loras:
            lora.validate(config)
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
class QwenImageArtifact:
    name: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class QwenImageFailure:
    code: str
    message: str
    stage: QwenImageStage | str
    retryable: bool
    attempt: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QwenImageJob:
    id: str
    status: QwenImageJobStatus
    request: dict[str, Any]
    stage: QwenImageStage | None = None
    progress: float = 0.0
    attempt: int = 0
    artifacts: list[QwenImageArtifact] = field(default_factory=list)
    failure: QwenImageFailure | None = None
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
        payload["refundReason"] = "qwen_image_generation_failed" if self.failure else None
        return payload


def _merged_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("inputs", "parameters"):
        value = payload.get(key)
        if isinstance(value, dict):
            merged.update(value)
    merged.update({key: value for key, value in payload.items() if key not in {"inputs", "parameters", "callback"}})
    return merged


def _lora_values(payload: dict[str, Any], parameters: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for source in (payload.get("loras"), payload.get("lora"), parameters.get("loras"), parameters.get("lora")):
        if isinstance(source, list):
            values.extend(item for item in source if isinstance(item, dict))
    inputs = payload.get("inputs")
    if isinstance(inputs, list):
        values.extend(
            item for item in inputs
            if isinstance(item, dict) and str(item.get("role") or item.get("type") or "").lower() in {"lora", "adapter", "lora_adapter"}
        )
    return values


def request_from_payload(payload: dict[str, Any], config: Any | None = None) -> QwenImageRequest:
    parameters = _merged_parameters(payload)
    default_width = int(getattr(config, "default_width", 1328)) if config is not None else 1328
    default_height = int(getattr(config, "default_height", 1328)) if config is not None else 1328
    default_steps = int(getattr(config, "default_steps", 20)) if config is not None else 20
    default_cfg = float(getattr(config, "default_cfg_scale", 2.5)) if config is not None else 2.5
    loras: list[QwenLoRARequest] = []
    seen: set[tuple[str, str, float, bool]] = set()
    for item in _lora_values(payload, parameters):
        source_url = str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "")
        file_name = Path(str(item.get("fileName") or item.get("file_name") or "adapter.safetensors")).name
        scale = float(item.get("scale", item.get("multiplier", 1.0)))
        high_noise = bool(item.get("isHighNoise", item.get("is_high_noise", False)))
        path = Path(str(item["path"])).expanduser() if item.get("path") else None
        key = (source_url, str(path or ""), scale, high_noise)
        if key in seen:
            continue
        seen.add(key)
        loras.append(QwenLoRARequest(source_url, file_name, scale, high_noise, path))
    cfg = parameters.get("cfg_scale", parameters.get("cfg", parameters.get("guidance_scale", parameters.get("true_cfg_scale", default_cfg))))
    return QwenImageRequest(
        prompt=str(parameters.get("prompt") or ""),
        negative_prompt=str(parameters.get("negative_prompt") or parameters.get("negativePrompt") or ""),
        width=int(parameters.get("width", default_width)),
        height=int(parameters.get("height", default_height)),
        steps=int(parameters.get("steps", parameters.get("num_inference_steps", default_steps))),
        cfg_scale=float(cfg),
        seed=int(parameters["seed"]) if parameters.get("seed") is not None else None,
        loras=tuple(loras),
        external_job_id=str(payload.get("job_id") or parameters.get("job_id") or parameters.get("external_job_id") or "") or None,
        payment_intent_id=str(parameters.get("payment_intent_id") or parameters.get("paymentIntentId") or "") or None,
    )

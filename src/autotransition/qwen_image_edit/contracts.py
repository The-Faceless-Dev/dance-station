from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit


def _safe_file_name(value: str, default: str) -> str:
    name = Path(value or default).name
    if not name or name in {".", ".."} or len(name) > 240:
        raise ValueError("reference file name is invalid")
    if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("reference images must be PNG, JPEG, or WebP")
    return name


@dataclass(frozen=True)
class QwenImageEditReference:
    source_url: str = ""
    file_name: str = "reference.png"
    path: Path | None = None

    def validate(self, config: Any) -> None:
        if self.path is None:
            parts = urlsplit(self.source_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("reference source_url must use an HTTP(S) URL")
        elif not config.allow_local_references:
            raise ValueError("local reference paths are disabled for this worker")
        elif not self.path.is_file():
            raise ValueError(f"reference image was not found: {self.path}")
        _safe_file_name(self.file_name, "reference.png")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "file_name": self.file_name,
            "path": str(self.path) if self.path else None,
        }


@dataclass(frozen=True)
class QwenImageEditLoRA:
    source_url: str = ""
    file_name: str = "adapter.safetensors"
    scale: float = 1.0
    path: Path | None = None

    def validate(self, config: Any) -> None:
        if self.path is None:
            parts = urlsplit(self.source_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("LoRA source_url must use an HTTP(S) URL")
        elif not config.allow_local_loras:
            raise ValueError("local LoRA paths are disabled for this worker")
        elif not self.path.is_file():
            raise ValueError(f"LoRA was not found: {self.path}")
        if Path(self.file_name).suffix.lower() != ".safetensors":
            raise ValueError("LoRA files must use the .safetensors format")
        if not self.file_name.strip() or len(self.file_name) > 240:
            raise ValueError("LoRA file name is invalid")
        if self.scale < 0 or self.scale > config.max_lora_scale:
            raise ValueError(f"LoRA scale must be between 0 and {config.max_lora_scale}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "file_name": self.file_name,
            "scale": self.scale,
            "path": str(self.path) if self.path else None,
        }


@dataclass(frozen=True)
class QwenImageEditRequest:
    prompt: str
    references: tuple[QwenImageEditReference, ...]
    negative_prompt: str = ""
    width: int = 1328
    height: int = 1328
    steps: int = 20
    cfg_scale: float = 2.5
    seed: int | None = None
    loras: tuple[QwenImageEditLoRA, ...] = ()
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
        if not self.references:
            raise ValueError("at least one reference image is required")
        if len(self.references) > config.max_references:
            raise ValueError(f"at most {config.max_references} reference images may be supplied")
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
        for reference in self.references:
            reference.validate(config)
        for lora in self.loras:
            lora.validate(config)
        if self.external_job_id and any(char in self.external_job_id for char in "\\/"):
            raise ValueError("external_job_id cannot contain path separators")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["references"] = [reference.to_dict() for reference in self.references]
        payload["loras"] = [lora.to_dict() for lora in self.loras]
        return payload


QwenImageEditStage = Literal[
    "validate_request",
    "model_preflight",
    "prepare_references",
    "prepare_loras",
    "load_model",
    "encode_prompt",
    "denoise",
    "decode",
    "validate_output",
    "finalizing",
    "cleanup",
]


@dataclass(frozen=True)
class QwenImageEditFailure:
    code: str
    message: str
    stage: QwenImageEditStage | str
    retryable: bool
    attempt: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merged_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("inputs", "parameters"):
        value = payload.get(key)
        if isinstance(value, dict):
            merged.update(value)
    merged.update({key: value for key, value in payload.items() if key not in {"inputs", "parameters", "callback"}})
    return merged


def _reference_values(payload: dict[str, Any], parameters: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in ("reference_images", "referenceImages", "references", "images"):
        source = payload.get(key) if key in payload else parameters.get(key)
        if isinstance(source, list):
            values.extend(source)
    inputs = payload.get("inputs")
    if isinstance(inputs, list):
        values.extend(
            item for item in inputs
            if isinstance(item, (str, dict))
            and str(item.get("role") or item.get("type") or "reference").lower()
            in {"reference", "reference_image", "image", "input_image"}
        )
    if not values:
        single = parameters.get("image") or parameters.get("reference_image")
        if single:
            values.append(single)
    return values


def _reference_from_value(value: Any, index: int) -> QwenImageEditReference:
    if isinstance(value, str):
        return QwenImageEditReference(source_url=value, file_name=f"reference-{index:02d}.png")
    if not isinstance(value, dict):
        raise ValueError("each reference image must be a URL or object")
    nested = value.get("image") if isinstance(value.get("image"), dict) else value
    source_url = str(
        nested.get("sourceUrl")
        or nested.get("source_url")
        or nested.get("url")
        or nested.get("imageUrl")
        or (value.get("image") if isinstance(value.get("image"), str) else "")
        or ""
    )
    path = Path(str(nested["path"])).expanduser() if nested.get("path") else None
    file_name = Path(str(nested.get("fileName") or nested.get("file_name") or f"reference-{index:02d}.png")).name
    return QwenImageEditReference(source_url=source_url, file_name=file_name, path=path)


def _lora_values(payload: dict[str, Any], parameters: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for source in (payload.get("loras"), payload.get("lora"), parameters.get("loras"), parameters.get("lora")):
        if isinstance(source, list):
            values.extend(source)
    inputs = payload.get("inputs")
    if isinstance(inputs, list):
        values.extend(
            item for item in inputs
            if isinstance(item, dict)
            and str(item.get("role") or item.get("type") or "").lower() in {"lora", "adapter", "lora_adapter"}
        )
    return values


def request_from_payload(payload: dict[str, Any], config: Any | None = None) -> QwenImageEditRequest:
    parameters = _merged_parameters(payload)
    default_width = int(getattr(config, "default_width", 1328)) if config is not None else 1328
    default_height = int(getattr(config, "default_height", 1328)) if config is not None else 1328
    default_steps = int(getattr(config, "default_steps", 20)) if config is not None else 20
    default_cfg = float(getattr(config, "default_cfg_scale", 2.5)) if config is not None else 2.5
    references = tuple(
        _reference_from_value(value, index)
        for index, value in enumerate(_reference_values(payload, parameters), start=1)
    )
    loras: list[QwenImageEditLoRA] = []
    seen: set[tuple[str, str, float]] = set()
    for item in _lora_values(payload, parameters):
        if not isinstance(item, dict):
            raise ValueError("each LoRA must be an object")
        source_url = str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "")
        file_name = Path(str(item.get("fileName") or item.get("file_name") or "adapter.safetensors")).name
        scale = float(item.get("scale", item.get("multiplier", 1.0)))
        path = Path(str(item["path"])).expanduser() if item.get("path") else None
        key = (source_url, str(path or ""), scale)
        if key in seen:
            continue
        seen.add(key)
        loras.append(QwenImageEditLoRA(source_url, file_name, scale, path))
    cfg = parameters.get(
        "cfg_scale",
        parameters.get("cfg", parameters.get("guidance_scale", parameters.get("true_cfg_scale", default_cfg))),
    )
    return QwenImageEditRequest(
        prompt=str(parameters.get("prompt") or ""),
        references=references,
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

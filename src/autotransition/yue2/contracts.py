from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Yue2Status = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class Yue2Request:
    lyrics: str
    style: str | None = None
    cot: Literal["off", "melody", "full"] | None = None
    num_inference_steps: int | None = None
    seed: int | None = None
    duration_seconds: float | None = None
    guidance_scale: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    request_options: dict[str, str | int | float | bool] = field(default_factory=dict)
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def validate(self, config: Any) -> None:
        if not self.lyrics.strip():
            raise ValueError("YuE2 requires non-empty lyrics/text")
        if len(self.lyrics) > config.max_lyrics_characters:
            raise ValueError(f"lyrics must be {config.max_lyrics_characters} characters or fewer")
        if self.style is not None and len(self.style) > config.max_style_characters:
            raise ValueError(f"style must be {config.max_style_characters} characters or fewer")
        if self.cot is not None and self.cot not in {"off", "melody", "full"}:
            raise ValueError("cot must be off, melody, or full")
        if self.num_inference_steps is not None and not config.min_inference_steps <= self.num_inference_steps <= config.max_inference_steps:
            raise ValueError("num_inference_steps is outside the configured bounds")
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError("seed must be a non-negative integer")
        for name, value in {
            "duration_seconds": self.duration_seconds,
            "guidance_scale": self.guidance_scale,
            "temperature": self.temperature,
        }.items():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.max_tokens is not None and (not isinstance(self.max_tokens, int) or self.max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        if len(self.request_options) > config.max_request_options:
            raise ValueError(f"request_options must contain {config.max_request_options} entries or fewer")
        for key, value in self.request_options.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(key)):
                raise ValueError(f"invalid YuE2 request option name: {key}")
            if "\x00" in str(value) or "\n" in str(value):
                raise ValueError(f"invalid YuE2 request option value for {key}")
        for name, value in (("external_job_id", self.external_job_id), ("payment_intent_id", self.payment_intent_id)):
            if value is not None and (not value.strip() or len(value) > 240):
                raise ValueError(f"{name} must be a non-empty value of 240 characters or fewer")
        if self.external_job_id and any(char in self.external_job_id for char in "\\/"):
            raise ValueError("external_job_id cannot contain path separators")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Yue2Artifact:
    name: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Yue2Failure:
    code: str
    message: str
    stage: str
    retryable: bool
    attempt: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Yue2Job:
    id: str
    status: Yue2Status
    request: dict[str, Any]
    stage: str | None = None
    progress: float = 0.0
    attempt: int = 0
    message: str | None = None
    artifacts: list[Yue2Artifact] = field(default_factory=list)
    failure: Yue2Failure | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(item) for item in self.artifacts]
        payload["failure"] = self.failure.to_dict() if self.failure else None
        payload["failureCode"] = self.failure.code if self.failure else None
        payload["refundRequired"] = bool(self.failure)
        payload["refundReason"] = "yue2_music_generation_failed" if self.failure else None
        return payload

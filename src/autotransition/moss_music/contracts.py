from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


MossMusicJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
MossMusicStage = Literal[
    "validate_request",
    "acquire_audio",
    "normalize_audio",
    "load_backend",
    "build_timeline",
    "moss_analysis",
    "parse_and_validate",
    "merge_analysis",
    "write_artifacts",
    "upload_artifacts",
    "finalizing",
]


def _safe_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


@dataclass(frozen=True)
class MossAudioInput:
    source_url: str = ""
    filename: str = "audio"
    path: Path | None = None

    def validate(self) -> None:
        if self.path is None:
            parts = urlsplit(self.source_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("audio requires an HTTP(S) source_url or a local path")
        elif not self.path.is_file():
            raise ValueError(f"audio file was not found: {self.path}")
        name = Path(self.filename).name
        if not name or name in {".", ".."}:
            raise ValueError("audio filename is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceUrl": _safe_url(self.source_url),
            "filename": Path(self.filename).name,
            "path": str(self.path) if self.path else None,
        }


@dataclass(frozen=True)
class MossMusicRequest:
    audio: MossAudioInput
    analysis_profile: str = "visual_sync_v1"
    prompt: str | None = None
    event_resolution_ms: int = 80
    include_semantic_events: bool = True
    include_dense_features: bool = True
    max_new_tokens: int = 4096
    temperature: float = 0.0
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def validate(self, config: Any) -> None:
        self.audio.validate()
        if self.audio.path is not None and not config.allow_local_audio_paths:
            raise ValueError("local audio paths are disabled for this worker")
        if not self.analysis_profile.strip() or len(self.analysis_profile) > 120:
            raise ValueError("analysis_profile is invalid")
        if self.prompt is not None and len(self.prompt) > config.max_prompt_characters:
            raise ValueError(f"prompt must be {config.max_prompt_characters} characters or fewer")
        if not config.min_event_resolution_ms <= self.event_resolution_ms <= config.max_event_resolution_ms:
            raise ValueError("event_resolution_ms is outside the configured bounds")
        if self.max_new_tokens < 1 or self.max_new_tokens > config.max_new_tokens:
            raise ValueError(f"max_new_tokens must be between 1 and {config.max_new_tokens}")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        for name, value in (("external_job_id", self.external_job_id), ("payment_intent_id", self.payment_intent_id)):
            if value is not None and (not value.strip() or len(value) > 240):
                raise ValueError(f"{name} must be a non-empty value of 240 characters or fewer")
        if self.external_job_id and any(char in self.external_job_id for char in "\\/"):
            raise ValueError("external_job_id cannot contain path separators")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["audio"] = self.audio.to_dict()
        return payload


@dataclass(frozen=True)
class MossMusicArtifact:
    name: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class MossMusicFailure:
    code: str
    message: str
    stage: MossMusicStage | str
    retryable: bool
    attempt: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MossMusicJob:
    id: str
    status: MossMusicJobStatus
    request: dict[str, Any]
    stage: MossMusicStage | None = None
    progress: float = 0.0
    attempt: int = 0
    message: str | None = None
    artifacts: list[MossMusicArtifact] = field(default_factory=list)
    failure: MossMusicFailure | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        payload["failure"] = self.failure.to_dict() if self.failure else None
        payload["failureCode"] = self.failure.code if self.failure else None
        payload["refundRequired"] = bool(self.failure)
        payload["refundReason"] = "moss_music_analysis_failed" if self.failure else None
        return payload

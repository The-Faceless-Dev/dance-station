"""Stable contracts shared by the avatar worker, adapters, and API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

AvatarJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
AvatarStage = Literal[
    "validate_request",
    "image_generation",
    "image_validation",
    "mesh_generation",
    "mesh_validation",
    "skeleton_fit",
    "rigging",
    "reskinning",
    "rig_validation",
    "runtime_validation",
    "finalizing",
]


@dataclass(frozen=True)
class AvatarRequest:
    """A user request after the upload has been stored in the job directory."""

    description: str
    reference_image: Path | None = None
    prompt: str | None = None
    negative_prompt: str = ""
    retry_prompts: tuple[str, ...] = ()
    prompt_policy_version: str | None = None
    quality: Literal["preview", "runtime", "quality"] = "runtime"
    seed: int | None = None
    max_attempts: int = 3
    retain_debug_artifacts: bool = False
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def validate(self, *, max_description_characters: int, max_prompt_characters: int, max_attempts: int) -> None:
        description = self.description.strip()
        prompt = (self.prompt or description).strip()
        if not prompt and self.reference_image is None:
            raise ValueError("description or prompt is required when no reference image is provided")
        if len(description) > max_description_characters:
            raise ValueError(f"description must be {max_description_characters} characters or fewer")
        if len(prompt) > max_prompt_characters:
            raise ValueError(f"prompt must be {max_prompt_characters} characters or fewer")
        if len(self.negative_prompt) > max_prompt_characters:
            raise ValueError(f"negative_prompt must be {max_prompt_characters} characters or fewer")
        if any(not item.strip() or len(item) > max_prompt_characters for item in self.retry_prompts):
            raise ValueError(f"retry_prompts must contain non-empty values of {max_prompt_characters} characters or fewer")
        if self.quality not in {"preview", "runtime", "quality"}:
            raise ValueError("quality must be preview, runtime, or quality")
        if self.max_attempts < 1 or self.max_attempts > max_attempts:
            raise ValueError(f"max_attempts must be between 1 and {max_attempts}")
        if self.reference_image is not None and not self.reference_image.is_file():
            raise ValueError(f"reference image was not found: {self.reference_image}")
        for field_name, value in (("external_job_id", self.external_job_id), ("payment_intent_id", self.payment_intent_id)):
            if value is not None and (not value.strip() or len(value) > 240):
                raise ValueError(f"{field_name} must be a non-empty value of 240 characters or fewer")
            if field_name == "external_job_id" and value is not None and any(char in value for char in "\\/"):
                raise ValueError("external_job_id cannot contain path separators")

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "reference_image": str(self.reference_image) if self.reference_image else None,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "retry_prompts": list(self.retry_prompts),
            "prompt_policy_version": self.prompt_policy_version,
            "quality": self.quality,
            "seed": self.seed,
            "max_attempts": self.max_attempts,
            "retain_debug_artifacts": self.retain_debug_artifacts,
            "external_job_id": self.external_job_id,
            "payment_intent_id": self.payment_intent_id,
        }


@dataclass(frozen=True)
class AvatarReskinRequest:
    """Reapply a canonical skeleton profile to an existing mesh."""

    mesh: Path
    profile: Path
    quality: Literal["preview", "runtime", "quality"] = "runtime"
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def validate(self, *, max_attempts: int) -> None:
        if self.mesh.suffix.lower() != ".glb" or not self.mesh.is_file():
            raise ValueError(f"source mesh GLB was not found: {self.mesh}")
        if self.profile.suffix.lower() != ".json" or not self.profile.is_file():
            raise ValueError(f"canonical skeleton profile was not found: {self.profile}")
        if self.quality not in {"preview", "runtime", "quality"}:
            raise ValueError("quality must be preview, runtime, or quality")
        for field_name, value in (("external_job_id", self.external_job_id), ("payment_intent_id", self.payment_intent_id)):
            if value is not None and (not value.strip() or len(value) > 240):
                raise ValueError(f"{field_name} must be a non-empty value of 240 characters or fewer")
            if field_name == "external_job_id" and value is not None and any(char in value for char in "\\/"):
                raise ValueError("external_job_id cannot contain path separators")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_type": "reskin",
            "mesh": str(self.mesh),
            "profile": str(self.profile),
            "quality": self.quality,
            "external_job_id": self.external_job_id,
            "payment_intent_id": self.payment_intent_id,
        }


@dataclass(frozen=True)
class AvatarArtifact:
    name: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class AvatarFailure:
    code: str
    message: str
    stage: AvatarStage | str
    retryable: bool
    attempt: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AvatarJob:
    id: str
    status: AvatarJobStatus
    request: dict[str, Any]
    stage: AvatarStage | None = None
    progress: float = 0.0
    attempt: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[AvatarArtifact] = field(default_factory=list)
    failure: AvatarFailure | None = None
    failure_summary: dict[str, Any] | None = None
    refund_required: bool = False
    refund_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        payload["failure"] = self.failure.to_dict() if self.failure else None
        payload["failureSummary"] = self.failure_summary
        payload["refundRequired"] = self.refund_required
        payload["refundReason"] = self.refund_reason
        payload["failureCode"] = self.failure.code if self.failure else None
        return payload


@dataclass(frozen=True)
class AvatarResult:
    status: Literal["succeeded", "failed", "cancelled"]
    job_id: str
    artifacts: tuple[AvatarArtifact, ...] = ()
    failure: AvatarFailure | None = None
    refund_required: bool = False
    refund_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        payload["failure"] = self.failure.to_dict() if self.failure else None
        payload["refundRequired"] = self.refund_required
        payload["refundReason"] = self.refund_reason
        payload["failureCode"] = self.failure.code if self.failure else None
        return payload

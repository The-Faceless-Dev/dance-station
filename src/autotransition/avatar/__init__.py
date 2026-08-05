"""Avatar generation worker primitives."""

from autotransition.avatar.contracts import AvatarJob, AvatarRequest, AvatarResult
from autotransition.avatar.pipeline import AvatarPipeline

__all__ = ["AvatarJob", "AvatarPipeline", "AvatarRequest", "AvatarResult"]

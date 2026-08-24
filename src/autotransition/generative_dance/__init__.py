"""Local proof-of-concept pipeline for image-driven generative dances."""

from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import (
    AvatarReference,
    CanvasContract,
    DanceComposition,
    DanceDriver,
    RenderedSegment,
)
from autotransition.generative_dance.service import GenerativeDanceService

__all__ = [
    "AvatarReference",
    "CanvasContract",
    "DanceComposition",
    "DanceDriver",
    "GenerativeDanceConfig",
    "GenerativeDanceService",
    "RenderedSegment",
]

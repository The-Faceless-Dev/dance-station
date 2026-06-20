"""Named transition presets for creator-friendly workflows."""

from __future__ import annotations

from dataclasses import dataclass

from autotransition.config import TransitionConfig


WORKING_ACE_STEP_PROMPT = (
    "suspenseful cinematic horror theme, dark ambient strings, low percussion, "
    "tense atmosphere, instrumental"
)


@dataclass(frozen=True)
class TransitionPreset:
    """A reusable transition style with defaults users can override."""

    slug: str
    name: str
    description: str
    caption: str
    config: TransitionConfig


PRESETS: dict[str, TransitionPreset] = {
    "smooth-continuation": TransitionPreset(
        slug="smooth-continuation",
        name="Smooth continuation",
        description="Preserve the current groove and continue naturally into the next section.",
        caption=WORKING_ACE_STEP_PROMPT,
        config=TransitionConfig(context_seconds=18.0, repaint_overlap_seconds=2.0, new_section_seconds=30.0),
    ),
    "energy-build": TransitionPreset(
        slug="energy-build",
        name="Energy build",
        description="Use the source tail as context and build intensity into the next clip.",
        caption=(
            "Continue from the existing ending and build energy with stronger drums, "
            "rising movement, and a clean transition into a more intense section."
        ),
        config=TransitionConfig(context_seconds=16.0, repaint_overlap_seconds=2.0, new_section_seconds=40.0),
    ),
    "breakdown": TransitionPreset(
        slug="breakdown",
        name="Breakdown",
        description="Ease out of the current clip into a lower-energy breakdown.",
        caption=(
            "Continue from the existing ending and transition into a spacious breakdown "
            "with reduced drums, clear atmosphere, and musical continuity."
        ),
        config=TransitionConfig(context_seconds=18.0, repaint_overlap_seconds=2.0, new_section_seconds=36.0),
    ),
    "genre-shift": TransitionPreset(
        slug="genre-shift",
        name="Genre shift",
        description="Bend the ending toward a different style while preserving a believable bridge.",
        caption=(
            "Continue from the existing ending. Preserve the tempo and musical center at first, "
            "then transition into the target style with a smooth bridge."
        ),
        config=TransitionConfig(context_seconds=20.0, repaint_overlap_seconds=2.0, new_section_seconds=48.0),
    ),
    "dj-bridge": TransitionPreset(
        slug="dj-bridge",
        name="DJ bridge",
        description="Create a practical beat-friendly segment for mixing or streaming.",
        caption=(
            "Continue from the existing ending with a DJ-friendly transition, steady beat, "
            "clean downbeats, and a mixable intro for the next section."
        ),
        config=TransitionConfig(context_seconds=24.0, repaint_overlap_seconds=2.0, new_section_seconds=48.0),
    ),
}


def get_preset(slug: str) -> TransitionPreset:
    try:
        return PRESETS[slug]
    except KeyError as exc:
        options = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{slug}'. Available presets: {options}") from exc

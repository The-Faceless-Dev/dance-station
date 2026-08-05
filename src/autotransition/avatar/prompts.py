"""Prompt policy for generating rig-friendly humanoid source images."""

from __future__ import annotations

PROMPT_POLICY_VERSION = "humanoid-source-v1"

HUMANOID_PROMPT_PREFIX = (
    "A single full-body humanoid character, centered and fully visible from head to toe, "
    "front-facing neutral A-pose, arms held away from the torso with clear space between "
    "upper arms and body, straight readable limbs, visible hands and feet, natural human "
    "proportions, symmetrical stance, clear silhouette, studio lighting, plain uncluttered "
    "background, high-quality character reference sheet, no props, no clothing fusion with "
    "the background, no crop, no extra people. "
)

HUMANOID_NEGATIVE_PROMPT = (
    "side view, rear view, sitting, crouching, walking pose, crossed limbs, arms touching torso, "
    "hidden hands, hidden feet, cropped body, extra limbs, extra fingers, duplicate character, "
    "vehicle, furniture, text, watermark, busy background, extreme perspective"
)


def compose_avatar_prompt(description: str) -> str:
    """Add fixed rigging constraints without allowing user text to replace them."""

    cleaned = " ".join(description.split())
    return f"{HUMANOID_PROMPT_PREFIX}Character design: {cleaned}"


def compose_retry_prompt(description: str, failure_codes: list[str]) -> str:
    """Make retries address observed source-image failures while preserving the request."""

    corrections: list[str] = []
    if "image_too_small" in failure_codes or "image_invalid" in failure_codes:
        corrections.append("increase subject resolution and keep the entire body in frame")
    if "mesh_missing" in failure_codes or "rig_missing" in failure_codes:
        corrections.append("make the silhouette and limb separation even clearer")
    if not corrections:
        corrections.append("use a cleaner, more explicit A-pose with all joints visible")
    return f"{compose_avatar_prompt(description)} Retry correction: {', '.join(corrections)}."

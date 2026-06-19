"""ACE-Step model profile registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """A model option that can be shown in UI and used by installers."""

    slug: str
    display_name: str
    repo_id: str
    local_dir_name: str
    family: str
    supports_repaint: bool
    quality_label: str
    speed_label: str
    vram_guidance: str
    default_inference_steps: int
    notes: str


ACE_STEP_MODELS: dict[str, ModelProfile] = {
    "acestep-v15-base": ModelProfile(
        slug="acestep-v15-base",
        display_name="ACE-Step 1.5 Base",
        repo_id="ACE-Step/acestep-v15-base",
        local_dir_name="acestep-v15-base",
        family="ACE-Step 1.5",
        supports_repaint=True,
        quality_label="Medium",
        speed_label="Standard",
        vram_guidance="2B DiT profile; use this when compatibility matters more than speed.",
        default_inference_steps=64,
        notes="Foundation 2B model with the broadest task support in the non-XL family.",
    ),
    "acestep-v15-sft": ModelProfile(
        slug="acestep-v15-sft",
        display_name="ACE-Step 1.5 SFT",
        repo_id="ACE-Step/acestep-v15-sft",
        local_dir_name="acestep-v15-sft",
        family="ACE-Step 1.5",
        supports_repaint=True,
        quality_label="High",
        speed_label="Standard",
        vram_guidance="2B DiT profile for users who want higher quality than base.",
        default_inference_steps=64,
        notes="Good default for quality-focused repaint workflows on consumer GPUs.",
    ),
    "acestep-v15-turbo": ModelProfile(
        slug="acestep-v15-turbo",
        display_name="ACE-Step 1.5 Turbo",
        repo_id="ACE-Step/Ace-Step1.5",
        local_dir_name="acestep-v15-turbo",
        family="ACE-Step 1.5",
        supports_repaint=True,
        quality_label="Very high",
        speed_label="Fast",
        vram_guidance="Consumer-friendly 2B profile; recommended starting point.",
        default_inference_steps=8,
        notes="Fast profile suited to iteration, previews, and creator-facing workflows.",
    ),
    "acestep-v15-xl-base": ModelProfile(
        slug="acestep-v15-xl-base",
        display_name="ACE-Step 1.5 XL Base",
        repo_id="ACE-Step/acestep-v15-xl-base",
        local_dir_name="acestep-v15-xl-base",
        family="ACE-Step 1.5 XL",
        supports_repaint=True,
        quality_label="High",
        speed_label="Large",
        vram_guidance="XL 4B profile; expects at least 12GB VRAM with offload/quantization, 20GB preferred.",
        default_inference_steps=64,
        notes="Larger foundation profile for high-quality output when hardware allows.",
    ),
    "acestep-v15-xl-sft": ModelProfile(
        slug="acestep-v15-xl-sft",
        display_name="ACE-Step 1.5 XL SFT",
        repo_id="ACE-Step/acestep-v15-xl-sft",
        local_dir_name="acestep-v15-xl-sft",
        family="ACE-Step 1.5 XL",
        supports_repaint=True,
        quality_label="Very high",
        speed_label="Large",
        vram_guidance="XL 4B profile; expects at least 12GB VRAM with offload/quantization, 20GB preferred.",
        default_inference_steps=64,
        notes="Quality-focused XL profile for users with stronger GPUs.",
    ),
    "acestep-v15-xl-turbo": ModelProfile(
        slug="acestep-v15-xl-turbo",
        display_name="ACE-Step 1.5 XL Turbo",
        repo_id="ACE-Step/acestep-v15-xl-turbo",
        local_dir_name="acestep-v15-xl-turbo",
        family="ACE-Step 1.5 XL",
        supports_repaint=True,
        quality_label="Very high",
        speed_label="Fast XL",
        vram_guidance="XL 4B profile; expects at least 12GB VRAM with offload/quantization, 20GB preferred.",
        default_inference_steps=8,
        notes="Fast XL profile when users want higher quality and have enough VRAM.",
    ),
}


def get_model_profile(slug: str) -> ModelProfile:
    try:
        return ACE_STEP_MODELS[slug]
    except KeyError as exc:
        options = ", ".join(sorted(ACE_STEP_MODELS))
        raise ValueError(f"Unknown model '{slug}'. Available models: {options}") from exc


def repaint_capable_models() -> list[ModelProfile]:
    return [profile for profile in ACE_STEP_MODELS.values() if profile.supports_repaint]

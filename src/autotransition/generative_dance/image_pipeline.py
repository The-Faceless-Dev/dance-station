"""Reference image creation and canvas normalization."""

from __future__ import annotations

import shutil
from pathlib import Path

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command
from autotransition.avatar.adapters.image_generator import CommandImageGenerator
from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import AvatarReference
from autotransition.generative_dance.video import normalize_image


REFERENCE_PROMPT_POLICY = (
    "A single full-body bipedal character, centered and fully visible, front-facing "
    "neutral A-pose, arms held away from the torso with clear space between arms and "
    "body, straight readable limbs, symmetrical stance, studio lighting, seamless "
    "solid contrasting background, high-quality character reference sheet, no props, "
    "no crop, no extra people, character does not cast a shadow. Character design: "
    "{description}"
)


def build_reference_prompt(description: str) -> str:
    value = description.strip()
    if not value:
        raise ValueError("avatar description is required")
    return REFERENCE_PROMPT_POLICY.format(description=value)


class ReferenceImagePipeline:
    def __init__(self, config: GenerativeDanceConfig, store: ArtifactStore):
        self.config = config
        self.store = store

    def create(
        self,
        *,
        reference_id: str,
        description: str,
        uploaded_image: Path | None = None,
        seed: int | None = None,
    ) -> AvatarReference:
        directory = self.store.create_id_dir("references", reference_id)
        prompt = build_reference_prompt(description)
        source = directory / (f"source{uploaded_image.suffix.lower()}" if uploaded_image else "generated.png")
        if uploaded_image is not None:
            if uploaded_image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise ValueError("reference image must be PNG, JPEG, or WebP")
            shutil.copy2(uploaded_image, source)
        else:
            if not self.config.image_command:
                raise AvatarAdapterError(
                    "generative_image_not_configured",
                    "no image generator is configured; set GENERATIVE_DANCE_IMAGE_COMMAND or upload a reference image",
                    retryable=False,
                )
            generator = CommandImageGenerator(self.config.image_command, timeout_seconds=self.config.job_timeout_seconds, cwd=self.config.image_cwd)
            generator.generate(
                prompt=prompt,
                negative_prompt="extra people, cropped body, hidden limbs, text, logo, prop, busy background",
                output=source,
                seed=seed,
                reference_image=None,
                quality="runtime",
            )

        normalized = directory / "normalized-reference.png"
        normalize_image(source, normalized, width=self.config.canvas.width, height=self.config.canvas.height)
        matte: Path | None = None
        if self.config.matte_command:
            matte = directory / "normalized-matte.png"
            run_adapter_command(
                parse_command(self.config.matte_command),
                values={
                    "input": normalized,
                    "source_image": source,
                    "output": matte,
                    "output_dir": directory,
                },
                cwd=self.config.matte_cwd,
                timeout_seconds=self.config.job_timeout_seconds,
                log_dir=directory,
                component="reference-matte",
            )
            if not matte.is_file() or matte.stat().st_size == 0:
                raise AvatarAdapterError(
                    "matte_output_missing",
                    "matte command completed without normalized-matte.png",
                    retryable=True,
                )
        metadata_path = directory / "reference.json"
        reference = AvatarReference(
            id=reference_id,
            description=description.strip(),
            prompt=prompt,
            source_image=source,
            normalized_image=normalized,
            matte_image=matte,
            canvas=self.config.canvas,
            metadata_path=metadata_path,
        )
        self.store.write_json(metadata_path, reference.to_dict())
        return reference

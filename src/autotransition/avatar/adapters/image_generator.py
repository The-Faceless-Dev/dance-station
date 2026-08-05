"""FLUX.2 Klein command boundary.

The official FLUX.2 Klein checkout evolves independently of this package. The
worker therefore invokes the pinned local inference entry point through an argv
template rather than importing a mutable model implementation into the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command


class CommandImageGenerator:
    """Run a text/image-to-image command with prompt files as inputs."""

    def __init__(self, command: str | Sequence[str] | None, *, timeout_seconds: float, cwd: Path | None = None):
        self.command = parse_command(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        output: Path,
        seed: int | None,
        reference_image: Path | None,
        quality: str,
    ) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        prompt_file = output.with_name(output.stem + ".prompt.txt")
        negative_file = output.with_name(output.stem + ".negative.txt")
        prompt_file.write_text(prompt, encoding="utf-8")
        negative_file.write_text(negative_prompt, encoding="utf-8")
        try:
            run_adapter_command(
                self.command,
                values={
                    "prompt_file": prompt_file,
                    "negative_prompt_file": negative_file,
                    "output": output,
                    "seed": seed if seed is not None else "",
                    "reference_image": reference_image or "",
                    "quality": quality,
                },
                cwd=self.cwd,
                timeout_seconds=self.timeout_seconds,
                log_dir=output.parent,
                component="image-generation",
            )
        finally:
            prompt_file.unlink(missing_ok=True)
            negative_file.unlink(missing_ok=True)
        if not output.is_file() or output.stat().st_size == 0:
            raise AvatarAdapterError("image_missing", "image generator completed without an image", retryable=True)
        return output

"""Adapter protocols and common failures."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AvatarAdapterError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = True, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class ImageGenerator(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        output: Path,
        seed: int | None,
        reference_image: Path | None,
        quality: str,
    ) -> Path: ...


class MeshGenerator(Protocol):
    def generate(self, *, image: Path, output_dir: Path, quality: str) -> Path: ...


class RigGenerator(Protocol):
    def generate(self, *, mesh: Path, output: Path, manifest: Path, quality: str) -> tuple[Path, Path]: ...

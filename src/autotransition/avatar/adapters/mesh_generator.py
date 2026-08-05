"""Stable Fast 3D command boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from autotransition.avatar.adapters.command import parse_command, run_adapter_command
from autotransition.avatar.adapters.base import AvatarAdapterError


class CommandMeshGenerator:
    def __init__(self, command: str | Sequence[str] | None, *, timeout_seconds: float, cwd: Path | None = None):
        self.command = parse_command(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def generate(self, *, image: Path, output_dir: Path, quality: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        expected = output_dir / "mesh.glb"
        run_adapter_command(
            self.command,
            values={"image": image, "output_dir": output_dir, "output": expected, "quality": quality},
            cwd=self.cwd,
            timeout_seconds=self.timeout_seconds,
            log_dir=output_dir,
            component="mesh-generation",
        )
        if expected.is_file():
            return expected
        # Stable Fast 3D writes the result under a numbered child directory
        # (for example ``output/0/mesh.glb``) in addition to supporting a
        # flat output directory. Accept both layouts but reject ambiguity.
        candidates = sorted(output_dir.rglob("*.glb"))
        if len(candidates) == 1:
            return candidates[0]
        raise AvatarAdapterError("mesh_missing", "mesh generator completed without a unique GLB", retryable=True)

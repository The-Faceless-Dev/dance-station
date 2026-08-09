"""SkinTokens/TokenRig command boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command


class CommandRigGenerator:
    def __init__(self, command: str | Sequence[str] | None, *, timeout_seconds: float, cwd: Path | None = None):
        self.command = parse_command(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def generate(self, *, mesh: Path, output: Path, manifest: Path, quality: str) -> tuple[Path, Path]:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        run_adapter_command(
            self.command,
            values={"input": mesh, "output": output, "manifest_output": manifest, "quality": quality},
            cwd=self.cwd,
            timeout_seconds=self.timeout_seconds,
            log_dir=output.parent,
            component="rig-generation",
        )
        if not output.is_file():
            raise AvatarAdapterError("rig_missing", "rig generator completed without a GLB", retryable=True)
        if not manifest.is_file():
            raise AvatarAdapterError("manifest_missing", "rig generator completed without a humanoid manifest", retryable=True)
        return output, manifest


class CommandReskinGenerator:
    """Run SkinTokens against a mesh that already has a canonical skeleton."""

    def __init__(self, command: str | Sequence[str] | None, *, timeout_seconds: float, cwd: Path | None = None):
        self.command = parse_command(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def generate(
        self,
        *,
        skeleton: Path,
        output: Path,
        manifest: Path,
        profile: Path,
        quality: str,
    ) -> tuple[Path, Path]:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        run_adapter_command(
            self.command,
            values={
                "input": skeleton,
                "skeleton": skeleton,
                "output": output,
                "manifest_output": manifest,
                "profile": profile,
                "quality": quality,
            },
            cwd=self.cwd,
            timeout_seconds=self.timeout_seconds,
            log_dir=output.parent,
            component="avatar-reskin",
        )
        if not output.is_file():
            raise AvatarAdapterError("reskin_missing", "reskin generator completed without a GLB", retryable=True)
        if not manifest.is_file():
            raise AvatarAdapterError("reskin_manifest_missing", "reskin generator completed without a canonical manifest", retryable=True)
        return output, manifest

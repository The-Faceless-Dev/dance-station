"""Optional video enhancement boundaries for the VACE worker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command
from autotransition.generative_dance.video import probe_video

from .config import VaceStitchConfig


@dataclass(frozen=True)
class VideoStageResult:
    stage: str
    output_video: Path
    enabled: bool
    backend: str | None
    metadata_path: Path | None
    probe: object | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "outputVideo": str(self.output_video),
            "enabled": self.enabled,
            "backend": self.backend,
            "metadataPath": str(self.metadata_path) if self.metadata_path else None,
            "probe": self.probe.to_dict() if hasattr(self.probe, "to_dict") else None,
        }


class VaceVideoStage:
    """Run an optional external enhancer or interpolator behind one contract."""

    def __init__(self, config: VaceStitchConfig, *, stage: str):
        if stage not in {"enhancement", "motion-interpolation"}:
            raise ValueError(f"unsupported VACE video stage: {stage}")
        self.config = config
        self.stage = stage
        if stage == "enhancement":
            self.enabled = config.enhancement_enabled
            self.backend = config.enhancement_backend
            self.command = parse_command(config.enhancement_command)
            self.cwd = config.enhancement_cwd
        else:
            self.enabled = config.motion_interpolation_enabled
            self.backend = config.motion_interpolation_backend
            self.command = parse_command(config.motion_interpolation_command)
            self.cwd = config.motion_interpolation_cwd

    def process(self, *, input_video: Path, output_dir: Path, width: int, height: int, fps: int) -> VideoStageResult:
        if not self.enabled:
            return VideoStageResult(self.stage, input_video, False, None, None, None)
        if not self.command:
            raise AvatarAdapterError(
                f"vace_{self.stage.replace('-', '_')}_not_configured",
                f"VACE {self.stage} is enabled but no command is configured",
                retryable=False,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{self.stage}.mp4"
        metadata_path = output_dir / f"{self.stage}.json"
        target_fps = (
            self.config.enhancement_target_fps
            if self.stage == "enhancement"
            else self.config.motion_interpolation_target_fps
        )
        scale = self.config.enhancement_scale if self.stage == "enhancement" else 1.0
        tile_size = self.config.enhancement_tile_size if self.stage == "enhancement" else 0
        fp16 = self.config.enhancement_fp16 if self.stage == "enhancement" else False
        # Real-ESRGAN's stage CLI accepts an integer model scale. Keep the
        # public config numeric, but do not render a JSON/env float such as
        # 2.0 into argv as "2.0".
        command_scale = int(scale) if self.stage == "enhancement" else 1
        run_adapter_command(
            self.command,
            values={
                "input": input_video,
                "output": output,
                "output_dir": output_dir,
                "width": width,
                "height": height,
                "fps": fps,
                "target_fps": target_fps or fps,
                "scale": command_scale,
                "tile_size": tile_size,
                "fp16": "true" if fp16 else "false",
            },
            cwd=self.cwd,
            timeout_seconds=self.config.job_timeout_seconds,
            log_dir=output_dir,
            component=f"vace-{self.stage}",
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise AvatarAdapterError(
                f"vace_{self.stage.replace('-', '_')}_output_missing",
                f"VACE {self.stage} completed without an output video",
                retryable=True,
                details={"outputDir": str(output)},
            )
        probe = probe_video(output)
        metadata = {
            "stage": self.stage,
            "backend": self.backend,
            "input": str(input_video),
            "output": str(output),
            "requested": {
                "width": width,
                "height": height,
                "fps": fps,
                "targetFps": target_fps or fps,
                "scale": scale,
                "tileSize": tile_size,
                "fp16": fp16,
            },
            "probe": probe.to_dict(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return VideoStageResult(self.stage, output, True, self.backend, metadata_path, probe)

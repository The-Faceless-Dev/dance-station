"""BiRefNet video-matting boundary for generated dance clips."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.video import VideoProbe, probe_video


@dataclass(frozen=True)
class MatteResult:
    input_video: Path
    output_video: Path
    probe: VideoProbe
    metadata_path: Path
    model: str
    backend: str


class BiRefNetMattingAdapter:
    """Run BiRefNet once over every frame through a stable project boundary.

    The default native command points at the project-owned runner, while a
    configured command remains available for a separately packaged CUDA,
    TensorRT, or ONNX runtime. Both paths must write an alpha-capable video.
    """

    def __init__(self, config: GenerativeDanceConfig):
        self.config = config
        self.command = parse_command(config.matte_command)
        self.native_command = self._build_native_command()

    def _build_native_command(self) -> tuple[str, ...]:
        if self.config.matte_backend != "native":
            return ()
        repo_root = Path(__file__).resolve().parents[3]
        runner = repo_root / "tools" / "generative_dance" / "birefnet_video.py"
        command = [
            self.config.matte_python or sys.executable,
            str(runner),
            "--input",
            "{input_video}",
            "--output",
            "{output_video}",
            "--raw-output",
            "{raw_output}",
            "--model",
            "{matte_model}",
            "--device",
            "{matte_device}",
            "--dtype",
            "{matte_dtype}",
            "--batch-size",
            "{matte_batch_size}",
            "--input-size",
            "{matte_input_size}",
            "--threshold",
            "{matte_alpha_threshold}",
            "--edge-feather",
            "{matte_edge_feather}",
            "--despill-strength",
            "{matte_despill_strength}",
            "--background-key-distance",
            "{matte_background_key_distance}",
            "--background-key-strength",
            "{matte_background_key_strength}",
            "--crf",
            "{transparent_crf}",
        ]
        if self.config.matte_checkpoint:
            command.extend(("--checkpoint", "{matte_checkpoint}"))
        return tuple(command)

    @property
    def configured(self) -> bool:
        if self.command:
            return True
        return bool(self.native_command and (self.config.matte_checkpoint or self.config.matte_model))

    def process(self, *, input_video: Path, output_dir: Path) -> MatteResult:
        command = self.command or self.native_command
        if not command:
            raise AvatarAdapterError(
                "birefnet_not_configured",
                "BiRefNet matting is not configured; set GENERATIVE_DANCE_MATTE_BACKEND=native or GENERATIVE_DANCE_MATTE_COMMAND",
                retryable=False,
            )
        if not input_video.is_file():
            raise AvatarAdapterError(
                "birefnet_input_missing",
                f"BiRefNet input video was not found: {input_video}",
                retryable=False,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "birefnet-alpha.mov"
        input_probe = probe_video(input_video)
        metadata_path = output_dir / "birefnet.json"
        run_adapter_command(
            command,
            values={
                "input_video": input_video,
                "output_video": output_video,
                "raw_output": output_dir / "birefnet-alpha.rgba",
                "output_dir": output_dir,
                "matte_model": self.config.matte_model,
                "matte_checkpoint": self.config.matte_checkpoint or "",
                "matte_device": self.config.matte_device,
                "matte_dtype": self.config.matte_compute_dtype,
                "matte_batch_size": self.config.matte_batch_size,
                "matte_input_size": self.config.matte_input_size,
                "matte_alpha_threshold": self.config.matte_alpha_threshold,
                "matte_edge_feather": self.config.matte_edge_feather,
                "matte_despill_strength": self.config.matte_despill_strength,
                "matte_background_key_distance": self.config.matte_background_key_distance,
                "matte_background_key_strength": self.config.matte_background_key_strength,
                "transparent_codec": self.config.transparent_codec,
                "transparent_crf": self.config.transparent_crf,
            },
            cwd=self.config.matte_cwd,
            timeout_seconds=self.config.job_timeout_seconds,
            log_dir=output_dir,
            component="birefnet-matting",
        )
        if not output_video.is_file() or output_video.stat().st_size == 0:
            raise AvatarAdapterError(
                "birefnet_output_missing",
                "BiRefNet completed without an alpha video output",
                retryable=True,
                details={"outputDir": str(output_dir)},
            )
        try:
            output_probe = probe_video(output_video)
        except Exception as exc:
            raise AvatarAdapterError("birefnet_output_invalid", str(exc), retryable=True) from exc
        if output_probe.width != input_probe.width or output_probe.height != input_probe.height:
            raise AvatarAdapterError(
                "birefnet_dimensions_mismatch",
                "BiRefNet changed the generated video's dimensions",
                retryable=False,
                details={"input": input_probe.to_dict(), "output": output_probe.to_dict()},
            )
        if input_probe.frame_count is not None and output_probe.frame_count is not None:
            if input_probe.frame_count != output_probe.frame_count:
                raise AvatarAdapterError(
                    "birefnet_frame_count_mismatch",
                    "BiRefNet output does not contain one alpha frame for every RGB frame",
                    retryable=True,
                    details={"inputFrames": input_probe.frame_count, "outputFrames": output_probe.frame_count},
                )
        if input_probe.fps > 0 and output_probe.fps > 0:
            fps_tolerance = max(0.1, input_probe.fps * 0.02)
            if abs(input_probe.fps - output_probe.fps) > fps_tolerance:
                raise AvatarAdapterError(
                    "birefnet_fps_mismatch",
                    "BiRefNet output changed the source frame rate",
                    retryable=True,
                    details={"inputFps": input_probe.fps, "outputFps": output_probe.fps},
                )
        if input_probe.duration_seconds > 0 and output_probe.duration_seconds > 0:
            duration_tolerance = max(0.1, 2.0 / max(input_probe.fps, 1.0))
            if abs(input_probe.duration_seconds - output_probe.duration_seconds) > duration_tolerance:
                raise AvatarAdapterError(
                    "birefnet_duration_mismatch",
                    "BiRefNet output changed the source duration",
                    retryable=True,
                    details={
                        "inputDurationSeconds": input_probe.duration_seconds,
                        "outputDurationSeconds": output_probe.duration_seconds,
                    },
                )
        if not output_probe.has_alpha:
            raise AvatarAdapterError(
                "birefnet_alpha_missing",
                "BiRefNet output is not alpha-capable; expected an RGBA/yuva pixel format",
                retryable=False,
                details={"probe": output_probe.to_dict()},
            )
        payload = {
            "inputVideo": str(input_video),
            "outputVideo": str(output_video),
            "model": self.config.matte_model,
            "backend": "native" if not self.command and self.native_command else "command",
            "device": self.config.matte_device,
            "dtype": self.config.matte_compute_dtype,
            "batchSize": self.config.matte_batch_size,
            "alphaThreshold": self.config.matte_alpha_threshold,
            "edgeFeather": self.config.matte_edge_feather,
            "inputProbe": input_probe.to_dict(),
            "outputProbe": output_probe.to_dict(),
            "framesProcessed": output_probe.frame_count,
            "alphaValidated": True,
        }
        metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return MatteResult(
            input_video=input_video,
            output_video=output_video,
            probe=output_probe,
            metadata_path=metadata_path,
            model=self.config.matte_model,
            backend="native" if not self.command and self.native_command else "command",
        )

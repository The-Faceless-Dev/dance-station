"""External command adapter for the pinned Wan Animate 2 Lite runtime."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import AvatarReference, DanceDriver, RenderedSegment
from autotransition.generative_dance.video import probe_video


class WanAnimate2LiteAdapter:
    """Keep Wan's changing repository CLI behind one stable project boundary.

    The command template receives the placeholders documented in
    ``GenerativeDanceConfig``. The official Wan repository currently exposes
    reference-image and driving-video inputs; the adapter intentionally does
    not import its heavyweight runtime into the local web process.
    """

    def __init__(self, config: GenerativeDanceConfig):
        self.config = config
        self.command = parse_command(config.wan_command)
        self.native_command = self._build_native_command()

    def _build_native_command(self) -> tuple[str, ...]:
        if self.config.wan_backend != "native":
            return ()
        required = (
            self.config.wan_transformer_checkpoint,
            self.config.wan_official_source,
            self.config.wan_t5_checkpoint,
            self.config.wan_t5_tokenizer,
            self.config.wan_clip_checkpoint,
            self.config.wan_clip_tokenizer,
            self.config.wan_vae_checkpoint,
        )
        if not all(required):
            return ()
        repo_root = Path(__file__).resolve().parents[3]
        runner = repo_root / "tools" / "generative_dance" / "wan_animate_2_runner.py"
        return (
            self.config.wan_python or sys.executable,
            str(runner),
            "--model", "{wan_transformer}",
            "--official-source", "{wan_official_source}",
            "--t5", "{wan_t5_checkpoint}",
            "--t5-tokenizer", "{wan_t5_tokenizer}",
            "--clip", "{wan_clip_checkpoint}",
            "--clip-tokenizer", "{wan_clip_tokenizer}",
            "--vae", "{wan_vae_checkpoint}",
            "--device", "{wan_device}",
            "--dtype", "{wan_dtype}",
            "--reference-image", "{reference_image}",
            "--driver-video", "{driver_video}",
            "--output", "{output}",
            "--prompt", "{prompt}",
            "--width", "{width}",
            "--height", "{height}",
            "--fps", "{fps}",
            "--full-driver",
            "--max-clip-len", "{temporal_window}",
            "--temporal-context-frames", "{temporal_context_frames}",
            "--steps", "{inference_steps}",
            "--text-length", "{text_length}",
            "--reference-strength", "{reference_strength}",
            "--seed", "{seed}",
        )

    @property
    def configured(self) -> bool:
        return bool(self.command or self.native_command)

    def render(
        self,
        *,
        segment_id: str,
        reference: AvatarReference,
        driver: DanceDriver,
        output_dir: Path,
        prompt: str | None = None,
        seed: int | None = None,
        inference_steps: int | None = None,
        text_length: int | None = None,
        reference_strength: float | None = None,
        continuation_frames: Path | None = None,
        continuation_frame: Path | None = None,
    ) -> RenderedSegment:
        command = self.command or self.native_command
        effective_continuation = continuation_frames or continuation_frame
        if not self.command and effective_continuation is not None:
            command = (*command, "--continuation-frames", "{continuation_frames}")
        if not command:
            raise AvatarAdapterError(
                "wan_animate_not_configured",
                "Wan Animate 2 is not configured; set the native runtime paths or GENERATIVE_DANCE_WAN_COMMAND",
                retryable=False,
            )
        effective_steps = (
            inference_steps
            if inference_steps is not None
            else self.config.wan_inference_steps
        )
        effective_reference_strength = (
            reference_strength
            if reference_strength is not None
            else self.config.wan_reference_strength
        )
        if not 0 < effective_reference_strength <= 5:
            raise AvatarAdapterError(
                "wan_invalid_reference_strength",
                "Wan-Animate-2 reference strength must be greater than 0 and at most 5",
                retryable=False,
                details={"referenceStrength": effective_reference_strength},
            )
        if effective_steps < self.config.wan_min_inference_steps:
            raise AvatarAdapterError(
                "wan_invalid_inference_steps",
                (
                    "The Wan-Animate-2 distilled worker requires at least "
                    f"{self.config.wan_min_inference_steps} inference steps; "
                    f"received {effective_steps}."
                ),
                retryable=False,
                details={
                    "requestedSteps": effective_steps,
                    "minimumSteps": self.config.wan_min_inference_steps,
                    "modelRevision": self.config.wan_model_revision,
                },
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "render.mp4"
        prompt_value = prompt or reference.prompt
        prompt_file = output_dir / "prompt.txt"
        prompt_file.write_text(prompt_value, encoding="utf-8")
        try:
            run_adapter_command(
                command,
                values={
                    "reference_image": reference.normalized_image,
                    "driver_video": driver.normalized_video,
                    "output": output,
                    "output_dir": output_dir,
                    "prompt": prompt_value,
                    "prompt_file": prompt_file,
                    "model_revision": self.config.wan_model_revision,
                    "config_file": self.config.wan_config_file or "",
                    "width": self.config.canvas.width,
                    "height": self.config.canvas.height,
                    # The worker preserves the normalized driver's source
                    # rate. The local editor may still use its configured
                    # canvas rate when it explicitly normalizes a driver.
                    # The native runner's CLI accepts an integer FPS. Video
                    # probes may expose an equivalent rate as a float (for
                    # example, 20.0), so normalize it at this boundary.
                    "fps": int(round(driver.canvas.fps)),
                    # The native runner requires an integer argument; use its
                    # deterministic default when the request leaves seed unset.
                    "seed": seed if seed is not None else 0,
                    "inference_steps": effective_steps,
                    "text_length": text_length if text_length is not None else self.config.wan_text_length,
                    "reference_strength": effective_reference_strength,
                    "continuation_frame": continuation_frame or "",
                    "continuation_frames": effective_continuation or "",
                    "temporal_window": self.config.wan_temporal_window,
                    "temporal_context_frames": self.config.wan_temporal_context_frames,
                    "guidance_scale": self.config.wan_guidance_scale,
                    "wan_transformer": self.config.wan_transformer_checkpoint or "",
                    "wan_official_source": self.config.wan_official_source or "",
                    "wan_t5_checkpoint": self.config.wan_t5_checkpoint or "",
                    "wan_t5_tokenizer": self.config.wan_t5_tokenizer or "",
                    "wan_clip_checkpoint": self.config.wan_clip_checkpoint or "",
                    "wan_clip_tokenizer": self.config.wan_clip_tokenizer or "",
                    "wan_vae_checkpoint": self.config.wan_vae_checkpoint or "",
                    "wan_device": self.config.wan_device,
                    "wan_dtype": self.config.wan_compute_dtype,
                },
                cwd=self.config.wan_cwd,
                timeout_seconds=self.config.job_timeout_seconds,
                log_dir=output_dir,
                component="wan-animate-2-lite",
            )
        finally:
            prompt_file.unlink(missing_ok=True)
        if not output.is_file():
            candidates = sorted(
                (path for path in output_dir.iterdir() if path.suffix.lower() in {".mp4", ".webm", ".mov"} and path.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                shutil.copy2(candidates[0], output)
        if not output.is_file() or output.stat().st_size == 0:
            raise AvatarAdapterError(
                "wan_animate_output_missing",
                "Wan Animate 2 Lite completed without a render.mp4 output",
                retryable=True,
                details={"outputDir": str(output_dir)},
            )
        try:
            probe = probe_video(output)
        except Exception as exc:
            raise AvatarAdapterError("wan_animate_output_invalid", str(exc), retryable=True) from exc
        metadata_path = output_dir / "render.json"
        runtime_metadata: dict[str, object] = {}
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    runtime_metadata = loaded
            except (OSError, ValueError):
                # The runtime log remains available beside the output. The
                # adapter still writes its own validated segment metadata.
                runtime_metadata = {}
        metadata_path.write_text(
            json.dumps(
                {
                    **runtime_metadata,
                    "segmentId": segment_id,
                    "referenceId": reference.id,
                    "driverId": driver.id,
                    "modelRevision": self.config.wan_model_revision,
                    "backend": "native" if not self.command and self.native_command else "command",
                    "prompt": prompt_value,
                    "seed": seed if seed is not None else 0,
                    "referenceStrength": effective_reference_strength,
                    "probe": probe.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return RenderedSegment(
            id=segment_id,
            driver_id=driver.id,
            reference_id=reference.id,
            output_video=output,
            duration_seconds=probe.duration_seconds,
            canvas=self.config.canvas,
            metadata_path=metadata_path,
            model_revision=self.config.wan_model_revision,
            prompt=prompt_value,
            runtime_metadata=runtime_metadata,
        )

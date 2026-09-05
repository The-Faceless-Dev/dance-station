"""Stable command boundary around the official Wan2.1 VACE runtime."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command
from autotransition.generative_dance.video import probe_video

from .config import VaceStitchConfig


class VaceRuntime:
    """Run one VACE bridge while keeping the heavy runtime out of the API process."""

    def __init__(self, config: VaceStitchConfig):
        self.config = config
        self.command = parse_command(config.runtime_command)
        self.native_command = self._build_native_command()
        self.lightx2v_command = self._build_lightx2v_command()
        self._checkpoint_report: dict[str, object] | None = None

    def _validate_checkpoint(self) -> dict[str, object] | None:
        if self._checkpoint_report is not None:
            return self._checkpoint_report
        checkpoint = self.config.checkpoint_file
        if checkpoint is None:
            return None
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise AvatarAdapterError(
                "vace_checkpoint_missing",
                "Wan2.1 VACE checkpoint file is missing or empty",
                retryable=False,
                details={"checkpoint": str(checkpoint)},
            )
        try:
            from safetensors import safe_open

            with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:
            raise AvatarAdapterError(
                "vace_checkpoint_invalid",
                f"Wan2.1 VACE checkpoint header could not be read: {exc}",
                retryable=False,
                details={"checkpoint": str(checkpoint)},
            ) from exc
        scaled = "scaled_fp8" in keys and any(key.endswith(".scale_weight") for key in keys)
        self._checkpoint_report = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "tensorCount": len(keys),
            "scaledFp8": scaled,
        }
        if not scaled:
            raise AvatarAdapterError(
                "vace_checkpoint_format",
                "Wan2.1 VACE checkpoint is not the expected scaled-FP8 format",
                retryable=False,
                details=self._checkpoint_report,
            )
        return self._checkpoint_report

    def _build_native_command(self) -> tuple[str, ...]:
        if self.config.runtime_backend != "native" or not self.config.source_root or not self.config.checkpoint_dir:
            return ()
        script = self.config.source_root / "vace" / "vace_wan_inference.py"
        if not script.is_file():
            return ()
        command: list[str] = [
            self.config.python_executable or sys.executable,
            "-m",
            "autotransition.vace_stitch.native_runner",
            "--script",
            str(script),
            "--model_name",
            "{model_name}",
            "--size",
            "{model_size}",
            "--frame_num",
            "{frame_num}",
            "--ckpt_dir",
            "{checkpoint_dir}",
            "--offload_model",
            "{offload_model}",
            "--save_file",
            "{output}",
            "--save_dir",
            "{output_dir}",
            "--src_video",
            "{source_video}",
            "--src_mask",
            "{source_mask}",
            "--prompt",
            "{prompt}",
            "--use_prompt_extend",
            "plain",
            "--base_seed",
            "{seed}",
            "--sample_solver",
            "{sample_solver}",
            "--sample_steps",
            "{sample_steps}",
            "--sample_shift",
            "{sample_shift}",
            "--sample_guide_scale",
            "{guide_scale}",
        ]
        if self.config.t5_cpu:
            command.append("--t5_cpu")
        return tuple(command)

    def _build_lightx2v_command(self) -> tuple[str, ...]:
        if self.config.runtime_backend != "lightx2v":
            return ()
        if not self.config.lightx2v_source_root or not self.config.lightx2v_config:
            return ()
        return (
            self.config.python_executable or sys.executable,
            "-m",
            "autotransition.vace_stitch.lightx2v_runner",
            "--source-root",
            str(self.config.lightx2v_source_root),
            "--config-json",
            "{lightx2v_config}",
            "--model-path",
            str(self.config.checkpoint_dir or ""),
            "--frame-num",
            "{frame_num}",
            "--src-video",
            "{source_video}",
            "--src-mask",
            "{source_mask}",
            "--prompt",
            "{prompt}",
            "--seed",
            "{seed}",
            "--save-result-path",
            "{output}",
        )

    @property
    def configured(self) -> bool:
        return bool(self.command or self.native_command or self.lightx2v_command)

    def generate(
        self,
        *,
        source_video: Path,
        source_mask: Path,
        output_dir: Path,
        prompt: str,
        frame_num: int,
        seed: int,
        sample_steps: int | None = None,
        sample_shift: float | None = None,
        guide_scale: float | None = None,
        model_name: str | None = None,
        model_size: str | None = None,
    ) -> Path:
        command = self.command or self.native_command or self.lightx2v_command
        if not command:
            raise AvatarAdapterError(
                "vace_runtime_not_configured",
                "Wan2.1 VACE is not configured; set VACE_STITCH_BACKEND/native paths or VACE_STITCH_COMMAND",
                retryable=False,
            )
        if not source_video.is_file() or not source_mask.is_file():
            raise AvatarAdapterError(
                "vace_input_missing",
                "VACE source video or source mask is missing",
                retryable=False,
                details={"sourceVideo": str(source_video), "sourceMask": str(source_mask)},
            )
        checkpoint_report = self._validate_checkpoint()
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "vace-output.mp4"
        prompt_file = output_dir / "prompt.txt"
        lightx2v_config_path = output_dir / "lightx2v-config.json"
        prompt_file.write_text(prompt, encoding="utf-8")
        if self.config.runtime_backend == "lightx2v":
            if not self.config.lightx2v_config or not self.config.lightx2v_config.is_file():
                raise AvatarAdapterError(
                    "vace_lightx2v_config_missing",
                    "LightX2V VACE config is missing",
                    retryable=False,
                    details={"config": str(self.config.lightx2v_config)},
                )
            payload = json.loads(self.config.lightx2v_config.read_text(encoding="utf-8"))
            payload.update(
                {
                    "target_video_length": frame_num,
                    "infer_steps": self.config.lightx2v_steps,
                    "sample_shift": sample_shift if sample_shift is not None else 5.0,
                    "sample_guide_scale": guide_scale if guide_scale is not None else 1.0,
                    "lora_configs": (
                        [{"path": str(self.config.lightx2v_lora), "strength": self.config.lightx2v_lora_strength}]
                        if self.config.lightx2v_lora
                        else []
                    ),
                    "self_attn_1_type": self.config.lightx2v_attention_backend,
                    "cross_attn_1_type": self.config.lightx2v_attention_backend,
                    "cross_attn_2_type": self.config.lightx2v_attention_backend,
                }
            )
            if self.config.lightx2v_t5_checkpoint:
                payload["t5_original_ckpt"] = str(self.config.lightx2v_t5_checkpoint)
            if self.config.lightx2v_t5_tokenizer:
                payload["t5_tokenizer_path"] = str(self.config.lightx2v_t5_tokenizer)
            if self.config.lightx2v_vae:
                payload["vae_path"] = str(self.config.lightx2v_vae)
            lightx2v_config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            run_adapter_command(
                command,
                values={
                    "source_video": source_video,
                    "source_mask": source_mask,
                    "output": output,
                    "output_dir": output_dir,
                    "prompt": prompt,
                    "prompt_file": prompt_file,
                    "frame_num": frame_num,
                    "seed": seed,
                    "sample_steps": sample_steps if sample_steps is not None else self.config.sample_steps,
                    "sample_shift": sample_shift if sample_shift is not None else self.config.sample_shift,
                    "guide_scale": guide_scale if guide_scale is not None else self.config.guide_scale,
                    "model_name": model_name or self.config.model_name,
                    "model_size": model_size or self.config.model_size,
                    "checkpoint_dir": self.config.checkpoint_dir or "",
                    "offload_model": "true" if self.config.offload_model else "false",
                    "context_scale": self.config.context_scale,
                    "sample_solver": self.config.sample_solver,
                    "lightx2v_config": lightx2v_config_path,
                },
                cwd=self.config.runtime_cwd,
                timeout_seconds=self.config.runtime_timeout_seconds,
                log_dir=output_dir,
                component="wan-vace-stitch",
            )
        finally:
            prompt_file.unlink(missing_ok=True)
            lightx2v_config_path.unlink(missing_ok=True)
        if not output.is_file():
            candidates = sorted(
                (
                    path
                    for path in output_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                shutil.copy2(candidates[0], output)
        if not output.is_file() or output.stat().st_size == 0:
            raise AvatarAdapterError(
                "vace_output_missing",
                "VACE completed without a usable video output",
                retryable=True,
                details={"outputDir": str(output)},
            )
        try:
            probe = probe_video(output)
        except Exception as exc:
            raise AvatarAdapterError(
                "vace_output_invalid",
                f"VACE output could not be decoded: {exc}",
                retryable=True,
            ) from exc
        metadata = {
            "runtime": "wan-vace-stitch",
            "backend": self.config.runtime_backend,
            "modelName": model_name or self.config.model_name,
            "modelSize": model_size or self.config.model_size,
            "prompt": prompt,
            "frameNum": frame_num,
            "seed": seed,
            "sampleSteps": sample_steps if sample_steps is not None else self.config.sample_steps,
            "sampleShift": sample_shift if sample_shift is not None else self.config.sample_shift,
            "guideScale": guide_scale if guide_scale is not None else self.config.guide_scale,
            "offloadModel": self.config.offload_model,
            "attentionBackend": self.config.attention_backend,
            "tf32": self.config.tf32,
            "lightx2v": {
                "enabled": self.config.runtime_backend == "lightx2v",
                "sourceRoot": str(self.config.lightx2v_source_root) if self.config.lightx2v_source_root else None,
                "steps": self.config.lightx2v_steps,
                "lora": str(self.config.lightx2v_lora) if self.config.lightx2v_lora else None,
                "loraStrength": self.config.lightx2v_lora_strength,
                "attentionBackend": self.config.lightx2v_attention_backend,
                "cpuOffload": self.config.offload_model,
                "t5CpuOffload": self.config.t5_cpu,
            },
            "checkpoint": checkpoint_report,
            "sourceVideo": str(source_video),
            "sourceMask": str(source_mask),
            "probe": probe.to_dict(),
        }
        (output_dir / "vace-output.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return output

from __future__ import annotations

import json
import math
import queue
import re
import secrets
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Yue2Config
from .contracts import Yue2Request


class Yue2RuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Yue2RuntimeResult:
    output_path: Path
    metadata: dict[str, Any]
    command: list[str]


class Yue2Runtime:
    """Native audio.cpp YuE2 runner; one process per request for isolation."""

    def __init__(self, config: Yue2Config):
        self.config = config

    def preflight(self) -> dict[str, Any]:
        return self.config.preflight()

    def resolve_request(self, request: Yue2Request) -> dict[str, Any]:
        request_options = dict(request.request_options)
        if request.max_tokens is not None and "semantic_max_tokens" not in request_options:
            request_options["semantic_max_tokens"] = request.max_tokens
            request_options.setdefault("semantic_min_tokens", min(200, request.max_tokens))
        if request.guidance_scale is not None and "cfg_scale" not in request_options:
            request_options["cfg_scale"] = request.guidance_scale
        if request.temperature is not None and "semantic_temperature" not in request_options:
            request_options["semantic_temperature"] = request.temperature
        if request.duration_seconds is not None and "semantic_max_tokens" not in request_options:
            max_tokens = max(1, math.ceil(request.duration_seconds * self.config.semantic_tokens_per_second))
            request_options["semantic_max_tokens"] = max_tokens
            request_options.setdefault("semantic_min_tokens", min(200, max_tokens))
        return {
            "style": request.style or self.config.default_style,
            "cot": request.cot or self.config.default_cot,
            "num_inference_steps": request.num_inference_steps or self.config.default_inference_steps,
            "seed": request.seed if request.seed is not None else secrets.randbelow(2**31 - 1),
            "duration_seconds": request.duration_seconds,
            "guidance_scale": request.guidance_scale,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "request_options": request_options,
        }

    def build_command(self, request: Yue2Request, output_path: Path) -> tuple[list[str], dict[str, Any]]:
        resolved = self.resolve_request(request)
        cli = self.config.resolve_cli()
        if not cli:
            raise Yue2RuntimeError(f"audio.cpp CLI was not found: {self.config.cli_path}")
        command = [
            cli,
            "--task", "gen",
            "--family", "yue2",
            "--model", str(self.config.model_root.resolve()),
            "--backend", self.config.backend,
            "--device", str(self.config.device),
            "--threads", str(self.config.threads),
            "--lyrics", request.lyrics,
            "--request-option", f"style={resolved['style']}",
            "--request-option", f"cot={resolved['cot']}",
            "--num-inference-steps", str(resolved["num_inference_steps"]),
            "--seed", str(resolved["seed"]),
            "--session-option", f"yue2.model_gguf={self.config.model_file}",
            "--session-option", f"yue2.vae_gguf={self.config.vae_file}",
            "--out", str(output_path),
            "--log",
            "--metrics",
        ]
        reserved = {
            "style",
            "cot",
            "lyrics",
            "num_inference_steps",
            "seed",
            "duration_seconds",
            "guidance_scale",
            "cfg_scale",
            "temperature",
            "semantic_temperature",
            "max_tokens",
            "semantic_max_tokens",
            "semantic_min_tokens",
        }
        for key in ("semantic_max_tokens", "semantic_min_tokens", "cfg_scale", "semantic_temperature"):
            if key in resolved["request_options"]:
                command.extend(["--request-option", f"{key}={resolved['request_options'][key]}"])
        for key, value in resolved["request_options"].items():
            if str(key) in reserved:
                continue
            command.extend(["--request-option", f"{key}={value}"])
        return command, resolved

    def generate(
        self,
        request: Yue2Request,
        attempt_dir: Path,
        progress: Callable[[float, str], None],
    ) -> Yue2RuntimeResult:
        report = self.preflight()
        if not report.get("ready"):
            raise Yue2RuntimeError(f"YuE2 preflight failed: {json.dumps(report, sort_keys=True)}")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        output_path = attempt_dir / "audio.wav"
        command, resolved = self.build_command(request, output_path)
        (attempt_dir / "command.json").write_text(
            json.dumps({"command": command, "resolved": resolved, "preflight": report}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log_path = attempt_dir / "audiocpp.log"
        started = time.monotonic()
        last_progress = 0.0

        def emit_progress(fraction: float, message: str) -> None:
            nonlocal last_progress
            last_progress = max(last_progress, min(1.0, float(fraction)))
            progress(last_progress, message)

        emit_progress(0.02, "Starting native audio.cpp YuE2 inference")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def drain() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=drain, name="yue2-audiocpp-log", daemon=True)
        reader.start()
        step_pattern = re.compile(r"(?:step|steps|denoise)\s*[:=]?\s*(\d+)\s*(?:/|of)\s*(\d+)", re.IGNORECASE)
        percent_pattern = re.compile(r"(?<!\d)(\d{1,3})%")
        captured: list[str] = []
        timed_out = False
        phase_progress = (
            (re.compile(r"yue2\.plan", re.IGNORECASE), 0.12, "YuE2 planning"),
            (re.compile(r"yue2\.ar\.", re.IGNORECASE), 0.34, "YuE2 semantic autoregressive pass"),
            (re.compile(r"yue2\.semantic", re.IGNORECASE), 0.50, "YuE2 semantic tokens"),
            (re.compile(r"yue2\.nar\.", re.IGNORECASE), 0.70, "YuE2 acoustic refinement"),
            (re.compile(r"yue2\.vae|oobleck_audio_vae", re.IGNORECASE), 0.88, "YuE2 audio VAE decode"),
        )
        try:
            with log_path.open("w", encoding="utf-8") as log:
                while True:
                    try:
                        line = lines.get(timeout=0.5)
                    except queue.Empty:
                        if time.monotonic() - started > self.config.request_timeout_seconds:
                            timed_out = True
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            break
                        continue
                    if line is None:
                        break
                    log.write(line)
                    log.flush()
                    captured.append(line)
                    step = step_pattern.search(line)
                    percent = percent_pattern.search(line)
                    matched_phase = False
                    if step:
                        current, total = (int(step.group(1)), max(1, int(step.group(2))))
                        emit_progress(min(0.94, 0.08 + 0.84 * current / total), f"YuE2 inference step {current}/{total}")
                    elif percent:
                        emit_progress(min(0.94, 0.08 + 0.84 * int(percent.group(1)) / 100), f"YuE2 runtime: {line.strip()[-240:]}")
                    else:
                        for pattern, fraction, message in phase_progress:
                            if pattern.search(line):
                                matched_phase = True
                                emit_progress(fraction, message)
                                break
                    if not step and not percent and not matched_phase and (len(captured) == 1 or len(captured) % 25 == 0):
                        emit_progress(0.08, f"YuE2 runtime: {line.strip()[-240:]}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
        reader.join(timeout=2)
        return_code = process.returncode if process.returncode is not None else process.wait(timeout=10)
        wall_seconds = time.monotonic() - started
        output_text = "".join(captured)
        if timed_out:
            raise Yue2RuntimeError(f"audio.cpp YuE2 inference timed out after {wall_seconds:.1f}s")
        if return_code != 0:
            tail = output_text[-4000:].strip()
            raise Yue2RuntimeError(f"audio.cpp YuE2 exited with code {return_code}: {tail}")
        if not output_path.is_file() or output_path.stat().st_size < 128:
            raise Yue2RuntimeError("audio.cpp completed without a non-empty WAV output")
        try:
            with wave.open(str(output_path), "rb") as handle:
                wave_info = {
                    "sampleRate": handle.getframerate(),
                    "channels": handle.getnchannels(),
                    "sampleWidthBytes": handle.getsampwidth(),
                    "frames": handle.getnframes(),
                    "durationSeconds": handle.getnframes() / max(1, handle.getframerate()),
                }
        except (wave.Error, OSError) as exc:
            raise Yue2RuntimeError(f"audio.cpp output is not a readable WAV: {exc}") from exc
        metadata = {
            "runtime": "yue2-audio-cpp",
            "backend": self.config.backend,
            "device": self.config.device,
            "model": self.config.model_file,
            "vae": self.config.vae_file,
            "audioCppRevision": self.config.audio_cpp_revision,
            "resolved": resolved,
            "returnCode": return_code,
            "wallSeconds": wall_seconds,
            "outputBytes": output_path.stat().st_size,
            "wave": wave_info,
            "logFile": log_path.name,
        }
        (attempt_dir / "runtime-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        emit_progress(0.97, "YuE2 audio generated and validated")
        return Yue2RuntimeResult(output_path=output_path, metadata=metadata, command=command)

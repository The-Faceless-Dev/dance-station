from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import QwenImageConfig
from .contracts import QwenImageRequest


ProgressCallback = Callable[[str, float, str], None]


class QwenImageRuntime:
    """Persistent stable-diffusion.cpp server wrapper for Qwen-Image GGUF."""

    def __init__(self, config: QwenImageConfig, *, emit: Callable[..., Any] | None = None):
        self.config = config
        self.emit = emit or (lambda *_args, **_kwargs: None)
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://{self.config.runtime_host}:{self.config.runtime_port}"

    def preflight(self) -> dict[str, Any]:
        return self.config.preflight()

    def command(self) -> list[str]:
        command = [
            str(self.config.runtime_binary),
            "--diffusion-model", str(self.config.diffusion_model),
            "--llm", str(self.config.text_encoder),
            "--vae", str(self.config.vae),
            "--listen-ip", self.config.runtime_host,
            "--listen-port", str(self.config.runtime_port),
            "--backend", self.config.device_backend,
            "--lora-model-dir", str(self.config.lora_root),
            "--lora-apply-mode", self.config.lora_apply_mode,
            "--log-level", "verbose",
        ]
        if self.config.cpu_offload:
            command.append("--offload-to-cpu")
        if self.config.flash_attention:
            command.append("--diffusion-fa")
        if self.config.mmap:
            command.append("--mmap")
        return command

    def ensure_server(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            report = self.preflight()
            if not report["ready"]:
                raise RuntimeError("Qwen image preflight failed: " + json.dumps(report, sort_keys=True))
            self.config.lora_root.mkdir(parents=True, exist_ok=True)
            log_path = self.config.artifact_root / "runtime-server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            command = self.command()
            self.emit(
                "runtime_server_starting",
                command=command,
                logPath=str(log_path),
                backend=self.config.device_backend,
                flashAttention=self.config.flash_attention,
                cpuOffload=self.config.cpu_offload,
                mmap=self.config.mmap,
            )
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self._environment(),
            )
            self._reader = threading.Thread(target=self._read_output, args=(self._process,), daemon=True)
            self._reader.start()
        try:
            self._wait_until_ready()
        except Exception:
            # Do not leave a failed native server consuming a GPU or keeping a
            # stale port occupied for the next job.
            self.close()
            raise

    def generate(self, request: QwenImageRequest, lora_paths: tuple[Any, ...], output_path: Path, progress: ProgressCallback) -> dict[str, Any]:
        """Submit one native img_gen job and persist its returned PNG."""
        self.ensure_server()
        body = self._request_body(request, lora_paths)
        self.emit("native_request_submitting", endpoint=f"{self.base_url}/sdcpp/v1/img_gen", request=body)
        started = time.monotonic()
        response = self._json_request("POST", "/sdcpp/v1/img_gen", body, timeout=120)
        native_id = str(response.get("id") or "")
        if not native_id:
            raise RuntimeError("stable-diffusion.cpp did not return a native job id")
        self.emit("native_job_accepted", nativeJobId=native_id, response=response)
        progress("denoise", 0.05, f"Qwen denoising job {native_id} queued")
        deadline = started + self.config.job_timeout_seconds
        last_status: tuple[Any, ...] | None = None
        while time.monotonic() < deadline:
            job = self._json_request("GET", f"/sdcpp/v1/jobs/{native_id}", None, timeout=30)
            status = str(job.get("status") or "").lower()
            result = job.get("result") or {}
            error = job.get("error") or {}
            key = (status, job.get("queue_position"), result.get("images") is not None, error.get("code"))
            if key != last_status:
                last_status = key
                self.emit("native_job_status", nativeJobId=native_id, status=status, queuePosition=job.get("queue_position"), response=job)
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"native Qwen job {native_id} {status}: {error.get('message') or error or job}")
            if status == "completed":
                image_bytes = self._image_bytes(result)
                self._write_png(image_bytes, output_path)
                progress("decode", 0.9, "Qwen image decoded")
                elapsed = time.monotonic() - started
                self.emit("native_job_completed", nativeJobId=native_id, elapsedSeconds=elapsed, output=str(output_path), bytes=len(image_bytes))
                return {
                    "nativeJobId": native_id,
                    "seed": request.seed if request.seed is not None else -1,
                    "steps": request.steps,
                    "cfgScale": request.cfg_scale,
                    "width": request.width,
                    "height": request.height,
                    "elapsedSeconds": elapsed,
                    "transformerQuantization": "Q8_0",
                    "textEncoderProfile": "Qwen2.5-VL-7B-Instruct-abliterated",
                    "loraCount": len(lora_paths),
                    "nativeStatus": job,
                }
            fraction = self._progress_fraction(status, started, request.steps)
            progress("denoise", fraction, f"Qwen native job {native_id}: {status or 'waiting'}")
            time.sleep(self.config.poll_interval_seconds)
        raise TimeoutError(f"Qwen native image job exceeded {self.config.job_timeout_seconds:.0f}s")

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"stable-diffusion.cpp exited during startup with code {self._process.returncode}; see runtime-server.log")
            try:
                capabilities = self._json_request("GET", "/sdcpp/v1/capabilities", None, timeout=5)
                self.emit("runtime_server_ready", capabilities=capabilities)
                return
            except Exception as exc:
                self.emit("runtime_server_waiting", errorType=type(exc).__name__, error=str(exc))
                time.sleep(2)
        raise TimeoutError(f"stable-diffusion.cpp did not become ready within {self.config.startup_timeout_seconds:.0f}s")

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                text = line.rstrip()
                if self._log_handle is not None:
                    self._log_handle.write(line)
                print(f"[qwen-native] {text}", flush=True)
                self.emit("native_log", line=text[:8000])
        except Exception as exc:
            self.emit("native_log_reader_failed", errorType=type(exc).__name__, error=str(exc))

    def _json_request(self, method: str, path: str, body: dict[str, Any] | None, *, timeout: float) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw) if raw else {}
                if not isinstance(parsed, dict):
                    raise RuntimeError(f"native server returned non-object JSON for {path}")
                return parsed
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"native server HTTP {exc.code} at {path}: {detail[:4000]}") from exc
        except (URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"native server request failed at {path}: {exc}") from exc

    def _request_body(self, request: QwenImageRequest, lora_paths: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "clip_skip": -1,
            "width": request.width,
            "height": request.height,
            "strength": 0.75,
            "seed": request.seed if request.seed is not None else -1,
            "batch_count": 1,
            "sample_params": {
                "scheduler": self.config.scheduler,
                "sample_method": self.config.sampler,
                "sample_steps": request.steps,
                "eta": 1.0,
                "flow_shift": self.config.flow_shift,
                "guidance": {"txt_cfg": request.cfg_scale},
            },
            "lora": [
                {
                    "path": str(lora.path),
                    "multiplier": lora.scale,
                    "is_high_noise": lora.is_high_noise,
                }
                for lora in lora_paths
            ],
            "output_format": "png",
            "output_compression": 100,
        }

    @staticmethod
    def _image_bytes(result: dict[str, Any]) -> bytes:
        value = result.get("b64_json")
        if not value and isinstance(result.get("images"), list) and result["images"]:
            first = result["images"][0]
            value = first.get("b64_json") if isinstance(first, dict) else first
        if not isinstance(value, str) or not value:
            raise RuntimeError("native Qwen job completed without a base64 image result")
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("native Qwen job returned invalid base64 image data") from exc

    @staticmethod
    def _write_png(data: bytes, path: Path) -> None:
        if (
            not data.startswith(b"\x89PNG\r\n\x1a\n")
            or b"\x00\x00\x00\rIHDR" not in data[:32]
            or b"IEND" not in data[-16:]
        ):
            raise RuntimeError("native Qwen result is not a valid non-empty PNG")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    @staticmethod
    def _progress_fraction(status: str, started: float, steps: int) -> float:
        if status in {"queued", ""}:
            return 0.08
        # Native server versions do not all expose step progress. Keep this as
        # an honest elapsed estimate and replace it with native progress when
        # the server supplies one in a future response.
        estimate = min(0.82, 0.10 + (time.monotonic() - started) / max(steps * 8.0, 1.0))
        return estimate

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.setdefault("GGML_CUDA_NO_VMM", "0")
        environment.setdefault("GGML_CUDA_FORCE_MMQ", "0")
        return environment

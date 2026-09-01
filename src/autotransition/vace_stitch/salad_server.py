"""HTTP entrypoint for the VACE stitch Salad worker."""

from __future__ import annotations

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from autotransition.generative_dance.video import resolve_ffmpeg, resolve_ffprobe

from .config import VaceStitchConfig
from .runtime import VaceRuntime
from .salad_adapter import process_queue_job
from .worker import VaceStitchWorker


config = VaceStitchConfig.from_env()
runtime = VaceRuntime(config)
worker = VaceStitchWorker(config, runtime=runtime)


def _asset(path: Path | None) -> dict[str, object]:
    return {"configured": bool(path), "exists": bool(path and (path.is_file() or path.is_dir()))}


def preflight() -> dict[str, object]:
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "deviceCount": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "deviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        cuda = {"available": False, "deviceCount": 0, "deviceName": None, "error": f"{type(exc).__name__}: {exc}"}
    source_script = config.source_root / "vace" / "vace_wan_inference.py" if config.source_root else None
    assets = {
        "sourceScript": _asset(source_script),
        "checkpointDir": {"configured": bool(config.checkpoint_dir), "exists": bool(config.checkpoint_dir and config.checkpoint_dir.is_dir())},
        "matteCheckpoint": _asset(config.matte_checkpoint),
    }
    required_assets = {"sourceScript", "checkpointDir"} if config.runtime_backend == "native" else set()
    missing = [name for name, report in assets.items() if not report["exists"]]
    missing_required = [name for name in missing if name in required_assets]
    missing_stages = []
    if config.enhancement_enabled and not config.enhancement_command:
        missing_stages.append("enhancementCommand")
    if config.motion_interpolation_enabled and not config.motion_interpolation_command:
        missing_stages.append("motionInterpolationCommand")
    runtime_ready = runtime.configured and not missing_required
    gpu_ready = config.runtime_backend == "command" or cuda["available"]
    return {
        "ready": bool(runtime_ready and gpu_ready and not missing_stages and resolve_ffmpeg() and resolve_ffprobe()),
        "runtime": "wan-vace-stitch",
        "modelName": config.model_name,
        "modelSize": config.model_size,
        "assets": assets,
        "missing": missing,
        "missingRequired": missing_required,
        "missingStages": missing_stages,
        "cuda": cuda,
        "ffmpeg": resolve_ffmpeg(),
        "ffprobe": resolve_ffprobe(),
        "offloadPolicy": "VACE model offload is configurable; keep it enabled for constrained GPUs and disable it when the full DiT model can remain resident",
        "attentionPolicy": "The official Wan attention path uses FlashAttention when available; flash_attention_2 can be required explicitly and is reported by the runtime self-test",
        "sequencePolicy": "one request contains the complete ordered sequence; all bridges and the optional loop bridge run under one durable job",
        "maskPolicy": "black preserves context; white generates the gray placeholder gap",
    }


def health() -> dict[str, object]:
    return {"ok": bool(preflight()["ready"]), **preflight()}


def ready() -> tuple[int, dict[str, object]]:
    report = preflight()
    return (200 if report["ready"] else 503), {"ok": bool(report["ready"]), **report}


def status() -> dict[str, object]:
    report = preflight()
    with worker._lock:
        active = list(worker._futures)
    return {**report, "activeJobs": active, "config": config.to_public_dict()}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, _Handler)
        self.worker = worker
        self.config = config


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        print(json.dumps({"event": "vace_http", "message": format % args}), flush=True)

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._write_json(200, health())
        elif path == "/ready":
            code, payload = ready()
            self._write_json(code, payload)
        elif path == "/v1/worker/status":
            self._write_json(200, status())
        else:
            self._write_json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path != "/process":
            self._write_json(404, {"detail": "not found"})
            return
        payload: dict[str, Any] = {}
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 16 * 1024 * 1024:
                raise ValueError("request body is missing or too large")
            decoded = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("request body must be a JSON object")
            payload = decoded
            if payload.get("runtime") not in {None, "wan-vace-stitch", "vace-stitch", "wan-vace"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            result = asyncio.run(process_queue_job(payload, self.server.worker, self.server.config))
            self._write_json(200, result)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "vace_queue_job_failed",
                        "jobId": payload.get("job_id"),
                        "errorType": type(exc).__name__,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
            self._write_json(500, {"detail": str(exc)})


if __name__ == "__main__":
    print(json.dumps({"event": "vace_worker_starting", **preflight(), "config": config.to_public_dict()}, default=str, sort_keys=True), flush=True)
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8080"))
    with _Server((host, port)) as server:
        print(json.dumps({"event": "vace_worker_ready", "host": host, "port": port}), flush=True)
        server.serve_forever()

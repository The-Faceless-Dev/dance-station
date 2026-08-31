"""HTTP entry point for the Salad queue worker."""

from __future__ import annotations

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import torch

from .config import GenerativeDanceConfig
from .salad_adapter import SaladRequestError, process_queue_job
from .worker import GenerativeDanceWorker


config = GenerativeDanceConfig.from_env()
worker = GenerativeDanceWorker(config)


def _asset_report(path: object) -> dict[str, object]:
    candidate = path if hasattr(path, "exists") else None
    return {"configured": bool(candidate), "exists": bool(candidate and candidate.exists())}


def preflight() -> dict[str, object]:
    paths = {
        "transformer": config.wan_transformer_checkpoint,
        "officialSource": config.wan_official_source,
        "t5": config.wan_t5_checkpoint,
        "t5Tokenizer": config.wan_t5_tokenizer,
        "clip": config.wan_clip_checkpoint,
        "clipTokenizer": config.wan_clip_tokenizer,
        "vae": config.wan_vae_checkpoint,
        "matte": config.matte_checkpoint,
    }
    assets = {name: _asset_report(value) for name, value in paths.items()}
    missing = [name for name, report in assets.items() if not report["exists"]]
    cuda = {
        "available": bool(torch.cuda.is_available()),
        "deviceCount": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "deviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return {
        "ready": not missing and (config.wan_device == "cpu" or cuda["available"]),
        "runtime": "wan-animate",
        "modelRevision": config.wan_model_revision,
        "assets": assets,
        "missing": missing,
        "cuda": cuda,
        "offloadPolicy": "GGUF linear weights memory-mapped on CPU; T5/CLIP load, encode, release; one GPU job",
        "temporalPolicy": (
            "full driver with configurable overlapping windows; "
            f"window={config.wan_temporal_window} context={config.wan_temporal_context_frames} frames; "
            "source FPS preserved"
        ),
    }


def health() -> dict[str, object]:
    report = preflight()
    return {"ok": bool(report["ready"]), **report}


def ready() -> tuple[int, dict[str, object]]:
    report = preflight()
    if not report["ready"]:
        return 503, {"detail": report}
    return 200, {"ok": True, **report}


def status() -> dict[str, object]:
    report = preflight()
    with worker._lock:
        active = list(worker._futures)
    return {**report, "activeJobs": active, "config": config.to_public_dict()}


class _WanHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, _WanRequestHandler)
        self.worker = worker
        self.config = config


class _WanRequestHandler(BaseHTTPRequestHandler):
    server: _WanHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        print(json.dumps({"event": "wan_http", "message": format % args}), flush=True)

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
            return
        if path == "/ready":
            code, payload = ready()
            self._write_json(code, payload)
            return
        if path == "/v1/worker/status":
            self._write_json(200, status())
            return
        self._write_json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path != "/process":
            self._write_json(404, {"detail": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 16 * 1024 * 1024:
                raise ValueError("request body is missing or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            if payload.get("runtime") not in {None, "wan-animate", "wan-animate-worker", "generative-dance", "generative-dance-worker"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            result = asyncio.run(process_queue_job(payload, self.server.worker, self.server.config))
            self._write_json(200, result)
        except SaladRequestError as exc:
            self._write_json(exc.status_code, {"detail": str(exc)})
        except Exception as exc:
            print(json.dumps({"event": "wan_queue_job_failed", "jobId": payload.get("job_id") if "payload" in locals() and isinstance(payload, dict) else None, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)
            self._write_json(500, {"detail": str(exc)})


if __name__ == "__main__":
    print(json.dumps({"event": "wan_worker_starting", **preflight(), "config": config.to_public_dict()}, default=str, sort_keys=True), flush=True)
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8080"))
    with _WanHTTPServer((host, port)) as server:
        print(json.dumps({"event": "wan_worker_ready", "host": host, "port": port}), flush=True)
        server.serve_forever()

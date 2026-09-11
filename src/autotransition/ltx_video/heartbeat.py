"""Publish LTX worker readiness to the launch server."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen


def _value(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _probe(url: str) -> tuple[bool, dict[str, object]]:
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=5) as response:
            body = response.read().decode("utf-8")
            value = json.loads(body) if body else {}
            return 200 <= response.status < 300, value if isinstance(value, dict) else {}
    except Exception as exc:
        return False, {"errorType": type(exc).__name__, "error": str(exc)}


def run() -> int:
    callback_url = _value("LAUNCH_SERVER_HEARTBEAT_URL")
    token = _value("WORKER_HEARTBEAT_TOKEN")
    if not callback_url or not token:
        print("[ltx-heartbeat] disabled: heartbeat configuration is absent", flush=True)
        return 0
    ready_url = _value("LTX_VIDEO_HEARTBEAT_READY_URL", "http://127.0.0.1:8080/ready")
    interval = max(5.0, float(_value("WORKER_HEARTBEAT_INTERVAL_SECONDS", "15")))
    while True:
        ready, diagnostics = _probe(ready_url)
        payload = {
            "provider": _value("WORKER_PROVIDER", "vast"),
            "container_group": _value("VAST_WORKER_GROUP", _value("VAST_INSTANCE_LABEL", "unknown")),
            "instance_id": _value("VAST_INSTANCE_ID", _value("CONTAINER_ID", _value("HOSTNAME", "unknown-instance"))),
            "runtime": "ltx-video",
            "model_revision": _value("LTX_VIDEO_MODEL_NAME", "LTX-2.5-22B-Distilled-NVFP4"),
            "state": "ready" if ready else "starting",
            "ready": ready,
            "loaded_models": [_value("LTX_VIDEO_MODEL_NAME", "LTX-2.5-22B-Distilled-NVFP4")],
            "capacity": diagnostics,
            "heartbeat_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            request = Request(callback_url, data=data, headers={"Content-Type": "application/json", "X-Worker-Heartbeat-Token": token}, method="POST")
            with urlopen(request, timeout=10) as response:
                response.read()
            print(f"[ltx-heartbeat] sent ready={ready}", flush=True)
        except Exception as exc:
            print(f"[ltx-heartbeat] send_failed error_type={type(exc).__name__} error={exc}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(run())

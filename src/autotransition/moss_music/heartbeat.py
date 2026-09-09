"""Publish MOSS-Music worker readiness to the launch server."""

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
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return 200 <= response.status < 300, parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        return False, {"errorType": type(exc).__name__}


def run() -> int:
    callback_url = _value("LAUNCH_SERVER_HEARTBEAT_URL")
    token = _value("WORKER_HEARTBEAT_TOKEN")
    if not callback_url or not token:
        print("[moss-heartbeat] disabled: heartbeat configuration is absent", flush=True)
        return 0
    ready_url = _value("MOSS_MUSIC_HEARTBEAT_READY_URL", "http://127.0.0.1:8080/ready")
    interval = max(5.0, float(_value("WORKER_HEARTBEAT_INTERVAL_SECONDS", "15")))
    while True:
        ready, diagnostics = _probe(ready_url)
        payload = {
            "provider": _value("WORKER_PROVIDER", "salad"),
            "container_group": _value("SALAD_CONTAINER_GROUP", "unknown"),
            "instance_id": _value("SALAD_INSTANCE_ID", _value("HOSTNAME", "unknown-instance")),
            "runtime": "moss-music",
            "model_revision": _value("MOSS_MUSIC_MODEL_NAME", "MOSS-Music-8B-Instruct"),
            "state": "ready" if ready else "starting",
            "ready": ready,
            "loaded_models": [_value("MOSS_MUSIC_MODEL_NAME", "MOSS-Music-8B-Instruct")],
            "capacity": diagnostics,
            "heartbeat_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            request = Request(callback_url, data=data, headers={"Content-Type": "application/json", "X-Worker-Heartbeat-Token": token}, method="POST")
            with urlopen(request, timeout=10) as response:
                response.read()
            print(f"[moss-heartbeat] sent ready={ready}", flush=True)
        except Exception as exc:
            print(f"[moss-heartbeat] send_failed error_type={type(exc).__name__}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(run())

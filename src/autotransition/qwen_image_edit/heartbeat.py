"""Publish Qwen Image Edit worker readiness to the launch server."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from autotransition.avatar.heartbeat import _env_value, probe_ready, send_heartbeat


def run() -> int:
    values = os.environ
    url = _env_value(values, "LAUNCH_SERVER_HEARTBEAT_URL")
    token = _env_value(values, "WORKER_HEARTBEAT_TOKEN")
    if not url or not token:
        print("[qwen-image-edit-heartbeat] disabled: launch heartbeat configuration is absent", flush=True)
        return 0
    ready_url = _env_value(values, "QWEN_IMAGE_EDIT_HEARTBEAT_READY_URL", default="http://127.0.0.1:8080/ready")
    interval = max(5.0, float(_env_value(values, "WORKER_HEARTBEAT_INTERVAL_SECONDS", default="15")))
    print(f"[qwen-image-edit-heartbeat] enabled interval_seconds={interval:g} ready_url={ready_url}", flush=True)
    while True:
        ready, capacity = probe_ready(ready_url)
        payload: dict[str, Any] = {
            "provider": _env_value(values, "WORKER_PROVIDER", default="runpod"),
            "organization": _env_value(values, "SALAD_ORGANIZATION", "SALAD_ORGANIZATION_NAME"),
            "project": _env_value(values, "SALAD_PROJECT", "SALAD_PROJECT_NAME"),
            "container_group": _env_value(values, "SALAD_CONTAINER_GROUP", "SALAD_CONTAINER_GROUP_NAME"),
            "instance_id": _env_value(values, "SALAD_INSTANCE_ID", "HOSTNAME", default="unknown-instance"),
            "machine_id": _env_value(values, "SALAD_MACHINE_ID") or None,
            "runtime": "qwen-image-edit-2511",
            "model_revision": _env_value(values, "WORKER_MODEL_REVISION", "QWEN_IMAGE_EDIT_MODEL_NAME", default="Qwen-Image-Edit-2511-Q8_0"),
            "state": "ready" if ready else "starting",
            "ready": ready,
            "loaded_models": ["Qwen-Image-Edit-2511-Q8_0", "Qwen-Image-Edit-VAE", "Qwen2.5-VL"],
            "capacity": capacity,
            "message": "Qwen Image Edit worker readiness probe passed" if ready else "Qwen Image Edit worker is waiting for readiness",
            "heartbeat_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            send_heartbeat(url, token, payload)
            print(f"[qwen-image-edit-heartbeat] sent state={payload['state']} ready={ready}", flush=True)
        except Exception as exc:
            print(f"[qwen-image-edit-heartbeat] send_failed error_type={type(exc).__name__}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(run())

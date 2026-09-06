"""Publish FLUX image-worker readiness to the launch server."""

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
        print("[flux-image-heartbeat] disabled: launch heartbeat configuration is absent", flush=True)
        return 0
    local_url = _env_value(values, "FLUX_IMAGE_HEARTBEAT_READY_URL", default="http://127.0.0.1:8080/ready")
    try:
        interval = max(5.0, float(_env_value(values, "WORKER_HEARTBEAT_INTERVAL_SECONDS", default="15")))
    except ValueError:
        interval = 15.0
    print(f"[flux-image-heartbeat] enabled interval_seconds={interval:g} ready_url={local_url}", flush=True)
    while True:
        ready, diagnostics = probe_ready(local_url)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "provider": _env_value(values, "WORKER_PROVIDER", default="salad"),
            "organization": _env_value(values, "SALAD_ORGANIZATION", "SALAD_ORGANIZATION_NAME"),
            "project": _env_value(values, "SALAD_PROJECT", "SALAD_PROJECT_NAME"),
            "container_group": _env_value(values, "SALAD_CONTAINER_GROUP", "SALAD_CONTAINER_GROUP_NAME"),
            "instance_id": _env_value(values, "SALAD_INSTANCE_ID", "HOSTNAME", default="unknown-instance"),
            "machine_id": _env_value(values, "SALAD_MACHINE_ID") or None,
            "runtime": "flux-image",
            "model_revision": _env_value(values, "WORKER_MODEL_REVISION", "FLUX_IMAGE_MODEL_NAME", default="flux.2-klein-4b"),
            "state": "ready" if ready else "starting",
            "ready": ready,
            "loaded_models": ["flux.2-klein-4b", "qwen3-4b", "flux-vae"],
            "capacity": diagnostics,
            "message": "FLUX image worker readiness probe passed" if ready else "FLUX image worker is waiting for readiness",
            "heartbeat_at": now,
        }
        try:
            send_heartbeat(url, token, payload)
            print(f"[flux-image-heartbeat] sent state={payload['state']} ready={ready}", flush=True)
        except Exception as exc:
            print(f"[flux-image-heartbeat] send_failed error_type={type(exc).__name__}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(run())

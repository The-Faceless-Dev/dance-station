"""Publish generative-dance readiness to the launch server."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from autotransition.avatar.heartbeat import _env_value, probe_ready, send_heartbeat


def run() -> int:
    values = os.environ
    url = _env_value(values, "LAUNCH_SERVER_HEARTBEAT_URL")
    token = _env_value(values, "WORKER_HEARTBEAT_TOKEN")
    if not url or not token:
        print("[wan-animate-heartbeat] disabled: launch heartbeat configuration is absent", flush=True)
        return 0
    local_url = _env_value(values, "WAN_ANIMATE_HEARTBEAT_READY_URL", default="http://127.0.0.1:8080/ready")
    interval = max(5.0, float(_env_value(values, "WORKER_HEARTBEAT_INTERVAL_SECONDS", default="15")))
    print(f"[wan-animate-heartbeat] enabled interval_seconds={interval:g} ready_url={local_url}", flush=True)
    while True:
        ready, diagnostics = probe_ready(local_url)
        payload = {
            "provider": _env_value(values, "WORKER_PROVIDER", default="salad"),
            "organization": _env_value(values, "SALAD_ORGANIZATION", "SALAD_ORGANIZATION_NAME", "VAST_ORGANIZATION"),
            "project": _env_value(values, "SALAD_PROJECT", "SALAD_PROJECT_NAME", "VAST_PROJECT"),
            "container_group": _env_value(values, "SALAD_CONTAINER_GROUP", "SALAD_CONTAINER_GROUP_NAME", "VAST_WORKER_GROUP", "VAST_INSTANCE_LABEL"),
            "instance_id": _env_value(values, "SALAD_INSTANCE_ID", "VAST_INSTANCE_ID", "CONTAINER_ID", "HOSTNAME", default="unknown-instance"),
            "machine_id": _env_value(values, "SALAD_MACHINE_ID", "VAST_MACHINE_ID") or None,
            "runtime": "wan-animate",
            "model_revision": _env_value(values, "WORKER_MODEL_REVISION", "GENERATIVE_DANCE_WAN_MODEL", default="Wan-Animate-2-Q6_K"),
            "state": "ready" if ready else "starting",
            "ready": ready,
            "loaded_models": ["Wan-Animate-2-Q6_K", "UMT5-XXL", "CLIP-XLM-R", "BiRefNet"],
            "capacity": diagnostics,
            "message": "Wan Animate readiness probe passed" if ready else "Wan Animate worker is waiting for readiness",
            "heartbeat_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            send_heartbeat(url, token, payload)
            print(f"[wan-animate-heartbeat] sent state={payload['state']} ready={ready}", flush=True)
        except Exception as exc:
            print(f"[wan-animate-heartbeat] send_failed error_type={type(exc).__name__} error={exc}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(run())

"""Launch-server heartbeat for the VACE stitch worker."""

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
        print("[vace-stitch-heartbeat] disabled: launch heartbeat configuration is absent", flush=True)
        return 0
    ready_url = _env_value(values, "VACE_STITCH_HEARTBEAT_READY_URL", default="http://127.0.0.1:8080/ready")
    interval = max(5.0, float(_env_value(values, "WORKER_HEARTBEAT_INTERVAL_SECONDS", default="15")))
    print(f"[vace-stitch-heartbeat] enabled interval_seconds={interval:g} ready_url={ready_url}", flush=True)
    while True:
        is_ready, diagnostics = probe_ready(ready_url)
        payload = {
            "provider": _env_value(values, "WORKER_PROVIDER", default="salad"),
            "organization": _env_value(values, "SALAD_ORGANIZATION", "SALAD_ORGANIZATION_NAME"),
            "project": _env_value(values, "SALAD_PROJECT", "SALAD_PROJECT_NAME"),
            "container_group": _env_value(values, "SALAD_CONTAINER_GROUP", "SALAD_CONTAINER_GROUP_NAME"),
            "instance_id": _env_value(values, "SALAD_INSTANCE_ID", "HOSTNAME", default="unknown-instance"),
            "machine_id": _env_value(values, "SALAD_MACHINE_ID") or None,
            "runtime": "wan-vace-stitch",
            "model_revision": _env_value(values, "WORKER_MODEL_REVISION", "VACE_STITCH_MODEL_NAME", default="vace-1.3B"),
            "state": "ready" if is_ready else "starting",
            "ready": is_ready,
            "loaded_models": [_env_value(values, "VACE_STITCH_MODEL_NAME", default="vace-1.3B"), "Wan2.1 VACE", "BiRefNet"],
            "capacity": diagnostics,
            "message": "Wan2.1 VACE stitch readiness probe passed" if is_ready else "VACE stitch worker is waiting for readiness",
            "heartbeat_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            send_heartbeat(url, token, payload)
            print(f"[vace-stitch-heartbeat] sent state={payload['state']} ready={is_ready}", flush=True)
        except Exception as exc:
            print(f"[vace-stitch-heartbeat] send_failed error_type={type(exc).__name__} error={exc}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(run())

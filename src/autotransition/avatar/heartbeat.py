"""Publish avatar-worker readiness to the launch server."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODELS = ("flux.2-klein-4b", "trellis.2-4b", "SkinTokens")


def _env_value(values: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = str(values.get(name, "")).strip()
        if value:
            return value
    return default


def _loaded_models(values: Mapping[str, str]) -> list[str]:
    configured = _env_value(values, "WORKER_LOADED_MODELS")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()][:32]
    return list(DEFAULT_MODELS)


def build_heartbeat_payload(
    values: Mapping[str, str],
    *,
    ready: bool,
    capacity: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the launch-server heartbeat payload without exposing secrets."""

    heartbeat_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    provider = _env_value(values, "WORKER_PROVIDER", default="salad")
    group = _env_value(values, "SALAD_CONTAINER_GROUP", "SALAD_CONTAINER_GROUP_NAME")
    instance = _env_value(values, "SALAD_INSTANCE_ID", "HOSTNAME", default="unknown-instance")
    machine = _env_value(values, "SALAD_MACHINE_ID")
    return {
        "provider": provider,
        "organization": _env_value(values, "SALAD_ORGANIZATION", "SALAD_ORGANIZATION_NAME"),
        "project": _env_value(values, "SALAD_PROJECT", "SALAD_PROJECT_NAME"),
        "container_group": group,
        "instance_id": instance,
        "machine_id": machine or None,
        "runtime": "avatar",
        "model_revision": _env_value(values, "WORKER_MODEL_REVISION", "AVATAR_MESH_MODEL_REVISION", default="trellis.2-4b"),
        "state": "ready" if ready else "starting",
        "ready": ready,
        "loaded_models": _loaded_models(values),
        "capacity": dict(capacity or {}),
        "message": "Avatar worker readiness probe passed" if ready else "Avatar worker is waiting for readiness",
        "heartbeat_at": heartbeat_at,
    }


def probe_ready(url: str, *, timeout: float = 5.0) -> tuple[bool, dict[str, Any]]:
    """Probe the local worker readiness endpoint and return safe diagnostics."""

    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            if response.status < 200 or response.status >= 300:
                return False, {"statusCode": response.status}
            return True, {"gpu": parsed.get("gpu", {}), "status": parsed.get("status", "ready")}
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return False, {"error": type(exc).__name__}


def send_heartbeat(url: str, token: str, payload: Mapping[str, Any], *, timeout: float = 10.0) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-Worker-Heartbeat-Token": token,
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        response.read()
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"heartbeat HTTP {response.status}")


def run() -> int:
    values = os.environ
    url = _env_value(values, "LAUNCH_SERVER_HEARTBEAT_URL")
    token = _env_value(values, "WORKER_HEARTBEAT_TOKEN")
    if not url or not token:
        print("[avatar-heartbeat] disabled: launch heartbeat configuration is absent", flush=True)
        return 0

    local_url = _env_value(values, "AVATAR_HEARTBEAT_READY_URL", default="http://127.0.0.1:8080/ready")
    try:
        interval = max(5.0, float(_env_value(values, "WORKER_HEARTBEAT_INTERVAL_SECONDS", default="15")))
    except ValueError:
        interval = 15.0
    print(f"[avatar-heartbeat] enabled interval_seconds={interval:g} ready_url={local_url}", flush=True)

    while True:
        ready, capacity = probe_ready(local_url)
        payload = build_heartbeat_payload(values, ready=ready, capacity=capacity)
        try:
            send_heartbeat(url, token, payload)
            print(f"[avatar-heartbeat] sent state={payload['state']} ready={ready}", flush=True)
        except Exception as exc:
            print(f"[avatar-heartbeat] send_failed error_type={type(exc).__name__}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(run())

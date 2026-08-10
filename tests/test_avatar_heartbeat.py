from datetime import datetime, timezone

from autotransition.avatar.heartbeat import build_heartbeat_payload


def test_heartbeat_payload_uses_salad_runtime_identity() -> None:
    payload = build_heartbeat_payload(
        {
            "WORKER_PROVIDER": "salad",
            "SALAD_ORGANIZATION": "faceless-dancer",
            "SALAD_PROJECT": "gen-workers",
            "SALAD_CONTAINER_GROUP": "faceless-avatar-trellis-v2-r8",
            "SALAD_INSTANCE_ID": "instance-1",
            "SALAD_MACHINE_ID": "machine-1",
            "AVATAR_MESH_MODEL_REVISION": "trellis.2-4b",
        },
        ready=True,
        capacity={"gpu": {"available": True}},
        now=datetime(2026, 8, 9, 21, 0, tzinfo=timezone.utc),
    )

    assert payload == {
        "provider": "salad",
        "organization": "faceless-dancer",
        "project": "gen-workers",
        "container_group": "faceless-avatar-trellis-v2-r8",
        "instance_id": "instance-1",
        "machine_id": "machine-1",
        "runtime": "avatar",
        "model_revision": "trellis.2-4b",
        "state": "ready",
        "ready": True,
        "loaded_models": ["flux.2-klein-4b", "trellis.2-4b", "SkinTokens"],
        "capacity": {"gpu": {"available": True}},
        "message": "Avatar worker readiness probe passed",
        "heartbeat_at": "2026-08-09T21:00:00Z",
    }


def test_heartbeat_payload_can_run_without_salad_specific_names() -> None:
    payload = build_heartbeat_payload({"HOSTNAME": "local-worker"}, ready=False)

    assert payload["instance_id"] == "local-worker"
    assert payload["state"] == "starting"
    assert payload["ready"] is False

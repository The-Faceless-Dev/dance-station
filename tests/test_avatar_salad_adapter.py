from __future__ import annotations

import asyncio
from pathlib import Path

from autotransition.avatar import salad_adapter
from autotransition.avatar.worker import AvatarWorker

from test_avatar_worker import make_pipeline, write_png


def test_queue_payload_maps_reference_and_request_fields(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    payload = {
        "runtime": "avatar",
        "job_id": "avatar-job-1",
        "inputs": [{"role": "reference", "sourceUrl": "https://cdn.example/reference.png", "fileName": "reference.png"}],
        "parameters": {"prompt": "a blue humanoid bear", "quality": "quality", "seed": 42, "maxAttempts": 2},
        "callback": {"url": "https://launcher.example/artifacts", "complete_url": "https://launcher.example/complete", "token": "callback-token"},
    }

    url, filename = salad_adapter._reference_url(payload)
    request = salad_adapter._request_from_payload(payload, reference)

    assert url == "https://cdn.example/reference.png"
    assert filename == "reference.png"
    assert request.description == "a blue humanoid bear"
    assert request.reference_image == reference
    assert request.quality == "quality"
    assert request.seed == 42
    assert request.max_attempts == 2
    assert request.external_job_id == "avatar-job-1"
    assert salad_adapter._callback_payload(payload) == (
        "https://launcher.example/artifacts",
        "https://launcher.example/complete",
        "",
        "callback-token",
    )


def test_progress_callback_matches_launch_server_schema(monkeypatch) -> None:
    sent: list[dict] = []

    def fake_post_json(url: str, token: str, payload: dict, *, timeout: float):
        sent.append(payload)
        return {}

    monkeypatch.setattr(salad_adapter, "_post_json", fake_post_json)
    job = {
        "stage": "mesh_generation",
        "progress": 0.37,
        "attempt": 1,
        "updated_at": "2026-08-06T12:00:00+00:00",
    }

    asyncio.run(
        salad_adapter._report_progress(
            "https://launcher.example/progress",
            "callback-token",
            "avatar-job-1",
            status="running",
            job=job,
            sequence=4,
        )
    )

    assert sent == [
        {
            "schema_version": 1,
            "job_id": "avatar-job-1",
            "runtime": "avatar",
            "status": "running",
            "phase": "mesh_generation",
            "progress": 0.37,
            "completed_steps": 37,
            "total_steps": 100,
            "attempt": 1,
            "message": None,
            "stage": "mesh_generation",
            "stage_index": None,
            "stage_count": None,
            "stage_kind": None,
            "task_id": None,
            "sequence": 4,
            "updated_at": "2026-08-06T12:00:00+00:00",
        }
    ]


def test_artifact_roles_match_shared_callback_contract() -> None:
    assert salad_adapter._artifact_role("source-image.png") == "preview"
    assert salad_adapter._artifact_role("avatar.glb") == "metadata"
    assert salad_adapter._artifact_role("manifest.json") == "metadata"


def test_queue_paper_run_completes_pipeline_and_callback_delivery(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "reference.png"
    write_png(source)
    pipeline, _ = make_pipeline(tmp_path, rig_failures=0)
    worker = AvatarWorker(pipeline)
    callbacks: list[dict] = []
    uploaded: list[str] = []

    def fake_download(_url: str, _filename: str, _config):
        return source

    def fake_post_json(_url: str, _token: str, payload: dict, *, timeout: float = 120):
        callbacks.append(payload)
        return {}

    def fake_upload(_url: str, _token: str, path: Path, artifact: dict) -> str:
        uploaded.append(path.name)
        return f"artifact-{path.name}"

    monkeypatch.setattr(salad_adapter, "_download_reference", fake_download)
    monkeypatch.setattr(salad_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(salad_adapter, "_upload_artifact", fake_upload)
    payload = {
        "runtime": "avatar",
        "job_id": "avatar-paper-job",
        "inputs": [{"role": "reference", "sourceUrl": "https://cdn.example/reference.png", "fileName": "reference.png"}],
        "parameters": {"description": "a paper-run humanoid", "quality": "runtime", "seed": 7, "max_attempts": 1, "payment_intent_id": "paper-payment"},
        "callback": {
            "url": "https://launcher.example/artifacts",
            "complete_url": "https://launcher.example/complete",
            "progress_url": "https://launcher.example/progress",
            "token": "paper-callback-token",
        },
    }
    try:
        result = asyncio.run(salad_adapter._run_queue_job(payload, worker, pipeline.config))
    finally:
        worker.shutdown()

    assert result["status"] == "succeeded"
    assert set(uploaded) >= {"avatar.glb", "manifest.json", "diagnostics.json"}
    assert callbacks
    progress_callbacks = [callback for callback in callbacks if "runtime" in callback]
    assert progress_callbacks
    assert all(callback["runtime"] == "avatar" for callback in progress_callbacks)
    assert progress_callbacks[-1]["status"] == "succeeded"
    assert progress_callbacks[-1]["updated_at"].endswith("Z")

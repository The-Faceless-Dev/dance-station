from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from autotransition.avatar import salad_adapter
from autotransition.avatar.canonical_skeleton import fit_profile
from autotransition.avatar.reskin_pipeline import AvatarReskinPipeline
from autotransition.avatar.contracts import AvatarReskinRequest
from autotransition.avatar.artifacts import AvatarArtifactStore
from autotransition.config import AvatarConfig
from autotransition.avatar.worker import AvatarWorker

from test_avatar_worker import make_pipeline, write_glb, write_png
from test_canonical_skeleton import write_mesh_glb


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


def test_queue_payload_maps_reskin_inputs_and_does_not_use_image_prompt(tmp_path: Path) -> None:
    mesh = tmp_path / "source-mesh.glb"
    profile = tmp_path / "canonical-profile.json"
    write_mesh_glb(mesh)
    profile.write_text("{}", encoding="utf-8")
    payload = {
        "runtime": "avatar",
        "job_id": "avatar-reskin-1",
        "inputs": [
            {"role": "mesh", "sourceUrl": "https://cdn.example/avatar.glb", "fileName": "avatar.glb"},
            {"role": "canonical_profile", "sourceUrl": "https://cdn.example/profile.json", "fileName": "profile.json"},
        ],
        "parameters": {"task_type": "avatar_reskin", "quality": "runtime"},
        "callback": {"url": "https://launcher.example/artifacts", "complete_url": "https://launcher.example/complete", "token": "callback-token"},
    }

    request = salad_adapter._request_from_payload(payload, None, mesh=mesh, profile=profile)

    assert isinstance(request, AvatarReskinRequest)
    assert request.mesh == mesh
    assert request.profile == profile


class FakeReskinGenerator:
    def generate(self, *, skeleton: Path, output: Path, manifest: Path, profile: Path, quality: str):
        write_glb(output)
        roles = (
            "hips", "spine", "chest", "head", "upperArmLeft", "upperArmRight",
            "forearmLeft", "forearmRight", "upperLegLeft", "upperLegRight",
            "lowerLegLeft", "lowerLegRight", "footLeft", "footRight",
        )
        manifest.write_text(json.dumps({"skeletonId": "humanoid-v1", "bones": {role: role for role in roles}}), encoding="utf-8")
        return output, manifest


def test_queue_paper_run_completes_reskin_and_uploads_all_artifacts(tmp_path: Path, monkeypatch) -> None:
    source_mesh = tmp_path / "source-mesh.glb"
    source_profile = tmp_path / "canonical-profile.json"
    write_mesh_glb(source_mesh)
    source_profile.write_text(json.dumps(fit_profile(source_mesh)), encoding="utf-8")
    pipeline, _ = make_pipeline(tmp_path, rig_failures=0)
    reskin = AvatarReskinPipeline(
        pipeline.config,
        reskin_generator=FakeReskinGenerator(),
        store=AvatarArtifactStore(pipeline.config.artifact_root),
    )
    worker = AvatarWorker(pipeline, reskin_pipeline=reskin)
    callbacks: list[dict] = []
    uploaded: list[str] = []

    def fake_download(_url: str, _filename: str, _config, **kwargs):
        return source_mesh if kwargs["prefix"] == "mesh" else source_profile

    def fake_post_json(_url: str, _token: str, payload: dict, *, timeout: float = 120):
        callbacks.append(payload)
        return {}

    def fake_upload(_url: str, _token: str, path: Path, artifact: dict) -> str:
        uploaded.append(path.name)
        return f"artifact-{path.name}"

    monkeypatch.setattr(salad_adapter, "_download_asset", fake_download)
    monkeypatch.setattr(salad_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(salad_adapter, "_upload_artifact", fake_upload)
    payload = {
        "runtime": "avatar",
        "job_id": "avatar-reskin-paper-job",
        "inputs": [
            {"role": "mesh", "sourceUrl": "https://cdn.example/avatar.glb", "fileName": "avatar.glb"},
            {"role": "canonical_profile", "sourceUrl": "https://cdn.example/profile.json", "fileName": "profile.json"},
        ],
        "parameters": {"task_type": "avatar_reskin", "quality": "runtime", "payment_intent_id": "paper-payment"},
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
    assert set(uploaded) >= {
        "source-mesh.glb", "canonical-profile.json", "canonical-skeleton.glb",
        "avatar.glb", "manifest.json", "diagnostics.json",
    }
    assert any(item.get("status") == "succeeded" for item in callbacks if "runtime" in item)


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


def test_queue_failure_uploads_debug_bundle_before_failure_callback(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "reference.png"
    write_png(source)
    pipeline, _ = make_pipeline(tmp_path, rig_failures=99)
    worker = AvatarWorker(pipeline)
    posts: list[tuple[str, dict]] = []
    uploaded: list[str] = []

    def fake_download(_url: str, _filename: str, _config):
        return source

    def fake_post_json(url: str, _token: str, payload: dict, *, timeout: float = 120):
        posts.append((url, payload))
        return {}

    def fake_upload(_url: str, _token: str, path: Path, artifact: dict) -> str:
        uploaded.append(path.name)
        return f"artifact-{path.name}"

    monkeypatch.setattr(salad_adapter, "_download_reference", fake_download)
    monkeypatch.setattr(salad_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(salad_adapter, "_upload_artifact", fake_upload)
    payload = {
        "runtime": "avatar",
        "job_id": "avatar-failure-paper-job",
        "inputs": [{"role": "reference", "sourceUrl": "https://cdn.example/reference.png", "fileName": "reference.png"}],
        "parameters": {"description": "a paper-run broken humanoid", "quality": "runtime", "seed": 7, "max_attempts": 1},
        "callback": {
            "url": "https://launcher.example/artifacts",
            "complete_url": "https://launcher.example/complete",
            "progress_url": "https://launcher.example/progress",
            "token": "paper-callback-token",
        },
    }
    try:
        with pytest.raises(RuntimeError, match="avatar output failed validation"):
            asyncio.run(salad_adapter._run_queue_job(payload, worker, pipeline.config))
    finally:
        worker.shutdown()

    assert "debug-attempt-1-mesh.glb" in uploaded
    assert "debug-attempt-1-rig.glb" in uploaded
    assert any(url.endswith("/fail") for url, _payload in posts)
    assert any(url.endswith("/progress") and payload.get("status") == "failed" for url, payload in posts)

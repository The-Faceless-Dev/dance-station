from __future__ import annotations

import json

from fastapi.testclient import TestClient

from autotransition.moss_music.artifacts import MossMusicArtifactStore
from autotransition.moss_music.config import MossMusicConfig
from autotransition.moss_music.contracts import MossMusicJob
from autotransition.moss_music.server import create_moss_music_worker_app


def test_mock_server_reports_ready_without_model_or_sglang(tmp_path) -> None:
    config = MossMusicConfig(artifact_root=tmp_path / "jobs", backend="mock", device="cpu", gpu_required=False)
    app = create_moss_music_worker_app(config)
    try:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["runtime"] == "moss-music"
            assert response.json()["ok"] is True
    finally:
        app.state.moss_music_worker.shutdown()


def test_direct_job_artifacts_are_downloadable(tmp_path) -> None:
    config = MossMusicConfig(artifact_root=tmp_path / "jobs", backend="mock", device="cpu", gpu_required=False)
    store = MossMusicArtifactStore(config.artifact_root)
    store.create_job(MossMusicJob(id="artifact-job", status="succeeded", request={}))
    store.finalize_json("artifact-job", "analysis.json", {"schema_version": 1, "runtime": "moss-music"})
    store._atomic_json(
        store.job_dir("artifact-job") / "job.json",
        {
            "id": "artifact-job",
            "status": "succeeded",
            "request": {},
            "artifacts": [store.artifact("artifact-job", "analysis.json", "application/json").__dict__],
        },
    )
    app = create_moss_music_worker_app(config)
    try:
        with TestClient(app) as client:
            job = client.get("/v1/moss/jobs/artifact-job")
            assert job.status_code == 200
            artifact = job.json()["artifacts"][0]
            assert "path" not in artifact
            assert artifact["downloadUrl"] == "/v1/moss/jobs/artifact-job/artifacts/analysis.json"
            response = client.get("/v1/moss/jobs/artifact-job/artifacts/analysis.json")
            assert response.status_code == 200
            assert json.loads(response.text)["runtime"] == "moss-music"
            assert response.headers["content-type"].startswith("application/json")
    finally:
        app.state.moss_music_worker.shutdown()

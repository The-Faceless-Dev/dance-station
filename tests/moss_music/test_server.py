from __future__ import annotations

from fastapi.testclient import TestClient

from autotransition.moss_music.config import MossMusicConfig
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

from pathlib import Path

from fastapi.testclient import TestClient

from autotransition.qwen_image.config import QwenImageConfig
from autotransition.qwen_image.server import create_qwen_image_worker_app


class FakeRuntime:
    def preflight(self):
        return {"ready": True, "runtime": "fake", "transformerQuantization": "Q8_0"}

    def close(self):
        pass


def test_server_exposes_provider_neutral_health_and_status_routes(tmp_path: Path) -> None:
    config = QwenImageConfig(artifact_root=tmp_path, gpu_required=False)
    app = create_qwen_image_worker_app(config, runtime=FakeRuntime())
    with TestClient(app) as client:
        assert client.get("/health").json()["ok"] is True
        assert client.get("/ready").status_code == 200
        status = client.get("/v1/worker/status").json()
        assert status["runtime"] == "qwen-image"
        assert status["activeJobs"] == []

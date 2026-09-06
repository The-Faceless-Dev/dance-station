import asyncio
from pathlib import Path

from PIL import Image

from autotransition.flux_image.config import FluxImageConfig
from autotransition.flux_image.salad_adapter import _run_queue_job
from autotransition.flux_image.worker import FluxImageWorker


class FakeRuntime:
    def preflight(self):
        return {"ready": True, "missing": []}

    def generate(self, request, output_path, progress):
        progress("denoise", 1.0, "done")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (request.width, request.height), "white").save(output_path, format="PNG")
        return {"width": request.width, "height": request.height, "loraCount": 0}


def test_salad_queue_uploads_and_completes(monkeypatch, tmp_path: Path) -> None:
    config = FluxImageConfig(artifact_root=tmp_path)
    worker = FluxImageWorker(config, runtime=FakeRuntime())
    posted = []
    uploaded = []

    def post(url, token, payload, **kwargs):
        posted.append((url, payload))
        return {}

    def upload(url, token, path, artifact):
        uploaded.append((path.name, artifact["name"]))
        return f"artifact-{len(uploaded)}"

    monkeypatch.setattr("autotransition.flux_image.salad_adapter._post_json", post)
    monkeypatch.setattr("autotransition.flux_image.salad_adapter._upload_artifact", upload)
    payload = {
        "runtime": "flux-image",
        "job_id": "queue-job-1",
        "parameters": {"prompt": "test", "seed": 7},
        "callback": {
            "url": "https://launcher.test/artifacts",
            "complete_url": "https://launcher.test/jobs/queue-job-1/complete",
            "progress_url": "https://launcher.test/jobs/queue-job-1/progress",
            "token": "callback-token",
        },
    }
    try:
        result = asyncio.run(_run_queue_job(payload, worker, config))
        assert result["status"] == "succeeded"
        assert {name for _, name in uploaded} == {"image.png", "image-metadata.json"}
        assert any(url.endswith("/complete") for url, _ in posted)
    finally:
        worker.shutdown()

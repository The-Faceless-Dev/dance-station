import time
from pathlib import Path

from PIL import Image

from autotransition.flux_image.config import FluxImageConfig
from autotransition.flux_image.contracts import FluxImageRequest
from autotransition.flux_image.worker import FluxImageWorker


class FakeRuntime:
    def preflight(self):
        return {"ready": True, "missing": []}

    def generate(self, request, output_path, progress):
        progress("load_model", 0.2, "fake model loaded")
        progress("denoise", 0.8, "fake denoise")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (request.width, request.height), (120, 70, 210, 255)).save(output_path, format="PNG")
        return {"width": request.width, "height": request.height, "seed": request.seed, "loraCount": len(request.loras)}


def wait_for(worker: FluxImageWorker, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = worker.get(job_id)
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError("worker did not reach a terminal state")


def test_worker_persists_png_and_metadata(tmp_path: Path) -> None:
    config = FluxImageConfig(artifact_root=tmp_path)
    worker = FluxImageWorker(config, runtime=FakeRuntime())
    try:
        job = __import__("asyncio").run(worker.submit(FluxImageRequest(prompt="test", seed=4)))
        result = wait_for(worker, job.id)
        assert result["status"] == "succeeded"
        assert {item["name"] for item in result["artifacts"]} == {"image.png", "image-metadata.json"}
        assert (tmp_path / job.id / "final" / "image.png").is_file()
        assert (tmp_path / job.id / "events.jsonl").is_file()
    finally:
        worker.shutdown()


def test_worker_external_job_id_is_idempotent(tmp_path: Path) -> None:
    config = FluxImageConfig(artifact_root=tmp_path)
    worker = FluxImageWorker(config, runtime=FakeRuntime())
    try:
        request = FluxImageRequest(prompt="test", external_job_id="external-1", payment_intent_id="payment-1")
        first = __import__("asyncio").run(worker.submit(request))
        second = __import__("asyncio").run(worker.submit(request))
        assert first.id == second.id == "external-1"
    finally:
        worker.shutdown()


def test_worker_persists_failure_diagnostics_without_staying_running(tmp_path: Path) -> None:
    class FailingRuntime(FakeRuntime):
        def generate(self, request, output_path, progress):
            raise RuntimeError("simulated denoise failure")

    config = FluxImageConfig(artifact_root=tmp_path)
    worker = FluxImageWorker(config, runtime=FailingRuntime())
    try:
        job = __import__("asyncio").run(worker.submit(FluxImageRequest(prompt="test")))
        result = wait_for(worker, job.id)
        assert result["status"] == "failed"
        assert result["failureCode"] == "flux_image_worker_failed"
        assert {item["name"] for item in result["artifacts"]} == {"failure-summary.json", "events.jsonl"}
        assert (tmp_path / job.id / "final" / "events.jsonl").is_file()
    finally:
        worker.shutdown()

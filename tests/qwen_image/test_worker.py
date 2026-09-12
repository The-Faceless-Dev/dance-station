import asyncio
import time
from pathlib import Path

from PIL import Image

from autotransition.qwen_image.config import QwenImageConfig
from autotransition.qwen_image.contracts import QwenImageRequest
from autotransition.qwen_image.worker import QwenImageWorker


class FakeRuntime:
    def __init__(self, *, fail: bool = False):
        self.emit = lambda *_args, **_kwargs: None
        self.fail = fail

    def preflight(self):
        return {"ready": True, "runtime": "fake", "transformerQuantization": "Q8_0"}

    def generate(self, request, lora_paths, output_path, progress):
        if self.fail:
            raise RuntimeError("fake denoise failure")
        progress("denoise", 0.5, "fake denoise")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (request.width, request.height), (80, 100, 120)).save(output_path, format="PNG")
        return {"nativeJobId": "fake-1", "loraCount": len(lora_paths)}

    def close(self):
        pass


def wait_for(worker: QwenImageWorker, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = worker.get(job_id)
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError("worker did not reach a terminal state")


def test_worker_persists_full_success_artifacts(tmp_path: Path) -> None:
    worker = QwenImageWorker(QwenImageConfig(artifact_root=tmp_path), runtime=FakeRuntime())
    try:
        job = asyncio.run(worker.submit(QwenImageRequest(prompt="test", width=256, height=256)))
        result = wait_for(worker, job.id)
        assert result["status"] == "succeeded"
        assert {item["name"] for item in result["artifacts"]} == {
            "image.png", "image-metadata.json", "preflight.json", "events.jsonl",
        }
        assert (tmp_path / job.id / "final" / "image.png").is_file()
        assert not (tmp_path / job.id / "tmp" / "loras").exists()
    finally:
        worker.shutdown()


def test_worker_persists_failure_without_staying_running(tmp_path: Path) -> None:
    worker = QwenImageWorker(QwenImageConfig(artifact_root=tmp_path), runtime=FakeRuntime(fail=True))
    try:
        job = asyncio.run(worker.submit(QwenImageRequest(prompt="test")))
        result = wait_for(worker, job.id)
        assert result["status"] == "failed"
        assert result["failureCode"] == "qwen_image_worker_failed"
        assert {item["name"] for item in result["artifacts"]} == {"failure-summary.json", "preflight.json", "events.jsonl"}
    finally:
        worker.shutdown()

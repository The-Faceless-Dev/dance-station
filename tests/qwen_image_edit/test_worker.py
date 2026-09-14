import asyncio
import time
from pathlib import Path

from PIL import Image

from autotransition.qwen_image_edit.config import QwenImageEditConfig
from autotransition.qwen_image_edit.contracts import request_from_payload
from autotransition.qwen_image_edit.worker import QwenImageEditWorker


class FakeRuntime:
    def __init__(self) -> None:
        self.emit = lambda *_args, **_kwargs: None

    def preflight(self):
        return {"ready": True, "runtime": "diffusers", "transformerQuantization": "Q8_0"}

    def generate(self, request, references, loras, output_path, progress):
        assert len(references) == 2
        assert [item.scale for item in loras] == [0.6, 1.1]
        progress("denoise", 0.5, "fake edit denoise")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (request.width, request.height), (80, 100, 120)).save(output_path, format="PNG")
        return {"runtime": "fake", "referenceCount": len(references), "loraCount": len(loras)}

    def close(self):
        pass


def wait_for(worker: QwenImageEditWorker, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = worker.get(job_id)
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError("worker did not reach a terminal state")


def test_worker_persists_multi_reference_success(tmp_path: Path) -> None:
    config = QwenImageEditConfig(artifact_root=tmp_path, allow_local_references=True, allow_local_loras=True)
    worker = QwenImageEditWorker(config, runtime=FakeRuntime())
    reference_a = tmp_path / "a.png"
    reference_b = tmp_path / "b.png"
    Image.new("RGB", (32, 32), "red").save(reference_a)
    Image.new("RGB", (32, 32), "blue").save(reference_b)
    try:
        request = request_from_payload(
            {
                "job_id": "edit-test",
                "prompt": "combine the subjects",
                "width": 256,
                "height": 256,
                "reference_images": [
                    {"path": str(reference_a), "fileName": "a.png"},
                    {"path": str(reference_b), "fileName": "b.png"},
                ],
                "loras": [
                    {"path": str(reference_a), "fileName": "one.safetensors", "scale": 0.6},
                    {"path": str(reference_b), "fileName": "two.safetensors", "scale": 1.1},
                ],
            },
            config,
        )
        result = wait_for(worker, asyncio.run(worker.submit(request))["id"])
        assert result["status"] == "succeeded"
        assert (tmp_path / "edit-test" / "final" / "image.png").is_file()
        assert (tmp_path / "edit-test" / "final" / "reference-metadata.json").is_file()
        assert not (tmp_path / "edit-test" / "tmp" / "references").exists()
        assert not (tmp_path / "edit-test" / "tmp" / "loras").exists()
    finally:
        worker.shutdown()

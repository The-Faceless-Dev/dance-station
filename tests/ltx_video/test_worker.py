from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from autotransition.ltx_video.config import LtxVideoConfig
from autotransition.ltx_video.contracts import LtxVideoRequest
from autotransition.ltx_video.runtime import LtxRuntimeResult
from autotransition.ltx_video.worker import LtxVideoWorker


class FakeRuntime:
    def preflight(self) -> dict[str, object]:
        return {"runtime": "ltx-video", "ready": True}

    def generate(self, request, output_dir: Path, progress):  # type: ignore[no-untyped-def]
        progress("load_models", 0.5, "fake model loaded", {"fake": True})
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "output.mp4"
        output.write_bytes(b"fake-video")
        metadata = {"runtime": "ltx-video", "memoryPlan": {"estimatedPeakGb": 1.0}}
        (output_dir / "generation-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        progress("finalize", 1.0, "fake generation complete")
        return LtxRuntimeResult(output, None, metadata)


class NotReadyRuntime:
    def preflight(self) -> dict[str, object]:
        return {"runtime": "ltx-video", "ready": False, "reason": "missing model components"}


def _wait(worker: LtxVideoWorker, job_id: str) -> dict[str, object]:
    for _ in range(100):
        job = worker.get(job_id)
        if job.get("status") in {"succeeded", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("fake LTX job did not reach a terminal state")


def test_worker_persists_success_artifacts(tmp_path: Path) -> None:
    config = LtxVideoConfig(artifact_root=tmp_path / "jobs")
    worker = LtxVideoWorker(config, runtime=FakeRuntime())
    try:
        import asyncio

        request = LtxVideoRequest(prompt="a dancer", num_frames=25, external_job_id="job-1")
        job = asyncio.run(worker.submit(request))
        result = _wait(worker, job.id)
        assert result["status"] == "succeeded"
        names = {item["name"] for item in result["artifacts"]}  # type: ignore[index]
        assert {"output.mp4", "generation-metadata.json", "request.json", "memory-plan.json", "events.jsonl"} <= names
        assert (tmp_path / "jobs" / "job-1" / "final" / "output.mp4").read_bytes() == b"fake-video"
        events = [
            json.loads(line)
            for line in (tmp_path / "jobs" / "job-1" / "final" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        progress = [event for event in events if event.get("event") == "progress"]
        assert progress
        assert [event["progress"] for event in progress] == sorted(event["progress"] for event in progress)
        assert all("stageProgress" in event for event in progress)
    finally:
        worker.shutdown()


def test_worker_rejects_jobs_before_model_readiness(tmp_path: Path) -> None:
    import asyncio
    from fastapi import HTTPException

    worker = LtxVideoWorker(LtxVideoConfig(artifact_root=tmp_path / "jobs"), runtime=NotReadyRuntime())
    try:
        with pytest.raises(HTTPException) as error:
            asyncio.run(worker.submit(LtxVideoRequest(prompt="a dancer", num_frames=25)))
        assert error.value.status_code == 503
        assert "missing model components" in str(error.value.detail)
    finally:
        worker.shutdown()

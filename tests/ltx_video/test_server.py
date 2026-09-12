from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from autotransition.ltx_video.config import LtxVideoConfig
from autotransition.ltx_video.runtime import LtxRuntimeResult
from autotransition.ltx_video.server import create_ltx_video_worker_app


class ServerFakeRuntime:
    def __init__(self) -> None:
        self.reset_calls = 0

    def preflight(self) -> dict[str, object]:
        return {"runtime": "ltx-video", "ready": True, "attention": {"flashSdp": True}}

    def residency_status(self) -> dict[str, object]:
        return {"requiresReset": False, "memory": {"allocatedGb": 0.0}}

    def reset_residency(self) -> dict[str, object]:
        self.reset_calls += 1
        return {"releasedAllocatedGb": 1.0, "requiresProcessRestart": False}

    def generate(self, request, output_dir: Path, progress):  # type: ignore[no-untyped-def]
        output_dir.mkdir(parents=True, exist_ok=True)
        progress("stage_1_denoise", 0.5, "fake denoise step 4/8", {"completedSteps": 4, "totalSteps": 8})
        output = output_dir / "output.mp4"
        output.write_bytes(b"fake-server-video")
        metadata = {"runtime": "ltx-video", "memoryPlan": {"estimatedPeakGb": 1.0}}
        (output_dir / "generation-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        progress("encode_output", 1.0, "fake output encoded")
        return LtxRuntimeResult(output, None, metadata)


def test_direct_http_lifecycle_hides_local_artifact_paths(tmp_path: Path) -> None:
    config = LtxVideoConfig(artifact_root=tmp_path / "jobs")
    runtime = ServerFakeRuntime()
    app = create_ltx_video_worker_app(config, runtime=runtime)
    worker = app.state.ltx_video_worker
    try:
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["ok"] is True
            reset = client.post("/v1/worker/reset")
            assert reset.status_code == 200
            assert reset.json()["reset"]["requiresProcessRestart"] is False
            assert runtime.reset_calls == 1

            submitted = client.post(
                "/v1/ltx/jobs",
                json={"job_id": "server-job", "parameters": {"prompt": "a dancer", "num_frames": 25}},
            )
            assert submitted.status_code == 200
            assert submitted.json()["id"] == "server-job"

            for _ in range(100):
                result = client.get("/v1/ltx/jobs/server-job")
                assert result.status_code == 200
                payload = result.json()
                if payload["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.01)
            assert payload["status"] == "succeeded"
            artifact = next(item for item in payload["artifacts"] if item["name"] == "output.mp4")
            assert "path" not in artifact
            download = client.get(artifact["downloadUrl"])
            assert download.status_code == 200
            assert download.content == b"fake-server-video"
    finally:
        worker.shutdown()

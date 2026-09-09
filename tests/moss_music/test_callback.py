from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import numpy as np

from autotransition.moss_music.callback import run_queue_job
from autotransition.moss_music.config import MossMusicConfig
from autotransition.moss_music.runtime import MossRuntimeResult
from autotransition.moss_music.worker import MossMusicWorker


def _write_wav(path: Path) -> None:
    values = (0.2 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(values.tobytes())


class FakeRuntime:
    def preflight(self):
        return {"ready": True, "backend": "mock"}

    def analyze(self, request, audio, progress):
        progress(1.0, "fake response")
        response = {"summary": "callback test", "events": [{"type": "hit", "time_seconds": 0.04}]}
        return MossRuntimeResult(response=response, raw_text=json.dumps(response), metadata={"backend": "mock"})


def test_callback_uploads_success_artifacts_and_completes(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source)
    config = MossMusicConfig(artifact_root=tmp_path / "jobs", backend="mock", device="cpu", gpu_required=False, allow_local_audio_paths=True)
    worker = MossMusicWorker(config, runtime=FakeRuntime())
    posted: list[tuple[str, dict]] = []
    uploaded: list[str] = []

    def post(url, token, payload, **kwargs):
        posted.append((url, payload))
        return {}

    def upload(url, token, path, artifact):
        uploaded.append(artifact["name"])
        return f"artifact-{len(uploaded)}"

    monkeypatch.setattr("autotransition.moss_music.callback._post_json", post)
    monkeypatch.setattr("autotransition.moss_music.callback._upload_artifact", upload)
    payload = {
        "runtime": "moss-music",
        "job_id": "callback-job-1",
        "parameters": {"audio": {"path": str(source), "filename": "source.wav"}},
        "callback": {
            "url": "https://launcher.test/artifacts",
            "complete_url": "https://launcher.test/jobs/callback-job-1/complete",
            "progress_url": "https://launcher.test/jobs/callback-job-1/progress",
            "token": "callback-token",
        },
    }
    try:
        result = asyncio.run(run_queue_job(payload, worker, config))
        assert result["status"] == "succeeded"
        assert "analysis.json" in uploaded
        assert any(url.endswith("/complete") for url, _ in posted)
        assert any(body.get("runtime") == "moss-music" for _, body in posted)
    finally:
        worker.shutdown()

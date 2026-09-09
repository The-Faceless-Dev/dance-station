from __future__ import annotations

import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from autotransition.moss_music.audio import normalize_audio
from autotransition.moss_music.config import MossMusicConfig
from autotransition.moss_music.contracts import MossAudioInput, MossMusicRequest
from autotransition.moss_music.parser import MossResponseError, parse_moss_response
from autotransition.moss_music.runtime import MossRuntimeResult
from autotransition.moss_music.runtime import SGLangMossClient
from autotransition.moss_music.timeline import build_dense_timeline
from autotransition.moss_music.worker import MossMusicWorker


def _write_wav(path: Path, duration: float = 0.24) -> None:
    sample_rate = 16000
    count = round(sample_rate * duration)
    values = (0.25 * np.sin(2 * np.pi * 440 * np.arange(count) / sample_rate) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(values.tobytes())


def _request(source: Path) -> MossMusicRequest:
    return MossMusicRequest(audio=MossAudioInput(filename=source.name, path=source))


def _wait_for(worker: MossMusicWorker, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = worker.get(job_id)
        if current["status"] in {"succeeded", "failed"}:
            return current
        time.sleep(0.02)
    raise AssertionError("MOSS-Music worker did not reach a terminal state")


class FakeRuntime:
    def preflight(self):
        return {"ready": True, "backend": "mock"}

    def analyze(self, request, audio, progress):
        progress(0.5, "fake MOSS analysis")
        response = {
            "summary": "A test song",
            "tempo_bpm": 120,
            "sections": [{"label": "intro", "start_seconds": 0, "end_seconds": 0.24, "confidence": 0.9}],
            "beats": [{"time_seconds": 0.08, "strength": 0.8, "confidence": 0.7}],
            "events": [{"type": "kick", "start_seconds": 0.08, "end_seconds": 0.10, "strength": 0.8, "confidence": 0.7}],
            "warnings": [],
        }
        return MossRuntimeResult(response=response, raw_text=json.dumps(response), metadata={"backend": "mock"})


def test_parser_normalizes_fenced_json_and_timed_collections() -> None:
    parsed, raw = parse_moss_response('```json\n{"beats":[{"time_seconds":0.08,"strength":0.5}]}\n```')
    assert raw.startswith("```json")
    assert parsed["beats"][0]["start_seconds"] == 0.08
    assert parsed["beats"][0]["end_seconds"] == 0.08
    assert parsed["events"][0]["type"] == "beat"


def test_parser_rejects_non_json() -> None:
    with pytest.raises(MossResponseError):
        parse_moss_response("MOSS says the song is energetic")


def test_sglang_client_sends_official_audio_request_shape(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class Response:
        status_code = 200
        text = '{"text":"{}"}'

        def json(self):
            return {"text": "{}"}

    class Client:
        def __init__(self, **kwargs):
            seen["timeout"] = kwargs["timeout"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            seen["url"] = url
            seen["payload"] = json
            return Response()

    monkeypatch.setattr("autotransition.moss_music.runtime.httpx.Client", Client)
    config = MossMusicConfig(backend="sglang", device="cuda", gpu_required=True, check_backend_on_preflight=False)
    audio = tmp_path / "normalized.wav"
    audio.write_bytes(b"audio")
    response = SGLangMossClient(config).generate(prompt="return json", audio_path=audio, max_new_tokens=128, temperature=0)
    assert response == {"text": "{}"}
    assert seen["url"] == "http://127.0.0.1:30000/generate"
    assert seen["payload"] == {
        "text": "return json",
        "audio_data": str(audio),
        "sampling_params": {"max_new_tokens": 128, "temperature": 0},
    }


def test_dense_timeline_covers_full_audio_at_80ms(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source)
    audio = normalize_audio(source, tmp_path / "normalized")
    timeline = build_dense_timeline(audio, resolution_ms=80)
    assert timeline["cell_count"] == 3
    assert timeline["cells"][0]["start_seconds"] == 0
    assert timeline["cells"][-1]["end_seconds"] == pytest.approx(0.24, abs=1e-5)
    assert all("chroma" in cell and len(cell["chroma"]) == 12 for cell in timeline["cells"])
    assert all(
        timeline["cells"][index]["end_seconds"] == pytest.approx(timeline["cells"][index + 1]["start_seconds"], abs=1e-6)
        for index in range(len(timeline["cells"]) - 1)
    )


def test_worker_persists_analysis_and_all_debug_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source)
    config = MossMusicConfig(artifact_root=tmp_path / "jobs", backend="mock", device="cpu", gpu_required=False, allow_local_audio_paths=True)
    worker = MossMusicWorker(config, runtime=FakeRuntime())
    try:
        job = asyncio.run(worker.submit(_request(source)))
        result = _wait_for(worker, job.id)
        assert result["status"] == "succeeded"
        names = {item["name"] for item in result["artifacts"]}
        assert {"analysis.json", "moss-response.json", "moss-raw.txt", "events.jsonl"}.issubset(names)
        analysis = json.loads((tmp_path / "jobs" / job.id / "final" / "analysis.json").read_text())
        assert analysis["timebase"]["resolution_ms"] == 80
        assert analysis["dense_features"]
        assert analysis["beats"][0]["start_seconds"] == 0.08
    finally:
        worker.shutdown()


def test_worker_turns_invalid_model_output_into_failure_with_raw_response(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source)

    class InvalidRuntime(FakeRuntime):
        def analyze(self, request, audio, progress):
            return MossRuntimeResult(response="not json", raw_text="not json", metadata={"backend": "mock"})

    config = MossMusicConfig(artifact_root=tmp_path / "jobs", backend="mock", device="cpu", gpu_required=False, allow_local_audio_paths=True)
    worker = MossMusicWorker(config, runtime=InvalidRuntime())
    try:
        job = asyncio.run(worker.submit(_request(source)))
        result = _wait_for(worker, job.id)
        assert result["status"] == "failed"
        assert result["failureCode"] == "moss_music_worker_failed"
        assert (tmp_path / "jobs" / job.id / "final" / "moss-raw.txt").read_text() == "not json"
        assert (tmp_path / "jobs" / job.id / "final" / "failure-summary.json").is_file()
    finally:
        worker.shutdown()


def test_worker_rejects_an_unbounded_run(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source)

    class SlowRuntime(FakeRuntime):
        def analyze(self, request, audio, progress):
            time.sleep(0.03)
            progress(1.0, "late response")
            return super().analyze(request, audio, progress)

    config = MossMusicConfig(
        artifact_root=tmp_path / "jobs",
        backend="mock",
        device="cpu",
        gpu_required=False,
        allow_local_audio_paths=True,
        job_timeout_seconds=0.001,
    )
    worker = MossMusicWorker(config, runtime=SlowRuntime())
    try:
        job = asyncio.run(worker.submit(_request(source)))
        result = _wait_for(worker, job.id)
        assert result["status"] == "failed"
        assert result["failure"]["details"]["errorType"] == "TimeoutError"
    finally:
        worker.shutdown()

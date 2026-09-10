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
from autotransition.moss_music.parser import merge_moss_passes
from autotransition.moss_music.prompting import build_analysis_prompts
from autotransition.moss_music.runtime import MossRuntimeResult
from autotransition.moss_music.runtime import SGLangMossClient, SGLangMossRuntime
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


def test_parser_keeps_valid_fields_when_backend_reports_length_stop() -> None:
    parsed, _ = parse_moss_response({
        "text": '{"summary":"partial","warnings":[]}',
        "meta_info": {"finish_reason": {"type": "length"}},
    })
    assert parsed["summary"] == "partial"
    assert any("output-length stop" in warning for warning in parsed["warnings"])


def test_parser_does_not_accept_a_nested_object_from_truncated_output() -> None:
    with pytest.raises(MossResponseError):
        parse_moss_response('{"summary":"partial", "sections": [{"label":"intro"')


def test_parser_recovers_complete_timed_records_from_truncated_pass() -> None:
    parsed, _ = parse_moss_response(
        '{"visual_events":[{"start":0,"end":1,"type":"drop","intensity":0.9},'
        '{"start":2,"end":3,"type":"rise"',
        required_keys={"visual_events", "warnings"},
    )
    assert len(parsed["visual_events"]) == 1
    assert parsed["visual_events"][0]["type"] == "drop"
    assert any("Recovered 1 complete timed record" in warning for warning in parsed["warnings"])


def test_parser_accepts_json_wrapped_by_model_preamble() -> None:
    parsed, _ = parse_moss_response('Here is the requested JSON:\n{"beats": [], "warnings": []}\nDone.')
    assert parsed["beats"] == []


def test_parser_retains_valid_pass_fields_when_optional_keys_are_missing() -> None:
    parsed, _ = parse_moss_response(
        '{"key":"B minor","chords":[{"chord":"Bm","start":0,"end":2}]}',
        required_keys={"key", "chords", "events", "warnings"},
    )
    assert parsed["key"] == "B minor"
    assert parsed["chords"][0]["type"] == "chord"
    assert parsed["event_count"] == 1
    assert any("omitted optional pass" in warning for warning in parsed["warnings"])


def test_analysis_pass_prompts_only_request_their_own_fields() -> None:
    request = MossMusicRequest(audio=MossAudioInput(source_url="https://example.test/song.mp3", filename="song.mp3"))
    prompts = {item.name: item.instruction for item in build_analysis_prompts(request)}
    assert '"beats": []' not in prompts["overview"]
    assert '"sections": []' not in prompts["rhythm"]
    assert 'exactly the keys shown below' in prompts["harmony"]
    assert 'Do not emit a regular beat grid' in prompts["rhythm"]
    assert 'visual_events' in prompts["rhythm"]


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


def test_sglang_client_uses_checkpoint_context_for_output_budget(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class Response:
        status_code = 200
        text = '{"text":"{}"}'

        def json(self):
            return {"text": "{}"}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            seen["payload"] = json
            return Response()

    monkeypatch.setattr("autotransition.moss_music.runtime.httpx.Client", Client)
    config = MossMusicConfig(
        backend="sglang",
        device="cuda",
        gpu_required=True,
        check_backend_on_preflight=False,
        model_context_length=40960,
        model_root=tmp_path / "model",
    )
    config.model_root.mkdir()
    (config.model_root / "config.json").write_text(json.dumps({"language_config": {"max_position_embeddings": 40960}}))
    audio = tmp_path / "normalized.wav"
    _write_wav(audio, duration=1.0)
    budget = SGLangMossClient(config).resolve_output_budget(prompt="return json", audio_path=audio)
    SGLangMossClient(config).generate(prompt="return json", audio_path=audio, max_new_tokens=budget["availableOutputTokens"], temperature=0)
    assert seen["payload"] == {
        "text": "return json",
        "audio_data": str(audio),
        "sampling_params": {"max_new_tokens": budget["availableOutputTokens"], "temperature": 0},
    }


def test_context_budget_recovers_from_sglang_measured_input() -> None:
    budget = {"contextLength": 40960, "availableOutputTokens": 37551}
    recovered = SGLangMossClient.recover_output_budget(
        "Requested token count exceeds the model's maximum context length of 40960 tokens. "
        "You requested a total of 41280 tokens: 3729 tokens from the input messages and 37551 tokens for the completion.",
        budget,
    )
    assert recovered is not None
    assert recovered["backendInputTokens"] == 3729
    assert recovered["availableOutputTokens"] == 37230
    assert recovered["budgetRecoveredFromBackendContextError"] is True


def test_request_has_no_caller_token_budget() -> None:
    request = MossMusicRequest(audio=MossAudioInput(source_url="https://example.test/song.mp3", filename="song.mp3"))
    assert "max_new_tokens" not in request.to_dict()


def test_merge_moss_passes_preserves_pass_provenance() -> None:
    merged = merge_moss_passes([
        ("rhythm", {"summary": "", "tempo_bpm": 120, "time_signature": None, "key": None, "sections": [], "beats": [{"id": "moss-beat-000001", "start_seconds": 0.08, "end_seconds": 0.08}], "chords": [], "lyrics": [], "instruments": [], "voices": [], "visual_cues": [], "visual_events": [{"id": "moss-visual-000001", "start_seconds": 0.08, "end_seconds": 0.4, "type": "drop", "intensity": 0.9}], "events": [], "warnings": [], "event_count": 1}),
        ("harmony", {"summary": "", "tempo_bpm": None, "time_signature": None, "key": "C minor", "sections": [], "beats": [], "chords": [{"id": "moss-chord-000001", "start_seconds": 0, "end_seconds": 1, "type": "Cm"}], "lyrics": [], "instruments": [], "voices": [], "visual_cues": [], "visual_events": [], "events": [], "warnings": [], "event_count": 1}),
    ])
    assert merged["passes"] == ["rhythm", "harmony"]
    assert merged["beats"][0]["source_pass"] == "rhythm"
    assert merged["chords"][0]["source_pass"] == "harmony"
    assert merged["visual_events"][0]["source_pass"] == "rhythm"


def test_sglang_runtime_runs_all_focused_passes_without_request_token_override(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source)
    audio = normalize_audio(source, tmp_path / "normalized")
    calls: list[dict[str, object]] = []

    class FakeClient:
        def resolve_output_budget(self, *, prompt, audio_path):
            return {"contextLength": 40960, "promptTokens": 100, "audioTokens": 3, "availableOutputTokens": 40857}

        def generate(self, *, prompt, audio_path, max_new_tokens, temperature):
            calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens})
            return {
                "text": json.dumps({
                    "summary": "pass",
                    "tempo_bpm": 120,
                    "time_signature": "4/4",
                    "key": "C",
                    "sections": [],
                    "beats": [],
                    "chords": [],
                    "lyrics": [],
                    "instruments": [],
                    "voices": [],
                    "visual_cues": [],
                    "events": [],
                    "warnings": [],
                }),
                "meta_info": {"finish_reason": {"type": "stop"}},
            }

    config = MossMusicConfig(model_root=tmp_path / "model", backend="sglang", device="cuda", gpu_required=True)
    result = SGLangMossRuntime(config, client=FakeClient()).analyze(_request(source), audio, lambda *_: None)
    assert len(calls) == 4
    assert set(result.responses) == {"overview", "rhythm", "harmony", "lyrics_and_voices"}
    assert [call["max_new_tokens"] for call in calls] == [1536, 1536, 1024, 2048]


def test_sglang_runtime_segments_long_harmony_pass_and_offsets_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source, duration=0.24)
    audio = normalize_audio(source, tmp_path / "normalized")
    calls: list[dict[str, object]] = []

    class FakeClient:
        def resolve_output_budget(self, *, prompt, audio_path):
            return {"contextLength": 40960, "promptTokens": 100, "audioTokens": 3, "availableOutputTokens": 40857}

        def generate(self, *, prompt, audio_path, max_new_tokens, temperature):
            calls.append({"prompt": prompt, "audio_path": audio_path, "max_new_tokens": max_new_tokens})
            return {
                "text": json.dumps({
                    "summary": "segment",
                    "tempo_bpm": 120,
                    "time_signature": "4/4",
                    "key": "C",
                    "sections": [],
                    "beats": [],
                    "chords": [{"type": "C", "start_seconds": 0, "end_seconds": 0.05}],
                    "lyrics": [],
                    "instruments": [],
                    "voices": [],
                    "visual_cues": [],
                    "events": [],
                    "warnings": [],
                }),
                "meta_info": {"finish_reason": {"type": "stop"}},
            }

    config = MossMusicConfig(
        model_root=tmp_path / "model",
        backend="sglang",
        device="cuda",
        gpu_required=True,
        semantic_window_seconds=0.1,
        min_semantic_window_seconds=0.05,
    )
    result = SGLangMossRuntime(config, client=FakeClient()).analyze(_request(source), audio, lambda *_: None)
    assert len(calls) == 12
    harmony = result.responses["harmony"]
    assert [item["start_seconds"] for item in harmony["chords"]] == [0.0, 0.1, 0.2]
    assert all(item["outputBudget"]["mode"] == "segmented" for item in result.metadata["passes"])


def test_segment_results_are_bounded_before_merge(tmp_path: Path) -> None:
    parsed = {
        "sections": [{"start_seconds": 0, "end_seconds": 2}],
        "beats": [{"start_seconds": 9, "end_seconds": 9}],
        "chords": [],
        "lyrics": [],
        "instruments": [],
        "voices": [],
        "visual_cues": [],
        "events": [{"start_seconds": 1, "end_seconds": 4}],
        "warnings": [],
        "event_count": 1,
    }
    bounded = SGLangMossRuntime._restrict_segment(parsed, 3)
    assert bounded["beats"] == []
    assert bounded["events"][0]["end_seconds"] == 3
    assert bounded["event_count"] == 1


def test_segment_visual_events_are_deduplicated_and_capped(tmp_path: Path) -> None:
    events = [
        {"type": f"event-{index}", "start_seconds": index, "end_seconds": index + 0.1, "intensity": index / 20}
        for index in range(20)
    ]
    bounded = SGLangMossRuntime._restrict_segment({"visual_events": events, "warnings": []}, 30)
    assert len(bounded["visual_events"]) == 12
    assert any("Limited visual_events" in warning for warning in bounded["warnings"])


def test_segment_parse_error_keeps_raw_text_without_retry(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source, duration=0.12)
    audio = normalize_audio(source, tmp_path / "normalized")
    calls: list[str] = []

    class FakeClient:
        def resolve_output_budget(self, *, prompt, audio_path):
            return {"contextLength": 40960, "promptTokens": 100, "audioTokens": 3, "availableOutputTokens": 40857}

        def generate(self, *, prompt, audio_path, max_new_tokens, temperature):
            calls.append(prompt)
            if len(calls) == 1:
                return {"text": '{"summary":"bad", "sections": [{"end": 1.000"}]}' }
            return {
                "text": json.dumps({
                    "summary": "retry",
                    "tempo_bpm": 120,
                    "time_signature": "4/4",
                    "key": "C",
                    "sections": [],
                    "beats": [],
                    "chords": [],
                    "lyrics": [],
                    "instruments": [],
                    "voices": [],
                    "visual_cues": [],
                    "events": [],
                    "warnings": [],
                }),
                "meta_info": {"finish_reason": {"type": "stop"}},
            }

    config = MossMusicConfig(
        model_root=tmp_path / "model",
        backend="sglang",
        device="cuda",
        gpu_required=True,
        semantic_window_seconds=0.1,
        min_semantic_window_seconds=0.05,
    )
    runtime = SGLangMossRuntime(config, client=FakeClient())
    result = runtime.analyze(_request(source), audio, lambda *_: None)
    assert len(calls) == 8
    assert not any("The previous response was malformed" in prompt for prompt in calls)
    assert '{"summary":"bad"' in result.raw_texts["overview"]
    assert result.metadata["passes"][0]["segments"][0]["parseError"]


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


def test_worker_returns_invalid_model_output_with_raw_response(tmp_path: Path) -> None:
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
        assert result["status"] == "succeeded"
        assert (tmp_path / "jobs" / job.id / "final" / "moss-raw.txt").read_text() == "not json"
        analysis = json.loads((tmp_path / "jobs" / job.id / "final" / "analysis.json").read_text())
        assert analysis["warnings"]
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

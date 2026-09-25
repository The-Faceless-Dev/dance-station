from __future__ import annotations

import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import pytest

from autotransition.mulacover.artifacts import MuLaCoverArtifactStore
from autotransition.mulacover.callback import request_from_payload
from autotransition.mulacover.config import MuLaCoverConfig
from autotransition.mulacover.contracts import MuLaCoverRequest
from autotransition.mulacover.runtime import MuLaCoverRuntime, MuLaCoverRuntimeResult
from autotransition.mulacover.server import create_mulacover_worker_app
from autotransition.mulacover.worker import MuLaCoverWorker


def _config(tmp_path: Path, *, allow_local: bool = False) -> MuLaCoverConfig:
    return MuLaCoverConfig(
        artifact_root=tmp_path / "jobs",
        model_root=tmp_path / "models",
        gpu_required=False,
        allow_local_inputs=allow_local,
        minimum_free_vram_gb=1,
        recommended_vram_gb=1,
    )


def test_request_accepts_reference_audio_and_exposes_controls() -> None:
    request = request_from_payload({
        "job_id": "job-a",
        "lyrics": "[Verse]\nnew words",
        "tags": "topic:[hope]; genre:[pop]; instrument:[piano]; mood:[warm]",
        "ref_audio": {"url": "https://example.test/reference.mp3"},
        "bpm": 120,
        "semitoneShift": -2,
        "octaveShift": 1,
        "durationSeconds": 18,
        "cfgScale": 1.7,
        "temperature": 0.9,
        "topK": 300,
        "seed": 42,
        "decodeSeed": 43,
        "outputFormat": "flac",
    })
    request.validate(_config(Path(".")))
    assert request.conditioning_mode == "reference_audio"
    assert request.semitone_shift == -2
    assert request.octave_shift == 1
    assert request.output_format == "flac"
    assert request.to_dict()["ref_audio_url"] == "https://example.test/reference.mp3"


def test_reference_audio_can_omit_lyrics_for_automatic_transcription() -> None:
    request = request_from_payload({
        "job_id": "job-auto-lyrics",
        "tags": "genre:[techno]",
        "ref_audio": {"url": "https://example.test/reference.mp3"},
    })
    request.validate(_config(Path(".")))
    assert request.lyrics == ""
    assert request.conditioning_mode == "reference_audio"


def test_reference_audio_url_string_is_accepted_for_automatic_transcription() -> None:
    request = request_from_payload({
        "job_id": "job-auto-lyrics-url",
        "tags": "genre:[techno]",
        "ref_audio_url": "https://example.test/reference.mp3",
    })
    request.validate(_config(Path(".")))
    assert request.ref_audio_url == "https://example.test/reference.mp3"
    assert request.lyrics == ""


def test_midi_conditioning_still_requires_explicit_lyrics() -> None:
    with pytest.raises(ValueError, match="lyrics"):
        request_from_payload({
            "job_id": "job-midi-no-lyrics",
            "tags": "genre:[techno]",
            "melody_midi": {"url": "https://example.test/melody.mid"},
            "chord_midi": {"url": "https://example.test/chord.mid"},
        })


def test_reference_audio_lyrics_are_extracted_and_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWord:
        word = " hello"
        start = 0.1
        end = 0.4
        probability = 0.98

    class FakeSegment:
        start = 0.0
        end = 0.5
        text = " Hello"
        avg_logprob = -0.1
        words = [FakeWord()]

    class FakeInfo:
        language = "en"
        language_probability = 0.99

    class FakeWhisperModel:
        def __init__(self, model, *, device, compute_type):
            assert model == "small"
            assert device == "cpu"
            assert compute_type == "int8"

        def transcribe(self, path, **kwargs):
            assert Path(path).name == "reference-audio.wav"
            assert kwargs["word_timestamps"] is True
            return iter([FakeSegment()]), FakeInfo()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    config = _config(tmp_path, allow_local=True)
    config = replace(config, device="cpu")
    runtime = __import__("autotransition.mulacover.runtime", fromlist=["MuLaCoverRuntime"]).MuLaCoverRuntime(config)
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    request = MuLaCoverRequest(lyrics="", tags="genre:[techno]", ref_audio_path=source)
    values, manifest = runtime.acquire_inputs(request, tmp_path / "attempt")
    assert Path(values["lyrics"]).read_text(encoding="utf-8") == "Hello"
    assert manifest["lyrics"]["source"] == "faster-whisper"
    assert manifest["lyrics"]["segments"][0]["words"][0]["word"] == "hello"


def test_request_accepts_midi_and_rejects_mixed_modes() -> None:
    config = _config(Path("."))
    midi = MuLaCoverRequest(
        lyrics="words", tags="genre:[pop]", melody_midi_url="https://example.test/melody.mid", chord_midi_url="https://example.test/chords.mid"
    )
    midi.validate(config)
    assert midi.conditioning_mode == "midi"

    mixed = MuLaCoverRequest(
        lyrics="words", tags="genre:[pop]", ref_audio_url="https://example.test/ref.mp3", melody_midi_url="https://example.test/melody.mid", chord_midi_url="https://example.test/chords.mid"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        mixed.validate(config)


def test_local_reference_path_is_preserved_for_development_mode() -> None:
    request = request_from_payload({
        "job_id": "local-job",
        "lyrics": "words",
        "tags": "genre:[pop]",
        "ref_audio_path": "D:/testing/reference.wav",
    })
    assert request.ref_audio_url == ""
    assert request.ref_audio_path == Path("D:/testing/reference.wav")
    request.validate(_config(Path("."), allow_local=True))


def test_preflight_reports_missing_bundle_and_gpu_requirements(tmp_path: Path) -> None:
    report = _config(tmp_path).preflight()
    assert report["runtime"] == "mulacover"
    assert report["ready"] is False
    assert "MuLaCover/config.json" in report["missing"]


def test_pipeline_kwargs_follow_installed_signature() -> None:
    def current_forward(inputs, *, max_audio_length_ms, disable_progress=False):
        return inputs, max_audio_length_ms, disable_progress

    values = {
        "max_audio_length_ms": 1000,
        "disable_progress": True,
        "cancelled": lambda: False,
        "on_progress": lambda *_args: None,
        "decode_seed": 42,
    }
    assert MuLaCoverRuntime._supported_pipeline_kwargs(current_forward, values) == {
        "max_audio_length_ms": 1000,
        "disable_progress": True,
    }


def test_pipeline_kwargs_preserve_var_keyword_hooks() -> None:
    def compatible_forward(inputs, **kwargs):
        return inputs, kwargs

    values = {"disable_progress": True, "on_progress": lambda *_args: None}
    assert MuLaCoverRuntime._supported_pipeline_kwargs(compatible_forward, values) == values


class _FakeRuntime:
    def __init__(self, root: Path):
        self.root = root

    def preflight(self):
        return {"runtime": "mulacover", "ready": True, "missing": []}

    def residency_status(self):
        return {"pipelineObjectLoaded": False, "lazyLoad": True}

    def reset(self):
        return self.residency_status()

    def generate(self, request, attempt_dir, progress):
        progress(0.5, "test_inference", "fake inference")
        output = attempt_dir / f"cover.{request.output_format}"
        output.write_bytes(b"RIFF" + b"0" * 256)
        return MuLaCoverRuntimeResult(
            output_path=output,
            metadata={"runtime": "fake", "seed": request.seed},
            input_manifest={"conditioningMode": request.conditioning_mode},
        )


def test_worker_persists_successful_audio_and_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    worker = MuLaCoverWorker(config, runtime=_FakeRuntime(tmp_path))
    request = MuLaCoverRequest(
        lyrics="words", tags="genre:[pop]", melody_midi_url="https://example.test/melody.mid", chord_midi_url="https://example.test/chords.mid", external_job_id="job-success"
    )
    job = __import__("asyncio").run(worker.submit(request))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = worker.get(job.id)
        if current["status"] == "succeeded":
            break
        time.sleep(0.02)
    current = worker.get(job.id)
    assert current["status"] == "succeeded"
    assert {item["name"] for item in current["artifacts"]} >= {"cover.wav", "request.json", "runtime-metadata.json", "events.jsonl"}
    worker.shutdown()


def test_health_endpoint_reports_preflight() -> None:
    config = _config(Path("."))
    app = create_mulacover_worker_app(config=config, runtime=_FakeRuntime(Path(".")))
    assert app.title == "MuLaCover Cover and Remix Worker"

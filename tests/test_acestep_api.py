import sys
from pathlib import Path
from types import SimpleNamespace

from autotransition.config import RuntimeConfig
from autotransition.models.acestep_api import (
    AceStepApiClient,
    AceStepApiError,
    _extract_audio_path,
    _extract_task_result,
    _raise_api_status,
    _repaint_defaults_for_profile,
)
from autotransition.models.registry import get_model_profile
from autotransition.pipeline import SourceSelectionPlan


def test_api_status_error_includes_method_url_and_compact_body() -> None:
    class Request:
        method = "POST"
        url = "http://127.0.0.1:8001/v1/init"

    class Response:
        status_code = 405
        text = "<!DOCTYPE html>\n<html>\nMethod Not Allowed\n</html>"
        request = Request()

        def json(self):
            raise ValueError("No JSON")

    try:
        _raise_api_status(Response(), "v1/init")
    except AceStepApiError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected AceStepApiError")

    assert "POST http://127.0.0.1:8001/v1/init" in message
    assert "HTTP 405" in message
    assert "<!DOCTYPE html> <html> Method Not Allowed </html>" in message


def test_extract_task_result_parses_result_json_string() -> None:
    data = [
        {
            "task_id": "abc",
            "status": 1,
            "result": '[{"file": "/v1/audio?path=C:/tmp/generated.wav", "status": 1}]',
        }
    ]

    result = _extract_task_result(data, "abc")

    assert result is not None
    assert result["file"] == "/v1/audio?path=C:/tmp/generated.wav"


def test_extract_audio_path_reads_file_download_url() -> None:
    path = _extract_audio_path({"file": "/v1/audio?path=C%3A%2Ftmp%2Fgenerated.wav"})

    assert path == "C:/tmp/generated.wav"


def test_repaint_defaults_are_conservative_for_base_model() -> None:
    defaults = _repaint_defaults_for_profile(get_model_profile("acestep-v15-base"))

    assert defaults["guidance_scale"] == 7.0
    assert defaults["shift"] == 3.0
    assert defaults["repaint_strength"] == 0.3
    assert defaults["repaint_latent_crossfade_frames"] == 24
    assert defaults["repaint_wav_crossfade_sec"] == 0.5


def test_repaint_defaults_use_faster_turbo_settings() -> None:
    defaults = _repaint_defaults_for_profile(get_model_profile("acestep-v15-turbo"))

    assert defaults["guidance_scale"] == 1.0
    assert defaults["repaint_strength"] == 0.45
    assert defaults["repaint_latent_crossfade_frames"] == 16


def test_repaint_uploads_scaffold_as_multipart_file(tmp_path: Path, monkeypatch) -> None:
    scaffold = tmp_path / "scaffold.wav"
    scaffold.write_bytes(b"audio")
    plan = SourceSelectionPlan(
        transition_id="test-generation",
        source_path=tmp_path / "source.mp3",
        source_extension=".mp3",
        source_format="MP3",
        source_duration_seconds=60.0,
        continuation_point_seconds=30.0,
        tail_start_seconds=9.0,
        tail_end_seconds=30.0,
        scaffold_path=scaffold,
        metadata_path=tmp_path / "metadata.json",
        caption="cinematic horror",
        context_seconds=18.0,
        repaint_overlap_seconds=3.0,
        new_section_seconds=36.0,
        requested_continuation_seconds=36.0,
        effective_continuation_seconds=None,
        repainting_start_seconds=18.0,
        repainting_end_seconds=-1.0,
        audio_format="wav",
        bpm_hint=None,
        key_hint=None,
        seed=None,
    )
    calls = []

    class Response:
        def __init__(self, body, status_code=200, content=b"") -> None:
            self._body = body
            self.status_code = status_code
            self.content = content
            self.text = str(body)

        def json(self):
            return self._body

    def fake_get(url, **kwargs):
        calls.append(("get", url, kwargs))
        if url.endswith("/v1/models"):
            return Response({"data": {"models": [{"name": "acestep-v15-turbo"}]}})
        if url.endswith("/v1/audio"):
            return Response({}, content=b"generated")
        return Response({})

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        if url.endswith("/release_task"):
            assert "json" not in kwargs
            assert "src_audio_path" not in kwargs["data"]
            assert kwargs["data"]["task_type"] == "repaint"
            assert kwargs["data"]["batch_size"] == "1"
            assert kwargs["data"]["thinking"] == "false"
            assert kwargs["data"]["repaint_mode"] == "balanced"
            assert kwargs["data"]["repaint_strength"] == "0.45"
            assert kwargs["data"]["repaint_wav_crossfade_sec"] == "0.25"
            assert "src_audio" in kwargs["files"]
            filename, file_obj, mime = kwargs["files"]["src_audio"]
            assert filename == "scaffold.wav"
            assert file_obj.read() == b"audio"
            assert mime == "audio/wav"
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "generated.wav"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).repaint(
        plan=plan,
        profile=get_model_profile("acestep-v15-turbo"),
        save_dir=tmp_path / "generated",
    )

    assert result.output_path.read_bytes() == b"generated"
    assert any(call[1].endswith("/release_task") for call in calls)


def test_repaint_initializes_when_model_list_is_non_json(tmp_path: Path, monkeypatch) -> None:
    scaffold = tmp_path / "scaffold.wav"
    scaffold.write_bytes(b"audio")
    plan = SourceSelectionPlan(
        transition_id="test-generation",
        source_path=tmp_path / "source.mp3",
        source_extension=".mp3",
        source_format="MP3",
        source_duration_seconds=60.0,
        continuation_point_seconds=30.0,
        tail_start_seconds=9.0,
        tail_end_seconds=30.0,
        scaffold_path=scaffold,
        metadata_path=tmp_path / "metadata.json",
        caption="cinematic horror",
        context_seconds=18.0,
        repaint_overlap_seconds=3.0,
        new_section_seconds=36.0,
        requested_continuation_seconds=36.0,
        effective_continuation_seconds=None,
        repainting_start_seconds=18.0,
        repainting_end_seconds=-1.0,
        audio_format="wav",
        bpm_hint=None,
        key_hint=None,
        seed=None,
    )
    calls = []

    class Response:
        def __init__(self, body=None, status_code=200, content=b"", text="") -> None:
            self._body = body
            self.status_code = status_code
            self.content = content
            self.text = text or str(body)

        def json(self):
            if self._body is None:
                raise ValueError("No JSON")
            return self._body

    def fake_get(url, **kwargs):
        calls.append(("get", url, kwargs))
        if url.endswith("/v1/models"):
            return Response(None, text="")
        if url.endswith("/v1/audio"):
            return Response({}, content=b"generated")
        return Response({})

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        if url.endswith("/v1/init"):
            return Response({})
        if url.endswith("/release_task"):
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "generated.wav"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).repaint(
        plan=plan,
        profile=get_model_profile("acestep-v15-turbo"),
        save_dir=tmp_path / "generated",
    )

    assert result.output_path.read_bytes() == b"generated"
    assert any(call[1].endswith("/v1/init") for call in calls)

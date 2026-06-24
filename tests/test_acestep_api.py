import sys
from pathlib import Path
from types import SimpleNamespace

from autotransition.config import RuntimeConfig
from autotransition.models.acestep_api import (
    ACE_STEP_BASE_MODEL,
    ACE_STEP_BASE_SLOT,
    AceStepApiClient,
    AceStepApiError,
    DEFAULT_LM_MODEL_PATH,
    DEFAULT_TEXT2MUSIC_BPM,
    DEFAULT_TEXT2MUSIC_KEY_SCALE,
    _extract_audio_path,
    _extract_task_result,
    _raise_api_status,
    _repaint_defaults_for_profile,
    _text2music_defaults_for_profile,
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


def test_repaint_defaults_use_explicit_mask_for_base_model() -> None:
    defaults = _repaint_defaults_for_profile(get_model_profile("acestep-v15-base"))

    assert defaults["chunk_mask_mode"] == "explicit"
    assert defaults["guidance_scale"] == 7.0
    assert defaults["shift"] == 1.0
    assert defaults["repaint_strength"] == 0.5
    assert defaults["repaint_latent_crossfade_frames"] == 24
    assert defaults["repaint_wav_crossfade_sec"] == 0.5


def test_repaint_defaults_use_faster_turbo_settings() -> None:
    defaults = _repaint_defaults_for_profile(get_model_profile("acestep-v15-turbo"))

    assert defaults["chunk_mask_mode"] == "explicit"
    assert defaults["guidance_scale"] == 1.0
    assert defaults["repaint_strength"] == 0.5
    assert defaults["repaint_latent_crossfade_frames"] == 16


def test_text2music_defaults_do_not_include_repaint_controls() -> None:
    defaults = _text2music_defaults_for_profile(get_model_profile("acestep-v15-turbo"))

    assert defaults["guidance_scale"] == 1.0
    assert "chunk_mask_mode" not in defaults
    assert "repaint_strength" not in defaults


def test_base_profile_keeps_non_turbo_step_default() -> None:
    profile = get_model_profile("acestep-v15-base")
    defaults = _text2music_defaults_for_profile(profile)

    assert profile.default_inference_steps == 50
    assert defaults["guidance_scale"] == 7.0
    assert defaults["shift"] == 1.0
    assert defaults["infer_method"] == "ode"
    assert defaults["sampler_mode"] == "euler"


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
        repainting_end_seconds=57.0,
        audio_format="wav",
        bpm_hint=None,
        key_hint=None,
        seed=None,
        ace_step_settings={
            "inference_steps": 12,
            "repaint_strength": 0.25,
            "chunk_mask_mode": "auto",
        },
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
        if url.endswith("/v1/model_inventory"):
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
            assert kwargs["data"]["repainting_start"] == "18.0"
            assert kwargs["data"]["repainting_end"] == "57.0"
            assert kwargs["data"]["lyrics"] == "[Instrumental]"
            assert kwargs["data"]["vocal_language"] == "unknown"
            assert kwargs["data"]["batch_size"] == "1"
            assert kwargs["data"]["thinking"] == "false"
            assert kwargs["data"]["inference_steps"] == "12"
            assert kwargs["data"]["chunk_mask_mode"] == "auto"
            assert kwargs["data"]["repaint_mode"] == "balanced"
            assert kwargs["data"]["repaint_strength"] == "0.25"
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


def test_text2music_generates_prompted_section_without_source_audio(tmp_path: Path, monkeypatch) -> None:
    scaffold = tmp_path / "scaffold.wav"
    scaffold.write_bytes(b"audio")
    plan = SourceSelectionPlan(
        transition_id="test-generation",
        source_path=tmp_path / "source.mp3",
        source_extension=".mp3",
        source_format="MP3",
        source_duration_seconds=60.0,
        continuation_point_seconds=30.0,
        tail_start_seconds=12.0,
        tail_end_seconds=30.0,
        scaffold_path=scaffold,
        metadata_path=tmp_path / "metadata.json",
        caption="instrumental cinematic horror",
        context_seconds=18.0,
        repaint_overlap_seconds=3.0,
        new_section_seconds=36.0,
        requested_continuation_seconds=36.0,
        effective_continuation_seconds=None,
        repainting_start_seconds=15.0,
        repainting_end_seconds=54.0,
        audio_format="wav",
        bpm_hint=120.0,
        key_hint="A minor",
        seed=42,
        ace_step_settings={
            "inference_steps": 16,
            "guidance_scale": 2.0,
            "chunk_mask_mode": "auto",
            "repaint_strength": 0.8,
        },
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
        if url.endswith("/v1/model_inventory"):
            return Response({"data": {"models": [{"name": "acestep-v15-turbo"}]}})
        if url.endswith("/v1/audio"):
            return Response({}, content=b"generated")
        return Response({})

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        if url.endswith("/v1/init"):
            raise AssertionError("text2music should use the active ACE-Step runtime model without forcing init")
        if url.endswith("/release_task"):
            assert "files" not in kwargs or kwargs["files"] is None
            assert "data" not in kwargs
            payload = kwargs["json"]
            assert payload["task_type"] == "text2music"
            assert "model" not in payload
            assert payload["thinking"] is True
            assert payload["prompt"] == "instrumental cinematic horror"
            assert payload["lyrics"] == "[Instrumental]"
            assert payload["audio_duration"] == 36.0
            assert payload["audio_format"] == "flac"
            assert payload["time_signature"] == "4"
            assert payload["lm_model_path"] == DEFAULT_LM_MODEL_PATH
            assert payload["use_random_seed"] is False
            assert payload["seed"] == 42
            assert payload["bpm"] == 120
            assert payload["key_scale"] == "A minor"
            assert payload["inference_steps"] == 16
            assert payload["guidance_scale"] == 2.0
            assert payload["use_format"] is False
            assert payload["lm_temperature"] == 0.85
            assert payload["lm_cfg_scale"] == 2.5
            assert payload["lm_top_p"] == 0.9
            assert payload["lm_negative_prompt"] == "NO USER INPUT"
            assert "vocal_language" not in payload
            assert "instruction" not in payload
            assert "infer_method" not in payload
            assert "use_tiled_decode" not in payload
            assert "constrained_decoding" not in payload
            assert "use_cot_caption" not in payload
            assert "use_cot_language" not in payload
            assert "allow_lm_batch" not in payload
            assert "chunk_mask_mode" not in payload
            assert "repaint_strength" not in payload
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "generated.flac"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).text2music(
        plan=plan,
        profile=get_model_profile("acestep-v15-turbo"),
        save_dir=tmp_path / "generated",
    )

    assert result.output_path.read_bytes() == b"generated"
    assert result.output_path.suffix == ".flac"
    assert (tmp_path / "generated" / "ace-request.json").exists()
    assert (tmp_path / "generated" / "ace-release-response.json").exists()
    assert (tmp_path / "generated" / "ace-query-response-final.json").exists()
    assert not any(call[1].endswith("/v1/init") for call in calls)


def test_repaint_transition_uses_active_runtime_without_forcing_init(tmp_path: Path, monkeypatch) -> None:
    scaffold = tmp_path / "transition-scaffold.wav"
    scaffold.write_bytes(b"audio")
    plan = SourceSelectionPlan(
        transition_id="test-generation",
        source_path=tmp_path / "source.mp3",
        source_extension=".mp3",
        source_format="MP3",
        source_duration_seconds=60.0,
        continuation_point_seconds=30.0,
        tail_start_seconds=0.0,
        tail_end_seconds=60.0,
        scaffold_path=scaffold,
        metadata_path=tmp_path / "metadata.json",
        caption="cinematic horror",
        context_seconds=18.0,
        repaint_overlap_seconds=2.0,
        new_section_seconds=30.0,
        requested_continuation_seconds=30.0,
        effective_continuation_seconds=None,
        repainting_start_seconds=28.0,
        repainting_end_seconds=60.0,
        audio_format="wav",
        bpm_hint=120.0,
        key_hint="C minor",
        seed=99,
        ace_step_settings={
            "inference_steps": 12,
            "repaint_strength": 0.35,
            "chunk_mask_mode": "auto",
            "lm_temperature": 0.1,
        },
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
        if url.endswith("/v1/model_inventory"):
            raise AssertionError("transition repaint should not force model inventory checks")
        if url.endswith("/v1/audio"):
            return Response({}, content=b"generated")
        return Response({})

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        if url.endswith("/v1/init"):
            raise AssertionError("transition repaint should not force ACE-Step model init")
        if url.endswith("/release_task"):
            payload = kwargs["data"]
            assert payload["task_type"] == "repaint"
            assert "model" not in payload
            assert payload["repainting_start"] == "28.0"
            assert payload["repainting_end"] == "60.0"
            assert payload["prompt"] == "cinematic horror"
            assert payload["lyrics"] == "[Instrumental]"
            assert payload["audio_format"] == "wav"
            assert payload["inference_steps"] == "12"
            assert payload["chunk_mask_mode"] == "auto"
            assert payload["repaint_strength"] == "0.35"
            assert payload["use_random_seed"] == "false"
            assert payload["seed"] == "99"
            assert payload["bpm"] == "120"
            assert payload["key_scale"] == "C minor"
            assert "audio_duration" not in payload
            assert "lm_temperature" not in payload
            assert "src_audio" in kwargs["files"]
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "generated.wav"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).repaint_transition(
        plan=plan,
        profile=get_model_profile("acestep-v15-turbo"),
        save_dir=tmp_path / "generated",
    )

    assert result.output_path.read_bytes() == b"generated"
    assert not any(call[1].endswith("/v1/init") for call in calls)


def test_extract_track_initializes_base_slot_and_uploads_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "song.wav"
    source.write_bytes(b"source-audio")
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
        if url.endswith("/v1/model_inventory"):
            return Response({"data": {"models": [{"name": "acestep-v15-turbo", "is_loaded": True}]}})
        if url.endswith("/v1/audio"):
            return Response({}, content=b"extracted")
        return Response({})

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        if url.endswith("/v1/init"):
            assert kwargs["json"] == {"model": ACE_STEP_BASE_MODEL, "slot": ACE_STEP_BASE_SLOT, "init_llm": False}
            return Response({"data": {"loaded_model": ACE_STEP_BASE_MODEL}})
        if url.endswith("/release_task"):
            payload = kwargs["data"]
            assert payload["task_type"] == "extract"
            assert payload["model"] == ACE_STEP_BASE_MODEL
            assert payload["track_name"] == "vocals"
            assert payload["audio_format"] == "flac"
            assert payload["inference_steps"] == "50"
            assert payload["guidance_scale"] == "7.0"
            assert payload["shift"] == "1.0"
            assert "lyrics" not in payload
            assert "src_audio" in kwargs["files"]
            filename, file_obj, mime = kwargs["files"]["src_audio"]
            assert filename == "song.wav"
            assert file_obj.read() == b"source-audio"
            assert mime == "application/octet-stream"
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "extracted.flac"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).extract_track(
        source_path=source,
        track_name="vocals",
        save_dir=tmp_path / "extract",
    )

    assert result.output_path.read_bytes() == b"extracted"


def test_base_text2music_test_uses_non_turbo_schedule(tmp_path: Path, monkeypatch) -> None:
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
        if url.endswith("/v1/model_inventory"):
            return Response({"data": {"models": [{"name": ACE_STEP_BASE_MODEL, "is_loaded": True}]}})
        if url.endswith("/v1/audio"):
            return Response({}, content=b"base-audio")
        return Response({})

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        if url.endswith("/release_task"):
            payload = kwargs["json"]
            assert payload["task_type"] == "text2music"
            assert payload["model"] == ACE_STEP_BASE_MODEL
            assert payload["prompt"] == "dark strings"
            assert payload["inference_steps"] == 50
            assert payload["guidance_scale"] == 7.0
            assert payload["shift"] == 1.0
            assert payload["infer_method"] == "ode"
            assert payload["sampler_mode"] == "euler"
            assert payload["use_adg"] is False
            assert payload["use_tiled_decode"] is False
            assert payload["dcw_enabled"] is False
            assert payload["velocity_norm_threshold"] == 2.5
            assert payload["velocity_ema_factor"] == 0.35
            assert payload["thinking"] is True
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "base.flac"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).text2music_base_test(
        prompt="dark strings",
        save_dir=tmp_path / "base-test",
        use_tiled_decode=False,
        dcw_enabled=False,
        velocity_norm_threshold=2.5,
        velocity_ema_factor=0.35,
    )

    assert result.output_path.read_bytes() == b"base-audio"
    assert not any(call[1].endswith("/v1/init") for call in calls)


def test_text2music_uses_working_bpm_and_key_defaults_when_ui_omits_them(tmp_path: Path, monkeypatch) -> None:
    plan = SourceSelectionPlan(
        transition_id="test-generation",
        source_path=tmp_path / "source.mp3",
        source_extension=".mp3",
        source_format="MP3",
        source_duration_seconds=60.0,
        continuation_point_seconds=30.0,
        tail_start_seconds=12.0,
        tail_end_seconds=30.0,
        scaffold_path=tmp_path / "scaffold.wav",
        metadata_path=tmp_path / "metadata.json",
        caption="instrumental cinematic horror",
        context_seconds=18.0,
        repaint_overlap_seconds=0.0,
        new_section_seconds=30.0,
        requested_continuation_seconds=30.0,
        effective_continuation_seconds=None,
        repainting_start_seconds=18.0,
        repainting_end_seconds=48.0,
        audio_format="wav",
        bpm_hint=None,
        key_hint=None,
        seed=None,
    )

    class Response:
        def __init__(self, body, status_code=200, content=b"") -> None:
            self._body = body
            self.status_code = status_code
            self.content = content
            self.text = str(body)

        def json(self):
            return self._body

    def fake_get(url, **kwargs):
        if url.endswith("/v1/model_inventory"):
            raise AssertionError("text2music should not check model inventory before generation")
        if url.endswith("/v1/audio"):
            return Response({}, content=b"generated")
        return Response({})

    def fake_post(url, **kwargs):
        if url.endswith("/release_task"):
            payload = kwargs["json"]
            assert "model" not in payload
            assert payload["bpm"] == DEFAULT_TEXT2MUSIC_BPM
            assert payload["key_scale"] == DEFAULT_TEXT2MUSIC_KEY_SCALE
            assert payload["audio_duration"] == 30.0
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "generated.flac"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).text2music(
        plan=plan,
        profile=get_model_profile("acestep-v15-xl-base"),
        save_dir=tmp_path / "generated",
    )

    assert result.output_path.read_bytes() == b"generated"


def test_text2music_skips_init_when_dit_and_lm_are_loaded(tmp_path: Path, monkeypatch) -> None:
    plan = SourceSelectionPlan(
        transition_id="test-generation",
        source_path=tmp_path / "source.mp3",
        source_extension=".mp3",
        source_format="MP3",
        source_duration_seconds=60.0,
        continuation_point_seconds=30.0,
        tail_start_seconds=12.0,
        tail_end_seconds=30.0,
        scaffold_path=tmp_path / "scaffold.wav",
        metadata_path=tmp_path / "metadata.json",
        caption="instrumental cinematic horror",
        context_seconds=18.0,
        repaint_overlap_seconds=0.0,
        new_section_seconds=36.0,
        requested_continuation_seconds=36.0,
        effective_continuation_seconds=None,
        repainting_start_seconds=18.0,
        repainting_end_seconds=54.0,
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
        if url.endswith("/v1/model_inventory"):
            return Response(
                {
                    "data": {
                        "models": [{"name": "acestep-v15-turbo", "is_loaded": True}],
                        "lm_models": [{"name": DEFAULT_LM_MODEL_PATH, "is_loaded": True}],
                        "loaded_lm_model": DEFAULT_LM_MODEL_PATH,
                        "llm_initialized": True,
                    }
                }
            )
        if url.endswith("/v1/audio"):
            return Response({}, content=b"generated")
        return Response({})

    def fake_post(url, **kwargs):
        calls.append(("post", url, kwargs))
        if url.endswith("/v1/init"):
            raise AssertionError("model init should be skipped when DiT and LM are already loaded")
        if url.endswith("/release_task"):
            return Response({"data": {"task_id": "task-1"}})
        if url.endswith("/query_result"):
            return Response({"data": [{"task_id": "task-1", "status": 1, "file": "generated.flac"}]})
        return Response({})

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = AceStepApiClient(RuntimeConfig()).text2music(
        plan=plan,
        profile=get_model_profile("acestep-v15-turbo"),
        save_dir=tmp_path / "generated",
    )

    assert result.output_path.read_bytes() == b"generated"
    assert not any(call[1].endswith("/v1/init") for call in calls)


def test_text2music_surfaces_runtime_lm_errors_without_forcing_init(tmp_path: Path, monkeypatch) -> None:
    plan = SourceSelectionPlan(
        transition_id="test-generation",
        source_path=tmp_path / "source.mp3",
        source_extension=".mp3",
        source_format="MP3",
        source_duration_seconds=60.0,
        continuation_point_seconds=30.0,
        tail_start_seconds=12.0,
        tail_end_seconds=30.0,
        scaffold_path=tmp_path / "scaffold.wav",
        metadata_path=tmp_path / "metadata.json",
        caption="instrumental cinematic horror",
        context_seconds=18.0,
        repaint_overlap_seconds=0.0,
        new_section_seconds=36.0,
        requested_continuation_seconds=36.0,
        effective_continuation_seconds=None,
        repainting_start_seconds=18.0,
        repainting_end_seconds=54.0,
        audio_format="wav",
        bpm_hint=None,
        key_hint=None,
        seed=None,
    )

    class Response:
        status_code = 200
        text = "{}"

        def __init__(self, body) -> None:
            self._body = body

        def json(self):
            return self._body

    def fake_get(url, **kwargs):
        if url.endswith("/v1/model_inventory"):
            raise AssertionError("text2music should not check model inventory before generation")
        return Response({})

    def fake_post(url, **kwargs):
        if url.endswith("/v1/init"):
            raise AssertionError("text2music should not force LM initialization")
        if url.endswith("/release_task"):
            return Response({"error": "5Hz LM init failed"})
        raise AssertionError(f"unexpected post: {url}")

    fake_httpx = SimpleNamespace(get=fake_get, post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    try:
        AceStepApiClient(RuntimeConfig()).text2music(
            plan=plan,
            profile=get_model_profile("acestep-v15-turbo"),
            save_dir=tmp_path / "generated",
        )
    except AceStepApiError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected AceStepApiError")

    assert "5Hz LM init failed" in message


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
        if url.endswith("/v1/model_inventory"):
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


def test_repaint_wraps_init_timeout_as_api_error(tmp_path: Path, monkeypatch) -> None:
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

    class TimeoutException(Exception):
        pass

    class HTTPError(Exception):
        pass

    class Response:
        status_code = 200
        text = "{}"

        def json(self):
            return {"data": {"models": []}}

    def fake_get(url, **kwargs):
        return Response()

    def fake_post(url, **kwargs):
        if url.endswith("/v1/init"):
            assert kwargs["timeout"] == RuntimeConfig().generation_timeout_seconds
            raise TimeoutException("timed out")
        return Response()

    fake_httpx = SimpleNamespace(
        get=fake_get,
        post=fake_post,
        TimeoutException=TimeoutException,
        HTTPError=HTTPError,
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    try:
        AceStepApiClient(RuntimeConfig()).repaint(
            plan=plan,
            profile=get_model_profile("acestep-v15-turbo"),
            save_dir=tmp_path / "generated",
        )
    except AceStepApiError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected AceStepApiError")

    assert "v1/init timed out" in message

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from autotransition.config import RuntimeConfig
from autotransition.ui import create_app


def make_wav(path: Path, duration_ms: int = 3000) -> Path:
    from pydub import AudioSegment

    AudioSegment.silent(duration=duration_ms, frame_rate=44100).export(path, format="wav")
    return path


def test_ui_status_endpoint_reports_environment(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))

    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert data["models_dir"] == str(tmp_path)
    assert data["repaint_model_count"] >= 1
    assert ".mp3" in data["supported_input_formats"]
    assert data["default_scaffold_format"] == "wav"
    assert "python_version" in data


def test_ui_presets_endpoint_returns_creator_presets(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))

    response = client.get("/api/presets")

    assert response.status_code == 200
    slugs = {preset["slug"] for preset in response.json()}
    assert "smooth-continuation" in slugs
    assert "dj-bridge" in slugs


def test_ui_models_endpoint_includes_install_status(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))

    response = client.get("/api/models")

    assert response.status_code == 200
    models = response.json()
    assert any(model["slug"] == "acestep-v15-turbo" for model in models)
    assert all("status" in model for model in models)


def test_ui_models_endpoint_exposes_working_xl_base_generation_defaults(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))

    response = client.get("/api/models")

    assert response.status_code == 200
    models = response.json()
    xl_base = next(model for model in models if model["slug"] == "acestep-v15-xl-base")
    assert xl_base["generation_defaults"]["inference_steps"] == 50
    assert xl_base["generation_defaults"]["guidance_scale"] == 7.0
    assert xl_base["generation_defaults"]["shift"] == 3.0


def test_ui_runtime_status_endpoint_returns_setup_commands(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))

    response = client.get("/api/runtime/status")

    assert response.status_code == 200
    data = response.json()
    assert "install_commands" in data
    assert "uv sync" in data["install_commands"]
    assert data["simple_setup_command"] == "autotransition runtime setup"
    assert data["simple_start_command"] == "autotransition runtime start"


def test_ui_runtime_status_uses_configured_runtime_port(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            models_dir=tmp_path,
            runtime_config=RuntimeConfig(ace_step_dir=tmp_path / "runtime", api_port=9101),
        )
    )

    response = client.get("/api/runtime/status")

    assert response.status_code == 200
    assert response.json()["api_url"] == "http://127.0.0.1:9101"


def test_ui_scaffold_endpoint_validates_missing_source(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))

    response = client.post(
        "/api/scaffolds",
        json={
            "source_path": str(tmp_path / "missing.wav"),
            "preset": "smooth-continuation",
        },
    )

    assert response.status_code == 400
    assert "Source audio not found" in response.json()["detail"]


def test_ui_logs_endpoint_records_validation_errors(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))
    client.post(
        "/api/scaffolds",
        json={
            "source_path": str(tmp_path / "missing.wav"),
            "preset": "smooth-continuation",
        },
    )

    response = client.get("/api/logs")

    assert response.status_code == 200
    assert any("Source audio not found" in entry["message"] for entry in response.json())


def test_ui_source_probe_endpoint_returns_duration(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "song.wav", duration_ms=3000)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post("/api/source/probe", json={"source_path": str(source)})

    assert response.status_code == 200
    data = response.json()
    assert data["path"] == str(source)
    assert data["source_extension"] == ".wav"
    assert data["source_format"] == "WAV"
    assert data["duration_seconds"] == 3.0


def test_ui_source_audio_endpoint_serves_file(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "song.wav", duration_ms=1000)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.get("/api/source/audio", params={"path": str(source)})

    assert response.status_code == 200
    assert response.content


def test_ui_generated_audio_can_be_loaded_as_next_source(tmp_path: Path) -> None:
    generated = make_wav(tmp_path / "generated-transition.wav", duration_ms=2500)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    probe = client.post("/api/source/probe", json={"source_path": str(generated)})
    audio = client.get("/api/source/audio", params={"path": str(generated)})

    assert probe.status_code == 200
    assert probe.json()["duration_seconds"] == 2.5
    assert audio.status_code == 200
    assert audio.content


def test_ui_extraction_tracks_endpoint_lists_ace_tracks(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.get("/api/extractions/tracks")

    assert response.status_code == 200
    assert "vocals" in response.json()
    assert "drums" in response.json()


def test_ui_run_extraction_writes_history(tmp_path: Path, monkeypatch) -> None:
    from autotransition.models.acestep_api import AceStepApiClient
    from autotransition.models.base import RepaintResult

    monkeypatch.chdir(tmp_path)
    source = make_wav(tmp_path / "song.wav", duration_ms=3000)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    def fake_extract_track(
        self,
        source_path,
        track_name,
        save_dir,
        *,
        audio_format="flac",
        inference_steps=50,
        guidance_scale=7.0,
        shift=3.0,
        seed=None,
        instruction=None,
    ):
        assert source_path == source
        assert track_name == "vocals"
        assert audio_format == "flac"
        assert inference_steps == 32
        output = save_dir / "vocals.flac"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"audio")
        metadata = save_dir / "vocals.json"
        metadata.write_text("{}", encoding="utf-8")
        return RepaintResult(output_path=output, metadata_path=metadata, model_name="ACE-Step API")

    monkeypatch.setattr(AceStepApiClient, "extract_track", fake_extract_track)

    response = client.post(
        "/api/extractions/run",
        json={
            "source_path": str(source),
            "track_name": "vocals",
            "output_format": "flac",
            "inference_steps": 32,
            "guidance_scale": 7.0,
            "shift": 3.0,
        },
    )
    history = client.get("/api/extractions")

    assert response.status_code == 200
    extraction = response.json()["extraction"]
    assert extraction["status"] == "complete"
    assert extraction["track_name"] == "vocals"
    assert Path(extraction["generated_audio_path"]).exists()
    assert history.status_code == 200
    assert history.json()[0]["extraction_id"] == extraction["extraction_id"]


def test_ui_selection_scaffold_uses_configured_future_length(tmp_path: Path) -> None:
    from pydub import AudioSegment

    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    output_dir = tmp_path / "out"
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/scaffolds/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "output_dir": str(output_dir),
            "continuation_point_seconds": 4.0,
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 1.0,
            "new_section_seconds": 5.0,
        },
    )

    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["tail_start_seconds"] == 2.0
    assert plan["tail_end_seconds"] == 4.0
    assert plan["source_format"] == "WAV"
    assert plan["audio_format"] == "wav"
    assert plan["requested_continuation_seconds"] == 5.0
    assert plan["repainting_end_seconds"] == 7.0
    assert Path(plan["scaffold_path"]).exists()
    assert len(AudioSegment.from_file(plan["scaffold_path"])) == 2000


def test_ui_selection_scaffold_rejects_early_marker(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/scaffolds/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "continuation_point_seconds": 1.0,
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 1.0,
            "new_section_seconds": 5.0,
        },
    )

    assert response.status_code == 400
    assert "too early" in response.json()["detail"]


def test_ui_selection_scaffold_honors_zero_repaint_margin(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/scaffolds/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "continuation_point_seconds": 4.0,
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 0.0,
            "new_section_seconds": 5.0,
        },
    )

    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["repaint_overlap_seconds"] == 0.0
    assert plan["repainting_start_seconds"] == 2.0


def test_ui_source_upload_accepts_audio_file(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "picked.wav", duration_ms=1200)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    with source.open("rb") as audio_file:
        response = client.post(
            "/api/source/upload",
            files={"file": ("picked.wav", audio_file, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["original_filename"] == "picked.wav"
    assert Path(data["stored_path"]).exists()
    assert data["probe"]["source_format"] == "WAV"
    assert data["probe"]["duration_seconds"] == 1.2


def test_ui_source_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/source/upload",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_ui_generate_from_selection_does_not_block_on_app_model_install_status(tmp_path: Path, monkeypatch) -> None:
    import autotransition.runtime.ace_step as ace_step_runtime
    from autotransition.models.acestep_api import AceStepApiClient
    from autotransition.models.base import RepaintResult

    monkeypatch.setattr(
        ace_step_runtime,
        "runtime_status",
        lambda config=None: SimpleNamespace(api_running=True, message="ACE-Step API is ready."),
    )

    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    output_dir = tmp_path / "out"
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    def fake_text2music(self, plan, profile, save_dir):
        generated = output_dir / "generated.flac"
        generated.parent.mkdir(parents=True, exist_ok=True)
        from pydub import AudioSegment

        AudioSegment.silent(duration=1000, frame_rate=44100).export(generated, format="flac")
        metadata = output_dir / "generated.json"
        metadata.write_text("{}", encoding="utf-8")
        return RepaintResult(output_path=generated, metadata_path=metadata, model_name="ACE-Step API")

    def fake_repaint_transition(self, plan, profile, save_dir):
        generated = output_dir / "repainted.wav"
        generated.parent.mkdir(parents=True, exist_ok=True)
        from pydub import AudioSegment

        AudioSegment.silent(duration=5000, frame_rate=44100).export(generated, format="wav")
        metadata = output_dir / "repainted.json"
        metadata.write_text("{}", encoding="utf-8")
        return RepaintResult(output_path=generated, metadata_path=metadata, model_name="ACE-Step API")

    monkeypatch.setattr(AceStepApiClient, "text2music", fake_text2music)
    monkeypatch.setattr(AceStepApiClient, "repaint_transition", fake_repaint_transition)

    response = client.post(
        "/api/generate/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "model_slug": "acestep-v15-turbo",
            "output_dir": str(output_dir),
            "continuation_point_seconds": 4.0,
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 0.0,
            "new_section_seconds": 5.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["status"] == "complete"
    assert payload["plan"]["requested_continuation_seconds"] == 5.0
    assert payload["plan"]["repainting_end_seconds"] == 7.0


def test_ui_generate_from_selection_records_advanced_settings_and_region(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "song.wav", duration_ms=10000)
    output_dir = tmp_path / "out"
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/generate/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "model_slug": "acestep-v15-turbo",
            "output_dir": str(output_dir),
            "continuation_point_seconds": 4.0,
            "generation_region": "repaint_existing",
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 1.0,
            "new_section_seconds": 5.0,
            "ace_step": {
                "inference_steps": 16,
                "chunk_mask_mode": "auto",
                "repaint_mode": "aggressive",
                "repaint_strength": 0.8,
            },
        },
    )

    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["generation_region"] == "repaint_existing"
    assert plan["repainting_start_seconds"] == 1.0
    assert plan["repainting_end_seconds"] == 7.0
    assert plan["ace_step_settings"]["inference_steps"] == 16
    assert plan["ace_step_settings"]["chunk_mask_mode"] == "auto"
    assert plan["ace_step_settings"]["repaint_mode"] == "aggressive"
    assert plan["ace_step_settings"]["repaint_strength"] == 0.8


def test_ui_generate_from_selection_rejects_existing_repaint_without_enough_audio(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/generate/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "model_slug": "acestep-v15-turbo",
            "continuation_point_seconds": 4.0,
            "generation_region": "repaint_existing",
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 1.0,
            "new_section_seconds": 5.0,
        },
    )

    assert response.status_code == 400
    assert "not enough source audio" in response.json()["detail"]


def test_ui_generate_from_selection_reports_missing_runtime_when_model_ready(tmp_path: Path, monkeypatch) -> None:
    import autotransition.runtime.ace_step as ace_step_runtime

    monkeypatch.setattr(
        ace_step_runtime,
        "runtime_status",
        lambda config=None: SimpleNamespace(api_running=False, message="ACE-Step API is not ready."),
    )
    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    model_dir = tmp_path / "models" / "acestep-v15-turbo"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/generate/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "model_slug": "acestep-v15-turbo",
            "continuation_point_seconds": 4.0,
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 1.0,
            "new_section_seconds": 5.0,
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "failed"
    assert "ACE-Step API is not ready" in result["message"]
    assert Path(result["scaffold_metadata_path"]).exists()


def test_ui_generate_from_selection_uses_text2music_and_composite_by_default(tmp_path: Path, monkeypatch) -> None:
    import autotransition.ui.app as ui_app
    from autotransition.models.base import RepaintResult

    calls = []

    class Adapter:
        def __init__(self, profile, model_path, runtime_config=None) -> None:
            pass

        def text2music(self, plan):
            calls.append(("text2music", plan.transition_id))
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            raw_audio = make_wav(raw_dir / "raw.wav", duration_ms=1000)
            raw_metadata = raw_dir / "raw.json"
            raw_metadata.write_text("{}", encoding="utf-8")
            return RepaintResult(output_path=raw_audio, metadata_path=raw_metadata, model_name="fake")

        def repaint_transition(self, plan):
            calls.append(("repaint", plan.transition_id))
            final_dir = tmp_path / "final"
            final_dir.mkdir()
            final_audio = make_wav(final_dir / "final.wav", duration_ms=2000)
            final_metadata = final_dir / "final.json"
            final_metadata.write_text("{}", encoding="utf-8")
            return RepaintResult(output_path=final_audio, metadata_path=final_metadata, model_name="fake")

    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    output_dir = tmp_path / "out"
    model_dir = tmp_path / "models" / "acestep-v15-turbo"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")
    monkeypatch.setattr(ui_app, "AceStepRepaintAdapter", Adapter)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/generate/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "model_slug": "acestep-v15-turbo",
            "output_dir": str(output_dir),
            "continuation_point_seconds": 2.0,
            "context_seconds": 1.0,
            "repaint_overlap_seconds": 0.0,
            "new_section_seconds": 1.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["status"] == "complete"
    assert calls == [
        ("text2music", payload["plan"]["transition_id"]),
        ("repaint", payload["plan"]["transition_id"]),
    ]
    assert Path(payload["result"]["generated_audio_path"]).exists()
    assert payload["result"]["generated_audio_path"].endswith("final.wav")


def test_ui_generate_from_selection_can_boundary_repaint_composite(tmp_path: Path, monkeypatch) -> None:
    import autotransition.ui.app as ui_app
    from autotransition.models.base import RepaintResult

    seen_boundary = {}

    class Adapter:
        def __init__(self, profile, model_path, runtime_config=None) -> None:
            pass

        def text2music(self, plan):
            raw_dir = tmp_path / "raw-boundary"
            raw_dir.mkdir()
            raw_audio = make_wav(raw_dir / "raw.wav", duration_ms=2000)
            raw_metadata = raw_dir / "raw.json"
            raw_metadata.write_text("{}", encoding="utf-8")
            return RepaintResult(output_path=raw_audio, metadata_path=raw_metadata, model_name="fake")

        def repaint_transition(self, plan):
            seen_boundary["start"] = plan.repainting_start_seconds
            seen_boundary["end"] = plan.repainting_end_seconds
            seen_boundary["scaffold_path"] = str(plan.scaffold_path)
            final_dir = tmp_path / "final-boundary"
            final_dir.mkdir()
            final_audio = make_wav(final_dir / "final.wav", duration_ms=4000)
            final_metadata = final_dir / "final.json"
            final_metadata.write_text("{}", encoding="utf-8")
            return RepaintResult(output_path=final_audio, metadata_path=final_metadata, model_name="fake")

    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    output_dir = tmp_path / "out"
    model_dir = tmp_path / "models" / "acestep-v15-turbo"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")
    monkeypatch.setattr(ui_app, "AceStepRepaintAdapter", Adapter)
    client = TestClient(create_app(models_dir=tmp_path / "models"))

    response = client.post(
        "/api/generate/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "model_slug": "acestep-v15-turbo",
            "output_dir": str(output_dir),
            "continuation_point_seconds": 3.0,
            "context_seconds": 1.0,
            "repaint_overlap_seconds": 1.0,
            "new_section_seconds": 2.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["status"] == "complete"
    assert seen_boundary["start"] == 2.0
    assert seen_boundary["end"] == 5.0
    assert Path(seen_boundary["scaffold_path"]).exists()
    assert payload["result"]["generated_audio_path"].endswith("final.wav")


def test_ui_generate_from_selection_passes_configured_runtime_port(tmp_path: Path, monkeypatch) -> None:
    import autotransition.ui.app as ui_app

    seen_ports = []

    class Adapter:
        def __init__(self, profile, model_path, runtime_config=None) -> None:
            seen_ports.append(runtime_config.api_port)

        def text2music(self, plan):
            from autotransition.models import AceStepRuntimeError

            raise AceStepRuntimeError("stop after config capture")

    source = make_wav(tmp_path / "song.wav", duration_ms=6000)
    model_dir = tmp_path / "models" / "acestep-v15-turbo"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")
    monkeypatch.setattr(ui_app, "AceStepRepaintAdapter", Adapter)
    client = TestClient(
        create_app(
            models_dir=tmp_path / "models",
            runtime_config=RuntimeConfig(api_port=9101),
        )
    )

    response = client.post(
        "/api/generate/from-selection",
        json={
            "source_path": str(source),
            "preset": "smooth-continuation",
            "model_slug": "acestep-v15-turbo",
            "continuation_point_seconds": 4.0,
            "context_seconds": 2.0,
            "repaint_overlap_seconds": 1.0,
            "new_section_seconds": 5.0,
        },
    )

    assert response.status_code == 200
    assert seen_ports == [9101]

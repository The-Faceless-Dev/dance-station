from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from autotransition.dataset_manager.app import create_app
from autotransition.dataset_manager.config import DatasetManagerConfig
from autotransition.dataset_manager.contracts import validate_annotation
from autotransition.dataset_manager.db import DatasetRepository


def _wav_bytes(duration: float = 0.25, sample_rate: int = 16000) -> bytes:
    samples = (np.sin(np.arange(int(duration * sample_rate)) * 2 * np.pi * 220 / sample_rate) * 0.2 * 32767).astype(np.int16)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return stream.getvalue()


def _config(tmp_path: Path) -> DatasetManagerConfig:
    return DatasetManagerConfig(storage_root=tmp_path / "data", model_root=tmp_path / "models", max_workers=1)


def test_repository_recovers_and_persists_draft_and_revision(tmp_path: Path) -> None:
    repository = DatasetRepository(tmp_path / "db")
    song = repository.create_song({"source_filename": "test.wav", "source_sha256": "abc", "duration_ms": 1000, "language": "en"})
    annotation = {"schema_version": 1, "song_id": song["id"], "duration_ms": 1000, "words": [{"id": "word_1", "text": "hello", "start_ms": 100, "end_ms": 500, "phonemes": [{"id": "phone_1", "symbol": "HH", "start_ms": 100, "end_ms": 250, "alignment_status": "estimated"}]}]}
    validation = validate_annotation(annotation, 1000)
    assert validation["valid"]
    draft = repository.upsert_draft(song["id"], annotation, validation)
    assert draft["version"] == 1
    revision = repository.create_revision(song["id"], annotation, validation, {"annotation": "revision.json"})
    assert revision["version"] == 1
    assert repository.get_song(song["id"])["status"] == "Saved"
    assert repository.stats()["saved_songs"] == 1


def test_annotation_rejects_out_of_bounds_and_duplicate_ids() -> None:
    result = validate_annotation({"words": [{"id": "same", "text": "x", "start_ms": 0, "end_ms": 200, "phonemes": [{"id": "same", "symbol": "X", "start_ms": 0, "end_ms": 50}]}]}, 100)
    assert not result["valid"]
    assert any("bounds" in error for error in result["errors"])
    assert any("duplicate" in error for error in result["errors"])


def test_standalone_api_imports_audio_and_preserves_user_lyrics(tmp_path: Path) -> None:
    app = create_app(_config(tmp_path))
    with TestClient(app) as client:
        response = client.post("/api/dataset/songs", files={"file": ("demo.wav", _wav_bytes(), "audio/wav")}, data={"title": "Demo", "lyrics": "Hello world", "vocals_only": "true"})
        assert response.status_code == 200, response.text
        song = response.json()
        assert song["title"] == "Demo"
        assert song["lyrics"] == "Hello world"
        assert client.get("/api/dataset/stats").json()["songs"] == 1
        detail = client.get(f"/api/dataset/songs/{song['id']}").json()
        assert detail["status"] == "New"
        assert client.get(f"/api/dataset/songs/{song['id']}/audio?kind=original").status_code == 200

        annotation = {
            "schema_version": 1,
            "song_id": song["id"],
            "duration_ms": song["duration_ms"],
            "words": [{
                "id": "word_1",
                "text": "Hello",
                "start_ms": 0,
                "end_ms": song["duration_ms"],
                "phonemes": [],
            }],
        }
        saved = client.patch(f"/api/dataset/songs/{song['id']}/draft", json=annotation)
        assert saved.status_code == 200, saved.text
        assert saved.json()["annotation"]["words"][0]["text"] == "Hello"
        assert client.get(f"/api/dataset/songs/{song['id']}").json()["draft"]["annotation"] == annotation

        lyrics = client.patch(f"/api/dataset/songs/{song['id']}/lyrics", json={"lyrics": "New lyrics"})
        assert lyrics.status_code == 200, lyrics.text
        updated = client.get(f"/api/dataset/songs/{song['id']}").json()
        assert updated["lyrics"] == "New lyrics"
        assert updated["draft"] is None

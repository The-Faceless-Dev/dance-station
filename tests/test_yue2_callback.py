from __future__ import annotations

import json
from pathlib import Path

from autotransition.yue2 import callback


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"id": "artifact-id"}).encode("utf-8")


def test_artifact_upload_uses_launcher_audio_and_metadata_roles(monkeypatch, tmp_path: Path) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response()

    monkeypatch.setattr(callback, "urlopen", fake_urlopen)
    audio = tmp_path / "audio.wav"
    metadata = tmp_path / "runtime-metadata.json"
    audio.write_bytes(b"wav")
    metadata.write_text("{}", encoding="utf-8")

    assert callback._upload_artifact("https://launcher.test/artifacts", "token", audio, {"name": "audio.wav"}) == "artifact-id"
    assert callback._upload_artifact("https://launcher.test/artifacts", "token", metadata, {"name": "runtime-metadata.json"}) == "artifact-id"

    audio_request, _ = requests[0]
    metadata_request, _ = requests[1]
    assert audio_request.get_header("X-artifact-role") == "audio"
    assert audio_request.get_header("X-artifact-variant") == "merged"
    assert metadata_request.get_header("X-artifact-role") == "metadata"
    assert metadata_request.get_header("X-artifact-variant") is None

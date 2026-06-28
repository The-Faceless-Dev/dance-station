from pathlib import Path
from types import SimpleNamespace

from autotransition.library.publish import LibraryPublisher, LibraryPublishSettings
from autotransition.library.schema import LibraryFile, LibraryItem


class FakeResponse:
    def __init__(self, payload: dict, ok: bool = True) -> None:
        self._payload = payload
        self.is_success = ok
        self.text = "error"

    def json(self) -> dict:
        return self._payload


class FakeClient:
    calls: list[tuple[str, str]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append(("POST", url))
        if url.endswith("/publish/items"):
            return FakeResponse({"item": {"id": "remote-1", "files": []}})
        if url.endswith("/files"):
            return FakeResponse({"item": {"id": "remote-1", "files": [{"id": "file-1"}]}})
        if url.endswith("/publish"):
            return FakeResponse({"item": {"id": "remote-1", "status": "published", "visibility": "public", "files": [{"id": "file-1"}]}})
        return FakeResponse({})

    def delete(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append(("DELETE", url))
        return FakeResponse({"item": {"id": "remote-1", "files": []}})


def test_library_publisher_uploads_and_publishes(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    item = LibraryItem(
        id="local-1",
        kind="generation",
        title="Clip",
        files=[LibraryFile(role="audio", mime_type="audio/wav", size_bytes=audio.stat().st_size, path=str(audio))],
    )
    FakeClient.calls = []
    monkeypatch.setattr("autotransition.library.publish.httpx", SimpleNamespace(Client=FakeClient))

    result = LibraryPublisher(LibraryPublishSettings(site_url="http://site.test", token="token")).publish(item)

    assert result["remote_item_id"] == "remote-1"
    assert result["remote_status"] == "published"
    assert result["file_count"] == 1
    assert ("DELETE", "http://site.test/api/library/publish/items/remote-1/files") in FakeClient.calls

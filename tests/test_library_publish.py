from pathlib import Path
from types import SimpleNamespace

from autotransition.library.publish import LibraryPublisher, LibraryPublishSettings, PublicLibraryClient
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


class FakeImportClient:
    def __init__(self, timeout: float, follow_redirects: bool = False) -> None:
        self.timeout = timeout
        self.follow_redirects = follow_redirects

    def __enter__(self) -> "FakeImportClient":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/api/library/remote-1"):
            return FakeResponse(
                {
                    "item": {
                        "id": "remote-1",
                        "kind": "generation",
                        "title": "Remote Clip",
                        "tags": ["public"],
                        "files": [
                            {
                                "id": "file-1",
                                "role": "audio",
                                "mimeType": "audio/wav",
                                "publicUrl": "http://cdn.test/clip.wav",
                                "metadata": {"originalName": "clip.wav"},
                            }
                        ],
                        "creator": {"displayName": "Creator"},
                    }
                }
            )
        response = FakeResponse({}, ok=True)
        response.content = b"audio"
        response.status_code = 200
        return response


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


def test_public_library_client_imports_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("autotransition.library.publish.httpx", SimpleNamespace(Client=FakeImportClient))

    item = PublicLibraryClient(LibraryPublishSettings(site_url="http://site.test")).import_item("remote-1", root=tmp_path)

    assert item.id == "imported-remote-1"
    assert item.title == "Remote Clip"
    assert item.metadata["imported"] is True
    assert item.metadata["creator"]["display_name"] == "Creator"
    assert len(item.files) == 1
    assert Path(item.files[0].path).read_bytes() == b"audio"

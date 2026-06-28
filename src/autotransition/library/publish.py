from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from autotransition.library.schema import LibraryItem


@dataclass
class LibraryPublishSettings:
    site_url: str = "http://localhost:3001"
    token: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.site_url.strip() and self.token.strip())


class LibraryPublishError(RuntimeError):
    pass


class LibraryPublisher:
    def __init__(self, settings: LibraryPublishSettings, timeout_seconds: float = 120.0) -> None:
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def publish(self, item: LibraryItem, *, publish_public: bool = True) -> dict[str, Any]:
        if not self.settings.configured:
            raise LibraryPublishError("Public library connection is not configured.")

        api_base = self.settings.site_url.rstrip("/") + "/api/library"
        headers = {"Authorization": f"Bearer {self.settings.token.strip()}"}
        payload = _item_payload(item, publish_public=publish_public)

        with httpx.Client(timeout=self.timeout_seconds) as client:
            item_response = _checked_json(
                client.post(f"{api_base}/publish/items", headers=headers, json=payload),
                "create public library item",
            )
            remote_item = item_response.get("item") or {}
            remote_item_id = str(remote_item.get("id") or "")
            if not remote_item_id:
                raise LibraryPublishError("Site did not return a remote library item id.")

            _checked_json(
                client.delete(f"{api_base}/publish/items/{remote_item_id}/files", headers=headers),
                "replace public library files",
            )

            uploaded_files: list[dict[str, Any]] = []
            for file_record in item.files:
                file_path = Path(file_record.path).expanduser()
                if not file_path.exists() or not file_path.is_file():
                    raise LibraryPublishError(f"File not found: {file_path}")
                with file_path.open("rb") as handle:
                    upload_response = _checked_json(
                        client.post(
                            f"{api_base}/publish/items/{remote_item_id}/files",
                            headers=headers,
                            data={
                                "role": file_record.role,
                                "metadata": json.dumps(file_record.metadata),
                            },
                            files={
                                "file": (
                                    file_path.name,
                                    handle,
                                    file_record.mime_type or "application/octet-stream",
                                )
                            },
                        ),
                        f"upload {file_path.name}",
                    )
                remote_item = upload_response.get("item") or remote_item
                uploaded_files = list(remote_item.get("files") or uploaded_files)

            if publish_public:
                published_response = _checked_json(
                    client.post(f"{api_base}/publish/items/{remote_item_id}/publish", headers=headers),
                    "publish public library item",
                )
                remote_item = published_response.get("item") or remote_item
                uploaded_files = list(remote_item.get("files") or uploaded_files)

        return {
            "remote_item_id": remote_item_id,
            "remote_status": remote_item.get("status") or "draft",
            "remote_visibility": remote_item.get("visibility") or "private",
            "file_count": len(uploaded_files),
            "public_url": f"{self.settings.site_url.rstrip('/')}/library",
            "remote_item": remote_item,
        }


def load_publish_settings(path: Path = Path("data/library/publish-connection.json")) -> LibraryPublishSettings:
    if not path.exists():
        return LibraryPublishSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return LibraryPublishSettings()
    return LibraryPublishSettings(site_url=str(data.get("site_url") or "http://localhost:3001"), token=str(data.get("token") or ""))


def save_publish_settings(settings: LibraryPublishSettings, path: Path = Path("data/library/publish-connection.json")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"site_url": settings.site_url.rstrip("/"), "token": settings.token}, indent=2),
        encoding="utf-8",
    )


def public_settings_response(settings: LibraryPublishSettings) -> dict[str, Any]:
    return {
        "site_url": settings.site_url,
        "has_token": bool(settings.token.strip()),
        "configured": settings.configured,
    }


def _item_payload(item: LibraryItem, *, publish_public: bool) -> dict[str, Any]:
    metadata = dict(item.metadata)
    metadata.pop("public_library", None)
    return {
        "localId": item.id,
        "visibility": "public" if publish_public else "private",
        "kind": item.kind,
        "title": item.title,
        "description": item.description or None,
        "tags": item.tags,
        "metadata": metadata,
        "sourceLineage": {**item.source_lineage, "localId": item.id},
        "license": item.license or None,
        "attribution": item.attribution or None,
    }


def _checked_json(response: httpx.Response, action: str) -> dict[str, Any]:
    if response.is_success:
        return response.json()
    try:
        body = response.json()
    except Exception:
        body = {"error": response.text}
    message = body.get("error") or body.get("detail") or response.text
    raise LibraryPublishError(f"Could not {action}: {message}")

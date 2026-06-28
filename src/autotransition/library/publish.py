from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from autotransition.library.schema import LibraryItem
from autotransition.library.schema import LibraryFile


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


class PublicLibraryClient:
    def __init__(self, settings: LibraryPublishSettings, timeout_seconds: float = 120.0) -> None:
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def list_items(self, *, kind: str | None = None, limit: int = 80) -> list[dict[str, Any]]:
        if not self.settings.site_url.strip():
            raise LibraryPublishError("Public library site URL is not configured.")
        params: dict[str, Any] = {"limit": limit}
        if kind and kind != "all":
            params["kind"] = kind
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = _checked_json(
                client.get(f"{self.settings.site_url.rstrip('/')}/api/library", params=params),
                "load public library",
            )
        return list(response.get("items") or [])

    def import_item(self, item_id: str, *, root: Path = Path("data/library/imports")) -> LibraryItem:
        if not self.settings.site_url.strip():
            raise LibraryPublishError("Public library site URL is not configured.")
        api_base = self.settings.site_url.rstrip("/")
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = _checked_json(client.get(f"{api_base}/api/library/{item_id}"), "load public library item")
            remote = response.get("item") or {}
            files = list(remote.get("files") or [])
            if not files:
                raise LibraryPublishError("Public library item has no downloadable files.")

            local_item_id = f"imported-{item_id}"
            import_dir = root / _safe_path_part(local_item_id)
            import_dir.mkdir(parents=True, exist_ok=True)
            local_files: list[LibraryFile] = []
            for remote_file in files:
                public_url = str(remote_file.get("publicUrl") or "")
                if not public_url:
                    continue
                file_id = str(remote_file.get("id") or "file")
                original_name = str((remote_file.get("metadata") or {}).get("originalName") or Path(public_url).name or file_id)
                local_path = import_dir / f"{_safe_path_part(file_id)}-{_safe_path_part(original_name)}"
                download = client.get(public_url)
                if not download.is_success:
                    raise LibraryPublishError(f"Could not download {public_url}: HTTP {download.status_code}")
                local_path.write_bytes(download.content)
                local_files.append(
                    LibraryFile(
                        id=f"import-{file_id}",
                        role=remote_file.get("role") or "audio",
                        mime_type=remote_file.get("mimeType") or "application/octet-stream",
                        size_bytes=local_path.stat().st_size,
                        storage_provider="local",
                        path=str(local_path),
                        public_url=public_url,
                        sha256=remote_file.get("sha256") or None,
                        metadata={
                            **(remote_file.get("metadata") or {}),
                            "remote_file_id": file_id,
                            "remote_public_url": public_url,
                        },
                    )
                )

        creator = remote.get("creator") or {}
        return LibraryItem(
            id=f"imported-{item_id}",
            owner_id=remote.get("ownerId") or None,
            visibility="local",
            status="draft",
            kind=remote.get("kind") or "audio",
            title=remote.get("title") or item_id,
            description=remote.get("description") or None,
            tags=list(remote.get("tags") or []),
            files=local_files,
            metadata={
                **(remote.get("metadata") or {}),
                "imported": True,
                "category": remote.get("kind") or "audio",
                "public_library": {
                    "remote_item_id": item_id,
                    "remote_status": remote.get("status") or "",
                    "remote_visibility": remote.get("visibility") or "",
                    "public_url": f"{api_base}/library",
                },
                "creator": {
                    "display_name": creator.get("displayName") or "",
                    "creator_slug": creator.get("creatorSlug") or "",
                    "avatar_url": creator.get("avatarUrl") or "",
                    "banner_url": creator.get("bannerUrl") or "",
                },
            },
            source_lineage={
                **(remote.get("sourceLineage") or {}),
                "remote_item_id": item_id,
                "imported_from": api_base,
            },
            license=remote.get("license") or None,
            attribution=remote.get("attribution") or None,
            created_at=remote.get("createdAt") or "",
            updated_at=remote.get("updatedAt") or "",
        )


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


def _safe_path_part(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")
    return (clean or "file")[:140]

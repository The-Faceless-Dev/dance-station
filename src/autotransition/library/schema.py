from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

LibraryVisibility = Literal["local", "private", "unlisted", "public"]
LibraryItemStatus = Literal["draft", "submitted", "approved", "rejected", "published", "archived"]
LibraryItemKind = Literal[
    "audio",
    "generation",
    "transition",
    "extraction",
    "stem",
    "merge",
    "edit",
    "instrument",
    "instrumenttrack",
    "dataset",
    "lokr",
    "rhythm_game",
    "voice",
    "speech",
    "sound_effect",
    "tool",
]
LibraryFileRole = Literal[
    "audio",
    "preview",
    "cover",
    "metadata",
    "dataset_manifest",
    "dataset_sample",
    "adapter_weights",
    "chart",
    "stem",
    "project",
    "voice_reference",
    "voice_embedding",
    "voice_model",
    "voice_index",
]
LibraryStorageProvider = Literal["local", "bunny"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class LibraryFile(BaseModel):
    id: str = Field(default_factory=lambda: f"file-{uuid4().hex[:12]}")
    role: LibraryFileRole
    mime_type: str
    size_bytes: int = Field(ge=0)
    storage_provider: LibraryStorageProvider = "local"
    path: str
    public_url: str | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class LibraryItem(BaseModel):
    id: str = Field(default_factory=lambda: f"library-{uuid4().hex[:12]}")
    owner_id: str | None = None
    visibility: LibraryVisibility = "local"
    status: LibraryItemStatus = "draft"
    kind: LibraryItemKind
    title: str = Field(min_length=1, max_length=160)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    files: list[LibraryFile] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_lineage: dict[str, Any] = Field(default_factory=dict)
    license: str | None = None
    attribution: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


def library_item_from_editor_asset(asset: dict[str, Any]) -> LibraryItem | None:
    audio_path_raw = str(asset.get("audio_path") or "")
    if not audio_path_raw:
        return None

    audio_path = Path(audio_path_raw).expanduser()
    if not audio_path.exists() or not audio_path.is_file():
        return None

    category = str(asset.get("category") or "audio")
    item_id = str(asset.get("asset_id") or audio_path.stem)
    title = str(asset.get("label") or item_id).strip() or item_id
    created_at = str(asset.get("created_at") or "") or utc_now_iso()

    return LibraryItem(
        id=item_id,
        visibility="local",
        status="draft",
        kind=_category_to_kind(category),
        title=title,
        files=[
            LibraryFile(
                role="audio",
                mime_type=audio_mime_type_for_path(audio_path),
                size_bytes=audio_path.stat().st_size,
                path=str(audio_path),
                metadata={
                    "duration_seconds": asset.get("duration_seconds") or 0,
                    "audio_url": asset.get("audio_url") or "",
                },
            )
        ],
        metadata={
            "category": category,
            "metadata_path": asset.get("metadata_path") or "",
            "message": asset.get("message") or "",
            "source_path": asset.get("source_path") or "",
        },
        source_lineage={
            "source_asset_id": asset.get("source_asset_id") or "",
        },
        created_at=created_at,
        updated_at=created_at,
    )


def _category_to_kind(category: str) -> LibraryItemKind:
    allowed: set[str] = {
        "audio",
        "generation",
        "transition",
        "extraction",
        "stem",
        "merge",
        "edit",
        "instrument",
        "instrumenttrack",
        "dataset",
        "lokr",
        "rhythm_game",
        "voice",
        "speech",
        "sound_effect",
        "tool",
    }
    return category if category in allowed else "audio"  # type: ignore[return-value]


def audio_mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"

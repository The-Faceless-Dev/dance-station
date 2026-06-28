from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from autotransition.library.schema import LibraryItem, library_item_from_editor_asset, utc_now_iso


class LocalLibraryIndex:
    def __init__(self, root: Path = Path("data/library")) -> None:
        self.root = root
        self.index_path = root / "index.json"
        self.items_dir = root / "items"

    def list_items(self) -> list[LibraryItem]:
        items: list[LibraryItem] = []
        for item_id in self._index_ids():
            item = self.read_item(item_id)
            if item is not None:
                items.append(item)
        return sorted(items, key=lambda item: item.updated_at or item.created_at, reverse=True)

    def read_item(self, item_id: str) -> LibraryItem | None:
        manifest_path = self._manifest_path(item_id)
        if not manifest_path.exists():
            return None
        try:
            return LibraryItem.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def write_item(self, item: LibraryItem) -> LibraryItem:
        manifest_path = self._manifest_path(item.id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(_model_dump(item), indent=2), encoding="utf-8")
        self._write_index(sorted({*self._index_ids(), item.id}))
        return item

    def reindex_from_editor_assets(self, assets: Iterable[dict[str, Any]]) -> list[LibraryItem]:
        return self.reindex_items(item for asset in assets if (item := library_item_from_editor_asset(asset)) is not None)

    def reindex_items(self, scanned_items: Iterable[LibraryItem]) -> list[LibraryItem]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.items_dir.mkdir(parents=True, exist_ok=True)

        item_ids: list[str] = []
        for scanned in scanned_items:
            existing = self.read_item(scanned.id)
            item = _merge_reindexed_item(existing, scanned)
            self.write_item(item)
            item_ids.append(item.id)

        self._write_index(sorted(set(item_ids)))
        return self.list_items()

    def update_item(self, item_id: str, updates: dict[str, Any]) -> LibraryItem:
        item = self.read_item(item_id)
        if item is None:
            raise FileNotFoundError(f"Library item not found: {item_id}")

        title = str(updates.get("title") or item.title).strip()
        if title:
            item.title = title
        if "description" in updates:
            description = updates.get("description")
            item.description = str(description).strip() if description else None
        if "tags" in updates:
            tags = updates.get("tags") or []
            if isinstance(tags, list):
                item.tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        if "license" in updates:
            license_value = updates.get("license")
            item.license = str(license_value).strip() if license_value else None
        if "attribution" in updates:
            attribution = updates.get("attribution")
            item.attribution = str(attribution).strip() if attribution else None

        item.updated_at = utc_now_iso()
        return self.write_item(item)

    def update_publish_metadata(self, item_id: str, publish_metadata: dict[str, Any]) -> LibraryItem:
        item = self.read_item(item_id)
        if item is None:
            raise FileNotFoundError(f"Library item not found: {item_id}")
        item.metadata["public_library"] = publish_metadata
        item.updated_at = utc_now_iso()
        return self.write_item(item)

    def _index_ids(self) -> list[str]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        ids = data.get("items") if isinstance(data, dict) else []
        if not isinstance(ids, list):
            return []
        return [str(item_id) for item_id in ids if str(item_id).strip()]

    def _write_index(self, item_ids: list[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps({"version": 1, "updated_at": utc_now_iso(), "items": item_ids}, indent=2),
            encoding="utf-8",
        )

    def _manifest_path(self, item_id: str) -> Path:
        return self.items_dir / _safe_item_id(item_id) / "manifest.json"


def _merge_reindexed_item(existing: LibraryItem | None, scanned: LibraryItem) -> LibraryItem:
    if existing is None:
        return scanned

    scanned.title = existing.title or scanned.title
    scanned.description = existing.description
    scanned.tags = existing.tags
    scanned.license = existing.license
    scanned.attribution = existing.attribution
    scanned.visibility = existing.visibility
    scanned.status = existing.status
    scanned.created_at = existing.created_at or scanned.created_at
    scanned.updated_at = utc_now_iso()
    scanned.metadata = {**scanned.metadata, **existing.metadata}
    scanned.source_lineage = {**scanned.source_lineage, **existing.source_lineage}
    return scanned


def _safe_item_id(item_id: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in item_id).strip("._")
    if not clean:
        raise ValueError("Library item id is empty.")
    return clean[:160]


def _model_dump(item: LibraryItem) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()

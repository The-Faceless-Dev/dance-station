"""Persistent storage for source separation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotransition.library.schema import LibraryFile, LibraryItem, audio_mime_type_for_path, utc_now_iso


def root() -> Path:
    return Path("data/source-separation")


def generations_root() -> Path:
    return root() / "generations"


def generation_path(generation_id: str) -> Path:
    safe_id = Path(generation_id).name
    if not safe_id or safe_id != generation_id:
        raise ValueError("Invalid source separation generation id.")
    return generations_root() / safe_id / "generation.json"


def read_generation(generation_id: str) -> dict[str, Any]:
    path = generation_path(generation_id)
    if not path.exists():
        raise FileNotFoundError(f"Source separation generation not found: {generation_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata_path"] = str(path)
    return payload


def write_generation(payload: dict[str, Any]) -> dict[str, Any]:
    path = generation_path(str(payload["generation_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["metadata_path"] = str(path)
    return payload


def list_generations() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for metadata_path in generations_root().glob("*/generation.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["metadata_path"] = str(metadata_path)
        items.append(payload)
    return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def _primary_output_path(metadata: dict[str, Any]) -> Path:
    instrumental = Path(str(metadata.get("instrumental_audio_path") or "")).expanduser()
    if instrumental.exists():
        return instrumental
    generated = Path(str(metadata.get("generated_audio_path") or "")).expanduser()
    return generated


def _stem_item(
    *,
    generation_id: str,
    metadata: dict[str, Any],
    audio_path: Path,
    metadata_file_path: Path,
    source_label: str,
    model_filename: str,
    asset_suffix: str,
    stem_name: str,
    audio_metadata_key: str,
) -> LibraryItem | None:
    if not audio_path.exists() or not audio_path.is_file():
        return None

    safe_stem = stem_name.strip().lower() or "stem"
    safe_suffix = asset_suffix.strip().lower() or safe_stem
    label = str(metadata.get("label") or generation_id).strip() or generation_id
    item_title = f"{label}_{safe_suffix}"
    stem_path = audio_path
    return LibraryItem(
        id=f"{generation_id}_{safe_suffix}",
        visibility="local",
        status="draft",
        kind="stem",
        title=item_title,
        description=f"{source_label} · {model_filename}".strip(" ·") or None,
        files=[
            LibraryFile(
                role="audio",
                mime_type=audio_mime_type_for_path(stem_path),
                size_bytes=stem_path.stat().st_size,
                path=str(stem_path),
                metadata={
                    "stem": safe_stem,
                    "source_label": source_label,
                    "model_filename": model_filename,
                    "output_format": metadata.get("output_format") or stem_path.suffix.lstrip(".") or "wav",
                },
            ),
            LibraryFile(
                role="metadata",
                mime_type="application/json",
                size_bytes=metadata_file_path.stat().st_size if metadata_file_path.exists() else 0,
                path=str(metadata_file_path),
            ),
        ],
        metadata={
            "category": "stem",
            "type": metadata.get("type") or "separation",
            "model_filename": model_filename,
            "source_label": source_label,
            "source_path": metadata.get("source_path") or "",
            "output_format": metadata.get("output_format") or stem_path.suffix.lstrip(".") or "wav",
            "primary_stem": safe_stem,
            audio_metadata_key: str(stem_path),
            "source_asset_id": metadata.get("source_asset_id") or "",
            "source_asset_category": metadata.get("source_asset_category") or "",
        },
        source_lineage={
            "source_asset_id": metadata.get("source_asset_id") or "",
            "source_asset_category": metadata.get("source_asset_category") or "",
            "generation_id": generation_id,
            "stem": safe_stem,
        },
        created_at=str(metadata.get("created_at") or utc_now_iso()),
        updated_at=str(metadata.get("created_at") or utc_now_iso()),
    )


def library_items_from_generation(metadata: dict[str, Any]) -> list[LibraryItem]:
    generation_id = str(metadata.get("generation_id") or "")
    primary_audio = _primary_output_path(metadata)
    vocals_audio = Path(str(metadata.get("vocals_audio_path") or "")).expanduser()
    if not generation_id or not primary_audio.exists() or not primary_audio.is_file():
        return []

    metadata_file_path = Path(str(metadata.get("metadata_path") or generation_path(generation_id))).expanduser()
    model_filename = str(metadata.get("model_filename") or "").strip()
    source_label = str(metadata.get("source_label") or metadata.get("source_path") or "").strip()
    vocals_file = vocals_audio if vocals_audio.exists() and vocals_audio.is_file() else None
    label = str(metadata.get("label") or generation_id).strip() or generation_id
    items: list[LibraryItem] = []
    instrumental_item = _stem_item(
        generation_id=generation_id,
        metadata=metadata,
        audio_path=primary_audio,
        metadata_file_path=metadata_file_path,
        source_label=source_label,
        model_filename=model_filename,
        asset_suffix="instrumental",
        stem_name="instrumental",
        audio_metadata_key="instrumental_audio_path",
    )
    vocals_item = _stem_item(
        generation_id=generation_id,
        metadata=metadata,
        audio_path=vocals_file or Path(),
        metadata_file_path=metadata_file_path,
        source_label=source_label,
        model_filename=model_filename,
        asset_suffix="vocal",
        stem_name="vocals",
        audio_metadata_key="vocals_audio_path",
    )
    if instrumental_item is not None:
        items.append(instrumental_item)
    if vocals_item is not None:
        items.append(vocals_item)
    return items


def library_item_from_generation(metadata: dict[str, Any]) -> LibraryItem | None:
    items = library_items_from_generation(metadata)
    return items[0] if items else None

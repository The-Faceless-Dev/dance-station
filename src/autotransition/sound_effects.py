"""TangoFlux sound effect generation storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotransition.library.schema import LibraryFile, LibraryItem, audio_mime_type_for_path, utc_now_iso


def root() -> Path:
    return Path("data/sound-effects")


def generations_root() -> Path:
    return root() / "generations"


def generation_path(generation_id: str) -> Path:
    safe_id = Path(generation_id).name
    if not safe_id or safe_id != generation_id:
        raise ValueError("Invalid sound effect generation id.")
    return generations_root() / safe_id / "generation.json"


def read_generation(generation_id: str) -> dict[str, Any]:
    path = generation_path(generation_id)
    if not path.exists():
        raise FileNotFoundError(f"Sound effect generation not found: {generation_id}")
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


def library_item_from_generation(metadata: dict[str, Any]) -> LibraryItem | None:
    generation_id = str(metadata.get("generation_id") or "")
    audio_path = Path(str(metadata.get("generated_audio_path") or "")).expanduser()
    if not generation_id or not audio_path.exists() or not audio_path.is_file():
        return None

    metadata_file_path = Path(str(metadata.get("metadata_path") or generation_path(generation_id))).expanduser()
    label = str(metadata.get("label") or generation_id).strip() or generation_id
    prompt = str(metadata.get("prompt") or "").strip()
    duration_seconds = float(metadata.get("duration_seconds") or 0)
    steps = metadata.get("steps")
    return LibraryItem(
        id=generation_id,
        visibility="local",
        status="draft",
        kind="sound_effect",
        title=label,
        description=prompt[:600] or None,
        files=[
            LibraryFile(
                role="audio",
                mime_type=audio_mime_type_for_path(audio_path),
                size_bytes=audio_path.stat().st_size,
                path=str(audio_path),
                metadata={
                    "duration_seconds": duration_seconds,
                    "prompt": prompt,
                    "steps": steps or 0,
                    "generation_type": metadata.get("type") or "sound_effect",
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
            "category": "sound_effect",
            "type": metadata.get("type") or "sound_effect",
            "prompt": prompt,
            "duration_seconds": duration_seconds,
            "steps": steps or 0,
            "output_format": metadata.get("output_format") or "wav",
        },
        created_at=str(metadata.get("created_at") or utc_now_iso()),
        updated_at=str(metadata.get("created_at") or utc_now_iso()),
    )

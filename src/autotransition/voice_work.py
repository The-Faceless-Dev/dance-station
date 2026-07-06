"""Seed-VC target voice and conversion storage."""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from autotransition.library.schema import LibraryFile, LibraryItem, utc_now_iso

VOICE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm"}
VOICE_EMBEDDING_EXTENSIONS = {".npy", ".npz"}


def root() -> Path:
    return Path("data/seed-vc")


def voices_root() -> Path:
    return root() / "target-voices"


def generations_root() -> Path:
    return root() / "generations"


def voice_path(voice_id: str) -> Path:
    return voices_root() / safe_id(voice_id, "target voice") / "voice.json"


def generation_path(generation_id: str) -> Path:
    return generations_root() / safe_id(generation_id, "conversion") / "generation.json"


def safe_id(value: str, label: str) -> str:
    clean = Path(value).name
    if not clean or clean != value:
        raise ValueError(f"Invalid {label} id.")
    return clean


def safe_label_stem(label: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    return clean or fallback


def read_voice(voice_id: str) -> dict[str, Any]:
    path = voice_path(voice_id)
    if not path.exists():
        raise FileNotFoundError(f"Target voice not found: {voice_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata_path"] = str(path)
    return payload


def write_voice(payload: dict[str, Any]) -> dict[str, Any]:
    path = voice_path(str(payload["voice_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["metadata_path"] = str(path)
    return payload


def update_voice(voice_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    payload = read_voice(voice_id)
    payload.update(updates)
    payload["updated_at"] = utc_now_iso()
    return write_voice(payload)


def delete_voice(voice_id: str) -> None:
    shutil.rmtree(voices_root() / safe_id(voice_id, "target voice"), ignore_errors=True)


def voice_runtime_model_name(metadata: dict[str, Any]) -> str:
    return str(metadata.get("voice_id") or "target-voice")


def voice_runtime_index_path(metadata: dict[str, Any]) -> str:
    return ""


def read_generation(generation_id: str) -> dict[str, Any]:
    path = generation_path(generation_id)
    if not path.exists():
        raise FileNotFoundError(f"Conversion not found: {generation_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata_path"] = str(path)
    return payload


def write_generation(payload: dict[str, Any]) -> dict[str, Any]:
    path = generation_path(str(payload["generation_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["metadata_path"] = str(path)
    return payload


def list_voices() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for metadata_path in voices_root().glob("*/voice.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["metadata_path"] = str(metadata_path)
        items.append(payload)
    return sorted(items, key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)


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


def _copy_voice_reference(source_path: Path, target_dir: Path) -> dict[str, Any]:
    suffix = source_path.suffix.lower()
    target_path = target_dir / f"{safe_label_stem(source_path.stem, 'reference')}{suffix}"
    if target_path.exists():
        target_path = target_dir / f"{safe_label_stem(source_path.stem, 'reference')}-{uuid4().hex[:6]}{suffix}"
    shutil.copyfile(source_path, target_path)
    return file_summary(target_path, role="voice_reference")


def create_voice_asset(
    *,
    label: str,
    language: str = "auto",
    description: str = "",
    reference_paths: list[Path],
    embedding_paths: list[Path] | None = None,
    source_asset_id: str = "",
    source_asset_label: str = "",
    source_asset_category: str = "",
) -> dict[str, Any]:
    if not reference_paths:
        raise ValueError("Add at least one reference audio file for the target voice.")
    voice_id = f"voice-{uuid4().hex[:12]}"
    target_dir = voices_root() / voice_id
    references_dir = target_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)

    stored_references = [_copy_voice_reference(source_path, references_dir) for source_path in reference_paths]
    timestamp = utc_now_iso()
    payload = {
        "voice_id": voice_id,
        "label": label.strip(),
        "description": description.strip(),
        "language": language.strip() or "auto",
        "voice_kind": "target_voice",
        "voice_dir": str(target_dir),
        "reference_files": stored_references,
        "embedding_files": [],
        "source_asset_id": source_asset_id,
        "source_asset_label": source_asset_label,
        "source_asset_category": source_asset_category,
        "preview_audio_path": stored_references[0]["path"] if stored_references else "",
        "training_status": "ready",
        "training_error": "",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    return write_voice(payload)


def create_generation_record(
    *,
    label: str,
    text: str,
    language: str,
    voice: dict[str, Any] | None,
    output_audio_path: Path,
    runtime_payload: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return create_voice_render_record(
        label=label,
        voice=voice,
        source_path=str(runtime_payload.get("source_path") or ""),
        output_audio_path=output_audio_path,
        runtime_payload=runtime_payload,
        render_type="conversion",
        text=text,
        language=language,
        extra=extra,
    )


def create_voice_render_record(
    *,
    label: str,
    voice: dict[str, Any] | None,
    source_path: str,
    output_audio_path: Path,
    runtime_payload: dict[str, Any],
    render_type: str,
    mode: str = "singing",
    text: str = "",
    language: str = "auto",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation_id = f"conversion-{uuid4().hex[:12]}"
    target_dir = generations_root() / generation_id
    target_dir.mkdir(parents=True, exist_ok=True)
    final_audio_path = target_dir / output_audio_path.name
    if output_audio_path.resolve() != final_audio_path.resolve():
        shutil.move(str(output_audio_path), final_audio_path)
    timestamp = utc_now_iso()
    payload = {
        "generation_id": generation_id,
        "label": label.strip(),
        "text": text,
        "language": language,
        "voice_id": str((voice or {}).get("voice_id") or ""),
        "voice_label": str((voice or {}).get("label") or ""),
        "output_audio_path": str(final_audio_path),
        "duration_seconds": 0,
        "render_type": render_type,
        "mode": mode,
        "source_path": source_path,
        "runtime_payload": runtime_payload,
        "created_at": timestamp,
    }
    if extra:
        payload.update(extra)
    return write_generation(payload)


def file_summary(path: Path, *, role: str) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "role": role,
        "path": str(path),
        "mime_type": mime_type,
        "size_bytes": path.stat().st_size,
        "name": path.name,
    }


def library_item_from_voice(metadata: dict[str, Any]) -> LibraryItem | None:
    voice_id = str(metadata.get("voice_id") or "")
    voice_dir = Path(str(metadata.get("voice_dir") or "")).expanduser()
    if not voice_id or not voice_dir.exists():
        return None

    files: list[LibraryFile] = [
        LibraryFile(
            role="metadata",
            mime_type="application/json",
            size_bytes=Path(str(metadata["metadata_path"])).stat().st_size if metadata.get("metadata_path") else 0,
            path=str(metadata.get("metadata_path") or voice_path(voice_id)),
        )
    ]
    for reference in metadata.get("reference_files", []):
        path = Path(str(reference.get("path") or "")).expanduser()
        if not path.exists():
            continue
        files.append(
            LibraryFile(
                role="voice_reference",
                mime_type=str(reference.get("mime_type") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"),
                size_bytes=int(reference.get("size_bytes") or path.stat().st_size),
                path=str(path),
                metadata={"label": path.stem},
            )
        )
    return LibraryItem(
        id=voice_id,
        visibility="local",
        status="draft",
        kind="voice",
        title=str(metadata.get("label") or voice_id),
        description=str(metadata.get("description") or "") or None,
        files=files,
        metadata={
            "category": "target_voice",
            "voice_kind": "target_voice",
            "language": str(metadata.get("language") or "auto"),
            "voice_dir": str(voice_dir),
            "voice_status": str(metadata.get("training_status") or "ready"),
            "source_asset_id": str(metadata.get("source_asset_id") or ""),
            "source_asset_label": str(metadata.get("source_asset_label") or ""),
            "source_asset_category": str(metadata.get("source_asset_category") or ""),
        },
        created_at=str(metadata.get("created_at") or utc_now_iso()),
        updated_at=str(metadata.get("updated_at") or utc_now_iso()),
    )


def library_item_from_generation(metadata: dict[str, Any]) -> LibraryItem | None:
    generation_id = str(metadata.get("generation_id") or "")
    audio_path = Path(str(metadata.get("output_audio_path") or "")).expanduser()
    if not generation_id or not audio_path.exists() or not audio_path.is_file():
        return None
    metadata_file_path = Path(str(metadata.get("metadata_path") or generation_path(generation_id))).expanduser()
    return LibraryItem(
        id=generation_id,
        visibility="local",
        status="draft",
        kind="speech",
        title=str(metadata.get("label") or generation_id),
        description=str(metadata.get("text") or "")[:600] or None,
        files=[
            LibraryFile(
                role="audio",
                mime_type=mimetypes.guess_type(audio_path.name)[0] or "audio/wav",
                size_bytes=audio_path.stat().st_size,
                path=str(audio_path),
                metadata={
                    "duration_seconds": metadata.get("duration_seconds") or 0,
                    "voice_id": metadata.get("voice_id") or "",
                    "voice_label": metadata.get("voice_label") or "",
                    "render_type": metadata.get("render_type") or "conversion",
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
            "category": "seed_vc",
            "text": str(metadata.get("text") or ""),
            "language": str(metadata.get("language") or "auto"),
            "voice_id": str(metadata.get("voice_id") or ""),
            "voice_label": str(metadata.get("voice_label") or ""),
            "source_path": str(metadata.get("source_path") or ""),
            "request_id": str(metadata.get("request_id") or ""),
        },
        created_at=str(metadata.get("created_at") or utc_now_iso()),
        updated_at=str(metadata.get("created_at") or utc_now_iso()),
    )


# Backward-compatible aliases for existing imports while the UI migrates.
list_target_voices = list_voices
list_conversions = list_generations
read_target_voice = read_voice
write_target_voice = write_voice
update_target_voice = update_voice
delete_target_voice = delete_voice
create_target_voice_asset = create_voice_asset
create_conversion_record = create_voice_render_record
library_item_from_target_voice = library_item_from_voice
library_item_from_conversion = library_item_from_generation

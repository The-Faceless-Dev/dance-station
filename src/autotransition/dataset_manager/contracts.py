"""Portable contracts and validation for dataset-manager annotations."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_PHONE_RE = re.compile(r"^[A-Za-z0-9_@'./-]+$")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(item: dict[str, Any], *, label: str, duration_ms: int) -> list[str]:
    errors: list[str] = []
    try:
        start = float(item["start_ms"])
        end = float(item["end_ms"])
    except (KeyError, TypeError, ValueError):
        return [f"{label} is missing numeric start_ms/end_ms"]
    if start < 0 or end > duration_ms:
        errors.append(f"{label} is outside the audio bounds")
    if end <= start:
        errors.append(f"{label} must have a positive duration")
    return errors


def validate_annotation(annotation: dict[str, Any], duration_ms: int) -> dict[str, Any]:
    """Validate a draft/revision without imposing a model frame grid."""

    errors: list[str] = []
    warnings: list[str] = []
    words = annotation.get("words") or []
    if not isinstance(words, list):
        errors.append("words must be a list")
        words = []
    seen: set[str] = set()
    previous_end = 0.0
    phoneme_count = 0
    for word_index, word in enumerate(words):
        if not isinstance(word, dict):
            errors.append(f"word {word_index} is not an object")
            continue
        word_id = str(word.get("id") or "")
        if not word_id or word_id in seen:
            errors.append(f"word {word_index} has a missing or duplicate id")
        seen.add(word_id)
        errors.extend(_interval(word, label=f"word {word_index}", duration_ms=duration_ms))
        try:
            start = float(word["start_ms"])
            end = float(word["end_ms"])
        except (KeyError, TypeError, ValueError):
            start = previous_end
            end = previous_end
        if start < previous_end - 0.5 and not word.get("allow_overlap"):
            warnings.append(f"word {word_index} overlaps the preceding word")
        previous_end = max(previous_end, end)
        phones = word.get("phonemes") or []
        if not isinstance(phones, list):
            errors.append(f"word {word_index} phonemes must be a list")
            continue
        for phone_index, phone in enumerate(phones):
            if not isinstance(phone, dict):
                errors.append(f"phoneme {word_index}:{phone_index} is not an object")
                continue
            phone_id = str(phone.get("id") or "")
            if not phone_id or phone_id in seen:
                errors.append(f"phoneme {word_index}:{phone_index} has a missing or duplicate id")
            seen.add(phone_id)
            symbol = str(phone.get("symbol") or "").strip()
            if not symbol or not _PHONE_RE.match(symbol):
                errors.append(f"phoneme {word_index}:{phone_index} has an invalid symbol")
            errors.extend(_interval(phone, label=f"phoneme {word_index}:{phone_index}", duration_ms=duration_ms))
            try:
                phone_start = float(phone["start_ms"])
                phone_end = float(phone["end_ms"])
                if phone_start < float(word.get("start_ms", 0)) - 0.5 or phone_end > float(word.get("end_ms", duration_ms)) + 0.5:
                    errors.append(f"phoneme {word_index}:{phone_index} escapes its parent word")
            except (TypeError, ValueError):
                pass
            if phone.get("alignment_status") in {"estimated", "unaligned"}:
                warnings.append(f"phoneme {word_index}:{phone_index} is not verified")
            phoneme_count += 1
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "word_count": len(words),
        "phoneme_count": phoneme_count,
    }

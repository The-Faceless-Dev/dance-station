"""Conversion between extraction output and the editable millisecond schema."""

from __future__ import annotations

from typing import Any

from autotransition.vevo2_dataset.alignment import AlignmentResult

from .contracts import SCHEMA_VERSION, new_id


def _origin(status: str) -> str:
    if status in {"aligned", "automatic"}:
        return "automatic_aligned"
    if status in {"estimated", "estimated_missing_phone"}:
        return "estimated"
    return status or "automatic"


def annotation_from_alignment(
    alignment: AlignmentResult,
    *,
    song_id: str,
    duration_ms: int,
    lyrics: str,
    language: str,
    feature_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    for word in alignment.words:
        phonemes = []
        for phone in word.phonemes:
            phonemes.append(
                {
                    "id": new_id("phone"),
                    "symbol": phone.phone,
                    "start_ms": round(phone.start * 1000, 3),
                    "end_ms": round(phone.end * 1000, 3),
                    "alignment_status": "aligned" if phone.timing_quality == "aligned" else "estimated",
                    "origin": _origin(phone.timing_quality),
                    "parent_word_id": None,
                    "syllable_index": phone.syllable_index,
                }
            )
        word_id = new_id("word")
        for phone in phonemes:
            phone["parent_word_id"] = word_id
        words.append(
            {
                "id": word_id,
                "text": word.text,
                "pronunciation_text": word.text,
                "start_ms": round(word.start * 1000, 3),
                "end_ms": round(word.end * 1000, 3),
                "kind": "sung",
                "origin": "automatic_aligned" if word.timing_quality == "aligned" else "automatic",
                "confidence": word.confidence,
                "phonemes": phonemes,
                "tags": [],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "song_id": song_id,
        "language": language,
        "duration_ms": duration_ms,
        "timing_reference": "song_start",
        "lyrics": lyrics,
        "lyrics_source": "user_provided" if lyrics else "asr_provisional",
        "alignment_engine": alignment.provider,
        "phone_timing_status": "aligned" if alignment.quality == "aligned" else "estimated_or_unaligned",
        "words": words,
        "features": feature_paths or {},
        "quality": {
            "machine_extraction": True,
            "approved": False,
            "warnings": list(alignment.warnings),
        },
    }

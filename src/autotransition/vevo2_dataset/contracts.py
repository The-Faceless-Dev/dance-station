"""Small, JSON-compatible contracts used by the dataset enrichment pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


class DatasetEnrichmentError(RuntimeError):
    """Base error for an enrichment run that cannot produce a valid artifact."""


class ProviderUnavailable(DatasetEnrichmentError):
    """Raised when an optional ASR/alignment provider is not installed."""


def normalize_text_token(value: str) -> str:
    """Normalize text for matching while preserving the original display text."""

    return re.sub(r"[^a-z0-9']+", "", value.lower())


def tokenize_words(text: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(text)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

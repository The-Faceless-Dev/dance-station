"""Small HTTP-facing helpers kept separate from the FastAPI route module."""

from __future__ import annotations

from pathlib import Path

from autotransition.dataset_manager.extraction import _probe


def _probe_upload(path: Path) -> dict[str, int]:
    value = _probe(path)
    return {key: int(value.get(key, 0)) for key in ("duration_ms", "sample_rate", "channels")}

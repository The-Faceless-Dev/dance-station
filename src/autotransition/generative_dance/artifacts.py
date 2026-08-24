"""Durable, inspectable artifact storage for the local POC."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_id_dir(self, category: str, item_id: str) -> Path:
        if not category or not item_id or Path(item_id).name != item_id or item_id in {".", ".."}:
            raise ValueError("invalid artifact id")
        path = self.root / category / item_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("artifact is outside the configured artifact root") from exc

    def resolve_relative(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact path is outside the configured artifact root") from exc
        return candidate

    def write_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def describe(self, path: Path, *, media_type: str | None = None) -> dict[str, Any]:
        return {
            "path": self.relative(path),
            "name": path.name,
            "sizeBytes": path.stat().st_size if path.is_file() else 0,
            "sha256": self.sha256(path) if path.is_file() else None,
            "mediaType": media_type,
        }

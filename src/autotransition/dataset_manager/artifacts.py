"""Filesystem artifact layout for dataset-manager records."""

from __future__ import annotations

import shutil
from pathlib import Path

from .contracts import sha256_file


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def song_root(self, song_id: str) -> Path:
        return self.root / "songs" / song_id

    def original_dir(self, song_id: str) -> Path:
        return self.song_root(song_id) / "original"

    def normalized_dir(self, song_id: str) -> Path:
        return self.song_root(song_id) / "normalized"

    def run_dir(self, song_id: str, run_id: str) -> Path:
        return self.song_root(song_id) / "extractions" / run_id

    def revision_dir(self, song_id: str, revision_id: str) -> Path:
        return self.song_root(song_id) / "revisions" / revision_id

    def export_dir(self) -> Path:
        return self.root / "exports"

    def store_upload(self, song_id: str, filename: str, source) -> Path:
        safe_name = Path(filename or "source_audio").name
        destination = self.original_dir(song_id) / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
        return destination

    def copy_artifact(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def describe(self, path: Path) -> dict[str, object]:
        path = path.resolve()
        return {
            "path": self.relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

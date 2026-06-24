"""Runtime state helpers for the local UI."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from autotransition.audio.ffmpeg import resolve_ffmpeg
from autotransition.models import repaint_capable_models


@dataclass(frozen=True)
class LogEntry:
    timestamp: str
    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class UiLog:
    """Small in-memory log buffer for local UI sessions."""

    def __init__(self, limit: int = 200) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=limit)

    def add(self, level: str, message: str) -> LogEntry:
        entry = LogEntry(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            level=level,
            message=message,
        )
        self._entries.appendleft(entry)
        return entry

    def entries(self) -> list[dict[str, str]]:
        return [entry.to_dict() for entry in self._entries]


def system_status(models_dir: Path = Path("models")) -> dict[str, object]:
    ffmpeg_path = resolve_ffmpeg()
    return {
        "python_version": sys.version.split()[0],
        "ffmpeg_available": ffmpeg_path is not None,
        "ffmpeg_path": ffmpeg_path,
        "models_dir": str(models_dir),
        "models_dir_exists": models_dir.exists(),
        "repaint_model_count": len(repaint_capable_models()),
        "cwd": str(Path.cwd()),
    }

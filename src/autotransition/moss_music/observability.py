from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MossMusicEventLogger:
    def __init__(self, path: Path, job_id: str):
        self.path = path
        self.job_id = job_id

    def emit(self, name: str, **fields: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {"event": name, "jobId": self.job_id, "createdAt": _now(), **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def exception(self, name: str, exc: BaseException, **fields: Any) -> None:
        self.emit(
            name,
            errorType=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            **fields,
        )

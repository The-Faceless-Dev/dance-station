from __future__ import annotations

import json
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class QwenImageEventLogger:
    def __init__(self, job_id: str, path: Path | None = None):
        self.job_id = job_id
        self.path = path
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        payload = {
            "event": event,
            "runtime": "qwen-image",
            "jobId": self.job_id,
            "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **fields,
        }
        line = json.dumps(payload, sort_keys=True, default=str)
        with self._lock:
            print(line, file=sys.stdout, flush=True)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return payload

    def exception(self, event: str, exc: BaseException, **fields: Any) -> dict[str, Any]:
        return self.emit(
            event,
            errorType=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            **fields,
        )

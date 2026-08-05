"""Structured, durable diagnostics for avatar worker jobs."""

from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager


_current_event_logger: contextvars.ContextVar["AvatarEventLogger | None"] = contextvars.ContextVar(
    "avatar_event_logger", default=None
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


class AvatarEventLogger:
    """Append JSONL events to a job and mirror them to container stdout."""

    def __init__(self, path: Path, *, job_id: str):
        self.path = path
        self.job_id = job_id
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            payload: dict[str, Any] = {
                "timestamp": utc_now(),
                "elapsedSeconds": round(time.monotonic() - self.started, 3),
                "sequence": self._sequence,
                "event": event,
                "jobId": self.job_id,
                "pid": os.getpid(),
                **{key: _json_value(value) for key, value in fields.items()},
            }
            line = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                # Diagnostics must never take down a paid job. The stdout copy
                # still preserves the event for the deployment log.
                print(
                    json.dumps(
                        {
                            "timestamp": utc_now(),
                            "event": "event_log_write_failed",
                            "jobId": self.job_id,
                            "path": str(self.path),
                            "error": str(exc),
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            print(line, flush=True)
            return payload

    def exception(self, event: str, exc: BaseException, **fields: Any) -> dict[str, Any]:
        return self.emit(
            event,
            errorType=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            **fields,
        )


def emit_worker_event(event: str, **fields: Any) -> dict[str, Any]:
    """Emit a process-level event when no job logger exists yet."""

    payload = {
        "timestamp": utc_now(),
        "event": event,
        "pid": os.getpid(),
        **{key: _json_value(value) for key, value in fields.items()},
    }
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), flush=True)
    return payload


def current_event_logger() -> AvatarEventLogger | None:
    return _current_event_logger.get()


@contextmanager
def use_event_logger(logger: AvatarEventLogger) -> Iterator[AvatarEventLogger]:
    token = _current_event_logger.set(logger)
    try:
        yield logger
    finally:
        _current_event_logger.reset(token)


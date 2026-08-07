"""Safe argv-template support for heavyweight external model runtimes."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Mapping, Sequence

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.resources import AvatarProcessError, run_command


def parse_command(template: str | Sequence[str] | None) -> tuple[str, ...]:
    if template is None:
        return ()
    if isinstance(template, str):
        return tuple(shlex.split(template, posix=os.name != "nt"))
    return tuple(str(value) for value in template)


def render_command(template: Sequence[str], **values: object) -> list[str]:
    rendered: list[str] = []
    replacements = {key: str(value) if value is not None else "" for key, value in values.items()}
    for token in template:
        try:
            token = token.format(**replacements)
        except KeyError as exc:
            raise AvatarAdapterError("adapter_command_invalid", f"unknown adapter placeholder: {exc}", retryable=False) from exc
        if token:
            rendered.append(token)
    return rendered


def run_adapter_command(
    template: Sequence[str],
    *,
    values: Mapping[str, object],
    cwd: Path | None,
    timeout_seconds: float,
    log_dir: Path | None = None,
    component: str = "avatar-adapter",
) -> None:
    command = render_command(template, **values)
    if not command:
        raise AvatarAdapterError("adapter_not_configured", "avatar adapter command is not configured", retryable=False)
    stdout_path = log_dir / f"{component}.stdout.log" if log_dir is not None else None
    stderr_path = log_dir / f"{component}.stderr.log" if log_dir is not None else None
    try:
        run_command(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            component=component,
        )
    except AvatarProcessError as exc:
        diagnostic = f"{exc}\n{exc.stderr[-4000:]}".lower()
        non_retryable_markers = (
            "no such file",
            "file not found",
            "modulenotfounderror",
            "cuda out of memory",
            "cuda is unavailable",
            "requires an nvidia",
            "out of memory",
            "permission denied",
            "authentication",
        )
        retryable = not any(marker in diagnostic for marker in non_retryable_markers)
        raise AvatarAdapterError(
            "adapter_process_failed",
            str(exc),
            retryable=retryable,
            details={
                "stderr": exc.stderr[-4000:],
                "stdout": exc.stdout[-4000:],
                "stdoutLog": str(stdout_path) if stdout_path is not None else None,
                "stderrLog": str(stderr_path) if stderr_path is not None else None,
                "command": command,
                "returnCode": exc.returncode,
            },
        ) from exc

"""Optional identity-audit command boundary for generated dance segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.avatar.adapters.command import parse_command, run_adapter_command

from .config import GenerativeDanceConfig


def audit_segment_identity(
    config: GenerativeDanceConfig,
    *,
    reference_image: Path,
    render_video: Path,
    output_dir: Path,
    segment_id: str,
    seed: int,
    attempt: int,
) -> dict[str, Any]:
    """Run a caller-provided scorer and require a numeric identity score."""

    command = parse_command(config.identity_audit_command)
    if not config.identity_audit_enabled:
        return {"enabled": False, "passed": True, "attempt": attempt, "seed": seed}
    if not command:
        raise AvatarAdapterError(
            "identity_audit_not_configured",
            "identity audit is enabled but no scorer command is configured",
            retryable=False,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "identity-audit.json"
    run_adapter_command(
        command,
        values={
            "reference_image": reference_image,
            "render_video": render_video,
            "output": audit_path,
            "output_dir": output_dir,
            "segment_id": segment_id,
            "seed": seed,
            "attempt": attempt,
            "threshold": config.identity_audit_threshold,
        },
        cwd=config.identity_audit_cwd,
        timeout_seconds=config.job_timeout_seconds,
        log_dir=output_dir,
        component="wan-identity-audit",
    )
    if not audit_path.is_file():
        raise AvatarAdapterError(
            "identity_audit_output_missing",
            "identity audit completed without identity-audit.json",
            retryable=False,
            details={"outputDir": str(output_dir)},
        )
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AvatarAdapterError(
            "identity_audit_output_invalid",
            f"identity audit output is not valid JSON: {exc}",
            retryable=False,
            details={"path": str(audit_path)},
        ) from exc
    if not isinstance(payload, dict):
        raise AvatarAdapterError(
            "identity_audit_output_invalid",
            "identity audit output must be a JSON object",
            retryable=False,
            details={"path": str(audit_path)},
        )
    try:
        score = float(payload["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AvatarAdapterError(
            "identity_audit_score_missing",
            "identity audit output must contain a numeric score",
            retryable=False,
            details={"path": str(audit_path)},
        ) from exc
    if not 0 <= score <= 1:
        raise AvatarAdapterError(
            "identity_audit_score_invalid",
            "identity audit score must be between 0 and 1",
            retryable=False,
            details={"score": score, "path": str(audit_path)},
        )
    return {
        **payload,
        "enabled": True,
        "score": score,
        "threshold": config.identity_audit_threshold,
        "passed": score >= config.identity_audit_threshold,
        "attempt": attempt,
        "seed": seed,
        "path": str(audit_path),
    }

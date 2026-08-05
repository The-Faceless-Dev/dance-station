from __future__ import annotations

import json
import sys
from pathlib import Path

from autotransition.avatar.observability import AvatarEventLogger, use_event_logger
from autotransition.avatar.resources import run_command


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_adapter_output_is_streamed_to_events_and_complete_logs(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    stdout_path = tmp_path / "adapter.stdout.log"
    stderr_path = tmp_path / "adapter.stderr.log"
    logger = AvatarEventLogger(event_path, job_id="job-observe")
    command = [
        sys.executable,
        "-c",
        "import sys; print('stdout-before'); print('stderr-before', file=sys.stderr); print('stdout-after')",
    ]

    with use_event_logger(logger):
        result = run_command(
            command,
            timeout_seconds=10,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            component="test-adapter",
        )

    assert result.returncode == 0
    assert "stdout-before" in stdout_path.read_text(encoding="utf-8")
    assert "stdout-after" in stdout_path.read_text(encoding="utf-8")
    assert "stderr-before" in stderr_path.read_text(encoding="utf-8")
    events = read_events(event_path)
    names = [event["event"] for event in events]
    assert names[0] == "adapter_process_started"
    assert "adapter_process_output" in names
    assert names[-1] == "adapter_process_finished"
    output_lines = [event["line"] for event in events if event["event"] == "adapter_process_output"]
    assert "stdout-before" in output_lines
    assert "stderr-before" in output_lines


def test_pipeline_persists_stage_and_terminal_events(tmp_path: Path) -> None:
    from tests.test_avatar_worker import make_pipeline, write_png
    from autotransition.avatar.contracts import AvatarRequest

    source = tmp_path / "source.png"
    write_png(source)
    pipeline, _ = make_pipeline(tmp_path, rig_failures=0)
    job = pipeline.create_job(AvatarRequest(description="an observable humanoid", reference_image=source))

    result = pipeline.run(AvatarRequest(description="an observable humanoid", reference_image=source), job_id=job.id)

    assert result.status == "succeeded"
    events = read_events(pipeline.store.event_log_path(job.id))
    names = [event["event"] for event in events]
    assert names[0] == "job_run_started"
    assert "job_progress_updated" in names
    assert "image_generation_finished" in names
    assert "mesh_validation_finished" in names
    assert "rig_validation_finished" in names
    assert "runtime_validation_finished" in names
    assert "finalization_finished" in names
    assert names[-1] == "job_run_finished"

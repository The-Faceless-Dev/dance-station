from pathlib import Path
from types import SimpleNamespace

from autotransition.config import RuntimeConfig
from autotransition.runtime.ace_step import (
    PortListener,
    RuntimeProcess,
    api_health,
    api_health_detail,
    build_runtime_env,
    build_debug_start_api_command,
    build_install_commands,
    build_start_api_command,
    build_uv_install_command,
    ensure_runtime_api,
    _parse_posix_port_listeners,
    read_runtime_pid,
    resolve_uv_executable,
    runtime_status,
    start_api_background,
    stop_runtime_process_tree,
)
import autotransition.runtime.ace_step as ace_step_runtime


def test_runtime_install_commands_target_runtime_folder(tmp_path: Path) -> None:
    config = RuntimeConfig(ace_step_dir=tmp_path / "ACE-Step-1.5")

    commands = build_install_commands(config)

    assert "uv/install.ps1" in commands[0]
    assert f"git clone https://github.com/ACE-Step/ACE-Step-1.5.git {config.ace_step_dir}" == commands[1]
    assert commands[-1] == "uv sync"


def test_uv_install_command_uses_shell_script_on_posix(monkeypatch) -> None:
    monkeypatch.setattr(ace_step_runtime.sys, "platform", "linux")

    command = build_uv_install_command()

    assert command == "curl -LsSf https://astral.sh/uv/install.sh | sh"


def test_uv_install_command_uses_powershell_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(ace_step_runtime.sys, "platform", "win32")

    command = build_uv_install_command()

    assert "powershell" in command
    assert "install.ps1" in command


def test_resolve_uv_executable_finds_posix_home_uv(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    uv = home / ".local" / "bin" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(ace_step_runtime.Path, "home", lambda: home)
    monkeypatch.setattr(ace_step_runtime.shutil, "which", lambda name: None)

    assert resolve_uv_executable() == uv


def test_runtime_start_command_uses_localhost_api(tmp_path: Path) -> None:
    config = RuntimeConfig(ace_step_dir=tmp_path / "ACE-Step-1.5", api_port=9001)

    command = build_debug_start_api_command(config)

    assert "run acestep-api" in command
    assert "--host 127.0.0.1" in command
    assert "--port 9001" in command


def test_runtime_user_start_command_is_simple() -> None:
    assert build_start_api_command() == "autotransition runtime start"


def test_runtime_status_reports_missing_install(tmp_path: Path) -> None:
    config = RuntimeConfig(ace_step_dir=tmp_path / "missing", api_port=65530)

    status = runtime_status(config)

    assert not status.installed
    assert not status.api_running
    assert "not installed" in status.message


def test_api_health_rejects_html_proxy_response(monkeypatch) -> None:
    class Response:
        status_code = 200
        text = "<!DOCTYPE html><html><title>502 | README | RunPod</title></html>"
        headers = {"content-type": "text/html; charset=utf-8"}

    monkeypatch.setitem(
        __import__("sys").modules,
        "httpx",
        SimpleNamespace(get=lambda *args, **kwargs: Response()),
    )

    assert not api_health()


def test_api_health_rejects_method_error(monkeypatch) -> None:
    class Response:
        status_code = 405
        text = '{"detail":"Method Not Allowed"}'
        headers = {"content-type": "application/json"}

    monkeypatch.setitem(
        __import__("sys").modules,
        "httpx",
        SimpleNamespace(get=lambda *args, **kwargs: Response()),
    )

    assert not api_health()


def test_api_health_detail_reports_unreachable_url(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setitem(
        __import__("sys").modules,
        "httpx",
        SimpleNamespace(get=fake_get),
    )

    detail = api_health_detail(RuntimeConfig(api_port=9911))

    assert not detail.running
    assert "127.0.0.1:9911/health" in detail.message
    assert "connection refused" in detail.message


def test_parse_posix_port_listeners_reads_ss_output() -> None:
    output = (
        "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n"
        'LISTEN 0 2048 127.0.0.1:8001 0.0.0.0:* users:(("python",pid=4675,fd=7))\n'
    )

    listeners = _parse_posix_port_listeners(output, 8001)

    assert listeners == [PortListener(pid=4675, command_line='LISTEN 0 2048 127.0.0.1:8001 0.0.0.0:* users:(("python",pid=4675,fd=7))')]


def test_ensure_runtime_api_reports_missing_install(tmp_path: Path) -> None:
    config = RuntimeConfig(ace_step_dir=tmp_path / "missing", api_port=65531)

    result = ensure_runtime_api(config)

    assert not result.started
    assert not result.already_running
    assert "autotransition setup" in result.message


def test_ensure_runtime_api_reports_port_conflict(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "ACE-Step-1.5"
    runtime_dir.mkdir()
    (runtime_dir / "pyproject.toml").write_text("[project]\nname = 'ace-step'\n", encoding="utf-8")
    config = RuntimeConfig(ace_step_dir=runtime_dir, api_port=8001)

    monkeypatch.setattr(ace_step_runtime, "api_health", lambda config: False)
    monkeypatch.setattr(ace_step_runtime, "find_runtime_processes", lambda config: [])
    monkeypatch.setattr(
        ace_step_runtime,
        "find_port_listeners",
        lambda config: [PortListener(pid=4675, command_line="python -m some.server")],
    )
    monkeypatch.setattr(
        ace_step_runtime,
        "start_api_background",
        lambda config: (_ for _ in ()).throw(AssertionError("should not start when port is occupied")),
    )

    result = ensure_runtime_api(config)

    assert not result.started
    assert not result.already_running
    assert result.pid == 4675
    assert "Port 8001 is already in use" in result.message
    assert "python -m some.server" in result.message


def test_runtime_env_disables_hf_transfer_by_default(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.delenv("AUTOTRANSITION_ALLOW_HF_TRANSFER", raising=False)
    monkeypatch.delenv("UV_LINK_MODE", raising=False)

    env = build_runtime_env()

    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert env["UV_LINK_MODE"] == "copy"
    assert env["UV_CACHE_DIR"].replace("\\", "/").endswith("data/runtime/uv-cache")
    assert env["TMPDIR"].replace("\\", "/").endswith("data/runtime/tmp")
    assert env["ACESTEP_CONFIG_PATH"] == "acestep-v15-turbo"
    assert env["ACESTEP_CONFIG_PATH2"] == "acestep-v15-base"


def test_runtime_env_allows_explicit_hf_transfer(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setenv("AUTOTRANSITION_ALLOW_HF_TRANSFER", "1")
    monkeypatch.setenv("UV_LINK_MODE", "clone")

    env = build_runtime_env()

    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert env["UV_LINK_MODE"] == "clone"


def test_run_install_retries_uv_sync_after_partial_venv_failure(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "ACE-Step-1.5"
    runtime_dir.mkdir()
    (runtime_dir / "pyproject.toml").write_text("[project]\nname = 'ace-step'\n", encoding="utf-8")
    venv = runtime_dir / ".venv"
    venv.mkdir()
    (venv / "partial.txt").write_text("partial", encoding="utf-8")
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ace_step_runtime, "resolve_uv_executable", lambda: Path("uv"))

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args == [str(Path("uv")), "sync", "--link-mode", "copy"] and len(calls) == 1:
            raise ace_step_runtime.subprocess.CalledProcessError(2, args)
        return ace_step_runtime.subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(ace_step_runtime.subprocess, "run", fake_run)

    ace_step_runtime.run_install(RuntimeConfig(ace_step_dir=runtime_dir))

    sync_calls = [call for call in calls if call[0] == [str(Path("uv")), "sync", "--link-mode", "copy"]]
    assert len(sync_calls) == 2
    assert not venv.exists()


def test_start_api_background_rotates_previous_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runtime_dir = tmp_path / "ACE-Step-1.5"
    runtime_dir.mkdir()
    (runtime_dir / "pyproject.toml").write_text("[project]\nname = 'ace-step'\n", encoding="utf-8")
    log_dir = tmp_path / "data/logs"
    log_dir.mkdir(parents=True)
    stdout_path = log_dir / "ace-step-api.log"
    stderr_path = log_dir / "ace-step-api.err.log"
    stdout_path.write_text("old stdout\n", encoding="utf-8")
    stderr_path.write_text("old stderr\nTraceback (most recent call last):\nKeyboardInterrupt\n", encoding="utf-8")

    class Process:
        pid = 1234

    def fake_popen(args, **kwargs):
        kwargs["stdout"].write(b"new stdout\n")
        kwargs["stderr"].write(b"new stderr\n")
        kwargs["stdout"].close()
        kwargs["stderr"].close()
        return Process()

    monkeypatch.setattr(ace_step_runtime, "resolve_uv_executable", lambda: Path("uv"))
    monkeypatch.setattr(ace_step_runtime.subprocess, "Popen", fake_popen)

    process = start_api_background(RuntimeConfig(ace_step_dir=runtime_dir))

    assert process.pid == 1234
    assert stdout_path.read_text(encoding="utf-8") == "new stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "new stderr\n"
    assert (log_dir / "ace-step-api.log.previous").read_text(encoding="utf-8") == "old stdout\n"
    assert "KeyboardInterrupt" in (log_dir / "ace-step-api.err.log.previous").read_text(encoding="utf-8")


def test_start_api_background_isolates_process_group(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runtime_dir = tmp_path / "ACE-Step-1.5"
    runtime_dir.mkdir()
    (runtime_dir / "pyproject.toml").write_text("[project]\nname = 'ace-step'\n", encoding="utf-8")
    popen_kwargs = {}

    class Process:
        pid = 1234

    def fake_popen(args, **kwargs):
        popen_kwargs.update(kwargs)
        kwargs["stdout"].close()
        kwargs["stderr"].close()
        return Process()

    monkeypatch.setattr(ace_step_runtime, "resolve_uv_executable", lambda: Path("uv"))
    monkeypatch.setattr(ace_step_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ace_step_runtime.sys, "platform", "linux")

    start_api_background(RuntimeConfig(ace_step_dir=runtime_dir))

    assert popen_kwargs["start_new_session"] is True


def test_ensure_runtime_api_reports_existing_unhealthy_process(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "ACE-Step-1.5"
    runtime_dir.mkdir()
    (runtime_dir / "pyproject.toml").write_text("[project]\nname = 'ace-step'\n", encoding="utf-8")
    config = RuntimeConfig(ace_step_dir=runtime_dir)

    monkeypatch.setattr(ace_step_runtime, "api_health", lambda config: False)
    monkeypatch.setattr(
        ace_step_runtime,
        "find_runtime_processes",
        lambda config: [RuntimeProcess(pid=1234, command_line="uv run acestep-api --port 8001")],
    )
    monotonic_values = iter([0, 21])
    monkeypatch.setattr(ace_step_runtime.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(ace_step_runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        ace_step_runtime,
        "start_api_background",
        lambda config: (_ for _ in ()).throw(AssertionError("should not start a duplicate runtime")),
    )

    result = ensure_runtime_api(config)

    assert not result.started
    assert not result.already_running
    assert result.pid == 1234
    assert "already running but the API is not reachable" in result.message


def test_ensure_runtime_api_stops_process_when_startup_times_out(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "ACE-Step-1.5"
    runtime_dir.mkdir()
    (runtime_dir / "pyproject.toml").write_text("[project]\nname = 'ace-step'\n", encoding="utf-8")
    config = RuntimeConfig(ace_step_dir=runtime_dir, api_startup_timeout_seconds=2)
    stopped = []

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(ace_step_runtime, "api_health", lambda config: False)
    monkeypatch.setattr(ace_step_runtime, "find_runtime_processes", lambda config: [])
    monkeypatch.setattr(ace_step_runtime, "start_api_background", lambda config: Process())
    monkeypatch.setattr(ace_step_runtime, "stop_runtime_process_tree", lambda pid, config: stopped.append(pid) or True)
    monotonic_values = iter([0, 1, 3])
    monkeypatch.setattr(ace_step_runtime.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(ace_step_runtime.time, "sleep", lambda seconds: None)

    result = ensure_runtime_api(config)

    assert not result.started
    assert not result.already_running
    assert result.pid == 4321
    assert stopped == [4321]
    assert "startup timeout" in result.message


def test_ensure_runtime_api_reports_startup_activity(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "ACE-Step-1.5"
    runtime_dir.mkdir()
    (runtime_dir / "pyproject.toml").write_text("[project]\nname = 'ace-step'\n", encoding="utf-8")
    log_dir = tmp_path / "data/logs"
    log_dir.mkdir(parents=True)
    (log_dir / "ace-step-api.err.log").write_text(
        "2026-06-18 INFO Initializing ACE-Step models\n",
        encoding="utf-8",
    )
    config = RuntimeConfig(ace_step_dir=runtime_dir, api_startup_timeout_seconds=2)
    messages = []

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ace_step_runtime, "api_health", lambda config: False)
    monkeypatch.setattr(ace_step_runtime, "find_runtime_processes", lambda config: [])
    monkeypatch.setattr(ace_step_runtime, "start_api_background", lambda config: Process())
    monkeypatch.setattr(ace_step_runtime, "stop_runtime_process_tree", lambda pid, config: True)
    monotonic_values = iter([0, 1, 3])
    monkeypatch.setattr(ace_step_runtime.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(ace_step_runtime.time, "sleep", lambda seconds: None)

    result = ensure_runtime_api(config, status_callback=messages.append)

    assert any("Started ACE-Step runtime process 4321" in message for message in messages)
    assert any("initializing" in message.lower() for message in messages)
    assert "Last runtime status: initializing" in result.message


def test_stop_runtime_process_tree_clears_matching_pid_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pid_path = tmp_path / "data/runtime/ace-step-api.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("1234", encoding="utf-8")

    class Completed:
        returncode = 0

    monkeypatch.setattr(ace_step_runtime.sys, "platform", "win32")
    monkeypatch.setattr(ace_step_runtime.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(ace_step_runtime, "find_runtime_processes", lambda config: [])

    stopped = stop_runtime_process_tree(1234)

    assert stopped
    assert read_runtime_pid() is None


def test_stop_runtime_process_tree_succeeds_when_processes_are_already_gone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pid_path = tmp_path / "data/runtime/ace-step-api.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("1234", encoding="utf-8")

    class Completed:
        returncode = 128

    monkeypatch.setattr(ace_step_runtime.sys, "platform", "win32")
    monkeypatch.setattr(ace_step_runtime.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(ace_step_runtime, "find_runtime_processes", lambda config: [])

    stopped = stop_runtime_process_tree(1234)

    assert stopped
    assert read_runtime_pid() is None

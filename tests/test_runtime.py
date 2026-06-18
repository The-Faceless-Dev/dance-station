from pathlib import Path
from types import SimpleNamespace

from autotransition.config import RuntimeConfig
from autotransition.runtime.ace_step import (
    RuntimeProcess,
    api_health,
    build_runtime_env,
    build_debug_start_api_command,
    build_install_commands,
    build_start_api_command,
    build_uv_install_command,
    ensure_runtime_api,
    read_runtime_pid,
    resolve_uv_executable,
    runtime_status,
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


def test_ensure_runtime_api_reports_missing_install(tmp_path: Path) -> None:
    config = RuntimeConfig(ace_step_dir=tmp_path / "missing", api_port=65531)

    result = ensure_runtime_api(config)

    assert not result.started
    assert not result.already_running
    assert "autotransition setup" in result.message


def test_runtime_env_disables_hf_transfer_by_default(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.delenv("AUTOTRANSITION_ALLOW_HF_TRANSFER", raising=False)

    env = build_runtime_env()

    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_runtime_env_allows_explicit_hf_transfer(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setenv("AUTOTRANSITION_ALLOW_HF_TRANSFER", "1")

    env = build_runtime_env()

    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


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

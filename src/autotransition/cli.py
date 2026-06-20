"""Command-line interface for Autotransition."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from autotransition.audio import build_repaint_scaffold
from autotransition.config import OutputConfig, RuntimeConfig, TransitionConfig
from autotransition.models import (
    ACE_STEP_MODELS,
    ModelInstallError,
    install_model,
    repaint_capable_models,
    resolve_model_status,
)
from autotransition.pipeline import TransitionRequest, create_scaffold_plan
from autotransition.presets import PRESETS, get_preset
from autotransition.runtime.ace_step import (
    build_debug_start_api_command,
    build_install_commands,
    build_start_api_command,
    ensure_runtime_api,
    run_install,
    runtime_doctor,
    runtime_status,
    start_api_background,
    start_api_foreground,
    stop_runtime_process_tree,
)
from autotransition.ui import create_app

app = typer.Typer(help="Build and manage AI music transition pipeline artifacts.")
models_app = typer.Typer(help="List, inspect, and install ACE-Step models.")
runtime_app = typer.Typer(help="Set up and inspect the ACE-Step runtime.")
app.add_typer(models_app, name="models")
app.add_typer(runtime_app, name="runtime")


@app.command()
def presets() -> None:
    """List available transition presets."""

    for preset in PRESETS.values():
        typer.echo(f"{preset.slug}: {preset.name} - {preset.description}")


@app.command()
def setup(
    runtime_dir: Path = typer.Option(Path("runtimes/ACE-Step-1.5"), help="ACE-Step runtime install directory."),
    skip_runtime: bool = typer.Option(False, "--skip-runtime", help="Skip ACE-Step runtime setup."),
    print_only: bool = typer.Option(False, "--print-only", help="Print setup commands without running."),
) -> None:
    """Set up Autotransition for first use."""

    config = RuntimeConfig(ace_step_dir=runtime_dir)
    if print_only:
        typer.echo("First-time setup commands:")
        for command in build_install_commands(config):
            typer.echo(command)
        return
    if skip_runtime:
        typer.echo("Skipped ACE-Step runtime setup.")
        return
    run_install(config)
    typer.echo("Setup complete.")
    typer.echo("Run the app with: autotransition run")


@app.command()
def run(
    host: str = typer.Option("127.0.0.1", help="Host for the local UI server."),
    port: int = typer.Option(7860, help="Port for the local UI server."),
    runtime_host: str = typer.Option("127.0.0.1", help="ACE-Step API host."),
    runtime_port: int = typer.Option(8001, help="ACE-Step API port."),
    models_dir: Path = typer.Option(Path("models"), help="Directory where models are installed."),
    ui_only: bool = typer.Option(False, "--ui-only", help="Start only the Autotransition UI."),
    no_runtime_autostart: bool = typer.Option(False, "--no-runtime-autostart", help="Do not auto-start ACE-Step API."),
) -> None:
    """Run the full local Autotransition app."""

    runtime_config = RuntimeConfig(api_host=runtime_host, api_port=runtime_port)
    managed_runtime_pid: int | None = None
    if not ui_only and not no_runtime_autostart:
        start_result = ensure_runtime_api(runtime_config, status_callback=typer.echo)
        typer.echo(start_result.message)
        if not start_result.started and not start_result.already_running:
            raise typer.Exit(1)
        if start_result.managed_by_current_run:
            managed_runtime_pid = start_result.pid
        typer.echo(f"ACE-Step API: {start_result.api_url}")

    typer.echo(f"Autotransition UI: http://{host}:{port}")
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("uvicorn is required to run the UI. Install project dependencies.") from exc
    try:
        uvicorn.run(create_app(models_dir=models_dir, runtime_config=runtime_config), host=host, port=port)
    finally:
        if managed_runtime_pid is not None:
            typer.echo("Stopping managed ACE-Step runtime...")
            if not stop_runtime_process_tree(managed_runtime_pid, runtime_config):
                typer.echo(
                    f"ACE-Step runtime process {managed_runtime_pid} did not stop cleanly. "
                    "Check it manually before running again."
                )


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Host for the local UI server."),
    port: int = typer.Option(7860, help="Port for the local UI server."),
    models_dir: Path = typer.Option(Path("models"), help="Directory where models are installed."),
) -> None:
    """Run the local web UI."""

    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("uvicorn is required to run the UI. Install project dependencies.") from exc

    typer.echo(f"Autotransition UI running at http://{host}:{port}")
    uvicorn.run(create_app(models_dir=models_dir), host=host, port=port)


@runtime_app.command()
def status(
    runtime_dir: Path = typer.Option(Path("runtimes/ACE-Step-1.5"), help="ACE-Step runtime install directory."),
    host: str = typer.Option("127.0.0.1", help="ACE-Step API host."),
    port: int = typer.Option(8001, help="ACE-Step API port."),
) -> None:
    """Show ACE-Step runtime install/API status."""

    config = RuntimeConfig(ace_step_dir=runtime_dir, api_host=host, api_port=port)
    typer.echo(json.dumps(runtime_status(config).to_dict(), indent=2))


@runtime_app.command("print-setup")
def print_setup(
    runtime_dir: Path = typer.Option(Path("runtimes/ACE-Step-1.5"), help="ACE-Step runtime install directory."),
) -> None:
    """Print first-time ACE-Step setup commands."""

    config = RuntimeConfig(ace_step_dir=runtime_dir)
    for command in build_install_commands(config):
        typer.echo(command)


@runtime_app.command("start-command")
def start_command(
    runtime_dir: Path = typer.Option(Path("runtimes/ACE-Step-1.5"), help="ACE-Step runtime install directory."),
    host: str = typer.Option("127.0.0.1", help="ACE-Step API host."),
    port: int = typer.Option(8001, help="ACE-Step API port."),
) -> None:
    """Print the ACE-Step API start command."""

    config = RuntimeConfig(ace_step_dir=runtime_dir, api_host=host, api_port=port)
    typer.echo(build_debug_start_api_command(config))


@runtime_app.command("setup")
def setup_runtime(
    runtime_dir: Path = typer.Option(Path("runtimes/ACE-Step-1.5"), help="ACE-Step runtime install directory."),
    print_only: bool = typer.Option(False, "--print-only", help="Print setup commands without running them."),
    execute: bool = typer.Option(False, "--execute", help="Deprecated; setup now runs by default."),
) -> None:
    """Run first-time ACE-Step runtime setup."""

    config = RuntimeConfig(ace_step_dir=runtime_dir)
    if print_only:
        typer.echo("First-time setup commands:")
        for command in build_install_commands(config):
            typer.echo(command)
        return

    run_install(config)
    typer.echo("ACE-Step runtime setup complete.")
    typer.echo("Next: autotransition runtime start")


@runtime_app.command("start")
def start_runtime(
    runtime_dir: Path = typer.Option(Path("runtimes/ACE-Step-1.5"), help="ACE-Step runtime install directory."),
    host: str = typer.Option("127.0.0.1", help="ACE-Step API host."),
    port: int = typer.Option(8001, help="ACE-Step API port."),
    background: bool = typer.Option(False, "--background", help="Start in the background and write logs to data/logs."),
) -> None:
    """Start the ACE-Step API."""

    config = RuntimeConfig(ace_step_dir=runtime_dir, api_host=host, api_port=port)
    if background:
        process = start_api_background(config)
        typer.echo(f"ACE-Step API starting in background as process {process.pid}.")
        typer.echo(f"API URL: {config.api_base_url}")
        typer.echo("Logs: data/logs/ace-step-api.log and data/logs/ace-step-api.err.log")
        return

    typer.echo(f"Starting ACE-Step API at {config.api_base_url}. Press Ctrl+C to stop.")
    raise typer.Exit(start_api_foreground(config))


@runtime_app.command("doctor")
def doctor(
    runtime_dir: Path = typer.Option(Path("runtimes/ACE-Step-1.5"), help="ACE-Step runtime install directory."),
    host: str = typer.Option("127.0.0.1", help="ACE-Step API host."),
    port: int = typer.Option(8001, help="ACE-Step API port."),
) -> None:
    """Check ACE-Step runtime setup."""

    config = RuntimeConfig(ace_step_dir=runtime_dir, api_host=host, api_port=port)
    for check in runtime_doctor(config):
        typer.echo(f"{check.status.value}: {check.name} - {check.message}")


@models_app.command("list")
def list_models(
    all_models: bool = typer.Option(False, "--all", help="Show all registered models, not only repaint-capable ones."),
    models_dir: Path = typer.Option(Path("models"), help="Directory where models are installed."),
) -> None:
    """List ACE-Step model profiles."""

    profiles = list(ACE_STEP_MODELS.values()) if all_models else repaint_capable_models()
    for profile in profiles:
        status = resolve_model_status(profile.slug, models_dir=models_dir)
        typer.echo(
            f"{profile.slug}: {profile.display_name} "
            f"[{status.state.value}] - {profile.quality_label}, {profile.speed_label}; {profile.vram_guidance}"
        )


@models_app.command()
def status(
    model: str = typer.Argument(..., help="Model slug."),
    models_dir: Path = typer.Option(Path("models"), help="Directory where models are installed."),
) -> None:
    """Show install status for one model."""

    try:
        install_status = resolve_model_status(model, models_dir=models_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(json.dumps(install_status.to_dict(), indent=2))


@models_app.command()
def install(
    model: str = typer.Argument(..., help="Model slug."),
    models_dir: Path = typer.Option(Path("models"), help="Directory where models are installed."),
) -> None:
    """Download an ACE-Step model profile from Hugging Face."""

    try:
        install_status = install_model(model, models_dir=models_dir)
    except (ValueError, ModelInstallError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(json.dumps(install_status.to_dict(), indent=2))


@app.command()
def scaffold(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source audio clip."),
    preset: str = typer.Option("smooth-continuation", help="Transition preset slug."),
    caption: str | None = typer.Option(None, help="Override the preset caption."),
    output_dir: Path | None = typer.Option(None, help="Directory for scaffold outputs."),
    context_seconds: float | None = typer.Option(None, help="Preserved context before repainting."),
    repaint_overlap_seconds: float | None = typer.Option(
        None,
        help="Deprecated legacy repaint setting; normal continuation generation does not use it.",
    ),
    new_section_seconds: float | None = typer.Option(None, help="Blank future duration to generate."),
    bpm: float | None = typer.Option(None, help="Optional BPM hint."),
    key: str | None = typer.Option(None, help="Optional key hint."),
    seed: int | None = typer.Option(None, help="Optional generation seed."),
) -> None:
    """Build a tail-plus-silence repaint scaffold and write metadata."""

    selected = get_preset(preset)
    base = selected.config
    output = base.output
    if output_dir is not None:
        output = OutputConfig(
            root_dir=output_dir,
            scaffold_dir=output_dir,
            generated_dir=output_dir / "generated",
            export_dir=output_dir / "exports",
            audio_format=output.audio_format,
        )

    config = TransitionConfig(
        context_seconds=context_seconds if context_seconds is not None else base.context_seconds,
        repaint_overlap_seconds=(
            repaint_overlap_seconds if repaint_overlap_seconds is not None else base.repaint_overlap_seconds
        ),
        new_section_seconds=new_section_seconds if new_section_seconds is not None else base.new_section_seconds,
        output=output,
        candidate_count=base.candidate_count,
        seed=seed if seed is not None else base.seed,
        bpm_hint=bpm if bpm is not None else base.bpm_hint,
        key_hint=key if key is not None else base.key_hint,
    )

    request = TransitionRequest(
        source_path=source,
        caption=caption or selected.caption,
        config=config,
    )
    plan = create_scaffold_plan(request)
    build_repaint_scaffold(
        source_path=plan.source_path,
        output_path=plan.scaffold_path,
        tail_seconds=config.tail_seconds,
        blank_seconds=config.new_section_seconds,
        output_format=plan.audio_format,
    )

    plan.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    plan.metadata_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    typer.echo(f"Scaffold written: {plan.scaffold_path}")
    typer.echo(f"Metadata written: {plan.metadata_path}")


if __name__ == "__main__":
    app()

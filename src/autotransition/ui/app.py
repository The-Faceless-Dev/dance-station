"""FastAPI app for the local Autotransition UI."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from autotransition.audio import build_continuation_composite, build_repaint_scaffold, build_selection_scaffold, probe_audio
from autotransition.audio.formats import DEFAULT_SCAFFOLD_FORMAT, SUPPORTED_INPUT_FORMATS, validate_supported_source
from autotransition.config import OutputConfig, RuntimeConfig, TransitionConfig
from autotransition.generation import GenerationResult, GenerationStatus
from autotransition.models import (
    AceStepRepaintAdapter,
    AceStepRuntimeError,
    ModelInstallError,
    get_model_profile,
    install_model,
    repaint_capable_models,
    resolve_model_status,
)
from autotransition.models.acestep_api import _repaint_defaults_for_profile, _text2music_defaults_for_profile
from autotransition.models.download import local_model_path
from autotransition.models.status import InstallState
from autotransition.pipeline import (
    SourceSelectionPlan,
    SourceSelectionRequest,
    TransitionRequest,
    create_scaffold_plan,
    create_source_selection_plan,
)
from autotransition.presets import PRESETS, get_preset
from autotransition.ui.activity import summarize_runtime_activity
from autotransition.ui.state import UiLog, system_status


class ScaffoldRequest(BaseModel):
    source_path: str = Field(..., min_length=1)
    preset: str = "smooth-continuation"
    caption: str | None = None
    output_dir: str | None = None
    context_seconds: float | None = Field(None, gt=0)
    repaint_overlap_seconds: float | None = Field(None, ge=0)
    new_section_seconds: float | None = Field(None, gt=0)
    bpm: float | None = Field(None, gt=0)
    key: str | None = None
    seed: int | None = None


class ProbeRequest(BaseModel):
    source_path: str = Field(..., min_length=1)


class SelectionScaffoldRequest(ScaffoldRequest):
    continuation_point_seconds: float = Field(..., gt=0)
    generation_region: Literal["extend", "repaint_existing"] = "extend"


class AceStepAdvancedSettings(BaseModel):
    inference_steps: int | None = Field(None, ge=1, le=200)
    guidance_scale: float | None = Field(None, ge=0)
    shift: float | None = Field(None, ge=0)
    chunk_mask_mode: Literal["explicit", "auto"] | None = None
    repaint_mode: Literal["balanced", "conservative", "aggressive"] | None = None
    repaint_strength: float | None = Field(None, ge=0, le=1)
    repaint_latent_crossfade_frames: int | None = Field(None, ge=0, le=200)
    repaint_wav_crossfade_sec: float | None = Field(None, ge=0, le=10)

    def to_payload(self) -> dict[str, object]:
        return {key: value for key, value in self.model_dump().items() if value is not None}


class GenerateSelectionRequest(SelectionScaffoldRequest):
    model_slug: str = "acestep-v15-xl-base"
    auto_install: bool = False
    ace_step: AceStepAdvancedSettings | None = None


def _setting_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def create_app(models_dir: Path = Path("models"), runtime_config: RuntimeConfig | None = None) -> FastAPI:
    runtime_config = runtime_config or RuntimeConfig()
    app = FastAPI(title="Autotransition", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    ui_log = UiLog()
    ui_log.add("info", "UI server started.")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    def get_status() -> dict[str, object]:
        status = system_status(models_dir=models_dir)
        status["supported_input_formats"] = list(SUPPORTED_INPUT_FORMATS)
        status["default_scaffold_format"] = DEFAULT_SCAFFOLD_FORMAT
        return status

    @app.get("/api/runtime/status")
    def get_runtime_status() -> dict[str, object]:
        from autotransition.runtime.ace_step import build_install_commands, build_start_api_command, runtime_status

        status = runtime_status(runtime_config).to_dict()
        status["install_commands"] = build_install_commands(runtime_config)
        status["start_api_command"] = build_start_api_command(runtime_config)
        status["simple_setup_command"] = "autotransition runtime setup"
        status["simple_start_command"] = "autotransition runtime start"
        return status

    @app.get("/api/runtime/activity")
    def get_runtime_activity() -> dict[str, object]:
        from autotransition.runtime.ace_step import runtime_status

        activity = summarize_runtime_activity().to_dict()
        status = runtime_status(runtime_config)
        activity["api_running"] = status.api_running
        activity["api_url"] = status.api_url
        activity["runtime_message"] = status.message
        return activity

    @app.get("/api/source/audio")
    def get_source_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        source_path = Path(path).expanduser()
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(status_code=404, detail=f"Source audio not found: {source_path}")
        return FileResponse(source_path)

    @app.get("/api/audio")
    def get_audio_file(path: str = Query(..., min_length=1)) -> FileResponse:
        audio_path = Path(path).expanduser()
        if not audio_path.exists() or not audio_path.is_file():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
        validate_supported_source(audio_path)
        return FileResponse(audio_path)

    @app.post("/api/source/probe")
    def probe_source(request: ProbeRequest) -> dict[str, object]:
        source_path = Path(request.source_path).expanduser()
        try:
            result = probe_audio(source_path)
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ui_log.add(
            "info",
            f"Loaded {result.source_format} source audio: {source_path}. "
            f"Scaffolds will be normalized to {DEFAULT_SCAFFOLD_FORMAT.upper()}.",
        )
        return result.to_dict()

    @app.post("/api/source/upload")
    def upload_source(file: UploadFile = File(...)) -> dict[str, object]:
        original_name = Path(file.filename or "source").name
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name).strip("._") or "source"
        suffix = Path(safe_name).suffix.lower()
        temp_path = Path("data/input") / f"{Path(safe_name).stem}-{uuid4().hex[:8]}{suffix}"

        try:
            validate_supported_source(temp_path)
        except ValueError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        temp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            result = probe_audio(temp_path)
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            file.file.close()

        ui_log.add(
            "info",
            f"Uploaded {result.source_format} source '{original_name}' to {temp_path}. "
            f"Scaffolds will be normalized to {DEFAULT_SCAFFOLD_FORMAT.upper()}.",
        )
        return {
            "original_filename": original_name,
            "stored_path": str(temp_path),
            "probe": result.to_dict(),
        }

    @app.get("/api/presets")
    def get_presets() -> list[dict[str, Any]]:
        return [
            {
                "slug": preset.slug,
                "name": preset.name,
                "description": preset.description,
                "caption": preset.caption,
                "config": {
                    "context_seconds": preset.config.context_seconds,
                    "new_section_seconds": preset.config.new_section_seconds,
                    "candidate_count": preset.config.candidate_count,
                },
            }
            for preset in PRESETS.values()
        ]

    @app.get("/api/models")
    def get_models() -> list[dict[str, Any]]:
        models = []
        for profile in repaint_capable_models():
            status = resolve_model_status(profile.slug, models_dir=models_dir)
            models.append(
                {
                    "slug": profile.slug,
                    "display_name": profile.display_name,
                    "repo_id": profile.repo_id,
                    "family": profile.family,
                    "supports_repaint": profile.supports_repaint,
                    "quality_label": profile.quality_label,
                    "speed_label": profile.speed_label,
                    "vram_guidance": profile.vram_guidance,
                    "default_inference_steps": profile.default_inference_steps,
                    "generation_defaults": {
                        "inference_steps": profile.default_inference_steps,
                        **_text2music_defaults_for_profile(profile),
                    },
                    "repaint_defaults": {
                        "inference_steps": profile.default_inference_steps,
                        **_repaint_defaults_for_profile(profile),
                    },
                    "notes": profile.notes,
                    "status": status.to_dict(),
                }
            )
        return models

    @app.post("/api/models/{slug}/install")
    def install_selected_model(slug: str) -> dict[str, str]:
        ui_log.add("info", f"Installing model '{slug}' from Hugging Face.")
        try:
            status = install_model(slug, models_dir=models_dir)
        except (ValueError, ModelInstallError) as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ui_log.add("info", f"Model '{slug}' installed at {status.local_path}.")
        return status.to_dict()

    @app.post("/api/scaffolds")
    def create_scaffold(request: ScaffoldRequest) -> dict[str, Any]:
        try:
            selected = get_preset(request.preset)
        except ValueError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        source_path = Path(request.source_path).expanduser()
        if not source_path.exists():
            message = f"Source audio not found: {source_path}"
            ui_log.add("error", message)
            raise HTTPException(status_code=400, detail=message)

        base = selected.config
        output = base.output
        if request.output_dir:
            output_dir = Path(request.output_dir).expanduser()
            output = OutputConfig(
                root_dir=output_dir,
                scaffold_dir=output_dir,
                generated_dir=output_dir / "generated",
                export_dir=output_dir / "exports",
                audio_format=output.audio_format,
            )

        config = TransitionConfig(
            context_seconds=_setting_or_default(request.context_seconds, base.context_seconds),
            repaint_overlap_seconds=_setting_or_default(request.repaint_overlap_seconds, base.repaint_overlap_seconds),
            new_section_seconds=_setting_or_default(request.new_section_seconds, base.new_section_seconds),
            output=output,
            candidate_count=base.candidate_count,
            seed=request.seed if request.seed is not None else base.seed,
            bpm_hint=request.bpm if request.bpm is not None else base.bpm_hint,
            key_hint=request.key if request.key else base.key_hint,
        )
        plan = create_scaffold_plan(
            TransitionRequest(
                source_path=source_path,
                caption=request.caption or selected.caption,
                config=config,
            )
        )

        try:
            ui_log.add(
                "info",
                f"Decoding {source_path.suffix.lower() or 'source'} and normalizing scaffold to "
                f"{plan.audio_format.upper()}.",
            )
            build_repaint_scaffold(
                source_path=plan.source_path,
                output_path=plan.scaffold_path,
                tail_seconds=config.tail_seconds,
                blank_seconds=config.new_section_seconds,
                output_format=plan.audio_format,
            )
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        plan.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        plan.metadata_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        ui_log.add("info", f"Scaffold created: {plan.scaffold_path}")
        return {"plan": plan.to_dict()}

    @app.post("/api/scaffolds/from-selection")
    def create_scaffold_from_selection(request: SelectionScaffoldRequest) -> dict[str, Any]:
        try:
            selected = get_preset(request.preset)
        except ValueError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        source_path = Path(request.source_path).expanduser()
        try:
            probe = probe_audio(source_path)
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        base = selected.config
        output = base.output
        if request.output_dir:
            output_dir = Path(request.output_dir).expanduser()
            output = OutputConfig(
                root_dir=output_dir,
                scaffold_dir=output_dir,
                generated_dir=output_dir / "generated",
                export_dir=output_dir / "exports",
                audio_format=output.audio_format,
            )

        config = TransitionConfig(
            context_seconds=_setting_or_default(request.context_seconds, base.context_seconds),
            repaint_overlap_seconds=_setting_or_default(request.repaint_overlap_seconds, base.repaint_overlap_seconds),
            new_section_seconds=_setting_or_default(request.new_section_seconds, base.new_section_seconds),
            output=output,
            candidate_count=base.candidate_count,
            seed=request.seed if request.seed is not None else base.seed,
            bpm_hint=request.bpm if request.bpm is not None else base.bpm_hint,
            key_hint=request.key if request.key else base.key_hint,
        )

        try:
            plan = create_source_selection_plan(
                SourceSelectionRequest(
                    source_path=source_path,
                    source_duration_seconds=probe.duration_seconds,
                    continuation_point_seconds=request.continuation_point_seconds,
                    caption=request.caption or selected.caption,
                    config=config,
                    generation_region=request.generation_region,
                )
            )
            ui_log.add(
                "info",
                f"Decoding {plan.source_format} source selection and normalizing scaffold to "
                f"{plan.audio_format.upper()}.",
            )
            build_selection_scaffold(
                source_path=plan.source_path,
                output_path=plan.scaffold_path,
                tail_start_seconds=plan.tail_start_seconds,
                tail_end_seconds=plan.tail_end_seconds,
                blank_seconds=config.new_section_seconds,
                output_format=plan.audio_format,
                target_end_seconds=(
                    request.continuation_point_seconds + config.new_section_seconds
                    if request.generation_region == "repaint_existing"
                    else None
                ),
                append_silence=request.generation_region != "extend",
            )
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        plan.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        plan.metadata_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        ui_log.add(
            "info",
            f"Selection scaffold created from {plan.tail_start_seconds:.2f}s to {plan.tail_end_seconds:.2f}s: "
            f"{plan.scaffold_path}",
        )
        return {"plan": plan.to_dict()}

    @app.post("/api/generate/from-selection")
    def generate_from_selection(request: GenerateSelectionRequest) -> dict[str, object]:
        generation_id = f"generation-{uuid4().hex[:12]}"
        try:
            profile = get_model_profile(request.model_slug)
            selected = get_preset(request.preset)
        except ValueError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        source_path = Path(request.source_path).expanduser()
        try:
            probe = probe_audio(source_path)
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        base = selected.config
        output = base.output
        if request.output_dir:
            output_dir = Path(request.output_dir).expanduser()
            output = OutputConfig(
                root_dir=output_dir,
                scaffold_dir=output_dir / "scaffolds",
                generated_dir=output_dir / "generated",
                export_dir=output_dir / "exports",
                audio_format=output.audio_format,
            )

        config = TransitionConfig(
            context_seconds=_setting_or_default(request.context_seconds, base.context_seconds),
            repaint_overlap_seconds=_setting_or_default(request.repaint_overlap_seconds, base.repaint_overlap_seconds),
            new_section_seconds=_setting_or_default(request.new_section_seconds, base.new_section_seconds),
            output=output,
            candidate_count=base.candidate_count,
            seed=request.seed if request.seed is not None else base.seed,
            bpm_hint=request.bpm if request.bpm is not None else base.bpm_hint,
            key_hint=request.key if request.key else base.key_hint,
        )

        try:
            plan = create_source_selection_plan(
                SourceSelectionRequest(
                    source_path=source_path,
                    source_duration_seconds=probe.duration_seconds,
                    continuation_point_seconds=request.continuation_point_seconds,
                    caption=request.caption or selected.caption,
                    config=config,
                    transition_id=generation_id,
                    generation_region=request.generation_region,
                    ace_step_settings=request.ace_step.to_payload() if request.ace_step else None,
                )
            )
            if request.generation_region == "repaint_existing":
                ui_log.add("info", "Preparing internal repaint scaffold for generation.")
                build_selection_scaffold(
                    source_path=plan.source_path,
                    output_path=plan.scaffold_path,
                    tail_start_seconds=plan.tail_start_seconds,
                    tail_end_seconds=plan.tail_end_seconds,
                    blank_seconds=config.new_section_seconds,
                    output_format=plan.audio_format,
                    target_end_seconds=request.continuation_point_seconds + config.new_section_seconds,
                    append_silence=True,
                )
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        plan.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        plan.metadata_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

        try:
            ui_log.add("info", "Running ACE-Step text-to-music continuation with the active runtime model.")
            adapter = AceStepRepaintAdapter(
                profile=profile,
                model_path=local_model_path(profile, models_dir),
                runtime_config=runtime_config,
            )
            raw_generation = adapter.text2music(plan)
            composite_dir = output.generated_dir / generation_id
            composite_path = composite_dir / f"{generation_id}-composite.{plan.audio_format}"
            composite_metadata_path = composite_dir / "composite.json"
            ui_log.add("info", "Stitching generated section after the selected source point.")
            build_continuation_composite(
                source_path=plan.source_path,
                generated_path=raw_generation.output_path,
                output_path=composite_path,
                continuation_point_seconds=plan.continuation_point_seconds,
                output_format=plan.audio_format,
            )
            composite_metadata = {
                "generation_id": generation_id,
                "raw_generated_audio_path": str(raw_generation.output_path),
                "raw_generated_metadata_path": str(raw_generation.metadata_path),
                "composite_audio_path": str(composite_path),
                "continuation_point_seconds": plan.continuation_point_seconds,
                "new_section_seconds": plan.new_section_seconds,
                "boundary_repaint": False,
            }
            composite_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            composite_metadata_path.write_text(json.dumps(composite_metadata, indent=2), encoding="utf-8")

            final_audio_path = composite_path
            final_metadata_path = composite_metadata_path
            if plan.repaint_margin_seconds > 0:
                boundary_start = max(0.0, plan.continuation_point_seconds - plan.repaint_margin_seconds)
                boundary_end = min(
                    plan.continuation_point_seconds + plan.repaint_margin_seconds,
                    plan.continuation_point_seconds + plan.new_section_seconds,
                )
                ui_log.add(
                    "info",
                    f"Running ACE-Step boundary repaint from {boundary_start:.2f}s to {boundary_end:.2f}s.",
                )
                boundary_plan = SourceSelectionPlan(
                    **{
                        **plan.to_dict(),
                        "source_path": plan.source_path,
                        "scaffold_path": composite_path,
                        "metadata_path": composite_metadata_path,
                        "tail_start_seconds": 0.0,
                        "tail_end_seconds": plan.continuation_point_seconds + plan.new_section_seconds,
                        "repainting_start_seconds": boundary_start,
                        "repainting_end_seconds": boundary_end,
                        "generation_region": "repaint_existing",
                    }
                )
                boundary_result = adapter.repaint(boundary_plan)
                final_audio_path = boundary_result.output_path
                final_metadata_path = boundary_result.metadata_path
                composite_metadata["boundary_repaint"] = True
                composite_metadata["boundary_repaint_start_seconds"] = boundary_start
                composite_metadata["boundary_repaint_end_seconds"] = boundary_end
                composite_metadata["boundary_repaint_audio_path"] = str(boundary_result.output_path)
                composite_metadata["boundary_repaint_metadata_path"] = str(boundary_result.metadata_path)
                composite_metadata_path.write_text(json.dumps(composite_metadata, indent=2), encoding="utf-8")
        except AceStepRuntimeError as exc:
            ui_log.add("error", str(exc))
            result = GenerationResult(
                generation_id=generation_id,
                status=GenerationStatus.FAILED,
                message=str(exc),
                model_slug=profile.slug,
                scaffold_path=plan.scaffold_path,
                scaffold_metadata_path=plan.metadata_path,
            )
            return {"result": result.to_dict(), "plan": plan.to_dict()}

        result = GenerationResult(
            generation_id=generation_id,
            status=GenerationStatus.COMPLETE,
            message="Generation complete.",
            model_slug=profile.slug,
            scaffold_path=plan.scaffold_path,
            scaffold_metadata_path=plan.metadata_path,
            generated_audio_path=final_audio_path,
            generated_metadata_path=final_metadata_path,
        )
        ui_log.add("info", f"Generated transition: {final_audio_path}")
        return {"result": result.to_dict(), "plan": plan.to_dict()}

    @app.get("/api/logs")
    def get_logs() -> list[dict[str, str]]:
        return ui_log.entries()

    return app

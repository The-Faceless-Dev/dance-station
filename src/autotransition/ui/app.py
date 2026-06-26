"""FastAPI app for the local Autotransition UI."""

from __future__ import annotations

import json
import re
import shutil
from urllib.parse import quote
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from autotransition.audio import (
    build_continuation_composite,
    build_repaint_scaffold,
    build_selection_scaffold,
    merge_audio_files,
    probe_audio,
)
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
from autotransition.models.acestep_api import (
    AceStepApiClient,
    AceStepApiError,
    BASE_EXTRACT_GUIDANCE_SCALE,
    BASE_RUNTIME_DCW_ENABLED,
    BASE_RUNTIME_GUIDANCE_SCALE,
    BASE_RUNTIME_INFERENCE_STEPS,
    BASE_RUNTIME_INFER_METHOD,
    BASE_RUNTIME_SHIFT,
    BASE_RUNTIME_USE_TILED_DECODE,
    BASE_RUNTIME_VELOCITY_EMA_FACTOR,
    BASE_RUNTIME_VELOCITY_NORM_THRESHOLD,
    _repaint_defaults_for_profile,
    _text2music_defaults_for_profile,
)
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
    model_slug: str = "acestep-v15-turbo"
    auto_install: bool = False
    ace_step: AceStepAdvancedSettings | None = None


EXTRACT_TRACKS = [
    "vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "fx",
    "brass",
    "woodwinds",
]


class ExtractionRunRequest(BaseModel):
    source_path: str = Field(..., min_length=1)
    track_name: str = "vocals"
    label: str | None = None
    output_format: Literal["flac", "wav", "wav32", "mp3", "opus", "aac"] = "flac"
    inference_steps: int = Field(BASE_RUNTIME_INFERENCE_STEPS, ge=1, le=200)
    guidance_scale: float = Field(BASE_EXTRACT_GUIDANCE_SCALE, ge=0)
    shift: float = Field(BASE_RUNTIME_SHIFT, ge=0)
    infer_method: Literal["ode", "sde"] = BASE_RUNTIME_INFER_METHOD
    use_tiled_decode: bool = BASE_RUNTIME_USE_TILED_DECODE
    dcw_enabled: bool = BASE_RUNTIME_DCW_ENABLED
    velocity_norm_threshold: float = Field(BASE_RUNTIME_VELOCITY_NORM_THRESHOLD, ge=0)
    velocity_ema_factor: float = Field(BASE_RUNTIME_VELOCITY_EMA_FACTOR, ge=0, le=1)
    seed: int | None = None
    instruction: str | None = None


class BaseGenerationTestRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    output_format: Literal["flac", "wav", "wav32", "mp3", "opus", "aac"] = "flac"
    audio_duration: float = Field(30.0, ge=10.0, le=120.0)
    inference_steps: int = Field(BASE_RUNTIME_INFERENCE_STEPS, ge=1, le=200)
    guidance_scale: float = Field(BASE_RUNTIME_GUIDANCE_SCALE, ge=0)
    shift: float = Field(BASE_RUNTIME_SHIFT, ge=0)
    infer_method: Literal["ode", "sde"] = BASE_RUNTIME_INFER_METHOD
    use_tiled_decode: bool = BASE_RUNTIME_USE_TILED_DECODE
    dcw_enabled: bool = BASE_RUNTIME_DCW_ENABLED
    velocity_norm_threshold: float = Field(BASE_RUNTIME_VELOCITY_NORM_THRESHOLD, ge=0)
    velocity_ema_factor: float = Field(BASE_RUNTIME_VELOCITY_EMA_FACTOR, ge=0, le=1)
    seed: int | None = None


class ExtractionRenameRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)


class ExtractionMergeRequest(BaseModel):
    extraction_ids: list[str] = Field(..., min_length=2)
    label: str = Field(..., min_length=1, max_length=120)
    output_format: Literal["flac", "wav", "wav32", "mp3", "opus", "aac"] = "flac"


class MusicGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: Literal["acestep-v15-turbo", "acestep-v15-base"] = "acestep-v15-turbo"
    label: str | None = None
    output_format: Literal["flac", "wav", "wav32", "mp3", "opus", "aac"] = "flac"
    audio_duration: float = Field(30.0, ge=10.0, le=120.0)
    inference_steps: int = Field(8, ge=1, le=200)
    guidance_scale: float = Field(1.0, ge=0)
    shift: float = Field(3.0, ge=0)
    infer_method: Literal["ode", "sde"] = "ode"
    use_tiled_decode: bool = True
    dcw_enabled: bool = False
    velocity_norm_threshold: float = Field(0.0, ge=0)
    velocity_ema_factor: float = Field(0.0, ge=0, le=1)
    seed: int | None = None


def _setting_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _extraction_metadata_root() -> Path:
    return Path("data/extractions")


def _extraction_metadata_path(extraction_id: str) -> Path:
    safe_id = Path(extraction_id).name
    if not safe_id or safe_id != extraction_id:
        raise HTTPException(status_code=400, detail="Invalid extraction id.")
    return _extraction_metadata_root() / safe_id / "extraction.json"


def _read_extraction_metadata(extraction_id: str) -> dict[str, Any]:
    metadata_path = _extraction_metadata_path(extraction_id)
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"Extraction not found: {extraction_id}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read extraction metadata: {extraction_id}") from exc
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _write_extraction_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(str(metadata["metadata_path"]))
    return _write_metadata(metadata_path, metadata)


def _write_metadata(metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _music_generation_root() -> Path:
    return Path("data/generations")


def _transition_root() -> Path:
    return Path("data/generated")


def _edit_root() -> Path:
    return Path("data/edits")


def _instrument_lab_root() -> Path:
    return Path("data/instrument-lab")


def _instrument_bank_root() -> Path:
    return _instrument_lab_root() / "instruments"


def _safe_item_id(item_id: str, label: str) -> str:
    safe_id = Path(item_id).name
    if not safe_id or safe_id != item_id:
        raise HTTPException(status_code=400, detail=f"Invalid {label} id.")
    return safe_id


def _read_json_file(metadata_path: Path, label: str) -> dict[str, Any]:
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {label} metadata.") from exc
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _transition_metadata_path(generation_id: str) -> Path:
    return _transition_root() / _safe_item_id(generation_id, "transition") / "result.json"


def _music_metadata_path(generation_id: str) -> Path:
    return _music_generation_root() / _safe_item_id(generation_id, "music generation") / "generation.json"


def _edit_metadata_path(edit_id: str) -> Path:
    return _edit_root() / _safe_item_id(edit_id, "edit") / "edit.json"


def _instrument_lab_metadata_path(clip_id: str) -> Path:
    return _instrument_lab_root() / _safe_item_id(clip_id, "instrument clip") / "clip.json"


def _safe_label_stem(label: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    return clean or fallback


def _midi_note(value: str | int | float | None, default: int = 60) -> int:
    if value is None:
        return default
    if isinstance(value, int | float):
        return max(0, min(127, int(value)))
    text = str(value).strip().lower()
    if text.lstrip("-").isdigit():
        return max(0, min(127, int(text)))
    match = re.fullmatch(r"([a-g])([#b]?)(-?\d+)", text)
    if not match:
        return default
    note_name, accidental, octave_text = match.groups()
    semitone = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[note_name]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    return max(0, min(127, (int(octave_text) + 1) * 12 + semitone))


def _parse_sfz_regions(sfz_text: str) -> list[dict[str, str]]:
    regions: list[dict[str, str]] = []
    current_group: dict[str, str] = {}
    current_region: dict[str, str] | None = None
    token_pattern = re.compile(r"(<group>|<region>)|([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]+\"|'[^']+'|[^<\s]+)")
    for raw_line in sfz_text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        for match in token_pattern.finditer(line):
            marker, key, value = match.groups()
            if marker == "<group>":
                current_group = {}
                current_region = None
            elif marker == "<region>":
                current_region = dict(current_group)
                regions.append(current_region)
            elif key and value:
                target = current_region if current_region is not None else current_group
                target[key.lower()] = value.strip().strip("\"'")
    return [region for region in regions if region.get("sample")]


def _sfz_instrument_from_regions(
    *,
    instrument_id: str,
    label: str,
    regions: list[dict[str, str]],
    stored_samples: dict[str, Path],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    missing_samples: set[str] = set()
    for region in regions:
        sample_name = Path(region["sample"].replace("\\", "/")).name
        sample_path = stored_samples.get(sample_name.lower())
        if sample_path is None:
            missing_samples.add(sample_name)
            continue
        key = _midi_note(region.get("key")) if region.get("key") else None
        root = _midi_note(region.get("pitch_keycenter"), key if key is not None else 60)
        low = _midi_note(region.get("lokey"), key if key is not None else root)
        high = _midi_note(region.get("hikey"), key if key is not None else root)
        samples.append(
            {
                "note": key if key is not None else root,
                "root": root,
                "low": min(low, high),
                "high": max(low, high),
                "path": str(sample_path),
                "url": f"/api/instrument-lab/instruments/sample?path={quote(str(sample_path))}",
                "volume": float(region.get("volume", 0) or 0),
            }
        )
    return {
        "id": instrument_id,
        "name": label,
        "category": "SoundFonts / User Instruments",
        "type": "sample",
        "source": "sfz",
        "samples": samples,
        "missing_samples": sorted(missing_samples),
        "envelope": {"attack": 0.005, "release": 0.2},
    }


def _list_user_instruments() -> list[dict[str, Any]]:
    instruments: list[dict[str, Any]] = []
    for metadata in _list_metadata(_instrument_bank_root(), "instrument.json"):
        instrument = metadata.get("instrument")
        if isinstance(instrument, dict):
            instruments.append(instrument)
    return sorted(instruments, key=lambda item: str(item.get("name") or ""))


def _asset_from_metadata(metadata: dict[str, Any], category: str, id_key: str) -> dict[str, Any] | None:
    audio_path = metadata.get("generated_audio_path")
    if not audio_path:
        return None
    path = Path(str(audio_path)).expanduser()
    if not path.exists() or not path.is_file():
        return None
    asset_id = str(metadata.get(id_key) or path.stem)
    label = str(metadata.get("label") or metadata.get("track_name") or metadata.get("prompt") or asset_id)
    return {
        "asset_id": asset_id,
        "category": category,
        "label": label,
        "audio_path": str(path),
        "audio_url": f"/api/editor/audio?path={quote(str(path))}",
        "duration_seconds": metadata.get("duration_seconds")
        or metadata.get("source_duration_seconds")
        or metadata.get("raw_generated_duration_seconds")
        or 0,
        "created_at": metadata.get("created_at") or "",
        "metadata_path": metadata.get("metadata_path") or "",
        "message": metadata.get("message") or "",
        "source_path": metadata.get("source_path") or "",
        "source_asset_id": metadata.get("source_asset_id") or "",
    }


def _list_metadata(root: Path, filename: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for metadata_path in root.glob(f"*/{filename}"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata["metadata_path"] = str(metadata_path)
        items.append(metadata)
    return items


def _editor_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []

    for metadata in _list_metadata(_transition_root(), "result.json"):
        asset = _asset_from_metadata(metadata, "transition", "generation_id")
        if asset:
            assets.append(asset)

    for metadata in _list_metadata(_music_generation_root(), "generation.json"):
        asset = _asset_from_metadata(metadata, "generation", "generation_id")
        if asset:
            assets.append(asset)

    for metadata in _list_metadata(_extraction_metadata_root(), "extraction.json"):
        if metadata.get("type") == "base_test":
            continue
        category = "merge" if metadata.get("type") == "merge" else "extraction"
        asset = _asset_from_metadata(metadata, category, "extraction_id")
        if asset:
            assets.append(asset)

    for metadata in _list_metadata(_edit_root(), "edit.json"):
        asset = _asset_from_metadata(metadata, "edit", "edit_id")
        if asset:
            assets.append(asset)

    for metadata in _list_metadata(_instrument_lab_root(), "clip.json"):
        category = "instrumenttrack" if metadata.get("type") == "instrumenttrack" else "instrument"
        asset = _asset_from_metadata(metadata, category, "clip_id")
        if asset:
            assets.append(asset)

    return sorted(assets, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def create_app(models_dir: Path = Path("models"), runtime_config: RuntimeConfig | None = None) -> FastAPI:
    runtime_config = runtime_config or RuntimeConfig()
    app = FastAPI(title="Dance Station", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    audiomass_dir = Path(__file__).resolve().parents[1] / "vendor" / "audiomass"
    ui_log = UiLog()
    ui_log.add("info", "UI server started.")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/audiomass", StaticFiles(directory=audiomass_dir, html=True), name="audiomass")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/audiomass")
    def audiomass_index() -> RedirectResponse:
        return RedirectResponse("/audiomass/")

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

    @app.get("/api/extractions/tracks")
    def get_extraction_tracks() -> list[str]:
        return EXTRACT_TRACKS

    @app.get("/api/extractions")
    def list_extractions() -> list[dict[str, Any]]:
        root = _extraction_metadata_root()
        if not root.exists():
            return []
        items: list[dict[str, Any]] = []
        for metadata_path in root.glob("*/extraction.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(metadata)
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    @app.get("/api/extractions/audio")
    def get_extraction_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        return get_audio_file(path)

    @app.get("/api/music-generations")
    def list_music_generations() -> list[dict[str, Any]]:
        root = _music_generation_root()
        if not root.exists():
            return []
        items: list[dict[str, Any]] = []
        for metadata_path in root.glob("*/generation.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(metadata)
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    @app.get("/api/music-generations/audio")
    def get_music_generation_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        return get_audio_file(path)

    @app.get("/api/editor/assets")
    def list_editor_assets() -> list[dict[str, Any]]:
        return _editor_assets()

    @app.get("/api/editor/audio")
    def get_editor_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        return get_audio_file(path)

    @app.get("/api/edits")
    def list_edits() -> list[dict[str, Any]]:
        return _list_metadata(_edit_root(), "edit.json")

    @app.get("/api/edits/audio")
    def get_edit_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        return get_audio_file(path)

    @app.get("/api/instrument-lab/clips")
    def list_instrument_lab_clips() -> list[dict[str, Any]]:
        return sorted(
            _list_metadata(_instrument_lab_root(), "clip.json"),
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )

    @app.get("/api/instrument-lab/audio")
    def get_instrument_lab_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        return get_audio_file(path)

    @app.get("/api/instrument-lab/instruments")
    def list_instrument_lab_instruments() -> list[dict[str, Any]]:
        return _list_user_instruments()

    @app.get("/api/instrument-lab/instruments/sample")
    def get_instrument_lab_instrument_sample(path: str = Query(..., min_length=1)) -> FileResponse:
        sample_path = Path(path).expanduser()
        try:
            sample_path.resolve().relative_to(_instrument_bank_root().resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Sample path is outside the instrument bank.") from exc
        return get_audio_file(str(sample_path))

    @app.post("/api/instrument-lab/instruments/sfz")
    def import_instrument_lab_sfz(
        sfz_file: UploadFile = File(...),
        sample_files: list[UploadFile] = File(default=[]),
        label: str = Form(..., min_length=1, max_length=120),
    ) -> dict[str, object]:
        import datetime as _datetime

        def fail(detail: str) -> None:
            ui_log.add("error", f"SFZ import failed: {detail}")
            raise HTTPException(status_code=400, detail=detail)

        sfz_name = Path(sfz_file.filename or "instrument.sfz").name
        if Path(sfz_name).suffix.lower() != ".sfz":
            fail("Upload an .sfz file.")
        instrument_id = f"user.sfz.{uuid4().hex[:12]}"
        instrument_dir = _instrument_bank_root() / instrument_id
        sample_dir = instrument_dir / "samples"
        metadata_path = instrument_dir / "instrument.json"
        instrument_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            sfz_text = sfz_file.file.read().decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            ui_log.add("error", "SFZ import failed: SFZ file must be UTF-8 text.")
            raise HTTPException(status_code=400, detail="SFZ file must be UTF-8 text.") from exc
        regions = _parse_sfz_regions(sfz_text)
        if not regions:
            fail("No playable SFZ regions found.")
        if not sample_files:
            referenced = sorted({Path(region["sample"].replace("\\", "/")).name for region in regions})
            fail(f"Upload the SFZ sample files too. Referenced samples: {', '.join(referenced[:12])}")

        stored_samples: dict[str, Path] = {}
        for sample in sample_files:
            sample_name = Path(sample.filename or "").name
            if not sample_name:
                continue
            sample_path = sample_dir / sample_name
            try:
                validate_supported_source(sample_path)
            except ValueError as exc:
                ui_log.add("error", f"SFZ import failed: {sample_name}: {exc}")
                raise HTTPException(status_code=400, detail=f"{sample_name}: {exc}") from exc
            with sample_path.open("wb") as output:
                shutil.copyfileobj(sample.file, output)
            stored_samples[sample_name.lower()] = sample_path

        instrument = _sfz_instrument_from_regions(
            instrument_id=instrument_id,
            label=label.strip(),
            regions=regions,
            stored_samples=stored_samples,
        )
        if not instrument["samples"]:
            missing = instrument.get("missing_samples") or []
            detail = "None of the SFZ sample references matched uploaded sample files."
            if missing:
                detail += f" Missing: {', '.join(missing[:12])}"
            fail(detail)
        if instrument.get("missing_samples"):
            ui_log.add("warning", f"SFZ import skipped missing samples: {', '.join(instrument['missing_samples'][:12])}")

        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
        metadata = {
            "instrument_id": instrument_id,
            "type": "sfz",
            "label": label.strip(),
            "created_at": created_at,
            "updated_at": created_at,
            "source_sfz_name": sfz_name,
            "metadata_path": str(metadata_path),
            "instrument": instrument,
        }
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Imported SFZ instrument: {label.strip()}")
        return {"instrument": instrument}

    @app.post("/api/instrument-lab/clips")
    def save_instrument_lab_clip(
        file: UploadFile = File(...),
        label: str = Form(..., min_length=1, max_length=120),
        project_json: str = Form(..., min_length=2),
        clip_type: Literal["instrument", "instrumenttrack"] = Form("instrument"),
    ) -> dict[str, object]:
        import datetime as _datetime

        try:
            project = json.loads(project_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid instrument project JSON.") from exc

        original_name = Path(file.filename or "instrument-lab.wav").name
        suffix = Path(original_name).suffix.lower() or ".wav"
        clip_id = f"{clip_type}-{uuid4().hex[:12]}"
        save_dir = _instrument_lab_root() / clip_id
        output_path = save_dir / f"{_safe_label_stem(label, clip_id)}{suffix}"
        metadata_path = save_dir / "clip.json"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()

        try:
            validate_supported_source(output_path)
        except ValueError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            probe = probe_audio(output_path)
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        metadata = {
            "clip_id": clip_id,
            "type": clip_type,
            "status": "complete",
            "label": label.strip(),
            "created_at": created_at,
            "updated_at": created_at,
            "generated_audio_path": str(output_path),
            "metadata_path": str(metadata_path),
            "duration_seconds": probe.duration_seconds,
            "source_format": probe.source_format,
            "project": project,
            "message": f"Instrument Lab {'track' if clip_type == 'instrumenttrack' else 'clip'} saved as {output_path.name}",
        }
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Saved Instrument Lab {clip_type}: {output_path}")
        return {"clip": metadata}

    @app.post("/api/edits")
    def save_edit(
        file: UploadFile = File(...),
        label: str = Form(..., min_length=1, max_length=120),
        source_asset_id: str | None = Form(None),
        source_category: str | None = Form(None),
    ) -> dict[str, object]:
        import datetime as _datetime

        original_name = Path(file.filename or "edit.wav").name
        suffix = Path(original_name).suffix.lower() or ".wav"
        temp_name = f"{_safe_label_stem(label, 'edit')}{suffix}"
        edit_id = f"edit-{uuid4().hex[:12]}"
        save_dir = _edit_root() / edit_id
        output_path = save_dir / temp_name
        metadata_path = save_dir / "edit.json"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()

        try:
            validate_supported_source(output_path)
        except ValueError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            probe = probe_audio(output_path)
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            file.file.close()

        metadata = {
            "edit_id": edit_id,
            "type": "edit",
            "status": "complete",
            "message": "Edit saved.",
            "created_at": created_at,
            "label": label.strip(),
            "original_filename": original_name,
            "source_asset_id": source_asset_id,
            "source_category": source_category,
            "output_format": suffix.lstrip("."),
            "generated_audio_path": str(output_path),
            "metadata_path": str(metadata_path),
            "duration_seconds": probe.duration_seconds,
            "source_format": probe.source_format,
        }
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Saved edited audio: {output_path}")
        return {"edit": metadata}

    @app.post("/api/music-generations/run")
    def run_music_generation(request: MusicGenerationRequest) -> dict[str, object]:
        import datetime as _datetime

        generation_id = f"music-{uuid4().hex[:12]}"
        save_dir = _music_generation_root() / generation_id
        metadata_path = save_dir / "generation.json"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
        prompt = request.prompt.strip()
        label = request.label.strip() if request.label else prompt[:80]
        ui_log.add("info", f"Running ACE-Step {request.model} text-to-music generation.")
        try:
            result = AceStepApiClient(runtime_config).text2music_standalone(
                prompt=prompt,
                model=request.model,
                save_dir=save_dir,
                audio_duration=request.audio_duration,
                audio_format=request.output_format,
                inference_steps=request.inference_steps,
                guidance_scale=request.guidance_scale,
                shift=request.shift,
                infer_method=request.infer_method,
                use_tiled_decode=request.use_tiled_decode,
                dcw_enabled=request.dcw_enabled,
                velocity_norm_threshold=request.velocity_norm_threshold,
                velocity_ema_factor=request.velocity_ema_factor,
                seed=request.seed,
            )
        except AceStepApiError as exc:
            ui_log.add("error", str(exc))
            metadata = {
                "generation_id": generation_id,
                "status": "failed",
                "message": str(exc),
                "created_at": created_at,
                "label": label,
                "prompt": prompt,
                "model": request.model,
                "output_format": request.output_format,
                "metadata_path": str(metadata_path),
                "settings": request.model_dump(),
            }
            _write_metadata(metadata_path, metadata)
            return {"generation": metadata}

        metadata = {
            "generation_id": generation_id,
            "status": "complete",
            "message": "Music generation complete.",
            "created_at": created_at,
            "label": label,
            "prompt": prompt,
            "model": request.model,
            "output_format": request.output_format,
            "generated_audio_path": str(result.output_path),
            "generated_metadata_path": str(result.metadata_path),
            "metadata_path": str(metadata_path),
            "settings": request.model_dump(),
        }
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Generated music: {result.output_path}")
        return {"generation": metadata}

    @app.post("/api/music-generations/{generation_id}/rename")
    def rename_music_generation(generation_id: str, request: ExtractionRenameRequest) -> dict[str, object]:
        metadata_path = _music_metadata_path(generation_id)
        metadata = _read_json_file(metadata_path, "Music generation")
        metadata["label"] = request.label.strip()
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Renamed music generation {generation_id}: {metadata['label']}")
        return {"generation": metadata}

    @app.post("/api/transitions/{generation_id}/rename")
    def rename_transition(generation_id: str, request: ExtractionRenameRequest) -> dict[str, object]:
        metadata_path = _transition_metadata_path(generation_id)
        metadata = _read_json_file(metadata_path, "Transition")
        metadata["label"] = request.label.strip()
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Renamed transition {generation_id}: {metadata['label']}")
        return {"transition": metadata}

    @app.post("/api/edits/{edit_id}/rename")
    def rename_edit(edit_id: str, request: ExtractionRenameRequest) -> dict[str, object]:
        metadata_path = _edit_metadata_path(edit_id)
        metadata = _read_json_file(metadata_path, "Edit")
        metadata["label"] = request.label.strip()
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Renamed edit {edit_id}: {metadata['label']}")
        return {"edit": metadata}

    @app.post("/api/instrument-lab/clips/{clip_id}/rename")
    def rename_instrument_lab_clip(clip_id: str, request: ExtractionRenameRequest) -> dict[str, object]:
        metadata_path = _instrument_lab_metadata_path(clip_id)
        metadata = _read_json_file(metadata_path, "Instrument clip")
        metadata["label"] = request.label.strip()
        metadata["updated_at"] = metadata.get("updated_at") or metadata.get("created_at") or ""
        _write_metadata(metadata_path, metadata)
        ui_log.add("info", f"Renamed Instrument Lab clip {clip_id}: {metadata['label']}")
        return {"clip": metadata}

    @app.post("/api/extractions/{extraction_id}/rename")
    def rename_extraction(extraction_id: str, request: ExtractionRenameRequest) -> dict[str, object]:
        metadata = _read_extraction_metadata(extraction_id)
        if metadata.get("type") == "base_test":
            raise HTTPException(status_code=400, detail="Base Test items cannot be renamed here.")
        metadata["label"] = request.label.strip()
        _write_extraction_metadata(metadata)
        ui_log.add("info", f"Renamed extraction {extraction_id}: {metadata['label']}")
        return {"extraction": metadata}

    @app.post("/api/extractions/merge")
    def merge_extractions(request: ExtractionMergeRequest) -> dict[str, object]:
        import datetime as _datetime

        selected = [_read_extraction_metadata(extraction_id) for extraction_id in request.extraction_ids]
        source_paths: list[Path] = []
        for metadata in selected:
            if metadata.get("type") == "base_test":
                raise HTTPException(status_code=400, detail="Base Test items cannot be merged.")
            if metadata.get("status") != "complete":
                raise HTTPException(status_code=400, detail=f"Only complete items can be merged: {metadata.get('extraction_id')}")
            audio_path = metadata.get("generated_audio_path")
            if not audio_path:
                raise HTTPException(status_code=400, detail=f"Item has no generated audio: {metadata.get('extraction_id')}")
            source_paths.append(Path(str(audio_path)))

        merge_id = f"merge-{uuid4().hex[:12]}"
        save_dir = _extraction_metadata_root() / merge_id
        output_path = save_dir / f"{merge_id}.{request.output_format}"
        metadata_path = save_dir / "extraction.json"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
        label = request.label.strip()
        try:
            merge_audio_files(source_paths, output_path, request.output_format)
            probe = probe_audio(output_path)
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        metadata = {
            "extraction_id": merge_id,
            "type": "merge",
            "status": "complete",
            "message": "Merge complete.",
            "created_at": created_at,
            "label": label,
            "track_name": label,
            "output_format": request.output_format,
            "generated_audio_path": str(output_path),
            "generated_metadata_path": str(metadata_path),
            "metadata_path": str(metadata_path),
            "source_extraction_ids": request.extraction_ids,
            "source_audio_paths": [str(path) for path in source_paths],
            "source_duration_seconds": probe.duration_seconds,
            "settings": {
                "output_format": request.output_format,
                "source_extraction_ids": request.extraction_ids,
            },
        }
        _write_extraction_metadata(metadata)
        ui_log.add("info", f"Merged {len(source_paths)} extraction items: {output_path}")
        return {"extraction": metadata}

    @app.post("/api/extractions/source/probe")
    def probe_extraction_source(request: ProbeRequest) -> dict[str, object]:
        return probe_source(request)

    @app.post("/api/extractions/source/upload")
    def upload_extraction_source(file: UploadFile = File(...)) -> dict[str, object]:
        return upload_source(file)

    @app.post("/api/extractions/run")
    def run_extraction(request: ExtractionRunRequest) -> dict[str, object]:
        import datetime as _datetime

        track_name = request.track_name.strip().lower()
        if track_name not in EXTRACT_TRACKS:
            raise HTTPException(status_code=400, detail=f"Unknown extract track: {request.track_name}")

        source_path = Path(request.source_path).expanduser()
        try:
            probe = probe_audio(source_path)
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        extraction_id = f"extraction-{uuid4().hex[:12]}"
        save_dir = Path("data/extractions") / extraction_id
        metadata_path = save_dir / "extraction.json"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
        ui_log.add("info", f"Running ACE-Step extract for {track_name}; base model will be loaded in the ACE runtime if needed.")

        try:
            result = AceStepApiClient(runtime_config).extract_track(
                source_path=source_path,
                track_name=track_name,
                save_dir=save_dir,
                audio_format=request.output_format,
                inference_steps=request.inference_steps,
                guidance_scale=request.guidance_scale,
                shift=request.shift,
                infer_method=request.infer_method,
                use_tiled_decode=request.use_tiled_decode,
                dcw_enabled=request.dcw_enabled,
                velocity_norm_threshold=request.velocity_norm_threshold,
                velocity_ema_factor=request.velocity_ema_factor,
                seed=request.seed,
                instruction=request.instruction.strip() if request.instruction else None,
            )
        except AceStepApiError as exc:
            ui_log.add("error", str(exc))
            metadata = {
                "extraction_id": extraction_id,
                "status": "failed",
                "message": str(exc),
                "created_at": created_at,
                "source_path": str(source_path),
                "source_format": probe.source_format,
                "source_duration_seconds": probe.duration_seconds,
                "track_name": track_name,
                "label": request.label.strip() if request.label else track_name,
                "output_format": request.output_format,
                "metadata_path": str(metadata_path),
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return {"extraction": metadata}

        metadata = {
            "extraction_id": extraction_id,
            "status": "complete",
            "message": "Extraction complete.",
            "created_at": created_at,
            "source_path": str(source_path),
            "source_format": probe.source_format,
            "source_duration_seconds": probe.duration_seconds,
            "track_name": track_name,
            "label": request.label.strip() if request.label else track_name,
            "output_format": request.output_format,
            "generated_audio_path": str(result.output_path),
            "generated_metadata_path": str(result.metadata_path),
            "metadata_path": str(metadata_path),
            "settings": {
                "inference_steps": request.inference_steps,
                "guidance_scale": request.guidance_scale,
                "shift": request.shift,
                "infer_method": request.infer_method,
                "use_tiled_decode": request.use_tiled_decode,
                "dcw_enabled": request.dcw_enabled,
                "velocity_norm_threshold": request.velocity_norm_threshold,
                "velocity_ema_factor": request.velocity_ema_factor,
                "seed": request.seed,
                "instruction": request.instruction,
            },
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        ui_log.add("info", f"Extracted {track_name}: {result.output_path}")
        return {"extraction": metadata}

    @app.post("/api/extractions/base-test")
    def run_base_generation_test(request: BaseGenerationTestRequest) -> dict[str, object]:
        import datetime as _datetime

        generation_id = f"base-test-{uuid4().hex[:12]}"
        save_dir = Path("data/extractions") / generation_id
        metadata_path = save_dir / "extraction.json"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
        prompt = request.prompt.strip()
        ui_log.add("info", "Running ACE-Step Base text-to-music test generation.")

        try:
            result = AceStepApiClient(runtime_config).text2music_base_test(
                prompt=prompt,
                save_dir=save_dir,
                audio_duration=request.audio_duration,
                audio_format=request.output_format,
                inference_steps=request.inference_steps,
                guidance_scale=request.guidance_scale,
                shift=request.shift,
                infer_method=request.infer_method,
                use_tiled_decode=request.use_tiled_decode,
                dcw_enabled=request.dcw_enabled,
                velocity_norm_threshold=request.velocity_norm_threshold,
                velocity_ema_factor=request.velocity_ema_factor,
                seed=request.seed,
            )
        except AceStepApiError as exc:
            ui_log.add("error", str(exc))
            metadata = {
                "extraction_id": generation_id,
                "type": "base_test",
                "status": "failed",
                "message": str(exc),
                "created_at": created_at,
                "track_name": "Base text2music test",
                "prompt": prompt,
                "output_format": request.output_format,
                "metadata_path": str(metadata_path),
                "settings": request.model_dump(),
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return {"extraction": metadata}

        metadata = {
            "extraction_id": generation_id,
            "type": "base_test",
            "status": "complete",
            "message": "Base text-to-music test complete.",
            "created_at": created_at,
            "track_name": "Base text2music test",
            "prompt": prompt,
            "output_format": request.output_format,
            "generated_audio_path": str(result.output_path),
            "generated_metadata_path": str(result.metadata_path),
            "metadata_path": str(metadata_path),
            "settings": request.model_dump(),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        ui_log.add("info", f"Generated Base test audio: {result.output_path}")
        return {"extraction": metadata}

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
        import datetime as _datetime

        generation_id = f"generation-{uuid4().hex[:12]}"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
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
            raw_probe = probe_audio(raw_generation.output_path)
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
            repaint_start = max(0.0, plan.continuation_point_seconds - plan.repaint_margin_seconds)
            repaint_end = plan.continuation_point_seconds + raw_probe.duration_seconds
            composite_metadata = {
                "generation_id": generation_id,
                "raw_generated_audio_path": str(raw_generation.output_path),
                "raw_generated_metadata_path": str(raw_generation.metadata_path),
                "raw_generated_duration_seconds": raw_probe.duration_seconds,
                "composite_audio_path": str(composite_path),
                "continuation_point_seconds": plan.continuation_point_seconds,
                "new_section_seconds": plan.new_section_seconds,
                "boundary_repaint": True,
                "boundary_repaint_start_seconds": repaint_start,
                "boundary_repaint_end_seconds": repaint_end,
            }
            composite_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            composite_metadata_path.write_text(json.dumps(composite_metadata, indent=2), encoding="utf-8")

            ui_log.add(
                "info",
                f"Running ACE-Step turbo repaint from {repaint_start:.2f}s to {repaint_end:.2f}s.",
            )
            boundary_plan = SourceSelectionPlan(
                **{
                    **plan.to_dict(),
                    "source_path": plan.source_path,
                    "scaffold_path": composite_path,
                    "metadata_path": composite_metadata_path,
                    "tail_start_seconds": 0.0,
                    "tail_end_seconds": plan.continuation_point_seconds + plan.new_section_seconds,
                    "repainting_start_seconds": repaint_start,
                    "repainting_end_seconds": repaint_end,
                    "generation_region": "repaint_existing",
                }
            )
            boundary_result = adapter.repaint_transition(boundary_plan)
            final_audio_path = boundary_result.output_path
            final_metadata_path = boundary_result.metadata_path
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
        transition_metadata_path = output.generated_dir / generation_id / "result.json"
        transition_metadata = {
            **result.to_dict(),
            "type": "transition",
            "created_at": created_at,
            "label": (plan.caption or generation_id)[:80],
            "caption": plan.caption,
            "source_path": str(plan.source_path),
            "source_format": plan.source_format,
            "continuation_point_seconds": plan.continuation_point_seconds,
            "new_section_seconds": plan.new_section_seconds,
            "settings": {
                "preset": request.preset,
                "model_slug": profile.slug,
                "context_seconds": config.context_seconds,
                "repaint_overlap_seconds": config.repaint_overlap_seconds,
                "new_section_seconds": config.new_section_seconds,
                "bpm": config.bpm_hint,
                "key": config.key_hint,
                "seed": config.seed,
                "ace_step": request.ace_step.to_payload() if request.ace_step else None,
            },
            "metadata_path": str(transition_metadata_path),
        }
        _write_metadata(transition_metadata_path, transition_metadata)
        ui_log.add("info", f"Generated transition: {final_audio_path}")
        return {"result": transition_metadata, "plan": plan.to_dict()}

    @app.get("/api/logs")
    def get_logs() -> list[dict[str, str]]:
        return ui_log.entries()

    return app

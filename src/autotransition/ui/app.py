"""FastAPI app for the local Autotransition UI."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
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
from autotransition.library.index import LocalLibraryIndex
from autotransition.library.publish import (
    LibraryPublishError,
    LibraryPublisher,
    LibraryPublishSettings,
    PublicLibraryClient,
    load_publish_settings,
    public_settings_response,
    save_publish_settings,
)
from autotransition.library.schema import LibraryFile, LibraryItem, audio_mime_type_for_path, library_item_from_editor_asset
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
from autotransition.runtime.side_step import build_side_step_command, side_step_status
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
    audio_duration: float = Field(30.0, ge=10.0, le=300.0)
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


class LocalLibraryUpdateRequest(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=160)
    description: str | None = Field(None, max_length=3000)
    tags: list[str] = Field(default_factory=list)
    license: str | None = Field(None, max_length=160)
    attribution: str | None = Field(None, max_length=1000)


class PublicLibraryConnectionRequest(BaseModel):
    site_url: str = Field("http://localhost:3001", min_length=1, max_length=500)
    token: str | None = Field(None, max_length=400)


class PublicLibraryPublishRequest(BaseModel):
    publish_public: bool = True


class ExtractionMergeRequest(BaseModel):
    extraction_ids: list[str] = Field(..., min_length=2)
    label: str = Field(..., min_length=1, max_length=120)
    output_format: Literal["flac", "wav", "wav32", "mp3", "opus", "aac"] = "flac"


class MusicGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "acestep-v15-turbo"
    label: str | None = None
    instrumental: bool = True
    lyrics: str | None = None
    vocal_language: str = "unknown"
    output_format: Literal["flac", "wav", "wav32", "mp3", "opus", "aac"] = "flac"
    audio_duration: float = Field(30.0, ge=10.0, le=300.0)
    inference_steps: int = Field(8, ge=1, le=200)
    guidance_scale: float = Field(1.0, ge=0)
    shift: float = Field(3.0, ge=0)
    infer_method: Literal["ode", "sde"] = "ode"
    use_tiled_decode: bool = True
    dcw_enabled: bool = False
    velocity_norm_threshold: float = Field(0.0, ge=0)
    velocity_ema_factor: float = Field(0.0, ge=0, le=1)
    seed: int | None = None
    lokr_adapter_id: str | None = None
    lokr_scale: float = Field(1.0, ge=0.0, le=1.0)


class LokrDatasetCreateRequest(BaseModel):
    label: str = Field("New LoKr dataset", min_length=1, max_length=120)
    custom_tag: str | None = None
    default_genre: str | None = None
    default_language: str = "unknown"
    tag_position: Literal["prepend", "append", "replace"] = "prepend"
    genre_ratio: int = Field(0, ge=0, le=100)
    all_instrumental: bool = True


class LokrDatasetSaveRequest(BaseModel):
    dataset: dict[str, Any]


class LokrDatasetAssetRequest(BaseModel):
    asset_id: str = Field(..., min_length=1)


class LokrPreprocessRequest(BaseModel):
    model: Literal["turbo", "base"] = "turbo"
    sidestep_command: str = "uv run sidestep"
    checkpoint_dir: str = "runtimes/ACE-Step-1.5/checkpoints"


class LokrTrainRequest(BaseModel):
    model: Literal["turbo", "base"] = "turbo"
    sidestep_command: str = "uv run sidestep"
    checkpoint_dir: str = "runtimes/ACE-Step-1.5/checkpoints"
    tensor_dir: str | None = None
    epochs: int = Field(500, ge=1)
    lokr_linear_dim: int = Field(64, ge=1)
    lokr_linear_alpha: int = Field(128, ge=1)
    save_every: int = Field(10, ge=1)
    optimizer_type: str = "adamw8bit"
    batch_size: int = Field(1, ge=1)
    gradient_accumulation: int = Field(4, ge=1)
    gradient_checkpointing: bool = True
    offload_encoder: bool = True
    chunk_duration: int | None = Field(None, ge=1)


def _music_generation_model(value: str) -> str:
    model = (value or "").strip()
    if model in {"acestep-v15-base", "acestep-v15-xl-base"}:
        return "acestep-v15-base"
    return "acestep-v15-turbo"


def _lokr_training_model_to_generation_model(value: str | None) -> str:
    model = (value or "").strip()
    if model == "base":
        return "acestep-v15-base"
    return "acestep-v15-turbo"


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


def _lokr_root() -> Path:
    return Path("data/lokr-training")


def _lokr_dataset_root() -> Path:
    return _lokr_root() / "datasets"


def _lokr_run_root() -> Path:
    return _lokr_root() / "runs"


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


def _lokr_dataset_path(dataset_id: str) -> Path:
    return _lokr_dataset_root() / _safe_item_id(dataset_id, "LoKr dataset") / "dataset.json"


def _lokr_run_path(run_id: str) -> Path:
    return _lokr_run_root() / _safe_item_id(run_id, "LoKr run") / "run.json"


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


def _lokr_dataset_audio_dir(dataset_id: str) -> Path:
    return _lokr_dataset_root() / _safe_item_id(dataset_id, "LoKr dataset") / "audio"


def _lokr_audio_path_for_response(dataset_id: str, audio_path: str) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    dataset_dir = _lokr_dataset_root() / _safe_item_id(dataset_id, "LoKr dataset")
    relative = str(audio_path).replace("\\", "/")
    if relative.startswith("./"):
        relative = relative[2:]
    return dataset_dir / relative


def _lokr_clean_sample(
    dataset_id: str,
    sample: dict[str, Any],
    fallback_tag: str = "",
    fallback_genre: str = "",
    fallback_language: str = "unknown",
) -> dict[str, Any]:
    sample_id = str(sample.get("id") or uuid4().hex[:8])
    audio_path = str(sample.get("audio_path") or "")
    filename = str(sample.get("filename") or Path(audio_path).name or "audio")
    is_instrumental = bool(sample.get("is_instrumental", True))
    lyrics = str(sample.get("lyrics") or "").strip()
    if is_instrumental or not lyrics:
        lyrics = "[Instrumental]"
        is_instrumental = True
    caption = str(sample.get("caption") or "").strip()
    return {
        "id": sample_id,
        "audio_path": audio_path,
        "filename": filename,
        "label": str(sample.get("label") or Path(filename).stem),
        "caption": caption,
        "genre": str(sample.get("genre") or fallback_genre or "").strip(),
        "lyrics": lyrics,
        "raw_lyrics": str(sample.get("raw_lyrics") or "").strip(),
        "formatted_lyrics": str(sample.get("formatted_lyrics") or lyrics).strip() or lyrics,
        "bpm": sample.get("bpm") if sample.get("bpm") not in ("", None) else "N/A",
        "keyscale": str(sample.get("keyscale") or "N/A"),
        "timesignature": str(sample.get("timesignature") or "4"),
        "duration": float(sample.get("duration") or 0),
        "language": str(sample.get("language") or fallback_language or "unknown"),
        "is_instrumental": is_instrumental,
        "custom_tag": str(sample.get("custom_tag") or fallback_tag or ""),
        "prompt_override": sample.get("prompt_override") or None,
        "labeled": bool(sample.get("labeled", bool(caption))),
        "source_asset_id": str(sample.get("source_asset_id") or ""),
        "source_category": str(sample.get("source_category") or ""),
    }


def _lokr_clean_dataset(dataset: dict[str, Any], dataset_id: str | None = None) -> dict[str, Any]:
    import datetime as _datetime

    metadata = dict(dataset.get("metadata") or {})
    dataset_id = dataset_id or str(metadata.get("dataset_id") or dataset.get("dataset_id") or f"lokr-{uuid4().hex[:12]}")
    created_at = str(metadata.get("created_at") or dataset.get("created_at") or _datetime.datetime.now(_datetime.UTC).isoformat())
    updated_at = _datetime.datetime.now(_datetime.UTC).isoformat()
    label = str(metadata.get("label") or metadata.get("name") or dataset.get("label") or "LoKr dataset").strip()
    custom_tag = str(metadata.get("custom_tag") or "").strip()
    default_genre = str(metadata.get("default_genre") or "").strip()
    default_language = str(metadata.get("default_language") or "unknown").strip() or "unknown"
    samples = [
        _lokr_clean_sample(dataset_id, sample, custom_tag, default_genre, default_language)
        for sample in list(dataset.get("samples") or [])
    ]
    all_instrumental = bool(metadata.get("all_instrumental", True))
    return {
        "metadata": {
            "dataset_id": dataset_id,
            "label": label,
            "name": str(metadata.get("name") or label),
            "custom_tag": custom_tag,
            "default_genre": default_genre,
            "default_language": default_language,
            "tag_position": str(metadata.get("tag_position") or "prepend"),
            "genre_ratio": int(metadata.get("genre_ratio") or 0),
            "all_instrumental": all_instrumental,
            "created_at": created_at,
            "updated_at": updated_at,
            "num_samples": len(samples),
        },
        "samples": samples,
    }


def _lokr_dataset_for_response(dataset: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(dataset.get("metadata", {}).get("dataset_id") or "")
    response = json.loads(json.dumps(dataset))
    response["metadata_path"] = str(_lokr_dataset_path(dataset_id)) if dataset_id else ""
    for sample in response.get("samples", []):
        audio_path = str(sample.get("audio_path") or "")
        resolved = _lokr_audio_path_for_response(dataset_id, audio_path) if dataset_id and audio_path else Path()
        sample["resolved_audio_path"] = str(resolved) if str(resolved) != "." else ""
        sample["audio_url"] = f"/api/lokr/audio?path={quote(str(resolved))}" if resolved and str(resolved) != "." else ""
    return response


def _read_lokr_dataset(dataset_id: str) -> dict[str, Any]:
    metadata_path = _lokr_dataset_path(dataset_id)
    dataset = _read_json_file(metadata_path, "LoKr dataset")
    dataset.pop("metadata_path", None)
    return _lokr_clean_dataset(dataset, dataset_id=dataset_id)


def _write_lokr_dataset(dataset: dict[str, Any], dataset_id: str | None = None) -> dict[str, Any]:
    clean = _lokr_clean_dataset(dataset, dataset_id=dataset_id)
    metadata_path = _lokr_dataset_path(str(clean["metadata"]["dataset_id"]))
    _write_metadata(metadata_path, clean)
    return clean


def _copy_lokr_audio(dataset_id: str, source_path: Path, label: str) -> tuple[str, Path, float]:
    validate_supported_source(source_path)
    audio_dir = _lokr_dataset_audio_dir(dataset_id)
    audio_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_label_stem(label or source_path.stem, "sample")
    target = audio_dir / f"{stem}{source_path.suffix.lower()}"
    if target.exists():
        target = audio_dir / f"{stem}-{uuid4().hex[:8]}{source_path.suffix.lower()}"
    shutil.copy2(source_path, target)
    duration = 0.0
    try:
        duration = float(probe_audio(target).duration_seconds)
    except Exception:
        duration = 0.0
    return f"./audio/{target.name}", target, duration


def _lokr_sample_from_audio(
    *,
    dataset_id: str,
    source_path: Path,
    label: str,
    default_genre: str = "",
    default_language: str = "unknown",
    source_asset_id: str = "",
    source_category: str = "",
) -> dict[str, Any]:
    relative_audio_path, target, duration = _copy_lokr_audio(dataset_id, source_path, label)
    return _lokr_clean_sample(
        dataset_id,
        {
            "id": f"sample-{uuid4().hex[:10]}",
            "audio_path": relative_audio_path,
            "filename": target.name,
            "label": label or target.stem,
            "caption": "",
            "genre": default_genre,
            "lyrics": "[Instrumental]",
            "duration": duration,
            "language": default_language,
            "is_instrumental": True,
            "source_asset_id": source_asset_id,
            "source_category": source_category,
        },
    )


def _lokr_latest_tensor_dir(dataset_id: str) -> str:
    candidates: list[tuple[str, Path]] = []
    for metadata in _list_metadata(_lokr_run_root(), "run.json"):
        if (
            metadata.get("dataset_id") == dataset_id
            and metadata.get("type") == "preprocess"
            and metadata.get("status") == "running"
        ):
            proc = _LOKR_PROCESSES.get(str(metadata.get("run_id")))
            if proc is not None and proc.poll() is not None:
                metadata["status"] = "complete" if proc.returncode == 0 else "failed"
                metadata["returncode"] = proc.returncode
                metadata["completed_at"] = _now_iso()
                _write_metadata(_lokr_run_path(str(metadata["run_id"])), metadata)
        if (
            metadata.get("dataset_id") == dataset_id
            and metadata.get("type") == "preprocess"
            and metadata.get("status") == "complete"
            and metadata.get("tensor_dir")
        ):
            candidates.append((str(metadata.get("created_at") or ""), Path(str(metadata["tensor_dir"]))))
    if not candidates:
        return ""
    return str(sorted(candidates, key=lambda item: item[0], reverse=True)[0][1])


def _now_iso() -> str:
    import datetime as _datetime

    return _datetime.datetime.now(_datetime.UTC).isoformat()


_LOKR_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}


def _active_lokr_run() -> dict[str, Any] | None:
    for run in _lokr_runs():
        if run.get("status") == "running":
            return run
    return None


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _lokr_preprocess_log_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("type") != "preprocess":
        return {}
    log_path = Path(str(metadata.get("log_path") or ""))
    if not log_path.exists():
        return {}
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    processed_match = re.search(r"Preprocessing complete:\s*(\d+)/(\d+)\s+processed,\s*(\d+)\s+failed", log_text)
    if processed_match:
        return {
            "processed_samples": int(processed_match.group(1)),
            "total_samples": int(processed_match.group(2)),
            "failed_samples": int(processed_match.group(3)),
            "summary": f"Processed {processed_match.group(1)}/{processed_match.group(2)} samples",
        }
    processed_line = re.search(r"Processed:\s*(\d+)/(\d+)", log_text)
    if processed_line:
        return {
            "processed_samples": int(processed_line.group(1)),
            "total_samples": int(processed_line.group(2)),
            "summary": f"Processed {processed_line.group(1)}/{processed_line.group(2)} samples",
        }
    return {}


def _lokr_train_log_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("type") != "train":
        return {}
    progress_summary = _lokr_train_progress_summary(metadata)
    if progress_summary:
        return progress_summary
    log_path = Path(str(metadata.get("log_path") or ""))
    if not log_path.is_file():
        return {}
    log_text = _strip_ansi(log_path.read_text(encoding="utf-8", errors="replace"))
    if "Training summary complete" in log_text:
        steps = re.search(r"Training summary complete \(steps=(\d+)\)", log_text)
        summary = f"Training complete ({steps.group(1)} steps)" if steps else "Training complete"
        return {"summary": summary}
    session_match = re.search(r"\[INFO\]\s+Session:\s*(.+)", log_text)
    sample_match = re.search(r"PreprocessedTensorDataset:\s*(\d+)\s+samples", log_text)
    epoch_matches = re.findall(r"Epoch\s+(\d+)(?:/(\d+))?", log_text)
    loss_matches = re.findall(r"(?:train/)?loss[=:\s]+([0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)", log_text, flags=re.IGNORECASE)
    parts: list[str] = []
    if epoch_matches:
        epoch, total = epoch_matches[-1]
        parts.append(f"Epoch {epoch}{f'/{total}' if total else ''}")
    if loss_matches:
        parts.append(f"loss {loss_matches[-1]}")
    if sample_match:
        parts.append(f"{sample_match.group(1)} samples")
    if parts:
        return {"summary": "Training " + " | ".join(parts)}
    meaningful_lines = [
        line.strip()
        for line in log_text.replace("\r", "\n").splitlines()
        if line.strip() and not set(line.strip()) <= {"=", "-", "*"}
    ]
    if meaningful_lines:
        last_line = meaningful_lines[-1]
        if len(last_line) > 180:
            last_line = f"{last_line[:177]}..."
        result = {"summary": last_line}
        if session_match:
            result["session"] = session_match.group(1).strip()
        return result
    return {}


def _lokr_train_progress_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    progress_path = _lokr_train_progress_path(metadata)
    if not progress_path.exists():
        return {}
    last: dict[str, Any] | None = None
    for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    if not last:
        return {}
    epoch = last.get("epoch")
    max_epochs = last.get("max_epochs")
    step = last.get("step")
    loss = last.get("loss")
    summary_parts = []
    if epoch is not None:
        summary_parts.append(f"Epoch {epoch}{f'/{max_epochs}' if max_epochs else ''}")
    if step is not None:
        summary_parts.append(f"step {step}")
    if loss is not None:
        try:
            summary_parts.append(f"loss {float(loss):.4f}")
        except (TypeError, ValueError):
            summary_parts.append(f"loss {loss}")
    if not summary_parts:
        return {}
    return {
        "summary": "Training " + " | ".join(summary_parts),
        "progress_path": str(progress_path),
        "current_epoch": epoch,
        "max_epochs": max_epochs,
        "current_step": step,
        "loss": loss,
    }


def _lokr_train_progress_path(metadata: dict[str, Any]) -> Path:
    output_dir = Path(str(metadata.get("output_dir") or ""))
    if not output_dir:
        return Path()
    return output_dir / ".progress.jsonl"


def _lokr_train_session_log_path(metadata: dict[str, Any]) -> Path:
    output_dir = Path(str(metadata.get("output_dir") or ""))
    if not output_dir:
        return Path()
    session_dir = output_dir / "session_logs"
    if not session_dir.exists():
        return Path()
    logs = sorted(session_dir.glob("*_ui.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else Path()


def _lokr_enrich_run(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = _lokr_preprocess_log_summary(metadata)
    if not summary:
        summary = _lokr_train_log_summary(metadata)
    if summary:
        metadata.update(summary)
    if metadata.get("type") == "preprocess" and metadata.get("status") == "complete":
        metadata["ready_to_train"] = bool(metadata.get("tensor_dir")) and Path(str(metadata["tensor_dir"])).exists()
    return metadata


def _hidden_subprocess_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


def _lokr_run_metadata(
    *,
    run_id: str,
    run_type: str,
    dataset_id: str,
    label: str,
    command: list[str],
    log_path: Path,
    model: str,
    tensor_dir: Path | None = None,
    output_dir: Path | None = None,
    cwd: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "type": run_type,
        "dataset_id": dataset_id,
        "label": label,
        "status": "pending",
        "model": model,
        "command": command,
        "log_path": str(log_path),
        "cwd": str(cwd) if cwd is not None else "",
        "created_at": _now_iso(),
        "started_at": "",
        "completed_at": "",
        "returncode": None,
    }
    if tensor_dir is not None:
        metadata["tensor_dir"] = str(tensor_dir)
    if output_dir is not None:
        metadata["output_dir"] = str(output_dir)
        metadata["adapter_dir"] = str(output_dir / "final")
    if extra:
        metadata.update(extra)
    return metadata


def _start_lokr_process(metadata: dict[str, Any]) -> dict[str, Any]:
    run_id = str(metadata["run_id"])
    run_dir = _lokr_run_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(metadata["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metadata["status"] = "running"
    metadata["started_at"] = _now_iso()
    _write_metadata(_lokr_run_path(run_id), metadata)
    try:
        log_file = log_path.open("ab")
        cwd = str(metadata.get("cwd") or "") or None
        process = subprocess.Popen(
            metadata["command"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            **_hidden_subprocess_kwargs(),
        )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["message"] = str(exc)
        metadata["completed_at"] = _now_iso()
        _write_metadata(_lokr_run_path(run_id), metadata)
        return metadata
    _LOKR_PROCESSES[run_id] = process
    metadata["pid"] = process.pid
    _write_metadata(_lokr_run_path(run_id), metadata)
    return metadata


def _stop_lokr_process(run_id: str) -> dict[str, Any]:
    metadata = _read_json_file(_lokr_run_path(run_id), "LoKr run")
    if metadata.get("status") != "running":
        return _lokr_enrich_run(_refresh_lokr_run(metadata))
    process = _LOKR_PROCESSES.get(run_id)
    if process is None:
        raise HTTPException(
            status_code=409,
            detail="This Side-Step run is not managed by the current app process. Stop it from the terminal or restart the machine/runtime.",
        )
    metadata = _refresh_lokr_run(metadata)
    if metadata.get("status") != "running":
        return _lokr_enrich_run(metadata)
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop Side-Step run: {exc}") from exc
    finally:
        _LOKR_PROCESSES.pop(run_id, None)
    metadata["status"] = "stopped"
    metadata["message"] = "Stopped by user."
    metadata["returncode"] = process.poll()
    metadata["completed_at"] = _now_iso()
    _write_metadata(_lokr_run_path(run_id), metadata)
    return _lokr_enrich_run(metadata)


def _refresh_lokr_run(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("status") != "running":
        return _lokr_enrich_run(metadata)
    run_id = str(metadata.get("run_id") or "")
    process = _LOKR_PROCESSES.get(run_id)
    log_summary = _lokr_preprocess_log_summary(metadata)
    if log_summary:
        metadata.update(log_summary)
        if (
            metadata.get("type") == "preprocess"
            and metadata.get("total_samples")
            and metadata.get("processed_samples") == metadata.get("total_samples")
            and int(metadata.get("failed_samples") or 0) == 0
        ):
            metadata["returncode"] = 0
            metadata["status"] = "complete"
            metadata["completed_at"] = metadata.get("completed_at") or _now_iso()
            _write_metadata(_lokr_run_path(run_id), metadata)
            return _lokr_enrich_run(metadata)
    if process is None:
        metadata["status"] = "unknown"
        metadata["message"] = "Run was started by a previous app process. Check the log file for progress."
        _write_metadata(_lokr_run_path(run_id), metadata)
        return _lokr_enrich_run(metadata)
    returncode = process.poll()
    if returncode is None:
        return _lokr_enrich_run(metadata)
    _LOKR_PROCESSES.pop(run_id, None)
    metadata["returncode"] = returncode
    metadata["status"] = "complete" if returncode == 0 else "failed"
    log_path = Path(str(metadata.get("log_path") or ""))
    if returncode == 0 and log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "No audio files found" in log_text or "Processed: 0/0" in log_text:
            metadata["status"] = "failed"
            metadata["message"] = "Side-Step finished without processing any audio. Check dataset paths and run logs."
    metadata["completed_at"] = _now_iso()
    _write_metadata(_lokr_run_path(run_id), metadata)
    return _lokr_enrich_run(metadata)


def _lokr_runs() -> list[dict[str, Any]]:
    runs = [_refresh_lokr_run(metadata) for metadata in _list_metadata(_lokr_run_root(), "run.json")]
    return sorted(runs, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def _lokr_adapter_weight_path(metadata: dict[str, Any]) -> Path | None:
    output_dir_raw = str(metadata.get("output_dir") or "")
    if not output_dir_raw:
        return None
    output_dir = Path(output_dir_raw).expanduser()
    best = output_dir / "best" / "lokr_weights.safetensors"
    if best.exists():
        return best
    checkpoint_root = output_dir / "checkpoints"
    checkpoints: list[Path] = []
    if checkpoint_root.exists():
        checkpoints = [path for path in checkpoint_root.glob("epoch_*/lokr_weights.safetensors") if path.exists()]
    if checkpoints:
        def checkpoint_sort_key(path: Path) -> tuple[int, float]:
            match = re.search(r"epoch_(\d+)", str(path.parent.name))
            return (int(match.group(1)) if match else -1, path.stat().st_mtime)

        return sorted(checkpoints, key=checkpoint_sort_key, reverse=True)[0]
    direct = output_dir / "lokr_weights.safetensors"
    if direct.exists():
        return direct
    return None


def _lokr_adapter_for_response(metadata: dict[str, Any]) -> dict[str, Any] | None:
    if metadata.get("type") != "train" or metadata.get("status") != "complete" or metadata.get("adapter_type") != "lokr":
        return None
    weight_path = _lokr_adapter_weight_path(metadata)
    if weight_path is None:
        return None
    run_id = str(metadata.get("run_id") or "")
    label = str(metadata.get("label") or run_id or "LoKr adapter")
    if label.lower().startswith("train lokr "):
        label = label[11:].strip() or label
    model = _lokr_training_model_to_generation_model(str(metadata.get("model") or "turbo"))
    return {
        "adapter_id": run_id,
        "run_id": run_id,
        "dataset_id": str(metadata.get("dataset_id") or ""),
        "label": label,
        "model": model,
        "training_model": str(metadata.get("model") or ""),
        "adapter_type": "lokr",
        "weights_path": str(weight_path),
        "output_dir": str(metadata.get("output_dir") or ""),
        "epochs": metadata.get("epochs"),
        "created_at": metadata.get("created_at"),
        "completed_at": metadata.get("completed_at"),
        "metadata_path": metadata.get("metadata_path"),
    }


def _lokr_adapters() -> list[dict[str, Any]]:
    adapters = []
    for metadata in _lokr_runs():
        adapter = _lokr_adapter_for_response(metadata)
        if adapter is not None:
            adapters.append(adapter)
    return sorted(adapters, key=lambda item: str(item.get("completed_at") or item.get("created_at") or ""), reverse=True)


def _find_lokr_adapter(adapter_id: str | None) -> dict[str, Any] | None:
    if not adapter_id:
        return None
    return next((adapter for adapter in _lokr_adapters() if adapter.get("adapter_id") == adapter_id), None)


def _sidestep_preprocess_command(
    request: LokrPreprocessRequest,
    *,
    dataset_dir: Path,
    dataset_json: Path,
    tensor_dir: Path,
) -> list[str]:
    return [
        *_sidestep_command_prefix(request.sidestep_command),
        "preprocess",
        "--audio-dir",
        str(dataset_dir.resolve()),
        "--dataset-json",
        str(dataset_json.resolve()),
        "--output",
        str(tensor_dir.resolve()),
        "--checkpoint-dir",
        str(Path(request.checkpoint_dir).expanduser().resolve()),
        "--model",
        request.model,
    ]


def _sidestep_train_command(request: LokrTrainRequest, *, tensor_dir: Path, output_dir: Path) -> list[str]:
    command = [
        *_sidestep_command_prefix(request.sidestep_command),
        "--yes",
        "train",
        "--checkpoint-dir",
        str(Path(request.checkpoint_dir).expanduser().resolve()),
        "--model",
        request.model,
        "--dataset-dir",
        str(tensor_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--adapter-type",
        "lokr",
        "--epochs",
        str(request.epochs),
        "--lokr-linear-dim",
        str(request.lokr_linear_dim),
        "--lokr-linear-alpha",
        str(request.lokr_linear_alpha),
        "--save-every",
        str(request.save_every),
        "--optimizer-type",
        request.optimizer_type,
        "--batch-size",
        str(request.batch_size),
        "--gradient-accumulation",
        str(request.gradient_accumulation),
    ]
    command.append("--gradient-checkpointing" if request.gradient_checkpointing else "--no-gradient-checkpointing")
    command.append("--offload-encoder" if request.offload_encoder else "--no-offload-encoder")
    if request.chunk_duration:
        command.extend(["--chunk-duration", str(request.chunk_duration)])
    return command


def _sidestep_command_prefix(command: str) -> list[str]:
    value = (command or "sidestep").strip()
    if not value:
        return ["sidestep"]
    try:
        parts = shlex.split(value, posix=os.name != "nt")
    except ValueError:
        parts = [value]
    return parts or ["sidestep"]


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

    for item in _local_library().list_items():
        if not bool((item.metadata or {}).get("imported")):
            continue
        audio_file = next((file for file in item.files if file.role in {"audio", "preview", "stem"}), None)
        if audio_file is None:
            continue
        audio_path = Path(audio_file.path)
        if not audio_path.exists() or not audio_path.is_file():
            continue
        creator = (item.metadata or {}).get("creator") or {}
        creator_name = creator.get("display_name") or creator.get("creator_slug") or ""
        assets.append(
            {
                "asset_id": item.id,
                "category": item.kind,
                "label": item.title,
                "audio_path": str(audio_path),
                "audio_url": f"/api/editor/audio?path={quote(str(audio_path))}",
                "duration_seconds": audio_file.metadata.get("duration_seconds") or 0,
                "created_at": item.created_at,
                "metadata_path": str(_local_library()._manifest_path(item.id)),
                "message": f"Imported public library item{f' by {creator_name}' if creator_name else ''}",
                "source_path": audio_file.public_url or "",
                "source_asset_id": item.source_lineage.get("remote_item_id") or "",
                "imported": True,
                "creator_name": creator_name,
            }
        )

    return sorted(assets, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def _local_library() -> LocalLibraryIndex:
    return LocalLibraryIndex(Path("data/library"))


def _library_item_response(item: LibraryItem) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()


def _library_items_from_lokr_datasets() -> list[LibraryItem]:
    items: list[LibraryItem] = []
    for metadata_path in _lokr_dataset_root().glob("*/dataset.json"):
        try:
            dataset = _lokr_dataset_for_response(_read_lokr_dataset(metadata_path.parent.name))
        except Exception:
            continue
        metadata = dataset.get("metadata") or {}
        dataset_id = str(metadata.get("dataset_id") or "")
        metadata_path = Path(str(dataset.get("metadata_path") or ""))
        if not dataset_id or not metadata_path.exists():
            continue
        files = [
            LibraryFile(
                role="dataset_manifest",
                mime_type="application/json",
                size_bytes=metadata_path.stat().st_size,
                path=str(metadata_path),
            )
        ]
        for sample in dataset.get("samples", []):
            sample_path_raw = str(sample.get("resolved_audio_path") or sample.get("audio_path") or "")
            if not sample_path_raw:
                continue
            sample_path = Path(sample_path_raw)
            if not sample_path.exists() and not sample_path.is_absolute():
                sample_path = _lokr_audio_path_for_response(dataset_id, sample_path_raw)
            if not sample_path.exists() or not sample_path.is_file():
                continue
            files.append(
                LibraryFile(
                    role="dataset_sample",
                    mime_type=audio_mime_type_for_path(sample_path),
                    size_bytes=sample_path.stat().st_size,
                    path=str(sample_path),
                    metadata={
                        "sample_id": sample.get("id") or "",
                        "label": sample.get("label") or "",
                        "caption": sample.get("caption") or "",
                        "lyrics": sample.get("lyrics") or "",
                        "genre": sample.get("genre") or "",
                        "language": sample.get("language") or "",
                        "duration": sample.get("duration") or 0,
                        "is_instrumental": bool(sample.get("is_instrumental", True)),
                        "source_asset_id": sample.get("source_asset_id") or "",
                        "source_category": sample.get("source_category") or "",
                    },
                )
            )
        items.append(
            LibraryItem(
                id=dataset_id,
                visibility="local",
                status="draft",
                kind="dataset",
                title=str(metadata.get("label") or dataset_id),
                files=files,
                metadata={
                    "category": "dataset",
                    "metadata_path": str(metadata_path),
                    "sample_count": metadata.get("num_samples", 0),
                    "indexed_sample_file_count": max(0, len(files) - 1),
                    "custom_tag": metadata.get("custom_tag") or "",
                    "default_genre": metadata.get("default_genre") or "",
                    "default_language": metadata.get("default_language") or "unknown",
                    "all_instrumental": bool(metadata.get("all_instrumental", True)),
                },
                created_at=str(metadata.get("created_at") or ""),
                updated_at=str(metadata.get("updated_at") or metadata.get("created_at") or ""),
            )
        )
    return items


def _library_items_from_lokr_adapters() -> list[LibraryItem]:
    items: list[LibraryItem] = []
    for adapter in _lokr_adapters():
        adapter_id = str(adapter.get("adapter_id") or "")
        weights_path = Path(str(adapter.get("weights_path") or ""))
        if not adapter_id or not weights_path.exists():
            continue
        metadata_path = str(adapter.get("metadata_path") or "")
        files = [
            LibraryFile(
                role="adapter_weights",
                mime_type="application/octet-stream",
                size_bytes=weights_path.stat().st_size,
                path=str(weights_path),
            )
        ]
        if metadata_path and Path(metadata_path).exists():
            files.append(
                LibraryFile(
                    role="metadata",
                    mime_type="application/json",
                    size_bytes=Path(metadata_path).stat().st_size,
                    path=metadata_path,
                )
            )
        items.append(
            LibraryItem(
                id=adapter_id,
                visibility="local",
                status="draft",
                kind="lokr",
                title=str(adapter.get("label") or adapter_id),
                files=files,
                metadata={
                    "category": "lokr",
                    "adapter_type": adapter.get("adapter_type") or "lokr",
                    "model": adapter.get("model") or "",
                    "training_model": adapter.get("training_model") or "",
                    "dataset_id": adapter.get("dataset_id") or "",
                    "epochs": adapter.get("epochs"),
                    "metadata_path": metadata_path,
                    "output_dir": adapter.get("output_dir") or "",
                },
                source_lineage={"dataset_id": adapter.get("dataset_id") or ""},
                created_at=str(adapter.get("created_at") or ""),
                updated_at=str(adapter.get("completed_at") or adapter.get("created_at") or ""),
            )
        )
    return items


def _local_library_scanned_items() -> list[LibraryItem]:
    items = [item for asset in _editor_assets() if (item := library_item_from_editor_asset(asset)) is not None]
    items.extend(_library_items_from_lokr_datasets())
    items.extend(_library_items_from_lokr_adapters())
    return items


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
        status["side_step"] = side_step_status(runtime_config).to_dict()
        status["side_step_command"] = build_side_step_command(runtime_config)
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

    @app.get("/api/library/local")
    def list_local_library() -> dict[str, object]:
        library = _local_library()
        items = [_library_item_response(item) for item in library.list_items()]
        return {"items": items, "count": len(items), "index_path": str(library.index_path)}

    @app.post("/api/library/local/reindex")
    def reindex_local_library() -> dict[str, object]:
        library = _local_library()
        items = [_library_item_response(item) for item in library.reindex_items(_local_library_scanned_items())]
        ui_log.add("info", f"Reindexed local library: {len(items)} items")
        return {"items": items, "count": len(items), "index_path": str(library.index_path)}

    @app.patch("/api/library/local/{item_id}")
    def update_local_library_item(item_id: str, request: LocalLibraryUpdateRequest) -> dict[str, object]:
        library = _local_library()
        try:
            item = library.update_item(item_id, request.model_dump(exclude_unset=True))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        ui_log.add("info", f"Updated local library item: {item.title}")
        return {"item": _library_item_response(item)}

    @app.get("/api/library/publish/connection")
    def get_public_library_connection() -> dict[str, Any]:
        return public_settings_response(load_publish_settings())

    @app.post("/api/library/publish/connection")
    def save_public_library_connection(request: PublicLibraryConnectionRequest) -> dict[str, Any]:
        existing = load_publish_settings()
        token = request.token if request.token is not None and request.token.strip() else existing.token
        settings = LibraryPublishSettings(site_url=request.site_url.strip().rstrip("/"), token=token)
        save_publish_settings(settings)
        ui_log.add("info", f"Saved public library connection for {settings.site_url}.")
        return public_settings_response(settings)

    @app.post("/api/library/local/{item_id}/publish")
    def publish_local_library_item(item_id: str, request: PublicLibraryPublishRequest) -> dict[str, Any]:
        library = _local_library()
        item = library.read_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Library item not found: {item_id}")

        try:
            publish_result = LibraryPublisher(load_publish_settings()).publish(
                item,
                publish_public=request.publish_public,
            )
            updated = library.update_publish_metadata(item_id, publish_result)
        except (FileNotFoundError, LibraryPublishError) as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        ui_log.add("info", f"Published local library item '{updated.title}' to the public library.")
        return {"item": _library_item_response(updated), "publish": publish_result}

    @app.get("/api/library/public")
    def list_public_library(kind: str = Query("all", max_length=80)) -> dict[str, Any]:
        try:
            items = PublicLibraryClient(load_publish_settings()).list_items(kind=kind, limit=80)
        except LibraryPublishError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"items": items, "count": len(items)}

    @app.post("/api/library/public/{item_id}/import")
    def import_public_library_item(item_id: str) -> dict[str, Any]:
        try:
            imported = PublicLibraryClient(load_publish_settings()).import_item(item_id)
            item = _local_library().write_item(imported)
        except LibraryPublishError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ui_log.add("info", f"Imported public library item: {item.title}")
        return {"item": _library_item_response(item)}

    @app.get("/api/lokr/datasets")
    def list_lokr_datasets() -> list[dict[str, Any]]:
        root = _lokr_dataset_root()
        if not root.exists():
            return []
        datasets: list[dict[str, Any]] = []
        for metadata_path in root.glob("*/dataset.json"):
            try:
                dataset = json.loads(metadata_path.read_text(encoding="utf-8"))
                clean = _lokr_clean_dataset(dataset, dataset_id=metadata_path.parent.name)
            except Exception:
                continue
            datasets.append(_lokr_dataset_for_response(clean))
        return sorted(datasets, key=lambda item: str(item.get("metadata", {}).get("updated_at") or ""), reverse=True)

    @app.get("/api/lokr/datasets/{dataset_id}")
    def get_lokr_dataset(dataset_id: str) -> dict[str, Any]:
        return _lokr_dataset_for_response(_read_lokr_dataset(dataset_id))

    @app.post("/api/lokr/datasets")
    def create_lokr_dataset(request: LokrDatasetCreateRequest) -> dict[str, Any]:
        import datetime as _datetime

        dataset_id = f"lokr-{uuid4().hex[:12]}"
        now = _datetime.datetime.now(_datetime.UTC).isoformat()
        label = request.label.strip()
        dataset = {
            "metadata": {
                "dataset_id": dataset_id,
                "label": label,
                "name": label,
                "custom_tag": (request.custom_tag or "").strip(),
                "default_genre": (request.default_genre or "").strip(),
                "default_language": (request.default_language or "unknown").strip() or "unknown",
                "tag_position": request.tag_position,
                "genre_ratio": request.genre_ratio,
                "all_instrumental": request.all_instrumental,
                "created_at": now,
                "updated_at": now,
                "num_samples": 0,
            },
            "samples": [],
        }
        saved = _write_lokr_dataset(dataset, dataset_id=dataset_id)
        ui_log.add("info", f"Created LoKr dataset: {label}")
        return {"dataset": _lokr_dataset_for_response(saved)}

    @app.post("/api/lokr/datasets/{dataset_id}")
    def save_lokr_dataset(dataset_id: str, request: LokrDatasetSaveRequest) -> dict[str, Any]:
        incoming = request.dataset
        incoming_metadata = dict(incoming.get("metadata") or {})
        incoming_metadata["dataset_id"] = dataset_id
        incoming["metadata"] = incoming_metadata
        saved = _write_lokr_dataset(incoming, dataset_id=dataset_id)
        ui_log.add("info", f"Saved LoKr dataset: {saved['metadata']['label']}")
        return {"dataset": _lokr_dataset_for_response(saved)}

    @app.post("/api/lokr/datasets/{dataset_id}/entries/upload")
    def upload_lokr_dataset_entry(dataset_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        original_name = Path(file.filename or "sample.wav").name
        upload_dir = _lokr_dataset_audio_dir(dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / f"upload-{uuid4().hex[:8]}{Path(original_name).suffix.lower()}"
        try:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            metadata = dataset.get("metadata", {})
            sample = _lokr_sample_from_audio(
                dataset_id=dataset_id,
                source_path=temp_path,
                label=Path(original_name).stem,
                default_genre=str(metadata.get("default_genre") or ""),
                default_language=str(metadata.get("default_language") or "unknown"),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
        dataset["samples"].append(sample)
        saved = _write_lokr_dataset(dataset, dataset_id=dataset_id)
        ui_log.add("info", f"Added LoKr dataset entry from upload: {original_name}")
        return {"dataset": _lokr_dataset_for_response(saved), "sample": sample}

    @app.post("/api/lokr/datasets/{dataset_id}/entries/from-asset")
    def add_lokr_dataset_entry_from_asset(dataset_id: str, request: LokrDatasetAssetRequest) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        asset = next((item for item in _editor_assets() if item.get("asset_id") == request.asset_id), None)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Creation not found: {request.asset_id}")
        audio_path = Path(str(asset.get("audio_path") or "")).expanduser()
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"Creation audio not found: {audio_path}")
        try:
            sample = _lokr_sample_from_audio(
                dataset_id=dataset_id,
                source_path=audio_path,
                label=str(asset.get("label") or audio_path.stem),
                default_genre=str(dataset.get("metadata", {}).get("default_genre") or ""),
                default_language=str(dataset.get("metadata", {}).get("default_language") or "unknown"),
                source_asset_id=str(asset.get("asset_id") or ""),
                source_category=str(asset.get("category") or ""),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        dataset["samples"].append(sample)
        saved = _write_lokr_dataset(dataset, dataset_id=dataset_id)
        ui_log.add("info", f"Added LoKr dataset entry from creation: {asset.get('label')}")
        return {"dataset": _lokr_dataset_for_response(saved), "sample": sample}

    @app.delete("/api/lokr/datasets/{dataset_id}/entries/{entry_id}")
    def delete_lokr_dataset_entry(dataset_id: str, entry_id: str) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        before = len(dataset.get("samples", []))
        dataset["samples"] = [sample for sample in dataset.get("samples", []) if sample.get("id") != entry_id]
        if len(dataset["samples"]) == before:
            raise HTTPException(status_code=404, detail=f"Dataset entry not found: {entry_id}")
        saved = _write_lokr_dataset(dataset, dataset_id=dataset_id)
        ui_log.add("info", f"Deleted LoKr dataset entry: {entry_id}")
        return {"dataset": _lokr_dataset_for_response(saved)}

    @app.get("/api/lokr/audio")
    def get_lokr_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        audio_path = Path(path).expanduser()
        try:
            audio_path.resolve().relative_to(_lokr_root().resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="LoKr audio path is outside the dataset root.") from exc
        return get_audio_file(str(audio_path))

    @app.get("/api/lokr/adapters")
    def list_lokr_adapters() -> list[dict[str, Any]]:
        return _lokr_adapters()

    @app.get("/api/lokr/runs")
    def list_lokr_runs() -> list[dict[str, Any]]:
        return _lokr_runs()

    @app.get("/api/lokr/runs/{run_id}")
    def get_lokr_run(run_id: str) -> dict[str, Any]:
        return _refresh_lokr_run(_read_json_file(_lokr_run_path(run_id), "LoKr run"))

    @app.get("/api/lokr/runs/{run_id}/logs")
    def get_lokr_run_logs(run_id: str) -> dict[str, str]:
        metadata = _refresh_lokr_run(_read_json_file(_lokr_run_path(run_id), "LoKr run"))
        log_paths = []
        if metadata.get("type") == "train":
            session_log = _lokr_train_session_log_path(metadata)
            if session_log.exists():
                log_paths.append(("Side-Step session log", session_log))
        primary_log = Path(str(metadata.get("log_path") or ""))
        if primary_log.exists() and all(path != primary_log for _, path in log_paths):
            log_paths.append(("Process log", primary_log))
        if not log_paths:
            return {"text": ""}
        chunks = []
        for label, path in log_paths:
            chunks.append(f"--- {label}: {path} ---\n{path.read_text(encoding='utf-8', errors='replace')[-20000:]}")
        return {"text": "\n\n".join(chunks)[-30000:]}

    @app.post("/api/lokr/runs/{run_id}/stop")
    def stop_lokr_run(run_id: str) -> dict[str, Any]:
        stopped = _stop_lokr_process(run_id)
        ui_log.add("info", f"Stopped Side-Step run: {run_id}")
        return {"run": stopped}

    @app.post("/api/lokr/datasets/{dataset_id}/preprocess")
    def preprocess_lokr_dataset(dataset_id: str, request: LokrPreprocessRequest) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        if not dataset.get("samples"):
            raise HTTPException(status_code=400, detail="Dataset has no samples.")
        if not side_step_status(runtime_config).installed:
            raise HTTPException(status_code=400, detail="Side-Step runtime is not installed. Run `autotransition runtime setup` or `autotransition runtime setup-sidestep`.")
        active_run = _active_lokr_run()
        if active_run is not None:
            raise HTTPException(status_code=400, detail=f"Side-Step is already running: {active_run.get('label') or active_run.get('run_id')}")
        run_id = f"preprocess-{uuid4().hex[:12]}"
        run_dir = _lokr_run_root() / run_id
        tensor_dir = run_dir / "tensors"
        dataset_json = _lokr_dataset_path(dataset_id)
        dataset_dir = dataset_json.parent
        command = _sidestep_preprocess_command(request, dataset_dir=dataset_dir, dataset_json=dataset_json, tensor_dir=tensor_dir)
        metadata = _lokr_run_metadata(
            run_id=run_id,
            run_type="preprocess",
            dataset_id=dataset_id,
            label=f"Preprocess {dataset['metadata']['label']}",
            command=command,
            log_path=run_dir / "sidestep-preprocess.log",
            model=request.model,
            tensor_dir=tensor_dir,
            cwd=runtime_config.side_step_dir,
            extra={"dataset_json": str(dataset_json), "dataset_dir": str(dataset_dir)},
        )
        started = _start_lokr_process(metadata)
        ui_log.add("info", f"Started Side-Step preprocess run: {run_id}")
        return {"run": started}

    @app.post("/api/lokr/datasets/{dataset_id}/train")
    def train_lokr_dataset(dataset_id: str, request: LokrTrainRequest) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        if not side_step_status(runtime_config).installed:
            raise HTTPException(status_code=400, detail="Side-Step runtime is not installed. Run `autotransition runtime setup` or `autotransition runtime setup-sidestep`.")
        active_run = _active_lokr_run()
        if active_run is not None:
            raise HTTPException(status_code=400, detail=f"Side-Step is already running: {active_run.get('label') or active_run.get('run_id')}")
        tensor_dir_value = request.tensor_dir or _lokr_latest_tensor_dir(dataset_id)
        if not tensor_dir_value:
            raise HTTPException(status_code=400, detail="No preprocessed tensor dataset found. Run preprocess first.")
        tensor_dir = Path(tensor_dir_value).expanduser()
        if not tensor_dir.exists():
            raise HTTPException(status_code=400, detail="No preprocessed tensor dataset found. Run preprocess first.")
        run_id = f"train-{uuid4().hex[:12]}"
        run_dir = _lokr_run_root() / run_id
        output_dir = run_dir / "adapter"
        command = _sidestep_train_command(request, tensor_dir=tensor_dir, output_dir=output_dir)
        metadata = _lokr_run_metadata(
            run_id=run_id,
            run_type="train",
            dataset_id=dataset_id,
            label=f"Train LoKr {dataset['metadata']['label']}",
            command=command,
            log_path=run_dir / "sidestep-train.log",
            model=request.model,
            tensor_dir=tensor_dir,
            output_dir=output_dir,
            cwd=runtime_config.side_step_dir,
            extra={
                "adapter_type": "lokr",
                "epochs": request.epochs,
                "lokr_linear_dim": request.lokr_linear_dim,
                "lokr_linear_alpha": request.lokr_linear_alpha,
                "save_every": request.save_every,
                "optimizer_type": request.optimizer_type,
                "batch_size": request.batch_size,
                "gradient_accumulation": request.gradient_accumulation,
                "gradient_checkpointing": request.gradient_checkpointing,
                "offload_encoder": request.offload_encoder,
                "chunk_duration": request.chunk_duration,
            },
        )
        started = _start_lokr_process(metadata)
        ui_log.add("info", f"Started Side-Step LoKr training run: {run_id}")
        return {"run": started}

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
        lyrics = "[Instrumental]" if request.instrumental else (request.lyrics or "").strip() or "[Instrumental]"
        model = _music_generation_model(request.model)
        vocal_language = (request.vocal_language or "unknown").strip() or "unknown"
        label = request.label.strip() if request.label else prompt[:80]
        lokr_adapter = _find_lokr_adapter(request.lokr_adapter_id)
        if request.lokr_adapter_id and lokr_adapter is None:
            raise HTTPException(status_code=404, detail=f"LoKr adapter not found: {request.lokr_adapter_id}")
        if lokr_adapter and lokr_adapter.get("model") != model:
            raise HTTPException(
                status_code=400,
                detail=f"Selected LoKr was trained for {lokr_adapter.get('model')}; choose that model before generating.",
            )
        lokr_path = str(lokr_adapter.get("weights_path") or "") if lokr_adapter else None
        lokr_adapter_name = f"dance_station_{lokr_adapter.get('adapter_id')}" if lokr_adapter else None
        lokr_label = str(lokr_adapter.get("label") or "") if lokr_adapter else ""
        if lokr_adapter:
            ui_log.add("info", f"Running ACE-Step {model} text-to-music generation with LoKr: {lokr_label}.")
        else:
            ui_log.add("info", f"Running ACE-Step {model} text-to-music generation.")
        try:
            result = AceStepApiClient(runtime_config).text2music_standalone(
                prompt=prompt,
                model=model,
                save_dir=save_dir,
                lyrics=lyrics,
                vocal_language=vocal_language,
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
                lokr_path=lokr_path,
                lokr_scale=request.lokr_scale,
                lokr_adapter_name=lokr_adapter_name,
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
                "model": model,
                "output_format": request.output_format,
                "lokr_adapter": lokr_adapter,
                "lokr_scale": request.lokr_scale if lokr_adapter else None,
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
            "model": model,
            "output_format": request.output_format,
            "lokr_adapter": lokr_adapter,
            "lokr_scale": request.lokr_scale if lokr_adapter else None,
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

    @app.delete("/api/logs")
    def clear_logs() -> list[dict[str, str]]:
        ui_log.clear()
        return ui_log.entries()

    return app

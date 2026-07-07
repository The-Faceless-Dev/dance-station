"""FastAPI app for the local Autotransition UI."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tempfile
from dataclasses import replace
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
from autotransition.audio.ffmpeg import resolve_ffmpeg
from autotransition.audio.formats import DEFAULT_SCAFFOLD_FORMAT, SUPPORTED_INPUT_FORMATS, validate_supported_source
from autotransition.config import OutputConfig, RuntimeConfig, TransitionConfig
from autotransition.generation import GenerationResult, GenerationStatus
from autotransition.library.index import LocalLibraryIndex
from autotransition.library.publish import (
    DEFAULT_PUBLIC_LIBRARY_SITE_URL,
    LibraryPublishError,
    LibraryPublisher,
    LibraryPublishSettings,
    PublicLibraryClient,
    authenticate_wallet_signature,
    is_expired_session_error,
    load_publish_settings,
    logout_site_session,
    public_settings_response,
    refresh_site_session,
    request_wallet_nonce,
    save_publish_settings,
)
from autotransition.library.schema import (
    LibraryFile,
    LibraryItem,
    audio_mime_type_for_path,
    library_item_from_editor_asset,
    utc_now_iso,
)
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
    ACE_STEP_BASE_MODEL,
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
from autotransition.runtime.seed_vc import (
    api_health as rvc_api_health,
    ensure_runtime_api as ensure_rvc_runtime_api,
    managed_runtime_alive as rvc_managed_runtime_alive,
    read_runtime_pid as read_rvc_runtime_pid,
    run_install as run_rvc_install,
    startup_progress_snapshot as rvc_startup_progress_snapshot,
    stop_runtime as stop_rvc_runtime,
    runtime_status as rvc_runtime_status,
)
from autotransition.runtime.side_step import build_side_step_command, side_step_status
from autotransition.runtime.tango_flux import (
    generate_wav as generate_sound_effect_wav,
    run_install as run_sound_effect_runtime_install,
    runtime_status as sound_effect_runtime_status,
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
from autotransition.rhythm_beats import (
    copy_uploaded_source_audio,
    create_project as create_rhythm_project,
    library_item_from_rhythm_project,
    list_volumes as list_rhythm_volumes,
    list_projects as list_rhythm_projects,
    read_project as read_rhythm_project,
    remove_volume as remove_rhythm_volume,
    safe_project_id as safe_rhythm_project_id,
    upsert_volume as upsert_rhythm_volume,
    write_project as write_rhythm_project,
)
from autotransition.sound_effects import (
    generation_path as sound_effect_generation_path,
    library_item_from_generation as library_item_from_sound_effect_generation,
    list_generations as list_sound_effect_generations,
    write_generation as write_sound_effect_generation,
)
from autotransition.ui.activity import summarize_runtime_activity
from autotransition.ui.state import UiLog, system_status
from autotransition.voice_work import (
    VOICE_AUDIO_EXTENSIONS,
    VOICE_EMBEDDING_EXTENSIONS,
    create_voice_asset,
    create_generation_record,
    create_voice_render_record,
    delete_voice,
    library_item_from_voice,
    library_item_from_generation,
    list_generations,
    list_voices,
    read_voice,
    update_voice,
    voice_runtime_index_path,
    voice_runtime_model_name,
)


_extract_retry_lock = threading.Lock()
_extract_retry_threads: dict[str, threading.Thread] = {}


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


class RhythmBeatProjectCreateRequest(BaseModel):
    label: str = Field("New rhythm beat project", min_length=1, max_length=160)


class RhythmBeatProjectSaveRequest(BaseModel):
    project: dict[str, Any]


class RhythmBeatProjectAssetRequest(BaseModel):
    asset_id: str = Field(..., min_length=1)


class RhythmBeatExtractionRequest(BaseModel):
    track_name: str = "vocals"
    label: str | None = None
    attach_to_project: bool = True
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


class RhythmBeatLyricsExtractionRequest(BaseModel):
    model: str = Field("small", min_length=1, max_length=80)
    language: str | None = Field(None, max_length=32)
    disable_word_timestamps: bool = False


class RhythmBeatVolumeUpsertRequest(BaseModel):
    volume_id: str | None = Field(None, max_length=80)
    label: str = Field(..., min_length=1, max_length=160)
    description: str = Field("", max_length=500)
    sort_order: int = Field(0, ge=0, le=9999)


class RhythmBeatGameAssetSettingsRequest(BaseModel):
    game_enabled: bool
    volume_id: str | None = Field(None, max_length=80)
    step_arrows_enabled: bool = True
    orb_beat_enabled: bool = False


class PublicLibraryConnectionRequest(BaseModel):
    site_url: str = Field(DEFAULT_PUBLIC_LIBRARY_SITE_URL, min_length=1, max_length=500)


class PublicLibraryAuthNonceRequest(BaseModel):
    public_key: str = Field(..., min_length=20, max_length=120)


class PublicLibraryAuthVerifyRequest(BaseModel):
    public_key: str = Field(..., min_length=20, max_length=120)
    nonce: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=2000)
    signature: list[int] = Field(..., min_length=1)


class PublicLibraryPublishRequest(BaseModel):
    publish_public: bool = True


class VoiceWorkTrainRequest(BaseModel):
    epochs: int = Field(20, ge=1, le=1000)
    save_every: int = Field(5, ge=1, le=100)
    batch_size: int = Field(1, ge=1, le=16)
    use_f0: bool = True
    sample_rate: Literal["32k", "40k", "48k"] = "48k"
    version: Literal["v1", "v2"] = "v2"
    cpu_threads: int | None = Field(None, ge=1, le=64)


class VoiceWorkConvertRequest(BaseModel):
    request_id: str | None = Field(None, max_length=80)
    voice_id: str = Field(..., min_length=1)
    source_audio_path: str = Field(..., min_length=1)
    label: str | None = Field(None, max_length=160)
    mode: Literal["speaking", "singing"] = "singing"
    diffusion_steps: int = Field(25, ge=1, le=200)
    length_adjust: float = Field(1.0, ge=0.5, le=2.0)
    inference_cfg_rate: float = Field(0.7, ge=0.0, le=1.0)


class VoiceWorkTtsRequest(BaseModel):
    voice_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=10000)
    label: str | None = Field(None, max_length=160)
    language: str = Field("auto", max_length=32)


class VoiceWorkTargetVoiceFromAssetRequest(BaseModel):
    asset_id: str = Field(..., min_length=1)
    label: str | None = Field(None, max_length=160)
    description: str = Field("", max_length=2000)
    language: str = Field("auto", max_length=32)


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


class Vocal2BgmRequest(BaseModel):
    source_audio_path: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1, max_length=120)
    prompt: str | None = Field(None, max_length=240)
    output_format: Literal["flac", "wav", "wav32", "mp3", "opus", "aac"] = "flac"
    audio_duration: float | None = Field(None, ge=1.0, le=600.0)
    inference_steps: int = Field(BASE_RUNTIME_INFERENCE_STEPS, ge=1, le=200)
    guidance_scale: float = Field(BASE_RUNTIME_GUIDANCE_SCALE, ge=0)
    shift: float = Field(BASE_RUNTIME_SHIFT, ge=0)
    infer_method: Literal["ode", "sde"] = BASE_RUNTIME_INFER_METHOD
    use_tiled_decode: bool = True
    dcw_enabled: bool = False
    velocity_norm_threshold: float = Field(BASE_RUNTIME_VELOCITY_NORM_THRESHOLD, ge=0)
    velocity_ema_factor: float = Field(BASE_RUNTIME_VELOCITY_EMA_FACTOR, ge=0, le=1)
    seed: int | None = None
    audio_cover_strength: float = Field(1.0, ge=0.0, le=1.0)


class SoundEffectRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)
    prompt: str = Field(..., min_length=1, max_length=240)
    duration_seconds: float = Field(10.0, ge=1.0, le=30.0)
    steps: int = Field(50, ge=1, le=200)
    output_format: Literal["wav", "wav32", "flac", "mp3", "opus", "aac"] = "wav"


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


class LokrDatasetEntryAssetRequest(BaseModel):
    entry_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)


class LokrDatasetImportRequest(BaseModel):
    label: str | None = Field(None, max_length=120)


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


def _copy_upload_to_temp(file: UploadFile, temp_root: Path) -> Path:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    safe_stem = _safe_label_stem(Path(filename).stem, "upload")
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_path = temp_root / f"{safe_stem}-{uuid4().hex[:8]}{suffix}"
    with temp_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return temp_path


def _voice_upload_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VOICE_AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VOICE_EMBEDDING_EXTENSIONS:
        return "embedding"
    return "unknown"


def _save_float32_pcm_to_wav(pcm_path: Path, wav_path: Path, *, sample_rate: int = 44100, channels: int = 1) -> None:
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert PCM output into WAV.")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            str(pcm_path),
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _update_extraction_metadata(
    extraction_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        metadata = _read_extraction_metadata(extraction_id)
    except HTTPException:
        return None
    metadata.update(updates)
    return _write_extraction_metadata(metadata)


def _music_generation_root() -> Path:
    return Path("data/generations")


def _sound_effect_root() -> Path:
    return Path("data/sound-effects")


def _sound_effect_output_extension(output_format: str) -> str:
    format_name = (output_format or "wav").strip().lower()
    if format_name == "wav32":
        return ".wav"
    return f".{format_name}" if format_name else ".wav"


def _sound_effect_transcode_output(wav_path: Path, output_path: Path, output_format: str) -> Path:
    output_format = (output_format or "wav").strip().lower()
    if output_format == "wav":
        if wav_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.replace(output_path)
        return output_path

    ffmpeg = resolve_ffmpeg()
    if output_format == "wav32":
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to export 32-bit WAV output.")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(wav_path),
                "-c:a",
                "pcm_s32le",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav_path.unlink(missing_ok=True)
        return output_path

    from pydub import AudioSegment

    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to export TangoFlux output.")
    segment = AudioSegment.from_wav(wav_path)
    segment.export(output_path, format=output_format)
    wav_path.unlink(missing_ok=True)
    return output_path


def _transition_root() -> Path:
    return Path("data/generated")


def _edit_root() -> Path:
    return Path("data/edits")


def _instrument_lab_root() -> Path:
    return Path("data/instrument-lab")


def _lokr_root() -> Path:
    return Path("data/lokr-training")


def _voice_work_root() -> Path:
    return Path("data/seed-vc")


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


def _lokr_empty_sample(
    dataset_id: str,
    *,
    label: str = "",
    default_genre: str = "",
    default_language: str = "unknown",
) -> dict[str, Any]:
    return _lokr_clean_sample(
        dataset_id,
        {
            "id": f"sample-{uuid4().hex[:10]}",
            "audio_path": "",
            "filename": "",
            "label": label or "Untitled entry",
            "caption": "",
            "genre": default_genre,
            "lyrics": "[Instrumental]",
            "duration": 0,
            "language": default_language,
            "is_instrumental": True,
        },
    )


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
    missing_audio_entries: list[str] = []
    for sample in response.get("samples", []):
        audio_path = str(sample.get("audio_path") or "")
        resolved = _lokr_audio_path_for_response(dataset_id, audio_path) if dataset_id and audio_path else Path()
        resolved_path = str(resolved) if str(resolved) != "." else ""
        has_audio = bool(resolved_path and resolved.exists() and resolved.is_file())
        sample["resolved_audio_path"] = resolved_path
        sample["audio_url"] = f"/api/lokr/audio?path={quote(str(resolved))}" if has_audio else ""
        sample["has_audio"] = has_audio
        if not has_audio:
            missing_audio_entries.append(str(sample.get("id") or ""))
    response["validation"] = {
        "missing_audio_entries": [entry_id for entry_id in missing_audio_entries if entry_id],
        "missing_audio_count": len([entry_id for entry_id in missing_audio_entries if entry_id]),
    }
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
    try:
        _sync_local_library_index()
    except Exception:
        pass
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


def _lokr_attach_audio_to_entry(
    dataset: dict[str, Any],
    *,
    dataset_id: str,
    entry_id: str,
    source_path: Path,
    label: str,
    source_asset_id: str = "",
    source_category: str = "",
) -> dict[str, Any]:
    samples = list(dataset.get("samples") or [])
    index = next((i for i, sample in enumerate(samples) if str(sample.get("id") or "") == entry_id), -1)
    if index < 0:
        raise FileNotFoundError(f"Dataset entry not found: {entry_id}")
    original = dict(samples[index])
    metadata = dataset.get("metadata", {})
    sample = _lokr_sample_from_audio(
        dataset_id=dataset_id,
        source_path=source_path,
        label=label or str(original.get("label") or source_path.stem),
        default_genre=str(original.get("genre") or metadata.get("default_genre") or ""),
        default_language=str(original.get("language") or metadata.get("default_language") or "unknown"),
        source_asset_id=source_asset_id or str(original.get("source_asset_id") or ""),
        source_category=source_category or str(original.get("source_category") or ""),
    )
    merged = {
        **original,
        **sample,
        "id": entry_id,
        "label": str(original.get("label") or label or sample.get("label") or "Untitled entry"),
        "caption": str(original.get("caption") or ""),
        "lyrics": str(original.get("lyrics") or "[Instrumental]"),
        "formatted_lyrics": str(original.get("formatted_lyrics") or original.get("lyrics") or "[Instrumental]"),
        "raw_lyrics": str(original.get("raw_lyrics") or ""),
        "genre": str(original.get("genre") or sample.get("genre") or ""),
        "language": str(original.get("language") or sample.get("language") or "unknown"),
        "custom_tag": str(original.get("custom_tag") or ""),
        "prompt_override": original.get("prompt_override") or None,
        "is_instrumental": bool(original.get("is_instrumental", True)),
        "labeled": bool(original.get("labeled", bool(original.get("caption")))),
        "source_asset_id": source_asset_id or str(original.get("source_asset_id") or ""),
        "source_category": source_category or str(original.get("source_category") or ""),
    }
    samples[index] = _lokr_clean_sample(
        dataset_id,
        merged,
        str(metadata.get("custom_tag") or ""),
        str(metadata.get("default_genre") or ""),
        str(metadata.get("default_language") or "unknown"),
    )
    dataset["samples"] = samples
    return samples[index]


def _lokr_missing_audio_entries(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_id = str((dataset.get("metadata") or {}).get("dataset_id") or "")
    missing: list[dict[str, Any]] = []
    for sample in list(dataset.get("samples") or []):
        audio_path = str(sample.get("audio_path") or "")
        if not audio_path:
            missing.append(sample)
            continue
        resolved = _lokr_audio_path_for_response(dataset_id, audio_path) if dataset_id else Path()
        if not resolved.exists() or not resolved.is_file():
            missing.append(sample)
    return missing


def _coerce_import_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_import_audio_source(raw_value: str) -> Path | None:
    location = str(raw_value or "").strip()
    if not location:
        return None
    candidate = Path(location).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate
    try:
        resolved = (Path.cwd() / location).expanduser().resolve()
    except Exception:
        return None
    if resolved.exists() and resolved.is_file():
        return resolved
    return None


def _extract_import_samples(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(payload, list):
        return {}, [_coerce_import_dict(item) for item in payload]
    root = _coerce_import_dict(payload)
    for key in ("samples", "entries", "items", "data"):
        value = root.get(key)
        if isinstance(value, list):
            return root, [_coerce_import_dict(item) for item in value]
    if any(key in root for key in ("caption", "name", "label", "title", "lyrics", "audio_path", "audio", "audio_location", "audioLocation")):
        return {}, [root]
    return root, []


def _import_entry_label(entry: dict[str, Any], index: int) -> str:
    return (
        str(entry.get("label") or "").strip()
        or str(entry.get("name") or "").strip()
        or str(entry.get("title") or "").strip()
        or f"Imported entry {index + 1}"
    )


def _normalize_imported_dataset_entry(
    dataset_id: str,
    entry: dict[str, Any],
    *,
    index: int,
    default_genre: str,
    default_language: str,
    fallback_tag: str,
) -> dict[str, Any]:
    label = _import_entry_label(entry, index)
    audio_location = (
        str(entry.get("audio_path") or "").strip()
        or str(entry.get("audio") or "").strip()
        or str(entry.get("audio_location") or "").strip()
        or str(entry.get("audioLocation") or "").strip()
        or str(entry.get("path") or "").strip()
    )
    resolved_audio = _resolve_import_audio_source(audio_location)
    imported = _lokr_empty_sample(
        dataset_id,
        label=label,
        default_genre=str(entry.get("genre") or default_genre or "").strip(),
        default_language=str(entry.get("language") or default_language or "unknown").strip() or "unknown",
    )
    imported.update(
        {
            "id": str(entry.get("id") or imported["id"]),
            "label": label,
            "caption": str(entry.get("caption") or "").strip(),
            "genre": str(entry.get("genre") or default_genre or "").strip(),
            "lyrics": str(entry.get("lyrics") or "[Instrumental]").strip() or "[Instrumental]",
            "raw_lyrics": str(entry.get("raw_lyrics") or "").strip(),
            "formatted_lyrics": str(entry.get("formatted_lyrics") or entry.get("lyrics") or "[Instrumental]").strip() or "[Instrumental]",
            "bpm": entry.get("bpm") if entry.get("bpm") not in ("", None) else "N/A",
            "keyscale": str(entry.get("keyscale") or entry.get("key") or "N/A"),
            "timesignature": str(entry.get("timesignature") or entry.get("time_signature") or "4"),
            "language": str(entry.get("language") or default_language or "unknown").strip() or "unknown",
            "is_instrumental": bool(entry.get("is_instrumental", not str(entry.get("lyrics") or "").strip() or str(entry.get("lyrics") or "").strip() == "[Instrumental]")),
            "custom_tag": str(entry.get("custom_tag") or fallback_tag or "").strip(),
            "prompt_override": entry.get("prompt_override") or None,
            "labeled": bool(entry.get("labeled", bool(str(entry.get("caption") or "").strip()))),
            "source_audio_location": audio_location,
        }
    )
    known_keys = {
        "id","label","name","title","caption","genre","lyrics","raw_lyrics","formatted_lyrics","bpm","keyscale","key",
        "timesignature","time_signature","language","is_instrumental","custom_tag","prompt_override","labeled",
        "audio_path","audio","audio_location","audioLocation","path",
    }
    extras = {key: value for key, value in entry.items() if key not in known_keys}
    if extras:
        imported["extra_metadata"] = extras
    imported = _lokr_clean_sample(dataset_id, imported, fallback_tag, default_genre, default_language)
    if resolved_audio is None:
        imported["audio_path"] = ""
        imported["filename"] = str(Path(audio_location).name) if audio_location else imported.get("filename", "")
        imported["duration"] = float(entry.get("duration") or imported.get("duration") or 0)
        return imported
    copied = _lokr_sample_from_audio(
        dataset_id=dataset_id,
        source_path=resolved_audio,
        label=label,
        default_genre=str(imported.get("genre") or default_genre or ""),
        default_language=str(imported.get("language") or default_language or "unknown"),
    )
    merged = {**imported, **copied, "id": imported["id"], "caption": imported["caption"], "lyrics": imported["lyrics"], "raw_lyrics": imported["raw_lyrics"], "formatted_lyrics": imported["formatted_lyrics"], "custom_tag": imported["custom_tag"], "prompt_override": imported["prompt_override"], "labeled": imported["labeled"]}
    if extras:
        merged["extra_metadata"] = extras
    merged["source_audio_location"] = audio_location
    return _lokr_clean_sample(dataset_id, merged, fallback_tag, default_genre, default_language)


def _dataset_from_import_payload(
    *,
    dataset_id: str,
    label: str,
    payload: Any,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = existing or {"metadata": {}, "samples": []}
    root, raw_entries = _extract_import_samples(payload)
    metadata = dict(existing.get("metadata") or {})
    root_meta = _coerce_import_dict(root.get("metadata"))
    resolved_label = label.strip() or str(root_meta.get("label") or root_meta.get("name") or root.get("label") or root.get("name") or metadata.get("label") or "Imported LoKr dataset").strip()
    default_genre = str(root_meta.get("default_genre") or root.get("default_genre") or metadata.get("default_genre") or "").strip()
    default_language = str(root_meta.get("default_language") or root.get("default_language") or metadata.get("default_language") or "unknown").strip() or "unknown"
    custom_tag = str(root_meta.get("custom_tag") or root.get("custom_tag") or metadata.get("custom_tag") or "").strip()
    merged_dataset = {
        "metadata": {
            **metadata,
            **root_meta,
            "dataset_id": dataset_id,
            "label": resolved_label,
            "name": str(root_meta.get("name") or root.get("name") or resolved_label),
            "default_genre": default_genre,
            "default_language": default_language,
            "custom_tag": custom_tag,
        },
        "samples": list(existing.get("samples") or []),
    }
    for index, entry in enumerate(raw_entries):
        merged_dataset["samples"].append(
            _normalize_imported_dataset_entry(
                dataset_id,
                entry,
                index=index,
                default_genre=default_genre,
                default_language=default_language,
                fallback_tag=custom_tag,
            )
        )
    return merged_dataset


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
_VOICE_RUNTIME_ACTION_LOCK = threading.Lock()
_VOICE_RUNTIME_ACTION_STATE: dict[str, Any] = {
    "active": False,
    "action": "idle",
    "message": "No Seed-VC runtime action is in progress.",
    "error": "",
    "phase": "idle",
    "started_at": None,
    "completed_at": None,
}
_VOICE_WORK_JOB_LOCK = threading.Lock()
_VOICE_WORK_JOB_STATE: dict[str, Any] = {
    "active": False,
    "action": "idle",
    "message": "No Seed-VC conversion is in progress.",
    "error": "",
    "phase": "idle",
    "started_at": None,
    "completed_at": None,
    "details": {},
}
_VOICE_RUNTIME_OWNED_PID: int | None = None
_VOICE_RUNTIME_OWNED_APP_PID: int | None = None


def _active_lokr_run() -> dict[str, Any] | None:
    for run in _lokr_runs():
        if run.get("status") == "running":
            return run
    return None


def _voice_runtime_action_state() -> dict[str, Any]:
    with _VOICE_RUNTIME_ACTION_LOCK:
        return dict(_VOICE_RUNTIME_ACTION_STATE)


def _set_voice_runtime_action_state(**updates: Any) -> None:
    with _VOICE_RUNTIME_ACTION_LOCK:
        _VOICE_RUNTIME_ACTION_STATE.update(updates)


def _voice_work_job_state() -> dict[str, Any]:
    with _VOICE_WORK_JOB_LOCK:
        return dict(_VOICE_WORK_JOB_STATE)


def _set_voice_work_job_state(**updates: Any) -> None:
    with _VOICE_WORK_JOB_LOCK:
        _VOICE_WORK_JOB_STATE.update(updates)


def _set_owned_voice_runtime_pid(pid: int | None) -> None:
    global _VOICE_RUNTIME_OWNED_PID, _VOICE_RUNTIME_OWNED_APP_PID
    _VOICE_RUNTIME_OWNED_PID = pid
    _VOICE_RUNTIME_OWNED_APP_PID = os.getpid() if pid is not None else None


def _owned_voice_runtime_pid() -> int | None:
    if _VOICE_RUNTIME_OWNED_APP_PID != os.getpid():
        return None
    return _VOICE_RUNTIME_OWNED_PID


def _start_voice_runtime_action(
    action: str,
    runner: callable,
    *,
    ui_log: UiLog,
) -> bool:
    with _VOICE_RUNTIME_ACTION_LOCK:
        if _VOICE_RUNTIME_ACTION_STATE.get("active"):
            return False
        _VOICE_RUNTIME_ACTION_STATE.update(
            {
                "active": True,
                "action": action,
                "message": f"{action.capitalize()} in progress.",
                "error": "",
                "phase": f"{action}ing",
                "started_at": _now_iso(),
                "completed_at": None,
            }
        )

    def _run() -> None:
        try:
            message = runner()
            _set_voice_runtime_action_state(
                active=False,
                action=action,
                message=message,
                error="",
                phase="idle",
                completed_at=_now_iso(),
            )
            ui_log.add("info", message)
        except Exception as exc:
            _set_voice_runtime_action_state(
                active=False,
                action=action,
                message=f"{action.capitalize()} failed.",
                error=str(exc),
                phase="failed",
                completed_at=_now_iso(),
            )
            ui_log.add("error", f"Seed-VC runtime {action} failed: {exc}")

    thread = threading.Thread(target=_run, name=f"voice-runtime-{action}", daemon=True)
    thread.start()
    return True


def _start_voice_work_job(
    action: str,
    runner: callable,
    *,
    ui_log: UiLog,
    details: dict[str, Any] | None = None,
) -> bool:
    job_details = dict(details or {})
    request_id = str(job_details.get("request_id") or "").strip()
    with _VOICE_WORK_JOB_LOCK:
        if _VOICE_WORK_JOB_STATE.get("active"):
            return False
        _VOICE_WORK_JOB_STATE.update(
            {
                "active": True,
                "action": action,
                "message": f"{action.capitalize()} in progress." + (f" Request {request_id}." if request_id else ""),
                "error": "",
                "phase": f"{action}ing",
                "started_at": _now_iso(),
                "completed_at": None,
                "details": job_details,
            }
        )

    def _run() -> None:
        try:
            result = runner()
            _set_voice_work_job_state(
                active=False,
                action=action,
                message=result if isinstance(result, str) else f"{action.capitalize()} complete.",
                error="",
                phase="idle",
                completed_at=_now_iso(),
            )
            if isinstance(result, str):
                ui_log.add("info", result)
        except Exception as exc:
            voice_id = str(_VOICE_WORK_JOB_STATE.get("details", {}).get("voice_id") or "")
            request_id = str(_VOICE_WORK_JOB_STATE.get("details", {}).get("request_id") or "")
            if action == "train" and voice_id:
                try:
                    update_voice(voice_id, {"training_status": "failed", "training_error": str(exc)})
                except Exception:
                    pass
            _set_voice_work_job_state(
                active=False,
                action=action,
                message=f"{action.capitalize()} failed.",
                error=str(exc),
                phase="failed",
                completed_at=_now_iso(),
            )
            ui_log.add("error", f"Voice Work {action} failed{f' [{request_id}]' if request_id else ''}: {exc}")

    thread = threading.Thread(target=_run, name=f"voice-work-{action}", daemon=True)
    thread.start()
    return True


def _voice_runtime_status_payload(runtime_config: RuntimeConfig) -> dict[str, Any]:
    status = rvc_runtime_status(runtime_config).to_dict()
    action = _voice_runtime_action_state()
    managed_pid = read_rvc_runtime_pid()
    managed_pid_alive = rvc_managed_runtime_alive(runtime_config)
    startup_progress = rvc_startup_progress_snapshot()

    phase = "installed"
    phase_message = str(status.get("message") or "Seed-VC runtime status is unknown.")

    if action.get("active"):
        phase = str(action.get("phase") or "working")
        phase_message = str(action.get("message") or phase_message)
    elif action.get("phase") == "failed":
        phase = "failed"
        phase_message = str(action.get("error") or action.get("message") or phase_message)
    elif bool(status.get("api_running")):
        phase = "ready"
        phase_message = "Seed-VC runtime is reachable."
    elif not bool(status.get("installed")):
        phase = "missing"
        phase_message = "Seed-VC runtime is not installed."
    else:
        phase = "installed"
        phase_message = "Seed-VC runtime is installed but not running."

    if phase == "installed" and managed_pid_alive:
        if startup_progress.get("phase") not in {"idle", "failed", "interrupted"}:
            phase = "starting"
            phase_message = str(startup_progress.get("message") or "Seed-VC runtime is still starting.")
        else:
            phase = "stale"
            phase_message = str(startup_progress.get("message") or "Seed-VC process exists, but the UI is not reachable.")
    elif phase == "installed" and action.get("active"):
        phase = str(action.get("phase") or "working")
        phase_message = str(action.get("message") or phase_message)

    status["action"] = action
    status["managed_pid"] = managed_pid
    status["managed_pid_alive"] = managed_pid_alive
    start_owned = bool(action.get("active")) and str(action.get("action") or "") in {"start", "restart"} and managed_pid is not None
    status["owned_by_app"] = (_owned_voice_runtime_pid() == managed_pid and managed_pid is not None) or start_owned
    status["simple_setup_command"] = "Install Runtime"
    status["simple_start_command"] = "Restart Runtime" if phase in {"ready", "starting", "stale"} else "Start Runtime"
    status["startup_progress"] = startup_progress
    status["phase"] = phase
    status["phase_message"] = phase_message
    return status


def _voice_work_runtime_python(runtime_config: RuntimeConfig) -> Path:
    install_dir = runtime_config.rvc_dir.expanduser()
    if sys.platform == "win32":
        return install_dir / ".venv" / "Scripts" / "python.exe"
    return install_dir / ".venv" / "bin" / "python"


def _voice_work_runtime_log_path() -> Path:
    return Path("data/logs/seed-vc.log")


def _voice_work_training_progress_summary(log_text: str, *, total_epochs: int) -> str | None:
    epoch_matches = re.findall(r"Train Epoch:\s*(\d+)\s*\[(\d+)%\]", log_text)
    if epoch_matches:
        epoch, percent = epoch_matches[-1]
        return f"Epoch {epoch}/{total_epochs} ({percent}%)"
    epoch_records = re.findall(r"====>\s*Epoch:\s*(\d+)\s*\[([^\]]+)\]", log_text)
    if epoch_records:
        epoch, elapsed = epoch_records[-1]
        return f"Epoch {epoch}/{total_epochs} ({elapsed})"
    return None


def _voice_work_log_tail(log_text: str, *, lines: int = 20) -> str:
    tail = [line.strip() for line in log_text.replace("\r", "\n").splitlines() if line.strip()]
    if len(tail) > lines:
        tail = tail[-lines:]
    return "\n".join(tail)


def _run_rvc_client(api_name: str, args: list[Any], runtime_config: RuntimeConfig) -> Any:
    python_exe = _voice_work_runtime_python(runtime_config)
    if not python_exe.exists():
        raise RuntimeError("Voice Work runtime dependencies are not installed.")
    payload = {
        "base_url": rvc_runtime_status(runtime_config).api_url,
        "api_name": api_name,
        "args": args,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        temp_path = Path(handle.name)
    try:
        script = """
import json
import sys
import tempfile
from gradio_client import Client, handle_file
from pathlib import Path

import numpy as np
import soundfile as sf

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
client = Client(payload["base_url"])
result = client.predict(*payload["args"], api_name=payload["api_name"])

def materialize(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            return value
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], (int, float)) and not isinstance(value[1], (str, bytes, bytearray)):
            sample_rate = int(value[0])
            data = np.asarray(value[1])
            out_dir = Path(tempfile.mkdtemp(prefix="voice-work-rvc-"))
            out_path = out_dir / "output.wav"
            sf.write(out_path, data, sample_rate)
            return {"kind": "audio_file", "path": str(out_path), "sample_rate": sample_rate}
        return [materialize(item) for item in value]
    return value

print(json.dumps({"result": materialize(result)}, default=str))
"""
        command = [str(python_exe), "-c", "from pathlib import Path; " + script, str(temp_path)]
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("LC_ALL", "C.UTF-8")
        env.setdefault("LANG", "C.UTF-8")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "Voice Work runtime call failed."
            raise RuntimeError(stderr)
        output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        output = output_lines[-1] if output_lines else ""
        if not output:
            return None
        return json.loads(output).get("result")
    finally:
        temp_path.unlink(missing_ok=True)


def _run_rvc_voice_conversion(
    *,
    model_name: str,
    source_path: Path,
    f0_up_key: int,
    f0_method: str,
    index_path: str,
    index_rate: float,
    filter_radius: int,
    resample_sr: int,
    rms_mix_rate: float,
    protect: float,
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    python_exe = _voice_work_runtime_python(runtime_config)
    if not python_exe.exists():
        raise RuntimeError("Voice Work runtime dependencies are not installed.")
    runtime_dir = runtime_config.rvc_dir.expanduser().resolve()
    payload = {
        "runtime_dir": str(runtime_dir),
        "model_name": model_name,
        "source_path": str(source_path),
        "f0_up_key": int(f0_up_key),
        "f0_method": "rmvpe" if f0_method == "rmvpe_gpu" else f0_method,
        "index_path": index_path,
        "index_rate": float(index_rate),
        "filter_radius": int(filter_radius),
        "resample_sr": int(resample_sr),
        "rms_mix_rate": float(rms_mix_rate),
        "protect": float(protect),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        temp_path = Path(handle.name)
    try:
        script = """
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

sys.argv = [sys.argv[0]]
payload = json.loads(Path(os.environ["VOICE_WORK_PAYLOAD_PATH"]).read_text(encoding="utf-8"))
runtime_dir = Path(payload["runtime_dir"])
os.chdir(runtime_dir)
load_dotenv(runtime_dir / ".env")

from configs.config import Config
from infer.modules.vc.modules import VC

config = Config()
vc = VC(config)
vc.get_vc(payload["model_name"])
info, audio = vc.vc_single(
    0,
    payload["source_path"],
    int(payload["f0_up_key"]),
    None,
    payload["f0_method"],
    payload["index_path"],
    "",
    float(payload["index_rate"]),
    int(payload["filter_radius"]),
    int(payload["resample_sr"]),
    float(payload["rms_mix_rate"]),
    float(payload["protect"]),
)

result: dict[str, object] = {"info": info}
if isinstance(audio, tuple) and len(audio) == 2 and audio[0] is not None and audio[1] is not None:
    sample_rate = int(audio[0])
    data = np.asarray(audio[1])
    out_dir = Path(tempfile.mkdtemp(prefix="voice-work-rvc-"))
    out_path = out_dir / "output.wav"
    sf.write(out_path, data, sample_rate)
    result["result"] = {"kind": "audio_file", "path": str(out_path), "sample_rate": sample_rate}
else:
    result["result"] = None

print(json.dumps(result, default=str))
"""
        command = [str(python_exe), "-c", "from pathlib import Path; " + script]
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("LC_ALL", "C.UTF-8")
        env.setdefault("LANG", "C.UTF-8")
        env["weight_root"] = "assets/weights"
        env["weight_uvr5_root"] = "assets/uvr5_weights"
        env["index_root"] = "logs"
        env["outside_index_root"] = "assets/indices"
        env["rmvpe_root"] = "assets/rmvpe"
        env["VOICE_WORK_PAYLOAD_PATH"] = str(temp_path)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
            cwd=runtime_dir,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "Voice Work runtime conversion failed."
            raise RuntimeError(stderr)
        output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        output = output_lines[-1] if output_lines else ""
        if not output:
            raise RuntimeError("Voice Work runtime returned no output.")
        payload_result = json.loads(output)
        info = str(payload_result.get("info") or "")
        runtime_result = payload_result.get("result")
        if runtime_result is None:
            raise RuntimeError(info or "Voice Work runtime returned no audio file.")
        if info and not info.startswith("Success"):
            raise RuntimeError(info)
        return {"info": info, "result": runtime_result}
    finally:
        temp_path.unlink(missing_ok=True)


def _voice_work_spawn_tts_wav(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{str(output_path).replace("'", "''")}')
$synth.Speak('{text.replace("'", "''")}')
$synth.Dispose()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _voice_work_gpu_text() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else ""
    except Exception:
        return ""


def _voice_work_default_threads() -> int:
    return max(1, min(4, (os.cpu_count() or 4) // 2))


def _voice_work_runtime_assets_dir(runtime_config: RuntimeConfig) -> Path:
    return runtime_config.rvc_dir.expanduser() / "assets" / "weights"


def _voice_work_runtime_logs_dir(runtime_config: RuntimeConfig) -> Path:
    return runtime_config.rvc_dir.expanduser() / "logs"


def _voice_work_latest_index_path(experiment_name: str, runtime_config: RuntimeConfig) -> Path | None:
    logs_dir = _voice_work_runtime_logs_dir(runtime_config) / experiment_name
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("added_*.index")) or sorted(logs_dir.glob("trained_*.index"))
    if candidates:
        return candidates[-1]
    return None


def _voice_work_training_run_name(voice_id: str) -> str:
    return f"{voice_id}-{uuid4().hex[:8]}"


def _voice_work_extract_existing_path(result: Any) -> Path | None:
    if isinstance(result, str):
        candidate = Path(result).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate
        return None
    if isinstance(result, dict):
        for key in ("path", "file", "value", "name"):
            candidate = result.get(key)
            if isinstance(candidate, str):
                path = Path(candidate).expanduser()
                if path.exists() and path.is_file():
                    return path
        for value in result.values():
            path = _voice_work_extract_existing_path(value)
            if path is not None:
                return path
        return None
    if isinstance(result, (list, tuple)):
        for value in result:
            path = _voice_work_extract_existing_path(value)
            if path is not None:
                return path
    return None


def _voice_work_normalize_render_output(
    *,
    voice: dict[str, Any],
    source_path: str,
    runtime_result: Any,
    label: str,
    render_type: str,
    mode: str = "singing",
    text: str = "",
    language: str = "auto",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = _voice_work_extract_existing_path(runtime_result)
    if output_path is None:
        raise RuntimeError("Voice Work runtime returned no audio file.")
    return create_voice_render_record(
        label=label,
        voice=voice,
        source_path=source_path,
        output_audio_path=output_path,
        runtime_payload={"result": runtime_result, "mode": mode},
        render_type=render_type,
        mode=mode,
        text=text,
        language=language,
        extra=extra,
    )


def _voice_work_train_voice_asset(
    voice_id: str,
    *,
    runtime_config: RuntimeConfig,
    epochs: int = 20,
    save_every: int = 5,
    batch_size: int = 1,
    use_f0: bool = True,
    sample_rate: str = "48k",
    version: str = "v2",
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    voice = read_voice(voice_id)
    training_run_name = _voice_work_training_run_name(voice_id)
    reference_dir = (Path(str(voice.get("voice_dir") or "")).expanduser().resolve() / "references")
    if not reference_dir.exists() or not any(reference_dir.iterdir()):
        raise RuntimeError("Voice clone needs reference audio before training.")

    result = ensure_rvc_runtime_api(runtime_config)
    if not result.started and not result.already_running and not rvc_api_health(runtime_config):
        raise RuntimeError(result.message)

    gpu_text = _voice_work_gpu_text()
    f0_method = "rmvpe_gpu" if gpu_text else "rmvpe"
    gpus_rmvpe = gpu_text if gpu_text else "-"
    threads = int(cpu_threads or _voice_work_default_threads())
    model_path = runtime_config.rvc_dir.expanduser() / "assets" / "weights" / f"{training_run_name}.pth"
    log_path = _voice_work_runtime_log_path()
    log_start_size = log_path.stat().st_size if log_path.exists() else 0
    _set_voice_work_job_state(
        active=True,
        action="train",
        message=f"Training voice '{voice.get('label') or voice_id}'.",
        error="",
        phase="training",
        started_at=_now_iso(),
        completed_at=None,
        details={
            "voice_id": voice_id,
            "training_run_id": training_run_name,
            "stage": "training",
            "reference_dir": str(reference_dir),
        },
    )
    runtime_result = _run_rvc_client(
        "/train_start_all",
        [
            training_run_name,
            sample_rate,
            use_f0,
            str(reference_dir),
            0,
            threads,
            f0_method,
            save_every,
            epochs,
            batch_size,
            "No",
            "",
            "",
            gpu_text,
            "No",
            "Yes",
            version,
            gpus_rmvpe,
        ],
        runtime_config,
    )
    timeout_seconds = max(1800, int(epochs) * 120)
    deadline = time.monotonic() + timeout_seconds
    last_progress_message = ""
    while time.monotonic() < deadline:
        log_text = ""
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(log_start_size)
                log_text = handle.read()
        if "Traceback (most recent call last):" in log_text:
            raise RuntimeError(_voice_work_log_tail(log_text))
        progress_message = _voice_work_training_progress_summary(log_text, total_epochs=epochs)
        if progress_message and progress_message != last_progress_message:
            last_progress_message = progress_message
            _set_voice_work_job_state(
                active=True,
                action="train",
                message=f"Training voice '{voice.get('label') or voice_id}': {progress_message}.",
                error="",
                phase="training",
                started_at=_VOICE_WORK_JOB_STATE.get("started_at"),
                completed_at=None,
                details={
                    "voice_id": voice_id,
                    "training_run_id": training_run_name,
                    "stage": "training",
                    "reference_dir": str(reference_dir),
                    "progress": progress_message,
                },
            )
        completion_marker = "Training is done. The program is closed."
        if model_path.exists() and log_text and completion_marker in log_text:
            break
        time.sleep(2)
    if not model_path.exists():
        raise RuntimeError(
            f"Voice training timed out before writing the final model checkpoint: {model_path}"
        )
    index_path = _voice_work_latest_index_path(training_run_name, runtime_config)
    updated = update_voice(
        voice_id,
        {
            "trained_model_path": str(model_path) if model_path.exists() else "",
            "trained_index_path": str(index_path) if index_path and index_path.exists() else "",
            "trained_model_name": model_path.name if model_path.exists() else "",
            "trained_index_name": index_path.name if index_path and index_path.exists() else "",
            "training_status": "trained",
            "training_error": "",
            "training_run_id": training_run_name,
        },
    )
    _set_voice_work_job_state(
        active=False,
        action="train",
        message=f"Voice '{updated.get('label') or voice_id}' trained.",
        error="",
        phase="idle",
        started_at=_VOICE_WORK_JOB_STATE.get("started_at"),
        completed_at=_now_iso(),
        details={
            "voice_id": voice_id,
            "training_run_id": training_run_name,
            "stage": "complete",
            "model_path": str(model_path),
            "index_path": str(index_path) if index_path else "",
        },
    )
    return {
        "voice": updated,
        "runtime_result": runtime_result,
        "model_path": str(model_path) if model_path.exists() else "",
        "index_path": str(index_path) if index_path and index_path.exists() else "",
    }


def _voice_work_target_reference_path(voice: dict[str, Any]) -> Path:
    preview_path = str(voice.get("preview_audio_path") or "")
    if preview_path:
        candidate = Path(preview_path).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    for reference in voice.get("reference_files", []) or []:
        candidate = Path(str(reference.get("path") or "")).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    raise RuntimeError("Target voice does not have a usable reference audio file.")


def _run_seed_vc_voice_conversion(
    *,
    source_path: Path,
    reference_path: Path,
    mode: str,
    diffusion_steps: int,
    length_adjust: float,
    inference_cfg_rate: float,
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    if not rvc_api_health(runtime_config):
        raise RuntimeError("Seed-VC runtime is not running.")
    if mode not in {"speaking", "singing"}:
        mode = "singing"
    payload = {
        "source_path": str(source_path),
        "reference_path": str(reference_path),
        "mode": mode,
        "diffusion_steps": int(diffusion_steps),
        "length_adjust": float(length_adjust),
        "inference_cfg_rate": float(inference_cfg_rate),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        temp_path = Path(handle.name)
    try:
        script = """
import json
import sys
from pathlib import Path

import torch

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
module_name = "app_svc" if payload["mode"] == "singing" else "app_vc"
seed_vc = __import__(module_name)
seed_vc.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Args:
    checkpoint = None
    config = None
    fp16 = True
    gpu = 0
    share = False
    port = 7860


args = Args()
loaded = seed_vc.load_models(args)
loaded_values = list(loaded) if isinstance(loaded, (list, tuple)) else [loaded]
if len(loaded_values) < 6:
    raise RuntimeError(f"Seed-VC load_models returned an unexpected result shape: {len(loaded_values)}")
if payload["mode"] == "singing":
    seed_vc.model_f0 = loaded_values[0]
    seed_vc.semantic_fn = loaded_values[1]
    seed_vc.vocoder_fn = loaded_values[2]
    seed_vc.campplus_model = loaded_values[3]
    seed_vc.to_mel_f0 = loaded_values[4]
    seed_vc.mel_fn_args = loaded_values[5]
    seed_vc.f0_fn = loaded_values[6] if len(loaded_values) > 6 else None
    seed_vc.max_context_window = seed_vc.sr // seed_vc.hop_length * 30
    seed_vc.overlap_wave_len = seed_vc.overlap_frame_len * seed_vc.hop_length
    result = seed_vc.voice_conversion(
        payload["source_path"],
        payload["reference_path"],
        payload["diffusion_steps"],
        payload["length_adjust"],
        payload["inference_cfg_rate"],
    )
else:
    seed_vc.model = loaded_values[0]
    seed_vc.semantic_fn = loaded_values[1]
    seed_vc.vocoder_fn = loaded_values[2]
    seed_vc.campplus_model = loaded_values[3]
    seed_vc.to_mel = loaded_values[4]
    seed_vc.mel_fn_args = loaded_values[5]
    seed_vc.f0_fn = loaded_values[6] if len(loaded_values) > 6 else None
    seed_vc.max_context_window = seed_vc.sr // seed_vc.hop_length * 30
    seed_vc.overlap_wave_len = seed_vc.overlap_frame_len * seed_vc.hop_length
    result = seed_vc.voice_conversion(
        payload["source_path"],
        payload["reference_path"],
        payload["diffusion_steps"],
        payload["length_adjust"],
        payload["inference_cfg_rate"],
    )

print(json.dumps({"result": result}, default=str))
"""
        completed = subprocess.run(
            [str(_voice_work_runtime_python(runtime_config)), "-c", script, str(temp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            cwd=str(runtime_config.rvc_dir.expanduser()),
            env={
                **os.environ.copy(),
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
            },
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "Seed-VC runtime conversion failed."
            raise RuntimeError(stderr)
        output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        output = output_lines[-1] if output_lines else ""
        if not output:
            raise RuntimeError("Seed-VC runtime returned no output.")
        payload_result = json.loads(output)
        return {"result": payload_result.get("result")}
    finally:
        temp_path.unlink(missing_ok=True)


def _voice_work_convert_audio(
    *,
    voice_id: str,
    source_audio_path: str,
    label: str,
    mode: str,
    diffusion_steps: int,
    length_adjust: float,
    inference_cfg_rate: float,
    runtime_config: RuntimeConfig,
    request_id: str | None = None,
    render_type: str = "conversion",
    text: str = "",
    language: str = "auto",
) -> dict[str, Any]:
    voice = read_voice(voice_id)
    source_path = Path(source_audio_path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise RuntimeError(f"Source audio not found: {source_audio_path}")
    try:
        validate_supported_source(source_path)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    reference_path = _voice_work_target_reference_path(voice)
    runtime_result = _run_seed_vc_voice_conversion(
        source_path=source_path,
        reference_path=reference_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        mode=mode,
        runtime_config=runtime_config,
    )
    return _voice_work_normalize_render_output(
        voice=voice,
        source_path=str(source_path),
        runtime_result=runtime_result,
        label=label or f"{voice.get('label') or voice_id} conversion",
        render_type=render_type,
        mode=mode,
        text=text,
        language=language,
        extra={"request_id": (request_id or "").strip()} if (request_id or "").strip() else None,
    )


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

    for item in _local_library().list_items():
        metadata = item.metadata or {}
        if metadata.get("category") != "sound_effect":
            continue
        audio_file = next((file for file in item.files if file.role in {"audio", "preview", "stem"}), None)
        if audio_file is None:
            continue
        audio_path = Path(audio_file.path)
        if not audio_path.exists() or not audio_path.is_file():
            continue
        assets.append(
            {
                "asset_id": item.id,
                "category": "sound_effect",
                "label": item.title,
                "audio_path": str(audio_path),
                "audio_url": f"/api/editor/audio?path={quote(str(audio_path))}",
                "duration_seconds": audio_file.metadata.get("duration_seconds") or 0,
                "created_at": item.created_at,
                "metadata_path": str(_local_library()._manifest_path(item.id)),
                "message": "Sound effect generation",
                "source_path": metadata.get("prompt") or "",
                "source_asset_id": "",
                "imported": False,
                "creator_name": "",
            }
        )

    return sorted(assets, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def _local_library() -> LocalLibraryIndex:
    return LocalLibraryIndex(Path("data/library"))


def _find_editor_asset(asset_id: str) -> dict[str, Any]:
    for asset in _editor_assets():
        if str(asset.get("asset_id") or "") == asset_id:
            return asset
    raise FileNotFoundError(f"Editor asset not found: {asset_id}")


def _sync_local_library_index() -> list[LibraryItem]:
    library = _local_library()
    return library.reindex_items(_local_library_scanned_items())


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


def _attach_extraction_to_rhythm_project(
    *,
    project_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    project = read_rhythm_project(project_id)
    output_path = Path(str(metadata.get("generated_audio_path") or "")).expanduser()
    extracted_probe = probe_audio(output_path)
    project_tracks = list(project.get("tracks") or [])
    extraction_ref = str(metadata.get("extraction_id") or "")
    if not any(str(track.get("source_asset_id") or "") == extraction_ref for track in project_tracks):
        project_tracks.append(
            {
                "track_id": f"track-{uuid4().hex[:10]}",
                "label": str(metadata.get("label") or metadata.get("track_name") or output_path.stem),
                "audio_path": str(output_path),
                "audio_url": f"/api/audio?path={quote(str(output_path))}",
                "duration_seconds": extracted_probe.duration_seconds,
                "source_asset_id": extraction_ref,
                "source_category": "extraction",
                "created_at": utc_now_iso(),
            }
        )
        project["tracks"] = project_tracks
        project = write_rhythm_project(project)
        _sync_local_library_index()
    return project


def _schedule_extraction_retry(
    *,
    extraction_id: str,
    runtime_config: RuntimeConfig,
    source_path: Path,
    track_name: str,
    label: str,
    output_format: str,
    inference_steps: int,
    guidance_scale: float,
    shift: float,
    infer_method: str,
    use_tiled_decode: bool,
    dcw_enabled: bool,
    velocity_norm_threshold: float,
    velocity_ema_factor: float,
    seed: int | None,
    instruction: str | None,
    rhythm_project_id: str | None = None,
) -> None:
    from autotransition.runtime.ace_step import api_health, runtime_recovery_state

    def runner() -> None:
        try:
            deadline = time.monotonic() + runtime_config.api_startup_timeout_seconds + 120
            while time.monotonic() < deadline:
                recovery = runtime_recovery_state()
                if not recovery.active and api_health(runtime_config):
                    break
                _update_extraction_metadata(
                    extraction_id,
                    {
                        "status": "recovering",
                        "message": "ACE-Step runtime is recovering. Extraction will retry automatically when the runtime is ready.",
                        "retry_state": "waiting_for_runtime",
                    },
                )
                time.sleep(2)
            else:
                _update_extraction_metadata(
                    extraction_id,
                    {
                        "status": "failed",
                        "message": "ACE-Step runtime did not recover in time. Retry the extraction.",
                        "retry_state": "recovery_timeout",
                    },
                )
                return

            _update_extraction_metadata(
                extraction_id,
                {
                    "status": "retrying",
                    "message": "ACE-Step runtime recovered. Retrying extraction now.",
                    "retry_state": "retrying",
                },
            )

            result = AceStepApiClient(runtime_config).extract_track(
                source_path=source_path,
                track_name=track_name,
                save_dir=Path("data/extractions") / extraction_id,
                audio_format=output_format,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                infer_method=infer_method,
                use_tiled_decode=use_tiled_decode,
                dcw_enabled=dcw_enabled,
                velocity_norm_threshold=velocity_norm_threshold,
                velocity_ema_factor=velocity_ema_factor,
                seed=seed,
                instruction=instruction.strip() if instruction else None,
            )
            metadata = _update_extraction_metadata(
                extraction_id,
                {
                    "status": "complete",
                    "message": "Extraction complete after ACE-Step runtime recovery.",
                    "generated_audio_path": str(result.output_path),
                    "generated_metadata_path": str(result.metadata_path),
                    "retry_state": "complete",
                },
            )
            if metadata and rhythm_project_id:
                _attach_extraction_to_rhythm_project(project_id=rhythm_project_id, metadata=metadata)
        except Exception as exc:
            _update_extraction_metadata(
                extraction_id,
                {
                    "status": "failed",
                    "message": f"Extraction retry failed after runtime recovery: {exc}",
                    "retry_state": "failed",
                },
            )
        finally:
            with _extract_retry_lock:
                _extract_retry_threads.pop(extraction_id, None)

    with _extract_retry_lock:
        existing = _extract_retry_threads.get(extraction_id)
        if existing and existing.is_alive():
            return
        thread = threading.Thread(target=runner, name=f"extract-retry-{extraction_id}", daemon=True)
        _extract_retry_threads[extraction_id] = thread
        thread.start()


def _schedule_post_extract_runtime_recycle(
    runtime_config: RuntimeConfig,
    ui_log: UiLog,
    *,
    reason: str,
) -> dict[str, Any] | None:
    import sys

    if sys.platform != "win32":
        return None
    from autotransition.runtime.ace_step import runtime_recovery_state, schedule_runtime_recycle

    recovery = runtime_recovery_state()
    if recovery.active:
        return recovery.to_dict()
    recycled = schedule_runtime_recycle(reason=reason, config=runtime_config)
    ui_log.add("info", "Restarting ACE-Step runtime after base extraction to release Windows paging pressure.")
    return recycled.to_dict()


def _run_extraction_job(
    *,
    runtime_config: RuntimeConfig,
    ui_log: UiLog,
    source_path: Path,
    track_name: str,
    label: str,
    output_format: str,
    inference_steps: int,
    guidance_scale: float,
    shift: float,
    infer_method: str,
    use_tiled_decode: bool,
    dcw_enabled: bool,
    velocity_norm_threshold: float,
    velocity_ema_factor: float,
    seed: int | None,
    instruction: str | None,
    rhythm_project_id: str | None = None,
) -> dict[str, Any]:
    import datetime as _datetime

    if track_name not in EXTRACT_TRACKS:
        raise HTTPException(status_code=400, detail=f"Unknown extract track: {track_name}")

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
            audio_format=output_format,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            infer_method=infer_method,
            use_tiled_decode=use_tiled_decode,
            dcw_enabled=dcw_enabled,
            velocity_norm_threshold=velocity_norm_threshold,
            velocity_ema_factor=velocity_ema_factor,
            seed=seed,
            instruction=instruction.strip() if instruction else None,
        )
    except AceStepApiError as exc:
        failure = _handle_ace_runtime_failure(runtime_config, ui_log, "ACE-Step track extraction failed.", exc)
        metadata = {
            "extraction_id": extraction_id,
            "status": "recovering" if failure["recovery_active"] else "failed",
            "message": failure["message"],
            "created_at": created_at,
            "source_path": str(source_path),
            "source_format": probe.source_format,
            "source_duration_seconds": probe.duration_seconds,
            "track_name": track_name,
            "label": label,
            "output_format": output_format,
            "metadata_path": str(metadata_path),
            "runtime_recovery": failure["recovery"],
            "retry_state": "waiting_for_runtime" if failure["recovery_active"] else "failed",
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if failure["recovery_active"]:
            _schedule_extraction_retry(
                extraction_id=extraction_id,
                runtime_config=runtime_config,
                source_path=source_path,
                track_name=track_name,
                label=label,
                output_format=output_format,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                infer_method=infer_method,
                use_tiled_decode=use_tiled_decode,
                dcw_enabled=dcw_enabled,
                velocity_norm_threshold=velocity_norm_threshold,
                velocity_ema_factor=velocity_ema_factor,
                seed=seed,
                instruction=instruction,
                rhythm_project_id=rhythm_project_id,
            )
            return metadata
        recycle = _schedule_post_extract_runtime_recycle(
            runtime_config,
            ui_log,
            reason="Restarting ACE-Step after base extraction failure to release Windows paging pressure.",
        )
        if recycle is not None:
            metadata["runtime_recovery"] = recycle
            metadata["post_extract_runtime_recycle"] = True
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    metadata = {
        "extraction_id": extraction_id,
        "status": "complete",
        "message": "Extraction complete.",
        "created_at": created_at,
        "source_path": str(source_path),
        "source_format": probe.source_format,
        "source_duration_seconds": probe.duration_seconds,
        "track_name": track_name,
        "label": label,
        "output_format": output_format,
        "generated_audio_path": str(result.output_path),
        "generated_metadata_path": str(result.metadata_path),
        "metadata_path": str(metadata_path),
        "settings": {
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "shift": shift,
            "infer_method": infer_method,
            "use_tiled_decode": use_tiled_decode,
            "dcw_enabled": dcw_enabled,
            "velocity_norm_threshold": velocity_norm_threshold,
            "velocity_ema_factor": velocity_ema_factor,
            "seed": seed,
            "instruction": instruction,
        },
    }
    recycle = _schedule_post_extract_runtime_recycle(
        runtime_config,
        ui_log,
        reason="Restarting ACE-Step after base extraction completion to release Windows paging pressure.",
    )
    if recycle is not None:
        metadata["runtime_recovery"] = recycle
        metadata["post_extract_runtime_recycle"] = True
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ui_log.add("info", f"Extracted {track_name}: {result.output_path}")
    return metadata


def _handle_ace_runtime_failure(
    runtime_config: RuntimeConfig,
    ui_log: Any,
    prefix: str,
    exc: Exception,
) -> dict[str, Any]:
    from autotransition.runtime.ace_step import api_health, runtime_recovery_state, schedule_runtime_recovery

    error_text = str(exc)
    if api_health(runtime_config):
        message = f"{prefix} {error_text}"
        ui_log.add("error", message)
        return {
            "message": message,
            "recovery_active": False,
            "recovery": runtime_recovery_state().to_dict(),
        }

    recovery = schedule_runtime_recovery(
        reason=f"{prefix} Runtime became unreachable after an ACE-Step failure.",
        config=runtime_config,
    )
    message = (
        "ACE-Step runtime crashed or became unreachable. "
        "Dance Station is restarting it in the background. "
        f"{error_text}"
    )
    ui_log.add("error", message)
    return {
        "message": message,
        "recovery_active": True,
        "recovery": recovery.to_dict(),
    }


def _extract_lyrics_from_audio(
    *,
    audio_path: Path,
    model_name: str,
    language: str | None,
    include_word_timestamps: bool,
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper is not installed in this environment. Run `python -m pip install -e \".[dev]\"` again."
        ) from exc

    selected_language = (language or "").strip() or None
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"} else "cpu"
    if device == "cpu":
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
    whisper = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, info = whisper.transcribe(
        str(audio_path),
        language=selected_language,
        word_timestamps=include_word_timestamps,
        vad_filter=True,
    )
    lyric_segments: list[dict[str, Any]] = []
    text_lines: list[str] = []
    for index, segment in enumerate(segments):
        segment_text = (segment.text or "").strip()
        if not segment_text:
            continue
        text_lines.append(segment_text)
        lyric_segments.append(
            {
                "id": f"segment-{index + 1}",
                "text": segment_text,
                "startSeconds": float(segment.start),
                "endSeconds": float(segment.end),
                "words": [
                    {
                        "text": (word.word or "").strip(),
                        "startSeconds": float(word.start),
                        "endSeconds": float(word.end),
                    }
                    for word in (segment.words or [])
                    if (word.word or "").strip()
                ],
            }
        )
    language_probability = getattr(info, "language_probability", None)
    return {
        "enabled": bool(lyric_segments),
        "source": "extracted",
        "provider": "faster-whisper",
        "model": model_name,
        "language": getattr(info, "language", None),
        "language_probability": float(language_probability) if language_probability is not None else None,
        "text": "\n".join(text_lines),
        "segments": lyric_segments,
        "updated_at_iso": utc_now_iso(),
    }


def _default_rhythm_track_label(source_path: Path, track_name: str) -> str:
    source_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._") or "source"
    clean_track_name = re.sub(r"[^A-Za-z0-9._-]+", "_", track_name.strip().lower()).strip("._") or "track"
    return f"{source_stem}_{clean_track_name}"


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


def _library_items_from_voice_work() -> list[LibraryItem]:
    items: list[LibraryItem] = []
    for voice in list_voices():
        item = library_item_from_voice(voice)
        if item is not None:
            items.append(item)
    for generation in list_generations():
        item = library_item_from_generation(generation)
        if item is not None:
            items.append(item)
    return items


def _library_items_from_rhythm_projects() -> list[LibraryItem]:
    items: list[LibraryItem] = []
    for summary in list_rhythm_projects():
        project_id = str(summary.get("project_id") or "")
        if not project_id or not summary.get("has_final_asset"):
            continue
        try:
            project = read_rhythm_project(project_id)
        except Exception:
            continue
        item = library_item_from_rhythm_project(project)
        if item is not None:
            items.append(item)
    return items


def _local_library_scanned_items() -> list[LibraryItem]:
    items = [item for asset in _editor_assets() if (item := library_item_from_editor_asset(asset)) is not None]
    items.extend(_library_items_from_music_generations())
    items.extend(_library_items_from_lokr_datasets())
    items.extend(_library_items_from_lokr_adapters())
    items.extend(_library_items_from_sound_effects())
    items.extend(_library_items_from_voice_work())
    items.extend(_library_items_from_rhythm_projects())
    return items


def _library_items_from_sound_effects() -> list[LibraryItem]:
    items: list[LibraryItem] = []
    for generation in list_sound_effect_generations():
        item = library_item_from_sound_effect_generation(generation)
        if item is not None:
            items.append(item)
    return items


def _library_items_from_music_generations() -> list[LibraryItem]:
    items: list[LibraryItem] = []
    for metadata_path in _music_generation_root().glob("*/generation.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        generation_id = str(metadata.get("generation_id") or "")
        audio_path = Path(str(metadata.get("generated_audio_path") or "")).expanduser()
        if not generation_id or not audio_path.exists() or not audio_path.is_file():
            continue
        items.append(
            LibraryItem(
                id=generation_id,
                visibility="local",
                status="draft",
                kind="generation",
                title=str(metadata.get("label") or generation_id),
                description=str(metadata.get("prompt") or "")[:600] or None,
                files=[
                    LibraryFile(
                        role="audio",
                        mime_type=audio_mime_type_for_path(audio_path),
                        size_bytes=audio_path.stat().st_size,
                        path=str(audio_path),
                        metadata={
                            "duration_seconds": metadata.get("audio_duration") or 0,
                            "model": metadata.get("model") or "",
                            "lokr_adapter": (metadata.get("lokr_adapter") or {}).get("adapter_id") if isinstance(metadata.get("lokr_adapter"), dict) else "",
                            "render_type": metadata.get("type") or "music",
                        },
                    ),
                    LibraryFile(
                        role="metadata",
                        mime_type="application/json",
                        size_bytes=metadata_path.stat().st_size if metadata_path.exists() else 0,
                        path=str(metadata_path),
                    ),
                ],
                metadata={
                    "category": "generation",
                    "type": metadata.get("type") or "music",
                    "prompt": metadata.get("prompt") or "",
                    "model": metadata.get("model") or "",
                    "output_format": metadata.get("output_format") or "flac",
                },
                created_at=str(metadata.get("created_at") or utc_now_iso()),
                updated_at=str(metadata.get("created_at") or utc_now_iso()),
            )
        )
    return sorted(items, key=lambda item: item.created_at or item.updated_at, reverse=True)


def _dataset_source_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _dataset_source_summary_from_local(dataset: dict[str, Any]) -> dict[str, Any]:
    metadata = dataset.get("metadata") or {}
    dataset_id = str(metadata.get("dataset_id") or "")
    return {
        "source_id": _dataset_source_id("local", dataset_id),
        "source_kind": "local",
        "dataset_id": dataset_id,
        "label": metadata.get("label") or metadata.get("name") or dataset_id,
        "sample_count": len(dataset.get("samples") or []),
        "metadata": metadata,
    }


def _dataset_source_from_library_item(item: LibraryItem) -> dict[str, Any] | None:
    if item.kind != "dataset":
        return None
    samples: list[dict[str, Any]] = []
    for file in item.files:
        if file.role != "dataset_sample":
            continue
        path = Path(file.path).expanduser()
        if not path.exists() or not path.is_file():
            continue
        sample_metadata = dict(file.metadata or {})
        sample_id = str(sample_metadata.get("sample_id") or file.id)
        label = str(sample_metadata.get("label") or path.stem or sample_id)
        samples.append(
            {
                "id": sample_id,
                "label": label,
                "filename": path.name,
                "audio_path": str(path),
                "resolved_audio_path": str(path),
                "audio_url": f"/api/audio?path={quote(str(path))}",
                "caption": str(sample_metadata.get("caption") or ""),
                "lyrics": str(sample_metadata.get("lyrics") or "[Instrumental]"),
                "formatted_lyrics": str(sample_metadata.get("lyrics") or "[Instrumental]"),
                "genre": str(sample_metadata.get("genre") or ""),
                "language": str(sample_metadata.get("language") or "unknown"),
                "duration": sample_metadata.get("duration") or sample_metadata.get("duration_seconds") or 0,
                "is_instrumental": bool(sample_metadata.get("is_instrumental", True)),
                "source_asset_id": sample_metadata.get("source_asset_id") or "",
                "source_category": sample_metadata.get("source_category") or "",
                "prompt_override": sample_metadata.get("prompt_override"),
                "custom_tag": sample_metadata.get("custom_tag") or "",
                "labeled": bool(sample_metadata.get("label")),
                "bpm": sample_metadata.get("bpm") or "N/A",
                "keyscale": sample_metadata.get("keyscale") or "N/A",
                "timesignature": sample_metadata.get("timesignature") or "4",
            }
        )
    return {
        "source_id": _dataset_source_id("library", item.id),
        "source_kind": "library",
        "library_item_id": item.id,
        "label": item.title,
        "sample_count": len(samples),
        "metadata": {
            **(item.metadata or {}),
            "dataset_id": item.id,
            "label": item.title,
        },
        "samples": samples,
    }


def _dataset_sources() -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for metadata_path in _lokr_dataset_root().glob("*/dataset.json"):
        try:
            dataset = _lokr_dataset_for_response(_read_lokr_dataset(metadata_path.parent.name))
        except Exception:
            continue
        sources.append(_dataset_source_summary_from_local(dataset))
    for item in _local_library().list_items():
        if not bool((item.metadata or {}).get("imported")):
            continue
        source = _dataset_source_from_library_item(item)
        if source:
            source.pop("samples", None)
            sources.append(source)
    return sorted(sources, key=lambda item: str(item.get("metadata", {}).get("updated_at") or item.get("metadata", {}).get("created_at") or ""), reverse=True)


def _dataset_source_detail(source_id: str) -> dict[str, Any]:
    if source_id.startswith("local:"):
        dataset_id = source_id.split(":", 1)[1]
        dataset = _lokr_dataset_for_response(_read_lokr_dataset(dataset_id))
        return {
            "source_id": source_id,
            "source_kind": "local",
            "dataset_id": dataset_id,
            "label": (dataset.get("metadata") or {}).get("label") or dataset_id,
            "sample_count": len(dataset.get("samples") or []),
            "metadata": dataset.get("metadata") or {},
            "samples": dataset.get("samples") or [],
        }
    if source_id.startswith("library:"):
        item_id = source_id.split(":", 1)[1]
        item = _local_library().read_item(item_id)
        if item is None:
            raise FileNotFoundError(f"Imported dataset not found: {item_id}")
        source = _dataset_source_from_library_item(item)
        if source is None:
            raise FileNotFoundError(f"Dataset source not found: {source_id}")
        return source
    raise FileNotFoundError(f"Dataset source not found: {source_id}")


def create_app(models_dir: Path = Path("models"), runtime_config: RuntimeConfig | None = None) -> FastAPI:
    runtime_config = runtime_config or RuntimeConfig()
    app = FastAPI(title="Dance Station", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    audiomass_dir = Path(__file__).resolve().parents[1] / "vendor" / "audiomass"
    ui_log = UiLog()
    ui_log.add("info", "UI server started.")

    def _current_voice_runtime_config() -> RuntimeConfig:
        return runtime_config

    @app.on_event("shutdown")
    def _shutdown_managed_voice_runtime() -> None:
        action = _voice_runtime_action_state()
        runtime_config = _current_voice_runtime_config()
        should_stop = (
            _owned_voice_runtime_pid() is not None
            or (
                action.get("active")
                and action.get("action") in {"start", "restart"}
                and rvc_managed_runtime_alive(runtime_config)
            )
            or rvc_managed_runtime_alive(runtime_config)
            or rvc_runtime_status(runtime_config).api_running
        )
        if not should_stop:
            return
        try:
            stop_rvc_runtime(runtime_config)
        finally:
            _set_owned_voice_runtime_pid(None)

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
        from autotransition.runtime.ace_step import (
            build_install_commands,
            build_start_api_command,
            managed_runtime_alive,
            read_runtime_pid,
            runtime_recovery_state,
            runtime_status,
        )

        status = runtime_status(runtime_config).to_dict()
        status["recovery"] = runtime_recovery_state().to_dict()
        status["managed_pid"] = read_runtime_pid()
        status["managed_pid_alive"] = managed_runtime_alive(runtime_config)
        status["install_commands"] = build_install_commands(runtime_config)
        status["start_api_command"] = build_start_api_command(runtime_config)
        status["simple_setup_command"] = "autotransition runtime setup"
        status["simple_start_command"] = "autotransition runtime start"
        status["side_step"] = side_step_status(runtime_config).to_dict()
        status["side_step_command"] = build_side_step_command(runtime_config)
        return status

    @app.get("/api/runtime/activity")
    def get_runtime_activity() -> dict[str, object]:
        from autotransition.runtime.ace_step import runtime_recovery_state, runtime_status

        activity = summarize_runtime_activity().to_dict()
        status = runtime_status(runtime_config)
        recovery = runtime_recovery_state()
        activity["api_running"] = status.api_running
        activity["api_url"] = status.api_url
        activity["runtime_message"] = status.message
        activity["recovery"] = recovery.to_dict()
        if recovery.active:
            activity["phase"] = "recovering"
            activity["message"] = recovery.message
            activity["detail"] = recovery.reason
        return activity

    @app.get("/api/voice-work/runtime/url")
    def get_voice_work_runtime_url() -> dict[str, str]:
        return {"url": rvc_runtime_status(_current_voice_runtime_config()).ui_url}

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
        try:
            validate_supported_source(audio_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(audio_path)

    @app.get("/api/voice-work/status")
    def get_voice_work_status() -> dict[str, object]:
        status = _voice_runtime_status_payload(_current_voice_runtime_config())
        status["job"] = _voice_work_job_state()
        return status

    @app.post("/api/voice-work/runtime/install")
    def install_voice_work_runtime() -> dict[str, object]:
        def _runner() -> str:
            run_rvc_install(_current_voice_runtime_config())
            status = _voice_runtime_status_payload(_current_voice_runtime_config())
            if status.get("phase") == "missing":
                raise RuntimeError("Seed-VC runtime install did not complete.")
            return "Seed-VC runtime setup complete."

        if not _start_voice_runtime_action("install", _runner, ui_log=ui_log):
            raise HTTPException(status_code=409, detail="Another Voice Work runtime action is already in progress.")
        return {"started": True, "action": "install"}

    @app.post("/api/voice-work/runtime/start")
    def start_voice_work_runtime() -> dict[str, object]:
        def _runner() -> str:
            effective_runtime_config = _current_voice_runtime_config()
            result = ensure_rvc_runtime_api(effective_runtime_config)
            if result.started or result.already_running:
                _set_owned_voice_runtime_pid(read_rvc_runtime_pid())
            status = _voice_runtime_status_payload(effective_runtime_config)
            if status.get("phase") not in {"ready", "starting", "stale"}:
                raise RuntimeError(str(result.message))
            return result.message

        if not _start_voice_runtime_action("start", _runner, ui_log=ui_log):
            raise HTTPException(status_code=409, detail="Another Voice Work runtime action is already in progress.")
        return {"started": True, "action": "start"}

    @app.post("/api/voice-work/runtime/restart")
    def restart_voice_work_runtime() -> dict[str, object]:
        def _runner() -> str:
            effective_runtime_config = _current_voice_runtime_config()
            if rvc_managed_runtime_alive(effective_runtime_config):
                stop_rvc_runtime(effective_runtime_config)
                _set_owned_voice_runtime_pid(None)
            result = ensure_rvc_runtime_api(effective_runtime_config)
            if result.started or result.already_running:
                _set_owned_voice_runtime_pid(read_rvc_runtime_pid())
            status = _voice_runtime_status_payload(effective_runtime_config)
            if status.get("phase") not in {"ready", "starting", "stale"}:
                raise RuntimeError(str(result.message))
            return "Seed-VC runtime restarted."

        if not _start_voice_runtime_action("restart", _runner, ui_log=ui_log):
            raise HTTPException(status_code=409, detail="Another Voice Work runtime action is already in progress.")
        return {"started": True, "action": "restart"}

    @app.post("/api/voice-work/runtime/stop")
    def stop_voice_work_runtime() -> dict[str, object]:
        stopped = stop_rvc_runtime(_current_voice_runtime_config())
        if stopped:
            _set_owned_voice_runtime_pid(None)
        return {"stopped": stopped, "message": "Seed-VC runtime stopped."}

    @app.get("/api/voice-work/voices")
    def get_voice_work_voices() -> list[dict[str, Any]]:
        return list_voices()

    @app.post("/api/voice-work/voices/upload")
    def upload_voice_work_voice(
        label: str = Form(...),
        language: str = Form("auto"),
        description: str = Form(""),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        ui_log.add(
            "info",
            f"Voice Work upload requested: label={label!r}, language={language!r}, files={len(files)}",
        )
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one reference audio file.")
        temp_dir = _voice_work_root() / ".uploads" / uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_paths: list[Path] = []
        try:
            for file in files:
                temp_paths.append(_copy_upload_to_temp(file, temp_dir))
            reference_paths = [path for path in temp_paths if _voice_upload_type(path) == "audio"]
            if not reference_paths:
                allowed = ", ".join(sorted(VOICE_AUDIO_EXTENSIONS))
                raise HTTPException(status_code=400, detail=f"Supported target voice files: {allowed}")
            voice = create_voice_asset(
                label=label,
                language=language,
                description=description,
                reference_paths=reference_paths,
            )
            ui_log.add("info", f"Voice Work upload stored {len(reference_paths)} reference file(s) for '{voice['label']}'.")
        finally:
            for file in files:
                try:
                    file.file.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)
        _sync_local_library_index()
        ui_log.add("info", f"Stored target voice '{voice['label']}'.")
        return {"voice": voice}

    @app.post("/api/voice-work/voices/from-asset")
    def create_voice_work_voice_from_asset(request: VoiceWorkTargetVoiceFromAssetRequest) -> dict[str, Any]:
        try:
            asset = _find_editor_asset(request.asset_id)
            audio_path = Path(str(asset.get("audio_path") or "")).expanduser()
            if not audio_path.exists() or not audio_path.is_file():
                raise FileNotFoundError(f"Asset audio not found: {audio_path}")
            voice = create_voice_asset(
                label=request.label or str(asset.get("label") or audio_path.stem),
                language=request.language,
                description=request.description,
                reference_paths=[audio_path],
                source_asset_id=str(asset.get("asset_id") or ""),
                source_asset_label=str(asset.get("label") or audio_path.stem),
                source_asset_category=str(asset.get("category") or ""),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        ui_log.add("info", f"Stored target voice from asset '{voice['label']}'.")
        return {"voice": voice}

    @app.post("/api/voice-work/tmp-upload")
    def upload_voice_work_temp_audio(file: UploadFile = File(...)) -> dict[str, str]:
        temp_dir = _voice_work_root() / ".uploads" / uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            temp_path = _copy_upload_to_temp(file, temp_dir)
        finally:
            try:
                file.file.close()
            except Exception:
                pass
        return {"path": str(temp_path), "name": temp_path.name}

    @app.get("/api/voice-work/generations")
    def get_voice_work_generations() -> list[dict[str, Any]]:
        return list_generations()

    @app.patch("/api/voice-work/voices/{voice_id}")
    def update_voice_work_voice(
        voice_id: str,
        label: str | None = Form(None),
        language: str | None = Form(None),
        description: str | None = Form(None),
    ) -> dict[str, Any]:
        try:
            updates: dict[str, Any] = {}
            if label is not None:
                updates["label"] = label.strip()
            if language is not None:
                updates["language"] = language.strip() or "auto"
            if description is not None:
                updates["description"] = description.strip()
            voice = update_voice(voice_id, updates)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _sync_local_library_index()
        return {"voice": voice}

    @app.delete("/api/voice-work/voices/{voice_id}")
    def delete_voice_work_voice(voice_id: str) -> dict[str, Any]:
        try:
            delete_voice(voice_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _sync_local_library_index()
        return {"deleted": True, "voice_id": voice_id}

    @app.post("/api/voice-work/voices/{voice_id}/train")
    def train_voice_work_voice(voice_id: str, request: VoiceWorkTrainRequest) -> dict[str, Any]:
        raise HTTPException(status_code=410, detail="Voice training has been removed. Use target voice uploads instead.")

    @app.post("/api/voice-work/voices/{voice_id}/convert")
    def convert_voice_work_sample(voice_id: str, request: VoiceWorkConvertRequest) -> dict[str, Any]:
        request_id = (request.request_id or uuid4().hex[:12]).strip()
        ui_log.add(
            "info",
            "Voice Work convert request "
            f"{request_id}: voice={voice_id}, mode={request.mode}, source={request.source_audio_path}, label={request.label or ''}",
        )

        def _runner() -> str:
            source_path = Path(request.source_audio_path).expanduser()
            if not request.label.strip():
                raise RuntimeError("Enter an output label before converting.")
            result = _voice_work_convert_audio(
                voice_id=voice_id,
                source_audio_path=request.source_audio_path,
                label=request.label or "",
                mode=request.mode,
                diffusion_steps=request.diffusion_steps,
                length_adjust=request.length_adjust,
                inference_cfg_rate=request.inference_cfg_rate,
                runtime_config=_current_voice_runtime_config(),
                request_id=request_id,
                render_type="conversion",
            )
            if _voice_work_root() in source_path.parents:
                shutil.rmtree(source_path.parent, ignore_errors=True)
            _sync_local_library_index()
            return f"Seed-VC conversion saved as '{result['label']}'."

        if not _start_voice_work_job(
            "convert",
            _runner,
            ui_log=ui_log,
            details={
                "request_id": request_id,
                "voice_id": voice_id,
                "source_audio_path": request.source_audio_path,
                "label": request.label or "",
                "mode": request.mode,
            },
        ):
            raise HTTPException(status_code=409, detail="Another Voice Work job is already in progress.")
        return {"started": True, "action": "convert", "voice_id": voice_id, "request_id": request_id}

    @app.post("/api/voice-work/voices/{voice_id}/tts")
    def generate_voice_work_tts(voice_id: str, request: VoiceWorkTtsRequest) -> dict[str, Any]:
        raise HTTPException(status_code=410, detail="TTS has been removed from Voice Work. Use sample conversion only.")

    @app.get("/api/library/file")
    def get_library_file(path: str = Query(..., min_length=1)) -> FileResponse:
        file_path = Path(path).expanduser()
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"Library file not found: {file_path}")
        return FileResponse(file_path)

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

    @app.get("/api/rhythm-beats/projects")
    def get_rhythm_beat_projects() -> list[dict[str, Any]]:
        return list_rhythm_projects()

    @app.get("/api/rhythm-beats/volumes")
    def get_rhythm_beat_volumes() -> dict[str, Any]:
        return {"volumes": list_rhythm_volumes()}

    @app.post("/api/rhythm-beats/volumes")
    def save_rhythm_beat_volume(request: RhythmBeatVolumeUpsertRequest) -> dict[str, Any]:
        volume = {
            "volume_id": request.volume_id,
            "label": request.label,
            "description": request.description,
            "sort_order": request.sort_order,
        }
        try:
            volumes = upsert_rhythm_volume(volume)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ui_log.add("info", f"Saved rhythm-game volume '{request.label}'.")
        return {"volumes": volumes}

    @app.delete("/api/rhythm-beats/volumes/{volume_id}")
    def delete_rhythm_beat_volume(volume_id: str) -> dict[str, Any]:
        try:
            volumes = remove_rhythm_volume(volume_id)
            for summary in list_rhythm_projects():
                project_id = str(summary.get("project_id") or "")
                if not project_id:
                    continue
                project = read_rhythm_project(project_id)
                game_asset = dict(project.get("game_asset") or {})
                if str(game_asset.get("volume_id") or "") != volume_id:
                    continue
                game_asset.update(
                    {
                        "volume_id": "",
                        "volume_label": "",
                        "volume_slug": "",
                        "official_volume": False,
                        "sort_order": 0,
                        "supported_game_modes": dict(
                            (game_asset.get("supported_game_modes") or {})
                        ),
                    }
                )
                project["game_asset"] = game_asset
                write_rhythm_project(project)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        ui_log.add("info", f"Removed rhythm-game volume '{volume_id}'.")
        return {"volumes": volumes}

    @app.get("/api/rhythm-beats/projects/{project_id}")
    def get_rhythm_beat_project(project_id: str) -> dict[str, Any]:
        try:
            return read_rhythm_project(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rhythm-beats/projects")
    def create_rhythm_beat_project_route(request: RhythmBeatProjectCreateRequest) -> dict[str, Any]:
        project = write_rhythm_project(create_rhythm_project(request.label))
        _sync_local_library_index()
        ui_log.add("info", f"Created rhythm beat project: {project['label']}")
        return {"project": project}

    @app.patch("/api/rhythm-beats/projects/{project_id}")
    def save_rhythm_beat_project(project_id: str, request: RhythmBeatProjectSaveRequest) -> dict[str, Any]:
        try:
            project = dict(request.project or {})
            project["project_id"] = safe_rhythm_project_id(project_id)
            saved = write_rhythm_project(project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        return {"project": saved}

    @app.patch("/api/rhythm-beats/projects/{project_id}/game-asset")
    def update_rhythm_game_asset_settings(project_id: str, request: RhythmBeatGameAssetSettingsRequest) -> dict[str, Any]:
        try:
            project = read_rhythm_project(project_id)
            selected_volume_id = str(request.volume_id or "").strip()
            volumes = list_rhythm_volumes()
            selected_volume = next((volume for volume in volumes if str(volume.get("volume_id") or "") == selected_volume_id), None)
            if selected_volume and bool(selected_volume.get("official", False)):
                raise ValueError("The official Faceless volume can only be managed from the official admin workflow.")
            project["game_asset"] = {
                "game_enabled": request.game_enabled,
                "volume_id": selected_volume_id if selected_volume else "",
                "volume_label": str(selected_volume.get("label") or "") if selected_volume else "",
                "volume_slug": str(selected_volume.get("slug") or "") if selected_volume else "",
                "official_volume": bool(selected_volume.get("official", False)) if selected_volume else False,
                "sort_order": int(selected_volume.get("sort_order") or 0) if selected_volume else 0,
                "supported_game_modes": {
                    "step_arrows": request.step_arrows_enabled,
                    "orb_beat": request.orb_beat_enabled,
                    "laser_shoot": request.step_arrows_enabled,
                },
            }
            saved = write_rhythm_project(project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        ui_log.add("info", f"Updated rhythm-game asset settings for '{saved['label']}'.")
        return {"project": saved, "volumes": volumes}

    @app.post("/api/rhythm-beats/projects/{project_id}/source/upload")
    def upload_rhythm_beat_source(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        original_name = Path(file.filename or "source").name
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name).strip("._") or "source"
        suffix = Path(safe_name).suffix.lower()
        temp_path = Path("data/rhythm-beats/.uploads") / f"{Path(safe_name).stem}-{uuid4().hex[:8]}{suffix}"
        try:
            validate_supported_source(temp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            probe = probe_audio(temp_path)
            project = read_rhythm_project(project_id)
            source = copy_uploaded_source_audio(project_id, temp_path)
            source["label"] = original_name
            source["duration_seconds"] = probe.duration_seconds
            project["source"] = source
            saved = write_rhythm_project(project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            file.file.close()
        _sync_local_library_index()
        ui_log.add("info", f"Attached uploaded source audio to rhythm beat project '{saved['label']}'.")
        return {"project": saved}

    @app.post("/api/rhythm-beats/projects/{project_id}/source/asset")
    def set_rhythm_beat_source_asset(project_id: str, request: RhythmBeatProjectAssetRequest) -> dict[str, Any]:
        try:
            asset = _find_editor_asset(request.asset_id)
            audio_path = Path(str(asset.get("audio_path") or "")).expanduser()
            if not audio_path.exists() or not audio_path.is_file():
                raise FileNotFoundError(f"Asset audio not found: {audio_path}")
            project = read_rhythm_project(project_id)
            probe = probe_audio(audio_path)
            project["source"] = {
                "label": str(asset.get("label") or audio_path.stem),
                "audio_path": str(audio_path),
                "audio_url": f"/api/audio?path={quote(str(audio_path))}",
                "duration_seconds": probe.duration_seconds,
                "source_asset_id": str(asset.get("asset_id") or ""),
                "source_category": str(asset.get("category") or ""),
            }
            saved = write_rhythm_project(project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        ui_log.add("info", f"Attached source asset to rhythm beat project '{saved['label']}'.")
        return {"project": saved}

    @app.post("/api/rhythm-beats/projects/{project_id}/tracks/asset")
    def add_rhythm_beat_track_asset(project_id: str, request: RhythmBeatProjectAssetRequest) -> dict[str, Any]:
        try:
            asset = _find_editor_asset(request.asset_id)
            audio_path = Path(str(asset.get("audio_path") or "")).expanduser()
            if not audio_path.exists() or not audio_path.is_file():
                raise FileNotFoundError(f"Asset audio not found: {audio_path}")
            probe = probe_audio(audio_path)
            project = read_rhythm_project(project_id)
            tracks = list(project.get("tracks") or [])
            tracks.append(
                {
                    "track_id": f"track-{uuid4().hex[:10]}",
                    "label": str(asset.get("label") or audio_path.stem),
                    "audio_path": str(audio_path),
                    "audio_url": f"/api/audio?path={quote(str(audio_path))}",
                    "duration_seconds": probe.duration_seconds,
                    "source_asset_id": str(asset.get("asset_id") or ""),
                    "source_category": str(asset.get("category") or ""),
                    "created_at": str(project.get("updated_at") or ""),
                }
            )
            project["tracks"] = tracks
            saved = write_rhythm_project(project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        ui_log.add("info", f"Added linked track to rhythm beat project '{saved['label']}'.")
        return {"project": saved}

    @app.post("/api/rhythm-beats/projects/{project_id}/extract-track")
    def extract_rhythm_beat_track(project_id: str, request: RhythmBeatExtractionRequest) -> dict[str, Any]:
        try:
            project = read_rhythm_project(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        source = dict(project.get("source") or {})
        source_path = Path(str(source.get("audio_path") or "")).expanduser()
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(status_code=400, detail="Attach a source song before extracting tracks.")
        label = request.label.strip() if request.label else _default_rhythm_track_label(source_path, request.track_name)
        metadata = _run_extraction_job(
            runtime_config=runtime_config,
            ui_log=ui_log,
            source_path=source_path,
            track_name=request.track_name.strip().lower(),
            label=label,
            output_format=request.output_format,
            inference_steps=request.inference_steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
            infer_method=request.infer_method,
            use_tiled_decode=request.use_tiled_decode,
            dcw_enabled=request.dcw_enabled,
            velocity_norm_threshold=request.velocity_norm_threshold,
            velocity_ema_factor=request.velocity_ema_factor,
            seed=request.seed,
            instruction=request.instruction,
            rhythm_project_id=project_id if request.attach_to_project else None,
        )
        if request.attach_to_project and metadata.get("status") == "complete":
            project = _attach_extraction_to_rhythm_project(project_id=project_id, metadata=metadata)
            return {"extraction": metadata, "project": project}
        _sync_local_library_index()
        return {"extraction": metadata}

    @app.post("/api/rhythm-beats/projects/{project_id}/lyrics/extract")
    def extract_rhythm_beat_lyrics(project_id: str, request: RhythmBeatLyricsExtractionRequest) -> dict[str, Any]:
        try:
            project = read_rhythm_project(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        source = dict(project.get("source") or {})
        source_path = Path(str(source.get("audio_path") or "")).expanduser()
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(status_code=400, detail="Attach a source song before extracting lyrics.")
        ui_log.add("info", f"Running lyrics extraction for rhythm beat project '{project.get('label')}'.")
        try:
            lyrics = _extract_lyrics_from_audio(
                audio_path=source_path,
                model_name=request.model.strip(),
                language=request.language,
                include_word_timestamps=not request.disable_word_timestamps,
            )
        except Exception as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        project["lyrics"] = lyrics
        saved = write_rhythm_project(project)
        _sync_local_library_index()
        ui_log.add("info", f"Extracted lyrics for rhythm beat project '{saved['label']}'.")
        return {"project": saved, "lyrics": lyrics}

    @app.delete("/api/rhythm-beats/projects/{project_id}/tracks/{track_id}")
    def remove_rhythm_beat_track(project_id: str, track_id: str) -> dict[str, Any]:
        try:
            project = read_rhythm_project(project_id)
            before = len(project.get("tracks") or [])
            project["tracks"] = [track for track in project.get("tracks") or [] if str(track.get("track_id") or "") != track_id]
            if len(project["tracks"]) == before:
                raise FileNotFoundError(f"Rhythm beat track not found: {track_id}")
            saved = write_rhythm_project(project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        ui_log.add("info", f"Removed linked track from rhythm beat project '{saved['label']}'.")
        return {"project": saved}

    @app.delete("/api/rhythm-beats/projects/{project_id}/selections/{selection_id}")
    def remove_rhythm_beat_selection(project_id: str, selection_id: str) -> dict[str, Any]:
        try:
            project = read_rhythm_project(project_id)
            before = len(project.get("selections") or [])
            project["selections"] = [
                selection for selection in project.get("selections") or []
                if str(selection.get("selection_id") or "") != selection_id
            ]
            if len(project["selections"]) == before:
                raise FileNotFoundError(f"Rhythm beat selection not found: {selection_id}")
            saved = write_rhythm_project(project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _sync_local_library_index()
        ui_log.add("info", f"Removed saved selection from rhythm beat project '{saved['label']}'.")
        return {"project": saved}

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

    @app.post("/api/library/local/{item_id}/cover")
    def set_local_library_cover(item_id: str, file: UploadFile = File(...)) -> dict[str, object]:
        content_type = str(file.content_type or "").lower()
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Card image must be an image file.")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise HTTPException(status_code=400, detail="Supported card image formats: PNG, JPG, JPEG, WEBP, GIF.")
        tmp_dir = Path("data/library/.uploads")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{uuid4().hex}{suffix}"
        try:
            tmp_path.write_bytes(file.file.read())
            item = _local_library().set_cover_image(item_id, tmp_path, mime_type=content_type or "application/octet-stream")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        ui_log.add("info", f"Updated card image for local library item: {item.title}")
        return {"item": _library_item_response(item)}

    @app.get("/api/library/publish/connection")
    def get_public_library_connection() -> dict[str, Any]:
        settings = load_publish_settings()
        if settings.authenticated and settings.refresh_token:
            try:
                refreshed = refresh_site_session(settings)
                save_publish_settings(refreshed)
                settings = refreshed
            except LibraryPublishError as exc:
                if is_expired_session_error(str(exc)):
                    settings = settings.cleared_session()
                    save_publish_settings(settings)
                ui_log.add("warn", f"Public library session refresh failed: {exc}")
        return public_settings_response(settings)

    @app.post("/api/library/publish/connection")
    def save_public_library_connection(request: PublicLibraryConnectionRequest) -> dict[str, Any]:
        existing = load_publish_settings()
        site_url = DEFAULT_PUBLIC_LIBRARY_SITE_URL
        settings = existing if existing.site_url.rstrip("/") == site_url else LibraryPublishSettings(site_url=site_url)
        settings.site_url = site_url
        save_publish_settings(settings)
        ui_log.add("info", f"Using public library connection for {settings.site_url}.")
        return public_settings_response(settings)

    @app.post("/api/library/publish/auth/nonce")
    def create_public_library_auth_nonce(request: PublicLibraryAuthNonceRequest) -> dict[str, Any]:
        settings = load_publish_settings()
        try:
            return request_wallet_nonce(settings, request.public_key.strip())
        except LibraryPublishError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/library/publish/auth/verify")
    def verify_public_library_auth(request: PublicLibraryAuthVerifyRequest) -> dict[str, Any]:
        settings = load_publish_settings()
        try:
            signature_bytes = bytes(request.signature)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Signature payload is invalid.") from exc
        try:
            authenticated = authenticate_wallet_signature(
                settings,
                public_key=request.public_key.strip(),
                nonce=request.nonce,
                message=request.message,
                signature_bytes=signature_bytes,
            )
        except LibraryPublishError as exc:
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_publish_settings(authenticated)
        display_name = (authenticated.creator_profile or {}).get("displayName") or authenticated.public_key[:10]
        ui_log.add("info", f"Connected public library wallet for {display_name}.")
        return public_settings_response(authenticated)

    @app.post("/api/library/publish/auth/logout")
    def logout_public_library_auth() -> dict[str, Any]:
        settings = load_publish_settings()
        cleared = logout_site_session(settings)
        save_publish_settings(cleared)
        ui_log.add("info", "Disconnected public library wallet session.")
        return public_settings_response(cleared)

    @app.post("/api/library/local/{item_id}/publish")
    def publish_local_library_item(item_id: str, request: PublicLibraryPublishRequest) -> dict[str, Any]:
        library = _local_library()
        item = library.read_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Library item not found: {item_id}")

        try:
            publisher = LibraryPublisher(load_publish_settings())
            publish_result = publisher.publish(
                item,
                publish_public=request.publish_public,
            )
            save_publish_settings(publisher.settings)
            updated = library.update_publish_metadata(item_id, publish_result)
        except (FileNotFoundError, LibraryPublishError) as exc:
            if isinstance(exc, LibraryPublishError) and is_expired_session_error(str(exc)):
                save_publish_settings(load_publish_settings().cleared_session())
                message = "Public library session expired. Connect your wallet again and retry publishing."
                ui_log.add("error", message)
                raise HTTPException(status_code=400, detail=message) from exc
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        ui_log.add("info", f"Published local library item '{updated.title}' to the public library.")
        return {"item": _library_item_response(updated), "publish": publish_result}

    @app.post("/api/library/local/{item_id}/revoke")
    def revoke_local_library_item(item_id: str) -> dict[str, Any]:
        library = _local_library()
        item = library.read_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Library item not found: {item_id}")

        try:
            publisher = LibraryPublisher(load_publish_settings())
            revoke_result = publisher.revoke(item)
            save_publish_settings(publisher.settings)
            updated = library.update_publish_metadata(item_id, revoke_result)
        except (FileNotFoundError, LibraryPublishError) as exc:
            if isinstance(exc, LibraryPublishError) and is_expired_session_error(str(exc)):
                save_publish_settings(load_publish_settings().cleared_session())
                message = "Public library session expired. Connect your wallet again and retry revoking."
                ui_log.add("error", message)
                raise HTTPException(status_code=400, detail=message) from exc
            ui_log.add("error", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        ui_log.add("info", f"Revoked public library item for '{updated.title}'.")
        return {"item": _library_item_response(updated), "publish": revoke_result}

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

    @app.get("/api/lokr/dataset-sources")
    def list_lokr_dataset_sources() -> list[dict[str, Any]]:
        return _dataset_sources()

    @app.get("/api/lokr/dataset-sources/{source_id:path}")
    def get_lokr_dataset_source(source_id: str) -> dict[str, Any]:
        try:
            return _dataset_source_detail(source_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @app.post("/api/lokr/datasets/import-json")
    def import_lokr_dataset_json(
        file: UploadFile = File(...),
        label: str = Form(""),
    ) -> dict[str, Any]:
        try:
            payload = json.loads(file.file.read().decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Dataset JSON could not be parsed.") from exc
        dataset_id = f"lokr-{uuid4().hex[:12]}"
        imported = _dataset_from_import_payload(dataset_id=dataset_id, label=label, payload=payload)
        saved = _write_lokr_dataset(imported, dataset_id=dataset_id)
        ui_log.add("info", f"Imported LoKr dataset JSON into {saved['metadata']['label']}")
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

    @app.post("/api/lokr/datasets/{dataset_id}/import-json")
    def append_lokr_dataset_json(
        dataset_id: str,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        try:
            payload = json.loads(file.file.read().decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Dataset JSON could not be parsed.") from exc
        imported = _dataset_from_import_payload(
            dataset_id=dataset_id,
            label=str((dataset.get("metadata") or {}).get("label") or ""),
            payload=payload,
            existing=dataset,
        )
        saved = _write_lokr_dataset(imported, dataset_id=dataset_id)
        ui_log.add("info", f"Appended JSON entries into LoKr dataset: {saved['metadata']['label']}")
        return {"dataset": _lokr_dataset_for_response(saved)}

    @app.post("/api/lokr/datasets/{dataset_id}/entries/empty")
    def create_empty_lokr_dataset_entry(dataset_id: str) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        metadata = dataset.get("metadata", {})
        sample = _lokr_empty_sample(
            dataset_id,
            default_genre=str(metadata.get("default_genre") or ""),
            default_language=str(metadata.get("default_language") or "unknown"),
        )
        dataset["samples"].append(sample)
        saved = _write_lokr_dataset(dataset, dataset_id=dataset_id)
        ui_log.add("info", f"Added empty LoKr dataset entry: {sample['id']}")
        return {"dataset": _lokr_dataset_for_response(saved), "sample": sample}

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

    @app.post("/api/lokr/datasets/{dataset_id}/entries/{entry_id}/upload")
    def attach_upload_to_lokr_dataset_entry(dataset_id: str, entry_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        original_name = Path(file.filename or "sample.wav").name
        upload_dir = _lokr_dataset_audio_dir(dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / f"upload-{uuid4().hex[:8]}{Path(original_name).suffix.lower()}"
        try:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            sample = _lokr_attach_audio_to_entry(
                dataset,
                dataset_id=dataset_id,
                entry_id=entry_id,
                source_path=temp_path,
                label=Path(original_name).stem,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
        saved = _write_lokr_dataset(dataset, dataset_id=dataset_id)
        ui_log.add("info", f"Attached upload to LoKr dataset entry: {entry_id}")
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

    @app.post("/api/lokr/datasets/{dataset_id}/entries/attach-asset")
    def attach_asset_to_lokr_dataset_entry(dataset_id: str, request: LokrDatasetEntryAssetRequest) -> dict[str, Any]:
        dataset = _read_lokr_dataset(dataset_id)
        asset = next((item for item in _editor_assets() if item.get("asset_id") == request.asset_id), None)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Creation not found: {request.asset_id}")
        audio_path = Path(str(asset.get("audio_path") or "")).expanduser()
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"Creation audio not found: {audio_path}")
        try:
            sample = _lokr_attach_audio_to_entry(
                dataset,
                dataset_id=dataset_id,
                entry_id=request.entry_id,
                source_path=audio_path,
                label=str(asset.get("label") or audio_path.stem),
                source_asset_id=str(asset.get("asset_id") or ""),
                source_category=str(asset.get("category") or ""),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        saved = _write_lokr_dataset(dataset, dataset_id=dataset_id)
        ui_log.add("info", f"Attached creation audio to LoKr dataset entry: {request.entry_id}")
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
        missing_audio = _lokr_missing_audio_entries(dataset)
        if missing_audio:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset has {len(missing_audio)} entr{'y' if len(missing_audio) == 1 else 'ies'} missing audio. Attach audio before preprocessing.",
            )
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
        missing_audio = _lokr_missing_audio_entries(dataset)
        if missing_audio:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset has {len(missing_audio)} entr{'y' if len(missing_audio) == 1 else 'ies'} missing audio. Attach audio before training.",
            )
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
            failure = _handle_ace_runtime_failure(runtime_config, ui_log, "ACE-Step music generation failed.", exc)
            metadata = {
                "generation_id": generation_id,
                "type": "music",
                "status": "recovering" if failure["recovery_active"] else "failed",
                "message": failure["message"],
                "created_at": created_at,
                "label": label,
                "prompt": prompt,
                "model": model,
                "output_format": request.output_format,
                "lokr_adapter": lokr_adapter,
                "lokr_scale": request.lokr_scale if lokr_adapter else None,
                "metadata_path": str(metadata_path),
                "settings": request.model_dump(),
                "runtime_recovery": failure["recovery"],
            }
            _write_metadata(metadata_path, metadata)
            return {"generation": metadata}

        metadata = {
            "generation_id": generation_id,
            "type": "music",
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
        _sync_local_library_index()
        ui_log.add("info", f"Generated music: {result.output_path}")
        return {"generation": metadata}

    @app.get("/api/sound-effects")
    def list_sound_effect_generations() -> list[dict[str, Any]]:
        root = _sound_effect_root()
        if not root.exists():
            return []
        items: list[dict[str, Any]] = []
        for metadata_path in (root / "generations").glob("*/generation.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(metadata)
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    @app.get("/api/sound-effects/runtime/status")
    def get_sound_effect_runtime_status() -> dict[str, object]:
        return sound_effect_runtime_status(runtime_config).to_dict()

    @app.post("/api/sound-effects/runtime/install")
    def install_sound_effect_runtime() -> dict[str, object]:
        run_sound_effect_runtime_install(runtime_config)
        return {"status": sound_effect_runtime_status(runtime_config).to_dict(), "message": "TangoFlux runtime setup complete."}

    @app.get("/api/sound-effects/audio")
    def get_sound_effect_audio(path: str = Query(..., min_length=1)) -> FileResponse:
        return get_audio_file(path)

    @app.post("/api/sound-effects/run")
    def run_sound_effect_generation(request: SoundEffectRequest) -> dict[str, object]:
        import datetime as _datetime

        generation_id = f"sound-effect-{uuid4().hex[:12]}"
        save_dir = _sound_effect_root() / "generations" / generation_id
        metadata_path = save_dir / "generation.json"
        created_at = _datetime.datetime.now(_datetime.UTC).isoformat()
        payload = request.model_dump()
        label = str(payload.get("label") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        duration_seconds = float(payload.get("duration_seconds") or 10)
        steps = int(payload.get("steps") or 50)
        output_format = str(payload.get("output_format") or "wav").strip().lower()
        if not label:
            raise HTTPException(status_code=400, detail="Enter a label for the sound effect.")
        if not prompt:
            raise HTTPException(status_code=400, detail="Enter a prompt for the sound effect.")
        if duration_seconds < 1 or duration_seconds > 30:
            raise HTTPException(status_code=400, detail="Sound effect duration must be between 1 and 30 seconds.")
        ui_log.add("info", f"Running TangoFlux sound effect generation: {label}.")
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            if not sound_effect_runtime_status(runtime_config).ready:
                run_sound_effect_runtime_install(runtime_config)
            wav_output = (save_dir / "sound_effect.wav").resolve()
            output_path = (save_dir / f"sound_effect{_sound_effect_output_extension(output_format)}").resolve()
            generated_wav = generate_sound_effect_wav(
                prompt,
                wav_output,
                steps=steps,
                duration_seconds=int(duration_seconds),
                config=runtime_config,
            )
            written_path = _sound_effect_transcode_output(generated_wav, output_path, output_format)
        except Exception as exc:
            ui_log.add("error", str(exc))
            metadata = {
                "generation_id": generation_id,
                "type": "sound_effect",
                "status": "failed",
                "message": str(exc),
                "created_at": created_at,
                "label": label,
                "prompt": prompt,
                "steps": steps,
                "duration_seconds": duration_seconds,
                "output_format": output_format,
                "generated_audio_path": "",
                "metadata_path": str(metadata_path),
                "settings": payload,
            }
            write_sound_effect_generation(metadata)
            return {"generation": metadata}

        metadata = {
            "generation_id": generation_id,
            "type": "sound_effect",
            "status": "complete",
            "message": "Sound effect generation complete.",
            "created_at": created_at,
            "label": label,
            "prompt": prompt,
            "model": "declare-lab/TangoFlux",
            "steps": steps,
            "duration_seconds": duration_seconds,
            "output_format": output_format,
            "generated_audio_path": str(written_path),
            "metadata_path": str(metadata_path),
            "settings": payload,
        }
        write_sound_effect_generation(metadata)
        _sync_local_library_index()
        ui_log.add("info", f"Generated sound effect audio: {written_path}")
        return {"generation": metadata}

    @app.post("/api/music-generations/vocal2bgm")
    def run_vocal2bgm_generation(request: Vocal2BgmRequest) -> dict[str, object]:
        raise HTTPException(status_code=410, detail="Vocal2BGM has been replaced by Sound Effects.")

    @app.post("/api/sound-effects/{generation_id}/rename")
    def rename_sound_effect_generation(generation_id: str, request: ExtractionRenameRequest) -> dict[str, object]:
        metadata_path = sound_effect_generation_path(generation_id)
        metadata = _read_json_file(metadata_path, "Sound effect generation")
        metadata["label"] = request.label.strip()
        write_sound_effect_generation(metadata)
        ui_log.add("info", f"Renamed sound effect generation {generation_id}: {metadata['label']}")
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
        source_path = Path(request.source_path).expanduser()
        metadata = _run_extraction_job(
            runtime_config=runtime_config,
            ui_log=ui_log,
            source_path=source_path,
            track_name=request.track_name.strip().lower(),
            label=request.label.strip() if request.label else request.track_name.strip().lower(),
            output_format=request.output_format,
            inference_steps=request.inference_steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
            infer_method=request.infer_method,
            use_tiled_decode=request.use_tiled_decode,
            dcw_enabled=request.dcw_enabled,
            velocity_norm_threshold=request.velocity_norm_threshold,
            velocity_ema_factor=request.velocity_ema_factor,
            seed=request.seed,
            instruction=request.instruction,
        )
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
            failure = _handle_ace_runtime_failure(runtime_config, ui_log, "ACE-Step Base text-to-music test failed.", exc)
            metadata = {
                "extraction_id": generation_id,
                "type": "base_test",
                "status": "recovering" if failure["recovery_active"] else "failed",
                "message": failure["message"],
                "created_at": created_at,
                "track_name": "Base text2music test",
                "prompt": prompt,
                "output_format": request.output_format,
                "metadata_path": str(metadata_path),
                "settings": request.model_dump(),
                "runtime_recovery": failure["recovery"],
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

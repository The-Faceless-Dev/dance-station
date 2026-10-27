"""Application service layer for the standalone dataset manager."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .config import DatasetManagerConfig
from .contracts import new_id, now_iso, validate_annotation, write_json
from .db import DatasetRepository
from .extraction import ExtractionRunner


class DatasetManagerService:
    def __init__(self, config: DatasetManagerConfig | None = None):
        self.config = config or DatasetManagerConfig.from_env()
        self.config.storage_root.mkdir(parents=True, exist_ok=True)
        self.config.model_root.mkdir(parents=True, exist_ok=True)
        self.repository = DatasetRepository(self.config.storage_root)
        self.artifacts = ArtifactStore(self.config.storage_root)
        self.runner = ExtractionRunner(self.repository, self.artifacts, self.config)
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="dataset-extract")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.RLock()
        self.repository.recover_running_runs()

    def close(self) -> None:
        # A clean shutdown must not leave a worker writing SQLite after the
        # process has released its service lifecycle.  Runs are bounded by the
        # provider timeouts and are recovered as failed on an unclean restart.
        self.executor.shutdown(wait=True, cancel_futures=False)

    def provider_status(self) -> dict[str, Any]:
        """Report configured providers without loading large model weights."""

        from autotransition.audio.ffmpeg import resolve_ffmpeg
        from autotransition.config import RuntimeConfig
        from autotransition.runtime.source_separation import runtime_status

        ffmpeg = resolve_ffmpeg()
        separation = runtime_status(RuntimeConfig(source_separation_dir=self.config.separator_runtime_dir)).to_dict()
        model_cache = Path(separation["install_dir"]) / "models"
        primary_cache = model_cache / self.config.separator_primary_model
        fallback_cache = model_cache / self.config.separator_fallback_model
        nfa_python = self.config.nfa_python
        nfa_runtime_ready = False
        nfa_runtime_error = ""
        if nfa_python:
            try:
                check = subprocess.run([nfa_python, "-c", "import nemo"], capture_output=True, text=True, timeout=30, check=False)
                nfa_runtime_ready = check.returncode == 0
                nfa_runtime_error = (check.stderr or check.stdout).strip()[-500:]
            except Exception as exc:
                nfa_runtime_error = str(exc)
        return {
            "ffmpeg": {"path": ffmpeg, "ready": bool(ffmpeg)},
            "source_separation": {**separation, "primary_model": self.config.separator_primary_model, "fallback_model": self.config.separator_fallback_model, "primary_model_cached": primary_cache.is_file() and primary_cache.stat().st_size > 0, "fallback_model_cached": fallback_cache.is_file() and fallback_cache.stat().st_size > 0, "download_timeout_seconds": self.config.separation_timeout_seconds},
            "sofa": {"configured": bool(self.config.sofa_command or (self.config.sofa_script and self.config.sofa_checkpoint and self.config.sofa_dictionary)), "script": str(self.config.sofa_script) if self.config.sofa_script else None, "checkpoint": str(self.config.sofa_checkpoint) if self.config.sofa_checkpoint else None, "dictionary": str(self.config.sofa_dictionary) if self.config.sofa_dictionary else None},
            "nfa": {"configured": bool(self.config.nfa_script and self.config.nfa_script.exists()), "runtime_ready": nfa_runtime_ready, "runtime_error": nfa_runtime_error, "script": str(self.config.nfa_script) if self.config.nfa_script else None, "model": self.config.nfa_model, "device": self.config.nfa_device, "python": self.config.nfa_python},
            "rmvpe": {"configured": bool(self.config.rmvpe_checkpoint and self.config.rmvpe_checkpoint.exists()), "checkpoint": str(self.config.rmvpe_checkpoint) if self.config.rmvpe_checkpoint else None, "device": self.config.rmvpe_device},
            "asr": {"provider": "faster-whisper", "model": self.config.asr_model, "device": self.config.asr_device},
        }

    def _progress(self, run_id: str):
        def callback(stage: str, status: str, progress: float, message: str) -> None:
            self.repository.upsert_stage(run_id, stage, status=status, progress=progress, message=message, completed_at=now_iso() if status in {"succeeded", "failed"} else None)
        return callback

    def upload_song(self, *, filename: str, source, title: str, artist: str, language: str, lyrics: str, vocals_only: bool, probe: dict[str, Any], source_sha256: str) -> dict[str, Any]:
        song_id = new_id("song")
        destination = self.artifacts.store_upload(song_id, filename, source)
        values = {
            "id": song_id,
            "source_filename": Path(filename).name,
            "title": title or Path(filename).stem,
            "artist": artist,
            "language": language or self.config.language,
            "lyrics": lyrics,
            "vocals_only": vocals_only,
            "duration_ms": probe.get("duration_ms", 0),
            "sample_rate": probe.get("sample_rate", 0),
            "channels": probe.get("channels", 0),
            "source_sha256": source_sha256,
        }
        try:
            return self.repository.create_song(values)
        except Exception:
            shutil.rmtree(self.artifacts.song_root(song_id), ignore_errors=True)
            raise

    def get_song(self, song_id: str) -> dict[str, Any] | None:
        song = self.repository.get_song(song_id)
        if not song:
            return None
        run = self.repository.get_run(song["active_run_id"]) if song.get("active_run_id") else None
        draft = self.repository.get_draft(song_id)
        revision = self.repository.get_revision(song["approved_revision_id"]) if song.get("approved_revision_id") else None
        return {**song, "run": run, "draft": draft, "approved_revision": revision}

    def update_lyrics(self, song_id: str, lyrics: str) -> dict[str, Any]:
        song = self.repository.get_song(song_id)
        if not song:
            raise KeyError(song_id)
        value = str(lyrics or "").strip()
        self.repository.delete_draft(song_id)
        return self.repository.update_song(
            song_id,
            lyrics=value,
            lyrics_source="user_provided" if value else "missing",
            status="New",
            last_error="",
            approved_revision_id=None,
        )

    def start_extraction(self, song_id: str) -> dict[str, Any]:
        with self._lock:
            song = self.repository.get_song(song_id)
            if not song:
                raise KeyError(song_id)
            if song.get("active_run_id"):
                active = self.repository.get_run(song["active_run_id"])
                if active and active["status"] in {"queued", "running"}:
                    return active
            run = self.repository.create_run(song_id, {"language": song["language"], "primary_separator": self.config.separator_primary_model, "fallback_separator": self.config.separator_fallback_model, "alignment": "sofa->nfa", "features": "rmvpe_or_parselmouth_equivalent"})
            future = self.executor.submit(self.runner.run, song_id, run["id"], self._progress(run["id"]))
            self._futures[run["id"]] = future
            return self.repository.get_run(run["id"]) or run

    def update_draft(self, song_id: str, annotation: dict[str, Any]) -> dict[str, Any]:
        song = self.repository.get_song(song_id)
        if not song:
            raise KeyError(song_id)
        validation = validate_annotation(annotation, int(song["duration_ms"]))
        return self.repository.upsert_draft(song_id, annotation, validation)

    def approve(self, song_id: str) -> dict[str, Any]:
        song = self.repository.get_song(song_id)
        draft = self.repository.get_draft(song_id)
        if not song or not draft:
            raise ValueError("A draft annotation is required before saving a revision.")
        validation = validate_annotation(draft["annotation"], int(song["duration_ms"]))
        if not validation["valid"]:
            raise ValueError("Cannot save invalid annotation: " + "; ".join(validation["errors"]))
        revision_id = new_id("revision")
        revision_dir = self.artifacts.revision_dir(song_id, revision_id)
        annotation = dict(draft["annotation"])
        annotation["revision_id"] = revision_id
        annotation["quality"] = {**annotation.get("quality", {}), "approved": True, "validation": validation}
        write_json(revision_dir / "annotation.json", annotation)
        export = {"annotation": self.artifacts.relative(revision_dir / "annotation.json")}
        revision = self.repository.create_revision(song_id, annotation, validation, export)
        return revision

    def export_dataset(self, song_ids: list[str] | None = None) -> dict[str, Any]:
        selected = self.repository.list_songs()
        if song_ids:
            selected = [song for song in selected if song["id"] in set(song_ids)]
        revisions = []
        for song in selected:
            if song.get("approved_revision_id"):
                revision = self.repository.get_revision(song["approved_revision_id"])
                if revision:
                    revisions.append(revision)
        if not revisions:
            raise ValueError("No approved revisions are available for export.")
        export_id = new_id("export")
        export_root = self.artifacts.export_dir() / export_id
        export_root.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 1, "export_id": export_id, "created_at": now_iso(), "songs": []}
        for revision in revisions:
            song = self.repository.get_song(revision["song_id"])
            if not song:
                continue
            entry = {"song_id": song["id"], "revision_id": revision["id"], "title": song["title"], "source_filename": song["source_filename"], "duration_ms": song["duration_ms"], "annotation": revision["annotation"], "provenance": {"source_sha256": song["source_sha256"], "approved_at": revision["created_at"]}}
            manifest["songs"].append(entry)
        write_json(export_root / "manifest.json", manifest)
        result = self.repository.create_export([item["id"] for item in revisions], self.artifacts.relative(export_root / "manifest.json"), {"valid": True, "song_count": len(manifest["songs"])})
        result["manifest"] = manifest
        return result

    def audio_path(self, song_id: str, kind: str) -> Path:
        if kind == "original":
            candidates = list(self.artifacts.original_dir(song_id).glob("*"))
        elif kind == "vocals":
            candidates = [self.artifacts.normalized_dir(song_id) / "vocals.wav"]
        elif kind == "instrumental":
            candidates = [self.artifacts.normalized_dir(song_id) / "instrumental.wav"]
        else:
            raise KeyError(kind)
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        raise FileNotFoundError(f"No {kind} audio is available for {song_id}.")

    def waveform_path(self, song_id: str, kind: str) -> Path:
        run = self.repository.get_song(song_id)
        if not run or not run.get("active_run_id"):
            raise FileNotFoundError("No extraction waveform is available yet.")
        path = self.artifacts.run_dir(song_id, run["active_run_id"]) / ("waveform_original.npz" if kind == "original" else "waveform_vocals.npz")
        if not path.exists():
            raise FileNotFoundError(path)
        return path

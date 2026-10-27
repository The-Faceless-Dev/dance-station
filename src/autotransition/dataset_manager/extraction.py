"""Real extraction stages for the standalone dataset manager."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from autotransition.audio.ffmpeg import resolve_ffmpeg
from autotransition.config import RuntimeConfig
from autotransition.runtime.source_separation import separate_audio
from autotransition.vevo2_dataset.alignment import build_alignment, reconcile_words, validate_alignment
from autotransition.vevo2_dataset.contracts import ProviderUnavailable, write_json
from autotransition.vevo2_dataset.forced_alignment import _find_tier, _read_textgrid, run_nfa
from autotransition.vevo2_dataset.prosody import Vevo2ProsodyConfig, normalize_audio
from autotransition.vevo2_dataset.transcription import TranscriptionResult, transcribe

from .annotation import annotation_from_alignment
from .artifacts import ArtifactStore
from .config import DatasetManagerConfig
from .contracts import now_iso, sha256_file
from .db import DatasetRepository
from .features import extract_features, waveform_peaks


Progress = Callable[[str, str, float, str], None]


class ExtractionFailure(RuntimeError):
    """An extraction stage failed without invalidating prior artifacts."""


def _probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            streams = [item for item in payload.get("streams", []) if item.get("codec_type") == "audio"]
            if streams:
                stream = streams[0]
                duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
                return {"duration_ms": round(duration * 1000), "sample_rate": int(stream.get("sample_rate") or 0), "channels": int(stream.get("channels") or 0), "probe": payload}
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {"duration_ms": round(float(info.duration) * 1000), "sample_rate": int(info.samplerate), "channels": int(info.channels), "probe": {"fallback": "soundfile"}}
    except Exception:
        # imageio-ffmpeg ships ffmpeg but not always ffprobe.  Pydub still
        # gives us a reliable media probe for MP3/M4A/OGG in that setup.
        from autotransition.audio.ffmpeg import require_pydub

        segment = require_pydub("probe uploaded audio").from_file(str(path))
        return {"duration_ms": len(segment), "sample_rate": int(segment.frame_rate), "channels": int(segment.channels), "probe": {"fallback": "pydub-ffmpeg"}}


def _stage_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def _copy_first_matching(paths: list[Path], destination: Path, *, include: str | None = None, exclude: str | None = None) -> Path:
    candidates = [path for path in paths if path.exists() and (include is None or include in path.name.lower()) and (exclude is None or exclude not in path.name.lower())]
    if not candidates:
        raise ExtractionFailure("Source separation returned no usable audio stem.")
    source = sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _sofa_alignment(
    audio_path: Path,
    lyrics: str,
    output_dir: Path,
    config: DatasetManagerConfig,
) -> tuple[list[Any], dict[int, list[tuple[str, float, float]]], dict[str, Any]]:
    if not config.sofa_command and not (config.sofa_script and config.sofa_checkpoint and config.sofa_dictionary):
        raise ProviderUnavailable(
            "SOFA is not configured. Set DATASET_MANAGER_SOFA_SCRIPT, "
            "DATASET_MANAGER_SOFA_CHECKPOINT, and DATASET_MANAGER_SOFA_DICTIONARY, "
            "or provide DATASET_MANAGER_SOFA_COMMAND."
        )
    input_dir = output_dir / "sofa_input" / "dataset_manager"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_audio = input_dir / "utterance.wav"
    input_lab = input_dir / "utterance.lab"
    shutil.copy2(audio_path, input_audio)
    input_lab.write_text(lyrics.strip() + "\n", encoding="utf-8")
    if config.sofa_command:
        tokens = shlex.split(config.sofa_command, posix=False)
        substitutions = {
            "{audio}": str(input_audio),
            "{lab}": str(input_lab),
            "{folder}": str(input_dir.parent),
            "{output_dir}": str(output_dir),
            "{checkpoint}": str(config.sofa_checkpoint or ""),
            "{dictionary}": str(config.sofa_dictionary or ""),
        }
        command = []
        for value in tokens:
            for key, replacement in substitutions.items():
                value = value.replace(key, replacement)
            command.append(value)
    else:
        python_executable = config.sofa_python or sys.executable
        command = [
            python_executable,
            str(config.sofa_script),
            "--ckpt", str(config.sofa_checkpoint),
            "--folder", str(input_dir.parent),
            "--dictionary", str(config.sofa_dictionary),
            "--out_formats", "TextGrid",
        ]
    completed = subprocess.run(command, cwd=output_dir, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=1800)
    (output_dir / "sofa.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "sofa.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise ExtractionFailure(f"SOFA failed with exit code {completed.returncode}: {(completed.stderr or completed.stdout)[-2000:]}")
    textgrids = sorted(output_dir.rglob("*.TextGrid")) + sorted(output_dir.rglob("*.textgrid"))
    if not textgrids:
        raise ExtractionFailure("SOFA completed without producing a TextGrid artifact.")
    textgrid = textgrids[0]
    tiers = _read_textgrid(textgrid)
    word_intervals = _find_tier(tiers, ("words", "word"))
    phone_intervals = _find_tier(tiers, ("phones", "phone", "phonemes", "phoneme"))
    from autotransition.vevo2_dataset.transcription import WordObservation

    observations: list[WordObservation] = []
    phone_map: dict[int, list[tuple[str, float, float]]] = {}
    for interval in word_intervals:
        label = str(interval.get("text", "")).strip()
        if not label or label.lower() in {"sil", "sp", "spn", "pau"}:
            continue
        start = float(interval["start"])
        end = float(interval["end"])
        index = len(observations)
        observations.append(WordObservation(label, start, end, None, "sofa"))
        phone_map[index] = []
        for phone in phone_intervals:
            phone_label = str(phone.get("text", "")).strip()
            phone_start = float(phone.get("start", 0))
            phone_end = float(phone.get("end", 0))
            overlap = min(end, phone_end) - max(start, phone_start)
            if overlap > 0 and phone_label and phone_label.lower() not in {"sil", "sp", "spn", "pau"}:
                phone_map[index].append((phone_label, max(start, phone_start), min(end, phone_end)))
    if not observations:
        raise ExtractionFailure("SOFA TextGrid contained no non-silence word intervals.")
    copied_textgrid = output_dir / "sofa.TextGrid"
    if textgrid.resolve() != copied_textgrid.resolve():
        shutil.copy2(textgrid, copied_textgrid)
    return observations, phone_map, {"provider": "sofa", "textgrid": copied_textgrid.name, "command": command}


class ExtractionRunner:
    def __init__(self, repository: DatasetRepository, artifacts: ArtifactStore, config: DatasetManagerConfig):
        self.repository = repository
        self.artifacts = artifacts
        self.config = config

    def run(self, song_id: str, run_id: str, progress: Progress | None = None) -> dict[str, Any]:
        song = self.repository.get_song(song_id)
        if not song:
            raise KeyError(song_id)
        run_dir = self.artifacts.run_dir(song_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        warnings: list[str] = []
        artifacts: dict[str, Any] = {}
        provenance: dict[str, Any] = {"pipeline_version": "dataset-manager-v1", "started_at": now_iso()}

        def notify(stage: str, status: str, fraction: float, message: str = "") -> None:
            value = max(0.0, min(1.0, float(fraction)))
            _stage_log(log_path, f"stage={stage} status={status} progress={value:.3f} message={message}")
            self.repository.upsert_stage(run_id, stage, status=status, progress=value, message=message, started_at=now_iso() if status == "running" else None, completed_at=now_iso() if status in {"succeeded", "failed"} else None, error=message if status == "failed" else "")
            run_updates: dict[str, Any] = {"progress": value, "current_stage": stage}
            if status == "running":
                run_updates["status"] = "running"
            self.repository.update_run(run_id, **run_updates)
            if progress:
                progress(stage, status, value, message)

        try:
            source = next(self.artifacts.original_dir(song_id).glob("*"))
            notify("media_ingestion", "running", 0.02, "Probing source audio")
            probe = _probe(source)
            source_info = {key: value for key, value in probe.items() if key != "probe"}
            write_json(run_dir / "probe.json", probe)
            waveform_peaks(source, run_dir / "waveform_original.npz")
            notify("media_ingestion", "succeeded", 0.12, "Original media and waveform cache stored")

            notify("vocal_separation", "running", 0.12, "Preparing vocal stem")
            vocals_path = self.artifacts.normalized_dir(song_id) / "vocals.wav"
            instrumental_path = self.artifacts.normalized_dir(song_id) / "instrumental.wav"
            separation_provenance: dict[str, Any]
            if song["vocals_only"]:
                vocals_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, vocals_path)
                separation_provenance = {"provider": "user_supplied_vocals_only", "model": None}
            else:
                separation_errors: list[str] = []
                result = None
                for model, label in ((self.config.separator_primary_model, "mel_band_roformer"), (self.config.separator_fallback_model, "demucs_htdemucs_ft")):
                    try:
                        separation_dir = run_dir / "separation" / label
                        result = separate_audio(source_path=source, output_dir=separation_dir, model_filename=model, output_format="wav", config=RuntimeConfig(source_separation_dir=self.config.separator_runtime_dir, generation_timeout_seconds=self.config.separation_timeout_seconds))
                        paths = [Path(item) for item in result["audio_files"]]
                        _copy_first_matching(paths, vocals_path, include="vocal")
                        instrumental_candidates = [path for path in paths if "vocal" not in path.name.lower() and "instrument" in path.name.lower()]
                        if instrumental_candidates:
                            shutil.copy2(instrumental_candidates[0], instrumental_path)
                        separation_provenance = {
                            "provider": "audio-separator",
                            "model": model,
                            "label": label,
                            "device": result.get("device"),
                            "torch_version": result.get("torch_version"),
                            "torch_cuda_version": result.get("torch_cuda_version"),
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                        }
                        if label != "mel_band_roformer":
                            warnings.append("Primary Mel-Band RoFormer failed; Demucs htdemucs_ft fallback was used.")
                        break
                    except Exception as exc:
                        separation_errors.append(f"{model}: {exc}")
                else:
                    raise ExtractionFailure("Vocal separation failed: " + " | ".join(separation_errors))
            shutil.copy2(vocals_path, run_dir / "vocals.wav")
            waveform_peaks(vocals_path, run_dir / "waveform_vocals.npz")
            artifacts["vocals"] = self.artifacts.describe(vocals_path)
            if instrumental_path.exists():
                artifacts["instrumental"] = self.artifacts.describe(instrumental_path)
            provenance["separation"] = separation_provenance
            notify("vocal_separation", "succeeded", 0.32, f"Vocal stem ready via {separation_provenance['provider']}")

            # NeMo's acoustic model consumes a mono [batch, time] waveform.
            # Keep the separated stem at its original analysis format, but
            # provide alignment providers with their explicit 16 kHz mono
            # input contract rather than relying on implicit channel handling.
            alignment_audio = run_dir / "alignment" / "audio_16k_mono.wav"
            normalize_audio(
                vocals_path,
                alignment_audio,
                Vevo2ProsodyConfig(sample_rate=16_000),
            )

            notify("lyrics", "running", 0.32, "Loading supplied lyrics or producing a provisional transcript")
            lyrics = str(song["lyrics"] or "").strip()
            if lyrics:
                transcription = TranscriptionResult(lyrics, [], "provided", song["language"], ["Supplied lyrics are authoritative text; timing comes from forced alignment."])
            else:
                transcription = transcribe(vocals_path, "faster-whisper", self.config.asr_model, song["language"], self.config.asr_device, self.config.asr_compute_type)
                lyrics = transcription.text
                self.repository.update_song(song_id, lyrics=lyrics, lyrics_source="asr_provisional")
                warnings.extend(transcription.warnings)
            write_json(run_dir / "transcript.json", transcription.to_dict())
            provenance["lyrics"] = {"provider": transcription.provider, "source": "user_provided" if song["lyrics"] else "faster-whisper_provisional"}
            notify("lyrics", "succeeded", 0.42, f"Lyrics source: {transcription.provider}")

            notify("alignment", "running", 0.42, "Running SOFA against current lyrics")
            exact_phones = None
            observed_words = []
            alignment_provider = "sofa"
            provider_label = "sofa_word_phone_timestamps"
            try:
                observed_words, exact_phones, align_provenance = _sofa_alignment(alignment_audio, lyrics, run_dir / "alignment", self.config)
                provenance["alignment"] = align_provenance
            except Exception as sofa_error:
                warnings.append(f"SOFA unavailable or unusable: {sofa_error}")
                alignment_provider = "nfa"
                provider_label = "nfa_word_timestamps"
                if not self.config.nfa_script:
                    raise ExtractionFailure(f"SOFA failed and NeMo fallback is not configured: {sofa_error}")
                observed_words = run_nfa(alignment_audio, lyrics, run_dir / "alignment" / "nfa", script=self.config.nfa_script, model_name=self.config.nfa_model, python_executable=self.config.nfa_python or sys.executable, device=self.config.nfa_device)
                provenance["alignment"] = {"provider": "nfa", "model": self.config.nfa_model, "device": self.config.nfa_device, "fallback_from": str(sofa_error)}
                warnings.append("NeMo provided word timings only; phoneme labels are G2P estimates and explicitly unaligned.")
            duration = max(float(song["duration_ms"]) / 1000.0, 0.001)
            words, reconcile_warnings = reconcile_words(lyrics, observed_words, duration)
            warnings.extend(reconcile_warnings)
            alignment = build_alignment(words, "g2p_en", alignment_provider, exact_phones=exact_phones, provider_label=provider_label)
            validate_alignment(alignment, duration)
            write_json(run_dir / "alignment.json", alignment.to_dict())
            notify("alignment", "succeeded", 0.62, f"{alignment.provider}; phone quality={alignment.quality}")

            notify("melodic_features", "running", 0.62, "Extracting F0, voicing, energy, and waveform features")
            feature_info = extract_features(vocals_path, run_dir / "features", sample_rate=self.config.sample_rate, rmvpe_checkpoint=self.config.rmvpe_checkpoint, rmvpe_device=self.config.rmvpe_device)
            if feature_info.get("warning"):
                warnings.append(str(feature_info["warning"]))
            provenance["melodic_features"] = feature_info
            notify("melodic_features", "succeeded", 0.78, f"F0 provider={feature_info['provider']}")

            annotation = annotation_from_alignment(alignment, song_id=song_id, duration_ms=int(song["duration_ms"]), lyrics=lyrics, language=song["language"], feature_paths={"f0": self.artifacts.relative(run_dir / "features" / "f0_hz.npy"), "voiced": self.artifacts.relative(run_dir / "features" / "voiced.npy"), "energy": self.artifacts.relative(run_dir / "features" / "energy_rms_100hz.npy")})
            annotation["quality"]["warnings"].extend(warnings)
            write_json(run_dir / "annotation_machine.json", annotation)
            artifacts["run_directory"] = self.artifacts.relative(run_dir)
            artifacts["annotation"] = self.artifacts.relative(run_dir / "annotation_machine.json")
            provenance["completed_at"] = now_iso()
            provenance["source_sha256"] = song["source_sha256"]
            run = self.repository.update_run(run_id, status="succeeded", progress=1.0, current_stage="complete", artifacts=artifacts, provenance=provenance, warnings=warnings, completed_at=now_iso(), error="")
            self.repository.upsert_stage(run_id, "complete", status="succeeded", progress=1.0, message="Extraction complete", completed_at=now_iso(), artifacts=artifacts)
            self.repository.upsert_draft(song_id, annotation, {"valid": True, "errors": [], "warnings": warnings, "word_count": len(annotation["words"]), "phoneme_count": sum(len(item["phonemes"]) for item in annotation["words"])})
            self.repository.update_song(song_id, status="Ready for Review", last_error="", has_unsaved_changes=1)
            write_json(run_dir / "metadata.json", {"run_id": run_id, "song_id": song_id, "source": source_info, "provenance": provenance, "warnings": warnings, "artifacts": artifacts})
            return run
        except Exception as exc:
            message = str(exc)
            _stage_log(log_path, f"run status=failed error={message}")
            previous_run = self.repository.get_run(run_id) or {}
            current_stage = previous_run.get("current_stage") or "unknown"
            run = self.repository.update_run(run_id, status="failed", error=message, current_stage=current_stage, warnings=warnings)
            self.repository.update_song(song_id, status="Failed", last_error=message)
            self.repository.upsert_stage(run_id, current_stage, status="failed", progress=run.get("progress", 0), message=message, error=message, completed_at=now_iso())
            raise

"""Orchestration for one inspectable Vevo2 dataset-enrichment run."""

from __future__ import annotations

import time
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .alignment import build_alignment, reconcile_words, validate_alignment
from .contracts import SCHEMA_VERSION, write_json
from .forced_alignment import run_mfa, run_nfa
from .prosody import Vevo2ProsodyConfig, extract_prosody, normalize_audio
from .transcription import TranscriptionResult, transcribe


@dataclass
class EnrichmentConfig:
    asr_provider: str = "parakeet"
    asr_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    asr_language: str = "en"
    asr_device: str = "auto"
    asr_compute_type: str = "int8"
    alignment_provider: str = "nfa+mfa"
    phoneme_provider: str = "g2p_en"
    nfa_script: Path | None = None
    nfa_model: str = "stt_en_fastconformer_hybrid_large_pc"
    nfa_python: str = ""
    mfa_binary: str = "mfa"
    mfa_dictionary: Path | None = None
    mfa_acoustic_model: Path | None = None
    include_f0_diagnostic: bool = True
    prosody: Vevo2ProsodyConfig = field(default_factory=Vevo2ProsodyConfig)


def _log(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{timestamp}] {message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _provided_transcription(text: str, language: str) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        words=[],
        provider="provided",
        language=language,
        warnings=["Transcript text was provided by the caller; timing still requires an alignment provider."],
    )


def _write_manifest(run_dir: Path) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(run_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    write_json(run_dir / "manifest.json", {"schema_version": SCHEMA_VERSION, "files": files})


def enrich_dataset(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    config: EnrichmentConfig | None = None,
    transcript: str | None = None,
    transcript_file: str | Path | None = None,
) -> dict[str, Any]:
    """Enrich one vocal stem without extracting Vevo2 codec tokens."""

    config = config or EnrichmentConfig()
    source_path = Path(audio_path).expanduser().resolve()
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    started = time.monotonic()
    warnings: list[str] = []
    stages: list[dict[str, Any]] = []

    def stage_start(name: str) -> float:
        start = time.monotonic()
        _log(log_path, f"stage={name} status=started")
        return start

    def stage_done(name: str, start: float) -> None:
        elapsed = time.monotonic() - start
        stages.append({"name": name, "status": "succeeded", "elapsed_seconds": round(elapsed, 3)})
        _log(log_path, f"stage={name} status=succeeded elapsed_seconds={elapsed:.3f}")

    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if transcript_file is not None:
        transcript = Path(transcript_file).read_text(encoding="utf-8").strip()
    if transcript is not None:
        transcript = transcript.strip()

    normalize_stage = stage_start("normalize_audio")
    normalized_path = run_dir / "audio_24k_mono.wav"
    source_info = normalize_audio(source_path, normalized_path, config.prosody)
    stage_done("normalize_audio", normalize_stage)

    asr_stage = stage_start("transcription")
    if config.asr_provider == "none" and transcript:
        transcription = _provided_transcription(transcript, config.asr_language)
    else:
        transcription = transcribe(
            normalized_path,
            provider=config.asr_provider,
            model_name=config.asr_model,
            language=config.asr_language,
            device=config.asr_device,
            compute_type=config.asr_compute_type,
        )
        if transcript:
            transcription.text = transcript
            transcription.provider = f"{transcription.provider}+provided_text"
    write_json(run_dir / "transcript.json", transcription.to_dict())
    warnings.extend(transcription.warnings)
    stage_done("transcription", asr_stage)

    alignment_stage = stage_start("alignment")
    requested_text = transcript or transcription.text
    alignment_text = requested_text if requested_text else None
    exact_phones: dict[int, list[tuple[str, float, float]]] | None = None
    provider_label: str | None = None
    observed_words = transcription.words
    if config.alignment_provider in {"nfa", "nfa+mfa"}:
        if config.nfa_script is None:
            raise ValueError("--alignment-provider nfa requires --nfa-script pointing to NeMo's align.py")
        observed_words = run_nfa(
            normalized_path,
            alignment_text,
            run_dir / "nfa",
            script=config.nfa_script,
            model_name=config.nfa_model,
            python_executable=config.nfa_python or sys.executable,
        )
        provider_label = "nfa_word_timestamps"
    if config.alignment_provider in {"mfa", "nfa+mfa"}:
        if not alignment_text:
            raise ValueError("MFA alignment requires known transcript text")
        if config.mfa_dictionary is None or config.mfa_acoustic_model is None:
            raise ValueError("--alignment-provider mfa requires dictionary and acoustic-model paths")
        mfa_words, exact_phones = run_mfa(
            normalized_path,
            alignment_text,
            run_dir / "mfa",
            dictionary=config.mfa_dictionary,
            acoustic_model=config.mfa_acoustic_model,
            binary=config.mfa_binary,
        )
        if config.alignment_provider == "nfa+mfa":
            if len(observed_words) != len(mfa_words):
                warnings.append("NFA and MFA produced different word counts; MFA intervals were used for phone nesting.")
            observed_words = mfa_words
            provider_label = "nfa+mfa_word_phone_timestamps"
            if not exact_phones or not any(exact_phones.values()):
                warnings.append(
                    "MFA produced no usable non-silence phone intervals for this audio; "
                    "word timings remain aligned, while phoneme and syllable timings are estimated from G2P."
                )
        else:
            observed_words = mfa_words
            provider_label = "mfa_word_phone_timestamps"
    alignment_provider_for_builder = "mfa" if config.alignment_provider in {"mfa", "nfa+mfa"} else config.alignment_provider
    words, reconcile_warnings = reconcile_words(alignment_text, observed_words, source_info["duration_seconds"])
    warnings.extend(reconcile_warnings)
    alignment = build_alignment(
        words,
        config.phoneme_provider,
        alignment_provider_for_builder,
        exact_phones=exact_phones,
        provider_label=provider_label,
    )
    validate_alignment(alignment, source_info["duration_seconds"])
    write_json(run_dir / "alignment.json", alignment.to_dict())
    warnings.extend(alignment.warnings)
    stage_done("alignment", alignment_stage)

    prosody_stage = stage_start("prosody")
    prosody_dir = run_dir / "prosody"
    prosody = extract_prosody(
        normalized_path,
        prosody_dir,
        config=config.prosody,
        include_f0_diagnostic=config.include_f0_diagnostic,
    )
    stage_done("prosody", prosody_stage)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "runtime": "vevo2-dataset-enrichment",
        "status": "succeeded",
        "source": {
            "original_path": str(source_path),
            "normalized_audio": "audio_24k_mono.wav",
            **source_info,
        },
        "transcript": {
            "artifact": "transcript.json",
            "text": transcript or transcription.text,
            "provider": transcription.provider,
            "language": transcription.language,
        },
        "alignment": {
            "artifact": "alignment.json",
            "provider": alignment.provider,
            "quality": alignment.quality,
            "word_count": len(alignment.words),
            "syllable_count": len(alignment.syllables),
            "phoneme_count": len(alignment.phonemes),
        },
        "manifest": "manifest.json",
        "prosody": {"directory": "prosody", **prosody},
        "tokens": {
            "codec_tokens_extracted": False,
            "reason": "This run stops at training metadata and Vevo2-compatible prosody pre-quantization features.",
        },
        "diagnostics": {
            "warnings": warnings,
            "stages": stages,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "configuration": {
            "asr_provider": config.asr_provider,
            "asr_model": config.asr_model,
            "alignment_provider": config.alignment_provider,
            "phoneme_provider": config.phoneme_provider,
            "nfa_model": config.nfa_model,
            "mfa_binary": config.mfa_binary,
            "include_f0_diagnostic": config.include_f0_diagnostic,
            "prosody": {
                "sample_rate": config.prosody.sample_rate,
                "n_fft": config.prosody.n_fft,
                "win_length": config.prosody.win_length,
                "hop_length": config.prosody.hop_length,
                "n_chroma": config.prosody.n_chroma,
                "prosody_stride": config.prosody.prosody_stride,
            },
        },
    }
    write_json(run_dir / "metadata.json", metadata)
    _log(log_path, f"run status=succeeded elapsed_seconds={metadata['diagnostics']['elapsed_seconds']}")
    _write_manifest(run_dir)
    return metadata

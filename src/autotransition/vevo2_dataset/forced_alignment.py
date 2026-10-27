"""Adapters for the external NeMo Forced Aligner and MFA CLIs.

These tools are intentionally invoked as versioned external providers. Their
large CUDA/Conda runtimes should not be imported into the lightweight local
pipeline, and a missing provider must fail rather than downgrade labels.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .contracts import ProviderUnavailable, tokenize_words
from .transcription import WordObservation


def _run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Alignment provider failed ({completed.returncode}): {details[-4000:]}")


def _find_ctm(root: Path, kind: str, utterance_id: str) -> Path:
    candidates = [
        root / "ctm" / kind / f"{utterance_id}.ctm",
        root / "ctm" / kind / f"{utterance_id}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list((root / "ctm" / kind).glob("*")) if (root / "ctm" / kind).exists() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not find NFA {kind} CTM under {root}")


def _read_ctm(path: Path, source: str) -> list[WordObservation]:
    observations: list[WordObservation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        start = float(parts[2])
        duration = float(parts[3])
        # NFA CTM rows include provider metadata after the recognized token:
        # ``<utt> <channel> <start> <duration> <token> NA lex NA``.  Only
        # the fifth field is the word; including the metadata makes every
        # token fail transcript reconciliation.
        token = parts[4]
        if not token or token in {"<blank>", "<eps>", "sil", "SIL"}:
            continue
        observations.append(WordObservation(token, start, start + duration, None, source))
    return observations


def run_nfa(
    audio_path: Path,
    text: str | None,
    output_dir: Path,
    *,
    script: Path,
    model_name: str,
    python_executable: str = sys.executable,
    segment_separator: str | None = None,
    device: str | None = None,
) -> list[WordObservation]:
    """Run NVIDIA NeMo Forced Aligner and return its word CTM intervals."""

    if not script.exists():
        raise ProviderUnavailable(f"NeMo Forced Aligner script does not exist: {script}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "nfa_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_filepath": str(audio_path.resolve()),
                **({"text": text} if text else {}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        python_executable,
        str(script),
        f"pretrained_name={model_name}",
        f"manifest_filepath={manifest}",
        f"output_dir={output_dir}",
    ]
    if segment_separator:
        command.append(f"additional_segment_grouping_separator={segment_separator}")
    if device and device != "auto":
        command.extend([f"transcribe_device={device}", f"viterbi_device={device}"])
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    _run(command, cwd=script.parent, env=env)
    return _read_ctm(_find_ctm(output_dir, "words", "manifest"), "nfa")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _read_textgrid(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the interval tiers needed from a Praat long TextGrid."""

    tiers: dict[str, list[dict[str, Any]]] = {}
    tier_name: str | None = None
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("name ="):
            tier_name = _unquote(line.split("=", 1)[1].strip()).lower()
            tiers.setdefault(tier_name, [])
            current = None
        elif re.match(r"intervals \[\d+\]:", line):
            current = {}
            if tier_name is not None:
                tiers.setdefault(tier_name, []).append(current)
        elif current is not None and line.startswith("xmin ="):
            current["start"] = float(line.split("=", 1)[1].strip())
        elif current is not None and line.startswith("xmax ="):
            current["end"] = float(line.split("=", 1)[1].strip())
        elif current is not None and line.startswith("text ="):
            current["text"] = _unquote(line.split("=", 1)[1].strip())
    return tiers


def _find_tier(tiers: dict[str, list[dict[str, Any]]], names: tuple[str, ...]) -> list[dict[str, Any]]:
    for name in names:
        if name in tiers:
            return tiers[name]
    for name, intervals in tiers.items():
        if any(name.startswith(candidate) for candidate in names):
            return intervals
    raise ValueError(f"TextGrid does not contain one of the required tiers: {names}")


def run_mfa(
    audio_path: Path,
    text: str,
    output_dir: Path,
    *,
    dictionary: Path,
    acoustic_model: Path,
    binary: str = "mfa",
) -> tuple[list[WordObservation], dict[int, list[tuple[str, float, float]]]]:
    """Run MFA and return exact word intervals plus phone intervals by word."""

    executable = shutil.which(binary) or (str(binary) if Path(binary).exists() else None)
    if executable is None:
        raise ProviderUnavailable(
            f"MFA executable was not found: {binary}. Install MFA and its English "
            "dictionary/acoustic model before selecting --alignment-provider mfa."
        )
    if not dictionary.exists() or not acoustic_model.exists():
        raise ProviderUnavailable("MFA requires both an existing dictionary and acoustic model path.")

    corpus_dir = output_dir / "mfa_corpus"
    result_dir = output_dir / "mfa_output"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    corpus_audio = corpus_dir / "utterance.wav"
    corpus_text = corpus_dir / "utterance.lab"
    shutil.copy2(audio_path, corpus_audio)
    corpus_text.write_text(text.strip() + "\n", encoding="utf-8")
    _run(
        [
            executable,
            "align",
            str(corpus_dir),
            str(dictionary),
            str(acoustic_model),
            str(result_dir),
            "--clean",
            "--single_speaker",
        ]
    )
    # Windows treats these two glob patterns as the same path.  Deduplicate by
    # resolved path so a valid single TextGrid is not reported as two files.
    textgrid_paths = {
        path.resolve()
        for pattern in ("*.TextGrid", "*.textgrid")
        for path in result_dir.rglob(pattern)
    }
    textgrids = sorted(textgrid_paths)
    if len(textgrids) != 1:
        raise RuntimeError(f"Expected one MFA TextGrid, found {len(textgrids)} under {result_dir}")
    tiers = _read_textgrid(textgrids[0])
    word_intervals = _find_tier(tiers, ("words", "word"))
    phone_intervals = _find_tier(tiers, ("phones", "phone", "phonemes", "phoneme"))
    observations: list[WordObservation] = []
    phone_map: dict[int, list[tuple[str, float, float]]] = {}
    word_index = -1
    for interval in word_intervals:
        label = str(interval.get("text", "")).strip()
        if not label or label.lower() in {"sil", "sp", "spn"}:
            continue
        word_index += 1
        start = float(interval["start"])
        end = float(interval["end"])
        observations.append(WordObservation(label, start, end, None, "mfa"))
        phone_map[word_index] = []
        for phone in phone_intervals:
            phone_label = str(phone.get("text", "")).strip()
            overlap = min(end, float(phone.get("end", 0))) - max(start, float(phone.get("start", 0)))
            if overlap <= 0 or not phone_label or phone_label.lower() in {"sil", "sp", "spn"}:
                continue
            phone_map[word_index].append(
                (phone_label, max(start, float(phone["start"])), min(end, float(phone["end"])))
            )
    return observations, phone_map

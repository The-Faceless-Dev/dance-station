"""ASR provider adapters for dataset enrichment."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .contracts import ProviderUnavailable, tokenize_words


@dataclass
class WordObservation:
    text: str
    start: float
    end: float
    confidence: float | None = None
    source: str = "asr"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["duration"] = self.duration
        return value


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordObservation]
    provider: str
    language: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "language": self.language,
            "words": [word.to_dict() for word in self.words],
            "warnings": self.warnings,
        }


def _segment_words(segment: Any) -> list[WordObservation]:
    words = getattr(segment, "words", None)
    if words:
        result = []
        for word in words:
            text = str(getattr(word, "word", "")).strip()
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if text and start is not None and end is not None:
                probability = getattr(word, "probability", None)
                result.append(WordObservation(text, float(start), float(end), probability))
        if result:
            return result

    text = str(getattr(segment, "text", "")).strip()
    start = float(getattr(segment, "start", 0.0))
    end = float(getattr(segment, "end", start))
    tokens = tokenize_words(text)
    if not tokens or end <= start:
        return []
    weights = [max(1, len(token)) for token in tokens]
    total = float(sum(weights))
    result = []
    cursor = start
    for token, weight in zip(tokens, weights):
        next_cursor = cursor + (end - start) * weight / total
        result.append(WordObservation(token, cursor, next_cursor, None, "asr_segment_estimate"))
        cursor = next_cursor
    return result


def transcribe_faster_whisper(
    audio_path: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
) -> TranscriptionResult:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProviderUnavailable(
            "faster-whisper is not installed; install the project runtime or "
            "choose a configured Parakeet/NeMo provider."
        ) from exc

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if _cuda_available() else "cpu"
    if resolved_device == "cpu" and compute_type == "float16":
        compute_type = "int8"
    model = WhisperModel(model_name, device=resolved_device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language=language or None,
        word_timestamps=True,
        vad_filter=True,
    )
    observations: list[WordObservation] = []
    segment_text: list[str] = []
    for segment in segments:
        segment_text.append(str(getattr(segment, "text", "")).strip())
        observations.extend(_segment_words(segment))
    detected_language = str(getattr(info, "language", language or "unknown"))
    return TranscriptionResult(
        text=" ".join(part for part in segment_text if part).strip(),
        words=observations,
        provider="faster-whisper",
        language=detected_language,
        warnings=["ASR word timings are provisional until forced alignment is run."],
    )


def transcribe_parakeet(
    audio_path: Path,
    model_name: str,
    language: str,
    device: str,
) -> TranscriptionResult:
    """Use the NeMo Parakeet boundary when NeMo is installed.

    NeMo has changed timestamp object shapes between releases. This adapter
    accepts the common word-timestamp shape and fails loudly for an unknown
    shape instead of silently writing inaccurate labels.
    """

    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProviderUnavailable(
            "The Parakeet provider requires NeMo. Install a compatible "
            "nemo_toolkit ASR environment before selecting --asr-provider parakeet."
        ) from exc

    model = ASRModel.from_pretrained(model_name=model_name)
    resolved_device = device if device != "auto" else ("cuda" if _cuda_available() else "cpu")
    model = model.to(resolved_device)
    hypotheses = model.transcribe([str(audio_path)], timestamps=True)
    hypothesis = hypotheses[0] if isinstance(hypotheses, (list, tuple)) else hypotheses
    text = str(getattr(hypothesis, "text", "")).strip()
    timestamp_data = (
        getattr(hypothesis, "timestamp", None)
        or getattr(hypothesis, "timestep", None)
        or getattr(hypothesis, "timestamps", None)
    )
    words: list[WordObservation] = []
    if isinstance(timestamp_data, dict):
        timestamp_data = timestamp_data.get("word") or timestamp_data.get("words")
    if timestamp_data:
        for item in timestamp_data:
            if isinstance(item, dict):
                token = item.get("word") or item.get("text")
                start = item.get("start")
                end = item.get("end")
            else:
                token = getattr(item, "word", None) or getattr(item, "text", None)
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
            if isinstance(item, dict) and start is None and "start" in item:
                start = item["start"]
            if isinstance(item, dict) and end is None and "end" in item:
                end = item["end"]
            if token and start is not None and end is not None:
                words.append(WordObservation(str(token), float(start), float(end), None, "parakeet"))
    if not words:
        raise RuntimeError(
            "Parakeet returned text but no recognized word timestamp structure. "
            "Pin the NeMo version and update the provider adapter before using it for labels."
        )
    return TranscriptionResult(text=text, words=words, provider="parakeet", language=language, warnings=[])


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def transcribe(
    audio_path: Path,
    provider: str,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
) -> TranscriptionResult:
    if provider == "faster-whisper":
        return transcribe_faster_whisper(audio_path, model_name, language, device, compute_type)
    if provider == "parakeet":
        return transcribe_parakeet(audio_path, model_name, language, device)
    if provider == "none":
        return TranscriptionResult("", [], "none", language, ["No ASR provider was selected."])
    raise ValueError(f"Unsupported ASR provider: {provider}")

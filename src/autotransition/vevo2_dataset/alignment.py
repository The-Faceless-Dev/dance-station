"""Transcript reconciliation, phonemization, and timing construction."""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from .contracts import DatasetEnrichmentError, ProviderUnavailable, normalize_text_token, tokenize_words
from .transcription import WordObservation


VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}

_G2P_INSTANCE: Any | None = None


@dataclass
class TimedPhoneme:
    phone: str
    start: float
    end: float
    duration: float
    word_index: int
    syllable_index: int | None
    timing_quality: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimedSyllable:
    text: str
    start: float
    end: float
    duration: float
    word_index: int
    syllable_index: int
    stress: str | None
    phoneme_indices: list[int]
    timing_quality: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimedWord:
    text: str
    start: float
    end: float
    duration: float
    word_index: int
    confidence: float | None
    timing_quality: str
    phonemes: list[TimedPhoneme]
    syllables: list[TimedSyllable]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phonemes"] = [phone.to_dict() for phone in self.phonemes]
        value["syllables"] = [syllable.to_dict() for syllable in self.syllables]
        return value


@dataclass
class AlignmentResult:
    provider: str
    quality: str
    words: list[TimedWord]
    phonemes: list[TimedPhoneme]
    syllables: list[TimedSyllable]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "quality": self.quality,
            "words": [word.to_dict() for word in self.words],
            "phonemes": [phone.to_dict() for phone in self.phonemes],
            "syllables": [syllable.to_dict() for syllable in self.syllables],
            "warnings": self.warnings,
        }


def validate_alignment(result: AlignmentResult, duration: float, tolerance: float = 0.05) -> None:
    """Reject malformed intervals before they can become training metadata."""

    errors: list[str] = []
    previous_end = 0.0
    for word in result.words:
        if word.start < -tolerance or word.end > duration + tolerance or word.end < word.start:
            errors.append(f"word {word.word_index} is outside the audio bounds")
        if word.start + tolerance < previous_end:
            errors.append(f"word {word.word_index} is not monotonic")
        previous_end = max(previous_end, word.end)
        if word.phonemes:
            if abs(sum(phone.duration for phone in word.phonemes) - word.duration) > tolerance:
                errors.append(f"phoneme durations do not conserve word {word.word_index} duration")
            for phone in word.phonemes:
                if phone.start < word.start - tolerance or phone.end > word.end + tolerance:
                    errors.append(f"phoneme {phone.phone} escapes word {word.word_index}")
        for syllable in word.syllables:
            if syllable.start < word.start - tolerance or syllable.end > word.end + tolerance:
                errors.append(f"syllable {syllable.syllable_index} escapes word {word.word_index}")
    if errors:
        raise DatasetEnrichmentError("Invalid alignment: " + "; ".join(errors))


def _base_phone(phone: str) -> str:
    return re.sub(r"[012]$", "", phone.upper())


def is_vowel(phone: str) -> bool:
    return _base_phone(phone) in VOWELS


def phone_stress(phone: str) -> str | None:
    value = phone.upper()
    if value and value[-1] in "012":
        return value[-1]
    return None


def phonemize_word(word: str, provider: str = "g2p_en") -> list[str]:
    if provider == "none":
        return []
    if provider == "g2p_en":
        global _G2P_INSTANCE
        try:
            from g2p_en import G2p
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ProviderUnavailable(
                "g2p-en is required for ARPAbet phonemes. Install the "
                "vevo2-dataset optional dependencies or choose --phoneme-provider none."
            ) from exc
        if _G2P_INSTANCE is None:
            _G2P_INSTANCE = G2p()
        values = _G2P_INSTANCE(word)
        return [str(value) for value in values if re.fullmatch(r"[A-Za-z]+[012]?", str(value))]
    if provider == "phonemizer":
        try:
            from phonemizer import phonemize
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ProviderUnavailable(
                "phonemizer is required for the phonemizer provider and must be "
                "installed together with espeak-ng."
            ) from exc
        output = phonemize(word, language="en-us", backend="espeak", strip=True, preserve_punctuation=False)
        return [token for token in output.split() if token]
    raise ValueError(f"Unsupported phoneme provider: {provider}")


def _uniform_observations(words: list[str], duration: float) -> list[WordObservation]:
    if not words:
        return []
    weights = [max(1, len(word)) for word in words]
    total = float(sum(weights))
    result: list[WordObservation] = []
    cursor = 0.0
    for word, weight in zip(words, weights):
        end = cursor + duration * weight / total
        result.append(WordObservation(word, cursor, end, None, "uniform_estimate"))
        cursor = end
    return result


def reconcile_words(
    known_text: str | None,
    observed: Iterable[WordObservation],
    duration: float,
) -> tuple[list[WordObservation], list[str]]:
    """Map a supplied transcript onto rough ASR boundaries without changing text."""

    observed = list(observed)
    known_words = tokenize_words(known_text or "")
    if not known_words:
        return observed, []
    if not observed:
        return _uniform_observations(known_words, duration), [
            "No timestamped ASR words were available; word boundaries are uniform estimates."
        ]

    observed_span_start = max(0.0, min(duration, min(item.start for item in observed)))
    observed_span_end = min(duration, max(observed_span_start, max(item.end for item in observed)))

    observed_keys = [normalize_text_token(item.text) for item in observed]
    known_keys = [normalize_text_token(item) for item in known_words]
    matcher = difflib.SequenceMatcher(a=known_keys, b=observed_keys, autojunk=False)
    mapped: list[WordObservation | None] = [None] * len(known_words)
    warnings: list[str] = []
    for block in matcher.get_matching_blocks():
        known_start, observed_start, size = block
        for offset in range(size):
            if known_start + offset < len(known_words) and observed_start + offset < len(observed):
                item = observed[observed_start + offset]
                mapped[known_start + offset] = WordObservation(
                    known_words[known_start + offset],
                    max(0.0, item.start),
                    min(duration, max(item.start, item.end)),
                    item.confidence,
                    "known_transcript_matched",
                )

    if all(item is not None for item in mapped):
        return [item for item in mapped if item is not None], warnings

    warnings.append("Known transcript and ASR transcript differed; unmatched word boundaries were interpolated.")
    if not any(block.size for block in matcher.get_matching_blocks()):
        warnings.append(
            "No reliable transcript anchors were found; unmatched words were constrained to the detected vocal span."
        )
        span = max(0.0, observed_span_end - observed_span_start)
        estimated = _uniform_observations(known_words, span)
        for item in estimated:
            item.start += observed_span_start
            item.end += observed_span_start
            item.source = "known_transcript_span_estimate"
        return estimated, warnings
    index = 0
    while index < len(mapped):
        if mapped[index] is not None:
            index += 1
            continue
        run_start = index
        while index + 1 < len(mapped) and mapped[index + 1] is None:
            index += 1
        run_end = index
        previous = next((mapped[position] for position in range(run_start - 1, -1, -1) if mapped[position]), None)
        following = next((mapped[position] for position in range(run_end + 1, len(mapped)) if mapped[position]), None)
        start = previous.end if previous else observed_span_start
        end = following.start if following else observed_span_end
        if end < start:
            end = start
        missing = known_words[run_start : run_end + 1]
        weights = [max(1, len(value)) for value in missing]
        total = float(sum(weights))
        span = max(0.0, end - start)
        cursor = start
        for offset, (word, weight) in enumerate(zip(missing, weights)):
            item_end = end if offset == len(missing) - 1 else cursor + span * weight / total
            mapped[run_start + offset] = WordObservation(
                word, cursor, item_end, None, "known_transcript_interpolated"
            )
            cursor = item_end
        index += 1
    return [item for item in mapped if item is not None], warnings


def _phone_weights(phones: list[str]) -> list[float]:
    return [2.0 if is_vowel(phone) else 1.0 for phone in phones]


def _timed_phones(
    phones: list[str],
    start: float,
    end: float,
    word_index: int,
    timing_quality: str,
) -> list[TimedPhoneme]:
    if not phones:
        return []
    weights = _phone_weights(phones)
    total = sum(weights)
    cursor = start
    result: list[TimedPhoneme] = []
    for index, (phone, weight) in enumerate(zip(phones, weights)):
        # Compare the position, not the phone label.  A repeated final phone
        # (for example the two N phones in "shinin") must not cause the first
        # matching label to consume the whole word interval.
        next_cursor = end if index == len(phones) - 1 else cursor + (end - start) * weight / total
        result.append(
            TimedPhoneme(
                phone=phone,
                start=cursor,
                end=next_cursor,
                duration=max(0.0, next_cursor - cursor),
                word_index=word_index,
                syllable_index=None,
                timing_quality=timing_quality,
            )
        )
        cursor = next_cursor
    return result


def _syllable_ranges(phones: list[str]) -> list[tuple[int, int]]:
    nuclei = [index for index, phone in enumerate(phones) if is_vowel(phone)]
    if not nuclei:
        return [(0, len(phones))] if phones else []
    ranges: list[tuple[int, int]] = []
    for index, nucleus in enumerate(nuclei):
        start = 0 if index == 0 else nuclei[index - 1] + 1
        end = nuclei[index + 1] if index + 1 < len(nuclei) else len(phones)
        ranges.append((start, end))
    return ranges


def build_alignment(
    words: list[WordObservation],
    phoneme_provider: str,
    alignment_provider: str,
    exact_phones: dict[int, list[tuple[str, float, float]]] | None = None,
    provider_label: str | None = None,
) -> AlignmentResult:
    if alignment_provider not in {"fallback", "nfa", "mfa", "sofa"}:
        raise ValueError(f"Unsupported alignment provider: {alignment_provider}")

    all_phones: list[TimedPhoneme] = []
    all_syllables: list[TimedSyllable] = []
    timed_words: list[TimedWord] = []
    warnings: list[str] = []
    if alignment_provider == "fallback":
        warnings.append(
            "Fallback alignment uses word boundaries plus weighted phoneme durations; "
            "phoneme and syllable timings are estimated, not forced-aligned ground truth."
        )
    elif alignment_provider == "nfa":
        warnings.append(
            "NFA word/token timing is authoritative for words; phoneme boundaries are "
            "still derived unless an MFA phone tier is also supplied."
        )
    for word_index, observation in enumerate(words):
        exact = (exact_phones or {}).get(word_index, [])
        # A real SOFA/MFA phone tier is the authoritative pronunciation for
        # this word.  Use its labels for syllable grouping too; falling back to
        # a separately generated G2P sequence can silently mis-index phones
        # when a lexicon emits a different variant.
        phones = [phone for phone, _start, _end in exact] if exact else phonemize_word(observation.text, phoneme_provider)
        if exact:
            timing_quality = "aligned"
            timed_phones = [
                TimedPhoneme(
                    phone=phone,
                    start=start,
                    end=end,
                    duration=max(0.0, end - start),
                    word_index=word_index,
                    syllable_index=None,
                    timing_quality=timing_quality,
                )
                for phone, start, end in exact
            ]
        else:
            timing_quality = "estimated" if alignment_provider != "mfa" else "estimated_missing_phone"
            timed_phones = _timed_phones(
                phones,
                observation.start,
                observation.end,
                word_index,
                timing_quality,
            )
        ranges = _syllable_ranges(phones)
        timed_syllables: list[TimedSyllable] = []
        for syllable_index, (start_index, end_index) in enumerate(ranges):
            selected = timed_phones[start_index:end_index]
            if not selected:
                continue
            phoneme_indices = list(range(len(all_phones) + start_index, len(all_phones) + end_index))
            stress = next((phone_stress(phone.phone) for phone in selected if is_vowel(phone.phone)), None)
            syllable = TimedSyllable(
                text="".join(phone.phone for phone in selected),
                start=selected[0].start,
                end=selected[-1].end,
                duration=selected[-1].end - selected[0].start,
                word_index=word_index,
                syllable_index=syllable_index,
                stress=stress,
                phoneme_indices=phoneme_indices,
                timing_quality=timing_quality,
            )
            timed_syllables.append(syllable)
            for phone in selected:
                phone.syllable_index = syllable_index
        all_phones.extend(timed_phones)
        all_syllables.extend(timed_syllables)
        timed_words.append(
            TimedWord(
                text=observation.text,
                start=observation.start,
                end=observation.end,
                duration=observation.duration,
                word_index=word_index,
                confidence=observation.confidence,
                timing_quality=timing_quality,
                phonemes=timed_phones,
                syllables=timed_syllables,
            )
        )
    if phoneme_provider == "none":
        warnings.append("Phonemization was disabled; no phoneme or syllable labels were emitted.")
    has_exact_phone_intervals = bool(exact_phones) and any(exact_phones.values())
    quality = (
        "aligned"
        if has_exact_phone_intervals
        else "word_aligned"
        if alignment_provider == "nfa" or (provider_label and "word" in provider_label)
        else "estimated"
    )
    return AlignmentResult(
        provider=provider_label or {
            "fallback": "fallback_word_timestamps",
            "nfa": "nfa_word_timestamps",
            "mfa": "mfa_word_phone_timestamps",
            "sofa": "sofa_word_phone_timestamps",
        }[alignment_provider],
        quality=quality,
        words=timed_words,
        phonemes=all_phones,
        syllables=all_syllables,
        warnings=warnings,
    )

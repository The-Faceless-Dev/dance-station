from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


MuLaCoverStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
MuLaCoverOutputFormat = Literal["wav", "flac"]


def _public_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


@dataclass(frozen=True)
class MuLaCoverRequest:
    """One cover/remix request.

    Exactly one conditioning mode is required: ``ref_audio`` or both
    ``melody_midi`` and ``chord_midi``. URL fields are used in provider jobs;
    local paths are available only when explicitly enabled for local testing.
    """

    lyrics: str = ""
    tags: str = ""
    ref_audio_url: str = ""
    ref_audio_path: Path | None = None
    melody_midi_url: str = ""
    melody_midi_path: Path | None = None
    chord_midi_url: str = ""
    chord_midi_path: Path | None = None
    drum_midi_url: str = ""
    drum_midi_path: Path | None = None
    lyrics_url: str = ""
    tags_url: str = ""
    bpm: float | None = None
    semitone_shift: int = 0
    octave_shift: int = 0
    duration_seconds: float = 30.0
    cfg_scale: float = 1.5
    temperature: float = 1.0
    top_k: int = 250
    seed: int | None = None
    decode_seed: int | None = None
    save_symbolic_midi: bool = True
    output_format: MuLaCoverOutputFormat = "wav"
    external_job_id: str | None = None
    payment_intent_id: str | None = None

    def validate(self, config: Any) -> None:
        if not self.lyrics.strip() and not self.lyrics_url and self.conditioning_mode != "reference_audio":
            raise ValueError("MuLaCover requires non-empty lyrics or lyrics_url")
        if self.lyrics_url and not self._is_remote_url(self.lyrics_url):
            raise ValueError("lyrics_url must be an HTTP(S) URL")
        if not self.tags.strip() and not self.tags_url:
            raise ValueError("MuLaCover requires non-empty tags or tags_url")
        if self.tags_url and not self._is_remote_url(self.tags_url):
            raise ValueError("tags_url must be an HTTP(S) URL")
        if len(self.lyrics) > config.max_lyrics_characters:
            raise ValueError(f"lyrics must be {config.max_lyrics_characters} characters or fewer")
        if len(self.tags) > config.max_tags_characters:
            raise ValueError(f"tags must be {config.max_tags_characters} characters or fewer")

        has_audio = bool(self.ref_audio_url or self.ref_audio_path)
        has_midi = bool(
            self.melody_midi_url
            or self.melody_midi_path
            or self.chord_midi_url
            or self.chord_midi_path
            or self.drum_midi_url
            or self.drum_midi_path
        )
        if has_audio and has_midi:
            raise ValueError("ref_audio and MIDI conditioning are mutually exclusive")
        if not has_audio and not has_midi:
            raise ValueError("provide ref_audio, or both melody_midi and chord_midi")
        if has_audio and self.ref_audio_url and not self._is_remote_url(self.ref_audio_url):
            raise ValueError("ref_audio_url must be an HTTP(S) URL")
        if has_midi:
            if not (self.melody_midi_url or self.melody_midi_path) or not (self.chord_midi_url or self.chord_midi_path):
                raise ValueError("MIDI conditioning requires both melody_midi and chord_midi")
            for name, value in (
                ("melody_midi_url", self.melody_midi_url),
                ("chord_midi_url", self.chord_midi_url),
                ("drum_midi_url", self.drum_midi_url),
            ):
                if value and not self._is_remote_url(value):
                    raise ValueError(f"{name} must be an HTTP(S) URL")
        if self.bpm is not None and (not math.isfinite(self.bpm) or not 0 < self.bpm <= 300):
            raise ValueError("bpm must be between 0 and 300")
        if not -24 <= self.semitone_shift <= 24:
            raise ValueError("semitone_shift must be between -24 and 24")
        if not -2 <= self.octave_shift <= 2:
            raise ValueError("octave_shift must be between -2 and 2")
        if not math.isfinite(self.duration_seconds) or not config.min_duration_seconds <= self.duration_seconds <= config.max_duration_seconds:
            raise ValueError(f"duration_seconds must be between {config.min_duration_seconds} and {config.max_duration_seconds}")
        for name, value in (("cfg_scale", self.cfg_scale), ("temperature", self.temperature)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.top_k <= config.max_top_k:
            raise ValueError(f"top_k must be between 1 and {config.max_top_k}")
        for name, value in (("seed", self.seed), ("decode_seed", self.decode_seed)):
            if value is not None and (isinstance(value, bool) or not 0 <= value < 2**63):
                raise ValueError(f"{name} must be an unsigned 63-bit integer")
        if self.output_format not in {"wav", "flac"}:
            raise ValueError("output_format must be wav or flac")
        for name, value in (("external_job_id", self.external_job_id), ("payment_intent_id", self.payment_intent_id)):
            if value is not None and (not value.strip() or len(value) > 240):
                raise ValueError(f"{name} must be a non-empty value of 240 characters or fewer")
        if self.external_job_id and any(char in self.external_job_id for char in "\\/"):
            raise ValueError("external_job_id cannot contain path separators")

    @staticmethod
    def _is_remote_url(value: str) -> bool:
        parsed = urlsplit(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @property
    def conditioning_mode(self) -> Literal["reference_audio", "midi"]:
        return "reference_audio" if (self.ref_audio_url or self.ref_audio_path) else "midi"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in tuple(payload):
            if key.endswith("_path") and payload[key] is not None:
                payload[key] = str(payload[key])
            if key.endswith("_url"):
                payload[key] = _public_url(str(payload[key] or ""))
        payload["conditioning_mode"] = self.conditioning_mode
        return payload


@dataclass(frozen=True)
class MuLaCoverArtifact:
    name: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str
    role: str = "metadata"


@dataclass(frozen=True)
class MuLaCoverFailure:
    code: str
    message: str
    stage: str
    retryable: bool = False
    attempt: int = 1
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MuLaCoverJob:
    id: str
    status: MuLaCoverStatus
    request: dict[str, Any]
    stage: str | None = None
    progress: float = 0.0
    attempt: int = 0
    message: str | None = None
    artifacts: list[MuLaCoverArtifact] = field(default_factory=list)
    failure: MuLaCoverFailure | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(item) for item in self.artifacts]
        payload["failure"] = self.failure.to_dict() if self.failure else None
        payload["failureCode"] = self.failure.code if self.failure else None
        payload["refundRequired"] = bool(self.failure)
        payload["refundReason"] = "mulacover_music_generation_failed" if self.failure else None
        return payload

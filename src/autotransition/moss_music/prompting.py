from __future__ import annotations

from dataclasses import dataclass

from .contracts import MossMusicRequest


DEFAULT_ANALYSIS_PROMPT = """Analyze the supplied musical audio for downstream visual synchronization.
Return only the fields requested for the current analysis pass. Use absolute seconds
from the beginning of the supplied audio. Do not invent timestamps for events you
cannot identify; use an empty list or null scalar and include a warning instead.

Every timed item must use start_seconds and end_seconds (or time_seconds for an
instantaneous event), plus type or label, strength when meaningful, and confidence
between 0 and 1 when it can be estimated. Keep event timing conservative and preserve
uncertainty in warnings rather than presenting guesses as facts."""


@dataclass(frozen=True)
class MossAnalysisPass:
    name: str
    instruction: str
    required_keys: tuple[str, ...]


ANALYSIS_PASSES = (
    MossAnalysisPass(
        name="overview",
        instruction=(
            "Return the musical overview: summary, tempo_bpm, time_signature, key, "
            "sections, instruments, voices, visual_cues, and warnings. Include all "
            "detectable sections with absolute timestamps."
        ),
        required_keys=("summary", "tempo_bpm", "time_signature", "key", "sections", "instruments", "voices", "visual_cues", "warnings"),
    ),
    MossAnalysisPass(
        name="rhythm",
        instruction=(
            "Return the complete rhythm interpretation: beats and musical events "
            "such as kicks, snares, claps, hats, percussion, accents, fills, drops, "
            "and other useful onset events. Preserve strength and confidence. Do not "
            "limit the number of events; use the full audio and absolute timestamps."
        ),
        required_keys=("beats", "events", "warnings"),
    ),
    MossAnalysisPass(
        name="harmony",
        instruction=(
            "Return the complete harmonic interpretation: key, chord changes, chord "
            "labels, and any meaningful harmonic events. Use absolute timestamps and "
            "include confidence where possible."
        ),
        required_keys=("key", "chords", "events", "warnings"),
    ),
    MossAnalysisPass(
        name="lyrics_and_voices",
        instruction=(
            "Return all detectable timestamped lyrics, vocal phrases, vocal sections, "
            "and voice descriptions. If the audio is instrumental, return an empty "
            "lyrics list and explain that in warnings rather than inventing words."
        ),
        required_keys=("lyrics", "voices", "events", "warnings"),
    ),
)


_FIELD_SHAPES = {
    "summary": "string",
    "tempo_bpm": "number or null",
    "time_signature": "string or null",
    "key": "string or null",
    "sections": [],
    "beats": [],
    "chords": [],
    "lyrics": [],
    "instruments": [],
    "voices": [],
    "visual_cues": [],
    "events": [],
    "warnings": [],
}


def _prompt_for_pass(request: MossMusicRequest, analysis_pass: MossAnalysisPass) -> str:
    user_context = request.prompt.strip() if request.prompt and request.prompt.strip() else DEFAULT_ANALYSIS_PROMPT
    schema = {key: _FIELD_SHAPES[key] for key in analysis_pass.required_keys}
    return (
        f"{user_context}\n\n"
        f"Analysis pass: {analysis_pass.name}. {analysis_pass.instruction}\n"
        "Return only one valid JSON object, with no Markdown fences and no prose outside it. "
        "The object must contain exactly the keys shown below, even when a list is empty "
        "or a scalar is null. Do not add data for another analysis pass. Use absolute "
        "seconds from the beginning of the supplied audio. "
        "Every timed item must use start_seconds and end_seconds, or time_seconds for an "
        "instantaneous event, plus type or label, strength when meaningful, and confidence "
        "between 0 and 1 when it can be estimated. Do not invent a regular beat grid or "
        "repeat events at fixed intervals. Include only events supported by the supplied "
        "audio, then stop after the last supported item.\n"
        f"Required JSON shape for this pass only: {schema}\n\n"
        f"Analysis profile: {request.analysis_profile}. The companion measured timeline uses "
        f"{request.event_resolution_ms} ms cells. Use the audio itself for interpretation."
    )


def build_analysis_prompts(request: MossMusicRequest) -> list[MossAnalysisPass]:
    return [
        MossAnalysisPass(
            name=analysis_pass.name,
            instruction=_prompt_for_pass(request, analysis_pass),
            required_keys=analysis_pass.required_keys,
        )
        for analysis_pass in ANALYSIS_PASSES
    ]


def build_analysis_prompt(request: MossMusicRequest) -> str:
    """Backward-compatible access to the full overview prompt."""

    return build_analysis_prompts(request)[0].instruction


def add_segment_context(*, start_seconds: float, end_seconds: float, full_duration_seconds: float) -> str:
    """Tell a bounded pass how to place its local audio results on the source timeline."""

    return (
        f"\n\nThis is a bounded segment of the original audio: "
        f"{start_seconds:.3f} to {end_seconds:.3f} seconds of {full_duration_seconds:.3f} seconds. "
        "Return only information supported by this segment. Use timestamps relative to the "
        "beginning of the supplied segment; the runtime will place them on the original "
        "timeline. Do not repeat events from outside this segment."
    )

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
    output_token_budget: int


ANALYSIS_PASSES = (
    MossAnalysisPass(
        name="overview",
        instruction=(
            "Return the musical overview: summary, tempo_bpm, time_signature, key, "
            "major sections, instruments, voices, visual_cues, and warnings. Include "
            "major intro, verse, pre-chorus, chorus, bridge, breakdown, and outro "
            "boundaries with absolute timestamps. Keep visual_cues sparse and useful "
            "for visual planning; do not emit a beat grid or repeated low-level events."
        ),
        required_keys=("summary", "tempo_bpm", "time_signature", "key", "sections", "instruments", "voices", "visual_cues", "warnings"),
        output_token_budget=1536,
    ),
    MossAnalysisPass(
        name="rhythm",
        instruction=(
            "Return sparse, salient within-section musical changes that matter for "
            "visual synchronization, under the visual_events field. Do not emit a "
            "regular beat grid, every onset, or repeated filler events. Prioritize "
            "meaningful builds, drops, breakdowns, fills, strong accents, groove or "
            "rhythmic-density changes, and vocal or instrument entries/exits. Return "
            "at most 12 salient visual_events for this segment. Each visual event "
            "should include start_seconds, end_seconds, type, intensity from 0 to 1, "
            "emotional_direction, a short description, musical_evidence, and confidence. "
            "Use an empty list when no salient change is supported by the audio."
        ),
        required_keys=("visual_events", "warnings"),
        output_token_budget=1536,
    ),
    MossAnalysisPass(
        name="harmony",
        instruction=(
            "Return a concise harmonic interpretation: key and meaningful chord changes "
            "only. Do not repeat a chord at regular intervals or emit a full beat grid. "
            "Use absolute timestamps and include confidence where possible."
        ),
        required_keys=("key", "chords"),
        output_token_budget=1024,
    ),
    MossAnalysisPass(
        name="lyrics_and_voices",
        instruction=(
            "Return timestamped lyric lines or meaningful vocal phrases, not individual "
            "word fragments or repeated filler. Include concise voice descriptions. If "
            "the audio is instrumental, return an empty lyrics list and explain that in "
            "warnings rather than inventing words."
        ),
        required_keys=("lyrics", "voices"),
        output_token_budget=2048,
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
    "visual_events": [],
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
        "audio, then stop after the last supported item. Missing optional fields are valid "
        "when the pass has no supported value.\n"
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
            output_token_budget=analysis_pass.output_token_budget,
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

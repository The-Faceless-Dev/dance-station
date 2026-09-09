from __future__ import annotations

from .contracts import MossMusicRequest


DEFAULT_ANALYSIS_PROMPT = """Analyze the supplied musical audio for downstream visual synchronization.
Return only one valid JSON object, with no Markdown fences and no prose outside the object.
Use absolute seconds from the beginning of the audio. Do not invent timestamps for events
you cannot identify; use null or an empty list and include a warning instead.

The object must contain these keys:
{
  "summary": "string",
  "tempo_bpm": number or null,
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
  "warnings": []
}

Every timed item must use start_seconds and end_seconds (or time_seconds for an
instantaneous event), plus type or label, strength when meaningful, and confidence
between 0 and 1 when it can be estimated. Keep event timing conservative and preserve
uncertainty in warnings rather than presenting guesses as facts."""


def build_analysis_prompt(request: MossMusicRequest) -> str:
    base = request.prompt.strip() if request.prompt and request.prompt.strip() else DEFAULT_ANALYSIS_PROMPT
    return (
        f"{base}\n\nAnalysis profile: {request.analysis_profile}. "
        f"The companion measured timeline uses {request.event_resolution_ms} ms cells. "
        "Use the audio itself for musical interpretation; do not describe the request."
    )

from __future__ import annotations

import json
import math
from typing import Any


class MossResponseError(ValueError):
    """Raised when the model response cannot become validated structured data."""


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if isinstance(response.get("text"), str):
            return response["text"]
        if isinstance(response.get("output"), str):
            return response["output"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            if isinstance(choice.get("text"), str):
                return choice["text"]
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        if "events" in response or "sections" in response:
            return json.dumps(response)
    raise MossResponseError("MOSS backend returned no text or structured JSON response")


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```") and stripped.endswith("```"):
        candidates.insert(0, stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise MossResponseError("MOSS response did not contain a valid JSON object")


def _number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if minimum is not None and result < minimum:
        return None
    if maximum is not None and result > maximum:
        return None
    return result


def _timed_item(value: Any, *, index: int, default_type: str, source: str = "moss") -> dict[str, Any] | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = {"time_seconds": value}
    if not isinstance(value, dict):
        return None
    start = _number(value.get("start_seconds", value.get("start", value.get("time_seconds", value.get("time")))), minimum=0)
    if start is None:
        return None
    end = _number(value.get("end_seconds", value.get("end")), minimum=start)
    if end is None:
        duration = _number(value.get("duration_seconds", value.get("duration")), minimum=0)
        end = start + duration if duration is not None else start
    item_type = str(value.get("type") or value.get("label") or default_type).strip() or default_type
    normalized = dict(value)
    normalized.update(
        {
            "id": str(value.get("id") or f"moss-{default_type}-{index:06d}"),
            "type": item_type,
            "start_seconds": round(start, 6),
            "end_seconds": round(max(start, end), 6),
            "source": str(value.get("source") or source),
            "confidence": _number(value.get("confidence"), minimum=0, maximum=1),
            "strength": _number(value.get("strength"), minimum=0, maximum=1),
            "provenance": str(value.get("provenance") or "MOSS-Music model output"),
        }
    )
    return normalized


def parse_moss_response(response: Any) -> tuple[dict[str, Any], str]:
    """Parse and normalize model JSON while retaining its exact text."""

    text = _response_text(response)
    raw = _json_object(text)
    parsed: dict[str, Any] = {
        "summary": str(raw.get("summary") or ""),
        "tempo_bpm": _number(raw.get("tempo_bpm", raw.get("tempoBpm")), minimum=1, maximum=400),
        "time_signature": raw.get("time_signature", raw.get("timeSignature")),
        "key": raw.get("key"),
        "warnings": [str(value) for value in raw.get("warnings") or []],
    }
    collections = {
        "sections": "section",
        "beats": "beat",
        "chords": "chord",
        "lyrics": "lyric",
        "instruments": "instrument",
        "voices": "voice",
        "visual_cues": "visual_cue",
    }
    all_events: list[dict[str, Any]] = []
    invalid_counts: dict[str, int] = {}
    for collection, default_type in collections.items():
        values = raw.get(collection)
        if not isinstance(values, list):
            values = []
        normalized = []
        for index, value in enumerate(values):
            item = _timed_item(value, index=index, default_type=default_type)
            if item is None:
                invalid_counts[collection] = invalid_counts.get(collection, 0) + 1
            else:
                normalized.append(item)
        parsed[collection] = normalized
        all_events.extend(normalized)
    raw_events = raw.get("events") if isinstance(raw.get("events"), list) else []
    parsed_events = []
    for index, value in enumerate(raw_events):
        item = _timed_item(value, index=index, default_type="event")
        if item is None:
            invalid_counts["events"] = invalid_counts.get("events", 0) + 1
        else:
            parsed_events.append(item)
    parsed["events"] = parsed_events
    parsed["event_count"] = len(parsed_events)
    if invalid_counts:
        parsed["warnings"].extend(f"Dropped {count} invalid timed {collection} record(s)" for collection, count in invalid_counts.items())
    if not parsed["events"]:
        parsed["events"] = sorted(all_events, key=lambda value: (value["start_seconds"], value["id"]))
        parsed["event_count"] = len(parsed["events"])
    return parsed, text

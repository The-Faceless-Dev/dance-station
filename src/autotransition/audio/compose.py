"""Audio composition helpers for generated continuations."""

from __future__ import annotations

from pathlib import Path

from autotransition.audio.formats import validate_supported_source


def build_continuation_composite(
    source_path: Path,
    generated_path: Path,
    output_path: Path,
    continuation_point_seconds: float,
    output_format: str = "wav",
) -> Path:
    """Write ``source[:continuation_point] + generated`` to ``output_path``."""

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required to compose generated audio. Install the project dependencies.") from exc

    if continuation_point_seconds <= 0:
        raise ValueError("continuation_point_seconds must be greater than 0")
    if not source_path.exists():
        raise FileNotFoundError(f"Source audio not found: {source_path}")
    if not generated_path.exists():
        raise FileNotFoundError(f"Generated audio not found: {generated_path}")
    validate_supported_source(source_path)
    validate_supported_source(generated_path)

    source = AudioSegment.from_file(source_path)
    generated = AudioSegment.from_file(generated_path)
    source_duration_seconds = len(source) / 1000
    if continuation_point_seconds > source_duration_seconds:
        raise ValueError(
            f"Continuation point is {continuation_point_seconds:.2f}s, "
            f"but source is only {source_duration_seconds:.2f}s."
        )

    source_head = source[: int(continuation_point_seconds * 1000)]
    generated = (
        generated.set_frame_rate(source_head.frame_rate)
        .set_channels(source_head.channels)
        .set_sample_width(source_head.sample_width)
    )
    composite = source_head + generated

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.export(output_path, format=output_format)
    return output_path

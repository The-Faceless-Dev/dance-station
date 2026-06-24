"""Build repaint scaffolds from a selected point in a longer source song."""

from __future__ import annotations

from pathlib import Path

from autotransition.audio.ffmpeg import configure_pydub_ffmpeg
from autotransition.audio.formats import validate_supported_source
from autotransition.audio.segments import matching_silence


def build_selection_scaffold(
    source_path: Path,
    output_path: Path,
    tail_start_seconds: float,
    tail_end_seconds: float,
    blank_seconds: float,
    output_format: str = "wav",
    target_end_seconds: float | None = None,
    append_silence: bool = True,
) -> Path:
    """Write a source-selection scaffold to ``output_path``.

    By default this writes ``source[tail_start:tail_end] + silence``. Set
    ``append_silence=False`` for ACE-Step outpainting, where the runtime should
    create right-side padding from ``repainting_end``. When ``target_end_seconds``
    is provided, it writes ``source[tail_start:target_end]`` so ACE-Step can
    repaint existing audio.
    """

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required to build audio scaffolds. Install the project dependencies.") from exc
    configure_pydub_ffmpeg()

    if tail_start_seconds < 0:
        raise ValueError("tail_start_seconds cannot be negative")
    if tail_end_seconds <= tail_start_seconds:
        raise ValueError("tail_end_seconds must be greater than tail_start_seconds")
    if blank_seconds <= 0 and target_end_seconds is None and append_silence:
        raise ValueError("blank_seconds must be greater than 0")
    if not source_path.exists():
        raise FileNotFoundError(f"Source audio not found: {source_path}")
    validate_supported_source(source_path)

    source = AudioSegment.from_file(source_path)
    duration_seconds = len(source) / 1000
    scaffold_end_seconds = target_end_seconds or tail_end_seconds
    if scaffold_end_seconds > duration_seconds:
        raise ValueError(
            f"Selection ends at {scaffold_end_seconds:.2f}s, but source is only {duration_seconds:.2f}s."
        )
    if target_end_seconds is not None and target_end_seconds <= tail_end_seconds:
        raise ValueError("target_end_seconds must be greater than tail_end_seconds")

    start_ms = int(tail_start_seconds * 1000)
    end_ms = int(tail_end_seconds * 1000)
    selected_tail = source[start_ms:end_ms]
    if target_end_seconds is not None:
        scaffold = source[start_ms : int(target_end_seconds * 1000)]
    elif append_silence:
        blank_ms = int(blank_seconds * 1000)
        scaffold = selected_tail + matching_silence(selected_tail, blank_ms)
    else:
        scaffold = selected_tail

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaffold.export(output_path, format=output_format)
    return output_path

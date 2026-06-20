from pathlib import Path

from autotransition.audio import build_continuation_composite, build_repaint_scaffold, build_selection_scaffold


def make_stereo_wav(path: Path, duration_ms: int = 3000) -> Path:
    from pydub import AudioSegment

    segment = (
        AudioSegment.silent(duration=duration_ms, frame_rate=48000)
        .set_channels(2)
        .set_sample_width(2)
    )
    segment.export(path, format="wav")
    return path


def test_selection_scaffold_silence_matches_source_layout(tmp_path: Path) -> None:
    from pydub import AudioSegment

    source = make_stereo_wav(tmp_path / "source.wav")
    output = build_selection_scaffold(
        source_path=source,
        output_path=tmp_path / "selection.wav",
        tail_start_seconds=1.0,
        tail_end_seconds=2.0,
        blank_seconds=1.0,
    )

    scaffold = AudioSegment.from_file(output)

    assert scaffold.frame_rate == 48000
    assert scaffold.channels == 2
    assert scaffold.sample_width == 2


def test_selection_scaffold_can_write_tail_only_for_outpaint(tmp_path: Path) -> None:
    from pydub import AudioSegment

    source = make_stereo_wav(tmp_path / "source.wav", duration_ms=4000)
    output = build_selection_scaffold(
        source_path=source,
        output_path=tmp_path / "outpaint.wav",
        tail_start_seconds=1.0,
        tail_end_seconds=2.0,
        blank_seconds=2.0,
        append_silence=False,
    )

    scaffold = AudioSegment.from_file(output)

    assert len(scaffold) == 1000
    assert scaffold.channels == 2


def test_repaint_scaffold_silence_matches_source_layout(tmp_path: Path) -> None:
    from pydub import AudioSegment

    source = make_stereo_wav(tmp_path / "source.wav")
    output = build_repaint_scaffold(
        source_path=source,
        output_path=tmp_path / "repaint.wav",
        tail_seconds=1.0,
        blank_seconds=1.0,
    )

    scaffold = AudioSegment.from_file(output)

    assert scaffold.frame_rate == 48000
    assert scaffold.channels == 2
    assert scaffold.sample_width == 2


def test_selection_scaffold_can_include_existing_target_audio(tmp_path: Path) -> None:
    from pydub import AudioSegment

    source = make_stereo_wav(tmp_path / "source.wav", duration_ms=4000)
    output = build_selection_scaffold(
        source_path=source,
        output_path=tmp_path / "existing.wav",
        tail_start_seconds=1.0,
        tail_end_seconds=2.0,
        blank_seconds=2.0,
        target_end_seconds=4.0,
    )

    scaffold = AudioSegment.from_file(output)

    assert len(scaffold) == 3000
    assert scaffold.channels == 2


def test_continuation_composite_appends_generated_audio_at_marker(tmp_path: Path) -> None:
    from pydub import AudioSegment

    source = make_stereo_wav(tmp_path / "source.wav", duration_ms=4000)
    generated = make_stereo_wav(tmp_path / "generated.wav", duration_ms=1500)

    output = build_continuation_composite(
        source_path=source,
        generated_path=generated,
        output_path=tmp_path / "composite.wav",
        continuation_point_seconds=2.0,
    )

    composite = AudioSegment.from_file(output)

    assert len(composite) == 3500
    assert composite.channels == 2

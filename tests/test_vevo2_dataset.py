from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autotransition.vevo2_dataset.alignment import build_alignment, reconcile_words, validate_alignment
from autotransition.vevo2_dataset.forced_alignment import _read_ctm
from autotransition.vevo2_dataset.pipeline import EnrichmentConfig, enrich_dataset
from autotransition.vevo2_dataset.prosody import Vevo2ProsodyConfig, extract_prosody
from autotransition.vevo2_dataset.transcription import WordObservation


def _write_synthetic_audio(path: Path, seconds: float = 1.2, sample_rate: int = 24_000) -> None:
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * (180 + 35 * time) * time)
    audio += 0.05 * np.sin(2 * np.pi * 360 * time)
    sf.write(path, audio.astype(np.float32), sample_rate)


def test_vevo2_prosody_uses_exact_source_parameters(tmp_path: Path):
    audio_path = tmp_path / "source.wav"
    _write_synthetic_audio(audio_path)
    output = tmp_path / "prosody"
    config = Vevo2ProsodyConfig()
    metadata = extract_prosody(audio_path, output, config=config, include_f0_diagnostic=False)

    chroma = np.load(output / "chroma_50hz.npy")
    summary = np.load(output / "chroma_6_25hz_summary.npy")
    assert metadata["sample_rate"] == 24_000
    assert metadata["n_fft"] == 1_920
    assert metadata["hop_length"] == 480
    assert metadata["frame_rate_hz"] == 50.0
    assert metadata["prosody_frame_rate_hz"] == 6.25
    assert chroma.shape[1] == 24
    assert summary.shape[1] == 24
    assert np.isfinite(chroma).all()


def test_fallback_alignment_preserves_known_word_timing_and_syllables(monkeypatch):
    from autotransition.vevo2_dataset import alignment

    phonemes = {
        "I'm": ["AY1", "M"],
        "shining": ["SH", "AY1", "N", "IH0", "NG"],
    }
    monkeypatch.setattr(alignment, "phonemize_word", lambda word, provider: phonemes[word])
    observations = [
        WordObservation("I'm", 0.0, 0.35, 0.9),
        WordObservation("shining", 0.35, 1.1, 0.9),
    ]
    words, warnings = reconcile_words("I'm shining", observations, 1.1)
    result = build_alignment(words, phoneme_provider="g2p_en", alignment_provider="fallback")
    assert not warnings
    assert result.quality == "estimated"
    assert result.words[1].start == 0.35
    assert result.words[1].end == 1.1
    assert result.words[1].syllables
    assert result.words[1].phonemes
    assert all(phone.duration >= 0 for phone in result.phonemes)
    validate_alignment(result, 1.1)


def test_repeated_final_phoneme_does_not_escape_word(monkeypatch):
    from autotransition.vevo2_dataset import alignment

    monkeypatch.setattr(
        alignment,
        "phonemize_word",
        lambda word, provider: ["SH", "IH1", "N", "IH0", "N"],
    )
    words, warnings = reconcile_words(
        "shinin",
        [WordObservation("shinin", 0.0, 0.9, 0.9)],
        0.9,
    )
    result = build_alignment(words, phoneme_provider="g2p_en", alignment_provider="fallback")
    assert not warnings
    assert result.words[0].phonemes[-1].end == pytest.approx(0.9)
    assert result.words[0].phonemes[-2].end == pytest.approx(result.words[0].phonemes[-1].start)
    validate_alignment(result, 0.9)


def test_known_transcript_mismatch_interpolates_an_unmatched_run_once():
    observations = [WordObservation("hello", 0.0, 0.2, 0.9)]
    words, warnings = reconcile_words("one two six", observations, 0.9)
    assert warnings
    assert [word.start for word in words] == pytest.approx([0.0, 0.2 / 3, 0.4 / 3])
    assert [word.end for word in words] == pytest.approx([0.2 / 3, 0.4 / 3, 0.2])


def test_nfa_ctm_reader_ignores_provider_metadata(tmp_path: Path):
    ctm = tmp_path / "words.ctm"
    ctm.write_text(
        "audio 1 16.08 0.24 shining NA lex NA\n"
        "audio 1 16.32 0.08 ignored NA lex NA\n",
        encoding="utf-8",
    )
    observations = _read_ctm(ctm, "nfa")
    assert [item.text for item in observations] == ["shining", "ignored"]
    assert observations[0].start == pytest.approx(16.08)
    assert observations[0].end == pytest.approx(16.32)


def test_mismatched_transcript_stays_inside_observed_vocal_span():
    observations = [
        WordObservation("recognized", 16.08, 16.40, 0.9),
        WordObservation("ending", 194.10, 194.48, 0.9),
    ]
    words, warnings = reconcile_words("unrelated supplied words", observations, 194.608)
    assert warnings
    assert words[0].start == pytest.approx(16.08)
    assert words[-1].end == pytest.approx(194.48)


def test_real_g2p_en_produces_arpabet_stress_for_dataset_labels():
    from autotransition.vevo2_dataset.alignment import phonemize_word

    phones = phonemize_word("shining", provider="g2p_en")
    assert phones == ["SH", "AY1", "N", "IH0", "NG"]


def test_pipeline_records_no_codec_tokens_and_provider_provenance(tmp_path: Path):
    audio_path = tmp_path / "source.wav"
    _write_synthetic_audio(audio_path, seconds=0.8)
    output = tmp_path / "run"
    metadata = enrich_dataset(
        audio_path,
        output,
        config=EnrichmentConfig(
            asr_provider="none",
            alignment_provider="fallback",
            phoneme_provider="none",
            include_f0_diagnostic=False,
        ),
        transcript="hello world",
    )
    assert metadata["status"] == "succeeded"
    assert metadata["tokens"]["codec_tokens_extracted"] is False
    assert metadata["alignment"]["quality"] == "estimated"
    assert (output / "audio_24k_mono.wav").exists()
    assert (output / "prosody" / "chroma_50hz.npy").exists()
    assert (output / "manifest.json").exists()
    saved = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert saved["prosody"]["representation"] == "vevo2_chroma_prequantization"

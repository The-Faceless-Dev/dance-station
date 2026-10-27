"""Enrich a vocal stem with Vevo2 AR adapter training metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from autotransition.vevo2_dataset import EnrichmentConfig, enrich_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Vocal stem or audio file to enrich")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transcript")
    parser.add_argument("--transcript-file", type=Path)
    parser.add_argument("--asr-provider", choices=["faster-whisper", "parakeet", "none"], default="parakeet")
    parser.add_argument("--asr-model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--alignment-provider", choices=["fallback", "nfa", "mfa", "nfa+mfa"], default="nfa+mfa")
    parser.add_argument("--nfa-script", type=Path, help="NeMo tools/nemo_forced_aligner/align.py")
    parser.add_argument("--nfa-model", default="stt_en_fastconformer_hybrid_large_pc")
    parser.add_argument("--nfa-python", default="", help="Python executable for the NeMo environment")
    parser.add_argument("--mfa-binary", default="mfa")
    parser.add_argument("--mfa-dictionary", type=Path)
    parser.add_argument("--mfa-acoustic-model", type=Path)
    parser.add_argument("--phoneme-provider", choices=["g2p_en", "phonemizer", "none"], default="g2p_en")
    parser.add_argument("--no-f0-diagnostic", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = EnrichmentConfig(
        asr_provider=args.asr_provider,
        asr_model=args.asr_model,
        asr_language=args.language,
        asr_device=args.device,
        asr_compute_type=args.compute_type,
        alignment_provider=args.alignment_provider,
        phoneme_provider=args.phoneme_provider,
        nfa_script=args.nfa_script,
        nfa_model=args.nfa_model,
        nfa_python=args.nfa_python,
        mfa_binary=args.mfa_binary,
        mfa_dictionary=args.mfa_dictionary,
        mfa_acoustic_model=args.mfa_acoustic_model,
        include_f0_diagnostic=not args.no_f0_diagnostic,
    )
    metadata = enrich_dataset(
        args.audio,
        args.output_dir,
        config=config,
        transcript=args.transcript,
        transcript_file=args.transcript_file,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

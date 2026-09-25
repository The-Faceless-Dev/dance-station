# MuLaCover worker

This is the launcher-compatible MuLaCover cover/remix worker. It accepts either
reference audio or melody/chord MIDI conditioning, plus lyrics and named style
tags. The two conditioning modes cannot be mixed.

## Request controls

`lyrics`, `tags`, `ref_audio` or `melody_midi` + `chord_midi`, optional
`drum_midi`, `bpm`, `semitone_shift`, `octave_shift`, `duration_seconds`,
`cfg_scale`, `temperature`, `top_k`, `seed`, `decode_seed`,
`save_symbolic_midi`, and `output_format` (`wav` or `flac`). Inputs are normally
HTTP(S) URLs supplied by the launcher. Local paths require
`MULACOVER_ALLOW_LOCAL_INPUTS=true` and are intended only for local testing.

The worker exposes `/process` for the launcher queue contract,
`/v1/mulacover/jobs` for direct HTTP use, `/v1/worker/status`, `/health`,
`/ready`, and durable artifact download URLs under each job.

## Memory and deployment

The upstream pipeline is used with lazy loading: transcription, style embedding,
MuLaCover generation, and HeartCodec decoding are not resident simultaneously.
The worker still requires a real CUDA device and rejects a request when the
configured minimum free VRAM is unavailable. The production image defaults to
20 GiB free VRAM and 24 GiB recommended capacity. A 10 GiB development GPU can
run the contract and preflight tests but should not be used for inference.

The model-bearing image includes MuLaCover, HeartCodec, Qwen3-Embedding, YourMT3,
and five ChordNet checkpoints so both input modes work without startup downloads.
The upstream model weights and generated outputs are restricted by the upstream
`MODEL_LICENSE` and are not automatically licensed for commercial use.

## Public image

The CI workflow publishes:

`ghcr.io/the-faceless-dev/faceless-mulacover-worker:mulacover-20260925-r1`

The package must remain public. CI verifies anonymous manifest pullability before
declaring the image ready for Salad, Vast, RunPod, or launcher configuration.

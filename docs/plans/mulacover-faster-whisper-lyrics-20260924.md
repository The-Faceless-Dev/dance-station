# MuLaCover Automatic Lyrics Extraction

## Goal

Allow MuLaCover reference-audio jobs to omit lyrics. When lyrics are absent,
the worker will transcribe the downloaded source audio with `faster-whisper`
and pass the resulting transcript to MuLaCover. Explicit lyrics or a lyrics URL
remain authoritative and bypass transcription.

## Approach

- Relax request validation only for reference-audio requests; MIDI requests
  still require explicit lyrics because there is no audio source to transcribe.
- Download/acquire the reference audio before resolving lyrics.
- Run `faster-whisper` lazily with configurable model, device, compute type,
  language, beam size, and VAD settings, then release the transcriber before
  MuLaCover loads its inference pipeline.
- Write the transcript to the existing `inputs/lyrics.txt` consumed by
  MuLaCover. Record the source (`explicit` or `faster-whisper`), model,
  detected language, segment and word timestamps, and confidence metadata in
  the input manifest and runtime metadata.
- Install `faster-whisper` explicitly in the model-bearing worker image because
  the image installs this project with `--no-deps`.

## Affected Files

- `src/autotransition/mulacover/contracts.py`
- `src/autotransition/mulacover/callback.py`
- `src/autotransition/mulacover/config.py`
- `src/autotransition/mulacover/runtime.py`
- `containers/mulacover-worker/Dockerfile`
- `tests/test_mulacover_worker.py`

## Risks and Verification

- Whisper transcription adds first-use model download and latency; the worker
  must report this through the existing progress stages.
- Keeping Whisper resident on the MuLaCover GPU could cause an avoidable VRAM
  conflict, so the transcriber is created per automatic-transcription job and
  explicitly released before MuLaCover inference.
- Validate explicit lyrics precedence, automatic reference-audio lyrics,
  rejection of lyric-less MIDI jobs, manifest metadata, and existing worker
  behavior with the unit suite. A production image test should use a real
  reference audio only after the local contract tests pass.

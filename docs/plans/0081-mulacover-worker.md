# MuLaCover Worker

## Objective

Add a production-shaped MuLaCover cover/remix worker that accepts reference audio
or symbolic MIDI conditioning, exposes the supported generation controls, reports
durable progress and artifacts through the existing launcher callback contract,
and can be packaged as a public model-bearing GHCR image for Vast or RunPod.

## Approach

- Add an isolated `autotransition.mulacover` package following the existing
  worker boundaries: contracts, configuration/preflight, runtime, durable
  artifacts, progress/callback handling, heartbeat, and HTTP server.
- Use the pinned upstream MuLaCover Python implementation and its documented
  checkpoint layout. Reference audio and MIDI inputs remain mutually exclusive.
  The YourMT3 Space checkpoint follows MuLaCover's documented `main` download
  URL and is verified by its expected byte count; the old immutable Space URL
  no longer resolves even though the checkpoint remains available.
- Accept remote HTTP(S) inputs for reference audio, lyrics, tags, and MIDI files;
  optionally permit local paths only when explicitly enabled for local testing.
- Expose lyrics, style tags, reference-audio BPM, melody/chord/drum MIDI,
  semitone and octave transposition, CFG, temperature, top-k, generation seed,
  decode seed, output format, and the symbolic-artifact export option.
- Keep inference serialized per GPU and use upstream lazy loading. Preflight
  reports GPU memory and an explicit estimated minimum VRAM requirement instead
  of silently falling back to CPU or launching an unsafe run.
- Persist request, preflight, events, symbolic MIDI, runtime metadata, stdout-like
  diagnostics, and final audio artifacts in each job directory. Upload every
  declared artifact through the same callback contract used by the other workers.
- Add a model-bearing Docker build and GitHub Actions workflow using the public
  GHCR package `faceless-mulacover-worker`, pinned source/model revisions, and an
  anonymous manifest verification step.

## Verification

- Unit-test request validation, input-mode exclusivity, path safety, deterministic
  seed handling, preflight memory reporting, and public artifact serialization.
- Run the worker locally on the 10GB environment with synthetic/MIDI contract
  tests and a deliberate `insufficient_vram` preflight; do not claim a local
  audio inference that the GPU cannot safely run.
- Run a Docker build/lint/import/health smoke test where available.
- Trigger the model-bearing GHCR workflow, verify its exact tag is publicly
  pullable, and record the image reference for launcher configuration.

## Risks and tradeoffs

- The official MuLaCover weights and generated outputs have upstream
  non-commercial license restrictions; the image README must surface this.
- Reference-audio conditioning requires YourMT3 and five ChordNet checkpoints;
  MIDI-only jobs can omit their runtime load but the production image includes
  them so both modes work without a second image.
- The image is large. A single model-bearing image is preferred over downloading
  model components at startup because it makes readiness and deployment behavior
  deterministic, but provider disk requirements must account for the bundle.

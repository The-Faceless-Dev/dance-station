# MOSS-Music Analysis Worker

## Objective

Add a worker-shaped runtime for OpenMOSS MOSS-Music that analyzes complete musical
pieces and returns trustworthy, machine-readable musical events for agents and
visual generation/editing. The runtime must preserve enough timing and provenance
to make visual clips align to the audio rather than relying on prose descriptions
or approximate timestamps.

The first implementation will be provider-neutral in its core and compatible with
the existing managed-worker callback/artifact flow. The deployment can later target
Salad or Vast after a real GPU memory and throughput test.

## Findings That Shape The Design

* MOSS-Music currently provides 8B Instruct and 8B Thinking variants. The first
  production profile should use `MOSS-Music-8B-Instruct` for throughput, while the
  model identifier remains configurable so Thinking can be evaluated without a
  code rewrite.
* The official processor config uses `audio_tokens_per_second = 12.5`, which is an
  80 ms audio-representation cadence. It is not a guarantee that the language model
  will emit a correct event every 80 ms. Its default human-readable time markers
  are much coarser, at two-second intervals.
* MOSS-Music is an audio-conditioned language model. It is well suited to semantic
  interpretation, musical explanation, structure, lyrics, instruments, tempo,
  chords, and timestamped events, but its generated text must be treated as an
  untrusted response and validated before the client uses it.
* The official data pipeline combines the language model with deterministic MIR
  tools such as BeatNet, Chordino, Essentia, and SongFormer. The worker should keep
  that boundary: MOSS supplies semantic interpretation, while a dense timeline
  adapter supplies deterministic visual-sync features and can later host stronger
  beat/chord backends.
* The official usage guide recommends the OpenMOSS `moss-audio` SGLang branch for
  quality and throughput. SGLang is the preferred production backend; it is not a
  ComfyUI dependency. A small local/mock backend will be used for contract tests,
  and a Transformers backend may be retained only as an explicit diagnostic mode.

References:

* https://github.com/OpenMOSS/MOSS-Music
* https://raw.githubusercontent.com/OpenMOSS/MOSS-Music/main/moss_music_usage_guide.md
* https://raw.githubusercontent.com/OpenMOSS/MOSS-Music/main/src/processing_moss_music.py
* https://raw.githubusercontent.com/OpenMOSS/MOSS-Music/main/src/configuration_moss_music.py
* https://github.com/wx9Songs/MOSS-Music-Data-Pipeline
* https://arxiv.org/abs/2606.01802

## Runtime Shape

### Core modules

Create a new `src/autotransition/moss_music/` package with small, independently
testable modules:

* `config.py`: environment-backed model, backend, duration, token, resolution,
  concurrency, timeout, and artifact settings.
* `contracts.py`: request, job, progress, artifact, failure, event, and schema
  types. Use seconds as the external time unit and preserve integer sample/frame
  coordinates where useful.
* `audio.py`: acquire a source URL or configured local source, decode with ffmpeg,
  validate duration/sample rate/channels, and create a normalized 16 kHz analysis
  input in a per-job directory.
* `moss_client.py`: pinned SGLang client with request construction, response capture,
  backend health checks, and model metadata. Do not download weights during a job.
* `prompting.py`: versioned task prompts for structured analysis. Prompts are
  configuration/data, not scattered through inference code.
* `parser.py`: strict JSON extraction and schema validation. Preserve the raw model
  response and fail visibly when the response cannot be validated; do not silently
  turn prose into events.
* `timeline.py`: deterministic 80 ms feature grid with absolute timestamps and
  normalized values. Start with RMS/energy, onset strength, spectral bands, spectral
  flux, centroid, chroma, and silence/activity flags using the existing audio
  dependency strategy where possible.
* `merge.py`: combine semantic MOSS events and deterministic timeline data without
  losing source, confidence, or processing-version metadata. Keep duplicate events
  instead of silently collapsing them unless a documented rule applies.
* `artifacts.py`, `worker.py`, `server.py`, and `callback.py`: follow the existing
  durable worker patterns used by `flux_image`, including one GPU job at a time,
  restart-safe state, callbacks, artifact upload, and terminal failure artifacts.

### Request contract

The worker request should support:

```json
{
  "job_id": "...",
  "audio": {
    "url": "https://...",
    "filename": "song.wav",
    "duration_seconds": 183.4
  },
  "analysis_profile": "visual_sync_v1",
  "event_resolution_ms": 80,
  "include_semantic_events": true,
  "include_dense_features": true,
  "moss_model": "configured-default",
  "max_new_tokens": 4096,
  "temperature": 0.0,
  "callback": {}
}
```

The exact transport endpoint will mirror the existing worker conventions, with a
dedicated MOSS namespace. The core request must not depend on Salad-specific fields;
the Salad queue adapter and any future Vast adapter translate provider envelopes
around it.

### Canonical output

The primary `analysis.json` artifact should contain:

* schema version and runtime/model/backend metadata;
* source duration, sample rate, channel count, and the canonical timebase;
* a dense timeline with 80 ms cells (configurable but bounded), each carrying
  acoustic features and activity/silence state;
* sparse events with stable IDs, type, start/end/duration, strength, confidence,
  source, and provenance;
* tempo, beat/downbeat, key, chord, section, lyric, instrument, and voice data when
  available, with uncertainty represented explicitly;
* analysis warnings and coverage so a downstream agent knows what was inferred,
  measured, omitted, or uncertain.

Keep separate top-level collections for `dense_features`, `beats`, `chords`,
`sections`, `lyrics`, `instruments`, `voices`, `visual_cues`, and `annotations`.
This avoids forcing every consumer to interpret a single overloaded event list.

Additional artifacts:

* `moss-response.json` and `moss-raw.txt` for exact model output and debugging;
* `audio-metadata.json` for normalization and decoder details;
* `request.json` with secrets removed;
* `events.jsonl` and `progress.jsonl`;
* optional compact binary feature export only after the JSON contract is stable.

## Progress And Observability

Report real lifecycle stages instead of leaving the launch server at a static
percentage:

1. `validate_request`
2. `acquire_audio`
3. `normalize_audio`
4. `load_backend`
5. `build_timeline`
6. `moss_analysis`
7. `parse_and_validate`
8. `merge_analysis`
9. `write_artifacts`
10. `upload_artifacts`

Every stage records start/end timestamps, elapsed time, input/output sizes, model
and backend identifiers, token counts when available, GPU name/memory where
available, and the last successful substep. Failures must include the exception
type, stage, traceback, request fingerprint, backend health, and paths to retained
debug artifacts. Do not log secret callback tokens or signed URLs.

## GPU And Deployment Strategy

* Load exactly one MOSS model per worker process. Never load Instruct and Thinking
  together.
* Prefer a single 24 GB+ GPU with no CPU model offload for the first production
  profile. A 3090 is a possible target but should be measured; a 5090/32 GB class
  offer is the preferred high-throughput target if its cost is justified.
* Bound model context, output tokens, and concurrent jobs. Default concurrency is
  one because this is an inference worker, not a batch server.
* Keep model weights in the image or an explicitly managed persistent volume. A job
  must not trigger a large network download.
* Add startup preflight checks for CUDA, available VRAM, model readability, ffmpeg,
  SGLang health, and a tiny audio request. Fail before accepting paid work when the
  target GPU cannot meet the selected profile.
* Clean each job's temporary decoded audio and intermediate arrays in `finally`
  blocks. Preserve only the declared artifacts and failure diagnostics.
* Build the container with a model-bearing base and a small code/dependency layer,
  but do not add provider-specific image behavior to the core runtime. The final
  provider choice comes after measured local/remote startup and inference results.

### Initial VRAM Estimate

The downloaded Instruct checkpoint reports 9,052,463,488 parameters and
18,104,928,256 bytes of BF16 weights, or approximately 16.86 GiB before the
runtime, CUDA workspace, audio-encoder activations, and KV cache are counted.
The language configuration has 36 layers, 8 KV heads, and a 128-wide head. At
BF16 KV precision that is approximately 147,456 bytes per cached token, or
about 0.14 MiB per token. The processor represents audio at 12.5 tokens per
second, so a typical five-minute song contributes about 3,750 audio tokens
before the prompt and generated answer are added.

This makes the practical deployment tiers:

* **24 GiB:** theoretical single-request floor, but borderline. It requires
  one request at a time, bounded context/output, and careful SGLang static
  memory allocation. It is not the "run perfectly" target.
* **32 GiB:** recommended minimum for one full-song request with no CPU model
  offload, BF16 weights, and a bounded context. This is the first profile to
  benchmark on a 5090-class GPU.
* **48 GiB:** preferred production profile for long songs, larger context
  limits, or headroom for CUDA/SGLang workspace and future concurrency.

The estimate is derived from the actual checkpoint/config and SGLang's static
memory model; the final choice must still be confirmed by a full-song run that
records peak VRAM. SGLang's `--mem-fraction-static` and `--max-total-tokens`
remain runtime settings so the same image can be profiled on 24, 32, and 48
GiB GPUs without rebuilding it.

## Implementation Phases

1. Add the contracts/configuration/artifact schema and a fake backend that produces
   deterministic fixture responses.
2. Implement audio normalization and the bounded 80 ms dense timeline adapter,
   reusing or extending existing audio-analysis helpers instead of duplicating
   rhythm-beat logic.
3. Implement strict MOSS structured-output prompts, parser, validator, and merge
   rules while retaining the raw response.
4. Implement the durable worker, HTTP status/health endpoints, callback adapter,
   heartbeat, and granular progress events.
5. Add the pinned SGLang runtime and startup preflight. Keep the model path and
   backend selection configurable for Instruct/Thinking comparisons.
6. Add unit, contract, failure, idempotency, artifact, and integration tests.
7. Run a real full-song GPU benchmark, record peak VRAM, audio duration, backend
   load time, timeline time, MOSS generation time, total elapsed time, and output
   validity. Use that evidence to choose Salad versus Vast and the production GPU
   profile.
8. Only after the worker passes the full-song test, add the provider deployment
   image and launcher integration.

## Risks And Decisions To Preserve

* An 80 ms grid does not make LLM-generated beat/chord timestamps exact. Consumers
  should use deterministic measured features for frame-accurate visual timing and
  treat MOSS events as interpretation with confidence/provenance.
* Full-song structured output may exceed a practical token budget. The runtime must
  support bounded task-specific passes or time-windowed semantic analysis later,
  with absolute timestamp offsets and deterministic merge behavior. It must not
  silently truncate a song.
* JSON is the initial interoperability format because agents and the site can use
  it directly. Large dense arrays may later gain a compact companion artifact,
  while `analysis.json` remains the canonical manifest.
* A Transformers fallback should not be enabled implicitly in production. If it is
  retained for diagnostics, it must be explicit and clearly reported because CPU
  fallback would violate the high-speed worker profile.

## Verification Criteria

The worker is ready for deployment only when all of the following are true:

* a full-song request completes on the selected GPU without CPU model offload;
* the output validates against the versioned schema;
* the dense timeline covers the complete source duration at the requested bounded
  resolution with no timestamp gaps or overlaps beyond the documented endpoint
  rule;
* semantic events retain raw-response provenance and never become fabricated data
  after a parse failure;
* progress callbacks identify the active stage and continue moving during long
  analysis steps;
* a forced backend failure produces a useful terminal failure and does not leave a
  job stuck in `running`;
* retrying the same external job ID is idempotent;
* all declared artifacts are retained locally and uploadable through the existing
  launch-server callback contract;
* the real benchmark records enough VRAM, timing, and throughput data to make the
  Salad/Vast decision from evidence.

## Implementation Status

Completed in the first runtime pass:

* `src/autotransition/moss_music/` worker package with contracts, configuration,
  audio normalization, dense timeline extraction, strict response parsing, SGLang
  client, durable artifacts, callbacks, HTTP endpoints, and heartbeat support;
* model-bearing container definition with optional SGLang autostart, Salad queue
  mode, and direct HTTP mode for a future Vast adapter;
* focused unit, callback, HTTP smoke, parser, timeline, and failure tests.

Verified in the local readiness pass:

* staged and integrity-checked the four-shard 18.1 GB
  `MOSS-Music-8B-Instruct` checkpoint;
* built a model-bearing local image with the pinned OpenMOSS SGLang branch,
  corrected entrypoint metadata, and CuDNN 9.16 loaded by PyTorch;
* passed container `/health`, `/ready`, `/v1/worker/status`, and a complete
  mock audio job producing all seven declared artifacts.

The first real 5090 startup exposed two runtime-image assumptions that local
mock tests could not catch: the Blackwell import path attempted to initialize
DeepGEMM even though MOSS is a dense BF16 model, and SGLang's CUDA-graph capture
requires a C compiler for Triton. The container now disables DeepGEMM unless it
is explicitly enabled and includes `build-essential` with `CC=gcc` and `CXX=g++`.
The first issue was reproduced and fixed; the compiler fix is awaiting a second
real 5090 startup and inference test.

Remaining before a production publish:

* run the real full-song BF16 GPU benchmark on a 32 GiB or larger target and
  record peak VRAM, audio duration, backend load time, timeline time, MOSS
  generation time, total elapsed time, and output validity;
* use that benchmark to choose Salad versus Vast and the final GPU profile;
* integrate the verified runtime with launch-server pricing and client workflows.

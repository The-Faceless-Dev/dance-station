# MOSS Full Output and Startup Reliability

## Objective

Make the MOSS-Music worker return the complete useful analysis for audio up to
the existing 30-minute limit, remove the accidental SGLang 4096-token default,
and make the published image start through its normal entrypoint on Vast without
manual shell intervention.

## Approach

- Keep the existing complete 80 ms measured timeline as a local worker artifact.
- Remove the request-level `max_new_tokens` diagnostic override and do not impose
  an arbitrary response-token ceiling in the worker.
- Set the SGLang request budget from the backend's actual context capacity and
  prompt token count, so omission means maximum available context rather than
  SGLang's 4096 default. If SGLang reports its exact serialized input count,
  recover the remaining context and retry without imposing a fixed ceiling.
  Record the resolved context metadata.
- Preserve every model response as raw output. Build a best-effort normalized
  semantic view for convenience, but never retry or reject a job because the
  model returned malformed JSON or a length-marked response. Keep parse warnings
  and the original text alongside the successful full result.
- Make the container entrypoint usable with both the normal image command and
  Vast's SSH-direct launch mode, including writable HOME/cache setup and a
  readiness wait before the external queue/heartbeat process is started.
- Add unit tests for resolved output budgeting, full-schema requests, startup
  behavior, and artifact completeness.

## Affected Files

- `src/autotransition/moss_music/config.py`
- `src/autotransition/moss_music/runtime.py`
- `src/autotransition/moss_music/contracts.py`
- `src/autotransition/moss_music/callback.py`
- `containers/moss-music-worker/entrypoint.sh`
- `containers/moss-music-worker/Dockerfile`
- `containers/moss-music-worker/README.md`
- `tests/moss_music/test_moss_music.py`

## Risks and Verification

- MOSS may still produce invalid or repetitive JSON for a long song even when
  the backend can use its full context. The parser must expose that failure and
  preserve diagnostics rather than silently returning partial data.
- The model context is a backend capacity, not an arbitrary audio limit; the
  existing 30-minute audio limit remains the only user-facing duration cap.
- Run the complete local test suite and a production-shaped Vast run using
  `D:\goreset\crown.mp3`. Verify all artifacts and destroy the paid instance
  after the run.

## Follow-up From First Full Run

The first recovery image proved startup and GPU inference, but the full-song
harmony pass consumed the checkpoint context completion budget and was rejected
as truncated. The runtime now keeps the caller free of token controls while
using configurable audio windows for long audio, offsetting timestamps back
onto the original song timeline, and merging the best-effort structured view.
The default semantic window is 60 seconds and is controlled by worker
configuration rather than a caller token limit. All four semantic passes use
that bounded path for long audio. Each pass requests only its own output fields,
preventing the model from emitting unrelated event grids. Segment timestamps
are bounded when they can be normalized, while every original model response is
always retained as a raw artifact. Malformed or length-marked JSON is a warning,
not a second inference and not a job failure.

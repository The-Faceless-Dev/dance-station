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
  SGLang's 4096 default. Record the resolved context metadata.
- Preserve the full MOSS prompt/schema and reject only genuinely incomplete
  responses; retain raw and response metadata on failure.
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

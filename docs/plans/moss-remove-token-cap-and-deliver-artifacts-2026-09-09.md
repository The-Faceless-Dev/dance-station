# Remove MOSS Token Cap And Complete Artifact Delivery

## Objective

Keep the MOSS-Music audio limit at 30 minutes, remove the arbitrary semantic
output-token ceiling, and make the worker deliver a complete, validated result
through the normal callback contract without manual artifact recovery.

## Approach

1. Remove `max_new_tokens` as a worker/configuration limit. Treat it as an
   optional backend override only when explicitly supplied by a caller; omit it
   from SGLang sampling parameters by default so generation can finish on the
   model's stop condition.
2. Preserve the 30-minute audio limit and existing transport/security guards.
3. Detect SGLang length-truncated responses and reject them before building a
   successful analysis. Do not accept a partial nested JSON object as a full
   semantic result.
4. Make the canonical `analysis.json`, raw response, runtime metadata, request,
   and event log upload as one declared artifact set. Return both artifact IDs
   and the uploaded artifact manifest from `/process` completion.
5. Expose the same complete artifact manifest from direct job status, including
   worker-relative download URLs and no container-local paths, so a direct Vast
   caller can retrieve the result without searching the container.

## Affected Files

- `src/autotransition/moss_music/config.py`
- `src/autotransition/moss_music/contracts.py`
- `src/autotransition/moss_music/server.py`
- `src/autotransition/moss_music/callback.py`
- `src/autotransition/moss_music/runtime.py`
- `src/autotransition/moss_music/parser.py`
- `src/autotransition/moss_music/worker.py`
- `tests/moss_music/*`
- `containers/moss-music-worker/README.md`

## Risks And Verification

- Omitting `max_new_tokens` must be accepted by the pinned OpenMOSS SGLang
  endpoint; the integration smoke test will assert the request shape.
- A malformed or truncated response must fail with raw output and diagnostics,
  never silently produce an incomplete successful analysis.
- The existing seven-artifact callback flow must remain compatible with the
  launch server. Focused tests run before publishing.
- Publish the model-bearing image through the existing MOSS workflow, verify
  the public image manifest, run a complete Crown analysis on one RTX 5090 Vast
  instance, retain the callback response and artifacts under `tmp/`, then
  destroy the paid instance.

# ACE-Step Repaint Transition

## Problem

The current working generation path uses ACE-Step `text2music` and then appends
the generated clip after the selected source point. That can create a valid
new music clip, but it is not conditioned on the selected source audio. The
model has no audio context for the join, so a blank intro or unrelated opening
can land directly after the marker.

Autotransition should use ACE-Step's audio-conditioned edit path for the actual
transition. This is not a trimming/crossfade problem; the transition should be
created by the model.

## ACE-Step Behavior To Use

ACE-Step documents `repaint` as the task for regenerating a selected time
segment while keeping the rest of the source audio. The runtime request uses:

- `task_type=repaint`
- uploaded `src_audio`
- `repainting_start`
- `repainting_end`
- `prompt` / `lyrics`

The runtime locks repaint duration to the uploaded source audio length and
passes the repaint window into the DiT generation path. It also has repaint
controls for:

- `chunk_mask_mode`
- `repaint_latent_crossfade_frames`
- `repaint_wav_crossfade_sec`
- `repaint_mode`
- `repaint_strength`

For an append transition, the source audio sent to ACE must already span the
desired output length. The repaint window should cover the append boundary and
the new generated section, so ACE has real audio before the boundary and a
target region to regenerate.

## Intended Pipeline

1. Keep the now-working `text2music` path only as a content proposal generator.
   It can create the future section, but it is not the final transition.

2. Build an ACE repaint scaffold:
   - source audio up to the selected continuation point
   - generated future section after that point

3. Submit that scaffold to ACE-Step `repaint`:
   - `repainting_start = continuation_point - repaint_context_before_seconds`
   - `repainting_end = continuation_point + repaint_context_after_seconds`
   - clamp the start/end to the scaffold duration
   - use the same prompt, BPM/key hints, seed, and exposed ACE settings

4. Save all stages:
   - raw text-to-music candidate
   - pre-repaint scaffold
   - ACE repaint output
   - request/response JSON for both ACE calls
   - metadata with repaint window and settings

5. Return the ACE repaint output as the primary result.

## Request Rules

- Do not post-process the transition with custom trimming or waveform crossfade
  as the primary fix.
- For this transition repaint path, do not force `/v1/init` or a model name
  unless the user explicitly selects that behavior. The text-to-music fix showed
  that forcing a different model can break the working runtime path.
- Send `src_audio` as multipart upload, not an absolute path.
- Set `audio_duration` only if ACE requires it; repaint should use the uploaded
  source duration.
- Preserve ACE repaint defaults unless we intentionally expose and document a
  control.

## Affected Files

- `src/autotransition/models/acestep_api.py`
- `src/autotransition/models/acestep.py`
- `src/autotransition/audio/compose.py`
- `src/autotransition/pipeline.py`
- `src/autotransition/config.py`
- `src/autotransition/ui/app.py`
- `src/autotransition/ui/static/app.js`
- `tests/test_acestep_api.py`
- `tests/test_ui_api.py`
- audio composition/scaffold tests as needed

## UI Changes

- Keep the user-facing action as generation, not scaffold creation.
- Results should list each generated candidate and let the user play it in the
  UI.
- Add clear status labels for the ACE stages:
  - generating candidate
  - repainting transition
  - ready / failed
- Expose repaint-specific controls only when they map directly to ACE-Step
  parameters.

## Risks

- The active runtime model may support `repaint` but still behave differently
  from `text2music`; request JSON must be saved so failures can be compared
  directly against working curl tests.
- Repainting the entire generated section may change the candidate more than
  expected. Start with a boundary/new-section repaint window and keep the raw
  candidate for inspection.
- If a selected model is forced through `/v1/init`, the runtime can switch away
  from the model that produced working music. Avoid that in the default path.

## Verification

- Add tests that repaint requests omit absolute source paths and upload
  `src_audio`.
- Add tests that transition generation calls text-to-music first, then repaint
  with a scaffold and a repaint window crossing the continuation point.
- Run the full test suite.
- Manually verify one real run saves raw candidate, repaint scaffold, repaint
  output, and request JSON for both ACE calls.

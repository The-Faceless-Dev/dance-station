# Model-Aware Repaint Defaults Plan

## Problem

ACE-Step 1.5 Base repaint output can become garbled when Autotransition relies on ACE-Step server defaults. Base/SFT models are more sensitive than turbo models to guidance, shift, repaint strength, and boundary blending. Autotransition currently sends only the minimum repaint fields:

- task type
- repaint bounds
- prompt
- model
- duration
- batch size
- inference steps

It does not send repaint mode/strength, crossfade, guidance, shift, or inference method explicitly.

## Approach

- Add a small request-default helper in the ACE-Step API client.
- Send explicit repaint defaults for all models:
  - `repaint_mode`
  - `repaint_strength`
  - `repaint_latent_crossfade_frames`
  - `repaint_wav_crossfade_sec`
  - `guidance_scale`
  - `shift`
  - `infer_method`
- Use more conservative defaults for non-turbo base/SFT models.
- Raise base/SFT default inference steps into ACE-Step's recommended range.
- Add unit coverage that base and turbo send different defaults.

## Affected Files

- `src/autotransition/models/registry.py`
- `src/autotransition/models/acestep_api.py`
- `tests/test_acestep_api.py`

## Tradeoffs and Risks

- These defaults are still heuristics; generated music quality can vary by prompt/source.
- Conservative repainting may preserve more of the source overlap and transition more gradually, which is preferable to noisy output for the creator-facing default.

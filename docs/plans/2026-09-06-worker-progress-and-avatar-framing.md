# Worker Progress and Avatar Framing

## Objective

Improve production observability without changing the Wan inference process, and
make the Flux image worker return avatar references in the canonical `480x832`
format with a smaller subject and matching solid-background padding.

## Approach

- Add an internal progress callback to the Wan worker pipeline.
- Convert Wan render-window, inference-step, VACE bridge, assembly, and
  post-processing milestones into monotonic launch-server progress updates.
- Preserve the existing model settings, windowing, seeds, prompts, and runtime
  commands.
- Add Flux output framing after inference: preserve the generated image, fit the
  subject image to a `480x832` canvas at approximately 70% of the available
  content area, and fill the added border with the sampled/generated background
  color rather than introducing a new prompt or model step.
- Record the requested and effective Flux dimensions in metadata and tests.

## Affected Areas

- `src/autotransition/generative_dance/`: Wan progress propagation.
- `tools/generative_dance/wan_animate_2_runner.py`: render progress events.
- `src/autotransition/flux_image/`: output framing and metadata.
- `tests/`: focused worker and framing coverage.
- `containers/flux-image-worker/`: runtime defaults and image publish inputs if
  required.

## Risks and Verification

- Progress must remain monotonic and must not alter inference or callback
  completion behavior.
- Framing must not crop the generated subject and must always produce exactly
  `480x832`.
- Run focused tests, then validate the Flux container build/runtime contract.
- Keep the existing Wan Vast image and Salad Flux package names unchanged.

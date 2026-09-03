# Wan T5 Cache and LightX2V Quality Fix

## Objective

Fix the two regressions observed in the RTX 5090 Wan Animate run:

1. Encode T5 conditioning once per source clip/prompt and reuse it across all temporal windows.
2. Verify and correct the LightX2V integration so it follows the official adapter and sampling contract without degrading identity, detail, or temporal quality.

## Scope

- `tools/generative_dance/wan_animate_2_runner.py`
- `tools/generative_dance/wan_animate_2_runtime.py`
- Focused tests for conditioning reuse and LightX2V configuration/weight handling.
- A short single-clip production-shaped validation using the existing worker test harness.

## Approach

- Add an explicit per-segment conditioning object/cache. T5 output is prompt-dependent, not temporal-window-dependent, so the same tensor will be passed to every window in that clip.
- Preserve separate conditioning when adjacent source clips use different prompts.
- Audit LightX2V target-name normalization, LoRA tensor orientation/scaling, base-model dtype/device placement, and the official four-step sigma schedule against primary Wan/LightX2V sources and the current runtime implementation.
- The audit found that the runtime treated LightX2V's denoising indices
  `[1000, 750, 500, 250]` as final sigmas and skipped the required
  `sample_shift=5.0` transform. It also did not scale direct `.diff` and
  `.diff_b` tensors by the configured adapter strength.
- Keep the existing CUDA-only attention requirements and RTX 5090 target unchanged. No slow attention fallback will be introduced.
- Add diagnostics that report T5 encode count, LightX2V adapter counts, applied strength, sampling schedule, and active attention backends.

## Tradeoffs and Risks

- Reusing T5 conditioning lowers runtime and memory churn but requires strict invalidation when the prompt or text length changes.
- Changing LightX2V application can alter output character identity and motion fidelity; the short test will compare representative frames before any full production run.
- The 5090 has limited headroom with the Q6 model, so the fix must not permanently retain additional large model copies on CUDA.

## Acceptance Criteria

- A 3-second single-clip run completes successfully with nonblank output.
- T5 encode count is one for the clip, not one per temporal window.
- The run reports LightX2V enabled with the expected checkpoint, the shifted
  four-step schedule, CUDA device, and required attention backends.
- Random output frames are visually inspected for identity drift, severe artifacting, black/blank output, and obvious temporal discontinuity.
- No paid worker is left running after validation.

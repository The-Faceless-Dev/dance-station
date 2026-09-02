# Wan Animate LightX2V and Artifact Contract Fix

## Objective

Correct the Wan-Animate-2 Vast worker so the accelerated 5090 profile actually
uses the official Wan-Animate-2 LightX2V distillation LoRA with its compatible
four-step sampling profile, while making successful and failed outputs
uploadable through the launch-server callback contract.

## Approach

- Verify the official `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`
  checkpoint and its target module/key layout before adding it to the image.
- Add a strict worker configuration for the LoRA path, strength, and steps.
  When acceleration is enabled, the worker must fail during preflight if the
  LoRA is missing or incompatible; it must not silently run the slow profile.
- Keep FlashAttention and the official compiled FlexAttention path required for
  the Vast 5090 image, with no CPU/eager attention fallback.
- Store the LightX2V weights as a separate cacheable image layer and preserve
  the existing public image name/base layers.
- Expand the launch-server artifact variant contract for the worker's
  namespaced final, diagnostic, and failure artifacts. Log each callback
  attempt, response, and failure with the job, variant, filename, bytes, and
  elapsed time.

## Verification

- Run static/config tests and inspect the LoRA keys without loading the 14B
  model locally.
- Publish through the existing repository GitHub Actions credential path and
  verify the resulting GHCR image is publicly pullable.
- Run the existing short two-window `doll-17f.mp4` Vast request on an RTX 5090.
  Confirm logs show LightX2V enabled, four steps, required attention
  backends, successful artifact callbacks, a valid non-black MP4, and recorded
  timing.
- Download the returned artifacts into the test directory and destroy the Vast
  instance after verification.

## Risks

- The LightX2V LoRA must match the Wan-Animate-2 Q6 architecture. A key or shape
  mismatch is a hard configuration error, not a reason to omit the LoRA.
- The launch-server schema change must be backward-compatible with all existing
  artifact variants and migrations.

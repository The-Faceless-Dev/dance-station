# Qwen Image White Output Fix

## Objective

Make the Q8 Qwen-Image-2512 worker produce real images on the RTX 6000 Ada
RunPod target instead of completing with uniform-white PNGs.

## Findings

- CUDA initializes the RTX 6000 Ada correctly.
- The Q8 GGUF contains 1,933 diffusion tensors, including six BF16 input,
  timestep, normalization, and output-projection weights.
- The current native runtime prepares only 1,927/1,933 tensors.
- Disabling mmap did not change the six-tensor omission, so mmap is not the
  root cause.
- The worker wrapper starts the newer stable-diffusion.cpp lazy parameter path
  without an explicit eager-load or segmented-scheduling policy.
- Controlled 1024x1024 generations with 4, 20, and 40 steps all returned
  exact uniform-white PNGs. Disabling FlashAttention, eager loading, and
  segmented compute did not change that result.
- The pinned stable-diffusion.cpp source already applies Qwen's 16-channel
  latent mean/std conversion in `WanVAERunner::diffusion_to_vae_latents`.
  The missing-normalization hypothesis is therefore not established.
- The native VAE path clamps decoder output to `[0, 1]` before PNG conversion.
  That can hide NaN/Inf or saturation as a valid white image, so the final PNG
  does not identify which stage is broken.
- The diagnostic image build completed successfully, but the workflow's
  anonymous pull check returned HTTP 401 because its pending publicization
  step had been removed. The image is therefore built but not yet usable by a
  public RunPod caller.

## Approach

1. Keep the existing runtime settings and add a gated native tensor diagnostic
   patch to the existing model-bearing build. It must log sampled latent,
   denormalized VAE latent, raw decoder output before clamping, and scaled
   decoder output with shape/min/max/mean/std and NaN/Inf counts.
2. Preserve the existing model files, CUDA image, request schema, LoRA
   support, and production defaults. Enable the diagnostic only for the
   controlled investigation run.
3. Extend focused runtime/config tests and documentation.
4. Run all local Qwen tests and static checks available on this machine.
5. Make the already-published diagnostic tag public through the repository CI
   credential, verify its manifest, and test it on the current single RTX
   6000 Ada pod without rebuilding the model layers.
6. Use the measured failing boundary to implement and validate the actual
   correction before declaring the worker fixed.

## Risks

- Eager loading increases startup/RAM use, but the 48 GB target has sufficient
  headroom and parameters remain CPU-resident when CPU offload is enabled.
- Disabling segmentation may reduce memory flexibility on smaller GPUs; the
  settings remain environment-configurable for future targets.
- Native diagnostics add a linear pass over host tensors but are disabled for
  normal production runs; the diagnostic run is intentionally slower only by
  a negligible amount relative to inference.
- The local machine cannot run the full Q8 inference, so the paid validation
  remains necessary for the final GPU-specific check.

## Acceptance Criteria

- Native log shows all 1,933 Qwen diffusion tensors prepared/staged.
- A controlled 1024x1024 generation is non-uniform and visually valid.
- The worker returns and retains the normal artifacts and logs.
- The paid test pod is stopped after artifacts are collected.

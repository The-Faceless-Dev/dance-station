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

## Approach

1. Add explicit Qwen runtime settings for eager parameter loading and disabling
   segmented compute, enabled by default for the Qwen model-bearing worker.
2. Pass the settings to stable-diffusion.cpp without changing model files,
   CUDA image, request schema, LoRA support, or CPU-offload behavior.
3. Extend focused runtime/config tests and documentation.
4. Run all local Qwen tests and static checks available on this machine.
5. Publish a new model-bearing image through the existing Qwen GitHub Actions
   workflow, verify its manifest, and test it on a single RTX 6000 Ada pod.

## Risks

- Eager loading increases startup/RAM use, but the 48 GB target has sufficient
  headroom and parameters remain CPU-resident when CPU offload is enabled.
- Disabling segmentation may reduce memory flexibility on smaller GPUs; the
  settings remain environment-configurable for future targets.
- The local machine cannot run the full Q8 inference, so the paid validation
  remains necessary for the final GPU-specific check.

## Acceptance Criteria

- Native log shows all 1,933 Qwen diffusion tensors prepared/staged.
- A controlled 1024x1024 generation is non-uniform and visually valid.
- The worker returns and retains the normal artifacts and logs.
- The paid test pod is stopped after artifacts are collected.

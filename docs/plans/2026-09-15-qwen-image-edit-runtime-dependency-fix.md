# Qwen Image Edit Runtime Dependency Fix

## Objective

Make the Qwen Image Edit 2511 worker fail during startup preflight, with a
useful diagnostic, instead of accepting a job and failing while loading the
Diffusers Edit Plus pipeline.

## Approach

- Install the CUDA-compatible `torchvision` wheel alongside PyTorch in the
  model-bearing worker image.
- Include `torchvision` and the Qwen Edit Plus pipeline imports in the startup
  preflight report so provider logs expose missing or incompatible dependencies
  before a paid inference request runs.
- Add regression tests for the dependency declaration and preflight checks.
- Rebuild through the existing public GHCR workflow, verify the corrected tag,
  then run the multi-reference edit test on one eligible RunPod GPU.

## Affected Files

- `containers/qwen-image-edit-worker/Dockerfile`
- `containers/qwen-image-edit-worker/entrypoint.sh`
- `src/autotransition/qwen_image_edit/runtime.py`
- `tests/qwen_image_edit/test_entrypoint.py`
- `tests/qwen_image_edit/test_runtime.py`

## Risks

- The CUDA wheel must match the PyTorch wheel selected from the `cu128` index.
- The extra wheel slightly increases the dependency layer, but it is required
  by the actual Edit Plus processor and does not change the model layer.

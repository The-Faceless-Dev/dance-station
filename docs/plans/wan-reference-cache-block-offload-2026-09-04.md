# Wan reference cache and block offload

## Objective

Allow the native Wan-Animate-2 GGUF runner to process long reference windows on
the target 32 GB+ GPUs without keeping the complete transformer working set and
the complete reference K/V cache in VRAM.

## Approach

1. Add a host-backed reference cache that stores each completed layer's K/V
   tensor on pinned CPU memory and stages only the active layer onto CUDA.
2. Add block-level residency management around the official `forward_ref` and
   `forward_gen` loops. The active block and its temporary tensors stay on CUDA;
   inactive blocks are returned to CPU after each layer.
3. Keep the existing GGUF lazy dequantization, Flash/Flex attention, VAE
   offload, reference-strength scaling, and continuation behavior unchanged.
4. Add stage-level memory logs for cache host/device bytes, active block, and
   residency transitions so future OOMs identify the real allocation.
5. Add unit coverage for host cache staging/release and the default memory
   policy, then run the existing generative-dance test suite and syntax checks.

## Tradeoffs and risks

- Host-backed caching and block transfers add PCIe traffic and may reduce speed,
  but they avoid impossible 32 GB allocations.
- The implementation must preserve the official block ordering and cache
  indexing exactly; changing the mathematical attention path would risk output
  quality.
- The default will apply only to CUDA native Wan inference. CPU/tests retain a
  no-op path.
- A full 81-frame CUDA validation remains required after implementation. The
  current saved diagnostic run is incomplete because its diagnostic hook raised
  a `TypeError` after block zero.

## Affected files

- `tools/generative_dance/wan_animate_2_runner.py`
- `tools/generative_dance/wan_animate_2_runtime.py` if the loader needs a
  block-residency helper
- `tests/test_generative_dance.py`
- Container/runtime environment documentation if new variables are exposed

# LTX Video Residency And Reset

## Objective

Make the LTX 2.5 worker complete the expected 10-second text-to-video job on
the 48 GB RunPod MIG target without silently falling back to CPU, and make
GPU cleanup usable and verifiable after both successful and failed jobs.

## Observed Failure

The 9-frame smoke job completed with strict video Flash SDP, NVFP4, and no
offload. A 10-second request at 24 FPS resolves to 241 frames (`8 * K + 1`).
At `576x1024`, denoising completed, but ConvVAE decode reached approximately
`46.5 GB` allocated on a `47.4 GB` device and raised a PyTorch CUDA caching
allocator assertion. The preflight estimate reported `24.75 GB` because it
treated text encoding, denoising, and decode residency as non-overlapping.

The request with intermediate retention disabled failed the same way, so the
stage-1 preview was not the only cause. Failed requests also left most CUDA
allocation resident in the worker process; later requests saw less than one
GB free and were rejected before model loading.

## Approach

* Add explicit lifecycle boundaries around prompt encoding, denoising, latent
  upscale, and VAE decode so model modules and request tensors are released
  before the next phase when the upstream lifecycle permits it.
* Stream or temporally tile final VAE decode with a bounded working set and
  log the resolved tiling configuration. Do not make a long-video request
  appear safe using only the current underestimated phase estimate.
* Record per-phase allocated, reserved, and peak VRAM, plus cleanup results,
  in the durable event and metadata artifacts.
* Add a protected worker reset endpoint that is available only when no job is
  active. It clears request state and CUDA caches and reports measured memory;
  it is not a slow-attention fallback.
* On failed jobs, preserve diagnostics and mark the worker as requiring a
  reset if the allocator cannot be returned to the configured reserve. The
  worker must not accept another expensive job while its memory state is
  unsafe.
* Keep strict Flash SDP for video. Gemma prompt encoding uses an explicitly
  scoped math-SDP context, including a temporary math-backend enable because
  the global strict video policy disables it; all SDP flags are restored before
  denoising. Keep audio disabled for this request and do not add CPU offload.

## Affected Files

* `src/autotransition/ltx_video/runtime.py`
* `src/autotransition/ltx_video/worker.py`
* `src/autotransition/ltx_video/server.py`
* `src/autotransition/ltx_video/memory.py`
* `tests/ltx_video/`
* `containers/ltx-video-worker/Dockerfile.overlay`

## Verification

* Run focused unit tests, compilation, and the existing LTX test suite.
* Validate a clean worker preflight and reset behavior without model weights.
* Publish a code-only overlay from the known-good model-bearing lineage and
  verify its manifest before renting a GPU.
* Run the exact no-conditioning, 10-second T2V request on one capped RunPod
  48 GB MIG Pod. Preserve the full request, progress, diagnostics, and final
  video artifacts. Leave the Pod running only after a successful artifact
  verification; destroy it on a confirmed failed test.

## Upstream Reference

LTX documents the `8 * K + 1` temporal grid and recommends resolutions below
`720x1280` and frame counts below `257`. The LTX-2 release notes also describe
recent improvements to temporal decode budgeting and long-video chunking.

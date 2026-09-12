# LTX Gemma Attention Isolation

## Objective

Make the LTX 2.5 video worker run the Gemma prompt encoder with the attention
backend it can actually execute on the target 5090, without weakening the
strict fast-attention policy used by the video transformer.

## Cause

The production smoke test passed CUDA, NVFP4, and Flash SDP preflight, then
failed inside `transformers` Gemma 4 prompt encoding with `No available
kernel`. The worker currently enables Flash SDP globally, so Gemma inherits a
backend that is not valid for that encoder call.

## Approach

* Keep the existing strict Flash-only CUDA policy for LTX video denoising.
* Wrap only the Gemma prompt-encoding call in an explicit math-SDPA context,
  using PyTorch's modern attention context API with its legacy equivalent as
  an API-version compatibility implementation detail.
* Restore and log the strict video policy immediately after prompt encoding.
* Fail clearly if the isolated Gemma policy cannot be installed; never turn on
  a slower global fallback for video.
* Add unit coverage with a small fake PyTorch backend so the policy boundary
  and restoration behavior are tested without model weights or CUDA.

## Affected Files

* `src/autotransition/ltx_video/runtime.py`
* `tests/ltx_video/test_runtime_attention.py`

## Risks and Verification

Math SDPA is intentionally limited to the short Gemma text-encoder pass. It
may be slower than Flash for text encoding, but avoids the production failure
and does not affect video denoising. Run the LTX tests, the full test suite,
Python compilation, then publish a code-only image update and verify a real
video-only Vast generation before reporting success.

## Publishing Correction

The existing model-bearing workflow is now manual-only. Runtime changes use
`Dockerfile.overlay` and `publish-ltx-video-worker-code-overlay.yml` against
the public model-bearing tag, so a code fix adds one small layer instead of
rebuilding the model bundle.

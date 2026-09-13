# Qwen FP8 Text Encoder Test

## Objective

Test `qwen_2.5_vl_7b_fp8_scaled.safetensors` with the existing public Qwen-Image-2512 Q8 transformer and VAE to determine whether the text encoder is contributing to the all-NaN/white image failure.

## Approach

* Reuse the verified public Qwen worker image by digest.
* Add only the 9.38 GB FP8 text-encoder layer in a derived test image.
* Keep the existing worker runtime and request contract unchanged.
* Disable the filename-based abliterated-encoder requirement for this test image.
* Run a real image request with native tensor diagnostics enabled.
* Save the request, preflight, event log, runtime log, metadata, and image under `tmp/`.
* Validate the output pixels for finite, non-uniform image data before treating the run as successful.

## Risks and rollback

The FP8 file may be structurally incompatible with the Qwen Image native loader or may not resolve the NaN failure. The production Q8 image remains untouched, and the derived image can be abandoned without changing the current package tag.

## Verification

The test is successful only if the worker reaches ready, the native job completes, the output is a valid PNG, and its pixel statistics show finite non-uniform data. The paid test pod will be destroyed after the artifacts are copied locally.

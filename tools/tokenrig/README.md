# Adaptive TokenRig Runner

`adaptive_runner.py` launches the upstream SkinTokens checkout without copying
model or checkpoint files into this repository. The small upstream runtime
adapter reads `SKINTOKENS_ATTENTION` so the runner can select a compatible
attention backend per GPU.

The default `auto` profile uses:

* Less than 14 GiB VRAM: one beam and inference-only execution for the 10 GiB RTX 3080.
  It uses PyTorch SDPA, which is the reliable low-memory CUDA path for the
  current Qwen3/RTX 3080 combination.
* At least 14 GiB VRAM: the upstream ten-beam quality setting with
  FlashAttention 2, with no low-VRAM downgrade.

The runner also enables PyTorch expandable allocator segments to reduce fragmentation. It does not pretend that allocator settings create additional VRAM.

Run it from WSL2/Linux where FlashAttention's supported environment is more reliable:

```bash
python tools/tokenrig/adaptive_runner.py \
  --skintokens-repo /mnt/d/SkinTokens \
  --input /mnt/d/dance-avatar-poc/sf3d-character1/0/mesh.glb \
  --output /mnt/d/dance-avatar-poc/tokenrig/character1.glb
```

The SkinTokens environment used for the tested path is Python 3.11 with
`torch==2.5.1` (CUDA 12.4), `transformers==4.57.1`, `bpy`, and
`flash-attn --no-build-isolation`. SkinTokens currently resolves newer
Transformers releases unless they are pinned; the 4.57 line is the compatible
runtime for this checkout.

Use `--profile quality` to force the higher-quality profile when testing a
larger GPU. Use `--num-beams` or `--attention` to benchmark a specific setting
without changing the automatic policy. Add `--manifest-output path.json` to
write a conservative `humanoid-v1` mapping beside a generated GLB. Generated
manifests include a 180-degree front orientation for the current Stable Fast
3D/TokenRig output. Override it with `--front-yaw-degrees` when a different
model source has another forward axis. Missing roles remain unmapped so the
dance client reports incomplete coverage instead of silently applying the
wrong motion.

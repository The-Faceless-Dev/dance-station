# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE} AS runtime-base

ARG LTX_REPOSITORY=https://github.com/Lightricks/LTX-2.git
ARG LTX_REF=a95ab856bf29407b6b066ede0abe1846050db56c

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/opt/ltx2/packages/ltx-core/src:/opt/ltx2/packages/ltx-pipelines/src \
    LTX_VIDEO_MODEL_ROOT=/models/ltx-2.5 \
    LTX_VIDEO_TRANSFORMER_PATH=/models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
    LTX_VIDEO_TEXT_ENCODER_PATH=/models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    LTX_VIDEO_VAE_PATH=/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    LTX_VIDEO_AUDIO_VAE_PATH=/models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    LTX_VIDEO_SPATIAL_UPSAMPLER_PATH=/models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    LTX_VIDEO_MODEL_NAME=LTX-2.5-22B-Distilled-NVFP4 \
    LTX_VIDEO_PIPELINE=distilled \
    LTX_VIDEO_QUANTIZATION=nvfp4-prequant \
    LTX_VIDEO_DEVICE=cuda \
    LTX_VIDEO_DTYPE=bfloat16 \
    LTX_VIDEO_OFFLOAD_MODE=none \
    LTX_VIDEO_GPU_REQUIRED=true \
    LTX_VIDEO_ALLOW_LOCAL_INPUTS=false \
    LTX_VIDEO_ARTIFACT_ROOT=/var/lib/autotransition/ltx-video-jobs \
    LTX_VIDEO_ENFORCE_NVFP4=true \
    LTX_VIDEO_RESERVE_VRAM_GB=1.5 \
    LTX_VIDEO_KEEP_INTERMEDIATES=false \
    LTX_VIDEO_ENFORCE_FAST_ATTENTION=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && git init /opt/ltx2 \
    && git -C /opt/ltx2 remote add origin "${LTX_REPOSITORY}" \
    && git -C /opt/ltx2 fetch --depth 1 origin "${LTX_REF}" \
    && git -C /opt/ltx2 checkout --detach FETCH_HEAD

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY containers/ltx-video-worker/entrypoint.sh /usr/local/bin/ltx-video-worker-entrypoint

# LTX 2.5 NVFP4 is a Blackwell-only production profile. The kernel extension is
# built for the RTX 5090's SM 12.0 target rather than silently selecting a slow
# fallback or shipping a binary for the wrong Blackwell generation.
RUN TORCH_CUDA_ARCH_LIST=12.0 python -m pip install --no-cache-dir --no-build-isolation -e /opt/ltx2/packages/ltx-kernels \
    && python -m pip install --no-cache-dir -e /opt/ltx2/packages/ltx-core -e /opt/ltx2/packages/ltx-pipelines \
    && python -m pip install --no-cache-dir --no-deps . \
    && python -m pip install --no-cache-dir \
       "av>=14.0" \
       "fastapi>=0.115.0" \
       "imageio-ffmpeg>=0.5.0" \
       "numpy>=1.26,<3" \
       "pillow>=10.0" \
       "scipy>=1.14" \
       "safetensors>=0.5.3" \
       "transformers>=5.8.0,<5.15" \
       "uvicorn>=0.32.0" \
    && chmod 0755 /usr/local/bin/ltx-video-worker-entrypoint \
    && python -c "import ltx_core, ltx_pipelines, torch; print('ltx packages ready torch=%s cuda=%s' % (torch.__version__, torch.version.cuda))"

RUN useradd --create-home --uid 10005 ltx \
    && mkdir -p /models/ltx-2.5 /var/lib/autotransition/ltx-video-jobs \
    && chown -R ltx:ltx /app /var/lib/autotransition /opt/ltx2

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

# The model-bearing build uses a named context so code changes do not rebuild
# the largest model layer:
# docker buildx build --build-context ltxmodels=D:/models/ltx-2.5 -f containers/ltx-video-worker/Dockerfile -t ltx-video-worker:local .
FROM runtime-base AS runtime
COPY --from=ltxmodels / /models/ltx-2.5/

RUN test -s /models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
    && test "$(stat -c '%s' /models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors)" = "18721732720" \
    && test -s /models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    && test "$(stat -c '%s' /models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors)" = "26263860594" \
    && test -s /models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    && test "$(stat -c '%s' /models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors)" = "1452269922" \
    && test -s /models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    && test "$(stat -c '%s' /models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors)" = "364866540" \
    && test -s /models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    && test "$(stat -c '%s' /models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)" = "995778752"

USER ltx
ENTRYPOINT ["/usr/local/bin/ltx-video-worker-entrypoint"]

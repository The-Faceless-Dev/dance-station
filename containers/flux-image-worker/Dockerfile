ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE} AS runtime-base

ARG FLUX2_COMMIT=50fe5162777813d869182b139e83b10743caef15
ARG SALAD_WORKER_VERSION=v0.7.0
ARG SALAD_WORKER_SHA256=074a329cf6462e77fc7b72100f59d8a690831456d9420186a834a8f30634c9e4

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/opt/flux2 \
    FLUX_IMAGE_MODEL_ROOT=/models/flux2-klein-4b \
    FLUX_IMAGE_MODEL_PATH=/models/flux2-klein-4b/flux-2-klein-4b.safetensors \
    FLUX_IMAGE_TEXT_ENCODER_PATH=/models/flux2-klein-4b/text_encoder \
    FLUX_IMAGE_VAE_PATH=/models/flux2-klein-4b/flux_vae.safetensors \
    FLUX_IMAGE_MODEL_NAME=flux.2-klein-4b \
    FLUX_IMAGE_DEVICE=cuda \
    FLUX_IMAGE_DTYPE=bfloat16 \
    FLUX_IMAGE_CPU_OFFLOAD=true \
    FLUX_IMAGE_MAX_STEPS=4 \
    FLUX_IMAGE_AVATAR_WIDTH=480 \
    FLUX_IMAGE_AVATAR_HEIGHT=832 \
    FLUX_IMAGE_AVATAR_SUBJECT_SCALE=0.70 \
    FLUX_IMAGE_ARTIFACT_ROOT=/var/lib/autotransition/flux-image-jobs \
    KLEIN_4B_MODEL_PATH=/models/flux2-klein-4b/flux-2-klein-4b.safetensors \
    FLUX2_TEXT_ENCODER_PATH=/models/flux2-klein-4b/text_encoder \
    AE_MODEL_PATH=/models/flux2-klein-4b/flux_vae.safetensors \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/black-forest-labs/flux2.git /opt/flux2 \
    && git -C /opt/flux2 checkout "${FLUX2_COMMIT}" \
    && curl -fsSL "https://github.com/SaladTechnologies/salad-cloud-job-queue-worker/releases/download/${SALAD_WORKER_VERSION}/salad-http-job-queue-worker_x86_64.tar.gz" -o /tmp/salad-worker.tar.gz \
    && echo "${SALAD_WORKER_SHA256}  /tmp/salad-worker.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/salad-worker.tar.gz -C /usr/local/bin \
    && test -x /usr/local/bin/salad-http-job-queue-worker \
    && rm -f /tmp/salad-worker.tar.gz

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY containers/flux-image-worker/entrypoint.sh /usr/local/bin/flux-image-worker-entrypoint

RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip install --no-cache-dir \
       "accelerate>=1.6.0" \
       "einops==0.8.1" \
       "fastapi>=0.115.0" \
       "huggingface-hub>=0.30.0" \
       "numpy>=1.26" \
       "pillow>=10.0" \
       "python-multipart>=0.0.20" \
       "safetensors>=0.5.3" \
       "transformers==4.56.1" \
       "typer>=0.12.0" \
       "uvicorn>=0.32.0" \
    && python -m pip install --no-cache-dir --no-deps -e /opt/flux2 \
    && chmod 0755 /usr/local/bin/flux-image-worker-entrypoint

RUN useradd --create-home --uid 10002 flux \
    && mkdir -p /models/flux2-klein-4b /var/lib/autotransition/flux-image-jobs \
    && chown -R flux:flux /app /var/lib/autotransition /opt/flux2

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

ENTRYPOINT ["/usr/local/bin/flux-image-worker-entrypoint"]

# Each model file is copied separately so the registry can cache the weights
# independently. The named build context is a local staging directory only.
FROM runtime-base AS runtime

COPY --from=fluxmodel /flux-2-klein-4b.safetensors /models/flux2-klein-4b/flux-2-klein-4b.safetensors
COPY --from=fluxmodel /text_encoder /models/flux2-klein-4b/text_encoder
COPY --from=fluxmodel /flux_vae.safetensors /models/flux2-klein-4b/flux_vae.safetensors
RUN chown -R flux:flux /models/flux2-klein-4b

USER flux

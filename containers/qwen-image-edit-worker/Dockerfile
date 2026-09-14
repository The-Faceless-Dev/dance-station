ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-base
ARG SALAD_WORKER_VERSION=v0.7.0
ARG SALAD_WORKER_SHA256=074a329cf6462e77fc7b72100f59d8a690831456d9420186a834a8f30634c9e4

LABEL org.opencontainers.image.source="https://github.com/The-Faceless-Dev/dance-station"
LABEL org.opencontainers.image.description="Qwen Image Edit 2511 CUDA Diffusers worker"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/var/lib/autotransition/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    QWEN_IMAGE_EDIT_RUNTIME=diffusers \
    QWEN_IMAGE_EDIT_MODEL_ROOT=/models/qwen-image-edit-2511 \
    QWEN_IMAGE_EDIT_MODEL_NAME=Qwen-Image-Edit-2511-Q8_0 \
    QWEN_IMAGE_EDIT_DIFFUSERS_MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
    QWEN_IMAGE_EDIT_DIFFUSERS_REVISION=main \
    QWEN_IMAGE_EDIT_TRANSFORMER=/models/qwen-image-edit-2511/qwen-image-edit-2511-Q8_0.gguf \
    QWEN_IMAGE_EDIT_DIFFUSERS_DTYPE=bfloat16 \
    QWEN_IMAGE_EDIT_DIFFUSERS_COMPUTE_DTYPE=bfloat16 \
    QWEN_IMAGE_EDIT_DIFFUSERS_CPU_OFFLOAD=true \
    QWEN_IMAGE_EDIT_DIFFUSERS_ATTENTION_BACKEND=sdpa \
    QWEN_IMAGE_EDIT_DIFFUSERS_COMPILE=false \
    QWEN_IMAGE_EDIT_GPU_REQUIRED=true \
    QWEN_IMAGE_EDIT_ARTIFACT_ROOT=/var/lib/autotransition/qwen-image-edit-jobs \
    QWEN_IMAGE_EDIT_LORA_ROOT=/var/lib/autotransition/qwen-image-edit-loras \
    QWEN_IMAGE_EDIT_STARTUP_LOG_ROOT=/var/lib/autotransition/qwen-image-edit-startup \
    QWEN_IMAGE_EDIT_STARTUP_PROBE_SECONDS=300 \
    QWEN_IMAGE_EDIT_ALLOW_LOCAL_REFERENCES=false \
    QWEN_IMAGE_EDIT_ALLOW_LOCAL_LORAS=false \
    SALAD_QUEUE_WORKER_ENABLED=true \
    WORKER_HOST=0.0.0.0 \
    WORKER_PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/SaladTechnologies/salad-cloud-job-queue-worker/releases/download/${SALAD_WORKER_VERSION}/salad-http-job-queue-worker_x86_64.tar.gz" -o /tmp/salad-worker.tar.gz \
    && echo "${SALAD_WORKER_SHA256}  /tmp/salad-worker.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/salad-worker.tar.gz -C /usr/local/bin \
    && test -x /usr/local/bin/salad-http-job-queue-worker \
    && rm -f /tmp/salad-worker.tar.gz \
    && python3 -m venv /opt/qwen-image-edit-venv \
    && /opt/qwen-image-edit-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/qwen-image-edit-venv/bin/pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cu128 \
       "torch>=2.7,<3" \
    && /opt/qwen-image-edit-venv/bin/pip install --no-cache-dir \
       "diffusers==0.40.0" \
       "transformers==5.17.0" \
       "huggingface-hub>=1.31.0,<2" \
       "accelerate>=1.1.0" \
       "bitsandbytes>=0.46.0" \
       "gguf>=0.10.0" \
       "fastapi>=0.115.0" \
       "python-multipart>=0.0.20" \
       "safetensors>=0.5.0" \
       "sentencepiece>=0.2.0" \
       "peft>=0.14.0" \
       "pillow>=10.0.0" \
       "uvicorn>=0.32.0"

COPY pyproject.toml README.md ./
COPY src ./src
COPY containers/qwen-image-edit-worker/entrypoint.sh /usr/local/bin/qwen-image-edit-worker-entrypoint

WORKDIR /app

RUN chmod 0755 /usr/local/bin/qwen-image-edit-worker-entrypoint \
    && useradd --create-home --uid 10004 qwen \
    && mkdir -p /models/qwen-image-edit-2511 /var/lib/autotransition/qwen-image-edit-jobs /var/lib/autotransition/qwen-image-edit-loras /var/lib/autotransition/qwen-image-edit-startup /var/lib/autotransition/huggingface \
    && chown -R qwen:qwen /var/lib/autotransition

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD /opt/qwen-image-edit-venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

# The local target accepts an existing transformer through a named build context.
FROM runtime-base AS runtime-local
COPY --chown=qwen:qwen --from=qweneditmodel /qwen-image-edit-2511-Q8_0.gguf /models/qwen-image-edit-2511/qwen-image-edit-2511-Q8_0.gguf
USER qwen
ENTRYPOINT ["/usr/local/bin/qwen-image-edit-worker-entrypoint"]

# The model-bearing production target includes the verified 21.8 GB Q8
# transformer. Diffusers caches the official 2511 text encoder, tokenizer,
# scheduler, and VAE under HF_HOME when the runtime first initializes.
FROM runtime-base AS runtime
ARG QWEN_IMAGE_EDIT_TRANSFORMER_URL=https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q8_0.gguf?download=true
ARG QWEN_IMAGE_EDIT_TRANSFORMER_SHA256=ab4f0622fb002fccaaa679a2ecce6fd1b3190d8ea28a5b7b2b17b8669bc24afa

RUN curl --fail --location --retry 5 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --max-time 7200 \
      --output /models/qwen-image-edit-2511/qwen-image-edit-2511-Q8_0.gguf \
      "${QWEN_IMAGE_EDIT_TRANSFORMER_URL}" \
    && echo "${QWEN_IMAGE_EDIT_TRANSFORMER_SHA256}  /models/qwen-image-edit-2511/qwen-image-edit-2511-Q8_0.gguf" | sha256sum -c -

USER qwen
ENTRYPOINT ["/usr/local/bin/qwen-image-edit-worker-entrypoint"]

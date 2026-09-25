# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE} AS runtime-base

ARG MULACOVER_REPOSITORY=https://github.com/HeartMuLa/MuLaCover.git
ARG MULACOVER_REF=315d653bc9a4383859525f7c9187ece669cf6fe8
ARG SALAD_WORKER_VERSION=v0.7.0
ARG SALAD_WORKER_SHA256=074a329cf6462e77fc7b72100f59d8a690831456d9420186a834a8f30634c9e4

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MULACOVER_MODEL_ROOT=/models/mulacover \
    MULACOVER_MODEL_NAME=MuLaCover \
    MULACOVER_DEVICE=cuda:0 \
    MULACOVER_DTYPE=bfloat16 \
    MULACOVER_GPU_REQUIRED=true \
    MULACOVER_ALLOW_LOCAL_INPUTS=false \
    MULACOVER_ARTIFACT_ROOT=/var/lib/autotransition/mulacover-jobs \
    MULACOVER_MINIMUM_FREE_VRAM_GB=20 \
    MULACOVER_RECOMMENDED_VRAM_GB=24 \
    SALAD_QUEUE_WORKER_ENABLED=true \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTORCH_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/SaladTechnologies/salad-cloud-job-queue-worker/releases/download/${SALAD_WORKER_VERSION}/salad-http-job-queue-worker_x86_64.tar.gz" -o /tmp/salad-worker.tar.gz \
    && echo "${SALAD_WORKER_SHA256}  /tmp/salad-worker.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/salad-worker.tar.gz -C /usr/local/bin \
    && test -x /usr/local/bin/salad-http-job-queue-worker \
    && rm -f /tmp/salad-worker.tar.gz \
    && git clone --depth 1 "${MULACOVER_REPOSITORY}" /opt/mulacover \
    && git -C /opt/mulacover fetch --depth 1 origin "${MULACOVER_REF}" \
    && git -C /opt/mulacover checkout --detach "${MULACOVER_REF}"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY containers/mulacover-worker/entrypoint.sh /usr/local/bin/mulacover-worker-entrypoint

RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip install --no-cache-dir \
       "accelerate==1.12.0" \
       "einops==0.8.1" \
       "fastapi>=0.115.0" \
       "faster-whisper>=1.1.1" \
       "httpx>=0.27.0" \
       "huggingface_hub[hf_xet]>=0.34.0" \
       "librosa>=0.10.2,<0.12" \
       "mido>=1.3,<2" \
       "numpy==2.0.2" \
       "safetensors>=0.4" \
       "scipy>=1.10,<2" \
       "soundfile>=0.12" \
       "tokenizers==0.22.1" \
       "torchao==0.9.0" \
       "torchtune==0.4.0" \
       "tqdm==4.67.1" \
       "transformers==4.57.6" \
       "vector-quantize-pytorch==1.27.15" \
       "uvicorn>=0.32.0" \
    && python -m pip install --no-cache-dir --no-deps /opt/mulacover \
    && python -c "import torch; import mulacover; print('MuLaCover runtime imports:', torch.__version__, mulacover.__version__)" \
    && chmod 0755 /usr/local/bin/mulacover-worker-entrypoint

RUN useradd --create-home --uid 10004 mulacover \
    && mkdir -p /models/mulacover /var/lib/autotransition/mulacover-jobs \
    && chown -R mulacover:mulacover /app /var/lib/autotransition /opt/mulacover

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

# The model-bearing build uses a named context:
# docker buildx build --build-context mulamodel=/path/to/bundle .
FROM runtime-base AS runtime
COPY --from=mulamodel / /models/mulacover/
RUN python /app/tools/mulacover/verify_bundle.py /models/mulacover
USER mulacover
ENTRYPOINT ["/usr/local/bin/mulacover-worker-entrypoint"]

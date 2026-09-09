ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE} AS runtime-base

ARG SGLANG_REPO=https://github.com/OpenMOSS/sglang.git
ARG SGLANG_REF=moss-audio
ARG SALAD_WORKER_VERSION=v0.7.0
ARG SALAD_WORKER_SHA256=074a329cf6462e77fc7b72100f59d8a690831456d9420186a834a8f30634c9e4

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MOSS_MUSIC_MODEL_ROOT=/models/moss-music-8b-instruct \
    MOSS_MUSIC_MODEL_NAME=MOSS-Music-8B-Instruct \
    MOSS_MUSIC_BACKEND=sglang \
    MOSS_MUSIC_SGLANG_URL=http://127.0.0.1:30000 \
    MOSS_MUSIC_DEVICE=cuda \
    MOSS_MUSIC_DTYPE=bfloat16 \
    MOSS_MUSIC_GPU_REQUIRED=true \
    MOSS_MUSIC_ALLOW_LOCAL_AUDIO_PATHS=false \
    MOSS_MUSIC_EVENT_RESOLUTION_MS=80 \
    MOSS_MUSIC_ARTIFACT_ROOT=/var/lib/autotransition/moss-music-jobs \
    MOSS_MUSIC_SGLANG_AUTOSTART=true \
    MOSS_MUSIC_SGLANG_PORT=30000 \
    SGLANG_ENABLE_JIT_DEEPGEMM=0 \
    CC=gcc \
    CXX=g++ \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTORCH_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && test -x "${CUDA_HOME}/bin/nvcc" \
    && "${CUDA_HOME}/bin/nvcc" --version \
    && git clone --depth 1 --branch "${SGLANG_REF}" "${SGLANG_REPO}" /opt/sglang \
    && curl -fsSL "https://github.com/SaladTechnologies/salad-cloud-job-queue-worker/releases/download/${SALAD_WORKER_VERSION}/salad-http-job-queue-worker_x86_64.tar.gz" -o /tmp/salad-worker.tar.gz \
    && echo "${SALAD_WORKER_SHA256}  /tmp/salad-worker.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/salad-worker.tar.gz -C /usr/local/bin \
    && test -x /usr/local/bin/salad-http-job-queue-worker \
    && rm -f /tmp/salad-worker.tar.gz

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY containers/moss-music-worker/entrypoint.sh /usr/local/bin/moss-music-worker-entrypoint
COPY containers/moss-music-worker/patch_sglang_blackwell.py /tmp/patch_sglang_blackwell.py

RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip install --no-cache-dir \
       "fastapi>=0.115.0" \
       "httpx>=0.27.0" \
       "imageio-ffmpeg>=0.5.0" \
       "numpy>=1.26" \
       "uvicorn>=0.32.0" \
    && python -m pip install --no-cache-dir -e "/opt/sglang/python[all]" \
    && python -m pip install --no-cache-dir "nvidia-cudnn-cu12==9.16.0.29" \
    && python /tmp/patch_sglang_blackwell.py \
    && rm -f /tmp/patch_sglang_blackwell.py \
    && chmod 0755 /usr/local/bin/moss-music-worker-entrypoint

RUN useradd --create-home --uid 10003 moss \
    && mkdir -p /models/moss-music-8b-instruct /var/lib/autotransition/moss-music-jobs \
    && chown -R moss:moss /app /var/lib/autotransition /opt/sglang

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

# The model-bearing build uses a named build context:
#   docker buildx build --build-context mossmodel=D:/models/moss-music-8b-instruct .
FROM runtime-base AS runtime
COPY --from=mossmodel / /models/moss-music-8b-instruct/
USER moss
ENTRYPOINT ["/usr/local/bin/moss-music-worker-entrypoint"]

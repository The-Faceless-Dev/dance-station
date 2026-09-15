FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS runtime-base

ARG AUDIOCPP_REPO=https://github.com/0xShug0/audio.cpp.git
ARG AUDIOCPP_REF=6fbbee4efd1c2251d3774edd89a002a0039da5f7
ARG AUDIOCPP_CUDA_ARCH=86;89;90
ARG SALAD_WORKER_VERSION=v0.7.0
ARG SALAD_WORKER_SHA256=074a329cf6462e77fc7b72100f59d8a690831456d9420186a834a8f30634c9e4

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    YUE2_MODEL_ROOT=/models/yue2-3b-gguf \
    YUE2_MODEL_FILE=yue2-3b-q8_0.gguf \
    YUE2_VAE_FILE=yue2-vae-f16.gguf \
    YUE2_AUDIOCPP_CLI=/opt/audio.cpp/bin/audiocpp_cli \
    YUE2_AUDIOCPP_REVISION=${AUDIOCPP_REF} \
    YUE2_BACKEND=cuda \
    YUE2_DEVICE=0 \
    YUE2_THREADS=8 \
    YUE2_GPU_REQUIRED=true \
    YUE2_ARTIFACT_ROOT=/var/lib/autotransition/yue2-jobs \
    WORKER_HOST=0.0.0.0 \
    WORKER_PORT=8080 \
    YUE2_WAIT_FOR_READY=true \
    YUE2_STARTUP_TIMEOUT_SECONDS=1800 \
    SALAD_QUEUE_WORKER_ENABLED=true \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git build-essential cmake ninja-build python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch dev "${AUDIOCPP_REPO}" /opt/audio.cpp \
    && git -C /opt/audio.cpp fetch --depth 1 origin "${AUDIOCPP_REF}" \
    && git -C /opt/audio.cpp checkout --detach "${AUDIOCPP_REF}"

# The project build helper is a POSIX shell script. Keep the CUDA/model-family
# selection explicit so a future image can change the architecture without
# changing the worker code.
RUN cd /opt/audio.cpp \
    && ./scripts/build_linux.sh --backend cuda --cuda-arch "${AUDIOCPP_CUDA_ARCH}" --model-set custom --models yue2 --build-type Release --no-deployment-build --target audiocpp_cli \
    && test -x /opt/audio.cpp/build/linux-cuda-release/bin/audiocpp_cli \
    && mkdir -p /opt/audio.cpp/bin \
    && cp /opt/audio.cpp/build/linux-cuda-release/bin/audiocpp_cli /opt/audio.cpp/bin/audiocpp_cli

RUN curl -fsSL "https://github.com/SaladTechnologies/salad-cloud-job-queue-worker/releases/download/${SALAD_WORKER_VERSION}/salad-http-job-queue-worker_x86_64.tar.gz" -o /tmp/salad-worker.tar.gz \
    && echo "${SALAD_WORKER_SHA256}  /tmp/salad-worker.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/salad-worker.tar.gz -C /usr/local/bin \
    && test -x /usr/local/bin/salad-http-job-queue-worker \
    && rm -f /tmp/salad-worker.tar.gz

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY containers/yue2-worker/entrypoint.sh /usr/local/bin/yue2-worker-entrypoint

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir --no-deps . \
    && python3 -m pip install --no-cache-dir "fastapi>=0.115.0" "httpx>=0.27.0" "numpy>=1.26" "uvicorn>=0.32.0" \
    && ln -sf /opt/audio.cpp/bin/audiocpp_cli /usr/local/bin/audiocpp_cli \
    && chmod 0755 /usr/local/bin/yue2-worker-entrypoint

RUN useradd --create-home --uid 10004 yue2 \
    && mkdir -p /models/yue2-3b-gguf /var/lib/autotransition/yue2-jobs /home/yue2/.cache \
    && chown -R yue2:yue2 /app /opt/audio.cpp /var/lib/autotransition /home/yue2

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

# Model-bearing build context example:
# docker build --build-context yue2model=D:/models/yue2-3b-gguf -f Dockerfile .
FROM runtime-base AS runtime
COPY --from=yue2model / /models/yue2-3b-gguf/
USER yue2
ENTRYPOINT ["/usr/local/bin/yue2-worker-entrypoint"]

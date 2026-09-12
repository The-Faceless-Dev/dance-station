ARG CUDA_VERSION=12.8.1
ARG SDCPP_COMMIT=7f410a3793c5bba8eb198e962ce7a3d6095f9d89

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS sd-builder
ARG SDCPP_COMMIT

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/leejet/stable-diffusion.cpp.git /opt/stable-diffusion.cpp \
    && git -C /opt/stable-diffusion.cpp fetch --depth 1 origin "${SDCPP_COMMIT}" \
    && git -C /opt/stable-diffusion.cpp checkout --detach "${SDCPP_COMMIT}" \
    && git -C /opt/stable-diffusion.cpp submodule update --init --depth 1 \
    && cmake -S /opt/stable-diffusion.cpp -B /opt/stable-diffusion.cpp/build \
       -DGGML_CUDA=ON \
       -DSD_SERVER_BUILD_FRONTEND=OFF \
       -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/stable-diffusion.cpp/build --config Release --target sd-server -j"$(nproc)" \
    && test -x /opt/stable-diffusion.cpp/build/bin/sd-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-base
ARG SALAD_WORKER_VERSION=v0.7.0
ARG SALAD_WORKER_SHA256=074a329cf6462e77fc7b72100f59d8a690831456d9420186a834a8f30634c9e4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    QWEN_IMAGE_MODEL_ROOT=/models/qwen-image-2512 \
    QWEN_IMAGE_DIFFUSION_MODEL=/models/qwen-image-2512/qwen-image-2512-Q8_0.gguf \
    QWEN_IMAGE_TEXT_ENCODER=/models/qwen-image-2512/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf \
    QWEN_IMAGE_VAE=/models/qwen-image-2512/qwen_image_vae.safetensors \
    QWEN_IMAGE_RUNTIME_BINARY=/usr/local/bin/sd-server \
    QWEN_IMAGE_BACKEND=cuda \
    QWEN_IMAGE_MODEL_NAME=Qwen-Image-2512-Q8_0 \
    QWEN_IMAGE_CPU_OFFLOAD=true \
    QWEN_IMAGE_FLASH_ATTENTION=true \
    QWEN_IMAGE_MMAP=true \
    QWEN_IMAGE_LORA_APPLY_MODE=at_runtime \
    QWEN_IMAGE_ARTIFACT_ROOT=/var/lib/autotransition/qwen-image-jobs \
    QWEN_IMAGE_LORA_ROOT=/var/lib/autotransition/qwen-image-loras \
    QWEN_IMAGE_ALLOW_LOCAL_LORAS=false \
    SALAD_QUEUE_WORKER_ENABLED=true \
    WORKER_HOST=0.0.0.0 \
    WORKER_PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/SaladTechnologies/salad-cloud-job-queue-worker/releases/download/${SALAD_WORKER_VERSION}/salad-http-job-queue-worker_x86_64.tar.gz" -o /tmp/salad-worker.tar.gz \
    && echo "${SALAD_WORKER_SHA256}  /tmp/salad-worker.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/salad-worker.tar.gz -C /usr/local/bin \
    && test -x /usr/local/bin/salad-http-job-queue-worker \
    && rm -f /tmp/salad-worker.tar.gz \
    && python3 -m venv /opt/qwen-venv \
    && /opt/qwen-venv/bin/pip install --no-cache-dir \
       "fastapi>=0.115.0" "python-multipart>=0.0.20" "uvicorn>=0.32.0"

COPY --from=sd-builder /opt/stable-diffusion.cpp/build/bin/sd-server /usr/local/bin/sd-server

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY containers/qwen-image-worker/entrypoint.sh /usr/local/bin/qwen-image-worker-entrypoint

RUN chmod 0755 /usr/local/bin/sd-server /usr/local/bin/qwen-image-worker-entrypoint \
    && useradd --create-home --uid 10004 qwen \
    && mkdir -p /models/qwen-image-2512 /var/lib/autotransition/qwen-image-jobs /var/lib/autotransition/qwen-image-loras \
    && chown -R qwen:qwen /app /var/lib/autotransition /opt/qwen-venv

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD /opt/qwen-venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"

# Build with a named model context so every model component gets an independent
# registry layer and no paid provider download is needed at job time.
FROM runtime-base AS runtime-local
COPY --chown=qwen:qwen --from=qwenmodel /qwen-image-2512-Q8_0.gguf /models/qwen-image-2512/qwen-image-2512-Q8_0.gguf
COPY --chown=qwen:qwen --from=qwenmodel /Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf /models/qwen-image-2512/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf
COPY --chown=qwen:qwen --from=qwenmodel /split_files/vae/qwen_image_vae.safetensors /models/qwen-image-2512/qwen_image_vae.safetensors

USER qwen
ENTRYPOINT ["/usr/local/bin/qwen-image-worker-entrypoint"]

# The published target downloads and verifies each model in its own layer. This
# avoids retaining a second full model context on CI runners while preserving
# independent registry layers and build-cache reuse.
FROM runtime-base AS runtime
ARG QWEN_IMAGE_DIFFUSION_URL=https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf?download=true
ARG QWEN_IMAGE_TEXT_ENCODER_URL=https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf?download=true
ARG QWEN_IMAGE_VAE_URL=https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors?download=true
ARG QWEN_IMAGE_DIFFUSION_SHA256=E285A0692582ACF09BB4086D9B120EB0E357C4386565D169A033CB968C6FA9A5
ARG QWEN_IMAGE_TEXT_ENCODER_SHA256=669A5604C47C90C20110C6DB5FE10AD7E8EC99B553A785D7D20492F7D5B3E7D0
ARG QWEN_IMAGE_VAE_SHA256=A70580F0213E67967EE9C95F05BB400E8FB08307E017A924BF3441223E023D1F

RUN curl --fail --location --retry 5 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --max-time 7200 \
      --output /models/qwen-image-2512/qwen-image-2512-Q8_0.gguf \
      "${QWEN_IMAGE_DIFFUSION_URL}" \
    && echo "${QWEN_IMAGE_DIFFUSION_SHA256}  /models/qwen-image-2512/qwen-image-2512-Q8_0.gguf" | sha256sum -c - \
    && chown qwen:qwen /models/qwen-image-2512/qwen-image-2512-Q8_0.gguf

RUN curl --fail --location --retry 5 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --max-time 7200 \
      --output /models/qwen-image-2512/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf \
      "${QWEN_IMAGE_TEXT_ENCODER_URL}" \
    && echo "${QWEN_IMAGE_TEXT_ENCODER_SHA256}  /models/qwen-image-2512/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf" | sha256sum -c - \
    && chown qwen:qwen /models/qwen-image-2512/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf

RUN curl --fail --location --retry 5 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --max-time 7200 \
      --output /models/qwen-image-2512/qwen_image_vae.safetensors \
      "${QWEN_IMAGE_VAE_URL}" \
    && echo "${QWEN_IMAGE_VAE_SHA256}  /models/qwen-image-2512/qwen_image_vae.safetensors" | sha256sum -c - \
    && chown qwen:qwen /models/qwen-image-2512/qwen_image_vae.safetensors

USER qwen
ENTRYPOINT ["/usr/local/bin/qwen-image-worker-entrypoint"]

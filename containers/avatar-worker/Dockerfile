# Heavy model repositories and checkpoints are mounted at runtime. Keeping
# them out of this image avoids baking machine-specific weights into the API.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    AVATAR_ARTIFACT_ROOT=/var/lib/faceless/avatar-jobs \
    AVATAR_MAX_ATTEMPTS=3 \
    AVATAR_REQUIRE_DEFORMATION_VALIDATOR=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools

RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "python-multipart>=0.0.20" \
        "typer>=0.12.0" \
        "uvicorn>=0.32.0" \
        "einops==0.8.1" \
        "transformers==4.56.1" \
        "safetensors==0.4.5" \
        "accelerate==1.12.0" \
        "fire==0.7.1"

RUN useradd --create-home --uid 10001 avatar \
    && mkdir -p /var/lib/faceless/avatar-jobs /models /runtimes \
    && chown -R avatar:avatar /app /var/lib/faceless/avatar-jobs
USER avatar

EXPOSE 8090
CMD ["autotransition", "avatar-worker", "--host", "0.0.0.0", "--port", "8090"]

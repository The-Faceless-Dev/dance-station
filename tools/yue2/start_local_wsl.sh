#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${AUTOTRANSITION_ROOT:-/mnt/d/autotransition}"
MODEL_ROOT="${YUE2_MODEL_ROOT:-/mnt/d/models/yue2-3b-gguf}"
CLI="${YUE2_AUDIOCPP_CLI:-/home/jarion/audio.cpp-yue2/build/linux-cuda-release/bin/audiocpp_cli}"
PYTHON="${YUE2_PYTHON:-/home/jarion/yue2-venv/bin/python}"
PORT="${WORKER_PORT:-8092}"

[[ -x "$PYTHON" ]] || { echo "YuE2 Python environment not found: $PYTHON" >&2; exit 1; }
[[ -x "$CLI" ]] || { echo "audio.cpp CLI not found: $CLI" >&2; exit 1; }
[[ -d "$MODEL_ROOT" ]] || { echo "YuE2 model root not found: $MODEL_ROOT" >&2; exit 1; }

export PYTHONPATH="$ROOT/src"
export YUE2_MODEL_ROOT="$MODEL_ROOT"
export YUE2_AUDIOCPP_CLI="$CLI"
export YUE2_ARTIFACT_ROOT="${YUE2_ARTIFACT_ROOT:-$ROOT/tmp/yue2-jobs}"
export WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
export WORKER_PORT="$PORT"
export SALAD_QUEUE_WORKER_ENABLED="${SALAD_QUEUE_WORKER_ENABLED:-false}"

exec "$PYTHON" -m autotransition.yue2.server

#!/usr/bin/env bash
set -Eeuo pipefail

SALAD_PID=""
WORKER_PID=""
HEARTBEAT_PID=""

if [[ -z "${HOME:-}" || ! -d "${HOME}" || ! -w "${HOME}" ]]; then
  USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6 || true)"
  if [[ -n "${USER_HOME}" && -d "${USER_HOME}" ]]; then
    export HOME="${USER_HOME}"
  else
    export HOME="/tmp/qwen-home"
    mkdir -p "${HOME}"
  fi
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "${XDG_CACHE_HOME}" "${QWEN_IMAGE_LORA_ROOT}" "${QWEN_IMAGE_ARTIFACT_ROOT}"
echo "QWEN startup user=$(id -un) uid=$(id -u) home=${HOME} cache=${XDG_CACHE_HOME} backend=${QWEN_IMAGE_BACKEND:-cuda}"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$WORKER_PID" "$SALAD_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$WORKER_PID" "$SALAD_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "Starting Qwen-Image-2512 worker on ${WORKER_HOST}:${WORKER_PORT}"
/opt/qwen-venv/bin/python -m autotransition.qwen_image.server &
WORKER_PID=$!

if [[ "${SALAD_QUEUE_WORKER_ENABLED:-true}" == "true" ]]; then
  echo "Starting Salad queue worker"
  /usr/local/bin/salad-http-job-queue-worker &
  SALAD_PID=$!
else
  echo "Salad queue worker disabled; direct HTTP mode is enabled"
fi

if [[ -n "${LAUNCH_SERVER_HEARTBEAT_URL:-}" && -n "${WORKER_HEARTBEAT_TOKEN:-}" ]]; then
  echo "Starting launch-server heartbeat agent"
  /opt/qwen-venv/bin/python -m autotransition.qwen_image.heartbeat &
  HEARTBEAT_PID=$!
fi

PIDS=("$WORKER_PID")
[[ -z "$SALAD_PID" ]] || PIDS+=("$SALAD_PID")
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed Qwen worker process exited with status ${status}"
exit "$status"

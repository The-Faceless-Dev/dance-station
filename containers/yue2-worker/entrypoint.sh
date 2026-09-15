#!/usr/bin/env bash
set -Eeuo pipefail

WORKER_PID=""
SALAD_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$SALAD_PID" "$WORKER_PID"; do
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$SALAD_PID" "$WORKER_PID"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ -z "${HOME:-}" || ! -d "${HOME}" || ! -w "${HOME}" ]]; then
  export HOME="/tmp/yue2-home"
  mkdir -p "$HOME"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$XDG_CACHE_HOME" "${YUE2_ARTIFACT_ROOT:-/var/lib/autotransition/yue2-jobs}"

echo "YuE2 startup user=$(id -un) uid=$(id -u) home=${HOME} model=${YUE2_MODEL_ROOT:-unset} cli=${YUE2_AUDIOCPP_CLI:-unset} backend=${YUE2_BACKEND:-unset}"
echo "YuE2 profile main=${YUE2_MODEL_FILE:-unset} vae=${YUE2_VAE_FILE:-unset} audio_cpp=${YUE2_AUDIOCPP_REVISION:-unset}"

echo "Starting YuE2 worker on ${WORKER_HOST:-0.0.0.0}:${WORKER_PORT:-8080}"
python3 -m autotransition.yue2.server &
WORKER_PID=$!

if [[ "${YUE2_WAIT_FOR_READY:-true}" == "true" ]]; then
  deadline=$((SECONDS + ${YUE2_STARTUP_TIMEOUT_SECONDS:-1800}))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${WORKER_PORT:-8080}/ready" >/tmp/yue2-ready.json 2>/tmp/yue2-ready.error; then
      echo "YuE2 worker is ready: $(cat /tmp/yue2-ready.json)"
      break
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
      echo "YuE2 HTTP worker exited during startup" >&2
      cat /tmp/yue2-ready.error >&2 || true
      exit 1
    fi
    sleep 5
  done
  if (( SECONDS >= deadline )); then
    echo "YuE2 readiness timeout after ${YUE2_STARTUP_TIMEOUT_SECONDS:-1800}s" >&2
    cat /tmp/yue2-ready.error >&2 || true
    exit 1
  fi
fi

if [[ "${SALAD_QUEUE_WORKER_ENABLED:-true}" == "true" ]]; then
  echo "Starting Salad queue worker"
  /usr/local/bin/salad-http-job-queue-worker &
  SALAD_PID=$!
else
  echo "Salad queue worker disabled; direct HTTP mode is enabled"
fi

PIDS=("$WORKER_PID")
[[ -z "$SALAD_PID" ]] || PIDS+=("$SALAD_PID")
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed YuE2 process exited with status ${status}"
exit "$status"

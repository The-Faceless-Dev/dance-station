#!/usr/bin/env bash
set -Eeuo pipefail

SGLANG_PID=""
SALAD_PID=""
WORKER_PID=""
HEARTBEAT_PID=""

if [[ -z "${HOME:-}" || ! -d "${HOME}" || ! -w "${HOME}" ]]; then
  USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6 || true)"
  if [[ -n "${USER_HOME}" && -d "${USER_HOME}" ]]; then
    export HOME="${USER_HOME}"
  else
    export HOME="/tmp/moss-home"
    mkdir -p "${HOME}"
  fi
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${XDG_CACHE_HOME}/flashinfer}"
mkdir -p "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_DIR}"
echo "MOSS startup user=$(id -un) uid=$(id -u) home=${HOME} cache=${XDG_CACHE_HOME} flashinfer=${FLASHINFER_WORKSPACE_DIR}"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$WORKER_PID" "$SALAD_PID" "$SGLANG_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$WORKER_PID" "$SALAD_PID" "$SGLANG_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}

trap cleanup EXIT INT TERM

if [[ "${MOSS_MUSIC_SGLANG_AUTOSTART:-true}" == "true" ]]; then
  echo "Starting MOSS-Music SGLang backend on port ${MOSS_MUSIC_SGLANG_PORT:-30000}"
  read -r -a SGLANG_EXTRA_ARGS <<< "${MOSS_MUSIC_SGLANG_EXTRA_ARGS:-}"
  sglang serve \
    --model-path "${MOSS_MUSIC_MODEL_ROOT}" \
    --host "${MOSS_MUSIC_SGLANG_HOST:-127.0.0.1}" \
    --port "${MOSS_MUSIC_SGLANG_PORT:-30000}" \
    --trust-remote-code \
    "${SGLANG_EXTRA_ARGS[@]}" &
  SGLANG_PID=$!
else
  echo "MOSS-Music SGLang autostart disabled; using ${MOSS_MUSIC_SGLANG_URL:-external backend}"
fi

echo "Starting MOSS-Music worker on ${WORKER_HOST:-0.0.0.0}:${WORKER_PORT:-8080}"
python -m autotransition.moss_music.server &
WORKER_PID=$!

if [[ "${MOSS_MUSIC_WAIT_FOR_READY:-true}" == "true" ]]; then
  echo "Waiting for MOSS-Music readiness before accepting queue jobs"
  ready_deadline=$((SECONDS + ${MOSS_MUSIC_STARTUP_TIMEOUT_SECONDS:-1800}))
  while (( SECONDS < ready_deadline )); do
    if curl -fsS "http://127.0.0.1:${WORKER_PORT:-8080}/ready" >/tmp/moss-ready.json 2>/tmp/moss-ready.error; then
      echo "MOSS-Music worker is ready: $(cat /tmp/moss-ready.json)"
      break
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
      echo "MOSS-Music HTTP worker exited during startup"
      cat /tmp/moss-ready.error >&2 || true
      exit 1
    fi
    sleep 5
  done
  if (( SECONDS >= ready_deadline )); then
    echo "MOSS-Music readiness timeout after ${MOSS_MUSIC_STARTUP_TIMEOUT_SECONDS:-1800}s" >&2
    cat /tmp/moss-ready.error >&2 || true
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

if [[ -n "${LAUNCH_SERVER_HEARTBEAT_URL:-}" && -n "${WORKER_HEARTBEAT_TOKEN:-}" ]]; then
  echo "Starting launch-server heartbeat agent"
  python -m autotransition.moss_music.heartbeat &
  HEARTBEAT_PID=$!
fi

PIDS=("$WORKER_PID")
[[ -z "$SALAD_PID" ]] || PIDS+=("$SALAD_PID")
[[ -z "$SGLANG_PID" ]] || PIDS+=("$SGLANG_PID")
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed MOSS-Music process exited with status ${status}"
exit "$status"

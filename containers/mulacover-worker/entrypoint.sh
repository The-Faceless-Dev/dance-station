#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_PID=""
SALAD_PID=""
HEARTBEAT_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$SERVER_PID" "$SALAD_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$SERVER_PID" "$SALAD_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}

trap cleanup EXIT INT TERM

if [[ -z "${HOME:-}" || ! -d "${HOME}" || ! -w "${HOME}" ]]; then
  USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6 || true)"
  export HOME="${USER_HOME:-/tmp/mulacover-home}"
  mkdir -p "$HOME"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$XDG_CACHE_HOME"

echo "Starting MuLaCover worker on ${WORKER_HOST:-0.0.0.0}:${WORKER_PORT:-8080}"
python -m autotransition.mulacover.server &
SERVER_PID=$!

if [[ "${MULACOVER_WAIT_FOR_READY:-true}" == "true" ]]; then
  deadline=$((SECONDS + ${MULACOVER_STARTUP_TIMEOUT_SECONDS:-1800}))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${WORKER_PORT:-8080}/ready" >/tmp/mulacover-ready.json 2>/tmp/mulacover-ready.error; then
      echo "MuLaCover worker ready: $(cat /tmp/mulacover-ready.json)"
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "MuLaCover HTTP worker exited during startup" >&2
      cat /tmp/mulacover-ready.error >&2 || true
      exit 1
    fi
    sleep 5
  done
  if (( SECONDS >= deadline )); then
    echo "MuLaCover readiness timeout after ${MULACOVER_STARTUP_TIMEOUT_SECONDS:-1800}s" >&2
    cat /tmp/mulacover-ready.error >&2 || true
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
  python -m autotransition.mulacover.heartbeat &
  HEARTBEAT_PID=$!
fi

PIDS=("$SERVER_PID")
[[ -z "$SALAD_PID" ]] || PIDS+=("$SALAD_PID")
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed MuLaCover process exited with status ${status}"
exit "$status"

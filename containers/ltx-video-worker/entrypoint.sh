#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_PID=""
HEARTBEAT_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ -z "$SERVER_PID" ]] || kill -TERM "$SERVER_PID" 2>/dev/null || true
  [[ -z "$HEARTBEAT_PID" ]] || kill -TERM "$HEARTBEAT_PID" 2>/dev/null || true
  [[ -z "$SERVER_PID" ]] || wait "$SERVER_PID" 2>/dev/null || true
  [[ -z "$HEARTBEAT_PID" ]] || wait "$HEARTBEAT_PID" 2>/dev/null || true
  exit "$status"
}

trap cleanup EXIT INT TERM

echo "Starting LTX 2.5 video worker on ${WORKER_HOST:-0.0.0.0}:${WORKER_PORT:-8080}"
python -m autotransition.ltx_video.server &
SERVER_PID=$!

if [[ -n "${LAUNCH_SERVER_HEARTBEAT_URL:-}" && -n "${WORKER_HEARTBEAT_TOKEN:-}" ]]; then
  echo "Starting launch-server heartbeat agent"
  python -m autotransition.ltx_video.heartbeat &
  HEARTBEAT_PID=$!
fi

PIDS=("$SERVER_PID")
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed LTX process exited with status ${status}"
exit "$status"

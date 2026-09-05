#!/usr/bin/env bash
set -Eeuo pipefail

SALAD_PID=""
WORKER_PID=""
HEARTBEAT_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ -z "$SALAD_PID" ]] || kill -TERM "$SALAD_PID" 2>/dev/null || true
  [[ -z "$WORKER_PID" ]] || kill -TERM "$WORKER_PID" 2>/dev/null || true
  [[ -z "$HEARTBEAT_PID" ]] || kill -TERM "$HEARTBEAT_PID" 2>/dev/null || true
  [[ -z "$SALAD_PID" ]] || wait "$SALAD_PID" 2>/dev/null || true
  [[ -z "$WORKER_PID" ]] || wait "$WORKER_PID" 2>/dev/null || true
  [[ -z "$HEARTBEAT_PID" ]] || wait "$HEARTBEAT_PID" 2>/dev/null || true
  exit "$status"
}

trap cleanup EXIT INT TERM

echo "Starting Salad queue worker"
if [[ "${SALAD_QUEUE_WORKER_ENABLED:-true}" == "true" ]]; then
  /usr/local/bin/salad-http-job-queue-worker &
  SALAD_PID=$!
else
  echo "Salad queue worker disabled for local HTTP dispatch"
fi

echo "Starting Wan2.1 VACE stitch worker on ${WORKER_HOST:-0.0.0.0}:${WORKER_PORT:-8080}"
python -m autotransition.vace_stitch.salad_server &
WORKER_PID=$!

if [[ -n "${LAUNCH_SERVER_HEARTBEAT_URL:-}" && -n "${WORKER_HEARTBEAT_TOKEN:-}" ]]; then
  echo "Starting launch-server heartbeat agent"
  python -m autotransition.vace_stitch.heartbeat &
  HEARTBEAT_PID=$!
fi

PIDS=("$WORKER_PID")
if [[ -n "$SALAD_PID" ]]; then PIDS+=("$SALAD_PID"); fi
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed VACE process exited with status ${status}"
exit "$status"

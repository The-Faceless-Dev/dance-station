#!/usr/bin/env bash
set -Eeuo pipefail

SGLANG_PID=""
SALAD_PID=""
WORKER_PID=""
HEARTBEAT_PID=""

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

if [[ "${SALAD_QUEUE_WORKER_ENABLED:-true}" == "true" ]]; then
  echo "Starting Salad queue worker"
  /usr/local/bin/salad-http-job-queue-worker &
  SALAD_PID=$!
else
  echo "Salad queue worker disabled; direct HTTP mode is enabled"
fi

echo "Starting MOSS-Music worker on ${WORKER_HOST:-0.0.0.0}:${WORKER_PORT:-8080}"
python -m autotransition.moss_music.server &
WORKER_PID=$!

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

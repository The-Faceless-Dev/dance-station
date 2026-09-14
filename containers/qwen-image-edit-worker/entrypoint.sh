#!/usr/bin/env bash
set -Eeuo pipefail

SALAD_PID=""
WORKER_PID=""
HEARTBEAT_PID=""
STARTUP_FAILURE_REPORTED=0
STARTUP_READY=0

QWEN_IMAGE_EDIT_ARTIFACT_ROOT="${QWEN_IMAGE_EDIT_ARTIFACT_ROOT:-/tmp/qwen-image-edit-jobs}"
QWEN_IMAGE_EDIT_LORA_ROOT="${QWEN_IMAGE_EDIT_LORA_ROOT:-/tmp/qwen-image-edit-loras}"
HF_HOME="${HF_HOME:-/tmp/qwen-image-edit-huggingface}"
WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
WORKER_PORT="${WORKER_PORT:-8080}"
export QWEN_IMAGE_EDIT_ARTIFACT_ROOT QWEN_IMAGE_EDIT_LORA_ROOT HF_HOME WORKER_HOST WORKER_PORT

STARTUP_LOG_ROOT="${QWEN_IMAGE_EDIT_STARTUP_LOG_ROOT:-${QWEN_IMAGE_EDIT_ARTIFACT_ROOT:-/tmp}/startup}"
if ! mkdir -p "${STARTUP_LOG_ROOT}" 2>/dev/null || [[ ! -w "${STARTUP_LOG_ROOT}" ]]; then
  STARTUP_LOG_ROOT="/tmp/qwen-image-edit-startup"
  mkdir -p "${STARTUP_LOG_ROOT}"
fi
STARTUP_LOG="${STARTUP_LOG_ROOT}/startup.log"
STARTUP_REPORT="${STARTUP_LOG_ROOT}/startup-report.txt"
WORKER_LOG="${STARTUP_LOG_ROOT}/worker.log"
PROBE_BODY="${STARTUP_LOG_ROOT}/health-response.txt"
PROBE_ERROR="${STARTUP_LOG_ROOT}/health-error.txt"
touch "${STARTUP_LOG}" "${STARTUP_REPORT}" "${WORKER_LOG}" "${PROBE_BODY}" "${PROBE_ERROR}"
exec > >(tee -a "${STARTUP_LOG}") 2>&1

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
event() { log "STARTUP_EVENT phase=$1 ${*:2}"; }

safe_environment() {
  local name
  for name in \
    HOME XDG_CACHE_HOME HF_HOME PYTHONPATH HOSTNAME USER WORKER_PROVIDER WORKER_HOST WORKER_PORT \
    SALAD_QUEUE_WORKER_ENABLED QWEN_IMAGE_EDIT_RUNTIME QWEN_IMAGE_EDIT_MODEL_ROOT \
    QWEN_IMAGE_EDIT_MODEL_NAME QWEN_IMAGE_EDIT_DIFFUSERS_MODEL_ID QWEN_IMAGE_EDIT_DIFFUSERS_REVISION \
    QWEN_IMAGE_EDIT_TRANSFORMER QWEN_IMAGE_EDIT_DIFFUSERS_DTYPE QWEN_IMAGE_EDIT_DIFFUSERS_COMPUTE_DTYPE \
    QWEN_IMAGE_EDIT_DIFFUSERS_CPU_OFFLOAD QWEN_IMAGE_EDIT_DIFFUSERS_ATTENTION_BACKEND \
    QWEN_IMAGE_EDIT_DIFFUSERS_COMPILE QWEN_IMAGE_EDIT_GPU_REQUIRED QWEN_IMAGE_EDIT_ARTIFACT_ROOT \
    QWEN_IMAGE_EDIT_LORA_ROOT QWEN_IMAGE_EDIT_STARTUP_LOG_ROOT QWEN_IMAGE_EDIT_STARTUP_PROBE_SECONDS; do
    if [[ -v "${name}" ]]; then
      printf '%s=%s\n' "${name}" "${!name}"
    else
      printf '%s=<unset>\n' "${name}"
    fi
  done
}

collect_diagnostics() {
  local reason="${1:-unknown}" status="${2:-unknown}" line="${3:-unknown}" command="${4:-unknown}"
  {
    printf '\n=== diagnostic captured=%s reason=%s status=%s ===\n' "$(timestamp)" "${reason}" "${status}"
    printf 'source_line=%s\nlast_command=%s\n' "${line}" "${command}"
    printf '\n--- identity ---\n'
    id || true
    umask || true
    hostname || true
    uname -a || true
    cat /etc/os-release 2>/dev/null || true
    printf '\n--- process ---\n'
    ps -ef 2>/dev/null || ps aux 2>/dev/null || true
    printf '\n--- cgroup ---\n'
    cat /proc/1/cgroup 2>/dev/null || true
    printf '\n--- memory and disk ---\n'
    free -h 2>/dev/null || true
    df -h 2>/dev/null || true
    df -ih 2>/dev/null || true
    printf '\n--- safe environment ---\n'
    safe_environment
    printf '\n--- required commands ---\n'
    command -v bash || true
    command -v curl || true
    command -v nvidia-smi || true
    printf '\n--- paths ---\n'
    for path in \
      "${HOME:-<unset>}" "${XDG_CACHE_HOME:-<unset>}" "${HF_HOME:-<unset>}" \
      "${QWEN_IMAGE_EDIT_MODEL_ROOT:-<unset>}" "${QWEN_IMAGE_EDIT_TRANSFORMER:-<unset>}" \
      "${QWEN_IMAGE_EDIT_ARTIFACT_ROOT:-<unset>}" "${QWEN_IMAGE_EDIT_LORA_ROOT:-<unset>}"; do
      printf 'PATH=%s\n' "${path}"
      [[ "${path}" == '<unset>' ]] || ls -ld "${path}" 2>&1 || true
    done
    printf '\n--- NVIDIA ---\n'
    nvidia-smi -L 2>&1 || true
    nvidia-smi 2>&1 || true
    cat /proc/driver/nvidia/version 2>/dev/null || true
    printf '\n--- Python ---\n'
    if [[ -x "${PYTHON_BIN:-}" ]]; then
      "${PYTHON_BIN}" --version 2>&1 || true
      "${PYTHON_BIN}" - <<'PY' 2>&1 || true
import importlib.metadata
import os
import sys

print("python_executable=" + sys.executable)
print("python_version=" + sys.version.replace("\n", " "))
for name in ("torch", "diffusers", "transformers", "accelerate", "bitsandbytes", "gguf", "fastapi", "uvicorn", "PIL"):
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        print(f"import_ok={name} version={version}")
    except Exception as exc:
        print(f"import_failed={name} type={type(exc).__name__} error={exc}")
try:
    import torch
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"torch_cuda_device_{index}={torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch_cuda_probe_failed type={type(exc).__name__} error={exc}")
PY
    else
      printf 'python_executable_missing=%s\n' "${PYTHON_BIN:-<unset>}"
    fi
    printf '\n--- child logs ---\n'
    tail -n 200 "${WORKER_LOG}" 2>/dev/null || true
    printf '\n--- health probe ---\n'
    tail -n 100 "${PROBE_BODY}" 2>/dev/null || true
    tail -n 100 "${PROBE_ERROR}" 2>/dev/null || true
  } >>"${STARTUP_REPORT}" 2>&1 || true
  log "STARTUP_DIAGNOSTICS reason=${reason} status=${status} report=${STARTUP_REPORT} log=${STARTUP_LOG} workerLog=${WORKER_LOG}"
}

on_error() {
  local status=$? line="${1:-unknown}" command="${BASH_COMMAND:-unknown}"
  if [[ "${STARTUP_FAILURE_REPORTED}" -eq 0 ]]; then
    STARTUP_FAILURE_REPORTED=1
    event "shell_error" "status=${status} line=${line} command=$(printf '%q' "${command}")"
    collect_diagnostics "shell_error" "${status}" "${line}" "${command}"
  fi
  return "${status}"
}
trap 'on_error ${LINENO}' ERR

if [[ -z "${HOME:-}" || ! -d "${HOME}" || ! -w "${HOME}" ]]; then
  USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6 || true)"
  if [[ -n "${USER_HOME}" && -d "${USER_HOME}" ]]; then
    export HOME="${USER_HOME}"
  else
    export HOME="/tmp/qwen-image-edit-home"
    mkdir -p "${HOME}"
  fi
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "${XDG_CACHE_HOME}" "${QWEN_IMAGE_EDIT_LORA_ROOT}" "${QWEN_IMAGE_EDIT_ARTIFACT_ROOT}" "${HF_HOME}"
PYTHON_BIN="/opt/qwen-image-edit-venv/bin/python"
export PYTHON_BIN
event "entrypoint_invoked" "pid=$$ uid=$(id -u) user=$(id -un) hostname=${HOSTNAME:-unknown}"
log "QWEN EDIT startup user=$(id -un) uid=$(id -u) home=${HOME} cache=${XDG_CACHE_HOME} backend=${QWEN_IMAGE_EDIT_RUNTIME:-diffusers}"
log "Startup diagnostics root=${STARTUP_LOG_ROOT} report=${STARTUP_REPORT}"
safe_environment
collect_diagnostics "entrypoint_invoked" "0" "0" "startup"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  event "python_missing" "path=${PYTHON_BIN}"
  collect_diagnostics "python_missing" "127" "0" "${PYTHON_BIN}"
  exit 127
fi

event "python_import_probe_started" "executable=${PYTHON_BIN}"
if "${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${STARTUP_REPORT}"
import autotransition.qwen_image_edit.server
print("worker_module_import=ok")
PY
then
  event "python_import_probe_finished" "status=0"
else
  status=$?
  event "python_import_probe_failed" "status=${status}"
  collect_diagnostics "python_import_probe_failed" "${status}" "0" "import autotransition.qwen_image_edit.server"
  exit "${status}"
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$WORKER_PID" "$SALAD_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$WORKER_PID" "$SALAD_PID" "$HEARTBEAT_PID"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  if [[ "${status}" -ne 0 && "${STARTUP_FAILURE_REPORTED}" -eq 0 ]]; then
    STARTUP_FAILURE_REPORTED=1
    event "entrypoint_exit" "status=${status} ready=${STARTUP_READY}"
    collect_diagnostics "entrypoint_exit" "${status}" "0" "managed_process_exit"
  else
    event "entrypoint_exit" "status=${status} ready=${STARTUP_READY}"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

event "http_worker_starting" "host=${WORKER_HOST} port=${WORKER_PORT}"
echo "Starting Qwen Image Edit 2511 worker on ${WORKER_HOST}:${WORKER_PORT}"
"${PYTHON_BIN}" -m autotransition.qwen_image_edit.server 2>&1 | tee -a "${WORKER_LOG}" &
WORKER_PID=$!
event "http_worker_spawned" "pid=${WORKER_PID} log=${WORKER_LOG}"

if [[ "${SALAD_QUEUE_WORKER_ENABLED:-true}" == "true" ]]; then
  echo "Starting Salad queue worker"
  /usr/local/bin/salad-http-job-queue-worker &
  SALAD_PID=$!
else
  echo "Salad queue worker disabled; direct HTTP mode is enabled"
fi

if [[ -n "${LAUNCH_SERVER_HEARTBEAT_URL:-}" && -n "${WORKER_HEARTBEAT_TOKEN:-}" ]]; then
  echo "Starting launch-server heartbeat agent"
  /opt/qwen-image-edit-venv/bin/python -m autotransition.qwen_image_edit.heartbeat &
  HEARTBEAT_PID=$!
fi

probe_seconds="${QWEN_IMAGE_EDIT_STARTUP_PROBE_SECONDS:-300}"
event "health_probe_started" "url=http://127.0.0.1:${WORKER_PORT}/health timeoutSeconds=${probe_seconds}"
probe_deadline=$((SECONDS + probe_seconds))
while (( SECONDS < probe_deadline )); do
  if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    set +e
    wait "${WORKER_PID}"
    status=$?
    set -e
    [[ "${status}" -eq 0 ]] && status=1
    event "http_worker_exited_before_health" "status=${status}"
    collect_diagnostics "http_worker_exited_before_health" "${status}" "0" "${PYTHON_BIN} -m autotransition.qwen_image_edit.server"
    STARTUP_FAILURE_REPORTED=1
    exit "${status}"
  fi
  if curl -fsS --max-time 5 "http://127.0.0.1:${WORKER_PORT}/health" >"${PROBE_BODY}" 2>"${PROBE_ERROR}"; then
    STARTUP_READY=1
    event "health_ready" "response=$(tr '\n' ' ' <"${PROBE_BODY}" | cut -c1-1000)"
    break
  fi
  event "health_probe_pending" "remainingSeconds=$((probe_deadline - SECONDS)) error=$(tr '\n' ' ' <"${PROBE_ERROR}" | cut -c1-500)"
  sleep 5
done
if [[ "${STARTUP_READY}" -ne 1 ]]; then
  event "health_probe_timeout" "timeoutSeconds=${probe_seconds}"
  STARTUP_FAILURE_REPORTED=1
  collect_diagnostics "health_probe_timeout" "124" "0" "curl http://127.0.0.1:${WORKER_PORT}/health"
  exit 124
fi

PIDS=("$WORKER_PID")
[[ -z "$SALAD_PID" ]] || PIDS+=("$SALAD_PID")
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed Qwen Image Edit process exited with status ${status}"
if [[ "${status}" -ne 0 ]]; then
  STARTUP_FAILURE_REPORTED=1
  collect_diagnostics "managed_process_exit" "${status}" "0" "wait -n"
fi
exit "$status"

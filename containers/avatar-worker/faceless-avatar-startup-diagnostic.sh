#!/bin/sh
set -eu

PORT="${AVATAR_DIAGNOSTIC_PORT:-8090}"
export AVATAR_DIAGNOSTIC_PORT="$PORT"
MISSING=0

log() {
  printf '[avatar-startup] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

check_path() {
  path="$1"
  label="$2"
  if [ -e "$path" ]; then
    if [ -d "$path" ]; then
      log "$label present type=directory path=$path"
    else
      size=$(stat -c '%s' "$path" 2>/dev/null || printf 'unknown')
      log "$label present type=file bytes=$size path=$path"
    fi
  else
    log "$label MISSING path=$path"
    MISSING=1
  fi
}

log "diagnostic wrapper begin pid=$$ uid=$(id -u) gid=$(id -g) cwd=$(pwd) port=$PORT"
log "python=$(command -v python || printf 'missing')"
log "python_version=$(python --version 2>&1 || printf 'unavailable')"
log "working_directory=$(pwd)"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia_smi_present=true"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader 2>&1 || log "nvidia_smi_query_failed exit=$?"
else
  log "nvidia_smi_present=false"
fi

check_path "${KLEIN_4B_MODEL_PATH:-/models/flux2/flux-2-klein-4b.safetensors}" "flux_image_model"
check_path "${AE_MODEL_PATH:-/models/flux2/ae.safetensors}" "flux_autoencoder"
check_path "${FLUX2_TEXT_ENCODER_PATH:-/models/flux2/Qwen3-4B}" "flux_text_encoder"
check_path "${TRELLIS2_MODEL_PATH:-/models/trellis2}" "trellis_model"
check_path "/models/dinov3/config.json" "trellis_dinov3_config"
check_path "/models/dinov3/model.safetensors" "trellis_dinov3_weights"
check_path "/models/birefnet/config.json" "trellis_birefnet_config"
check_path "/models/birefnet/model.safetensors" "trellis_birefnet_weights"
check_path "/models/SkinTokens" "skintokens_model"
check_path "/app/tools/avatar/flux2_klein_generate.py" "flux_command_script"
check_path "/app/tools/avatar/trellis2_generate.py" "trellis_command_script"
check_path "/app/tools/tokenrig/adaptive_runner.py" "rig_command_script"
check_path "/app/src/autotransition/avatar/worker.py" "worker_source"

artifact_root="${AVATAR_ARTIFACT_ROOT:-/var/lib/faceless/avatar-jobs}"
if mkdir -p "$artifact_root"; then
  probe_file="$artifact_root/.startup-diagnostic-$$"
  if printf 'startup diagnostic\n' > "$probe_file"; then
    rm -f "$probe_file"
    log "artifact_root_writable=true path=$artifact_root"
  else
    log "artifact_root_writable=false path=$artifact_root"
    MISSING=1
  fi
else
  log "artifact_root_create_failed path=$artifact_root"
  MISSING=1
fi

if ! python - <<'PY'
import importlib
import os
import sys

print(f"[avatar-startup] python_executable={sys.executable}")
print(f"[avatar-startup] python_path={os.environ.get('PYTHONPATH', '')}")

for name in ("fastapi", "uvicorn", "torch"):
    module = importlib.import_module(name)
    print(f"[avatar-startup] import_ok module={name} version={getattr(module, '__version__', 'unknown')}")

import torch
print(f"[avatar-startup] torch_cuda_available={torch.cuda.is_available()}")
print(f"[avatar-startup] torch_cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[avatar-startup] torch_cuda_device_name={torch.cuda.get_device_name(0)}")

from autotransition.avatar.worker import create_avatar_worker_app
from autotransition.config import AvatarConfig

config = AvatarConfig.from_env()
print(f"[avatar-startup] avatar_config gpu_required={config.gpu_required} max_attempts={config.max_attempts}")
print(f"[avatar-startup] avatar_commands image={bool(config.image_command)} mesh={bool(config.mesh_command)} rig={bool(config.rig_command)}")
app = create_avatar_worker_app(config)
print(f"[avatar-startup] app_constructed routes={len(app.routes)}")
PY
then
  log "python_preflight_failed"
  exit 42
fi

if [ "$MISSING" -ne 0 ]; then
  log "filesystem_preflight_failed"
  exit 43
fi

log "preflight_passed launching_uvicorn port=$PORT"
exec python -c 'import os; from autotransition.avatar.worker import create_avatar_worker_app; from autotransition.config import AvatarConfig; import uvicorn; uvicorn.run(create_avatar_worker_app(AvatarConfig.from_env()), host="0.0.0.0", port=int(os.environ.get("AVATAR_DIAGNOSTIC_PORT", "8090")))'

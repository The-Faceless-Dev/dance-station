#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/app/src:/opt/TRELLIS.2:/opt/flux2'
export AVATAR_GPU_REQUIRED=1
export AVATAR_ARTIFACT_ROOT='/var/lib/faceless/avatar-jobs'
export AVATAR_MAX_ATTEMPTS=3
export AVATAR_REQUIRE_DEFORMATION_VALIDATOR=1
export AVATAR_MESH_MODEL_REVISION='trellis.2-4b'
export KLEIN_4B_MODEL_PATH='/models/flux2/flux-2-klein-4b.safetensors'
export AE_MODEL_PATH='/models/flux2/ae.safetensors'
export FLUX2_TEXT_ENCODER_PATH='/models/flux2/Qwen3-4B'
export TRELLIS2_MODEL_PATH='/models/trellis2'
export TRELLIS2_ALLOW_MODEL_DOWNLOAD=0
export TRELLIS2_PIPELINE_TYPE=1024
export TORCH_CUDA_ARCH_LIST=8.6
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export AVATAR_IMAGE_COMMAND='python /app/tools/avatar/flux2_klein_generate.py --prompt-file {prompt_file} --negative-prompt-file {negative_prompt_file} --output {output} --seed {seed} --reference-image {reference_image}'
export AVATAR_MESH_COMMAND='python /app/tools/avatar/trellis2_generate.py --image {image} --output-dir {output_dir} --output {output} --quality {quality}'
export AVATAR_RIG_COMMAND='python /app/tools/tokenrig/adaptive_runner.py --skintokens-repo /models/SkinTokens --input {input} --output {output} --manifest-output {manifest_output} --profile auto --use-transfer'

SALAD_PID=""
WORKER_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ -z "$SALAD_PID" ]] || kill -TERM "$SALAD_PID" 2>/dev/null || true
  [[ -z "$WORKER_PID" ]] || kill -TERM "$WORKER_PID" 2>/dev/null || true
  [[ -z "$SALAD_PID" ]] || wait "$SALAD_PID" 2>/dev/null || true
  [[ -z "$WORKER_PID" ]] || wait "$WORKER_PID" 2>/dev/null || true
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

echo "Starting avatar adapter on ${WORKER_HOST:-0.0.0.0}:${WORKER_PORT:-8080}"
python -m autotransition.avatar.salad_server &
WORKER_PID=$!

PIDS=("$WORKER_PID")
if [[ -n "$SALAD_PID" ]]; then
  PIDS+=("$SALAD_PID")
fi
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
echo "A managed worker process exited with status ${status}"
exit "$status"

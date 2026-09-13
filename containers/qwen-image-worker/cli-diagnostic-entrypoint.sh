#!/usr/bin/env bash
set -u

output_dir=/tmp/qwen-cli-diagnostic
mkdir -p "${output_dir}"
log_path="${output_dir}/sd-cli.log"
output_path="${output_dir}/sd-cli-output.png"

{
  echo "diagnostic=direct-sd-cli"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "binary=/usr/local/bin/sd-cli"
  /usr/local/bin/sd-cli \
    --diffusion-model "${QWEN_IMAGE_DIFFUSION_MODEL}" \
    --vae "${QWEN_IMAGE_VAE}" \
    --llm "${QWEN_IMAGE_TEXT_ENCODER}" \
    --cfg-scale 2.5 \
    --sampling-method euler \
    --steps 20 \
    --flow-shift 3 \
    --seed 184729 \
    -H 1024 \
    -W 1024 \
    -p "A small red ceramic teapot with a blue flower painted on it, centered on a wooden table, soft studio lighting" \
    -o "${output_path}" \
    -v
  status=$?
  echo "exit_status=${status}"
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${log_path}" 2>&1

cp "${log_path}" /tmp/sd-cli.log 2>/dev/null || true
if [[ -f "${output_path}" ]]; then
  cp "${output_path}" /tmp/sd-cli-output.png 2>/dev/null || true
fi

echo "CLI diagnostic complete; serving /tmp on port 8080" >&2
exec python3 -m http.server 8080 --bind 0.0.0.0 --directory /tmp

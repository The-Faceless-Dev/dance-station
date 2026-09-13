#!/usr/bin/env bash
set -u

output_dir=/tmp/qwen-cli-diagnostic
mkdir -p "${output_dir}"
export SD_QWEN_TENSOR_DIAGNOSTICS=1

# Keep the RunPod port observable while the probes run. The previous harness
# exposed it only after CPU inference, which made a healthy pod look absent.
file_server_pid=""
python3 -m http.server 8080 --bind 0.0.0.0 --directory /tmp >"${output_dir}/file-server.log" 2>&1 &
file_server_pid=$!
cleanup() {
  if [[ -n "${file_server_pid}" ]] && kill -0 "${file_server_pid}" 2>/dev/null; then
    kill -TERM "${file_server_pid}" 2>/dev/null || true
    wait "${file_server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run_direct_cli() {
  local log_path="$1"
  local output_path="$2"
  local width="$3"
  local height="$4"
  local steps="$5"
  local label="$6"

  {
    echo "diagnostic=${label}"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "binary=/usr/local/bin/sd-cli"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
    /usr/local/bin/sd-cli \
      --diffusion-model "${QWEN_IMAGE_DIFFUSION_MODEL}" \
      --vae "${QWEN_IMAGE_VAE}" \
      --llm "${QWEN_IMAGE_TEXT_ENCODER}" \
      --cfg-scale 2.5 \
      --sampling-method euler \
      --steps "${steps}" \
      --flow-shift 3 \
      --seed 184729 \
      -H "${height}" \
      -W "${width}" \
      -p "A small red ceramic teapot with a blue flower painted on it, centered on a wooden table, soft studio lighting" \
      -o "${output_path}" \
      -v
    local status=$?
    echo "exit_status=${status}"
    echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return "${status}"
  } >"${log_path}" 2>&1
}

run_direct_cli "${output_dir}/sd-cli.log" "${output_dir}/sd-cli-output.png" 1024 1024 20 "direct-sd-cli-cuda"
direct_status=$?
cp "${output_dir}/sd-cli.log" /tmp/sd-cli.log 2>/dev/null || true
if [[ -f "${output_dir}/sd-cli-output.png" ]]; then
  cp "${output_dir}/sd-cli-output.png" /tmp/sd-cli-output.png 2>/dev/null || true
fi

server_log="${output_dir}/sd-server.log"
server_port=1234
server_pid=""
{
  echo "diagnostic=sd-server-http"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "binary=/usr/local/bin/sd-server"
  echo "listen=127.0.0.1:${server_port}"
  /usr/local/bin/sd-server \
    --diffusion-model "${QWEN_IMAGE_DIFFUSION_MODEL}" \
    --vae "${QWEN_IMAGE_VAE}" \
    --llm "${QWEN_IMAGE_TEXT_ENCODER}" \
    --listen-ip 127.0.0.1 \
    --listen-port "${server_port}" \
    --backend cuda \
    --log-level verbose
  echo "server_exit_status=$?"
} >"${server_log}" 2>&1 &
server_pid=$!

export QWEN_DIAGNOSTIC_SERVER_PORT="${server_port}"

python3 - <<'PY' >"${output_dir}/sd-server-http-client.log" 2>&1
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

root = Path("/tmp/qwen-cli-diagnostic")
port = int(os.environ["QWEN_DIAGNOSTIC_SERVER_PORT"])
base = f"http://127.0.0.1:{port}"
request_body = {
    "prompt": "A small red ceramic teapot with a blue flower painted on it, centered on a wooden table, soft studio lighting",
    "negative_prompt": "",
    "clip_skip": -1,
    "width": 1024,
    "height": 1024,
    "strength": 0.75,
    "seed": 184729,
    "batch_count": 1,
    "sample_params": {
        "scheduler": "discrete",
        "sample_method": "euler",
        "sample_steps": 20,
        "eta": 1.0,
        "flow_shift": 3.0,
        "guidance": {"txt_cfg": 2.5},
    },
    "lora": [],
    "output_format": "png",
    "output_compression": 100,
}
(root / "sd-server-request.json").write_text(json.dumps(request_body, indent=2) + "\n", encoding="utf-8")

def request(method, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))

started = time.monotonic()
capabilities = None
last_error = None
while time.monotonic() - started < 180:
    try:
        status, capabilities = request("GET", "/sdcpp/v1/capabilities", timeout=5)
        print(f"capabilities_status={status}")
        break
    except Exception as exc:
        last_error = repr(exc)
        time.sleep(2)
else:
    raise RuntimeError(f"server did not become reachable: {last_error}")
(root / "sd-server-capabilities.json").write_text(json.dumps(capabilities, indent=2) + "\n", encoding="utf-8")

status, accepted = request("POST", "/sdcpp/v1/img_gen", request_body, timeout=30)
print(f"submit_status={status}")
(root / "sd-server-submit.json").write_text(json.dumps(accepted, indent=2) + "\n", encoding="utf-8")
job_id = accepted.get("id")
if not job_id:
    raise RuntimeError(f"server did not return a job id: {accepted}")

deadline = time.monotonic() + 900
last_status = None
while time.monotonic() < deadline:
    status, job = request("GET", f"/sdcpp/v1/jobs/{job_id}", timeout=30)
    current = job.get("status")
    if current != last_status:
        print(f"job_status={current}")
        last_status = current
    if current in {"completed", "failed", "cancelled"}:
        (root / "sd-server-final.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        if current != "completed":
            raise RuntimeError(f"server job did not complete: {job}")
        result = job.get("result") or {}
        images = result.get("images") or []
        if not images or not images[0].get("b64_json"):
            raise RuntimeError(f"server job completed without image: {job}")
        (root / "sd-server-output.png").write_bytes(base64.b64decode(images[0]["b64_json"], validate=True))
        break
    time.sleep(2)
else:
    raise TimeoutError(f"server job exceeded diagnostic deadline: {job_id}")
print(f"elapsed_seconds={time.monotonic() - started:.2f}")
PY
server_client_status=$?

if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
  kill -TERM "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
fi
echo "direct_cli_exit_status=${direct_status}" >"${output_dir}/diagnostic-summary.txt"
echo "server_http_client_exit_status=${server_client_status}" >>"${output_dir}/diagnostic-summary.txt"
echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${output_dir}/diagnostic-summary.txt"

# This is a finiteness-only CPU probe. It uses one tiny step so it can answer
# whether CUDA is the differentiator without spending a full CPU generation.
export CUDA_VISIBLE_DEVICES=-1
run_direct_cli "${output_dir}/sd-cli-cpu.log" "${output_dir}/sd-cli-cpu-output.png" 256 256 1 "direct-sd-cli-cpu-smoke"
cpu_status=$?
echo "cpu_cli_exit_status=${cpu_status}" >>"${output_dir}/diagnostic-summary.txt"

echo "CLI/server/CPU diagnostic complete; serving /tmp on port 8080" >&2
wait "${file_server_pid}"

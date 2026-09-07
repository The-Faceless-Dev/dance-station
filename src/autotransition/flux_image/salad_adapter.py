from __future__ import annotations

import asyncio
import json
import mimetypes
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request

from .config import FluxImageConfig
from .contracts import FluxImageRequest, FluxLoRARequest
from .worker import FluxImageWorker


TERMINAL = {"succeeded", "failed", "cancelled"}


def _job_id(payload: dict[str, Any]) -> str:
    value = str(payload.get("job_id") or "").strip()
    if not value or any(char in value for char in "\\/"):
        raise ValueError("job is missing a valid job_id")
    return value


def _callback(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    value = payload.get("callback") or {}
    callback_url = str(value.get("url") or "")
    complete_url = str(value.get("complete_url") or "")
    progress_url = str(value.get("progress_url") or "")
    token = str(value.get("token") or "")
    if not callback_url or not complete_url or not token:
        raise ValueError("job is missing its callback contract")
    return callback_url, complete_url, progress_url, token


def _post_json(url: str, token: str, payload: Any, *, timeout: float = 120) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Content-Length": str(len(data)), "X-Job-Callback-Token": token},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"callback HTTP {exc.code}: {detail[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"callback failed to {url}: {exc}") from exc


def _source(value: dict[str, Any]) -> tuple[str, str]:
    url = str(value.get("sourceUrl") or value.get("source_url") or value.get("url") or "").strip()
    filename = Path(str(value.get("fileName") or value.get("file_name") or "adapter.safetensors")).name
    return url, filename


def _download_lora(url: str, filename: str, config: FluxImageConfig, index: int) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LoRA input must use an HTTP(S) source URL")
    if Path(filename).suffix.lower() != ".safetensors":
        raise ValueError("LoRA input must be a .safetensors file")
    handle = tempfile.NamedTemporaryFile(prefix=f"flux-lora-{index}-", suffix=".safetensors", delete=False)
    path = Path(handle.name)
    total = 0
    try:
        request = UrlRequest(url, headers={"Accept": "application/octet-stream"})
        with urlopen(request, timeout=180) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > config.max_lora_bytes:
                raise ValueError("LoRA exceeds the worker size limit")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > config.max_lora_bytes:
                    raise ValueError("LoRA exceeds the worker size limit")
                handle.write(chunk)
        if total == 0:
            raise ValueError("LoRA was empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        handle.close()


def _loras_from_payload(payload: dict[str, Any], config: FluxImageConfig) -> tuple[tuple[FluxLoRARequest, ...], list[Path]]:
    parameters = payload.get("parameters") or {}
    raw = parameters.get("loras") if isinstance(parameters, dict) else None
    parameter_values = list(raw) if isinstance(raw, list) else []
    input_values: list[dict[str, Any]] = []
    for item in payload.get("inputs") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "lora" or role.startswith("lora_") or role.startswith("adapter"):
            input_values.append(item)
    # Launch-server payloads may describe the same adapter in parameters and
    # inputs. Merge those descriptions once instead of downloading it twice.
    values: list[dict[str, Any]] = []
    if parameter_values:
        for index, item in enumerate(parameter_values):
            if not isinstance(item, dict):
                raise ValueError("each LoRA must be an object")
            merged = dict(input_values[index]) if index < len(input_values) else {}
            merged.update(item)
            values.append(merged)
        values.extend(input_values[len(parameter_values) :])
    else:
        values = input_values
    if len(values) > config.max_loras:
        raise ValueError(f"at most {config.max_loras} LoRAs may be supplied")
    paths: list[Path] = []
    requests: list[FluxLoRARequest] = []
    try:
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                raise ValueError("each LoRA must be an object")
            url, filename = _source(item)
            path = _download_lora(url, filename, config, index)
            paths.append(path)
            requests.append(FluxLoRARequest(source_url=url, file_name=filename, scale=float(item.get("scale", 1.0)), path=path))
        return tuple(requests), paths
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def _request(payload: dict[str, Any], loras: tuple[FluxLoRARequest, ...]) -> FluxImageRequest:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("FLUX image job parameters must be an object")
    return FluxImageRequest(
        prompt=str(parameters.get("prompt") or payload.get("prompt") or ""),
        negative_prompt=str(parameters.get("negative_prompt") or parameters.get("negativePrompt") or ""),
        width=int(parameters.get("width", 960)),
        height=int(parameters.get("height", 1664)),
        steps=int(parameters.get("steps", parameters.get("num_inference_steps", 4))),
        true_cfg_scale=float(parameters.get("true_cfg_scale", parameters.get("trueCfgScale", 1.0))),
        seed=int(parameters["seed"]) if parameters.get("seed") is not None else None,
        loras=loras,
        external_job_id=_job_id(payload),
        payment_intent_id=str(parameters.get("payment_intent_id") or parameters.get("paymentIntentId") or "") or None,
    )


def _artifact_role(name: str) -> str:
    return "preview" if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else "metadata"


def _upload_artifact(url: str, token: str, path: Path, artifact: dict[str, Any]) -> str:
    name = Path(str(artifact.get("name") or path.name)).name
    body = path.read_bytes()
    request = UrlRequest(
        url,
        data=body,
        headers={
            "Content-Type": mimetypes.guess_type(name)[0] or "application/octet-stream",
            "Content-Length": str(len(body)),
            "X-Job-Callback-Token": token,
            "X-Artifact-Role": _artifact_role(name),
            "X-Artifact-Variant": "flux-image-output",
            "X-Artifact-File-Name": name,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=1800) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"artifact callback HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"artifact callback failed: {exc}") from exc
    artifact_id = str(result.get("id") or result.get("artifactId") or "")
    if not artifact_id:
        raise RuntimeError("artifact callback returned no artifact id")
    return artifact_id


async def _progress(url: str, token: str, job_id: str, job: dict[str, Any], sequence: int, message: str | None = None, status: str = "running") -> None:
    if not url:
        return
    progress = max(0.0, min(1.0, float(job.get("progress") or 0)))
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "runtime": "flux-image",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": progress,
        "completed_steps": round(progress * 100),
        "total_steps": 100,
        "attempt": job.get("attempt", 0),
        "message": message,
        "stage": job.get("stage"),
        "sequence": sequence,
        "updated_at": job.get("updated_at") or "",
    }
    try:
        await asyncio.to_thread(_post_json, url, token, payload, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "flux_progress_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _fail_callback(complete_url: str, token: str, job_id: str, code: str, message: str) -> None:
    """Tell launch-server about adapter failures before returning HTTP 500.

    Salad uses the adapter's HTTP response to decide whether to retry, but the
    launch server owns payment/refund state. Without this callback a worker
    exception can leave the launch job in ``running`` until the provider gives
    up, even though the adapter already knows the job failed.
    """
    fail_url = f"{complete_url.rsplit('/complete', 1)[0]}/fail"
    try:
        await asyncio.to_thread(
            _post_json,
            fail_url,
            token,
            {"errorCode": code[:120], "errorMessage": message[:4000], "artifactIds": []},
            timeout=10,
        )
    except Exception as exc:
        print(json.dumps({"event": "flux_failure_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _run_queue_job(payload: dict[str, Any], worker: FluxImageWorker, config: FluxImageConfig) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, token = _callback(payload)
    paths: list[Path] = []
    failure_callback_sent = False
    try:
        # This is deliberately before LoRA downloads, request validation, and
        # model execution. It proves that Salad reached this adapter and gives
        # launch-server a live timestamp before any expensive work begins.
        await _progress(
            progress_url,
            token,
            job_id,
            {"status": "running", "stage": "accepted", "progress": 0.0, "attempt": 0},
            1,
            "FLUX worker accepted the job",
        )
        loras, paths = _loras_from_payload(payload, config)
        request = _request(payload, loras)
        job = await worker.submit(request)
        sequence = 2
        await _progress(progress_url, token, job_id, worker.get(job.id), sequence, "FLUX worker accepted the job")
        deadline = time.monotonic() + config.job_timeout_seconds + 60
        last_key: tuple[Any, ...] | None = None
        while time.monotonic() < deadline:
            current = worker.get(job.id)
            key = (current.get("status"), current.get("stage"), current.get("progress"), current.get("updated_at"))
            if key != last_key:
                last_key = key
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence)
            if current.get("status") in TERMINAL:
                if current.get("status") != "succeeded":
                    failure = current.get("failure") or {}
                    message = f"[{failure.get('code', 'flux_image_worker_failed')}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'FLUX image generation failed')}"
                    artifact_ids = []
                    for artifact in current.get("artifacts") or []:
                        path = Path(str(artifact.get("path") or ""))
                        if path.is_file():
                            try:
                                artifact_ids.append(await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact))
                            except Exception as upload_error:
                                print(json.dumps({"event": "flux_failure_artifact_upload_failed", "jobId": job_id, "error": str(upload_error)}), flush=True)
                    await _fail_callback(complete_url, token, job_id, failure.get("code", "flux_image_worker_failed"), message)
                    failure_callback_sent = True
                    sequence += 1
                    await _progress(progress_url, token, job_id, current, sequence, message, "failed")
                    raise RuntimeError(message)
                artifact_ids = []
                for artifact in current.get("artifacts") or []:
                    path = Path(str(artifact.get("path") or ""))
                    if not path.is_file():
                        raise RuntimeError(f"FLUX artifact is missing: {artifact.get('name')}")
                    artifact_ids.append(await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact))
                await asyncio.to_thread(_post_json, complete_url, token, {"artifactIds": artifact_ids})
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence, "FLUX worker completed the job", "succeeded")
                return {"schema_version": 1, "runtime": "flux-image", "status": "succeeded", "job_id": job_id, "artifact_ids": artifact_ids}
            await asyncio.sleep(2)
        raise TimeoutError(f"FLUX image job exceeded the {config.job_timeout_seconds:.0f}s worker timeout")
    except Exception as exc:
        if not failure_callback_sent:
            code = "flux_image_worker_timeout" if isinstance(exc, TimeoutError) else "flux_image_worker_failed"
            await _fail_callback(complete_url, token, job_id, code, f"[{code}] {exc}")
        raise
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def install_salad_routes(app: FastAPI, worker: FluxImageWorker, config: FluxImageConfig) -> FastAPI:
    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "flux-image", "flux-image-worker", "flux_image"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            return await _run_queue_job(payload, worker, config)
        except HTTPException:
            raise
        except Exception as exc:
            job_id = str(payload.get("job_id") or "unknown")
            print(json.dumps({"event": "flux_queue_job_failed", "jobId": job_id, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app

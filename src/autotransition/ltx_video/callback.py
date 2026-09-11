from __future__ import annotations

import asyncio
import json
import mimetypes
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .config import LtxVideoConfig
from .contracts import LtxConditioningImage, LtxVideoRequest
from .worker import LtxVideoWorker


TERMINAL = {"succeeded", "failed"}


def _job_id(payload: dict[str, Any]) -> str:
    value = str(payload.get("job_id") or "").strip()
    if not value or any(char in value for char in "\\/"):
        raise ValueError("job is missing a valid job_id")
    return value


def _callback(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    value = payload.get("callback") or {}
    callback_url = str(value.get("url") or "")
    complete_url = str(value.get("complete_url") or value.get("completeUrl") or "")
    progress_url = str(value.get("progress_url") or value.get("progressUrl") or "")
    token = str(value.get("token") or "")
    if not callback_url or not complete_url or not token:
        raise ValueError("job is missing its callback contract")
    return callback_url, complete_url, progress_url, token


def _post_json(url: str, token: str, payload: Any, *, timeout: float = 120) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = UrlRequest(url, data=data, headers={"Content-Type": "application/json", "Content-Length": str(len(data)), "X-Job-Callback-Token": token}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"callback HTTP {exc.code}: {detail[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"callback failed to {url}: {exc}") from exc


def _source(value: Any, *, default_filename: str) -> tuple[str, Path | None, str]:
    if not isinstance(value, dict):
        raise ValueError("media input must be an object")
    source_url = str(value.get("sourceUrl") or value.get("source_url") or value.get("url") or "").strip()
    raw_path = value.get("path")
    path = Path(str(raw_path)).expanduser() if raw_path else None
    filename = Path(str(value.get("filename") or value.get("fileName") or default_filename)).name
    return source_url, path, filename


def request_from_payload(payload: dict[str, Any]) -> LtxVideoRequest:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("LTX video job parameters must be an object")
    prompt = str(parameters.get("prompt") or payload.get("prompt") or "")
    if not prompt.strip():
        raise ValueError("LTX video job requires a prompt")
    raw_conditions = parameters.get("conditioning_images") or parameters.get("conditioningImages") or []
    if not raw_conditions:
        raw_conditions = [item for item in payload.get("inputs") or [] if isinstance(item, dict) and str(item.get("role") or "").lower() in {"image", "reference", "conditioning", "character_reference"}]
    conditions: list[LtxConditioningImage] = []
    for item in raw_conditions:
        source_url, path, filename = _source(item, default_filename="conditioning.png")
        conditions.append(LtxConditioningImage(
            source_url=source_url,
            path=path,
            filename=filename,
            frame_index=int(item.get("frameIndex", item.get("frame_index", item.get("frame", 0)))),
            strength=float(item.get("strength", 1.0)),
            mode=str(item.get("mode", "guide")),
            crf=int(item["crf"]) if item.get("crf") is not None else None,
        ))
    audio_value = parameters.get("audio")
    audio_mode = str(parameters.get("audio_mode") or parameters.get("audioMode") or "off")
    source_audio_url, source_audio_path = "", None
    if isinstance(audio_value, dict):
        source_audio_url, source_audio_path, _ = _source(audio_value, default_filename="source-audio")
        audio_mode = "source"
    loras: list[dict[str, Any]] = []
    for item in parameters.get("loras") or parameters.get("LoRAs") or []:
        if not isinstance(item, dict):
            raise ValueError("each LTX LoRA must be an object")
        loras.append(dict(item))
    return LtxVideoRequest(
        prompt=prompt,
        negative_prompt=str(parameters.get("negative_prompt") or parameters.get("negativePrompt") or "") or None,
        aspect_ratio=str(parameters.get("aspect_ratio") or parameters.get("aspectRatio") or "16:9"),
        width=int(parameters["width"]) if parameters.get("width") is not None else None,
        height=int(parameters["height"]) if parameters.get("height") is not None else None,
        num_frames=int(parameters["num_frames"]) if parameters.get("num_frames") is not None else None,
        duration_seconds=float(parameters["duration_seconds"]) if parameters.get("duration_seconds") is not None else (float(parameters["durationSeconds"]) if parameters.get("durationSeconds") is not None else None),
        frame_rate=float(parameters.get("frame_rate", parameters.get("frameRate", 24.0))),
        stage_1_steps=int(parameters["stage_1_steps"]) if parameters.get("stage_1_steps") is not None else None,
        stage_2_steps=int(parameters["stage_2_steps"]) if parameters.get("stage_2_steps") is not None else None,
        seed=int(parameters["seed"]) if parameters.get("seed") is not None else None,
        conditioning_images=tuple(conditions),
        audio_mode=audio_mode,
        source_audio_url=source_audio_url,
        source_audio_path=source_audio_path,
        loras=tuple(loras),
        output_format=str(parameters.get("output_format") or parameters.get("outputFormat") or "mp4"),
        retain_intermediates=parameters.get("retain_intermediates", parameters.get("retainIntermediates")),
        external_job_id=_job_id(payload),
        payment_intent_id=str(parameters.get("payment_intent_id") or parameters.get("paymentIntentId") or "") or None,
    )


def _uploaded_artifact(artifact: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "artifactId": artifact_id,
        "name": Path(str(artifact.get("name") or "artifact")).name,
        "mediaType": str(artifact.get("media_type") or artifact.get("mediaType") or "application/octet-stream"),
        "sizeBytes": int(artifact.get("size_bytes") or artifact.get("sizeBytes") or 0),
        "sha256": str(artifact.get("sha256") or ""),
        "role": str(artifact.get("role") or "output"),
    }


def _upload_artifact(url: str, token: str, artifact: dict[str, Any]) -> str:
    path = Path(str(artifact.get("path") or ""))
    if not path.is_file():
        raise RuntimeError(f"LTX artifact is missing: {artifact.get('name')}")
    name = Path(str(artifact.get("name") or path.name)).name
    body = path.read_bytes()
    request = UrlRequest(url, data=body, headers={
        "Content-Type": str(artifact.get("media_type") or mimetypes.guess_type(name)[0] or "application/octet-stream"),
        "Content-Length": str(len(body)),
        "X-Job-Callback-Token": token,
        "X-Artifact-Role": str(artifact.get("role") or "output"),
        "X-Artifact-Variant": "ltx-2.5-video",
        "X-Artifact-File-Name": name,
    }, method="POST")
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
    value = max(0.0, min(1.0, float(job.get("progress") or 0)))
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "runtime": "ltx-video",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": value,
        "completed_steps": round(value * 100),
        "total_steps": 100,
        "attempt": job.get("attempt", 0),
        "message": message or job.get("message"),
        "stage": job.get("stage"),
        "sequence": sequence,
        "updated_at": job.get("updated_at") or "",
    }
    try:
        await asyncio.to_thread(_post_json, url, token, payload, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "ltx_progress_callback_failed", "jobId": job_id, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)


async def _fail_callback(complete_url: str, token: str, code: str, message: str, artifact_ids: list[str]) -> None:
    fail_url = f"{complete_url.rsplit('/complete', 1)[0]}/fail"
    await asyncio.to_thread(_post_json, fail_url, token, {"errorCode": code[:120], "errorMessage": message[:4000], "artifactIds": artifact_ids}, timeout=10)


async def run_queue_job(payload: dict[str, Any], worker: LtxVideoWorker, config: LtxVideoConfig) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, token = _callback(payload)
    failure_sent = False
    try:
        await _progress(progress_url, token, job_id, {"stage": "accepted", "progress": 0.0, "attempt": 0}, 1, "LTX video worker accepted the job")
        job = await worker.submit(request_from_payload(payload))
        sequence = 2
        deadline = time.monotonic() + config.job_timeout_seconds + 60
        last_key: tuple[Any, ...] | None = None
        while time.monotonic() < deadline:
            current = worker.get(job.id)
            key = (current.get("status"), current.get("stage"), current.get("progress"), current.get("updated_at"), current.get("message"))
            if key != last_key:
                last_key = key
                sequence += 1
                await _progress(progress_url, token, job_id, current, sequence)
            if current.get("status") in TERMINAL:
                if current.get("status") != "succeeded":
                    failure = current.get("failure") or {}
                    message = f"[{failure.get('code', 'ltx_video_worker_failed')}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'LTX video generation failed')}"
                    ids: list[str] = []
                    for artifact in current.get("artifacts") or []:
                        try:
                            ids.append(await asyncio.to_thread(_upload_artifact, callback_url, token, artifact))
                        except Exception as exc:
                            print(json.dumps({"event": "ltx_failure_artifact_upload_failed", "jobId": job_id, "name": artifact.get("name"), "error": str(exc)}), flush=True)
                    await _fail_callback(complete_url, token, failure.get("code", "ltx_video_worker_failed"), message, ids)
                    failure_sent = True
                    await _progress(progress_url, token, job_id, current, sequence + 1, message, "failed")
                    raise RuntimeError(message)
                ids = []
                uploaded = []
                for artifact in current.get("artifacts") or []:
                    artifact_id = await asyncio.to_thread(_upload_artifact, callback_url, token, artifact)
                    ids.append(artifact_id)
                    uploaded.append(_uploaded_artifact(artifact, artifact_id))
                await asyncio.to_thread(_post_json, complete_url, token, {"artifactIds": ids, "artifacts": uploaded})
                await _progress(progress_url, token, job_id, current, sequence + 1, "LTX video worker completed the job", "succeeded")
                return {"schema_version": 1, "runtime": "ltx-video", "status": "succeeded", "job_id": job_id, "artifact_ids": ids, "artifacts": uploaded}
            await asyncio.sleep(2)
        raise TimeoutError(f"LTX video job exceeded the {config.job_timeout_seconds:.0f}s worker timeout")
    except Exception as exc:
        if not failure_sent:
            try:
                await _fail_callback(complete_url, token, "ltx_video_worker_timeout" if isinstance(exc, TimeoutError) else "ltx_video_worker_failed", str(exc), [])
            except Exception as callback_exc:
                print(json.dumps({"event": "ltx_failure_callback_failed", "jobId": job_id, "error": str(callback_exc)}), flush=True)
        raise


def install_callback_routes(app: FastAPI, worker: LtxVideoWorker, config: LtxVideoConfig) -> None:
    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "ltx-video", "ltx-video-worker", "ltx_2_5"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            return await run_queue_job(payload, worker, config)
        except HTTPException:
            raise
        except Exception as exc:
            print(json.dumps({"event": "ltx_queue_job_failed", "jobId": payload.get("job_id"), "errorType": type(exc).__name__, "error": str(exc)}), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

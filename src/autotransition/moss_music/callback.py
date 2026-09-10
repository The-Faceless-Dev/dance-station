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

from .config import MossMusicConfig
from .contracts import MossAudioInput, MossMusicRequest
from .worker import MossMusicWorker


TERMINAL = {"succeeded", "failed", "cancelled"}


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


def _audio_from_payload(payload: dict[str, Any]) -> MossAudioInput:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("MOSS-Music job parameters must be an object")
    value = parameters.get("audio") or payload.get("audio")
    if not isinstance(value, dict):
        for item in payload.get("inputs") or []:
            if isinstance(item, dict) and str(item.get("role") or "").lower() in {"audio", "source_audio", "music"}:
                value = item
                break
    if not isinstance(value, dict):
        raise ValueError("MOSS-Music job requires an audio input")
    source_url = str(value.get("sourceUrl") or value.get("source_url") or value.get("url") or "").strip()
    raw_path = value.get("path")
    path = Path(str(raw_path)).expanduser() if raw_path else None
    filename = Path(str(value.get("filename") or value.get("fileName") or value.get("file_name") or "audio")).name
    return MossAudioInput(source_url=source_url, filename=filename, path=path)


def request_from_payload(payload: dict[str, Any]) -> MossMusicRequest:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("MOSS-Music job parameters must be an object")
    audio = _audio_from_payload(payload)
    return MossMusicRequest(
        audio=audio,
        analysis_profile=str(parameters.get("analysis_profile") or parameters.get("analysisProfile") or "visual_sync_v1"),
        prompt=str(parameters.get("prompt") or "") or None,
        event_resolution_ms=int(parameters.get("event_resolution_ms", parameters.get("eventResolutionMs", 80))),
        include_semantic_events=bool(parameters.get("include_semantic_events", parameters.get("includeSemanticEvents", True))),
        include_dense_features=bool(parameters.get("include_dense_features", parameters.get("includeDenseFeatures", True))),
        temperature=float(parameters.get("temperature", 0.0)),
        external_job_id=_job_id(payload),
        payment_intent_id=str(parameters.get("payment_intent_id") or parameters.get("paymentIntentId") or "") or None,
    )


def _artifact_role(name: str) -> str:
    return "primary" if Path(name).name == "analysis.json" else "metadata"


def _uploaded_artifact(artifact: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    """Return remote-safe metadata without exposing the worker filesystem path."""

    return {
        "id": artifact_id,
        "artifactId": artifact_id,
        "name": Path(str(artifact.get("name") or "artifact")).name,
        "mediaType": str(artifact.get("media_type") or artifact.get("mediaType") or "application/octet-stream"),
        "sizeBytes": int(artifact.get("size_bytes") or artifact.get("sizeBytes") or 0),
        "sha256": str(artifact.get("sha256") or ""),
    }


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
            "X-Artifact-Variant": "moss-music-analysis",
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
        "runtime": "moss-music",
        "status": status,
        "phase": job.get("stage") or "processing",
        "progress": progress,
        "completed_steps": round(progress * 100),
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
        print(json.dumps({"event": "moss_progress_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _fail_callback(
    complete_url: str,
    token: str,
    job_id: str,
    code: str,
    message: str,
    artifact_ids: list[str] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> None:
    fail_url = f"{complete_url.rsplit('/complete', 1)[0]}/fail"
    try:
        await asyncio.to_thread(
            _post_json,
            fail_url,
            token,
            {
                "errorCode": code[:120],
                "errorMessage": message[:4000],
                "artifactIds": artifact_ids or [],
                "artifacts": artifacts or [],
            },
            timeout=10,
        )
    except Exception as exc:
        print(json.dumps({"event": "moss_failure_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def run_queue_job(payload: dict[str, Any], worker: MossMusicWorker, config: MossMusicConfig) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, token = _callback(payload)
    failure_callback_sent = False
    try:
        await _progress(
            progress_url,
            token,
            job_id,
            {"status": "running", "stage": "accepted", "progress": 0.0, "attempt": 0},
            1,
            "MOSS-Music worker accepted the job",
        )
        request = request_from_payload(payload)
        job = await worker.submit(request)
        sequence = 2
        await _progress(progress_url, token, job_id, worker.get(job.id), sequence, "MOSS-Music worker accepted the job")
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
                    message = f"[{failure.get('code', 'moss_music_worker_failed')}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'MOSS-Music analysis failed')}"
                    failure_artifact_ids: list[str] = []
                    failure_artifacts: list[dict[str, Any]] = []
                    for artifact in current.get("artifacts") or []:
                        path = Path(str(artifact.get("path") or ""))
                        if path.is_file():
                            try:
                                artifact_id = await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact)
                                failure_artifact_ids.append(artifact_id)
                                failure_artifacts.append(_uploaded_artifact(artifact, artifact_id))
                            except Exception as upload_error:
                                print(json.dumps({"event": "moss_failure_artifact_upload_failed", "jobId": job_id, "error": str(upload_error)}), flush=True)
                    await _fail_callback(
                        complete_url,
                        token,
                        job_id,
                        failure.get("code", "moss_music_worker_failed"),
                        message,
                        failure_artifact_ids,
                        failure_artifacts,
                    )
                    failure_callback_sent = True
                    await _progress(progress_url, token, job_id, current, sequence + 1, message, "failed")
                    raise RuntimeError(message)
                artifact_ids = []
                uploaded_artifacts = []
                for artifact in current.get("artifacts") or []:
                    path = Path(str(artifact.get("path") or ""))
                    if not path.is_file():
                        raise RuntimeError(f"MOSS-Music artifact is missing: {artifact.get('name')}")
                    artifact_id = await asyncio.to_thread(_upload_artifact, callback_url, token, path, artifact)
                    artifact_ids.append(artifact_id)
                    uploaded_artifacts.append(_uploaded_artifact(artifact, artifact_id))
                await asyncio.to_thread(
                    _post_json,
                    complete_url,
                    token,
                    {"artifactIds": artifact_ids, "artifacts": uploaded_artifacts},
                )
                await _progress(progress_url, token, job_id, current, sequence + 1, "MOSS-Music worker completed the job", "succeeded")
                return {
                    "schema_version": 1,
                    "runtime": "moss-music",
                    "status": "succeeded",
                    "job_id": job_id,
                    "artifact_ids": artifact_ids,
                    "artifacts": uploaded_artifacts,
                }
            await asyncio.sleep(2)
        raise TimeoutError(f"MOSS-Music job exceeded the {config.job_timeout_seconds:.0f}s worker timeout")
    except Exception as exc:
        if not failure_callback_sent:
            code = "moss_music_worker_timeout" if isinstance(exc, TimeoutError) else "moss_music_worker_failed"
            await _fail_callback(complete_url, token, job_id, code, f"[{code}] {exc}")
        raise


def install_callback_routes(app: FastAPI, worker: MossMusicWorker, config: MossMusicConfig) -> FastAPI:
    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "moss-music", "moss-music-worker", "moss_music"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            return await run_queue_job(payload, worker, config)
        except HTTPException:
            raise
        except Exception as exc:
            job_id = str(payload.get("job_id") or "unknown")
            print(json.dumps({"event": "moss_queue_job_failed", "jobId": job_id, "errorType": type(exc).__name__, "error": str(exc)}), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app

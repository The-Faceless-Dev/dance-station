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

from .config import MuLaCoverConfig
from .contracts import MuLaCoverRequest
from .worker import MuLaCoverWorker


TERMINAL = {"succeeded", "failed", "cancelled"}


def _value(payload: dict[str, Any], key: str, *aliases: str) -> Any:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("MuLaCover job parameters must be an object")
    for source in (parameters, payload):
        for candidate in (key, *aliases):
            if candidate in source:
                return source[candidate]
    return None


def _source(payload: dict[str, Any], key: str, *aliases: str) -> tuple[str, str]:
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("MuLaCover job parameters must be an object")
    sources = (parameters, payload)
    url_keys = (key, f"{key}_url", f"{key}Url", *aliases)
    path_keys = (f"{key}_path", f"{key}Path")
    for source in sources:
        for candidate in url_keys:
            if candidate not in source:
                continue
            value = source[candidate]
            if isinstance(value, dict):
                return str(value.get("url") or value.get("source_url") or value.get("sourceUrl") or ""), str(value.get("path") or "")
            value = str(value or "")
            parsed = urlsplit(value)
            return (value, "") if parsed.scheme in {"http", "https"} and parsed.netloc else ("", value)
        for candidate in path_keys:
            if candidate in source:
                return "", str(source[candidate] or "")
    return "", ""


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _job_id(payload: dict[str, Any]) -> str:
    value = str(payload.get("job_id") or payload.get("jobId") or "").strip()
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


def request_from_payload(payload: dict[str, Any], config: MuLaCoverConfig | None = None) -> MuLaCoverRequest:
    config = config or MuLaCoverConfig.from_env()
    lyrics = _value(payload, "lyrics", "text")
    tags = _value(payload, "tags", "style", "style_tags", "styleTags")
    audio_url, audio_path = _source(payload, "ref_audio", "refAudio", "reference_audio", "referenceAudio")
    melody_url, melody_path = _source(payload, "melody_midi", "melodyMidi")
    chord_url, chord_path = _source(payload, "chord_midi", "chordMidi")
    drum_url, drum_path = _source(payload, "drum_midi", "drumMidi")
    lyrics_url, _ = _source(payload, "lyrics_url", "lyricsUrl")
    tags_url, _ = _source(payload, "tags_url", "tagsUrl")
    if lyrics is None and not lyrics_url and not audio_url and not audio_path:
        raise ValueError("MuLaCover job requires lyrics or lyrics_url")
    if tags is None and not tags_url:
        raise ValueError("MuLaCover job requires tags or tags_url")
    seed = _value(payload, "seed")
    decode_seed = _value(payload, "decode_seed", "decodeSeed")
    return MuLaCoverRequest(
        lyrics=str(lyrics or ""), tags=str(tags or ""), ref_audio_url=audio_url, ref_audio_path=Path(audio_path) if audio_path else None,
        melody_midi_url=melody_url, melody_midi_path=Path(melody_path) if melody_path else None,
        chord_midi_url=chord_url, chord_midi_path=Path(chord_path) if chord_path else None,
        drum_midi_url=drum_url, drum_midi_path=Path(drum_path) if drum_path else None,
        lyrics_url=lyrics_url, tags_url=tags_url,
        bpm=(float(_value(payload, "bpm")) if _value(payload, "bpm") is not None else None),
        semitone_shift=int(_value(payload, "semitone_shift", "semitoneShift") or 0),
        octave_shift=int(_value(payload, "octave_shift", "octaveShift") or 0),
        duration_seconds=float(_value(payload, "duration_seconds", "durationSeconds") or config.default_duration_seconds),
        cfg_scale=float(_value(payload, "cfg_scale", "cfgScale") or 1.5),
        temperature=float(_value(payload, "temperature") or 1.0),
        top_k=int(_value(payload, "top_k", "topK") or 250),
        seed=int(seed) if seed is not None else None,
        decode_seed=int(decode_seed) if decode_seed is not None else None,
        save_symbolic_midi=_bool_value(_value(payload, "save_symbolic_midi", "saveSymbolicMidi"), True),
        output_format=str(_value(payload, "output_format", "outputFormat") or "wav").lower(),
        external_job_id=_job_id(payload),
        payment_intent_id=str(_value(payload, "payment_intent_id", "paymentIntentId") or "") or None,
    )


def _post_json(url: str, token: str, payload: Any, *, timeout: float = 120) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = UrlRequest(url, data=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body)), "X-Job-Callback-Token": token}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raise RuntimeError(f"callback HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"callback failed to {url}: {exc}") from exc


def _upload_artifact(url: str, token: str, artifact: dict[str, Any]) -> str:
    path = Path(str(artifact.get("path") or ""))
    if not path.is_file():
        raise RuntimeError(f"MuLaCover artifact is missing: {artifact.get('name')}")
    name = path.name
    body = path.read_bytes()
    role = str(artifact.get("role") or ("audio" if path.suffix in {".wav", ".flac"} else "metadata"))
    headers = {"Content-Type": mimetypes.guess_type(name)[0] or "application/octet-stream", "Content-Length": str(len(body)), "X-Job-Callback-Token": token, "X-Artifact-Role": role, "X-Artifact-File-Name": name}
    request = UrlRequest(url, data=body, headers=headers, method="POST")
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
    payload = {"schema_version": 1, "job_id": job_id, "runtime": "mulacover", "status": status, "phase": job.get("stage") or "processing", "progress": value, "completed_steps": round(value * 100), "total_steps": 100, "attempt": job.get("attempt", 0), "message": message or job.get("message"), "stage": job.get("stage"), "sequence": sequence, "updated_at": job.get("updated_at") or ""}
    try:
        await asyncio.to_thread(_post_json, url, token, payload, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "mulacover_progress_callback_failed", "jobId": job_id, "error": str(exc)}), flush=True)


async def _fail_callback(complete_url: str, token: str, code: str, message: str, artifact_ids: list[str]) -> None:
    fail_url = f"{complete_url.rsplit('/complete', 1)[0]}/fail"
    try:
        await asyncio.to_thread(_post_json, fail_url, token, {"errorCode": code[:120], "errorMessage": message[:4000], "artifactIds": artifact_ids}, timeout=10)
    except Exception as exc:
        print(json.dumps({"event": "mulacover_failure_callback_failed", "error": str(exc)}), flush=True)


async def _run_queue_job(payload: dict[str, Any], worker: MuLaCoverWorker, config: MuLaCoverConfig) -> dict[str, Any]:
    job_id = _job_id(payload)
    callback_url, complete_url, progress_url, token = _callback(payload)
    failure_sent = False
    try:
        await _progress(progress_url, token, job_id, {"stage": "accepted", "progress": 0, "attempt": 0}, 1, "MuLaCover worker accepted the job")
        job = await worker.submit(request_from_payload(payload, config))
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
                    message = f"[{failure.get('code', 'mulacover_worker_failed')}] stage={failure.get('stage', 'unknown')}: {failure.get('message', 'MuLaCover generation failed')}"
                    uploaded = [await asyncio.to_thread(_upload_artifact, callback_url, token, artifact) for artifact in current.get("artifacts") or []]
                    await _fail_callback(complete_url, token, failure.get("code", "mulacover_worker_failed"), message, uploaded)
                    failure_sent = True
                    await _progress(progress_url, token, job_id, current, sequence + 1, message, "failed")
                    raise RuntimeError(message)
                uploaded = []
                uploaded_meta = []
                for artifact in current.get("artifacts") or []:
                    artifact_id = await asyncio.to_thread(_upload_artifact, callback_url, token, artifact)
                    uploaded.append(artifact_id)
                    uploaded_meta.append({"id": artifact_id, "artifactId": artifact_id, "name": artifact.get("name"), "role": artifact.get("role"), "mediaType": artifact.get("media_type"), "sizeBytes": artifact.get("size_bytes"), "sha256": artifact.get("sha256")})
                await asyncio.to_thread(_post_json, complete_url, token, {"artifactIds": uploaded, "artifacts": uploaded_meta})
                await _progress(progress_url, token, job_id, current, sequence + 1, "MuLaCover worker completed the job", "succeeded")
                return {"schema_version": 1, "runtime": "mulacover", "status": "succeeded", "job_id": job_id, "artifact_ids": uploaded, "artifacts": uploaded_meta}
            await asyncio.sleep(1)
        raise TimeoutError(f"MuLaCover job exceeded the {config.job_timeout_seconds:.0f}s worker timeout")
    except Exception as exc:
        if not failure_sent:
            code = "mulacover_worker_timeout" if isinstance(exc, TimeoutError) else "mulacover_worker_failed"
            await _fail_callback(complete_url, token, code, f"[{code}] {exc}", [])
        raise


def install_callback_routes(app: FastAPI, worker: MuLaCoverWorker, config: MuLaCoverConfig) -> None:
    background_tasks: set[asyncio.Task[Any]] = set()
    app.state.mulacover_background_tasks = background_tasks

    def done(task: asyncio.Task[Any]) -> None:
        background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            print(json.dumps({"event": "mulacover_queue_job_cancelled", "jobId": task.get_name().removeprefix("mulacover-job-")}), flush=True)
        except Exception as exc:
            print(json.dumps({"event": "mulacover_queue_job_failed", "errorType": type(exc).__name__, "error": str(exc)}), flush=True)

    @app.post("/process")
    async def process(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            if payload.get("runtime") not in {None, "mulacover", "mulacover-worker", "music-cover"}:
                raise ValueError(f"unsupported runtime: {payload.get('runtime')}")
            job_id = _job_id(payload)
            _callback(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        task = asyncio.create_task(_run_queue_job(payload, worker, config), name=f"mulacover-job-{job_id}")
        background_tasks.add(task)
        task.add_done_callback(done)
        print(json.dumps({"event": "mulacover_queue_job_accepted", "jobId": job_id}), flush=True)
        return {"schema_version": 1, "runtime": "mulacover", "status": "accepted", "id": job_id, "job_id": job_id}

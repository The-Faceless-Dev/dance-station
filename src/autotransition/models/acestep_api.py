"""ACE-Step REST API client."""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

from autotransition.config import RuntimeConfig
from autotransition.models.base import RepaintResult
from autotransition.models.registry import ModelProfile
from autotransition.pipeline import SourceSelectionPlan


class AceStepApiError(RuntimeError):
    """Raised when ACE-Step API generation fails."""


@dataclass(frozen=True)
class AceStepApiClient:
    config: RuntimeConfig

    def health(self) -> bool:
        import httpx

        try:
            response = httpx.get(f"{self.config.api_base_url}/health", timeout=self.config.api_timeout_seconds)
            return response.status_code < 500
        except Exception:
            return False

    def repaint(self, plan: SourceSelectionPlan, profile: ModelProfile, save_dir: Path) -> RepaintResult:
        import httpx

        self._ensure_model(profile)
        payload: dict[str, Any] = {
            "task_type": "repaint",
            "repainting_start": plan.repainting_start_seconds,
            "repainting_end": plan.repainting_end_seconds,
            "prompt": plan.caption,
            "lyrics": "[Instrumental]",
            "vocal_language": "unknown",
            "model": profile.slug,
            "audio_duration": plan.context_seconds + plan.requested_continuation_seconds,
            "audio_format": plan.audio_format,
            "batch_size": 1,
            "inference_steps": profile.default_inference_steps,
            "thinking": False,
        }
        payload.update(_repaint_defaults_for_profile(profile))
        payload.update(plan.ace_step_settings)
        if plan.seed is not None:
            payload["use_random_seed"] = False
            payload["seed"] = plan.seed
        if plan.bpm_hint is not None:
            payload["bpm"] = int(plan.bpm_hint)
        if plan.key_hint:
            payload["key_scale"] = plan.key_hint

        with plan.scaffold_path.open("rb") as scaffold_file:
            return self._submit_and_wait(
                payload=payload,
                save_dir=save_dir,
                files={"src_audio": (plan.scaffold_path.name, scaffold_file, "audio/wav")},
            )

    def text2music(self, plan: SourceSelectionPlan, profile: ModelProfile, save_dir: Path) -> RepaintResult:
        self._ensure_model(profile, init_llm=True)
        payload: dict[str, Any] = {
            "task_type": "text2music",
            "prompt": plan.caption,
            "lyrics": "[Instrumental]",
            "vocal_language": "unknown",
            "model": profile.slug,
            "audio_duration": plan.requested_continuation_seconds,
            "audio_format": plan.audio_format,
            "batch_size": 1,
            "inference_steps": profile.default_inference_steps,
            "thinking": True,
        }
        payload.update(_text2music_defaults_for_profile(profile))
        payload.update(_filter_settings(plan.ace_step_settings, {"inference_steps", "guidance_scale", "shift"}))
        if plan.seed is not None:
            payload["use_random_seed"] = False
            payload["seed"] = plan.seed
        if plan.bpm_hint is not None:
            payload["bpm"] = int(plan.bpm_hint)
        if plan.key_hint:
            payload["key_scale"] = plan.key_hint

        return self._submit_and_wait(payload=payload, save_dir=save_dir)

    def _submit_and_wait(
        self,
        payload: dict[str, Any],
        save_dir: Path,
        files: dict[str, Any] | None = None,
    ) -> RepaintResult:
        release = _request(
            "post",
            f"{self.config.api_base_url}/release_task",
            "release_task",
            data=_stringify_form_fields(payload),
            files=files,
            timeout=self.config.generation_timeout_seconds,
        )
        _raise_api_status(release, "release_task")
        release_body = _response_json(release, "release_task")
        if release_body.get("error"):
            raise AceStepApiError(str(release_body["error"]))

        task_id = (release_body.get("data") or {}).get("task_id")
        if not task_id:
            raise AceStepApiError(f"ACE-Step API did not return a task_id: {release_body}")

        deadline = time.monotonic() + self.config.generation_timeout_seconds
        while time.monotonic() < deadline:
            query = _request(
                "post",
                f"{self.config.api_base_url}/query_result",
                "query_result",
                json={"task_id_list": [task_id]},
                timeout=self.config.generation_timeout_seconds,
            )
            _raise_api_status(query, "query_result")
            query_body = _response_json(query, "query_result")
            if query_body.get("error"):
                raise AceStepApiError(str(query_body["error"]))

            result_data = query_body.get("data")
            task_result = _extract_task_result(result_data, task_id)
            if task_result:
                status = task_result.get("status")
                if status == 1 or status == "succeeded":
                    return self._download_result(task_result, save_dir, task_id)
                if status == 2 or status == "failed":
                    raise AceStepApiError(str(task_result.get("error") or task_result.get("message") or "Generation failed"))

            time.sleep(self.config.poll_interval_seconds)

        raise AceStepApiError("ACE-Step generation timed out.")

    def _download_result(self, task_result: dict[str, Any], save_dir: Path, task_id: str) -> RepaintResult:
        import httpx

        audio_path = _extract_audio_path(task_result)
        if not audio_path:
            raise AceStepApiError(f"ACE-Step task succeeded but no audio path was returned: {task_result}")

        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{task_id}.wav"
        metadata_path = save_dir / f"{task_id}.json"

        response = _request(
            "get",
            f"{self.config.api_base_url}/v1/audio",
            "v1/audio",
            params={"path": audio_path},
            timeout=self.config.generation_timeout_seconds,
        )
        _raise_api_status(response, "v1/audio")
        output_path.write_bytes(response.content)
        metadata_path.write_text(json.dumps(task_result, indent=2), encoding="utf-8")

        return RepaintResult(output_path=output_path, metadata_path=metadata_path, model_name="ACE-Step API")

    def _ensure_model(self, profile: ModelProfile, init_llm: bool = False) -> None:
        import httpx
        from autotransition.runtime.checkpoints import repair_incomplete_checkpoint

        body: dict[str, Any] = {}
        try:
            models = _request(
                "get",
                f"{self.config.api_base_url}/v1/models",
                "v1/models",
                timeout=self.config.api_timeout_seconds,
            )
            _raise_api_status(models, "v1/models")
            body = _response_json(models, "v1/models")
        except AceStepApiError as exc:
            print(f"[Autotransition] ACE-Step model list unavailable; initializing directly. {exc}")

        model_data = body.get("data") or {}
        model_items = model_data.get("models", []) if isinstance(model_data, dict) else model_data
        available = {
            item.get("name")
            for item in model_items
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if profile.slug in available and not init_llm:
            return

        repair = repair_incomplete_checkpoint(profile, self.config)
        if repair.repaired:
            print(f"[Autotransition] {repair.message}")

        init = _request(
            "post",
            f"{self.config.api_base_url}/v1/init",
            "v1/init",
            json={"model": profile.slug, "init_llm": init_llm},
            timeout=self.config.generation_timeout_seconds,
        )
        _raise_api_status(init, "v1/init")
        init_body = _response_json(init, "v1/init")
        if init_body.get("error"):
            raise AceStepApiError(str(init_body["error"]))


def _extract_task_result(data: Any, task_id: str) -> dict[str, Any] | None:
    if isinstance(data, dict):
        if task_id in data and isinstance(data[task_id], dict):
            return data[task_id]
        if "results" in data:
            return _extract_task_result(data["results"], task_id)
        if data.get("task_id") == task_id:
            if isinstance(data.get("result"), str):
                parsed = _parse_result_string(data["result"])
                if parsed:
                    return parsed
            return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and (item.get("task_id") == task_id or item.get("id") == task_id):
                if isinstance(item.get("result"), str):
                    parsed = _parse_result_string(item["result"])
                    if parsed:
                        return parsed
                return item
    return None


def _extract_audio_path(task_result: dict[str, Any]) -> str | None:
    for key in ("file", "audio_path", "path", "url"):
        value = task_result.get(key)
        if isinstance(value, str):
            return _normalize_audio_path(value)

    audios = task_result.get("audios") or task_result.get("audio")
    if isinstance(audios, list) and audios:
        first = audios[0]
        if isinstance(first, dict):
            for key in ("path", "audio_path", "url"):
                value = first.get(key)
                if isinstance(value, str):
                    return _normalize_audio_path(value)
        if isinstance(first, str):
            return _normalize_audio_path(first)
    return None


def _parse_result_string(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    if isinstance(parsed, dict):
        return parsed
    return None


def _normalize_audio_path(value: str) -> str:
    parsed = urlparse(value)
    if parsed.path.endswith("/v1/audio"):
        query_path = parse_qs(parsed.query).get("path")
        if query_path:
            return query_path[0]
    return value


def _stringify_form_fields(payload: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            fields[key] = "true" if value else "false"
        else:
            fields[key] = str(value)
    return fields


def _filter_settings(settings: dict[str, object], allowed: set[str]) -> dict[str, object]:
    return {key: value for key, value in settings.items() if key in allowed}


def _repaint_defaults_for_profile(profile: ModelProfile) -> dict[str, Any]:
    is_turbo = "turbo" in profile.slug
    common = {
        "infer_method": "ode",
        "chunk_mask_mode": "explicit",
        "repaint_mode": "balanced",
        "repaint_strength": 0.5,
    }
    if is_turbo:
        return {
            **common,
            "guidance_scale": 1.0,
            "shift": 3.0,
            "repaint_latent_crossfade_frames": 16,
            "repaint_wav_crossfade_sec": 0.25,
        }
    return {
        **common,
        "guidance_scale": 7.0,
        "shift": 3.0,
        "repaint_latent_crossfade_frames": 24,
        "repaint_wav_crossfade_sec": 0.5,
    }


def _text2music_defaults_for_profile(profile: ModelProfile) -> dict[str, Any]:
    if "turbo" in profile.slug:
        return {
            "guidance_scale": 1.0,
            "shift": 3.0,
        }
    return {
        "guidance_scale": 7.0,
        "shift": 3.0,
    }


def _raise_api_status(response: Any, operation: str) -> None:
    if response.status_code < 400:
        return
    try:
        detail = response.json()
    except Exception:
        detail = _response_text_preview(response)
    request = getattr(response, "request", None)
    method = getattr(request, "method", "")
    url = getattr(request, "url", "")
    target = f" {method} {url}" if method and url else ""
    raise AceStepApiError(f"ACE-Step {operation}{target} failed with HTTP {response.status_code}: {detail}")


def _request(method: str, url: str, operation: str, **kwargs: Any) -> Any:
    import httpx

    try:
        return getattr(httpx, method)(url, **kwargs)
    except httpx.TimeoutException as exc:
        raise AceStepApiError(f"ACE-Step {operation} timed out while calling {method.upper()} {url}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise AceStepApiError(f"ACE-Step {operation} request failed while calling {method.upper()} {url}: {exc}") from exc


def _response_json(response: Any, operation: str) -> dict[str, Any]:
    try:
        body = response.json()
    except Exception as exc:
        detail = _response_text_preview(response)
        raise AceStepApiError(f"ACE-Step {operation} returned a non-JSON response: {detail[:500]}") from exc
    if not isinstance(body, dict):
        raise AceStepApiError(f"ACE-Step {operation} returned unexpected JSON: {body}")
    return body


def _response_text_preview(response: Any, limit: int = 500) -> str:
    text = getattr(response, "text", "")
    compact = " ".join(str(text).split())
    return compact[:limit]

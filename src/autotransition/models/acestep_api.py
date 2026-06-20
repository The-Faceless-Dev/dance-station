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


DEFAULT_LM_MODEL_PATH = "acestep-5Hz-lm-1.7B"
DEFAULT_TEXT2MUSIC_BPM = 120
DEFAULT_TEXT2MUSIC_KEY_SCALE = "C minor"


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

    def repaint_transition(self, plan: SourceSelectionPlan, profile: ModelProfile, save_dir: Path) -> RepaintResult:
        payload: dict[str, Any] = {
            "task_type": "repaint",
            "repainting_start": plan.repainting_start_seconds,
            "repainting_end": plan.repainting_end_seconds,
            "prompt": plan.caption,
            "lyrics": "[Instrumental]",
            "vocal_language": "unknown",
            "audio_format": plan.audio_format,
            "batch_size": 1,
            "inference_steps": profile.default_inference_steps,
            "thinking": False,
        }
        payload.update(_repaint_defaults_for_profile(profile))
        payload.update(_filter_settings(plan.ace_step_settings, _REPAINT_SETTING_ALLOWLIST))
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
        lm_model_path = _configured_lm_model_path(plan, profile)
        payload: dict[str, Any] = {
            "task_type": "text2music",
            "prompt": plan.caption,
            "lyrics": "[Instrumental]",
            "audio_duration": plan.requested_continuation_seconds,
            "audio_format": _raw_text2music_audio_format(plan.audio_format),
            "batch_size": 1,
            "inference_steps": profile.default_inference_steps,
            "thinking": True,
            "use_format": False,
            "time_signature": "4",
            "bpm": DEFAULT_TEXT2MUSIC_BPM,
            "key_scale": DEFAULT_TEXT2MUSIC_KEY_SCALE,
            "lm_model_path": lm_model_path,
            "lm_temperature": 0.85,
            "lm_cfg_scale": 2.5,
            "lm_top_p": 0.9,
            "lm_negative_prompt": "NO USER INPUT",
        }
        payload.update(_text2music_defaults_for_profile(profile))
        payload.update(_filter_settings(plan.ace_step_settings, _TEXT2MUSIC_SETTING_ALLOWLIST))
        if plan.seed is not None:
            payload["use_random_seed"] = False
            payload["seed"] = plan.seed
        if plan.bpm_hint is not None:
            payload["bpm"] = int(plan.bpm_hint)
        if plan.key_hint:
            payload["key_scale"] = plan.key_hint

        return self._submit_and_wait(payload=payload, save_dir=save_dir, use_json=True)

    def _submit_and_wait(
        self,
        payload: dict[str, Any],
        save_dir: Path,
        files: dict[str, Any] | None = None,
        use_json: bool = False,
    ) -> RepaintResult:
        save_dir.mkdir(parents=True, exist_ok=True)
        _write_debug_json(save_dir / "ace-request.json", payload)
        request_kwargs: dict[str, Any] = {
            "timeout": self.config.generation_timeout_seconds,
        }
        if files is not None:
            request_kwargs["files"] = files
        if use_json and not files:
            request_kwargs["json"] = payload
        else:
            request_kwargs["data"] = _stringify_form_fields(payload)

        release = _request(
            "post",
            f"{self.config.api_base_url}/release_task",
            "release_task",
            **request_kwargs,
        )
        _raise_api_status(release, "release_task")
        release_body = _response_json(release, "release_task")
        _write_debug_json(save_dir / "ace-release-response.json", release_body)
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
            _write_debug_json(save_dir / "ace-query-response-latest.json", query_body)
            if query_body.get("error"):
                raise AceStepApiError(str(query_body["error"]))

            result_data = query_body.get("data")
            task_result = _extract_task_result(result_data, task_id)
            if task_result:
                status = task_result.get("status")
                if status == 1 or status == "succeeded":
                    _write_debug_json(save_dir / "ace-query-response-final.json", query_body)
                    return self._download_result(task_result, save_dir, task_id)
                if status == 2 or status == "failed":
                    _write_debug_json(save_dir / "ace-query-response-final.json", query_body)
                    raise AceStepApiError(str(task_result.get("error") or task_result.get("message") or "Generation failed"))

            time.sleep(self.config.poll_interval_seconds)

        raise AceStepApiError("ACE-Step generation timed out.")

    def _download_result(self, task_result: dict[str, Any], save_dir: Path, task_id: str) -> RepaintResult:
        import httpx

        audio_path = _extract_audio_path(task_result)
        if not audio_path:
            raise AceStepApiError(f"ACE-Step task succeeded but no audio path was returned: {task_result}")

        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{task_id}{_audio_extension(audio_path, task_result)}"
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
        metadata = {
            **task_result,
            "downloaded_audio_path": str(output_path),
            "ace_audio_path": audio_path,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return RepaintResult(output_path=output_path, metadata_path=metadata_path, model_name="ACE-Step API")

    def _ensure_model(
        self,
        profile: ModelProfile,
        init_llm: bool = False,
        lm_model_path: str | None = None,
    ) -> None:
        import httpx
        from autotransition.runtime.checkpoints import repair_incomplete_checkpoint

        body: dict[str, Any] = {}
        try:
            models = _request(
                "get",
                f"{self.config.api_base_url}/v1/model_inventory",
                "v1/model_inventory",
                timeout=self.config.api_timeout_seconds,
            )
            _raise_api_status(models, "v1/model_inventory")
            body = _response_json(models, "v1/model_inventory")
        except AceStepApiError as exc:
            print(f"[Autotransition] ACE-Step model inventory unavailable; initializing directly. {exc}")

        inventory = _model_inventory(body)
        if inventory.is_model_loaded(profile.slug):
            if not init_llm or inventory.is_lm_loaded(lm_model_path):
                return

        repair = repair_incomplete_checkpoint(profile, self.config)
        if repair.repaired:
            print(f"[Autotransition] {repair.message}")

        init_payload: dict[str, Any] = {"model": profile.slug, "init_llm": init_llm}
        if init_llm and lm_model_path:
            init_payload["lm_model_path"] = lm_model_path

        init = _request(
            "post",
            f"{self.config.api_base_url}/v1/init",
            "v1/init",
            json=init_payload,
            timeout=self.config.generation_timeout_seconds,
        )
        _raise_api_status(init, "v1/init")
        init_body = _response_json(init, "v1/init")
        if init_body.get("error"):
            raise AceStepApiError(str(init_body["error"]))
        _validate_init_response(init_body, profile, init_llm)


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


def _raw_text2music_audio_format(app_output_format: str) -> str:
    if app_output_format == "wav32":
        return "wav32"
    return "flac"


def _audio_extension(audio_path: str, task_result: dict[str, Any]) -> str:
    for value in (audio_path, str(task_result.get("file", "")), str(task_result.get("audio_path", ""))):
        suffix = Path(urlparse(value).path).suffix.lower()
        if suffix in {".flac", ".wav", ".mp3", ".opus", ".aac", ".m4a", ".ogg"}:
            return suffix
    audio_format = str(task_result.get("audio_format") or task_result.get("format") or "").lower()
    if audio_format == "wav32":
        return ".wav"
    if audio_format in {"flac", "wav", "mp3", "opus", "aac", "ogg"}:
        return f".{audio_format}"
    return ".flac"


def _write_debug_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_init_response(init_body: dict[str, Any], profile: ModelProfile, init_llm: bool) -> None:
    data = init_body.get("data")
    if not isinstance(data, dict):
        return
    loaded_model = data.get("loaded_model")
    if isinstance(loaded_model, str) and loaded_model and loaded_model != profile.slug:
        raise AceStepApiError(
            f"ACE-Step initialized '{loaded_model}' instead of requested model '{profile.slug}': {init_body}"
        )
    if init_llm and "llm_initialized" in data and data.get("llm_initialized") is not True:
        raise AceStepApiError(f"ACE-Step LM initialization did not complete: {init_body}")


@dataclass(frozen=True)
class _ModelInventory:
    loaded_models: frozenset[str]
    llm_initialized: bool
    loaded_lm_model: str | None

    def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self.loaded_models

    def is_lm_loaded(self, lm_model_path: str | None) -> bool:
        if not self.llm_initialized:
            return False
        if not lm_model_path:
            return True
        return _model_name(lm_model_path) == _model_name(self.loaded_lm_model)


def _model_inventory(body: dict[str, Any]) -> _ModelInventory:
    data = body.get("data") or {}
    model_items = data.get("models", []) if isinstance(data, dict) else data
    loaded_models = {
        item.get("name")
        for item in model_items
        if (
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item.get("is_loaded") is True
        )
    }
    if isinstance(data, dict):
        return _ModelInventory(
            loaded_models=frozenset(loaded_models),
            llm_initialized=data.get("llm_initialized") is True,
            loaded_lm_model=data.get("loaded_lm_model") if isinstance(data.get("loaded_lm_model"), str) else None,
        )
    return _ModelInventory(loaded_models=frozenset(loaded_models), llm_initialized=False, loaded_lm_model=None)


def _model_name(path_or_name: str | None) -> str | None:
    if path_or_name is None:
        return None
    text = str(path_or_name).strip().replace("\\", "/")
    if not text:
        return None
    return text.rstrip("/").split("/")[-1]


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


_TEXT2MUSIC_SETTING_ALLOWLIST = {
    "inference_steps",
    "guidance_scale",
    "shift",
    "lm_model_path",
    "lm_temperature",
    "lm_cfg_scale",
    "lm_top_p",
    "lm_negative_prompt",
}

_REPAINT_SETTING_ALLOWLIST = {
    "inference_steps",
    "guidance_scale",
    "shift",
    "chunk_mask_mode",
    "repaint_mode",
    "repaint_strength",
    "repaint_latent_crossfade_frames",
    "repaint_wav_crossfade_sec",
}


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


def _lm_model_path_for_profile(profile: ModelProfile) -> str:
    return DEFAULT_LM_MODEL_PATH


def _configured_lm_model_path(plan: SourceSelectionPlan, profile: ModelProfile) -> str:
    configured = plan.ace_step_settings.get("lm_model_path")
    if configured is None:
        return _lm_model_path_for_profile(profile)
    configured_text = str(configured).strip()
    return configured_text or _lm_model_path_for_profile(profile)


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

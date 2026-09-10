from __future__ import annotations

import hashlib
import json
import math
import time
import wave
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx

from .audio import AudioBuffer
from .config import MossMusicConfig
from .contracts import MossMusicRequest
from .prompting import build_analysis_prompts


class MossBackendError(RuntimeError):
    pass


class MossAnalysisError(MossBackendError):
    def __init__(self, message: str, *, responses: dict[str, Any], raw_texts: dict[str, str], metadata: dict[str, Any]):
        super().__init__(message)
        self.responses = responses
        self.raw_texts = raw_texts
        self.metadata = metadata


@dataclass(frozen=True)
class MossRuntimeResult:
    response: Any
    raw_text: str
    metadata: dict[str, Any]
    responses: dict[str, Any] = field(default_factory=dict)
    raw_texts: dict[str, str] = field(default_factory=dict)


class MossRuntime(Protocol):
    def preflight(self) -> dict[str, Any]: ...

    def analyze(
        self,
        request: MossMusicRequest,
        audio: AudioBuffer,
        progress: Callable[[float, str], None],
    ) -> MossRuntimeResult: ...


class SGLangMossClient:
    def __init__(self, config: MossMusicConfig):
        self.config = config
        self._tokenizer: Any | None = None

    def context_length(self) -> int:
        if self.config.model_context_length is not None:
            return self.config.model_context_length
        config_path = Path(self.config.model_root) / "config.json"
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MossBackendError(f"could not read MOSS model context from {config_path}: {exc}") from exc
        language_config = model_config.get("language_config") or {}
        value = language_config.get("max_position_embeddings") or model_config.get("max_position_embeddings")
        if not isinstance(value, int) or value < 1:
            raise MossBackendError(f"MOSS model config has no usable max_position_embeddings: {config_path}")
        return value

    def _prompt_token_count(self, prompt: str) -> tuple[int, str]:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_root,
                    use_fast=False,
                    local_files_only=True,
                )
            except Exception:
                self._tokenizer = False
        if self._tokenizer is not False:
            try:
                return len(self._tokenizer.encode(prompt, add_special_tokens=False)), "tokenizer"
            except Exception:
                pass
        return max(1, math.ceil(len(prompt) / 4)), "character_estimate"

    @staticmethod
    def _audio_token_count(audio_path: Path) -> int:
        with wave.open(str(audio_path), "rb") as handle:
            duration = handle.getnframes() / max(1, handle.getframerate())
        return max(1, math.ceil(duration * 12.5))

    def resolve_output_budget(self, *, prompt: str, audio_path: Path) -> dict[str, Any]:
        context_length = self.context_length()
        prompt_tokens, prompt_method = self._prompt_token_count(prompt)
        audio_tokens = self._audio_token_count(audio_path)
        available = context_length - prompt_tokens - audio_tokens
        if available < 1:
            raise MossBackendError(
                f"MOSS input exceeds model context: context={context_length}, "
                f"promptTokens={prompt_tokens}, audioTokens={audio_tokens}"
            )
        return {
            "contextLength": context_length,
            "promptTokens": prompt_tokens,
            "promptTokenMethod": prompt_method,
            "audioTokens": audio_tokens,
            "availableOutputTokens": available,
        }

    def health(self) -> dict[str, Any]:
        url = f"{self.config.sglang_url}{self.config.sglang_health_path}"
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
        except httpx.HTTPError as exc:
            return {"ready": False, "url": url, "errorType": type(exc).__name__, "error": str(exc)}
        return {
            "ready": 200 <= response.status_code < 300,
            "url": url,
            "statusCode": response.status_code,
            "body": response.text[-1000:],
        }

    def generate(self, *, prompt: str, audio_path: Path, max_new_tokens: int, temperature: float) -> Any:
        url = f"{self.config.sglang_url}{self.config.sglang_endpoint}"
        sampling_params: dict[str, Any] = {"temperature": temperature}
        sampling_params["max_new_tokens"] = max_new_tokens
        payload = {
            "text": prompt,
            "audio_data": str(audio_path),
            "sampling_params": sampling_params,
        }
        try:
            with httpx.Client(timeout=self.config.request_timeout_seconds) as client:
                response = client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise MossBackendError(f"SGLang request failed: {exc}") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise MossBackendError(f"SGLang returned HTTP {response.status_code}: {response.text[-2000:]}")
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text


class SGLangMossRuntime:
    def __init__(self, config: MossMusicConfig, client: SGLangMossClient | None = None):
        self.config = config
        self.client = client or SGLangMossClient(config)

    def preflight(self) -> dict[str, Any]:
        report = self.config.preflight()
        gpu = self._gpu_report()
        report["gpu"] = gpu
        if self.config.backend != "sglang":
            report.update({"ready": True, "backendHealth": {"ready": True}})
            return report
        model_ready = Path(self.config.model_root).exists()
        backend_health = self.client.health() if self.config.check_backend_on_preflight else {"ready": None, "skipped": True}
        gpu_ready = bool(gpu.get("available")) if self.config.gpu_required else True
        report.update({"modelReady": model_ready, "backendHealth": backend_health})
        report["ready"] = bool(model_ready and gpu_ready and (backend_health.get("ready") or not self.config.check_backend_on_preflight))
        return report

    def _gpu_report(self) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:
            return {"available": False, "errorType": type(exc).__name__, "error": str(exc)}
        if not torch.cuda.is_available():
            return {"available": False, "cudaAvailable": False}
        try:
            index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            return {
                "available": True,
                "cudaAvailable": True,
                "deviceIndex": index,
                "name": torch.cuda.get_device_name(index),
                "totalVramBytes": int(total_bytes),
                "freeVramBytes": int(free_bytes),
            }
        except Exception as exc:
            return {"available": True, "cudaAvailable": True, "errorType": type(exc).__name__, "error": str(exc)}

    def analyze(
        self,
        request: MossMusicRequest,
        audio: AudioBuffer,
        progress: Callable[[float, str], None],
    ) -> MossRuntimeResult:
        passes = build_analysis_prompts(request)
        responses: dict[str, Any] = {}
        raw_texts: dict[str, str] = {}
        pass_metadata: list[dict[str, Any]] = []
        for index, analysis_pass in enumerate(passes):
            start = index / len(passes)
            progress(start, f"Preparing MOSS-Music {analysis_pass.name} pass ({index + 1}/{len(passes)})")
            budget = self.client.resolve_output_budget(prompt=analysis_pass.instruction, audio_path=audio.path)
            prompt_hash = hashlib.sha256(analysis_pass.instruction.encode("utf-8")).hexdigest()
            started = time.monotonic()
            try:
                response = self.client.generate(
                    prompt=analysis_pass.instruction,
                    audio_path=audio.path,
                    max_new_tokens=int(budget["availableOutputTokens"]),
                    temperature=request.temperature,
                )
            except Exception as exc:
                raise MossAnalysisError(
                    f"MOSS-Music {analysis_pass.name} pass failed: {exc}",
                    responses=responses,
                    raw_texts=raw_texts,
                    metadata={"backend": "sglang", "model": self.config.model_name, "passes": pass_metadata},
                ) from exc
            responses[analysis_pass.name] = response
            if isinstance(response, str):
                raw_text = response
            elif isinstance(response, dict) and isinstance(response.get("text"), str):
                raw_text = response["text"]
            else:
                raw_text = json.dumps(response, sort_keys=True, default=str)
            raw_texts[analysis_pass.name] = raw_text
            pass_metadata.append(
                {
                    "name": analysis_pass.name,
                    "requiredKeys": list(analysis_pass.required_keys),
                    "promptSha256": prompt_hash,
                    "promptCharacters": len(analysis_pass.instruction),
                    "elapsedSeconds": round(time.monotonic() - started, 3),
                    "outputCharacters": len(raw_text),
                    "outputBudget": budget,
                }
            )
            progress((index + 1) / len(passes), f"MOSS-Music {analysis_pass.name} pass received")
        metadata = {
            "backend": "sglang",
            "model": self.config.model_name,
            "temperature": request.temperature,
            "passes": pass_metadata,
        }
        return MossRuntimeResult(
            response={"passes": responses},
            raw_text="\n\n".join(f"[{name}]\n{text}" for name, text in raw_texts.items()),
            metadata=metadata,
            responses=responses,
            raw_texts=raw_texts,
        )


class MockMossRuntime:
    """Deterministic backend for contract and HTTP smoke tests only."""

    def preflight(self) -> dict[str, Any]:
        return {"ready": True, "backend": "mock", "model": "fixture"}

    def analyze(
        self,
        request: MossMusicRequest,
        audio: AudioBuffer,
        progress: Callable[[float, str], None],
    ) -> MossRuntimeResult:
        progress(0.5, "Generating fixture MOSS-Music response")
        response = {
            "summary": "Deterministic MOSS-Music fixture response",
            "tempo_bpm": None,
            "time_signature": None,
            "key": None,
            "sections": [],
            "beats": [],
            "chords": [],
            "lyrics": [],
            "instruments": [],
            "voices": [],
            "visual_cues": [],
            "events": [],
            "warnings": ["mock backend; no model inference was performed"],
        }
        progress(1.0, "Fixture MOSS-Music response ready")
        return MossRuntimeResult(response=response, raw_text=json.dumps(response), metadata={"backend": "mock"})


def create_runtime(config: MossMusicConfig) -> MossRuntime:
    return MockMossRuntime() if config.backend == "mock" else SGLangMossRuntime(config)

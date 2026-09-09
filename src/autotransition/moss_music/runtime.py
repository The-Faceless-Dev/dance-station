from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx

from .audio import AudioBuffer
from .config import MossMusicConfig
from .contracts import MossMusicRequest
from .prompting import build_analysis_prompt


class MossBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class MossRuntimeResult:
    response: Any
    raw_text: str
    metadata: dict[str, Any]


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
        payload = {
            "text": prompt,
            "audio_data": str(audio_path),
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
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
        prompt = build_analysis_prompt(request)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        progress(0.05, "Preparing MOSS-Music structured analysis request")
        response = self.client.generate(
            prompt=prompt,
            audio_path=audio.path,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        if isinstance(response, str):
            raw_text = response
        elif isinstance(response, dict) and isinstance(response.get("text"), str):
            raw_text = response["text"]
        else:
            raw_text = json.dumps(response, sort_keys=True, default=str)
        progress(1.0, "MOSS-Music response received")
        return MossRuntimeResult(
            response=response,
            raw_text=raw_text,
            metadata={
                "backend": "sglang",
                "model": self.config.model_name,
                "promptSha256": prompt_hash,
                "maxNewTokens": request.max_new_tokens,
                "temperature": request.temperature,
            },
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

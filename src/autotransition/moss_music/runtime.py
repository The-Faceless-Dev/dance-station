from __future__ import annotations

import hashlib
import json
import math
import re
import time
import wave
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx

from .audio import AudioBuffer, write_audio_segment
from .config import MossMusicConfig
from .contracts import MossMusicRequest
from .parser import MossResponseError, merge_moss_segments, parse_moss_response
from .prompting import add_segment_context, build_analysis_prompts


class MossBackendError(RuntimeError):
    pass


class MossAnalysisError(MossBackendError):
    def __init__(self, message: str, *, responses: dict[str, Any], raw_texts: dict[str, str], metadata: dict[str, Any]):
        super().__init__(message)
        self.responses = responses
        self.raw_texts = raw_texts
        self.metadata = metadata


class MossSegmentError(MossResponseError):
    """A semantic segment failed with its response preserved for diagnostics."""

    def __init__(self, message: str, *, response: Any, raw_text: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.response = response
        self.raw_text = raw_text
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

    @staticmethod
    def recover_output_budget(error: str, budget: dict[str, Any]) -> dict[str, Any] | None:
        """Use SGLang's measured input count when its serializer rejects a budget."""

        match = re.search(
            r"(\d+) tokens from the input messages and (\d+) tokens for the completion",
            error,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        input_tokens = int(match.group(1))
        context_length = int(budget["contextLength"])
        available = context_length - input_tokens - 1
        if available < 1:
            return None
        return {
            **budget,
            "estimatedAvailableOutputTokens": budget["availableOutputTokens"],
            "backendInputTokens": input_tokens,
            "availableOutputTokens": available,
            "budgetRecoveredFromBackendContextError": True,
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

    @staticmethod
    def _raw_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and isinstance(response.get("text"), str):
            return response["text"]
        return json.dumps(response, sort_keys=True, default=str)

    @staticmethod
    def _is_truncated(response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        metadata = response.get("meta_info") or response.get("metaInfo") or {}
        candidates = [metadata, response]
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            candidates.append(choices[0])
        for container in candidates:
            if not isinstance(container, dict):
                continue
            reason = container.get("finish_reason") or container.get("finishReason")
            if isinstance(reason, dict):
                reason = reason.get("type") or reason.get("reason")
            if str(reason or "").strip().lower() in {"length", "max_tokens", "max_new_tokens"}:
                return True
        return False

    def _generate_with_budget(self, *, prompt: str, audio_path: Path, temperature: float) -> tuple[Any, dict[str, Any]]:
        budget = self.client.resolve_output_budget(prompt=prompt, audio_path=audio_path)
        try:
            response = self.client.generate(
                prompt=prompt,
                audio_path=audio_path,
                max_new_tokens=int(budget["availableOutputTokens"]),
                temperature=temperature,
            )
        except Exception as exc:
            recovered_budget = self.client.recover_output_budget(str(exc), budget)
            if recovered_budget is None:
                raise
            budget = recovered_budget
            response = self.client.generate(
                prompt=prompt,
                audio_path=audio_path,
                max_new_tokens=int(budget["availableOutputTokens"]),
                temperature=temperature,
            )
        return response, budget

    @staticmethod
    def _restrict_segment(parsed: dict[str, Any], duration_seconds: float) -> dict[str, Any]:
        """Keep model events inside the audio segment that produced them."""

        bounded = dict(parsed)
        collections = ("sections", "beats", "chords", "lyrics", "instruments", "voices", "visual_cues", "events")
        dropped = 0
        clipped = 0
        for collection in collections:
            values = parsed.get(collection)
            if not isinstance(values, list):
                continue
            kept: list[dict[str, Any]] = []
            for value in values:
                if not isinstance(value, dict):
                    continue
                start = float(value.get("start_seconds", value.get("time_seconds", 0)))
                end = float(value.get("end_seconds", start))
                if start > duration_seconds or end < 0:
                    dropped += 1
                    continue
                item = dict(value)
                bounded_start = max(0.0, start)
                bounded_end = min(duration_seconds, max(bounded_start, end))
                if bounded_start != start or bounded_end != end:
                    clipped += 1
                item["start_seconds"] = round(bounded_start, 6)
                item["end_seconds"] = round(bounded_end, 6)
                if "time_seconds" in item:
                    item["time_seconds"] = round(bounded_start, 6)
                kept.append(item)
            bounded[collection] = kept
        warnings = list(bounded.get("warnings") or [])
        if dropped:
            warnings.append(f"Dropped {dropped} model event(s) outside the supplied segment")
        if clipped:
            warnings.append(f"Clipped {clipped} model event endpoint(s) to the supplied segment")
        bounded["warnings"] = warnings
        bounded["event_count"] = len(bounded.get("events") or [])
        return bounded

    def _generate_segmented_pass(
        self,
        *,
        request: MossMusicRequest,
        audio: AudioBuffer,
        analysis_pass: Any,
        start_seconds: float,
        end_seconds: float,
        progress: Callable[[float, str], None],
        pass_start: float,
        pass_span: float,
        segment_index: int,
        segment_count_hint: int,
    ) -> tuple[list[tuple[float, dict[str, Any]]], list[str], list[dict[str, Any]]]:
        duration = end_seconds - start_seconds
        segment_path = audio.path.parent / "semantic-segments" / f"{analysis_pass.name}-{start_seconds:010.3f}.wav"
        write_audio_segment(audio, segment_path, start_seconds, end_seconds)
        instruction = analysis_pass.instruction + add_segment_context(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            full_duration_seconds=audio.duration_seconds,
        )
        response, budget = self._generate_with_budget(
            prompt=instruction,
            audio_path=segment_path,
            temperature=request.temperature,
        )
        raw_response_text = self._raw_text(response)
        if self._is_truncated(response):
            if duration <= self.config.min_semantic_window_seconds:
                raise MossSegmentError(
                    f"MOSS {analysis_pass.name} segment remained truncated at "
                    f"{duration:.3f}s, the configured minimum semantic window",
                    response=response,
                    raw_text=raw_response_text,
                    metadata={
                        "startSeconds": round(start_seconds, 6),
                        "endSeconds": round(end_seconds, 6),
                        "outputCharacters": len(raw_response_text),
                        "outputBudget": budget,
                        "truncated": True,
                    },
                )
            midpoint = start_seconds + duration / 2
            first, first_raw, first_meta = self._generate_segmented_pass(
                request=request,
                audio=audio,
                analysis_pass=analysis_pass,
                start_seconds=start_seconds,
                end_seconds=midpoint,
                progress=progress,
                pass_start=pass_start,
                pass_span=pass_span,
                segment_index=segment_index,
                segment_count_hint=segment_count_hint + 1,
            )
            second, second_raw, second_meta = self._generate_segmented_pass(
                request=request,
                audio=audio,
                analysis_pass=analysis_pass,
                start_seconds=midpoint,
                end_seconds=end_seconds,
                progress=progress,
                pass_start=pass_start,
                pass_span=pass_span,
                segment_index=segment_index,
                segment_count_hint=segment_count_hint + 1,
            )
            return first + second, first_raw + second_raw, first_meta + second_meta
        retry_metadata: dict[str, Any] | None = None
        try:
            parsed, raw_text = parse_moss_response(response, required_keys=set(analysis_pass.required_keys))
            parsed = self._restrict_segment(parsed, duration)
        except MossResponseError as initial_error:
            retry_instruction = (
                f"{instruction}\n\n"
                "The previous response was malformed. Retry the same analysis now. "
                "Output strict JSON only: use double quotes around every key and string, "
                "do not quote numeric values, close every array and object, and do not add "
                "any explanation before or after the JSON object."
            )
            retry_response, retry_budget = self._generate_with_budget(
                prompt=retry_instruction,
                audio_path=segment_path,
                temperature=request.temperature,
            )
            retry_raw_text = self._raw_text(retry_response)
            retry_metadata = {
                "initialParseError": str(initial_error),
                "initialOutputCharacters": len(raw_response_text),
                "initialOutputBudget": budget,
                "retryOutputCharacters": len(retry_raw_text),
                "retryOutputBudget": retry_budget,
            }
            if self._is_truncated(retry_response):
                raise MossSegmentError(
                    f"MOSS {analysis_pass.name} segment retry remained truncated",
                    response={"initial": response, "retry": retry_response},
                    raw_text=f"[initial response]\n{raw_response_text}\n[retry response]\n{retry_raw_text}",
                    metadata={
                        "startSeconds": round(start_seconds, 6),
                        "endSeconds": round(end_seconds, 6),
                        "outputCharacters": len(raw_response_text) + len(retry_raw_text),
                        "outputBudget": retry_budget,
                        "retry": retry_metadata,
                        "truncated": True,
                    },
                )
            try:
                parsed, _ = parse_moss_response(retry_response, required_keys=set(analysis_pass.required_keys))
                parsed = self._restrict_segment(parsed, duration)
            except MossResponseError as retry_error:
                raise MossSegmentError(
                    f"MOSS {analysis_pass.name} segment response could not be parsed after retry: {retry_error}",
                    response={"initial": response, "retry": retry_response},
                    raw_text=f"[initial response]\n{raw_response_text}\n[retry response]\n{retry_raw_text}",
                    metadata={
                        "startSeconds": round(start_seconds, 6),
                        "endSeconds": round(end_seconds, 6),
                        "outputCharacters": len(raw_response_text) + len(retry_raw_text),
                        "outputBudget": retry_budget,
                        "retry": retry_metadata,
                        "parseErrorType": type(retry_error).__name__,
                        "parseError": str(retry_error),
                    },
                ) from retry_error
            raw_text = f"[initial response]\n{raw_response_text}\n[accepted retry response]\n{retry_raw_text}"
        progress(
            min(pass_start + pass_span * 0.95, pass_start + pass_span),
            f"MOSS-Music {analysis_pass.name} segment {segment_index + 1}/{max(1, segment_count_hint)} received",
        )
        return (
            [(start_seconds, parsed)],
            [raw_text],
            [
                {
                    "startSeconds": round(start_seconds, 6),
                    "endSeconds": round(end_seconds, 6),
                    "outputCharacters": len(raw_text),
                    "outputBudget": retry_metadata.get("retryOutputBudget", budget) if retry_metadata else budget,
                    **({"retry": retry_metadata} if retry_metadata else {}),
                }
            ],
        )

    def _run_segmented_pass(
        self,
        *,
        request: MossMusicRequest,
        audio: AudioBuffer,
        analysis_pass: Any,
        progress: Callable[[float, str], None],
        pass_start: float,
        pass_span: float,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        window = self.config.semantic_window_seconds
        ranges = [
            (start, min(audio.duration_seconds, start + window))
            for start in (index * window for index in range(math.ceil(audio.duration_seconds / window)))
        ]
        results: list[tuple[float, dict[str, Any]]] = []
        raw_texts: list[str] = []
        metadata: list[dict[str, Any]] = []
        for index, (start_seconds, end_seconds) in enumerate(ranges):
            chunk_results, chunk_raw, chunk_metadata = self._generate_segmented_pass(
                request=request,
                audio=audio,
                analysis_pass=analysis_pass,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                progress=progress,
                pass_start=pass_start + pass_span * (index / len(ranges)),
                pass_span=pass_span / len(ranges),
                segment_index=index,
                segment_count_hint=len(ranges),
            )
            results.extend(chunk_results)
            raw_texts.extend(chunk_raw)
            metadata.extend(chunk_metadata)
        merged = merge_moss_segments(results)
        return merged, "\n\n".join(
            f"[segment {meta['startSeconds']:.3f}-{meta['endSeconds']:.3f}]\n{text}"
            for meta, text in zip(metadata, raw_texts, strict=True)
        ), metadata

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
            started = time.monotonic()
            prompt_hash = hashlib.sha256(analysis_pass.instruction.encode("utf-8")).hexdigest()
            force_segmented = audio.duration_seconds > self.config.semantic_window_seconds
            segment_metadata: list[dict[str, Any]] = []
            try:
                if force_segmented:
                    response, raw_text, segment_metadata = self._run_segmented_pass(
                        request=request,
                        audio=audio,
                        analysis_pass=analysis_pass,
                        progress=progress,
                        pass_start=start,
                        pass_span=1 / len(passes),
                    )
                    budget = {"mode": "segmented", "segmentCount": len(segment_metadata), "windowSeconds": self.config.semantic_window_seconds}
                else:
                    response, budget = self._generate_with_budget(
                        prompt=analysis_pass.instruction,
                        audio_path=audio.path,
                        temperature=request.temperature,
                    )
                    if self._is_truncated(response):
                        response, raw_text, segment_metadata = self._run_segmented_pass(
                            request=request,
                            audio=audio,
                            analysis_pass=analysis_pass,
                            progress=progress,
                            pass_start=start,
                            pass_span=1 / len(passes),
                        )
                        budget = {
                            "mode": "segmented_after_context_limit",
                            "segmentCount": len(segment_metadata),
                            "windowSeconds": self.config.semantic_window_seconds,
                        }
                    else:
                        raw_text = self._raw_text(response)
            except Exception as exc:
                failure_pass_metadata: dict[str, Any] = {"name": analysis_pass.name, "error": str(exc)}
                if isinstance(exc, MossSegmentError):
                    failure_key = f"{analysis_pass.name}.segment.{exc.metadata.get('startSeconds', 0):.3f}"
                    responses[failure_key] = exc.response
                    raw_texts[failure_key] = exc.raw_text
                    failure_pass_metadata["segmentFailure"] = exc.metadata
                pass_metadata.append(failure_pass_metadata)
                raise MossAnalysisError(
                    f"MOSS-Music {analysis_pass.name} pass failed: {exc}",
                    responses=responses,
                    raw_texts=raw_texts,
                    metadata={"backend": "sglang", "model": self.config.model_name, "passes": pass_metadata},
                ) from exc
            responses[analysis_pass.name] = response
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
                    "segments": segment_metadata,
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

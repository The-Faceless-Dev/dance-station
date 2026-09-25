from __future__ import annotations

import gc
import json
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import MuLaCoverConfig
from .contracts import MuLaCoverRequest


class MuLaCoverRuntimeError(RuntimeError):
    pass


class MuLaCoverPreflightError(MuLaCoverRuntimeError):
    code = "mulacover_preflight_failed"


@dataclass(frozen=True)
class MuLaCoverRuntimeResult:
    output_path: Path
    metadata: dict[str, Any]
    input_manifest: dict[str, Any]


class MuLaCoverRuntime:
    """Serialized, lazy-loaded MuLaCover execution on one CUDA device."""

    def __init__(self, config: MuLaCoverConfig):
        self.config = config
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def preflight(self) -> dict[str, Any]:
        return self.config.preflight()

    def _prepare_imports(self) -> None:
        try:
            import mulacover  # noqa: F401
        except ImportError as exc:
            raise MuLaCoverRuntimeError("the pinned mulacover package is not installed") from exc

    def _pipeline_for_job(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        self._prepare_imports()
        import torch
        from mulacover import MuLaCoverGenPipeline

        device = torch.device(self.config.device)
        dtype = getattr(torch, self.config.mulacover_dtype)
        self._pipeline = MuLaCoverGenPipeline.from_pretrained(
            str(self.config.model_root),
            device=device,
            dtype={"mulacover": dtype, "codec": torch.float32, "qwen": torch.float32, "transcriptor": torch.float32},
            lazy_load=True,
        )
        return self._pipeline

    @staticmethod
    def _load_or_download(value: str, destination: Path, max_bytes: int, allow_local: bool) -> Path:
        from urllib.parse import urlsplit
        from urllib.request import Request, urlopen

        if not value:
            raise ValueError("missing input source")
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            destination.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with urlopen(Request(value, headers={"Accept": "*/*"}), timeout=120) as response, destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"remote input exceeds {max_bytes} bytes")
                    output.write(chunk)
            if total == 0:
                raise ValueError("remote input is empty")
            return destination
        if not allow_local:
            raise ValueError("local input paths are disabled; use an HTTP(S) source URL")
        source = Path(value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size > max_bytes:
            raise ValueError(f"local input exceeds {max_bytes} bytes")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    @staticmethod
    def _text_source(literal: str, url: str, destination: Path, max_bytes: int, allow_local: bool) -> str:
        if url:
            path = MuLaCoverRuntime._load_or_download(url, destination, max_bytes, allow_local)
            return path.read_text(encoding="utf-8")
        return literal

    def acquire_inputs(self, request: MuLaCoverRequest, attempt_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs_dir = attempt_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"conditioningMode": request.conditioning_mode, "files": []}

        def audio_or_midi(url: str, path: Path | None, name: str) -> Path:
            source = url or (str(path) if path else "")
            result = self._load_or_download(source, inputs_dir / name, self.config.max_download_bytes, self.config.allow_local_inputs)
            manifest["files"].append({"role": name, "name": result.name, "bytes": result.stat().st_size})
            return result

        lyrics = self._text_source(request.lyrics, request.lyrics_url, inputs_dir / "lyrics.txt", self.config.max_download_bytes, self.config.allow_local_inputs)
        tags = self._text_source(request.tags, request.tags_url, inputs_dir / "tags.txt", self.config.max_download_bytes, self.config.allow_local_inputs)
        if not lyrics.strip() or not tags.strip():
            raise ValueError("lyrics and tags must contain non-whitespace text")
        (inputs_dir / "lyrics.txt").write_text(lyrics, encoding="utf-8")
        (inputs_dir / "tags.txt").write_text(tags, encoding="utf-8")
        manifest["text"] = {"lyricsCharacters": len(lyrics), "tagsCharacters": len(tags)}
        values: dict[str, Any] = {"lyrics": str(inputs_dir / "lyrics.txt"), "tags": str(inputs_dir / "tags.txt")}
        if request.conditioning_mode == "reference_audio":
            from urllib.parse import urlsplit
            source_name = Path(request.ref_audio_path).name if request.ref_audio_path else Path(urlsplit(request.ref_audio_url).path).name
            suffix = Path(source_name).suffix or ".audio"
            values["ref_audio"] = str(audio_or_midi(request.ref_audio_url, request.ref_audio_path, "reference-audio" + suffix))
            if request.bpm is not None:
                values["bpm"] = request.bpm
        else:
            values["melody_midi"] = str(audio_or_midi(request.melody_midi_url, request.melody_midi_path, "melody.mid"))
            values["chord_midi"] = str(audio_or_midi(request.chord_midi_url, request.chord_midi_path, "chord.mid"))
            if request.drum_midi_url or request.drum_midi_path:
                values["drum_midi"] = str(audio_or_midi(request.drum_midi_url, request.drum_midi_path, "drums.mid"))
        (attempt_dir / "input-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return values, manifest

    @staticmethod
    def _shifted_midi_inputs(pipe: Any, inputs: dict[str, Any], request: MuLaCoverRequest, output_dir: Path) -> dict[str, Any]:
        """Convert a requested transposition into MIDI before model preprocessing."""
        shift = int(request.semitone_shift) + int(request.octave_shift) * 12
        if shift == 0:
            return inputs
        from mulacover.symbolic import SymbolicCondition

        if "ref_audio" in inputs:
            condition_inputs = {"ref_audio": inputs["ref_audio"]}
            if "bpm" in inputs:
                condition_inputs["bpm"] = inputs["bpm"]
            condition = pipe._symbolic_condition(condition_inputs)
        else:
            condition = SymbolicCondition.from_midi(
                inputs["melody_midi"],
                inputs["chord_midi"],
                inputs.get("drum_midi"),
            )
        melody = condition.melody.clone()
        melody[:, 1] = (melody[:, 1] + shift).clamp(0, 127)
        chords = condition.chords.clone()
        if len(chords):
            chords[:, 1] = (chords[:, 1] + shift) % 12
        shifted = SymbolicCondition(melody=melody, chords=chords, drums=condition.drums.clone(), bpm=condition.bpm)
        paths = shifted.save_midi(output_dir)
        shifted_inputs = {
            "melody_midi": str(paths["melody"]),
            "chord_midi": str(paths["chord"]),
            "drum_midi": str(paths["drums"]),
            "_transposeSemitones": shift,
        }
        if "ref_audio" in inputs:
            shifted_inputs["_transposedFromReferenceAudio"] = True
        else:
            shifted_inputs["_transposedFromMidi"] = True
        return shifted_inputs

    def generate(
        self,
        request: MuLaCoverRequest,
        attempt_dir: Path,
        progress: Callable[[float, str, str], None],
    ) -> MuLaCoverRuntimeResult:
        report = self.preflight()
        if not report.get("ready"):
            if any("GiB free VRAM" in str(item) for item in report.get("missing", [])):
                error = MuLaCoverPreflightError(json.dumps(report, sort_keys=True))
                error.code = "mulacover_insufficient_vram"
                raise error
            raise MuLaCoverPreflightError(json.dumps(report, sort_keys=True))
        attempt_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        progress(0.03, "acquire_inputs", "Acquiring MuLaCover inputs")
        inputs, manifest = self.acquire_inputs(request, attempt_dir)
        progress(0.12, "load_pipeline", "Loading MuLaCover lazy pipeline")
        with self._lock:
            pipe = self._pipeline_for_job()
            progress(0.20, "transcribe_or_load_condition", "Preparing melody and harmony conditioning")
            symbolic_dir = attempt_dir / "symbolic" if request.save_symbolic_midi else attempt_dir / "transient-symbolic"
            symbolic_dir.mkdir(parents=True, exist_ok=True)
            inputs = self._shifted_midi_inputs(pipe, inputs, request, symbolic_dir) if request.semitone_shift or request.octave_shift else inputs
            preprocess = pipe.preprocess(
                inputs,
                cfg_scale=request.cfg_scale,
                max_audio_length_ms=round(request.duration_seconds * 1000),
                symbolic_save_dir=symbolic_dir,
            )
            progress(0.28, "encode_condition", "Conditioning and style embedding prepared")
            import torch

            device = torch.device(self.config.device)
            devices = [device.index or 0] if device.type == "cuda" else []
            seed = request.seed if request.seed is not None else secrets.randbelow(2**63 - 1)
            decode_seed = request.decode_seed if request.decode_seed is not None else secrets.randbelow(2**63 - 1)

            def on_progress(stage: str, completed: int, total: int) -> None:
                progress(min(0.79, 0.28 + 0.51 * completed / max(1, total)), stage, f"MuLaCover {stage} step {completed}/{total}")

            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                outputs = pipe._forward(
                    preprocess,
                    max_audio_length_ms=round(request.duration_seconds * 1000),
                    temperature=request.temperature,
                    topk=request.top_k,
                    cfg_scale=request.cfg_scale,
                    disable_progress=True,
                    cancelled=lambda: False,
                    on_progress=on_progress,
                )
            progress(0.82, "decode_audio", "Decoding generated audio")
            extension = request.output_format
            output_path = attempt_dir / f"cover.{extension}"
            pipe.postprocess(
                {"frames": outputs["frames"]},
                save_path=output_path,
                disable_progress=True,
                cancelled=lambda: False,
                on_progress=lambda stage, completed, total: progress(min(0.96, 0.82 + 0.14 * completed / max(1, total)), stage, f"MuLaCover audio decode step {completed}/{total}"),
                decode_seed=int(decode_seed),
            )
            del preprocess, outputs
            gc.collect()
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not output_path.is_file() or output_path.stat().st_size < 128:
            raise MuLaCoverRuntimeError("MuLaCover completed without a non-empty audio output")
        symbolic_files = sorted((attempt_dir / "symbolic").glob("*.mid")) if symbolic_dir and symbolic_dir.is_dir() else []
        metadata = {
            "runtime": "mulacover",
            "model": self.config.model_name,
            "modelRoot": str(self.config.model_root),
            "device": self.config.device,
            "dtype": self.config.mulacover_dtype,
            "conditioningMode": request.conditioning_mode,
            "durationSecondsRequested": request.duration_seconds,
            "resolved": {"cfgScale": request.cfg_scale, "temperature": request.temperature, "topK": request.top_k, "seed": seed, "decodeSeed": decode_seed, "bpm": request.bpm, "semitoneShift": request.semitone_shift, "octaveShift": request.octave_shift},
            "symbolicArtifacts": [item.name for item in symbolic_files],
            "wallSeconds": time.monotonic() - started,
            "outputBytes": output_path.stat().st_size,
        }
        (attempt_dir / "runtime-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress(0.98, "validate_output", "MuLaCover audio output validated")
        return MuLaCoverRuntimeResult(output_path=output_path, metadata=metadata, input_manifest=manifest)

    def residency_status(self) -> dict[str, Any]:
        return {"pipelineObjectLoaded": self._pipeline is not None, "lazyLoad": True, "device": self.config.device}

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._pipeline = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return self.residency_status()

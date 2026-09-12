from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from .config import QwenImageConfig
from .contracts import QwenLoRARequest


def prepare_loras(
    request_loras: tuple[QwenLoRARequest, ...],
    destination: Path,
    config: QwenImageConfig,
    emit,
) -> tuple[QwenLoRARequest, ...]:
    """Download adapters into a job-scoped directory and return local paths."""

    destination.mkdir(parents=True, exist_ok=True)
    prepared: list[QwenLoRARequest] = []
    for index, lora in enumerate(request_loras):
        lora.validate(config)
        target = destination / f"{index:02d}-{Path(lora.file_name).name}"
        if lora.path is not None:
            shutil.copyfile(lora.path, target)
            source = "local"
        else:
            emit("lora_download_started", index=index, sourceUrl=lora.source_url, fileName=lora.file_name)
            _download(lora.source_url, target, config.max_lora_bytes)
            source = "https"
        prepared.append(QwenLoRARequest(lora.source_url, lora.file_name, lora.scale, lora.is_high_noise, target))
        emit(
            "lora_prepared",
            index=index,
            source=source,
            path=str(target),
            sizeBytes=target.stat().st_size,
            sha256=_sha256(target),
            scale=lora.scale,
            isHighNoise=lora.is_high_noise,
        )
    return tuple(prepared)


def cleanup_loras(path: Path, emit) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    emit("lora_cleanup_finished", path=str(path))


def _download(url: str, destination: Path, max_bytes: int) -> None:
    request = Request(url, headers={"Accept": "application/octet-stream"})
    total = 0
    with urlopen(request, timeout=300) as response, destination.open("wb") as handle:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("LoRA exceeds the worker size limit")
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("LoRA exceeds the worker size limit")
            handle.write(chunk)
    if total == 0:
        raise ValueError("LoRA download returned an empty file")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

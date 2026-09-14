from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from .config import QwenImageEditConfig
from .contracts import QwenImageEditReference


@dataclass(frozen=True)
class PreparedReference:
    request: QwenImageEditReference
    path: Path
    width: int
    height: int
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "fileName": self.request.file_name,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "sizeBytes": self.size_bytes,
            "sha256": self.sha256,
        }


def prepare_references(
    request_references: tuple[QwenImageEditReference, ...],
    destination: Path,
    config: QwenImageEditConfig,
    emit,
) -> tuple[PreparedReference, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedReference] = []
    for index, reference in enumerate(request_references, start=1):
        reference.validate(config)
        raw_path = destination / f"{index:02d}-raw-{Path(reference.file_name).name}"
        if reference.path is not None:
            shutil.copyfile(reference.path, raw_path)
            source = "local"
        else:
            emit("reference_download_started", index=index, sourceUrl=reference.source_url, fileName=reference.file_name)
            _download(reference.source_url, raw_path, config.max_reference_bytes)
            source = "https"
        try:
            with Image.open(raw_path) as source_image:
                source_image.verify()
            with Image.open(raw_path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                width, height = image.size
                if width * height > config.max_reference_pixels:
                    raise ValueError(
                        f"reference image {index} exceeds {config.max_reference_pixels} pixels"
                    )
                target = destination / f"{index:02d}-{Path(reference.file_name).stem}.png"
                image.save(target, format="PNG", optimize=False)
        except Exception:
            raw_path.unlink(missing_ok=True)
            raise
        raw_path.unlink(missing_ok=True)
        item = PreparedReference(
            request=reference,
            path=target,
            width=width,
            height=height,
            size_bytes=target.stat().st_size,
            sha256=_sha256(target),
        )
        prepared.append(item)
        emit(
            "reference_prepared",
            index=index,
            source=source,
            path=str(target),
            width=width,
            height=height,
            sizeBytes=item.size_bytes,
            sha256=item.sha256,
        )
    return tuple(prepared)


def cleanup_references(path: Path, emit) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    emit("reference_cleanup_finished", path=str(path))


def _download(url: str, destination: Path, max_bytes: int) -> None:
    request = Request(url, headers={"Accept": "image/*"})
    total = 0
    with urlopen(request, timeout=300) as response, destination.open("wb") as handle:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("reference image exceeds the worker size limit")
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("reference image exceeds the worker size limit")
            handle.write(chunk)
    if total == 0:
        raise ValueError("reference image download returned an empty file")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

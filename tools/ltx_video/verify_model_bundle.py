from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_FILES = {
    Path("diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"): 18_721_732_720,
    Path("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"): 26_263_860_594,
    Path("vae/ltx-2.5-video-vae-conv-bf16.safetensors"): 1_452_269_922,
    Path("vae/ltx-2.5-audio-vae-bf16.safetensors"): 364_866_540,
    Path("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"): 995_778_752,
}


def inspect_bundle(root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    missing: list[str] = []
    invalid_size: list[str] = []
    total_bytes = 0
    for relative, expected_bytes in EXPECTED_FILES.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(relative.as_posix())
            continue
        size = path.stat().st_size
        total_bytes += size
        complete = size == expected_bytes
        if not complete:
            invalid_size.append(relative.as_posix())
        files.append({"path": relative.as_posix(), "sizeBytes": size, "expectedBytes": expected_bytes, "complete": complete})
    return {
        "root": str(root),
        "requiredFiles": [item.as_posix() for item in EXPECTED_FILES],
        "files": files,
        "missing": missing,
        "invalidSize": invalid_size,
        "totalBytes": total_bytes,
        "totalGiB": round(total_bytes / (1024**3), 3),
        "ready": not missing and not invalid_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LTX 2.5 model context before building the worker image")
    parser.add_argument("root", type=Path, help="model context root containing diffusion_models/, text_encoders/, vae/, and latent_upscale_models/")
    args = parser.parse_args()
    report = inspect_bundle(args.root.expanduser().resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

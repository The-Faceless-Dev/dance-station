from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


EXPECTED_FILES = {
    Path("diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"): 18_721_732_720,
    Path("text_encoders/gemma_ablit_fixed_bf16.safetensors"): 26_263_862_351,
    Path("vae/ltx-2.5-video-vae-conv-bf16.safetensors"): 1_452_269_922,
    Path("vae/ltx-2.5-audio-vae-bf16.safetensors"): 364_866_540,
    Path("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"): 995_778_752,
}

TEXT_ENCODER = Path("text_encoders/gemma_ablit_fixed_bf16.safetensors")
REQUIRED_TEXT_ENCODER_KEYS = {
    "tokenizer_json",
    "multi_modal_projector.embedding_projection.weight",
    "text_embedding_projection.video_aggregate_embed.weight",
    "text_embedding_projection.audio_aggregate_embed.weight",
    "hf_asset__tokenizer_config.json",
    "hf_asset__processor_config.json",
}


def inspect_text_encoder(path: Path) -> dict[str, object]:
    """Validate the packed Gemma header without mmapping the 26 GB weights."""
    try:
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                return {"ready": False, "error": "text encoder header is truncated"}
            header_length = struct.unpack("<Q", length_bytes)[0]
            header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            return {"ready": False, "error": "text encoder header is truncated"}
        header = json.loads(header_bytes.decode("utf-8"))
        if not isinstance(header, dict):
            return {"ready": False, "error": "text encoder header is not a JSON object"}
        metadata = header.get("__metadata__") or {}
        raw_config = metadata.get("gemma_config") if isinstance(metadata, dict) else None
        if raw_config is None:
            return {"ready": False, "error": "missing metadata key gemma_config"}
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            return {"ready": False, "error": f"gemma_config is not valid JSON: {exc}"}

        keys = set(header) - {"__metadata__"}
        missing_keys = sorted(REQUIRED_TEXT_ENCODER_KEYS - keys)
        model_type = config.get("model_type")
        if model_type != "gemma4_unified":
            return {
                "ready": False,
                "error": f"expected gemma4_unified text encoder, got {model_type!r}",
                "modelType": model_type,
                "missingKeys": missing_keys,
            }
        if missing_keys:
            return {
                "ready": False,
                "error": "packed Gemma text encoder is missing required LTX assets",
                "modelType": model_type,
                "missingKeys": missing_keys,
            }
        return {
            "ready": True,
            "modelType": model_type,
            "tensorCount": len(keys),
            "metadataKeys": sorted(metadata),
        }
    except Exception as exc:
        return {"ready": False, "error": f"unable to inspect packed text encoder: {type(exc).__name__}: {exc}"}


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
    encoder_path = root / TEXT_ENCODER
    text_encoder = inspect_text_encoder(encoder_path) if encoder_path.is_file() and encoder_path.stat().st_size > 0 else {
        "ready": False,
        "error": "text encoder file is missing",
    }
    return {
        "root": str(root),
        "requiredFiles": [item.as_posix() for item in EXPECTED_FILES],
        "files": files,
        "missing": missing,
        "invalidSize": invalid_size,
        "totalBytes": total_bytes,
        "totalGiB": round(total_bytes / (1024**3), 3),
        "textEncoder": text_encoder,
        "ready": not missing and not invalid_size and bool(text_encoder.get("ready")),
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

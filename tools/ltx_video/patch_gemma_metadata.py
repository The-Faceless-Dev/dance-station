from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path


HEADER_PREFIX_BYTES = 1 << 20
COPY_CHUNK_BYTES = 64 << 20


def _decode_header(raw: bytes) -> tuple[int, dict[str, object]]:
    if len(raw) < 8:
        raise ValueError("safetensors header is truncated before its length field")
    header_length = struct.unpack("<Q", raw[:8])[0]
    end = 8 + header_length
    if len(raw) < end:
        raise ValueError(f"safetensors header requires {end} bytes, only {len(raw)} were read")
    try:
        header = json.loads(raw[8:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header_length, header


def read_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as handle:
        return _decode_header(handle.read(HEADER_PREFIX_BYTES))


def read_remote_metadata(url: str) -> dict[str, str]:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_PREFIX_BYTES - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        _header_length, header = _decode_header(response.read(HEADER_PREFIX_BYTES))
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("gemma_config"), str):
        raise ValueError(f"metadata source {url!r} does not contain gemma_config")
    try:
        config = json.loads(metadata["gemma_config"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata source {url!r} contains invalid gemma_config JSON: {exc}") from exc
    if not isinstance(config, dict) or config.get("model_type") != "gemma4_unified":
        raise ValueError("metadata source gemma_config is not a Gemma 4 unified config")
    return {"format": "pt", "gemma_config": json.dumps(config, separators=(",", ":"))}


def _encoded_header(header: dict[str, object]) -> bytes:
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return encoded + (b" " * ((-len(encoded)) % 8))


def patch_safetensors_metadata(path: Path, metadata: dict[str, str]) -> dict[str, int]:
    old_header_length, header = read_header(path)
    old_data_start = 8 + old_header_length
    existing = header.get("__metadata__")
    existing_metadata = dict(existing) if isinstance(existing, dict) else {}
    existing_metadata.update(metadata)
    header["__metadata__"] = existing_metadata
    encoded_header = _encoded_header(header)
    new_header_length = len(encoded_header)
    new_data_start = 8 + new_header_length
    delta = new_data_start - old_data_start
    if delta == 0:
        with path.open("r+b") as handle:
            handle.seek(0)
            handle.write(struct.pack("<Q", new_header_length))
            handle.write(encoded_header)
        return {"oldHeaderBytes": old_header_length, "newHeaderBytes": new_header_length, "dataShiftBytes": 0}

    original_size = path.stat().st_size
    with path.open("r+b") as handle:
        handle.truncate(original_size + delta)
        if delta > 0:
            position = original_size
            while position > old_data_start:
                start = max(old_data_start, position - COPY_CHUNK_BYTES)
                handle.seek(start)
                chunk = handle.read(position - start)
                handle.seek(start + delta)
                handle.write(chunk)
                position = start
        else:
            position = old_data_start
            while position < original_size:
                end = min(original_size, position + COPY_CHUNK_BYTES)
                handle.seek(position)
                chunk = handle.read(end - position)
                handle.seek(position + delta)
                handle.write(chunk)
                position = end
        handle.seek(0)
        handle.write(struct.pack("<Q", new_header_length))
        handle.write(encoded_header)
    return {"oldHeaderBytes": old_header_length, "newHeaderBytes": new_header_length, "dataShiftBytes": delta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the canonical Gemma 4 config to a packed LTX text encoder")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--metadata-source-url", required=True)
    args = parser.parse_args()
    metadata = read_remote_metadata(args.metadata_source_url)
    result = patch_safetensors_metadata(args.checkpoint, metadata)
    print(json.dumps({"checkpoint": str(args.checkpoint), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

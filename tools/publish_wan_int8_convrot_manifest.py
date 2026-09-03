#!/usr/bin/env python3
"""Publish a Wan INT8 ConvRot image as registry layers over a working image."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import io
import json
import os
import sys
import tarfile
import urllib.parse
from pathlib import Path
from typing import Any

from publish_wan_overlay_manifest import (
    DOCKER_LAYER,
    DOCKER_MANIFEST,
    MANIFEST_ACCEPT,
    OCI_LAYER,
    OCI_MANIFEST,
    get_registry_token,
    make_overlay_layer,
    registry_request,
    upload_blob,
)


class _HashingWriter:
    def __init__(self, target: Any | None) -> None:
        self.target = target
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, value: bytes) -> int:
        self.digest.update(value)
        self.size += len(value)
        if self.target is None:
            return len(value)
        return self.target.write(value)

    def tell(self) -> int:
        return self.size

    def flush(self) -> None:
        if self.target is not None:
            self.target.flush()


def _tar_info(name: str, *, mode: int, size: int = 0, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = mode
    if directory:
        info.type = tarfile.DIRTYPE
    info.size = size
    return info


def make_comfy_kitchen_layer(package_root: Path) -> tuple[bytes, str, str]:
    """Package only the importable Linux comfy_kitchen module."""

    source_root = package_root / "comfy_kitchen"
    if not source_root.is_dir():
        raise FileNotFoundError(f"comfy-kitchen package was not extracted at {source_root}")
    raw_buffer = io.BytesIO()
    with tarfile.open(fileobj=raw_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        archive.addfile(_tar_info("app/vendor", mode=0o755, directory=True))
        archive.addfile(_tar_info("app/vendor/comfy_kitchen", mode=0o755, directory=True))
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(source_root).as_posix()
            archive_name = f"app/vendor/comfy_kitchen/{relative}"
            with path.open("rb") as source:
                archive.addfile(_tar_info(archive_name, mode=0o644, size=path.stat().st_size), source)
    raw = raw_buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    return compressed, hashlib.sha256(raw).hexdigest(), hashlib.sha256(compressed).hexdigest()


def _write_model_layer(checkpoint: Path, target: Any) -> _HashingWriter:
    """Write a deterministic model layer to a streaming gzip target."""

    archive_name = "models/wan-animate-2/wan-animate-2-14b-int8-convrot.safetensors"
    with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=1, mtime=0) as gzip_file:
        raw_writer = _HashingWriter(gzip_file)
        # USTAR cannot represent the 16+ GB checkpoint size. PAX keeps the
        # layer portable while encoding large file sizes deterministically.
        with tarfile.open(fileobj=raw_writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            archive.addfile(_tar_info("models", mode=0o755, directory=True))
            archive.addfile(_tar_info("models/wan-animate-2", mode=0o755, directory=True))
            with checkpoint.open("rb") as source:
                archive.addfile(
                    _tar_info(archive_name, mode=0o644, size=checkpoint.stat().st_size),
                    source,
                )
    return raw_writer


def make_model_layer(checkpoint: Path, output: Path) -> tuple[str, str, int, int]:
    """Create the large layer on runner disk using bounded Python buffers."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"INT8 ConvRot checkpoint was not found: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as compressed_file:
        compressed_writer = _HashingWriter(compressed_file)
        raw_writer = _write_model_layer(checkpoint, compressed_writer)
    return raw_writer.digest.hexdigest(), compressed_writer.digest.hexdigest(), raw_writer.size, compressed_writer.size


def upload_blob_file(repo: str, token: str, digest: str, payload: Path) -> None:
    """Upload a large layer with one bounded-read monolithic PUT."""

    base = f"https://ghcr.io/v2/{repo}"
    auth = {"Authorization": f"Bearer {token}"}
    try:
        status, _, _ = registry_request("HEAD", f"{base}/blobs/{digest}", headers=auth)
        if status == 200:
            return
    except RegistryError as exc:
        if exc.status != 404:
            raise
    _, response_headers, _ = registry_request(
        "POST", f"{base}/blobs/uploads/", headers=auth
    )
    location = response_headers.get("Location")
    if not location:
        raise RuntimeError("GHCR did not return a blob upload location")
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin("https://ghcr.io", location))
    request_path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    request_path += ("&" if "?" in request_path else "?") + "digest=" + urllib.parse.quote(digest)
    connection = http.client.HTTPSConnection(parsed.netloc, timeout=900)
    try:
        connection.putrequest("PUT", request_path)
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(payload.stat().st_size))
        connection.endheaders()
        with payload.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status not in {201, 202}:
            raise RuntimeError(f"PUT {request_path} returned HTTP {response.status}: {body[:500]}")
    finally:
        connection.close()


def publish(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    code_layer, code_raw_digest, code_compressed_digest = make_overlay_layer(repo_root)
    comfy_layer, comfy_raw_digest, comfy_compressed_digest = make_comfy_kitchen_layer(
        Path(args.comfy_kitchen_root).resolve()
    )
    model_layer_path = Path(args.model_layer).resolve()
    model_raw_digest, model_compressed_digest, model_raw_size, model_compressed_size = make_model_layer(
        checkpoint, model_layer_path
    )
    if args.dry_run:
        return {
            "dryRun": True,
            "codeLayerDigest": f"sha256:{code_compressed_digest}",
            "comfyKitchenLayerDigest": f"sha256:{comfy_compressed_digest}",
            "modelLayerDigest": f"sha256:{model_compressed_digest}",
            "modelRawBytes": model_raw_size,
            "modelCompressedBytes": model_compressed_size,
        }

    username = os.environ.get("REGISTRY_USERNAME")
    password = os.environ.get("REGISTRY_TOKEN")
    if not username or not password:
        raise RuntimeError("REGISTRY_USERNAME and REGISTRY_TOKEN are required")
    token = get_registry_token(args.repo, username, password)
    auth = {"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT}
    base_url = f"https://ghcr.io/v2/{args.repo}"
    _, _, manifest_body = registry_request("GET", f"{base_url}/manifests/{args.base_tag}", headers=auth)
    base_manifest = json.loads(manifest_body)
    base_media_type = base_manifest.get("mediaType")
    if base_media_type not in {DOCKER_MANIFEST, OCI_MANIFEST}:
        raise RuntimeError(f"base image is not a single image manifest: {base_media_type}")
    _, _, config_body = registry_request(
        "GET",
        f"{base_url}/blobs/{base_manifest['config']['digest']}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    config = json.loads(config_body)
    image_config = config.setdefault("config", {})
    env_overrides = {
        "PYTHONPATH": "/app/vendor:/app/src:/opt/wan-animate-2:/opt/wan-vace:/opt/wan-vace/vace:/opt/wan21",
        "GENERATIVE_DANCE_WAN_MODEL": "Wan-Animate-2-INT8-ConvRot",
        "GENERATIVE_DANCE_WAN_CHECKPOINT_FORMAT": "int8_convrot",
        "GENERATIVE_DANCE_WAN_TRANSFORMER": "/models/wan-animate-2/wan-animate-2-14b-int8-convrot.safetensors",
        "GENERATIVE_DANCE_WAN_STEPS": "6",
        "GENERATIVE_DANCE_WAN_MIN_STEPS": "6",
        "GENERATIVE_DANCE_WAN_GUIDANCE_SCALE": "1",
        "GENERATIVE_DANCE_WAN_REFERENCE_STRENGTH": "1.25",
        "GENERATIVE_DANCE_WAN_LIGHTX2V_ENABLED": "1",
        "GENERATIVE_DANCE_WAN_LIGHTX2V_CHECKPOINT": "/models/wan-animate-2/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "GENERATIVE_DANCE_WAN_LIGHTX2V_STRENGTH": "1.0",
        "WAN_ANIMATE_LOG_SCALE": "0.0",
        "WAN_INT8_CONVROT_BACKEND": "cuda",
        "WAN_REQUIRE_INT8_CONVROT": "1",
        "GENERATIVE_DANCE_WAN_TEMPORAL_WINDOW": "81",
        "GENERATIVE_DANCE_WAN_TEMPORAL_CONTEXT_FRAMES": "5",
        "GENERATIVE_DANCE_STAGE_TIMEOUT_SECONDS": "7200",
    }
    keys = set(env_overrides)
    env = [value for value in image_config.get("Env") or [] if value.split("=", 1)[0] not in keys]
    env.extend(f"{key}={value}" for key, value in env_overrides.items())
    image_config["Env"] = env
    upload_blob(args.repo, token, f"sha256:{code_compressed_digest}", code_layer)
    upload_blob(args.repo, token, f"sha256:{comfy_compressed_digest}", comfy_layer)
    upload_blob_file(args.repo, token, f"sha256:{model_compressed_digest}", model_layer_path)
    config.setdefault("rootfs", {}).setdefault("diff_ids", []).extend(
        (
            f"sha256:{code_raw_digest}",
            f"sha256:{comfy_raw_digest}",
            f"sha256:{model_raw_digest}",
        )
    )
    config.setdefault("history", []).extend(
        [
            {"created_by": "COPY runtime-overlay /", "comment": "Wan INT8 ConvRot runtime"},
            {"created_by": "COPY comfy-kitchen /", "comment": "Fused CUDA ConvRot kernel"},
            {"created_by": "COPY wan-animate-2-14b-int8-convrot.safetensors /", "comment": "Official non-distilled checkpoint"},
        ]
    )
    config_bytes = json.dumps(config, separators=(",", ":"), ensure_ascii=False).encode()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    upload_blob(args.repo, token, f"sha256:{config_digest}", config_bytes)
    base_layer_media_type = base_manifest.get("layers", [{}])[0].get("mediaType") or (
        OCI_LAYER if base_media_type == OCI_MANIFEST else DOCKER_LAYER
    )
    manifest = {
        "schemaVersion": 2,
        "mediaType": base_media_type,
        "config": {
            "mediaType": base_manifest["config"]["mediaType"],
            "size": len(config_bytes),
            "digest": f"sha256:{config_digest}",
        },
        "layers": [
            *base_manifest["layers"],
            {"mediaType": base_layer_media_type, "size": len(code_layer), "digest": f"sha256:{code_compressed_digest}"},
            {"mediaType": base_layer_media_type, "size": len(comfy_layer), "digest": f"sha256:{comfy_compressed_digest}"},
            {"mediaType": base_layer_media_type, "size": model_compressed_size, "digest": f"sha256:{model_compressed_digest}"},
        ],
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    registry_request(
        "PUT",
        f"{base_url}/manifests/{args.tag}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": base_media_type, "Accept": MANIFEST_ACCEPT},
        data=manifest_bytes,
    )
    return {
        "tag": args.tag,
        "layers": len(manifest["layers"]),
        "modelCompressedBytes": model_compressed_size,
        "modelLayerDigest": f"sha256:{model_compressed_digest}",
        "comfyKitchenLayerDigest": f"sha256:{comfy_compressed_digest}",
        "configDigest": f"sha256:{config_digest}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo", default="the-faceless-dev/faceless-wan-animate-worker")
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--comfy-kitchen-root", required=True)
    parser.add_argument("--model-layer", default="/tmp/wan-int8-convrot-model.tar.gz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(publish(args), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"publisher_error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()

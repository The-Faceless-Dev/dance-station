#!/usr/bin/env python3
"""Append the existing RIFE/Real-ESRGAN payload to the Wan VACE image.

The quality worker already contains the required small payload layers. This
publisher mounts those layers into the Wan image manifest without unpacking the
large Animate and VACE images on a build runner.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from publish_wan_overlay_manifest import (
    DOCKER_LAYER,
    DOCKER_MANIFEST,
    MANIFEST_ACCEPT,
    OCI_LAYER,
    OCI_MANIFEST,
    RegistryError,
    get_registry_token,
    registry_request,
    upload_blob,
)


QUALITY_ENV = {
    "VACE_STITCH_ENHANCEMENT_ENABLED": "false",
    "VACE_STITCH_ENHANCEMENT_BACKEND": "realesrgan",
    "VACE_STITCH_ENHANCEMENT_SCALE": "2",
    "VACE_STITCH_ENHANCEMENT_TILE_SIZE": "512",
    "VACE_STITCH_ENHANCEMENT_FP16": "true",
    "VACE_STITCH_ENHANCEMENT_COMMAND": (
        "python /app/tools/vace_stitch/realesrgan_stage.py "
        "--input {input} --output {output} --model "
        "/models/realesrgan/RealESRGAN_x2plus.pth --scale {scale} "
        "--tile-size {tile_size} --fp16"
    ),
    "VACE_STITCH_MOTION_INTERPOLATION_ENABLED": "false",
    "VACE_STITCH_MOTION_INTERPOLATION_BACKEND": "rife",
    "VACE_STITCH_MOTION_INTERPOLATION_TARGET_FPS": "48",
    "VACE_STITCH_MOTION_INTERPOLATION_COMMAND": (
        "python /app/tools/vace_stitch/rife_stage.py --input {input} "
        "--output {output} --source-root /opt/rife --checkpoint "
        "/models/rife/rife49.pth --target-fps {target_fps}"
    ),
}


def _tar_info(name: str, *, mode: int = 0o644, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = mode
    if directory:
        info.type = tarfile.DIRTYPE
        info.size = 0
    return info


def _add_file(archive: tarfile.TarFile, archive_name: str, source: Path) -> None:
    info = _tar_info(archive_name)
    info.size = source.stat().st_size
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def make_code_layer(repo_root: Path) -> tuple[bytes, str, str]:
    entries: list[tuple[str, Path]] = []

    for root, destination in (
        (repo_root / "src" / "autotransition" / "generative_dance", "app/src/autotransition/generative_dance"),
        (repo_root / "src" / "autotransition" / "vace_stitch", "app/src/autotransition/vace_stitch"),
        (repo_root / "tools" / "vace_stitch", "app/tools/vace_stitch"),
    ):
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                entries.append((f"{destination}/{path.relative_to(root).as_posix()}", path))

    for relative in (
        Path("tools/generative_dance/wan_animate_2_runner.py"),
        Path("tools/generative_dance/wan_animate_2_runtime.py"),
    ):
        entries.append((f"app/{relative.as_posix()}", repo_root / relative))

    raw_buffer = io.BytesIO()
    directories = {
        "app",
        "app/src",
        "app/src/autotransition",
        "app/src/autotransition/generative_dance",
        "app/src/autotransition/vace_stitch",
        "app/tools",
        "app/tools/generative_dance",
        "app/tools/vace_stitch",
    }
    with tarfile.open(fileobj=raw_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for directory in sorted(directories):
            archive.addfile(_tar_info(directory, mode=0o755, directory=True))
        for archive_name, path in sorted(entries):
            _add_file(archive, archive_name, path)

    raw = raw_buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    return compressed, hashlib.sha256(raw).hexdigest(), hashlib.sha256(compressed).hexdigest()


def _single_manifest(repo: str, tag: str, token: str) -> dict[str, Any]:
    base = f"https://ghcr.io/v2/{repo}"
    auth = {"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT}
    _, _, body = registry_request("GET", f"{base}/manifests/{tag}", headers=auth)
    manifest = json.loads(body)
    if "manifests" not in manifest:
        return manifest
    for descriptor in manifest["manifests"]:
        platform = descriptor.get("platform") or {}
        if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
            _, _, body = registry_request(
                "GET",
                f"{base}/manifests/{descriptor['digest']}",
                headers=auth,
            )
            return json.loads(body)
    raise RuntimeError(f"no linux/amd64 manifest found for {repo}:{tag}")


def _source_layer_indexes(config: dict[str, Any]) -> list[int]:
    history = [entry for entry in config.get("history", []) if not entry.get("empty_layer")]
    selections: dict[str, int] = {}
    for index, entry in enumerate(history):
        command = str(entry.get("created_by", ""))
        if "git clone --depth 1 --branch" in command and "Frame-Interpolation" in command:
            selections["rife_source"] = index
        elif "COPY /rife49.pth /models/rife/rife49.pth" in command:
            selections["rife_model"] = index
        elif "COPY /tmp/quality-layer/ /" in command:
            selections["quality"] = index
    missing = {"rife_source", "rife_model", "quality"} - selections.keys()
    if missing:
        raise RuntimeError(f"quality image does not contain expected payload layers: {sorted(missing)}")
    return [selections["rife_source"], selections["rife_model"], selections["quality"]]


def _download_layer(repo: str, digest: str, token: str) -> bytes:
    _, _, body = registry_request(
        "GET",
        f"https://ghcr.io/v2/{repo}/blobs/{digest}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if hashlib.sha256(body).hexdigest() != digest.removeprefix("sha256:"):
        raise RuntimeError(f"layer digest mismatch for {repo}@{digest}")
    return body


def _update_env(config: dict[str, Any]) -> None:
    image_config = config.setdefault("config", {})
    env = list(image_config.get("Env") or [])
    keys = set(QUALITY_ENV)
    env = [item for item in env if item.split("=", 1)[0] not in keys]
    env.extend(f"{key}={value}" for key, value in QUALITY_ENV.items())
    image_config["Env"] = env


def publish(args: argparse.Namespace) -> dict[str, Any]:
    username = os.environ.get("REGISTRY_USERNAME")
    password = os.environ.get("REGISTRY_TOKEN")
    if not username or not password:
        raise RuntimeError("REGISTRY_USERNAME and REGISTRY_TOKEN are required")

    target_repo = args.repo
    target_token = get_registry_token(target_repo, username, password)
    source_token = get_registry_token(args.quality_repo, username, password)
    target_base = f"https://ghcr.io/v2/{target_repo}"
    target_manifest = _single_manifest(target_repo, args.base_tag, target_token)
    quality_manifest = _single_manifest(args.quality_repo, args.quality_tag, source_token)
    if target_manifest.get("mediaType") not in {DOCKER_MANIFEST, OCI_MANIFEST}:
        raise RuntimeError(f"target tag is not a single image manifest: {target_manifest.get('mediaType')}")

    _, _, target_config_body = registry_request(
        "GET",
        f"{target_base}/blobs/{target_manifest['config']['digest']}",
        headers={"Authorization": f"Bearer {target_token}", "Accept": "application/json"},
    )
    _, _, quality_config_body = registry_request(
        "GET",
        f"https://ghcr.io/v2/{args.quality_repo}/blobs/{quality_manifest['config']['digest']}",
        headers={"Authorization": f"Bearer {source_token}", "Accept": "application/json"},
    )
    target_config = json.loads(target_config_body)
    quality_config = json.loads(quality_config_body)
    selected_indexes = _source_layer_indexes(quality_config)
    code_layer, code_raw_digest, code_digest = make_code_layer(Path(args.repo_root).resolve())

    selected_layers: list[dict[str, Any]] = []
    source_layer_bytes: list[bytes] = []
    source_diff_ids = quality_config["rootfs"]["diff_ids"]
    for index in selected_indexes:
        descriptor = quality_manifest["layers"][index]
        selected_layers.append(dict(descriptor))
        source_layer_bytes.append(_download_layer(args.quality_repo, descriptor["digest"], source_token))

    for descriptor, payload in zip(selected_layers, source_layer_bytes):
        upload_blob(target_repo, target_token, descriptor["digest"], payload)
    upload_blob(target_repo, target_token, f"sha256:{code_digest}", code_layer)

    rootfs = target_config.setdefault("rootfs", {})
    diff_ids = rootfs.setdefault("diff_ids", [])
    history = target_config.setdefault("history", [])
    for index in selected_indexes:
        diff_ids.append(source_diff_ids[index])
        history.append({
            "created_by": f"registry overlay from {args.quality_repo}:{args.quality_tag}",
            "comment": quality_config["history"][index].get("created_by", "quality payload"),
        })
    diff_ids.append(f"sha256:{code_raw_digest}")
    history.append({"created_by": "COPY current VACE quality runners and runtime /"})
    _update_env(target_config)
    config_bytes = json.dumps(target_config, separators=(",", ":"), ensure_ascii=False).encode()
    config_digest = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
    upload_blob(target_repo, target_token, config_digest, config_bytes)

    layer_media_type = target_manifest["layers"][0].get("mediaType") or (
        OCI_LAYER if target_manifest["mediaType"] == OCI_MANIFEST else DOCKER_LAYER
    )
    new_layers = [
        {"mediaType": layer_media_type, "size": item["size"], "digest": item["digest"]}
        for item in selected_layers
    ]
    new_layers.append({"mediaType": layer_media_type, "size": len(code_layer), "digest": f"sha256:{code_digest}"})
    manifest = {
        "schemaVersion": 2,
        "mediaType": target_manifest["mediaType"],
        "config": {
            "mediaType": target_manifest["config"]["mediaType"],
            "size": len(config_bytes),
            "digest": config_digest,
        },
        "layers": [*target_manifest["layers"], *new_layers],
    }
    registry_request(
        "PUT",
        f"{target_base}/manifests/{args.tag}",
        headers={
            "Authorization": f"Bearer {target_token}",
            "Content-Type": target_manifest["mediaType"],
            "Accept": MANIFEST_ACCEPT,
        },
        data=json.dumps(manifest, separators=(",", ":")).encode(),
    )
    return {
        "tag": args.tag,
        "layers": len(manifest["layers"]),
        "qualityLayerDigests": [item["digest"] for item in selected_layers],
        "qualityLayerBytes": sum(item["size"] for item in selected_layers),
        "codeLayerBytes": len(code_layer),
        "configDigest": config_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo", default="the-faceless-dev/faceless-wan-animate-worker")
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--quality-repo", default="the-faceless-dev/faceless-wan-vace-stitch-worker")
    parser.add_argument("--quality-tag", default="vace13b-20260829-quality-rife9")
    args = parser.parse_args()
    print(json.dumps(publish(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

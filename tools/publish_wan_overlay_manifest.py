#!/usr/bin/env python3
"""Publish a small Wan Animate overlay without rebuilding the model layers."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
MANIFEST_ACCEPT = ", ".join((DOCKER_MANIFEST, OCI_MANIFEST, "application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.oci.image.index.v1+json"))


class RegistryError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        super().__init__(f"{method} {url} returned HTTP {status}: {body[:500]}")
        self.status = status


def registry_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RegistryError(method, url, exc.code, body) from exc


def get_registry_token(repo: str, username: str, password: str) -> str:
    basic = base64.b64encode(f"{username}:{password}".encode()).decode()
    query = urllib.parse.urlencode(
        {"service": "ghcr.io", "scope": f"repository:{repo}:pull,push"}
    )
    _, _, body = registry_request(
        "GET",
        f"https://ghcr.io/token?{query}",
        headers={"Authorization": f"Basic {basic}"},
    )
    token = json.loads(body)["token"]
    if not token:
        raise RuntimeError("GHCR returned an empty registry token")
    return token


def add_file_entries(tar: tarfile.TarFile, repo_root: Path) -> None:
    entries: list[tuple[str, Path]] = []
    runtime_root = repo_root / "src" / "autotransition" / "generative_dance"
    for path in runtime_root.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(runtime_root).as_posix()
        entries.append((f"app/src/autotransition/generative_dance/{relative}", path))

    runner = repo_root / "tools" / "generative_dance" / "wan_animate_2_runner.py"
    entries.append(("app/tools/generative_dance/wan_animate_2_runner.py", runner))
    runtime = repo_root / "tools" / "generative_dance" / "wan_animate_2_runtime.py"
    entries.append(("app/tools/generative_dance/wan_animate_2_runtime.py", runtime))

    directories = {
        "app",
        "app/src",
        "app/src/autotransition",
        "app/src/autotransition/generative_dance",
        "app/tools",
        "app/tools/generative_dance",
    }
    for archive_name in sorted(directories):
        info = tarfile.TarInfo(archive_name)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.mode = 0o755
        info.type = tarfile.DIRTYPE
        info.size = 0
        tar.addfile(info)

    for archive_name, path in sorted(entries):
        info = tarfile.TarInfo(archive_name)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.mode = 0o644
        info.size = path.stat().st_size
        with path.open("rb") as source:
            tar.addfile(info, source)


def make_overlay_layer(repo_root: Path) -> tuple[bytes, str, str]:
    raw_buffer = io.BytesIO()
    with tarfile.open(fileobj=raw_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        add_file_entries(archive, repo_root)
    raw = raw_buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    raw_digest = hashlib.sha256(raw).hexdigest()
    compressed_digest = hashlib.sha256(compressed).hexdigest()
    return compressed, raw_digest, compressed_digest


def upload_blob(repo: str, token: str, digest: str, payload: bytes) -> None:
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
    location = urllib.parse.urljoin("https://ghcr.io", location)
    separator = "&" if "?" in location else "?"
    upload_url = f"{location}{separator}digest={urllib.parse.quote(digest)}"
    registry_request(
        "PUT",
        upload_url,
        headers={**auth, "Content-Type": "application/octet-stream"},
        data=payload,
    )


def publish(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    layer, raw_digest, compressed_digest = make_overlay_layer(repo_root)
    if args.dry_run:
        return {
            "dryRun": True,
            "layerDigest": f"sha256:{compressed_digest}",
            "layerDiffId": f"sha256:{raw_digest}",
            "layerBytes": len(layer),
        }

    username = os.environ.get("REGISTRY_USERNAME")
    password = os.environ.get("REGISTRY_TOKEN")
    if not username or not password:
        raise RuntimeError("REGISTRY_USERNAME and REGISTRY_TOKEN are required")

    token = get_registry_token(args.repo, username, password)
    auth = {
        "Authorization": f"Bearer {token}",
        "Accept": MANIFEST_ACCEPT,
    }
    base_url = f"https://ghcr.io/v2/{args.repo}"
    _, _, manifest_body = registry_request(
        "GET",
        f"{base_url}/manifests/{args.base_tag}",
        headers=auth,
    )
    base_manifest = json.loads(manifest_body)
    if base_manifest.get("mediaType") not in {DOCKER_MANIFEST, OCI_MANIFEST}:
        raise RuntimeError(
            f"base image is not a single Docker/OCI image manifest: {base_manifest.get('mediaType')}"
        )

    _, _, config_body = registry_request(
        "GET",
        f"{base_url}/blobs/{base_manifest['config']['digest']}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    config = json.loads(config_body)
    config.setdefault("rootfs", {}).setdefault("diff_ids", []).append(
        f"sha256:{raw_digest}"
    )
    config.setdefault("history", []).append(
        {"created_by": "COPY payload/ /", "comment": "Wan Animate identity overlay"}
    )
    image_config = config.setdefault("config", {})
    env = list(image_config.get("Env") or [])
    env_overrides = {
        "HOME": "/home/wan",
        "XDG_CACHE_HOME": "/home/wan/.cache",
        "TRITON_CACHE_DIR": "/home/wan/.cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/home/wan/.cache/torchinductor",
        "HF_HOME": "/home/wan/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/home/wan/.cache/huggingface/transformers",
        "TORCH_HOME": "/home/wan/.cache/torch",
        "MPLCONFIGDIR": "/home/wan/.cache/matplotlib",
        "WAN_GGUF_GPU_RAW_CACHE": "0",
        "WAN_GGUF_DEQUANT_BACKEND": "triton",
        "WAN_T5_DEVICE": "cpu",
        # `sdpa` forces the runner's eager masked-attention adapter. The
        # `chunked` value alone leaves official torch.compile Flex Attention
        # active whenever Triton is installed.
        "WAN_FLEX_ATTENTION_BACKEND": "sdpa",
        "WAN_SDPA_BACKEND": "chunked",
        "WAN_FLEX_ATTENTION_CHUNK_SIZE": "64",
        "WAN_SDPA_CHUNK_SIZE": "256",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "GENERATIVE_DANCE_WAN_REFERENCE_STRENGTH": "1.25",
    }
    keys = set(env_overrides)
    env = [value for value in env if value.split("=", 1)[0] not in keys]
    env.extend(f"{key}={value}" for key, value in env_overrides.items())
    image_config["Env"] = env
    config_bytes = json.dumps(config, separators=(",", ":"), ensure_ascii=False).encode()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    upload_blob(args.repo, token, f"sha256:{compressed_digest}", layer)
    upload_blob(args.repo, token, f"sha256:{config_digest}", config_bytes)

    base_media_type = base_manifest.get("mediaType", DOCKER_MANIFEST)
    base_layer_media_type = (
        base_manifest.get("layers", [{}])[0].get("mediaType")
        or (OCI_LAYER if base_media_type == OCI_MANIFEST else DOCKER_LAYER)
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
            {
                "mediaType": base_layer_media_type,
                "size": len(layer),
                "digest": f"sha256:{compressed_digest}",
            },
        ],
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    registry_request(
        "PUT",
        f"{base_url}/manifests/{args.tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": base_media_type,
            "Accept": MANIFEST_ACCEPT,
        },
        data=manifest_bytes,
    )
    return {
        "tag": args.tag,
        "layers": len(manifest["layers"]),
        "compressedBytes": sum(layer["size"] for layer in manifest["layers"]),
        "overlayBytes": len(layer),
        "configDigest": f"sha256:{config_digest}",
        "overlayDigest": f"sha256:{compressed_digest}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo", default="the-faceless-dev/faceless-wan-animate-worker")
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

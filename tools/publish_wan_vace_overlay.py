#!/usr/bin/env python3
"""Append the VACE runtime and checkpoint to a known-good Wan image.

This publisher intentionally talks to the OCI registry directly.  The existing
Wan image is large and already proven on Vast, so rebuilding it locally would
duplicate its layers on the developer disk.  The added layer contains the
small runtime/source trees and the VACE checkpoint only; shared Wan encoders
are represented by symlinks to the files already present in the base image.
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
import time
from dataclasses import dataclass
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator


DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DOCKER_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"
OCI_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
MANIFEST_ACCEPT = ", ".join(
    (
        DOCKER_MANIFEST,
        OCI_MANIFEST,
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    )
)

CHECKPOINT_NAME = "wan2.1_vace_14B_fp8_scaled.safetensors"
LIGHTX2V_LORA_NAME = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    size: int

    def open(self):
        return urllib.request.urlopen(self.url, timeout=600)

VACE_ENV = {
    "PYTHONPATH": "/app/src:/opt/wan-animate-2:/opt/wan-vace:/opt/wan-vace/vace:/opt/wan21",
    "VACE_STITCH_ENABLED": "true",
    "VACE_STITCH_BACKEND": "native",
    "VACE_STITCH_SOURCE_ROOT": "/opt/wan-vace",
    "VACE_STITCH_CWD": "/opt/wan-vace",
    "VACE_STITCH_CHECKPOINT_DIR": "/models/wan-vace-14b",
    "VACE_STITCH_CHECKPOINT_FILE": f"/models/wan-vace-14b/{CHECKPOINT_NAME}",
    "VACE_STITCH_MODEL_NAME": "vace-14B",
    "VACE_STITCH_MODEL_SIZE": "480p",
    "VACE_STITCH_MODEL_FPS": "16",
    "VACE_STITCH_OUTPUT_FPS": "24",
    "VACE_STITCH_OUTPUT_WIDTH": "480",
    "VACE_STITCH_OUTPUT_HEIGHT": "832",
    "VACE_STITCH_DEFAULT_PROMPT": "the character continues dancing",
    "VACE_STITCH_DEFAULT_LOOP_PROMPT": "the character continues dancing and returns smoothly to the starting motion",
    "VACE_STITCH_DEFAULT_GAP_SECONDS": "2",
    "VACE_STITCH_LOOP_ENABLED": "true",
    "VACE_STITCH_MIN_GAP_SECONDS": "0.25",
    "VACE_STITCH_MAX_GAP_SECONDS": "20",
    "VACE_STITCH_CONTEXT_BEFORE_SECONDS": "1",
    "VACE_STITCH_CONTEXT_AFTER_SECONDS": "1",
    "VACE_STITCH_MAX_WINDOW_FRAMES": "81",
    "VACE_STITCH_SAMPLE_STEPS": "50",
    "VACE_STITCH_SAMPLE_SHIFT": "16",
    "VACE_STITCH_GUIDE_SCALE": "5",
    "VACE_STITCH_OFFLOAD_MODEL": "true",
    "VACE_STITCH_T5_CPU": "true",
    "VACE_STITCH_ATTENTION_BACKEND": "auto",
    "VACE_STITCH_TF32": "true",
    "VACE_STITCH_TEMPORARY_BACKGROUND": "0x7f7f7f",
    "VACE_STITCH_TRANSPARENT_DEFAULT": "true",
    "VACE_STITCH_MATTE_BACKEND": "native",
    "VACE_STITCH_MATTE_CHECKPOINT": "/models/birefnet-matting",
    "VACE_STITCH_MATTE_DEVICE": "cuda",
    "VACE_STITCH_MATTE_DTYPE": "float16",
    "VACE_STITCH_MATTE_BATCH_SIZE": "2",
    "VACE_STITCH_MATTE_INPUT_SIZE": "1024",
    "VACE_STITCH_ENHANCEMENT_ENABLED": "false",
    "VACE_STITCH_MOTION_INTERPOLATION_ENABLED": "false",
    "VACE_STITCH_ARTIFACT_ROOT": "/var/lib/autotransition/generative-dance-jobs",
    "VACE_STITCH_STAGE_TIMEOUT_SECONDS": "7200",
    "VACE_STITCH_MAX_UPLOAD_BYTES": "2147483648",
}

LIGHTX2V_ENV = {
    **VACE_ENV,
    "PYTHONPATH": "/app/src:/opt/wan-animate-2:/opt/wan-vace:/opt/wan-vace/vace:/opt/wan21:/opt/lightx2v",
    "VACE_STITCH_BACKEND": "lightx2v",
    "VACE_STITCH_LIGHTX2V_SOURCE_ROOT": "/opt/lightx2v",
    "VACE_STITCH_LIGHTX2V_CONFIG": "/models/wan-vace-14b/lightx2v-vace.json",
    "VACE_STITCH_LIGHTX2V_LORA": f"/models/wan-vace-lightx2v/{LIGHTX2V_LORA_NAME}",
    "VACE_STITCH_LIGHTX2V_LORA_STRENGTH": "1.0",
    "VACE_STITCH_LIGHTX2V_STEPS": "4",
    "VACE_STITCH_LIGHTX2V_ATTENTION": "flash_attn2",
    "VACE_STITCH_SAMPLE_STEPS": "4",
    "VACE_STITCH_SAMPLE_SHIFT": "5",
    "VACE_STITCH_GUIDE_SCALE": "1",
    "VACE_STITCH_OFFLOAD_MODEL": "false",
    "VACE_STITCH_T5_CPU": "false",
    "VACE_STITCH_ATTENTION_BACKEND": "flash_attention_2",
    "VACE_STITCH_TF32": "true",
}


class RegistryError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        super().__init__(f"{method} {url} returned HTTP {status}: {body[:500]}")
        self.status = status


def request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | Iterable[bytes] | None = None,
    timeout: float = 180,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RegistryError(method, url, exc.code, body) from exc


def registry_token(repo: str, username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    query = urllib.parse.urlencode(
        {"service": "ghcr.io", "scope": f"repository:{repo}:pull,push"}
    )
    _, _, body = request(
        "GET",
        f"https://ghcr.io/token?{query}",
        headers={"Authorization": f"Basic {encoded}"},
    )
    token = json.loads(body)["token"]
    if not token:
        raise RuntimeError("GHCR returned an empty registry token")
    return token


def normalize_archive_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def iter_tree(root: Path, destination: str) -> Iterator[tuple[str, Path]]:
    destination = normalize_archive_path(destination)
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.is_symlink()
        ):
            continue
        relative = path.relative_to(root).as_posix()
        yield f"{destination}/{relative}", path


def add_dir(tar: tarfile.TarFile, name: str, directories: set[str]) -> None:
    name = normalize_archive_path(name).rstrip("/")
    if not name or name in directories:
        return
    parent = name.rpartition("/")[0]
    if parent:
        add_dir(tar, parent, directories)
    info = tarfile.TarInfo(name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755
    info.type = tarfile.DIRTYPE
    info.size = 0
    tar.addfile(info)
    directories.add(name)


def add_file(
    tar: tarfile.TarFile,
    archive_name: str,
    source: Path | bytes | RemoteFile,
    directories: set[str],
    *,
    mode: int = 0o644,
) -> None:
    archive_name = normalize_archive_path(archive_name)
    add_dir(tar, archive_name.rpartition("/")[0], directories)
    info = tarfile.TarInfo(archive_name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = mode
    if isinstance(source, bytes):
        info.size = len(source)
        tar.addfile(info, io.BytesIO(source))
        return
    if isinstance(source, RemoteFile):
        info.size = source.size
        handle = source.open()
    else:
        info.size = source.stat().st_size
        handle = source.open("rb")
    with handle:
        tar.addfile(info, handle)


def add_symlink(tar: tarfile.TarFile, archive_name: str, target: str, directories: set[str]) -> None:
    archive_name = normalize_archive_path(archive_name)
    add_dir(tar, archive_name.rpartition("/")[0], directories)
    info = tarfile.TarInfo(archive_name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o777
    info.type = tarfile.SYMTYPE
    info.linkname = target
    tar.addfile(info)


def build_entries(
    repo_root: Path,
    vace_root: Path,
    wan21_root: Path,
    checkpoint: Path | RemoteFile | None,
    *,
    lightx2v_root: Path | None = None,
    lightx2v_lora: Path | RemoteFile | None = None,
) -> list[tuple[str, str, Path | bytes | RemoteFile | str | None]]:
    entries: list[tuple[str, str, Path | bytes | RemoteFile | str | None]] = []
    for path in repo_root.joinpath("src", "autotransition", "generative_dance").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            entries.append((
                f"app/src/autotransition/generative_dance/{path.relative_to(repo_root.joinpath('src', 'autotransition', 'generative_dance')).as_posix()}",
                "file",
                path,
            ))
    for path in repo_root.joinpath("src", "autotransition", "vace_stitch").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            entries.append((
                f"app/src/autotransition/vace_stitch/{path.relative_to(repo_root.joinpath('src', 'autotransition', 'vace_stitch')).as_posix()}",
                "file",
                path,
            ))
    for relative in (
        Path("tools/generative_dance/wan_animate_2_runner.py"),
        Path("tools/generative_dance/wan_animate_2_runtime.py"),
    ):
        entries.append((f"app/{relative.as_posix()}", "file", repo_root / relative))

    for archive_name, path in iter_tree(vace_root, "opt/wan-vace/vace"):
        if archive_name.endswith("/annotators/__init__.py"):
            source: Path | bytes = (
                b"# VACE runtime-only annotator exports.\n"
                b"# Heavy optional annotators are intentionally not imported at startup.\n"
                b"from .common import PlainPromptAnnotator\n"
            )
        elif archive_name.endswith("/annotators/utils.py"):
            source = path.read_bytes().replace(
                b"import pycocotools.mask as mask_utils",
                b"try:\n    import pycocotools.mask as mask_utils\nexcept ModuleNotFoundError:  # optional for plain VACE runtime inputs\n    mask_utils = None",
            )
        else:
            source = path
        entries.append((archive_name, "file", source))
    for archive_name, path in iter_tree(wan21_root, "opt/wan21/wan"):
        entries.append((archive_name, "file", path))

    if lightx2v_root is not None:
        for archive_name, path in iter_tree(lightx2v_root, "opt/lightx2v"):
            entries.append((archive_name, "file", path))

    config = repo_root / "containers" / "wan-animate-worker" / "vace-model-config.json"
    entries.append(("models/wan-vace-14b/config.json", "file", config))
    if checkpoint is not None:
        entries.append((f"models/wan-vace-14b/{CHECKPOINT_NAME}", "file", checkpoint))
    lightx_config = repo_root / "containers" / "wan-animate-worker" / "lightx2v-vace.json"
    if lightx2v_root is not None:
        entries.append(("models/wan-vace-14b/lightx2v-vace.json", "file", lightx_config))
    if lightx2v_lora is not None:
        entries.append((f"models/wan-vace-lightx2v/{LIGHTX2V_LORA_NAME}", "file", lightx2v_lora))
    entries.extend(
        (
            ("models/wan-vace-14b/models_t5_umt5-xxl-enc-bf16.pth", "symlink", "../wan-animate-2/companions/models_t5_umt5-xxl-enc-bf16.pth"),
            ("models/wan-vace-14b/Wan2.1_VAE.pth", "symlink", "../wan-animate-2/companions/vae.pth"),
            ("models/wan-vace-14b/google/umt5-xxl", "symlink", "../../wan-animate-2/companions/umt5-xxl"),
        )
    )
    return entries


class LayerStream:
    def __init__(self, sink) -> None:
        self.sink = sink
        self.raw_hash = hashlib.sha256()
        self.compressed_hash = hashlib.sha256()
        self.raw_bytes = 0
        self.compressed_bytes = 0

    def write_raw(self, data: bytes) -> int:
        self.raw_hash.update(data)
        self.raw_bytes += len(data)
        return len(data)

    def write_compressed(self, data: bytes) -> int:
        self.compressed_hash.update(data)
        self.compressed_bytes += len(data)
        if self.sink is not None:
            self.sink.write(data)
        return len(data)

    def tell(self) -> int:
        return self.raw_bytes


class GzipSink:
    def __init__(self, layer: LayerStream) -> None:
        self.layer = layer
        # The checkpoint is already a packed tensor stream.  Level 1 avoids
        # spending CPU time trying to compress high-entropy FP8 bytes again.
        self.compressor = gzip.GzipFile(fileobj=self, mode="wb", compresslevel=1, mtime=0)

    def write(self, data: bytes) -> int:
        return self.layer.write_compressed(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.compressor.close()


class TarStream:
    """File-like tar sink that hashes raw tar bytes before compression."""

    def __init__(self, layer: LayerStream) -> None:
        self.layer = layer
        self.compressed = GzipSink(layer)

    def write(self, data: bytes) -> int:
        self.layer.write_raw(data)
        return self.compressed.compressor.write(data)

    def tell(self) -> int:
        return self.layer.raw_bytes

    def close(self) -> None:
        self.compressed.close()


def emit_layer(entries, sink) -> LayerStream:
    layer = LayerStream(sink)
    tar_sink = TarStream(layer)
    directories: set[str] = set()
    try:
        with tarfile.open(fileobj=tar_sink, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for archive_name, kind, source in sorted(entries, key=lambda item: item[0]):
                if kind == "file":
                    assert isinstance(source, (Path, bytes, RemoteFile))
                    add_file(archive, archive_name, source, directories)
                else:
                    assert kind == "symlink" and isinstance(source, str)
                    add_symlink(archive, archive_name, source, directories)
    finally:
        tar_sink.close()
    return layer


class UploadSink:
    def __init__(self, response):
        self.response = response

    def write(self, data: bytes) -> int:
        self.response.write(data)
        return len(data)


def layer_pass(entries, output=None) -> LayerStream:
    if output is None:
        return emit_layer(entries, None)
    return emit_layer(entries, output)


def upload_layer(repo: str, token: str, digest: str, entries, size: int) -> None:
    base = f"https://ghcr.io/v2/{repo}"
    auth = {"Authorization": f"Bearer {token}"}
    try:
        status, _, _ = request("HEAD", f"{base}/blobs/{digest}", headers=auth)
        if status == 200:
            print(f"[ghcr] layer already exists {digest} ({size} bytes)", flush=True)
            return
    except RegistryError as exc:
        if exc.status != 404:
            raise

    _, response_headers, _ = request("POST", f"{base}/blobs/uploads/", headers=auth)
    location = response_headers.get("Location")
    if not location:
        raise RuntimeError("GHCR did not return a blob upload location")
    location = urllib.parse.urljoin("https://ghcr.io", location)
    separator = "&" if "?" in location else "?"
    upload_url = f"{location}{separator}digest={urllib.parse.quote(digest)}"

    stream_upload(upload_url, auth, entries, size, digest)


def stream_upload(url: str, headers: dict[str, str], entries, size: int, expected_digest: str) -> None:
    # urllib does not expose a convenient producer/consumer request body.  A
    # temporary file would consume another ~18 GB, so use an HTTP connection
    # with chunked transfer and feed it from the deterministic tar stream.
    import http.client

    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPSConnection(parsed.netloc, timeout=600)
    path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    token = headers["Authorization"]
    connection.putrequest("PUT", path)
    connection.putheader("Authorization", token)
    connection.putheader("Content-Type", "application/octet-stream")
    connection.putheader("Content-Length", str(size))
    connection.endheaders()

    class HttpSink:
        def __init__(self) -> None:
            self.sent = 0
            self.last_report = 0
            self.started = time.monotonic()

        def write(self, data: bytes) -> int:
            connection.send(data)
            self.sent += len(data)
            if self.sent - self.last_report >= 256 * 1024 * 1024:
                elapsed = max(time.monotonic() - self.started, 0.001)
                rate = self.sent / elapsed / (1024 * 1024)
                print(
                    f"[ghcr] uploaded {self.sent / (1024 ** 3):.2f} GiB / {size / (1024 ** 3):.2f} GiB ({rate:.1f} MiB/s)",
                    flush=True,
                )
                self.last_report = self.sent
            return len(data)

    try:
        streamed = layer_pass(entries, HttpSink())
        if streamed.compressed_bytes != size:
            raise RuntimeError(
                f"streamed layer size changed between passes: expected {size}, got {streamed.compressed_bytes}"
            )
        actual_digest = f"sha256:{streamed.compressed_hash.hexdigest()}"
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"streamed layer digest changed between passes: expected {expected_digest}, got {actual_digest}"
            )
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status not in {200, 201, 202}:
            raise RegistryError("PUT", url, response.status, body)
    finally:
        connection.close()


def update_config(config: dict, raw_diff_id: str, *, lightx2v_enabled: bool = False) -> bytes:
    rootfs = config.setdefault("rootfs", {})
    rootfs.setdefault("diff_ids", []).append(raw_diff_id)
    config.setdefault("history", []).append(
        {"created_by": "VACE runtime and checkpoint registry overlay"}
    )
    image_config = config.setdefault("config", {})
    env = list(image_config.get("Env") or [])
    env_values = LIGHTX2V_ENV if lightx2v_enabled else VACE_ENV
    keys = set(env_values)
    env = [item for item in env if item.split("=", 1)[0] not in keys]
    env.extend(f"{key}={value}" for key, value in env_values.items())
    image_config["Env"] = env
    return json.dumps(config, separators=(",", ":"), ensure_ascii=False).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo", default="the-faceless-dev/faceless-wan-animate-worker")
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--vace-source", required=True)
    parser.add_argument("--wan21-source", required=True)
    parser.add_argument("--lightx2v-source")
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--lightx2v-lora")
    lora_group.add_argument("--lightx2v-lora-url")
    parser.add_argument("--lightx2v-lora-size", type=int)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument("--checkpoint-url")
    checkpoint_group.add_argument("--reuse-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    vace_root = Path(args.vace_source).resolve()
    wan21_root = Path(args.wan21_source).resolve()
    lightx2v_root = Path(args.lightx2v_source).resolve() if args.lightx2v_source else None
    if lightx2v_root is not None and not lightx2v_root.is_dir():
        raise SystemExit(f"LightX2V source is missing: {lightx2v_root}")
    if args.lightx2v_lora_url and (not args.lightx2v_lora_size or args.lightx2v_lora_size <= 0):
        raise SystemExit("--lightx2v-lora-size is required with --lightx2v-lora-url")
    lightx2v_lora: Path | RemoteFile | None = None
    if args.lightx2v_lora_url:
        lightx2v_lora = RemoteFile(args.lightx2v_lora_url, args.lightx2v_lora_size)
    elif args.lightx2v_lora:
        lightx2v_lora = Path(args.lightx2v_lora).resolve()
        if not lightx2v_lora.is_file() or lightx2v_lora.stat().st_size <= 0:
            raise SystemExit(f"LightX2V LoRA is missing or empty: {lightx2v_lora}")
    if lightx2v_root is not None and lightx2v_lora is None:
        raise SystemExit("--lightx2v-lora or --lightx2v-lora-url is required with --lightx2v-source")
    if not any((args.checkpoint, args.checkpoint_url, args.reuse_checkpoint)):
        raise SystemExit("one of --checkpoint, --checkpoint-url, or --reuse-checkpoint is required")
    if args.reuse_checkpoint:
        checkpoint = None
        checkpoint_size = 0
    elif args.checkpoint_url:
        if not args.checkpoint_size or args.checkpoint_size <= 0:
            raise SystemExit("--checkpoint-size is required and must be positive with --checkpoint-url")
        checkpoint: Path | RemoteFile = RemoteFile(args.checkpoint_url, args.checkpoint_size)
        checkpoint_size = checkpoint.size
    else:
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise SystemExit(f"checkpoint is missing or empty: {checkpoint}")
        checkpoint_size = checkpoint.stat().st_size
    entries = build_entries(
        repo_root,
        vace_root,
        wan21_root,
        checkpoint,
        lightx2v_root=lightx2v_root,
        lightx2v_lora=lightx2v_lora,
    )
    print(f"[overlay] entries={len(entries)} checkpointBytes={checkpoint_size}", flush=True)
    first = layer_pass(entries)
    layer_digest = f"sha256:{first.compressed_hash.hexdigest()}"
    layer_diff_id = f"sha256:{first.raw_hash.hexdigest()}"
    print(
        json.dumps(
            {
                "layerDigest": layer_digest,
                "layerDiffId": layer_diff_id,
                "compressedBytes": first.compressed_bytes,
                "rawBytes": first.raw_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    username = os.getenv("REGISTRY_USERNAME")
    password = os.getenv("REGISTRY_TOKEN")
    if not username or not password:
        raise SystemExit("REGISTRY_USERNAME and REGISTRY_TOKEN are required")
    token = registry_token(args.repo, username, password)
    auth = {"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT}
    base_url = f"https://ghcr.io/v2/{args.repo}"
    _, _, manifest_body = request("GET", f"{base_url}/manifests/{args.base_tag}", headers=auth)
    base_manifest = json.loads(manifest_body)
    if base_manifest.get("mediaType") not in {DOCKER_MANIFEST, OCI_MANIFEST}:
        raise SystemExit(f"base tag is not a single image manifest: {base_manifest.get('mediaType')}")
    _, _, config_body = request(
        "GET",
        f"{base_url}/blobs/{base_manifest['config']['digest']}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    image_config = update_config(json.loads(config_body), layer_diff_id, lightx2v_enabled=lightx2v_root is not None)
    config_digest = f"sha256:{hashlib.sha256(image_config).hexdigest()}"
    upload_layer(args.repo, token, layer_digest, entries, first.compressed_bytes)
    # Upload the small image config with the regular registry flow.
    _, response_headers, _ = request(
        "POST", f"{base_url}/blobs/uploads/", headers={"Authorization": f"Bearer {token}"}
    )
    location = response_headers.get("Location")
    if not location:
        raise RuntimeError("GHCR did not return a config upload location")
    location = urllib.parse.urljoin("https://ghcr.io", location)
    separator = "&" if "?" in location else "?"
    request(
        "PUT",
        f"{location}{separator}digest={urllib.parse.quote(config_digest)}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
        data=image_config,
    )
    layer_media_type = base_manifest.get("layers", [{}])[0].get("mediaType") or (
        OCI_LAYER if base_manifest.get("mediaType") == OCI_MANIFEST else DOCKER_LAYER
    )
    manifest = {
        "schemaVersion": 2,
        "mediaType": base_manifest["mediaType"],
        "config": {
            "mediaType": base_manifest["config"]["mediaType"],
            "size": len(image_config),
            "digest": config_digest,
        },
        "layers": [
            *base_manifest["layers"],
            {"mediaType": layer_media_type, "size": first.compressed_bytes, "digest": layer_digest},
        ],
    }
    request(
        "PUT",
        f"{base_url}/manifests/{args.tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": base_manifest["mediaType"],
            "Accept": MANIFEST_ACCEPT,
        },
        data=json.dumps(manifest, separators=(",", ":")).encode(),
    )
    print(json.dumps({"tag": args.tag, "layers": len(manifest["layers"]), "layerDigest": layer_digest}, indent=2), flush=True)


if __name__ == "__main__":
    main()

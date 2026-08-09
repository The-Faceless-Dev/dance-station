from __future__ import annotations

import json
import struct
from pathlib import Path

from autotransition.avatar.canonical_skeleton import (
    CANONICAL_PARENTS,
    REQUIRED_CANONICAL_ROLES,
    fit_profile,
    read_glb,
    validate_profile,
    write_manifest,
    write_skeleton_glb,
)
from autotransition.avatar.validation import read_glb_json, validate_glb, validate_manifest


def write_mesh_glb(path: Path) -> None:
    points = [
        (-0.4, -0.5, -0.1),
        (0.4, -0.5, -0.1),
        (-0.3, 0.5, 0.1),
        (0.3, 0.5, 0.1),
    ]
    binary = b"".join(struct.pack("<fff", *point) for point in points)
    payload = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "geometry_0", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "buffers": [{"byteLength": len(binary)}],
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    binary += b"\0" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def test_canonical_profile_and_skeleton_glb_are_named_and_valid(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    skeleton = tmp_path / "canonical-skeleton.glb"
    manifest = tmp_path / "manifest.json"
    write_mesh_glb(mesh)

    profile = fit_profile(mesh)
    assert validate_profile(profile) == []
    assert set(profile["joints"]) == set(CANONICAL_PARENTS)
    write_skeleton_glb(mesh, profile, skeleton)
    write_manifest(profile, manifest, model_file=skeleton.name)

    payload = read_glb_json(skeleton)
    names = {node.get("name") for node in payload["nodes"]}
    assert set(CANONICAL_PARENTS).issubset(names)
    assert not any(name.startswith("bone_") for name in names if isinstance(name, str))
    assert validate_glb(skeleton, require_skin=False).ok
    assert validate_manifest(manifest, payload).ok
    assert set(json.loads(manifest.read_text())["bones"]) == set(REQUIRED_CANONICAL_ROLES)
    assert read_glb(skeleton).binary


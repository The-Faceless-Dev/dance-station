"""Structural and optional deformation validation for generated avatars."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autotransition.avatar.resources import AvatarProcessError, run_command

REQUIRED_BONES = (
    "hips",
    "spine",
    "chest",
    "head",
    "upperArmLeft",
    "upperArmRight",
    "forearmLeft",
    "forearmRight",
    "upperLegLeft",
    "upperLegRight",
    "lowerLegLeft",
    "lowerLegRight",
    "footLeft",
    "footRight",
)


class AvatarValidationError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.retryable = retryable


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    checks: tuple[dict[str, Any], ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        failed = next((check for check in self.checks if not check.get("ok")), None)
        if failed is None:
            raise AvatarValidationError("avatar_validation_failed", "avatar validation failed")
        raise AvatarValidationError(
            str(failed.get("code", "avatar_validation_failed")),
            str(failed.get("message", "avatar validation failed")),
            details=self.details,
        )


def _check(ok: bool, code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": ok, "code": code, "message": message, **details}


def read_glb_json(path: Path) -> dict[str, Any]:
    """Read the JSON chunk without requiring trimesh or Blender."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AvatarValidationError("glb_unreadable", f"could not read GLB: {path}", retryable=True) from exc
    if len(data) < 20 or data[:4] != b"glTF":
        raise AvatarValidationError("glb_invalid_header", "output is not a valid glTF 2.0 binary", retryable=True)
    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length != len(data):
        raise AvatarValidationError("glb_invalid_header", "GLB version or declared length is invalid", retryable=True)
    offset = 12
    json_chunk: bytes | None = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_chunk = chunk
            break
    if json_chunk is None:
        raise AvatarValidationError("glb_missing_json", "GLB has no JSON chunk", retryable=True)
    try:
        payload = json.loads(json_chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvatarValidationError("glb_invalid_json", "GLB JSON chunk could not be parsed", retryable=True) from exc
    if not isinstance(payload, dict):
        raise AvatarValidationError("glb_invalid_json", "GLB JSON root must be an object", retryable=True)
    return payload


def _finite_values(value: Any, location: str = "root") -> Iterable[str]:
    if isinstance(value, float) and not math.isfinite(value):
        yield location
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _finite_values(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _finite_values(child, f"{location}[{index}]")


def validate_glb(path: Path, *, require_skin: bool = True) -> ValidationReport:
    checks: list[dict[str, Any]] = []
    try:
        payload = read_glb_json(path)
    except AvatarValidationError as exc:
        return ValidationReport(False, (_check(False, exc.code, str(exc)),))
    checks.append(_check(True, "glb_header", "GLB header and JSON chunk are valid"))
    non_finite = list(_finite_values(payload))
    checks.append(_check(not non_finite, "glb_non_finite", "GLB JSON contains only finite numbers", locations=non_finite))
    meshes = payload.get("meshes")
    nodes = payload.get("nodes")
    skins = payload.get("skins")
    accessors = payload.get("accessors", [])
    checks.append(_check(isinstance(meshes, list) and bool(meshes), "mesh_missing", "GLB contains at least one mesh"))
    checks.append(_check(isinstance(nodes, list) and bool(nodes), "nodes_missing", "GLB contains nodes"))
    if require_skin:
        checks.append(_check(isinstance(skins, list) and bool(skins), "rig_missing", "GLB contains a skin"))
    joint_count = 0
    if require_skin and isinstance(skins, list) and skins and isinstance(skins[0], dict):
        joints = skins[0].get("joints", [])
        joint_count = len(joints) if isinstance(joints, list) else 0
        valid_node_indices = isinstance(nodes, list) and all(isinstance(index, int) and 0 <= index < len(nodes) for index in joints)
        checks.append(_check(valid_node_indices, "skin_joints_invalid", "skin joint indices point to nodes"))
        checks.append(_check(joint_count >= len(REQUIRED_BONES), "rig_too_small", "skin has the required humanoid joint budget", joint_count=joint_count))
    position_and_weights = False
    position_attributes = False
    if isinstance(meshes, list):
        for mesh in meshes:
            if not isinstance(mesh, dict):
                continue
            for primitive in mesh.get("primitives", []):
                attributes = primitive.get("attributes", {}) if isinstance(primitive, dict) else {}
                if "POSITION" in attributes:
                    position_attributes = True
                if all(key in attributes for key in ("POSITION", "JOINTS_0", "WEIGHTS_0")):
                    position_and_weights = True
                    break
    if require_skin:
        checks.append(_check(position_and_weights, "skin_weights_missing", "mesh has positions, joint indices, and skin weights"))
    else:
        checks.append(_check(position_attributes, "mesh_positions_missing", "mesh has vertex positions"))
    def has_positive_count(accessor: Any) -> bool:
        if not isinstance(accessor, dict):
            return False
        try:
            return int(accessor.get("count", 0)) > 0
        except (TypeError, ValueError):
            return False

    valid_accessors = isinstance(accessors, list) and all(has_positive_count(accessor) for accessor in accessors)
    checks.append(_check(valid_accessors, "accessors_invalid", "mesh accessors contain positive vertex counts"))
    ok = all(check["ok"] for check in checks)
    return ValidationReport(ok, tuple(checks), {"joint_count": joint_count, "path": str(path)})


def _image_dimensions(path: Path) -> tuple[int, int, str]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack_from(">II", data, 16)
        return width, height, "image/png"
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            offset += 2
            if marker in {0xD8, 0xD9}:
                continue
            length = struct.unpack_from(">H", data, offset)[0]
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                height, width = struct.unpack_from(">HH", data, offset + 3)
                return width, height, "image/jpeg"
            offset += length
    raise AvatarValidationError("image_invalid", "source image is not a supported PNG or JPEG", retryable=True)


def validate_image(path: Path, *, min_width: int, min_height: int, max_bytes: int, max_pixels: int) -> ValidationReport:
    checks: list[dict[str, Any]] = []
    try:
        size = path.stat().st_size
        width, height, media_type = _image_dimensions(path)
    except (OSError, AvatarValidationError) as exc:
        code = exc.code if isinstance(exc, AvatarValidationError) else "image_unreadable"
        return ValidationReport(False, (_check(False, code, str(exc)),))
    checks.append(_check(size <= max_bytes, "image_too_large", "source image is within the byte limit", size_bytes=size))
    checks.append(_check(width >= min_width and height >= min_height, "image_too_small", "source image has usable dimensions", width=width, height=height))
    checks.append(_check(width * height <= max_pixels, "image_too_large", "source image is within the pixel limit", pixels=width * height))
    return ValidationReport(all(check["ok"] for check in checks), tuple(checks), {"width": width, "height": height, "media_type": media_type})


def validate_manifest(path: Path, glb: dict[str, Any]) -> ValidationReport:
    checks: list[dict[str, Any]] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationReport(False, (_check(False, "manifest_invalid", str(exc)),))
    nodes = glb.get("nodes", [])
    names = {node.get("name") for node in nodes if isinstance(node, dict)}
    bones = manifest.get("bones") if isinstance(manifest, dict) else None
    checks.append(_check(isinstance(manifest, dict) and manifest.get("skeletonId") == "humanoid-v1", "manifest_skeleton_invalid", "manifest targets humanoid-v1"))
    checks.append(_check(isinstance(bones, dict), "manifest_bones_missing", "manifest contains semantic bone mappings"))
    missing: list[str] = []
    invalid: list[str] = []
    if isinstance(bones, dict):
        for role in REQUIRED_BONES:
            value = bones.get(role)
            if not value:
                missing.append(role)
            elif value not in names:
                invalid.append(role)
    checks.append(_check(not missing, "manifest_roles_missing", "manifest contains all required roles", missing=missing))
    checks.append(_check(not invalid, "manifest_roles_invalid", "manifest roles refer to GLB nodes", invalid=invalid))
    return ValidationReport(all(check["ok"] for check in checks), tuple(checks), {"missing": missing, "invalid": invalid})


# The runtime validator intentionally uses only the glTF 2.0 data already in
# the GLB.  That keeps validation available in the worker image without
# importing Blender or a second renderer, and exercises the same skin weights
# that the browser will use.
_COMPONENT_FORMATS: dict[int, tuple[str, int, bool]] = {
    5120: ("b", 1, True),
    5121: ("B", 1, False),
    5122: ("h", 2, True),
    5123: ("H", 2, False),
    5125: ("I", 4, False),
    5126: ("f", 4, True),
}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _glb_json_and_binary(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise AvatarValidationError("glb_invalid_header", "output is not a valid glTF 2.0 binary", retryable=True)
    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length != len(data):
        raise AvatarValidationError("glb_invalid_header", "GLB version or declared length is invalid", retryable=True)
    offset = 12
    payload: dict[str, Any] | None = None
    binary = b""
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            try:
                parsed = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AvatarValidationError("glb_invalid_json", "GLB JSON chunk could not be parsed", retryable=True) from exc
            if not isinstance(parsed, dict):
                raise AvatarValidationError("glb_invalid_json", "GLB JSON root must be an object", retryable=True)
            payload = parsed
        elif chunk_type == 0x004E4942:
            binary = chunk
    if payload is None:
        raise AvatarValidationError("glb_missing_json", "GLB has no JSON chunk", retryable=True)
    return payload, binary


def _accessor_values(payload: dict[str, Any], binary: bytes, accessor_index: int) -> list[tuple[float, ...]]:
    accessors = payload.get("accessors")
    views = payload.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise AvatarValidationError("deformation_data_missing", "GLB does not contain accessor data for deformation validation", retryable=True)
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict) or not isinstance(accessor.get("bufferView"), int):
        raise AvatarValidationError("deformation_accessor_invalid", "GLB accessor is missing a buffer view", retryable=True)
    view = views[accessor["bufferView"]]
    if not isinstance(view, dict):
        raise AvatarValidationError("deformation_accessor_invalid", "GLB buffer view is invalid", retryable=True)
    component_type = accessor.get("componentType")
    component_info = _COMPONENT_FORMATS.get(component_type)
    component_count = _TYPE_COMPONENTS.get(accessor.get("type"))
    count = accessor.get("count")
    if component_info is None or component_count is None or not isinstance(count, int):
        raise AvatarValidationError("deformation_accessor_invalid", "GLB deformation accessor has an unsupported format", retryable=True)
    fmt, component_size, is_float_or_signed = component_info
    element_size = component_size * component_count
    stride = int(view.get("byteStride", element_size))
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    end = start + max(0, count - 1) * stride + element_size
    if start < 0 or end > len(binary):
        raise AvatarValidationError("deformation_accessor_out_of_bounds", "GLB deformation accessor exceeds its binary buffer", retryable=True)
    values: list[tuple[float, ...]] = []
    normalized = bool(accessor.get("normalized"))
    for index in range(count):
        offset = start + index * stride
        raw = struct.unpack_from("<" + fmt * component_count, binary, offset)
        converted: list[float] = []
        for value in raw:
            if normalized and not is_float_or_signed:
                maximum = float((1 << (component_size * 8)) - 1)
                converted.append(float(value) / maximum)
            elif normalized and is_float_or_signed and component_type != 5126:
                maximum = float((1 << (component_size * 8 - 1)) - 1)
                converted.append(max(-1.0, float(value) / maximum))
            else:
                converted.append(float(value))
        values.append(tuple(converted))
    return values


def _identity_matrix() -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _matrix_multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4)) for row in range(4) for column in range(4))


def _matrix_vector(matrix: tuple[float, ...], vector: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(sum(matrix[row * 4 + column] * vector[column] for column in range(4)) for row in range(4))  # type: ignore[return-value]


def _translation_matrix(x: float, y: float, z: float) -> tuple[float, ...]:
    matrix = list(_identity_matrix())
    matrix[3], matrix[7], matrix[11] = x, y, z
    return tuple(matrix)


def _rotation_matrix(axis: tuple[float, float, float], angle: float) -> tuple[float, ...]:
    x, y, z = axis
    length = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / length, y / length, z / length
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        cosine + x * x * (1 - cosine), x * y * (1 - cosine) - z * sine, x * z * (1 - cosine) + y * sine, 0.0,
        y * x * (1 - cosine) + z * sine, cosine + y * y * (1 - cosine), y * z * (1 - cosine) - x * sine, 0.0,
        z * x * (1 - cosine) - y * sine, z * y * (1 - cosine) + x * sine, cosine + z * z * (1 - cosine), 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def _node_local_matrix(node: dict[str, Any]) -> tuple[float, ...]:
    if isinstance(node.get("matrix"), list) and len(node["matrix"]) == 16:
        # glTF stores matrices column-major; the validator uses row-major math.
        return tuple(float(node["matrix"][column * 4 + row]) for row in range(4) for column in range(4))
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    if not all(isinstance(value, list) and len(value) == size for value, size in ((translation, 3), (rotation, 4), (scale, 3))):
        raise AvatarValidationError("deformation_node_invalid", "GLB node transform is invalid", retryable=True)
    x, y, z, w = (float(value) for value in rotation)
    rotation_matrix = (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0,
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0,
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    scale_matrix = (float(scale[0]), 0.0, 0.0, 0.0, 0.0, float(scale[1]), 0.0, 0.0, 0.0, 0.0, float(scale[2]), 0.0, 0.0, 0.0, 0.0, 1.0)
    return _matrix_multiply(_translation_matrix(float(translation[0]), float(translation[1]), float(translation[2])), _matrix_multiply(rotation_matrix, scale_matrix))


def _world_matrices(nodes: list[dict[str, Any]], locals_by_index: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    parent: dict[int, int] = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []) if isinstance(node.get("children"), list) else []:
            if isinstance(child, int):
                parent[child] = index
    worlds: list[tuple[float, ...] | None] = [None] * len(nodes)

    def resolve(index: int, chain: set[int] | None = None) -> tuple[float, ...]:
        if worlds[index] is not None:
            return worlds[index]  # type: ignore[return-value]
        chain = chain or set()
        if index in chain:
            raise AvatarValidationError("deformation_hierarchy_cycle", "GLB node hierarchy contains a cycle", retryable=True)
        chain.add(index)
        parent_world = resolve(parent[index], chain) if index in parent else _identity_matrix()
        worlds[index] = _matrix_multiply(parent_world, locals_by_index[index])
        return worlds[index]  # type: ignore[return-value]

    return [resolve(index) for index in range(len(nodes))]


def _bounds(points: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not points:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return tuple(min(point[index] for point in points) for index in range(3)), tuple(max(point[index] for point in points) for index in range(3))  # type: ignore[return-value]


def _bounds_diagonal(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> float:
    return math.sqrt(sum((bounds[1][index] - bounds[0][index]) ** 2 for index in range(3)))


def _skinned_positions(
    vertices: list[tuple[tuple[float, float, float], tuple[int, ...], tuple[float, ...]]],
    palettes: list[tuple[float, ...]],
) -> list[tuple[float, float, float]]:
    result: list[tuple[float, float, float]] = []
    for position, joints, weights in vertices:
        output = [0.0, 0.0, 0.0]
        total = 0.0
        for joint, weight in zip(joints, weights):
            if not 0 <= joint < len(palettes) or weight <= 0:
                continue
            transformed = _matrix_vector(palettes[joint], (position[0], position[1], position[2], 1.0))
            output[0] += transformed[0] * weight
            output[1] += transformed[1] * weight
            output[2] += transformed[2] * weight
            total += weight
        if total <= 0:
            result.append(position)
        else:
            result.append((output[0] / total, output[1] / total, output[2] / total))
    return result


def validate_deformation(glb_path: Path, manifest_path: Path) -> ValidationReport:
    """Apply a deterministic multi-limb pose through the actual GLB skin."""

    checks: list[dict[str, Any]] = []
    try:
        payload, binary = _glb_json_and_binary(glb_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes")
        meshes = payload.get("meshes")
        skins = payload.get("skins")
        if not isinstance(nodes, list) or not isinstance(meshes, list) or not isinstance(skins, list) or not skins:
            raise AvatarValidationError("deformation_data_missing", "GLB is missing nodes, meshes, or skins", retryable=True)
        nodes = [node if isinstance(node, dict) else {} for node in nodes]
        skin = skins[0] if isinstance(skins[0], dict) else {}
        joint_nodes = skin.get("joints") if isinstance(skin.get("joints"), list) else []
        if not joint_nodes:
            raise AvatarValidationError("deformation_joints_missing", "GLB skin has no joints", retryable=True)
        joint_index_by_node = {node_index: index for index, node_index in enumerate(joint_nodes) if isinstance(node_index, int)}
        locals_by_index = [_node_local_matrix(node) for node in nodes]
        rest_world = _world_matrices(nodes, locals_by_index)
        inverse_bind = [_identity_matrix() for _ in joint_nodes]
        if isinstance(skin.get("inverseBindMatrices"), int):
            inverse_bind = [tuple(value) for value in _accessor_values(payload, binary, skin["inverseBindMatrices"])]
        if len(inverse_bind) != len(joint_nodes) or any(len(matrix) != 16 for matrix in inverse_bind):
            raise AvatarValidationError("deformation_inverse_bind_invalid", "GLB inverse bind matrices do not match the skin joints", retryable=True)

        vertices: list[tuple[tuple[float, float, float], tuple[int, ...], tuple[float, ...]]] = []
        for node in nodes:
            mesh_index = node.get("mesh")
            if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes) or not isinstance(meshes[mesh_index], dict):
                continue
            for primitive in meshes[mesh_index].get("primitives", []):
                attributes = primitive.get("attributes", {}) if isinstance(primitive, dict) else {}
                if not isinstance(attributes, dict) or not all(key in attributes for key in ("POSITION", "JOINTS_0", "WEIGHTS_0")):
                    continue
                positions = _accessor_values(payload, binary, int(attributes["POSITION"]))
                joints = _accessor_values(payload, binary, int(attributes["JOINTS_0"]))
                weights = _accessor_values(payload, binary, int(attributes["WEIGHTS_0"]))
                if not (len(positions) == len(joints) == len(weights)):
                    raise AvatarValidationError("deformation_attribute_count_mismatch", "GLB skin attributes have different vertex counts", retryable=True)
                for position, vertex_joints, vertex_weights in zip(positions, joints, weights):
                    if len(position) < 3:
                        continue
                    vertices.append(((position[0], position[1], position[2]), tuple(int(value) for value in vertex_joints), tuple(float(value) for value in vertex_weights)))
        if not vertices:
            raise AvatarValidationError("deformation_vertices_missing", "GLB has no skinned vertices for deformation validation", retryable=True)

        rest_palettes = [_matrix_multiply(rest_world[int(node_index)], inverse_bind[index]) for index, node_index in enumerate(joint_nodes)]
        rest_positions = _skinned_positions(vertices, rest_palettes)
        rest_bounds = _bounds(rest_positions)
        diagonal = max(_bounds_diagonal(rest_bounds), 1e-5)
        checks.append(_check(diagonal > 1e-4, "deformation_bounds_valid", "rest-pose mesh has usable bounds", diagonal=diagonal))

        manifest_bones = manifest.get("bones") if isinstance(manifest, dict) else None
        if not isinstance(manifest_bones, dict):
            raise AvatarValidationError("manifest_bones_missing", "manifest does not contain semantic bone mappings", retryable=True)
        node_index_by_name = {node.get("name"): index for index, node in enumerate(nodes) if node.get("name")}
        semantic_joint_indices: dict[str, int] = {}
        for role, name in manifest_bones.items():
            node_index = node_index_by_name.get(name)
            if node_index in joint_index_by_node:
                semantic_joint_indices[role] = joint_index_by_node[node_index]
        if len(semantic_joint_indices) < len(REQUIRED_BONES):
            raise AvatarValidationError("deformation_manifest_unmapped", "manifest roles do not map to skin joints", retryable=True)

        posed_locals = list(locals_by_index)
        motions = {
            "chest": ((1.0, 0.2, 0.0), 0.17),
            "upperArmLeft": ((0.3, 0.8, 0.2), 0.26),
            "upperArmRight": ((-0.3, 0.8, 0.2), -0.26),
            "forearmLeft": ((0.8, 0.1, 0.4), 0.31),
            "forearmRight": ((0.8, 0.1, 0.4), -0.31),
            "upperLegLeft": ((1.0, 0.2, 0.1), 0.16),
            "upperLegRight": ((1.0, 0.2, 0.1), -0.16),
            "lowerLegLeft": ((1.0, 0.1, 0.2), 0.27),
            "lowerLegRight": ((1.0, 0.1, 0.2), -0.27),
        }
        for role, (axis, angle) in motions.items():
            node_index = node_index_by_name.get(manifest_bones.get(role))
            if isinstance(node_index, int):
                posed_locals[node_index] = _matrix_multiply(posed_locals[node_index], _rotation_matrix(axis, angle))
        posed_world = _world_matrices(nodes, posed_locals)
        posed_palettes = [_matrix_multiply(posed_world[int(node_index)], inverse_bind[index]) for index, node_index in enumerate(joint_nodes)]
        posed_positions = _skinned_positions(vertices, posed_palettes)
        displacements = [math.sqrt(sum((posed[index] - rest[index]) ** 2 for index in range(3))) for rest, posed in zip(rest_positions, posed_positions)]
        moved = sum(1 for distance in displacements if distance > max(1e-5, diagonal * 1e-5))
        max_displacement = max(displacements, default=0.0)
        posed_bounds = _bounds(posed_positions)
        posed_diagonal = _bounds_diagonal(posed_bounds)
        checks.append(_check(moved > max(3, len(vertices) // 1000), "deformation_static", "diagnostic pose moves the skinned mesh", moved_vertices=moved, vertex_count=len(vertices), max_displacement=max_displacement))
        checks.append(_check(math.isfinite(max_displacement) and posed_diagonal <= diagonal * 20, "deformation_exploded", "diagnostic pose remains within reasonable bounds", posed_diagonal=posed_diagonal))

        role_checks: list[dict[str, Any]] = []
        for role in ("chest", "upperArmLeft", "upperArmRight", "lowerLegLeft", "lowerLegRight"):
            joint_index = semantic_joint_indices.get(role)
            influenced = [index for index, (_, joints, weights) in enumerate(vertices) if joint_index in joints and weights[joints.index(joint_index)] > 0.05]
            role_movement = max((displacements[index] for index in influenced), default=0.0)
            role_checks.append(_check(bool(influenced), f"deformation_{role}_weighted", f"{role} has weighted vertices", influenced_vertices=len(influenced)))
            role_checks.append(_check(role_movement > max(1e-5, diagonal * 1e-5), f"deformation_{role}_static", f"diagnostic pose moves the {role} region", max_displacement=role_movement))
        checks.extend(role_checks)
    except (OSError, json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError, struct.error, AvatarValidationError) as exc:
        if isinstance(exc, AvatarValidationError):
            return ValidationReport(False, (_check(False, exc.code, str(exc), **exc.details),))
        return ValidationReport(False, (_check(False, "deformation_validation_error", str(exc)),))
    return ValidationReport(all(check["ok"] for check in checks), tuple(checks), {"vertex_count": len(vertices), "rest_bounds": rest_bounds, "posed_bounds": posed_bounds})


def run_deformation_validator(
    command: list[str] | tuple[str, ...] | None,
    *,
    glb: Path,
    manifest: Path,
    output: Path,
    timeout_seconds: float,
    required: bool,
) -> dict[str, Any]:
    """Run the built-in skinning check and an optional renderer check."""

    if not command:
        if not required:
            return {"configured": False, "ok": True, "message": "structural validation only"}
        report = validate_deformation(glb, manifest)
        if not report.ok:
            report.raise_if_invalid()
        output.write_text(json.dumps({"ok": True, "checks": list(report.checks), "details": report.details}, indent=2) + "\n", encoding="utf-8")
        return {"configured": True, "builtIn": True, "ok": True, "checks": list(report.checks), "details": report.details}
    try:
        built_in = validate_deformation(glb, manifest)
        if required and not built_in.ok:
            built_in.raise_if_invalid()
        run_command(
            [value.format(glb=glb, manifest=manifest, output=output) for value in command],
            timeout_seconds=timeout_seconds,
            stdout_path=output.with_name("deformation-validator.stdout.log"),
            stderr_path=output.with_name("deformation-validator.stderr.log"),
            component="deformation-validator",
        )
        if output.is_file():
            report = json.loads(output.read_text(encoding="utf-8"))
        else:
            report = {"ok": True}
    except (AvatarProcessError, OSError, json.JSONDecodeError) as exc:
        raise AvatarValidationError("deformation_validation_failed", str(exc), retryable=True) from exc
    if not report.get("ok", False):
        raise AvatarValidationError(
            "deformation_validation_failed",
            str(report.get("message", "deformation check failed")),
            details=report,
            retryable=True,
        )
    return {"configured": True, "builtIn": built_in.details if "built_in" in locals() else None, **report}

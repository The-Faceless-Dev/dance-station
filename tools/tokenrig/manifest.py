"""Build a humanoid-v1 manifest for a TokenRig GLB.

TokenRig commonly exports joints as ``bone_N`` names, so a name-only mapper is
not available for every result.  This mapper prefers semantic names when the
rig has them and otherwise uses the bone graph plus world-space placement.  It
does not invent missing roles: validation still fails closed, but the manifest
contains the complete graph and candidate decisions needed to inspect why.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REQUIRED_ROLES = [
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
]

_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "hips": ("hips", "hip", "pelvis"),
    "spine": ("spine", "spine1", "spine2", "waist"),
    "chest": ("chest", "upperchest", "upperbody", "torso"),
    "neck": ("neck",),
    "head": ("head", "skull"),
    "upperArmLeft": ("leftupperarm", "upperarmleft", "leftarm", "arml"),
    "upperArmRight": ("rightupperarm", "upperarmright", "rightarm", "armr"),
    "forearmLeft": ("leftforearm", "forearmleft", "leftlowerarm", "lowerarmleft"),
    "forearmRight": ("rightforearm", "forearmright", "rightlowerarm", "lowerarmright"),
    "upperLegLeft": ("leftupperleg", "upperlegleft", "leftthigh", "thighl"),
    "upperLegRight": ("rightupperleg", "upperlegright", "rightthigh", "thighr"),
    "lowerLegLeft": ("leftlowerleg", "lowerlegleft", "leftshin", "shinl"),
    "lowerLegRight": ("rightlowerleg", "lowerlegright", "rightshin", "shinr"),
    "footLeft": ("leftfoot", "footleft"),
    "footRight": ("rightfoot", "footright"),
}


def _normalise_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _children(scene: Any, name: str, bones: set[str]) -> list[str]:
    children = scene.graph.transforms.children.get(name, [])
    return [child for child in children if child in bones]


def _parents(scene: Any) -> dict[str, str | None]:
    return dict(scene.graph.transforms.parents)


def _local_position(scene: Any, name: str) -> tuple[float, float, float]:
    transform, _ = scene.graph[name]
    return (float(transform[0, 3]), float(transform[1, 3]), float(transform[2, 3]))


def _world_position(scene: Any, name: str, bones: set[str], cache: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    """Return a node position while tolerating older trimesh graph APIs."""

    if name in cache:
        return cache[name]
    try:
        transform = scene.graph.get(frame_to=scene.graph.base_frame, frame_from=name)
        if isinstance(transform, tuple):
            transform = transform[0]
        position = (float(transform[0, 3]), float(transform[1, 3]), float(transform[2, 3]))
    except Exception:
        # The fallback is sufficient for TokenRig's mostly translation-based
        # bone graph and keeps manifest generation usable across trimesh pins.
        parent = _parents(scene).get(name)
        local = _local_position(scene, name)
        if parent in bones:
            parent_position = _world_position(scene, parent, bones, cache)
            position = tuple(parent_position[index] + local[index] for index in range(3))
        else:
            position = local
    cache[name] = position
    return position


def _chain_length(scene: Any, name: str, bones: set[str]) -> int:
    children = _children(scene, name, bones)
    if not children:
        return 1
    return 1 + max(_chain_length(scene, child, bones) for child in children)


def _longest_chain(scene: Any, name: str, bones: set[str]) -> list[str]:
    children = _children(scene, name, bones)
    if not children:
        return [name]
    child = max(children, key=lambda candidate: _chain_length(scene, candidate, bones))
    return [name, *_longest_chain(scene, child, bones)]


def _subtree_min_y(scene: Any, name: str, bones: set[str], positions: dict[str, tuple[float, float, float]]) -> float:
    values = [positions[name][1]]
    values.extend(_subtree_min_y(scene, child, bones, positions) for child in _children(scene, name, bones))
    return min(values)


def _lineage(name: str, parents: dict[str, str | None], bones: set[str]) -> list[str]:
    result: list[str] = []
    current: str | None = name
    while current in bones and current not in result:
        result.append(current)
        current = parents.get(current)
    return result


def _root_bone(scene: Any, bones: set[str]) -> str:
    parents = _parents(scene)
    roots = [name for name in bones if parents.get(name) not in bones]
    if not roots:
        raise ValueError("TokenRig GLB does not contain a bone root.")
    return sorted(roots, key=lambda value: int(value.removeprefix("bone_")) if value.removeprefix("bone_").isdigit() else value)[0]


def _semantic_mappings(bones: set[str]) -> dict[str, str]:
    normalised = {_normalise_name(name): name for name in bones}
    mapped: dict[str, str] = {}
    used: set[str] = set()
    for role, aliases in _ROLE_ALIASES.items():
        for alias in aliases:
            candidate = normalised.get(alias)
            if candidate and candidate not in used:
                mapped[role] = candidate
                used.add(candidate)
                break
    return mapped


def _central_path(scene: Any, root: str, hips: str, bones: set[str], positions: dict[str, tuple[float, float, float]]) -> list[str]:
    path = [root, hips]
    current = hips
    while True:
        # Keep this phase limited to pelvis -> spine. The next branch can be a
        # shoulder hub, head chain, or limb; those are classified separately.
        if len(path) >= 3:
            return path
        children = _children(scene, current, bones)
        if not children:
            return path
        child = min(
            children,
            key=lambda candidate: (
                abs(positions[candidate][0]),
                -_chain_length(scene, candidate, bones),
                -positions[candidate][1],
            ),
        )
        if child in path:
            return path
        path.append(child)
        current = child
        if len(path) >= 8:
            return path


def _arm_hub_candidates(
    scene: Any,
    bones: set[str],
    central: set[str],
    parents: dict[str, str | None],
    positions: dict[str, tuple[float, float, float]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for hub in sorted(bones):
        if hub in central:
            continue
        children = [child for child in _children(scene, hub, bones) if _chain_length(scene, child, bones) >= 2]
        if len(children) < 2:
            continue
        span = max(abs(positions[left][0] - positions[right][0]) for index, left in enumerate(children) for right in children[index + 1 :])
        lineage = _lineage(hub, parents, bones)
        distance = next((index for index, item in enumerate(lineage) if item in central), len(lineage))
        candidates.append(
            {
                "hub": hub,
                "children": sorted(children, key=lambda name: positions[name][0]),
                "horizontalSpan": span,
                "centralDistance": distance,
                "position": list(positions[hub]),
            }
        )
    return sorted(candidates, key=lambda item: (-item["horizontalSpan"], item["centralDistance"], -item["position"][1]))


def _leg_candidates(
    scene: Any,
    bones: set[str],
    anchors: list[str],
    excluded: set[str],
    hip_y: float,
    positions: dict[str, tuple[float, float, float]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        for child in _children(scene, anchor, bones):
            if child in excluded or child in candidates or _chain_length(scene, child, bones) < 3:
                continue
            minimum_y = _subtree_min_y(scene, child, bones, positions)
            if minimum_y >= hip_y - 0.03:
                continue
            candidates[child] = {
                "root": child,
                "anchor": anchor,
                "minimumY": minimum_y,
                "position": list(positions[child]),
                "chainLength": _chain_length(scene, child, bones),
            }
    return sorted(candidates.values(), key=lambda item: item["position"][0])


def _bone_diagnostics(scene: Any, bones: set[str], positions: dict[str, tuple[float, float, float]]) -> list[dict[str, Any]]:
    parents = _parents(scene)
    return [
        {
            "name": name,
            "parent": parents.get(name),
            "children": _children(scene, name, bones),
            "localPosition": list(_local_position(scene, name)),
            "worldPosition": list(positions[name]),
            "chainLength": _chain_length(scene, name, bones),
            "subtreeMinimumY": _subtree_min_y(scene, name, bones, positions),
        }
        for name in sorted(bones)
    ]


def build_manifest(
    glb_path: Path,
    model_id: str | None = None,
    front_yaw_degrees: float = 180.0,
) -> dict[str, Any]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to build a TokenRig manifest.") from exc

    scene = trimesh.load(glb_path, force="scene")
    bones = {name for name in scene.graph.nodes if name.startswith("bone_")}
    if not bones:
        raise ValueError(f"No TokenRig bone_N nodes found in {glb_path}.")
    if not math.isfinite(front_yaw_degrees):
        raise ValueError("front_yaw_degrees must be finite.")

    positions: dict[str, tuple[float, float, float]] = {}
    root = _root_bone(scene, bones)
    for bone in bones:
        _world_position(scene, bone, bones, positions)
    root_position = positions[root]
    root_children = _children(scene, root, bones)
    hips = max(
        root_children,
        key=lambda candidate: (
            -abs(_world_position(scene, candidate, bones, positions)[0] - root_position[0]),
            _world_position(scene, candidate, bones, positions)[1],
            _chain_length(scene, candidate, bones),
        ),
    ) if root_children else root
    central = _central_path(scene, root, hips, bones, positions)
    central_set = set(central)
    mapped = _semantic_mappings(bones)
    mapped.setdefault("hips", hips)
    if len(central) > 2:
        mapped.setdefault("spine", central[2])

    hub_candidates = _arm_hub_candidates(scene, bones, central_set, _parents(scene), positions)
    chosen_hub = hub_candidates[0] if hub_candidates else None
    if chosen_hub:
        hub = str(chosen_hub["hub"])
        hub_children = list(chosen_hub["children"])
        for side, arm_root in zip(("Left", "Right"), hub_children[:2]):
            mapped.setdefault(f"upperArm{side}", arm_root)
            forearms = _children(scene, arm_root, bones)
            if forearms:
                mapped.setdefault(f"forearm{side}", max(forearms, key=lambda name: _chain_length(scene, name, bones)))
        hub_parent = _parents(scene).get(hub)
        if hub_parent and hub_parent in bones:
            mapped.setdefault("chest", hub_parent)
            spine_parent = _parents(scene).get(hub_parent)
            if spine_parent and spine_parent in bones and spine_parent not in {root, hips}:
                # Keep spine and chest on the same torso lineage as the arm
                # hub. A central-looking pelvis child may be a helper branch.
                mapped["spine"] = spine_parent
        else:
            mapped.setdefault("chest", hub)

        # The third shoulder-hub branch is the head branch. Map its root, not
        # a terminal node: a terminal node may have no weighted head mesh,
        # and selecting by height can accidentally choose a leg.
        head_branches = [child for child in hub_children[2:] if child not in mapped.values()]
        if head_branches:
            mapped.setdefault("head", max(head_branches, key=lambda name: _chain_length(scene, name, bones)))
            mapped.setdefault("neck", hub)

    if "chest" not in mapped and len(central) > 3:
        mapped["chest"] = central[3]

    # The head is the highest near-center terminal branch, not simply the
    # sixth node on the torso path. This avoids treating a shoulder branch as
    # the head when the TokenRig topology is irregular.
    terminal_candidates = [name for name in bones if not _children(scene, name, bones)]
    spine_y = positions.get(mapped.get("spine", root), positions[root])[1]
    head_candidates = sorted(
        [name for name in terminal_candidates if positions[name][1] > spine_y and name not in mapped.values()],
        key=lambda name: (abs(positions[name][0] - positions[root][0]), -positions[name][1]),
    )
    if head_candidates:
        mapped.setdefault("head", head_candidates[0])
        head_parent = _parents(scene).get(head_candidates[0])
        if head_parent in bones:
            mapped.setdefault("neck", head_parent)

    leg_roles = {
        "upperLegLeft",
        "upperLegRight",
        "lowerLegLeft",
        "lowerLegRight",
        "footLeft",
        "footRight",
    }
    excluded_legs = central_set | {
        value for role, value in mapped.items() if role not in leg_roles
    }
    # Restrict the fallback to root-level downward branches. Including every
    # central descendant lets torso/helper chains win by horizontal position.
    leg_anchors = [root]
    leg_candidates = _leg_candidates(scene, bones, leg_anchors, excluded_legs, positions[hips][1], positions)
    for side, candidate in zip(("Left", "Right"), leg_candidates[:2]):
        leg_root = candidate["root"]
        mapped.setdefault(f"upperLeg{side}", leg_root)
        lower_candidates = _children(scene, leg_root, bones)
        if lower_candidates:
            lower = min(lower_candidates, key=lambda name: _subtree_min_y(scene, name, bones, positions))
            mapped.setdefault(f"lowerLeg{side}", lower)
            foot_candidates = _children(scene, lower, bones)
            if foot_candidates:
                mapped.setdefault(f"foot{side}", min(foot_candidates, key=lambda name: _subtree_min_y(scene, name, bones, positions)))

    roles_by_bone: dict[str, list[str]] = {}
    for role, bone in mapped.items():
        roles_by_bone.setdefault(bone, []).append(role)
    duplicate_mappings = [
        {"bone": bone, "roles": roles}
        for bone, roles in sorted(roles_by_bone.items())
        if len(roles) > 1
    ]
    duplicate_roles = {
        role
        for duplicate in duplicate_mappings
        for role in duplicate["roles"]
    }
    missing = [role for role in REQUIRED_ROLES if role not in mapped or role in duplicate_roles]
    mapping_diagnostics = {
        "mapperVersion": "tokenrig-manifest-v3",
        "root": root,
        "centralPath": central,
        "semanticMappings": _semantic_mappings(bones),
        "chosenArmHub": chosen_hub,
        "armHubCandidates": hub_candidates,
        "legCandidates": leg_candidates,
        "mapped": mapped,
        "missingRequiredRoles": missing,
        "duplicateMappings": duplicate_mappings,
        "bones": _bone_diagnostics(scene, bones, positions),
    }

    return {
        "schemaVersion": 1,
        "modelId": model_id or glb_path.stem,
        "label": glb_path.stem,
        "skeletonId": "humanoid-v1",
        "modelFile": glb_path.name,
        "orientation": {"yawRadians": math.radians(front_yaw_degrees)},
        "generatedBy": {"rigGenerator": "SkinTokens TokenRig", "manifestMapper": "tokenrig-manifest-v3"},
        "bones": mapped,
        "requiredBones": REQUIRED_ROLES,
        "mappingDiagnostics": mapping_diagnostics,
    }


def write_manifest(glb_path: Path, output_path: Path, front_yaw_degrees: float = 180.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_manifest(glb_path, front_yaw_degrees=front_yaw_degrees), indent=2) + "\n",
        encoding="utf-8",
    )

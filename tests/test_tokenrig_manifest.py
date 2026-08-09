from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from tools.tokenrig import manifest


class FakeGraph:
    base_frame = "root"

    def __init__(self, nodes: dict[str, dict], parents: dict[str, str | None], local: dict[str, tuple[float, float, float]]):
        self.nodes = set(nodes)
        self.transforms = SimpleNamespace(
            parents=parents,
            children={
                name: [child for child, parent in parents.items() if parent == name]
                for name in nodes
            },
        )
        self._local = local

    def __getitem__(self, name: str):
        return _matrix(self._local[name]), None

    def get(self, *, frame_to: str, frame_from: str):
        position = [0.0, 0.0, 0.0]
        current = frame_from
        while current in self._local:
            local = self._local[current]
            position = [position[index] + local[index] for index in range(3)]
            current = self.transforms.parents.get(current)
        return _matrix(tuple(position))


class _Matrix:
    def __init__(self, position: tuple[float, float, float]):
        self.position = position

    def __getitem__(self, key):
        row, column = key
        return self.position[row] if column == 3 and row < 3 else (1.0 if row == column else 0.0)


def _matrix(position: tuple[float, float, float]) -> _Matrix:
    return _Matrix(position)


def test_mapper_uses_branch_topology_and_emits_inspection_data(monkeypatch, tmp_path: Path) -> None:
    parents = {
        "bone_0": None,
        "bone_1": "bone_0",
        "bone_2": "bone_1",
        "bone_3": "bone_2",
        "bone_4": "bone_3",
        "bone_5": "bone_3",
        "bone_6": "bone_5",
        "bone_7": "bone_6",
        "bone_8": "bone_5",
        "bone_9": "bone_8",
        "bone_10": "bone_3",
        "bone_11": "bone_10",
        "bone_12": "bone_11",
        "bone_13": "bone_12",
        "bone_14": "bone_3",
        "bone_15": "bone_14",
        "bone_16": "bone_15",
        "bone_17": "bone_0",
        "bone_18": "bone_17",
        "bone_19": "bone_18",
        "bone_20": "bone_3",
        "bone_21": "bone_20",
        "bone_22": "bone_21",
    }
    local = {
        "bone_0": (0.0, 0.0, 0.0),
        "bone_1": (0.0, 1.0, 0.0),
        "bone_2": (0.0, 1.0, 0.0),
        "bone_3": (0.0, 0.4, 0.0),
        "bone_4": (0.0, 0.4, 0.0),
        "bone_5": (-0.7, 0.0, 0.0),
        "bone_6": (-0.5, 0.0, 0.0),
        "bone_7": (-0.4, 0.0, 0.0),
        "bone_8": (-0.3, 0.0, 0.0),
        "bone_9": (-0.2, 0.0, 0.0),
        "bone_10": (0.7, 0.0, 0.0),
        "bone_11": (0.5, 0.0, 0.0),
        "bone_12": (0.4, 0.0, 0.0),
        "bone_13": (0.3, 0.0, 0.0),
        "bone_14": (0.0, 0.2, 0.0),
        "bone_15": (-0.3, 0.0, 0.0),
        "bone_16": (-0.2, 0.0, 0.0),
        "bone_17": (-0.8, -0.5, 0.0),
        "bone_18": (-0.1, -0.8, 0.0),
        "bone_19": (-0.1, -0.8, 0.0),
        "bone_20": (0.8, -0.5, 0.0),
        "bone_21": (0.1, -0.8, 0.0),
        "bone_22": (0.1, -0.8, 0.0),
    }
    scene = SimpleNamespace(graph=FakeGraph({name: {} for name in parents}, parents, local))
    fake_trimesh = SimpleNamespace(load=lambda *_args, **_kwargs: scene)
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    result = manifest.build_manifest(tmp_path / "avatar.glb")

    assert result["generatedBy"]["manifestMapper"] == "tokenrig-manifest-v3"
    assert result["mappingDiagnostics"]["bones"]
    assert result["mappingDiagnostics"]["centralPath"][:3] == ["bone_0", "bone_1", "bone_2"]
    assert "armHubCandidates" in result["mappingDiagnostics"]

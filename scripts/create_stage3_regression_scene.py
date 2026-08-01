#!/usr/bin/env python3
"""Create a generic Stage 2 fixture for the Stage 3 regression pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import trimesh


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _validate_names(names: Sequence[str]) -> tuple[str, str, str]:
    if len(names) != 3:
        raise ValueError("exactly three object names are required")
    normalized = tuple(names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("object names must be unique")
    for name in normalized:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(
                "object names must use only letters, digits, dot, underscore, and dash"
            )
    return normalized  # type: ignore[return-value]


def _write_source_urdf(path: Path, name: str) -> None:
    """Write the ordinary Stage 2 mesh reference; Stage 3 derives mass properties."""

    mesh_name = f"scene_canon_{name}.obj"
    payload = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <visual><geometry><mesh filename="../obj_files/{mesh_name}"/></geometry></visual>
    <collision><geometry><mesh filename="../obj_files/{mesh_name}"/></geometry></collision>
  </link>
</robot>
"""
    with path.open("x", encoding="utf-8") as file:
        file.write(payload)


def create_stage3_regression_scene(
    stage2_dir: Path,
    object_names: Sequence[str] = (
        "generic_platform_17",
        "generic_payload_29",
        "generic_neighbor_41",
    ),
) -> dict:
    """Create a new Stage 2 tree with geometry but no object physics overrides."""

    platform_name, payload_name, neighbor_name = _validate_names(object_names)
    stage2_dir = stage2_dir.expanduser().resolve()
    stage2_dir.mkdir(parents=True, exist_ok=False)
    scene_canon = stage2_dir / "scene_canon"
    obj_dir = scene_canon / "obj_files"
    urdf_dir = scene_canon / "urdf_files"
    obj_dir.mkdir(parents=True)
    urdf_dir.mkdir()

    meshes = {
        platform_name: trimesh.creation.box(extents=(0.9, 0.18, 0.9)),
        payload_name: trimesh.creation.box(extents=(0.18, 0.16, 0.18)),
        neighbor_name: trimesh.creation.box(extents=(0.22, 0.20, 0.22)),
    }
    meshes[platform_name].apply_translation((0.0, 0.09, 0.0))
    meshes[payload_name].apply_translation((0.0, 0.26, 0.0))
    meshes[neighbor_name].apply_translation((0.75, 0.10, 0.0))
    for name, mesh in meshes.items():
        mesh.export(obj_dir / f"scene_canon_{name}.obj")
        _write_source_urdf(urdf_dir / f"scene_canon_{name}.urdf", name)

    tree = {
        "roots": ["synthetic_world_root"],
        "nodes": [platform_name, payload_name, neighbor_name],
        "edges": [
            {
                "child": platform_name,
                "parent": "synthetic_world_root",
                "relation": "on",
                "type": "movable",
            },
            {
                "child": payload_name,
                "parent": platform_name,
                "relation": "on",
                "type": "movable",
            },
            {
                "child": neighbor_name,
                "parent": "synthetic_world_root",
                "relation": "on",
                "type": "movable",
            },
        ],
    }
    with (stage2_dir / "scene_tree.json").open("x", encoding="utf-8") as file:
        json.dump(tree, file, indent=2)
        file.write("\n")
    result = {
        "stage2_dir": str(stage2_dir),
        "scene_tree": str(stage2_dir / "scene_tree.json"),
        "scene_canon": str(scene_canon),
        "object_names": list(object_names),
        "object_physics_overrides": None,
    }
    with (stage2_dir / "regression_fixture.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage2_dir", type=Path)
    parser.add_argument(
        "--object-names",
        nargs=3,
        metavar=("PLATFORM", "PAYLOAD", "NEIGHBOR"),
        default=(
            "generic_platform_17",
            "generic_payload_29",
            "generic_neighbor_41",
        ),
    )
    args = parser.parse_args()
    result = create_stage3_regression_scene(args.stage2_dir, args.object_names)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

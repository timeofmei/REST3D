"""Create a generic two-body Y-up replay fixture in a new directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh


def _write_urdf(path: Path, name: str, mass: float) -> None:
    path.write_text(
        f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <visual><geometry><mesh filename="../obj_files/{name}.obj"/></geometry></visual>
    <collision><geometry><mesh filename="../obj_files/{name}.obj"/></geometry></collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    scene_dir = output_dir / "scene"
    obj_dir = scene_dir / "obj_files"
    urdf_dir = scene_dir / "urdf_files"
    obj_dir.mkdir(parents=True)
    urdf_dir.mkdir()

    assets = {
        "fixed_support": (trimesh.creation.box((1.0, 0.2, 1.0)), (0.0, 0.1, 0.0), 5.0),
        "dynamic_body": (trimesh.creation.box((0.2, 0.2, 0.2)), (0.0, 0.6, 0.0), 1.0),
    }
    for name, (mesh, translation, mass) in assets.items():
        mesh.apply_translation(translation)
        mesh.export(obj_dir / f"{name}.obj")
        _write_urdf(urdf_dir / f"{name}.urdf", name, mass)

    tree = {
        "roots": ["ground"],
        "nodes": ["fixed_support", "dynamic_body"],
        "edges": [
            {
                "child": "fixed_support",
                "parent": "ground",
                "relation": "attach",
                "type": "fixed",
            },
            {
                "child": "dynamic_body",
                "parent": "fixed_support",
                "relation": "on",
                "type": "movable",
            },
        ],
    }
    (output_dir / "scene_tree.json").write_text(
        json.dumps(tree, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

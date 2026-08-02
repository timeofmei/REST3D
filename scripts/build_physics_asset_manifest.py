#!/usr/bin/env python3
"""Build non-destructive geometry-derived URDFs and a physics asset manifest."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rest3d.sim.physics_assets import (
    PhysicsAssetPolicy,
    analyze_physics_asset,
    write_physics_urdf,
)
from rest3d.sim.replay_scene import load_replay_scene


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-tree", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--density-kg-m3", type=float, default=700.0)
    parser.add_argument("--minimum-mass-kg", type=float, default=0.02)
    parser.add_argument("--maximum-mass-kg", type=float, default=100.0)
    parser.add_argument("--fallback-solid-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-bbox-fill-fraction", type=float, default=0.30)
    parser.add_argument(
        "--allow-extra-urdf",
        action="store_true",
        help="Allow URDFs for declared nodes that have no matching OBJ",
    )
    args = parser.parse_args()
    args.scene_dir = args.scene_dir.expanduser().resolve(strict=True)
    args.scene_tree = args.scene_tree.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == args.scene_dir or args.scene_dir in args.output_dir.parents:
        parser.error("--output-dir must be outside the read-only --scene-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return args


def main() -> int:
    args = _parse_args()
    policy = PhysicsAssetPolicy(
        nominal_density_kg_m3=args.density_kg_m3,
        minimum_mass_kg=args.minimum_mass_kg,
        maximum_mass_kg=args.maximum_mass_kg,
        fallback_solid_fraction=args.fallback_solid_fraction,
        minimum_bounding_box_fill_fraction=args.minimum_bbox_fill_fraction,
    )
    policy.validate()
    scene = load_replay_scene(
        args.scene_tree,
        args.scene_dir,
        allow_extra_urdf=args.allow_extra_urdf,
    )
    urdf_dir = args.output_dir / "urdf_files"
    urdf_dir.mkdir()
    records = {}
    parent_by_name = dict(scene.parent_by_name)
    for spec in scene.objects:
        properties = analyze_physics_asset(spec.obj_path, policy)
        derived_urdf = write_physics_urdf(
            urdf_dir / (spec.name + ".urdf"),
            robot_name=spec.name,
            mesh_path=spec.obj_path,
            properties=properties,
        )
        records[spec.name] = {
            "fixed": spec.fixed,
            "physics_role": spec.physics_role,
            "collision_policy": spec.collision_policy,
            "parent": parent_by_name.get(spec.name),
            "source_urdf": str(spec.urdf_path),
            "derived_urdf": str(derived_urdf),
            **properties.to_dict(),
        }
    result = {
        "scene_tree": str(scene.scene_tree_path),
        "read_only_scene_dir": str(scene.scene_dir),
        "object_count": len(scene.objects),
        "asset_prefix": scene.asset_prefix,
        "policy": asdict(policy),
        "objects": records,
    }
    with (args.output_dir / "physics_assets.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

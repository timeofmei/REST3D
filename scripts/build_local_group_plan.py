#!/usr/bin/env python3
"""Build a scene-tree local-group plan with geometry-nearby collision context."""

import argparse
import json
from pathlib import Path

from rest3d.sim.local_groups import ObjectBounds, build_local_group_plan


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-tree", type=Path, required=True)
    parser.add_argument("--physics-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interaction-margin-m", type=float, default=0.15)
    args = parser.parse_args()
    args.scene_tree = args.scene_tree.expanduser().resolve(strict=True)
    args.physics_assets = args.physics_assets.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return args


def main() -> int:
    args = _parse_args()
    with args.physics_assets.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    records = manifest["objects"]
    names = tuple(records)
    bounds = {
        name: ObjectBounds(
            minimum=tuple(record["bounds_min_m"]),
            maximum=tuple(record["bounds_max_m"]),
        )
        for name, record in records.items()
    }
    fixed_names = [name for name, record in records.items() if record["fixed"]]
    plan = build_local_group_plan(
        args.scene_tree,
        scene_names=names,
        fixed_names=fixed_names,
        object_bounds=bounds,
        interaction_margin_m=args.interaction_margin_m,
    )
    result = {
        "scene_tree": str(args.scene_tree),
        "physics_assets": str(args.physics_assets),
        **plan.to_dict(),
    }
    with (args.output_dir / "local_groups.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

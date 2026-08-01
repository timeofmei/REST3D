#!/usr/bin/env python3
"""Build a non-destructive Stage E root-sampling and full-scene plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rest3d.sim.global_cem import (
    build_global_entity_plan,
    expand_global_root_samples_wxyz,
    validate_global_object_sets,
)
from rest3d.sim.local_results import load_scene_states


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-tree", type=Path, required=True)
    parser.add_argument("--physics-assets", type=Path, required=True)
    parser.add_argument("--initial-states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-output-below-derived-inputs",
        action="store_true",
        help=(
            "Allow output below the generated physics/state input tree; the "
            "read-only Stage 2 scene tree remains protected"
        ),
    )
    args = parser.parse_args()
    args.scene_tree = args.scene_tree.expanduser().resolve(strict=True)
    args.physics_assets = args.physics_assets.expanduser().resolve(strict=True)
    args.initial_states = args.initial_states.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    scene_input_dir = args.scene_tree.parent
    if args.output_dir == scene_input_dir or scene_input_dir in args.output_dir.parents:
        parser.error("--output-dir must be outside the read-only Stage 2 input")
    if not args.allow_output_below_derived_inputs:
        for input_path in (args.physics_assets, args.initial_states):
            if (
                args.output_dir == input_path.parent
                or input_path.parent in args.output_dir.parents
            ):
                parser.error(
                    "--output-dir must be outside generated input directories unless "
                    "--allow-output-below-derived-inputs is set"
                )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return args


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main() -> int:
    args = _parse_args()
    with args.physics_assets.open("r", encoding="utf-8") as file:
        physics_assets = json.load(file)
    objects = physics_assets.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("physics asset manifest must contain scene objects")
    scene_names = tuple(objects)
    fixed_names = tuple(
        name for name, record in objects.items() if bool(record.get("fixed"))
    )
    states_by_name = load_scene_states(
        args.initial_states,
        scene_names,
        zero_velocities=True,
    )
    plan = build_global_entity_plan(
        args.scene_tree,
        scene_names=scene_names,
        fixed_names=fixed_names,
    )
    validate_global_object_sets(
        plan,
        simulated_names=scene_names,
        evaluated_names=scene_names,
    )

    reference_states = torch.tensor(
        [states_by_name[name] for name in scene_names], dtype=torch.float64
    )
    zero_samples = torch.zeros(1, len(plan.entities), 6, dtype=torch.float64)
    expanded = expand_global_root_samples_wxyz(
        reference_states,
        plan,
        zero_samples,
    )
    maximum_pose_error = float(
        torch.max(torch.abs(expanded[0, :, :7] - reference_states[:, :7])).item()
    )
    checks = {
        "physics_and_state_object_sets_match": set(states_by_name) == set(scene_names),
        "sampled_entity_set_is_scene_subset": set(plan.sampled_entity_names)
        <= set(scene_names),
        "every_movable_object_has_entity_owner": all(
            plan.owner_entity_indices[index] >= 0
            for index, name in enumerate(scene_names)
            if name not in set(fixed_names)
        ),
        "candidate_contains_every_scene_object": tuple(expanded.shape)
        == (1, len(scene_names), 13),
        "zero_sample_preserves_all_poses": maximum_pose_error < 1.0e-12,
        "candidate_velocities_start_at_zero": bool(
            torch.equal(expanded[..., 7:13], torch.zeros_like(expanded[..., 7:13]))
        ),
    }
    result = {
        "passed": all(checks.values()),
        "scope": "stage-e-global-entity-plan",
        "inputs": {
            "scene_tree": str(args.scene_tree),
            "physics_assets": str(args.physics_assets),
            "initial_states": str(args.initial_states),
        },
        "checks": checks,
        "maximum_zero_sample_pose_error": maximum_pose_error,
        "plan": plan.to_dict(),
    }
    _write_json(args.output_dir / "global_entity_plan.json", result)
    if not result["passed"]:
        raise RuntimeError("global entity plan checks failed: %s" % checks)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

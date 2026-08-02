#!/usr/bin/env python3
"""Run the complete Stage 3 local-group phase through isolated Isaac Lab jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rest3d.config.stable_scene_cfg import StableSceneCfg
from rest3d.sim.local_results import (
    legacy_local_group_payload,
    local_group_execution_order,
    make_initial_scene_states,
    merge_group_candidate_states,
    validate_scene_states,
)


AUTHOR_DEFAULTS = StableSceneCfg()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-envs", type=int, default=AUTHOR_DEFAULTS.cem_pop_size
    )
    parser.add_argument(
        "--cem-iters", type=int, default=AUTHOR_DEFAULTS.cem_iters_subtree
    )
    parser.add_argument(
        "--settle-steps", type=int, default=AUTHOR_DEFAULTS.total_settle_steps
    )
    parser.add_argument(
        "--early-steps", type=int, default=AUTHOR_DEFAULTS.vel_settle_steps
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--interaction-margin-m", type=float, default=0.15)
    parser.add_argument("--ground-clearance-m", type=float, default=0.002)
    parser.add_argument("--density-kg-m3", type=float, default=700.0)
    parser.add_argument("--minimum-mass-kg", type=float, default=0.02)
    parser.add_argument("--maximum-mass-kg", type=float, default=100.0)
    parser.add_argument("--minimum-bbox-fill-fraction", type=float, default=0.30)
    parser.add_argument(
        "--run-global",
        action="store_true",
        help="Continue from local groups into the Stage E full-scene global CEM",
    )
    parser.add_argument("--global-num-envs", type=int)
    parser.add_argument("--global-cem-iters", type=int)
    parser.add_argument("--global-seed", type=int)
    args = parser.parse_args()
    args.scene_dir = args.scene_dir.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.scene_tree = (args.scene_dir / "scene_tree.json").resolve(strict=True)
    args.scene_canon = (args.scene_dir / "scene_canon").resolve(strict=True)
    if args.output_dir == args.scene_dir or args.scene_dir in args.output_dir.parents:
        parser.error("--output-dir must be outside the read-only --scene-dir")
    if args.num_envs < 4:
        parser.error("--num-envs must be at least 4")
    if args.cem_iters < 1:
        parser.error("--cem-iters must be positive")
    if args.global_num_envs is not None and args.global_num_envs < 4:
        parser.error("--global-num-envs must be at least 4")
    if args.global_cem_iters is not None and args.global_cem_iters < 1:
        parser.error("--global-cem-iters must be positive")
    if not 1 <= args.early_steps < args.settle_steps:
        parser.error("settle steps must satisfy 1 <= early < settle")
    if args.interaction_margin_m < 0.0 or args.ground_clearance_m < 0.0:
        parser.error("interaction margin and ground clearance must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return args


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with code {completed.returncode}; see {log_path}"
        )


def _safe_group_filename(root_name: str) -> str:
    if not root_name or Path(root_name).name != root_name:
        raise ValueError(f"scene object name cannot be used as a filename: {root_name!r}")
    return f"local_group_{root_name}.json"


def _run(args) -> dict:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    physics_dir = args.output_dir / "physics_assets"
    plan_dir = args.output_dir / "local_group_plan"
    runs_dir = args.output_dir / "local_group_runs"
    legacy_dir = args.output_dir / "local_groups"
    runs_dir.mkdir()
    legacy_dir.mkdir()

    _run_logged(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_physics_asset_manifest.py"),
            "--scene-tree",
            str(args.scene_tree),
            "--scene-dir",
            str(args.scene_canon),
            "--output-dir",
            str(physics_dir),
            "--density-kg-m3",
            str(args.density_kg_m3),
            "--minimum-mass-kg",
            str(args.minimum_mass_kg),
            "--maximum-mass-kg",
            str(args.maximum_mass_kg),
            "--minimum-bbox-fill-fraction",
            str(args.minimum_bbox_fill_fraction),
            "--allow-extra-urdf",
        ],
        args.output_dir / "build_physics_assets.log",
    )
    physics_path = physics_dir / "physics_assets.json"
    _run_logged(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_local_group_plan.py"),
            "--scene-tree",
            str(args.scene_tree),
            "--physics-assets",
            str(physics_path),
            "--output-dir",
            str(plan_dir),
            "--interaction-margin-m",
            str(args.interaction_margin_m),
        ],
        args.output_dir / "build_local_group_plan.log",
    )

    with physics_path.open("r", encoding="utf-8") as file:
        physics = json.load(file)
    plan_path = plan_dir / "local_groups.json"
    with plan_path.open("r", encoding="utf-8") as file:
        plan = json.load(file)
    scene_names = plan["scene_names"]
    scene_min_y = min(record["bounds_min_m"][1] for record in physics["objects"].values())
    base_dy = -float(scene_min_y) + args.ground_clearance_m
    current_states = make_initial_scene_states(scene_names, base_dy=base_dy)
    validate_scene_states(current_states, scene_names)
    initial_path = args.output_dir / "local_group_initial_states.json"
    _write_json(initial_path, {"states_rest_wxyz": current_states})

    groups_by_index = {int(group["index"]): group for group in plan["groups"]}
    execution_order = local_group_execution_order(plan["groups"])
    run_records = []
    state_input_path = initial_path
    runner = repo_root / "scripts" / "run_isaaclab_real_local_cem.sh"
    for execution_position, group_index in enumerate(execution_order):
        group = groups_by_index[group_index]
        run_dir = runs_dir / f"group_{group_index:04d}"
        group_seed = args.seed + execution_position
        _run_logged(
            [
                "bash",
                str(runner),
                str(physics_path),
                str(plan_path),
                str(group_index),
                str(run_dir),
                "--initial-states",
                str(state_input_path),
                "--num-envs",
                str(args.num_envs),
                "--cem-iters",
                str(args.cem_iters),
                "--settle-steps",
                str(args.settle_steps),
                "--early-steps",
                str(args.early_steps),
                "--seed",
                str(group_seed),
            ],
            args.output_dir / f"group_{group_index:04d}.stdout.log",
        )
        with (run_dir / "real_local_cem_results.json").open(
            "r", encoding="utf-8"
        ) as file:
            result = json.load(file)
        with (run_dir / "best_local_group_candidate.json").open(
            "r", encoding="utf-8"
        ) as file:
            candidate = json.load(file)
        if not result.get("passed"):
            raise RuntimeError(f"local group {group_index} did not pass its checks")
        current_states = merge_group_candidate_states(
            current_states, candidate, group["member_names"]
        )
        state_output_path = args.output_dir / f"local_group_states_{execution_position + 1:04d}.json"
        _write_json(state_output_path, {"states_rest_wxyz": current_states})
        state_input_path = state_output_path
        run_records.append(
            {
                "execution_position": execution_position,
                "group_index": group_index,
                "root_name": group["root_name"],
                "seed": group_seed,
                "run_dir": str(run_dir),
                "best_reward": result["cem"]["best_reward"],
                "simulation_seconds": result["simulation"]["simulation_seconds"],
                "physics_steps_per_second": result["simulation"][
                    "physics_steps_per_second"
                ],
                "peak_gpu_memory_mib": result["simulation"][
                    "system_gpu_memory_used_mib_peak"
                ],
                "state_tensor_device": result["simulation"]["state_tensor_device"],
            }
        )

    for group in plan["groups"]:
        payload = legacy_local_group_payload(group, current_states, base_dy=base_dy)
        _write_json(legacy_dir / _safe_group_filename(group["root_name"]), payload)

    final_states_path = args.output_dir / "local_group_final_states.json"
    _write_json(final_states_path, {"states_rest_wxyz": current_states})
    result = {
        "passed": True,
        "backend": "isaac-lab",
        "scope": "stage3-local-groups-only",
        "scene_dir": str(args.scene_dir),
        "output_dir": str(args.output_dir),
        "physics_assets": str(physics_path),
        "local_group_plan": str(plan_path),
        "legacy_local_group_dir": str(legacy_dir),
        "final_states": str(final_states_path),
        "scene_object_count": len(scene_names),
        "group_count": len(plan["groups"]),
        "execution_order": execution_order,
        "base_dy": base_dy,
        "num_envs": args.num_envs,
        "cem_iterations": args.cem_iters,
        "settle_steps": args.settle_steps,
        "runs": run_records,
        "total_seconds": time.perf_counter() - started,
    }
    _write_json(args.output_dir / "local_group_pipeline_results.json", result)

    if not args.run_global:
        return result

    global_plan_dir = args.output_dir / "global_entity_plan"
    global_run_dir = args.output_dir / "global_cem"
    _run_logged(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_global_entity_plan.py"),
            "--scene-tree",
            str(args.scene_tree),
            "--physics-assets",
            str(physics_path),
            "--initial-states",
            str(final_states_path),
            "--output-dir",
            str(global_plan_dir),
            "--allow-output-below-derived-inputs",
        ],
        args.output_dir / "build_global_entity_plan.log",
    )
    global_plan_path = global_plan_dir / "global_entity_plan.json"
    global_num_envs = args.global_num_envs or args.num_envs
    global_cem_iters = args.global_cem_iters or args.cem_iters
    global_seed = (
        args.global_seed
        if args.global_seed is not None
        else args.seed + len(execution_order)
    )
    _run_logged(
        [
            "bash",
            str(repo_root / "scripts" / "run_isaaclab_real_global_cem.sh"),
            str(physics_path),
            str(global_plan_path),
            str(final_states_path),
            str(global_run_dir),
            "--num-envs",
            str(global_num_envs),
            "--cem-iters",
            str(global_cem_iters),
            "--settle-steps",
            str(args.settle_steps),
            "--early-steps",
            str(args.early_steps),
            "--seed",
            str(global_seed),
        ],
        args.output_dir / "global_cem.stdout.log",
    )
    global_result_path = global_run_dir / "real_global_cem_results.json"
    with global_result_path.open("r", encoding="utf-8") as file:
        global_result = json.load(file)
    if not global_result.get("passed"):
        raise RuntimeError("global CEM did not pass its checks")

    stage3_result = {
        "passed": True,
        "backend": "isaac-lab",
        "scope": "stage3-local-and-global",
        "scene_dir": str(args.scene_dir),
        "output_dir": str(args.output_dir),
        "local_phase": str(args.output_dir / "local_group_pipeline_results.json"),
        "global_entity_plan": str(global_plan_path),
        "global_phase": str(global_result_path),
        "physics_assets": str(physics_path),
        "local_final_states": str(final_states_path),
        "final_candidate_states": global_result["outputs"]["candidate_states"],
        "final_settled_states": global_result["outputs"]["settled_states"],
        "scene_object_count": len(scene_names),
        "sampled_entity_count": global_result["simulation"][
            "sampled_entity_count"
        ],
        "local_group_count": len(plan["groups"]),
        "local_num_envs": args.num_envs,
        "local_cem_iterations": args.cem_iters,
        "global_num_envs": global_num_envs,
        "global_cem_iterations": global_cem_iters,
        "global_seed": global_seed,
        "global_best_reward": global_result["cem"]["best_reward"],
        "global_state_tensor_device": global_result["simulation"][
            "state_tensor_device"
        ],
        "global_contact_tensor_device": global_result["simulation"][
            "contact_tensor_device"
        ],
        "global_simulation_seconds": global_result["simulation"][
            "simulation_seconds"
        ],
        "global_peak_gpu_memory_mib": global_result["simulation"][
            "system_gpu_memory_used_mib_peak"
        ],
        "total_seconds": time.perf_counter() - started,
    }
    _write_json(args.output_dir / "stage3_pipeline_results.json", stage3_result)
    return stage3_result


def main() -> int:
    args = _parse_args()
    try:
        result = _run(args)
    except BaseException as error:
        failure = {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failure_name = (
            "stage3_pipeline_failure.json"
            if args.run_global
            else "local_group_pipeline_failure.json"
        )
        try:
            _write_json(args.output_dir / failure_name, failure)
        finally:
            raise
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

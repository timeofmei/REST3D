#!/usr/bin/env python3
"""Compare matched Isaac Gym and Isaac Lab full-scene replay artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rest3d.sim.backend_comparison import (
    summarize_replay_quality,
    validate_replay_comparison,
)
from rest3d.sim.global_cem import (
    build_global_entity_plan,
    evaluate_convex_hull_intersections_wxyz,
    evaluate_global_cem_energy,
)
from rest3d.sim.replay_scene import load_replay_scene
from rest3d.utils.mesh import load_trimesh_any


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-tree", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--isaac-gym-result-dir", type=Path, required=True)
    parser.add_argument("--isaac-lab-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--early-step", type=int, default=15)
    parser.add_argument("--evaluation-step", type=int, default=60)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--position-threshold", type=float, default=0.1)
    parser.add_argument("--rotation-threshold", type=float, default=0.1)
    parser.add_argument(
        "--collision-approximation",
        choices=("convex_hull", "convex_decomposition"),
        default="convex_decomposition",
    )
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.early_step < args.evaluation_step <= args.steps:
        parser.error("steps must satisfy 1 <= early < evaluation <= total")
    if args.physics_dt <= 0.0:
        parser.error("--physics-dt must be positive")
    if args.position_threshold <= 0.0 or args.rotation_threshold <= 0.0:
        parser.error("stability thresholds must be positive")
    if not args.device.startswith("cuda"):
        parser.error("Stage F common geometry evaluation requires a CUDA device")

    for name in (
        "scene_tree",
        "scene_dir",
        "isaac_gym_result_dir",
        "isaac_lab_result_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.output_dir = args.output_dir.expanduser().resolve()
    for input_path in (args.scene_dir, args.isaac_gym_result_dir, args.isaac_lab_result_dir):
        if args.output_dir == input_path or input_path in args.output_dir.parents:
            parser.error("--output-dir must be outside all read-only inputs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return args


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload):
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _git_environment(repo_root: Path):
    def command(*arguments):
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _tensor_summary(value: torch.Tensor):
    array = value.detach().cpu().numpy()
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.tolist()


def _common_energy(
    states: torch.Tensor,
    reference_poses: torch.Tensor,
    plan,
    names,
    hulls,
    *,
    early_step: int,
    evaluation_step: int,
):
    placed = states[0].unsqueeze(0)
    early = states[early_step].unsqueeze(0)
    settled = states[evaluation_step].unsqueeze(0)
    placement_geometry = evaluate_convex_hull_intersections_wxyz(placed, hulls)
    settled_geometry = evaluate_convex_hull_intersections_wxyz(settled, hulls)
    final_geometry = evaluate_convex_hull_intersections_wxyz(
        states[-1].unsqueeze(0), hulls
    )
    energy = evaluate_global_cem_energy(
        plan,
        names,
        placed,
        early,
        settled,
        reference_poses,
        placement_penetration=placement_geometry["total"],
        settled_penetration=settled_geometry["total"],
        placement_penetration_by_object=placement_geometry["per_object"],
        settled_penetration_by_object=settled_geometry["per_object"],
    )
    components = {
        name: _tensor_summary(value)
        for name, value in energy.items()
        if name != "per_object"
    }
    per_object = {}
    for object_index, object_name in enumerate(names):
        per_object[object_name] = {
            component: float(values[0, object_index].item())
            for component, values in energy["per_object"].items()
        }
    reconciliation_error = abs(
        float(energy["per_object"]["energy"].sum().item())
        - float(energy["energy"].item())
    )
    return {
        "device": str(states.device),
        "dtype": str(states.dtype),
        "components": components,
        "per_object": per_object,
        "placement_intersection_count": float(
            placement_geometry["total"].item()
        ),
        "settled_intersection_count": float(settled_geometry["total"].item()),
        "final_intersection_count": float(final_geometry["total"].item()),
        "maximum_object_energy_reconciliation_error": reconciliation_error,
    }


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    gym_result = _read_json(args.isaac_gym_result_dir / "replay_results.json")
    lab_result = _read_json(args.isaac_lab_result_dir / "replay_results.json")
    gym_states = np.load(args.isaac_gym_result_dir / "replay_states_rest.npy")
    lab_states = np.load(args.isaac_lab_result_dir / "replay_states_rest.npy")
    comparison = validate_replay_comparison(
        gym_result,
        lab_result,
        gym_states,
        lab_states,
        expected_steps=args.steps,
        expected_dt_seconds=args.physics_dt,
        expected_collision_approximation=args.collision_approximation,
        position_threshold_m=args.position_threshold,
        rotation_threshold_rad=args.rotation_threshold,
    )
    names = tuple(comparison["object_names"])
    scene = load_replay_scene(args.scene_tree, args.scene_dir)
    if scene.names != names:
        raise ValueError("comparison scene object order differs from replay artifacts")
    plan = build_global_entity_plan(
        args.scene_tree,
        scene_names=names,
        fixed_names=scene.fixed_names,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the common Stage F energy evaluation")
    hulls = []
    for spec in scene.objects:
        mesh = load_trimesh_any(str(spec.obj_path))
        vertices = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
        hulls.append(torch.as_tensor(vertices, device=args.device, dtype=torch.float64))
    gym_tensor = torch.as_tensor(gym_states, device=args.device, dtype=torch.float64)
    lab_tensor = torch.as_tensor(lab_states, device=args.device, dtype=torch.float64)
    reference_poses = gym_tensor[0, :, :7]
    gym_energy = _common_energy(
        gym_tensor,
        reference_poses,
        plan,
        names,
        hulls,
        early_step=args.early_step,
        evaluation_step=args.evaluation_step,
    )
    lab_energy = _common_energy(
        lab_tensor,
        reference_poses,
        plan,
        names,
        hulls,
        early_step=args.early_step,
        evaluation_step=args.evaluation_step,
    )
    torch.cuda.synchronize()

    gym_quality = summarize_replay_quality(
        gym_states,
        names,
        evaluation_step=args.evaluation_step,
        position_threshold_m=args.position_threshold,
        rotation_threshold_rad=args.rotation_threshold,
    )
    lab_quality = summarize_replay_quality(
        lab_states,
        names,
        evaluation_step=args.evaluation_step,
        position_threshold_m=args.position_threshold,
        rotation_threshold_rad=args.rotation_threshold,
    )
    common_checks = {
        **comparison["checks"],
        "all_objects_enter_common_energy": set(gym_energy["per_object"])
        == set(lab_energy["per_object"])
        == set(names),
        "common_energy_runs_on_cuda": gym_energy["device"].startswith("cuda")
        and lab_energy["device"].startswith("cuda"),
        "common_energy_reconciles_per_object": gym_energy[
            "maximum_object_energy_reconciliation_error"
        ]
        < 1.0e-5
        and lab_energy["maximum_object_energy_reconciliation_error"] < 1.0e-5,
    }

    gym_sim = gym_result["simulation"]
    lab_sim = lab_result["simulation"]
    performance = {
        "isaac_gym": {
            key: gym_sim.get(key)
            for key in (
                "startup_seconds",
                "asset_load_seconds",
                "simulation_seconds",
                "steps_per_second",
                "total_seconds",
                "system_gpu_memory_used_mib_peak",
            )
        },
        "isaac_lab": {
            key: lab_sim.get(key)
            for key in (
                "startup_seconds",
                "asset_load_seconds",
                "simulation_seconds",
                "steps_per_second",
                "total_seconds",
                "system_gpu_memory_used_mib_peak",
            )
        },
        "lab_over_gym_simulation_time_ratio": float(
            lab_sim["simulation_seconds"] / gym_sim["simulation_seconds"]
        ),
        "lab_over_gym_steps_per_second_ratio": float(
            lab_sim["steps_per_second"] / gym_sim["steps_per_second"]
        ),
    }
    result = {
        "passed": all(common_checks.values()),
        "scope": "stage-f-matched-full-scene-replay",
        "checks": common_checks,
        "inputs": {
            "scene_tree": str(args.scene_tree),
            "scene_dir": str(args.scene_dir),
            "isaac_gym_result_dir": str(args.isaac_gym_result_dir),
            "isaac_lab_result_dir": str(args.isaac_lab_result_dir),
        },
        "workload": {
            "object_count": len(names),
            "fixed_object_count": len(comparison["fixed_names"]),
            "parallel_environments": 1,
            "steps": args.steps,
            "early_energy_step": args.early_step,
            "stability_evaluation_step": args.evaluation_step,
            "physics_dt_seconds": args.physics_dt,
            "collision_approximation": args.collision_approximation,
            "seed": args.seed,
            "seed_usage": "recorded; replay and post-evaluation are deterministic",
            "cem_population": None,
            "cem_iterations": None,
        },
        "software_and_hardware": {
            "isaac_gym": {
                "versions": gym_result["versions"],
                "environment": gym_result["environment"],
            },
            "isaac_lab": {
                "versions": lab_result["versions"],
                "environment": lab_result["environment"],
            },
        },
        "devices": {
            "isaac_gym": {
                "physics": gym_sim["device"],
                "gpu_pipeline": gym_sim["physics_uses_gpu_pipeline"],
                "state_tensor": gym_sim["state_tensor_device"],
            },
            "isaac_lab": {
                "physics": lab_sim["device"],
                "gpu_pipeline": lab_sim["physics_uses_gpu_pipeline"],
                "state_tensor": lab_sim["state_tensor_device"],
            },
            "common_energy": str(gym_tensor.device),
        },
        "performance": performance,
        "quality": {
            "isaac_gym": {"stability": gym_quality, "common_energy": gym_energy},
            "isaac_lab": {"stability": lab_quality, "common_energy": lab_energy},
        },
        "comparison": {
            "maximum_initial_position_difference_m": comparison[
                "maximum_initial_position_difference_m"
            ],
            "maximum_initial_rotation_difference_rad": comparison[
                "maximum_initial_rotation_difference_rad"
            ],
            "simulation_time_difference_seconds": float(
                lab_sim["simulation_seconds"] - gym_sim["simulation_seconds"]
            ),
            "stable_object_count_difference_lab_minus_gym": int(
                lab_quality["stable_object_count"] - gym_quality["stable_object_count"]
            ),
            "common_energy_difference_lab_minus_gym": float(
                lab_energy["components"]["energy"]
                - gym_energy["components"]["energy"]
            ),
        },
        "total_comparison_seconds": time.perf_counter() - started,
    }
    environment = {
        "git": _git_environment(repo_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "comparison_device": str(gym_tensor.device),
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }
    command = " ".join(shlex.quote(argument) for argument in [sys.executable, *sys.argv])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (args.output_dir / "run.log").write_text(
        "Stage F matched replay comparison\n"
        + "passed=%s\n" % result["passed"]
        + "checks=%s\n" % json.dumps(common_checks, sort_keys=True)
        + "gym_simulation_seconds=%.9f\n" % gym_sim["simulation_seconds"]
        + "lab_simulation_seconds=%.9f\n" % lab_sim["simulation_seconds"],
        encoding="utf-8",
    )
    _write_json(args.output_dir / "environment.json", environment)
    _write_json(args.output_dir / "metrics.json", result)
    _write_json(args.output_dir / "comparison_results.json", result)
    np.save(
        args.output_dir / "states_initial.npy",
        np.stack([gym_states[0], lab_states[0]], axis=0),
    )
    np.save(
        args.output_dir / "states_final.npy",
        np.stack([gym_states[-1], lab_states[-1]], axis=0),
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Stage F comparison checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Backend-neutral validation and quality summaries for Stage F replay runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .stability import evaluate_replay_stability, quaternion_geodesic_distance_wxyz


def validate_replay_comparison(
    gym_result: Mapping,
    lab_result: Mapping,
    gym_states_rest: np.ndarray,
    lab_states_rest: np.ndarray,
    *,
    expected_steps: int,
    expected_dt_seconds: float,
    expected_collision_approximation: str,
    position_threshold_m: float,
    rotation_threshold_rad: float,
    initial_pose_tolerance: float = 1.0e-5,
) -> Mapping:
    """Prove that two replay results describe the same comparable workload."""

    gym_states = np.asarray(gym_states_rest, dtype=np.float64)
    lab_states = np.asarray(lab_states_rest, dtype=np.float64)
    if gym_states.ndim != 3 or gym_states.shape[-1] != 13:
        raise ValueError("Isaac Gym states must have shape [frames, objects, 13]")
    if lab_states.ndim != 3 or lab_states.shape[-1] != 13:
        raise ValueError("Isaac Lab states must have shape [frames, objects, 13]")

    gym_names = tuple(gym_result["scene"]["object_names"])
    lab_names = tuple(lab_result["scene"]["object_names"])
    gym_fixed = tuple(gym_result["scene"]["fixed_names"])
    lab_fixed = tuple(lab_result["scene"]["fixed_names"])
    position_error = np.linalg.norm(
        gym_states[0, :, :3] - lab_states[0, :, :3], axis=-1
    )
    rotation_error = quaternion_geodesic_distance_wxyz(
        gym_states[0, :, 3:7], lab_states[0, :, 3:7]
    )
    checks = {
        "both_runtime_results_passed": bool(gym_result.get("runtime_passed"))
        and bool(lab_result.get("runtime_passed")),
        "backend_labels_are_distinct": gym_result.get("backend") == "isaac-gym"
        and lab_result.get("backend") == "isaac-lab",
        "scene_inputs_match": Path(gym_result["scene"]["scene_tree"]).resolve()
        == Path(lab_result["scene"]["scene_tree"]).resolve()
        and Path(gym_result["scene"]["scene_dir"]).resolve()
        == Path(lab_result["scene"]["scene_dir"]).resolve(),
        "object_order_matches": gym_names == lab_names,
        "fixed_object_sets_match": set(gym_fixed) == set(lab_fixed),
        "state_shapes_match": gym_states.shape == lab_states.shape
        == (expected_steps + 1, len(gym_names), 13),
        "physics_steps_match": int(gym_result["simulation"]["steps"])
        == int(lab_result["simulation"]["steps"])
        == expected_steps,
        "physics_dt_matches": np.isclose(
            float(gym_result["simulation"]["dt_seconds"]), expected_dt_seconds
        )
        and np.isclose(
            float(lab_result["simulation"]["dt_seconds"]), expected_dt_seconds
        ),
        "collision_strategy_matches": gym_result["simulation"][
            "collision_approximation"
        ]
        == lab_result["simulation"]["collision_approximation"]
        == expected_collision_approximation,
        "data_collection_matches": gym_result["simulation"].get(
            "data_collection"
        )
        == lab_result["simulation"].get("data_collection")
        == "root_states_each_step",
        "stability_definition_matches": int(
            gym_result["stability"]["evaluation_step"]
        )
        == int(lab_result["stability"]["evaluation_step"])
        and np.isclose(
            float(gym_result["stability"]["position_threshold_m"]),
            position_threshold_m,
        )
        and np.isclose(
            float(lab_result["stability"]["position_threshold_m"]),
            position_threshold_m,
        )
        and np.isclose(
            float(gym_result["stability"]["rotation_threshold_rad"]),
            rotation_threshold_rad,
        )
        and np.isclose(
            float(lab_result["stability"]["rotation_threshold_rad"]),
            rotation_threshold_rad,
        ),
        "initial_poses_match": float(position_error.max(initial=0.0))
        <= initial_pose_tolerance
        and float(rotation_error.max(initial=0.0)) <= initial_pose_tolerance,
        "isaac_gym_uses_gpu_physx": bool(
            gym_result["simulation"]["physics_uses_gpu_sim"]
        ),
        "isaac_gym_uses_cpu_tensor_pipeline": not bool(
            gym_result["simulation"]["physics_uses_gpu_pipeline"]
        )
        and str(gym_result["simulation"]["state_tensor_device"]) == "cpu",
        "isaac_lab_uses_gpu_physx_and_tensors": bool(
            lab_result["simulation"]["physics_uses_gpu_sim"]
        )
        and bool(lab_result["simulation"]["physics_uses_gpu_pipeline"])
        and str(lab_result["simulation"]["state_tensor_device"]).startswith("cuda"),
        "same_physical_gpu": gym_result["environment"].get("gpu")
        == lab_result["environment"].get("gpu"),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "object_names": list(gym_names),
        "fixed_names": list(gym_fixed),
        "maximum_initial_position_difference_m": float(
            position_error.max(initial=0.0)
        ),
        "maximum_initial_rotation_difference_rad": float(
            rotation_error.max(initial=0.0)
        ),
    }


def summarize_replay_quality(
    states_rest: np.ndarray,
    object_names: Sequence[str],
    *,
    evaluation_step: int,
    position_threshold_m: float,
    rotation_threshold_rad: float,
    early_window_steps: int = 10,
    terminal_window_steps: int = 10,
) -> Mapping:
    """Apply one shared stability definition and return JSON-ready metrics."""

    metrics = evaluate_replay_stability(
        states_rest,
        evaluation_step=evaluation_step,
        position_threshold_m=position_threshold_m,
        rotation_threshold_rad=rotation_threshold_rad,
        early_window_steps=early_window_steps,
        terminal_window_steps=terminal_window_steps,
    )
    if len(object_names) != metrics.stable.shape[0]:
        raise ValueError("object names must match the replay state object axis")
    objects = {}
    for index, name in enumerate(object_names):
        objects[name] = {
            "stable": bool(metrics.stable[index]),
            "displacement_at_evaluation_m": float(
                metrics.displacement_at_evaluation_m[index]
            ),
            "rotation_at_evaluation_rad": float(
                metrics.rotation_at_evaluation_rad[index]
            ),
            "final_displacement_m": float(metrics.final_displacement_m[index]),
            "final_rotation_rad": float(metrics.final_rotation_rad[index]),
            "terminal_mean_linear_speed_m_s": float(
                metrics.terminal_mean_linear_speed_m_s[index]
            ),
            "terminal_mean_angular_speed_rad_s": float(
                metrics.terminal_mean_angular_speed_rad_s[index]
            ),
        }
    return {
        "scene_stable": metrics.scene_stable,
        "stable_object_count": int(np.count_nonzero(metrics.stable)),
        "object_count": len(object_names),
        "maximum_displacement_at_evaluation_m": float(
            metrics.displacement_at_evaluation_m.max(initial=0.0)
        ),
        "maximum_rotation_at_evaluation_rad": float(
            metrics.rotation_at_evaluation_rad.max(initial=0.0)
        ),
        "objects": objects,
    }

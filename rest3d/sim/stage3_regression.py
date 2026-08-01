"""Backend-neutral validation for Stage 3 regression artifacts."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _positive_finite_inertia(record: Mapping) -> bool:
    inertia = np.asarray(record.get("inertia_kg_m2"), dtype=np.float64)
    return (
        inertia.shape == (3, 3)
        and np.isfinite(inertia).all()
        and bool(np.all(np.linalg.eigvalsh(0.5 * (inertia + inertia.T)) > 0.0))
    )


def validate_stage3_regression(
    *,
    expected_names: Sequence[str],
    stage2_hashes_before: Mapping[str, str],
    stage2_hashes_after: Mapping[str, str],
    physics_assets: Mapping,
    local_plan: Mapping,
    pipeline_result: Mapping,
    global_result: Mapping,
    replay_result: Mapping,
    replay_state_shape: Sequence[int],
    comparison_result: Mapping,
) -> dict:
    """Validate one generic full-pipeline and dual-backend integration run."""

    names = tuple(expected_names)
    name_set = set(names)
    physics_objects = physics_assets.get("objects", {})
    plan_names = tuple(local_plan.get("scene_names", ()))
    global_names = tuple(global_result.get("plan", {}).get("scene_names", ()))
    replay_names = tuple(replay_result.get("scene", {}).get("object_names", ()))
    comparison_names = tuple(comparison_result.get("object_names", ()))
    global_checks = global_result.get("checks", {})
    comparison_checks = comparison_result.get("checks", {})

    mass_properties_valid = set(physics_objects) == name_set
    for record in physics_objects.values():
        mass = float(record.get("mass_kg", float("nan")))
        mass_properties_valid = mass_properties_valid and np.isfinite(mass) and mass > 0.0
        mass_properties_valid = mass_properties_valid and _positive_finite_inertia(record)

    simulation = global_result.get("simulation", {})
    replay_simulation = replay_result.get("simulation", {})
    stability = replay_result.get("stability", {})
    checks = {
        "stage2_inputs_unchanged": dict(stage2_hashes_before)
        == dict(stage2_hashes_after),
        "arbitrary_object_names_preserved": len(names) == 3
        and len(name_set) == len(names)
        and set(physics_objects) == name_set
        and set(plan_names) == name_set
        and set(global_names) == name_set
        and set(replay_names) == name_set
        and set(comparison_names) == name_set,
        "geometry_policy_generated_all_mass_and_inertia": bool(
            mass_properties_valid
        ),
        "no_object_physics_override_input": physics_assets.get("policy") is not None
        and all("source_path" in record for record in physics_objects.values()),
        "local_plan_covers_complete_scene": set(plan_names) == name_set
        and int(local_plan.get("group_count", -1)) >= 1,
        "local_and_global_pipeline_passed": bool(pipeline_result.get("passed"))
        and pipeline_result.get("backend") == "isaac-lab"
        and pipeline_result.get("scope") == "stage3-local-and-global"
        and int(pipeline_result.get("scene_object_count", -1)) == len(names),
        "global_runtime_checks_passed": bool(global_result.get("passed"))
        and bool(global_checks)
        and all(bool(value) for value in global_checks.values()),
        "global_candidate_contains_every_object": bool(
            global_checks.get("all_scene_objects_in_every_candidate")
        )
        and bool(global_checks.get("all_object_energy_reconciles_with_global"))
        and int(simulation.get("scene_object_count_per_env", -1)) == len(names),
        "global_physx_and_tensors_are_cuda": bool(
            global_checks.get("physics_uses_gpu_sim")
        )
        and bool(global_checks.get("physics_uses_gpu_pipeline"))
        and str(simulation.get("state_tensor_device", "")).startswith("cuda")
        and str(simulation.get("contact_tensor_device", "")).startswith("cuda"),
        "final_replay_runtime_passed": bool(replay_result.get("runtime_passed"))
        and all(
            bool(replay_result.get("checks", {}).get(key))
            for key in (
                "all_scene_objects_loaded",
                "physics_uses_gpu_sim",
                "physics_uses_gpu_pipeline",
                "state_tensor_is_cuda",
                "contact_tensor_is_cuda",
                "all_objects_evaluated_for_stability",
            )
        ),
        "final_replay_uses_paper_gate": bool(replay_result.get("passed"))
        and bool(stability.get("require_stable"))
        and bool(stability.get("scene_stable"))
        and int(stability.get("evaluation_step", -1)) == 60
        and np.isclose(float(stability.get("position_threshold_m", np.nan)), 0.1)
        and np.isclose(float(stability.get("rotation_threshold_rad", np.nan)), 0.1),
        "final_replay_state_shape_is_complete": tuple(replay_state_shape)
        == (61, len(names), 13)
        and str(replay_simulation.get("state_tensor_device", "")).startswith("cuda"),
        "dual_backend_comparison_passed": bool(comparison_result.get("passed"))
        and bool(comparison_checks)
        and all(bool(value) for value in comparison_checks.values()),
        "legacy_gym_and_lab_device_contracts_hold": bool(
            comparison_checks.get("isaac_gym_uses_gpu_physx")
        )
        and bool(comparison_checks.get("isaac_gym_uses_cpu_tensor_pipeline"))
        and bool(comparison_checks.get("isaac_lab_uses_gpu_physx_and_tensors")),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "passed": all(checks.values()),
        "scope": "stage-g-generic-headless-regression",
        "checks": checks,
        "object_names": list(names),
        "object_count": len(names),
        "local_group_count": int(local_plan.get("group_count", 0)),
        "sampled_global_entity_count": int(
            simulation.get("sampled_entity_count", 0)
        ),
        "parallel_environments": int(simulation.get("num_envs", 0)),
        "global_simulation_seconds": simulation.get("simulation_seconds"),
        "global_peak_gpu_memory_mib": simulation.get(
            "system_gpu_memory_used_mib_peak"
        ),
        "final_stable_object_count": stability.get("stable_object_count"),
        "final_scene_stable": stability.get("scene_stable"),
        "comparison_gate_count": len(comparison_checks),
    }

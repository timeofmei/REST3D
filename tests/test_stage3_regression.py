import tempfile
import unittest
from pathlib import Path

import numpy as np

from rest3d.sim.local_groups import ObjectBounds, build_local_group_plan
from rest3d.sim.physics_assets import analyze_physics_asset
from rest3d.sim.replay_scene import load_replay_scene
from rest3d.sim.stage3_regression import validate_stage3_regression
from scripts.create_stage3_regression_scene import create_stage3_regression_scene


class Stage3RegressionTest(unittest.TestCase):
    def test_arbitrary_stage2_names_need_no_object_physics_overrides(self):
        names = ("shape_x_101", "shape_y_203", "shape_z_307")
        with tempfile.TemporaryDirectory() as directory:
            stage2 = Path(directory) / "stage2"
            fixture = create_stage3_regression_scene(stage2, names)
            scene = load_replay_scene(
                fixture["scene_tree"], fixture["scene_canon"]
            )
            properties = {
                spec.name: analyze_physics_asset(spec.obj_path)
                for spec in scene.objects
            }
            plan = build_local_group_plan(
                fixture["scene_tree"],
                scene_names=scene.names,
                fixed_names=scene.fixed_names,
                object_bounds={
                    name: ObjectBounds(
                        minimum=record.bounds_min_m,
                        maximum=record.bounds_max_m,
                    )
                    for name, record in properties.items()
                },
            )

        self.assertEqual(set(scene.names), set(names))
        self.assertEqual(fixture["object_physics_overrides"], None)
        self.assertEqual(len(plan.groups), 1)
        self.assertEqual(plan.groups[0].root_name, names[0])
        self.assertEqual(plan.groups[0].direct_child_names, (names[1],))
        self.assertTrue(all(record.mass_kg > 0.0 for record in properties.values()))

    def test_complete_generic_integration_contract_passes(self):
        names = ("alpha", "beta", "gamma")
        inertia = np.diag([0.1, 0.2, 0.3]).tolist()
        physics = {
            "policy": {"nominal_density_kg_m3": 700.0},
            "objects": {
                name: {
                    "mass_kg": 1.0,
                    "inertia_kg_m2": inertia,
                    "source_path": f"/read-only/{name}.obj",
                }
                for name in names
            },
        }
        local_plan = {"scene_names": list(names), "group_count": 1}
        pipeline = {
            "passed": True,
            "backend": "isaac-lab",
            "scope": "stage3-local-and-global",
            "scene_object_count": 3,
        }
        required_global_checks = {
            "physics_uses_gpu_sim": True,
            "physics_uses_gpu_pipeline": True,
            "all_scene_objects_in_every_candidate": True,
            "all_object_energy_reconciles_with_global": True,
        }
        global_result = {
            "passed": True,
            "checks": required_global_checks,
            "plan": {"scene_names": list(names)},
            "simulation": {
                "scene_object_count_per_env": 3,
                "sampled_entity_count": 2,
                "num_envs": 4,
                "state_tensor_device": "cuda:0",
                "contact_tensor_device": "cuda:0",
            },
        }
        replay_checks = {
            "all_scene_objects_loaded": True,
            "physics_uses_gpu_sim": True,
            "physics_uses_gpu_pipeline": True,
            "state_tensor_is_cuda": True,
            "contact_tensor_is_cuda": True,
            "all_objects_evaluated_for_stability": True,
        }
        replay = {
            "passed": True,
            "runtime_passed": True,
            "checks": replay_checks,
            "scene": {"object_names": list(names)},
            "simulation": {"state_tensor_device": "cuda:0"},
            "stability": {
                "evaluation_step": 60,
                "position_threshold_m": 0.1,
                "rotation_threshold_rad": 0.1,
                "require_stable": True,
                "stable_object_count": 3,
                "scene_stable": True,
            },
        }
        comparison_checks = {
            "isaac_gym_uses_gpu_physx": True,
            "isaac_gym_uses_cpu_tensor_pipeline": True,
            "isaac_lab_uses_gpu_physx_and_tensors": True,
        }
        comparison = {
            "passed": True,
            "checks": comparison_checks,
            "object_names": list(names),
        }

        result = validate_stage3_regression(
            expected_names=names,
            stage2_hashes_before={"asset": "same"},
            stage2_hashes_after={"asset": "same"},
            physics_assets=physics,
            local_plan=local_plan,
            pipeline_result=pipeline,
            global_result=global_result,
            replay_result=replay,
            replay_state_shape=(61, 3, 13),
            comparison_result=comparison,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["object_names"], list(names))

    def test_changed_stage2_input_fails_regression(self):
        names = ("alpha", "beta", "gamma")
        inertia = np.eye(3).tolist()
        physics = {
            "policy": {"nominal_density_kg_m3": 700.0},
            "objects": {
                name: {
                    "mass_kg": 1.0,
                    "inertia_kg_m2": inertia,
                    "source_path": name,
                }
                for name in names
            },
        }
        global_checks = {
            "physics_uses_gpu_sim": True,
            "physics_uses_gpu_pipeline": True,
            "all_scene_objects_in_every_candidate": True,
            "all_object_energy_reconciles_with_global": True,
        }
        replay_checks = {
            key: True
            for key in (
                "all_scene_objects_loaded",
                "physics_uses_gpu_sim",
                "physics_uses_gpu_pipeline",
                "state_tensor_is_cuda",
                "contact_tensor_is_cuda",
                "all_objects_evaluated_for_stability",
            )
        }
        comparison_checks = {
            "isaac_gym_uses_gpu_physx": True,
            "isaac_gym_uses_cpu_tensor_pipeline": True,
            "isaac_lab_uses_gpu_physx_and_tensors": True,
        }
        result = validate_stage3_regression(
            expected_names=names,
            stage2_hashes_before={"asset": "old"},
            stage2_hashes_after={"asset": "new"},
            physics_assets=physics,
            local_plan={"scene_names": list(names), "group_count": 1},
            pipeline_result={
                "passed": True,
                "backend": "isaac-lab",
                "scope": "stage3-local-and-global",
                "scene_object_count": 3,
            },
            global_result={
                "passed": True,
                "checks": global_checks,
                "plan": {"scene_names": list(names)},
                "simulation": {
                    "scene_object_count_per_env": 3,
                    "sampled_entity_count": 2,
                    "num_envs": 4,
                    "state_tensor_device": "cuda:0",
                    "contact_tensor_device": "cuda:0",
                },
            },
            replay_result={
                "passed": True,
                "runtime_passed": True,
                "checks": replay_checks,
                "scene": {"object_names": list(names)},
                "simulation": {"state_tensor_device": "cuda:0"},
                "stability": {
                    "evaluation_step": 60,
                    "position_threshold_m": 0.1,
                    "rotation_threshold_rad": 0.1,
                    "require_stable": True,
                    "scene_stable": True,
                },
            },
            replay_state_shape=(61, 3, 13),
            comparison_result={
                "passed": True,
                "checks": comparison_checks,
                "object_names": list(names),
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["stage2_inputs_unchanged"])


if __name__ == "__main__":
    unittest.main()

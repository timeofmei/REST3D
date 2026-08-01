import copy
import unittest

import numpy as np

from rest3d.sim.backend_comparison import (
    summarize_replay_quality,
    validate_replay_comparison,
)


class BackendComparisonTest(unittest.TestCase):
    def _result(self, backend):
        is_lab = backend == "isaac-lab"
        return {
            "runtime_passed": True,
            "backend": backend,
            "environment": {"gpu": "test-gpu"},
            "scene": {
                "scene_tree": "/tmp/shared_tree.json",
                "scene_dir": "/tmp/shared_scene",
                "object_names": ["root", "child"],
                "fixed_names": [],
            },
            "simulation": {
                "steps": 120,
                "dt_seconds": 1.0 / 60.0,
                "collision_approximation": "convex_decomposition",
                "data_collection": "root_states_each_step",
                "physics_uses_gpu_sim": True,
                "physics_uses_gpu_pipeline": is_lab,
                "state_tensor_device": "cuda:0" if is_lab else "cpu",
            },
            "stability": {
                "evaluation_step": 60,
                "position_threshold_m": 0.1,
                "rotation_threshold_rad": 0.1,
            },
        }

    def _states(self):
        states = np.zeros((121, 2, 13), dtype=np.float64)
        states[..., 3] = 1.0
        return states

    def test_matched_workload_and_device_modes_pass(self):
        result = validate_replay_comparison(
            self._result("isaac-gym"),
            self._result("isaac-lab"),
            self._states(),
            self._states(),
            expected_steps=120,
            expected_dt_seconds=1.0 / 60.0,
            expected_collision_approximation="convex_decomposition",
            position_threshold_m=0.1,
            rotation_threshold_rad=0.1,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["object_names"], ["root", "child"])

    def test_workload_mismatch_is_reported(self):
        lab = copy.deepcopy(self._result("isaac-lab"))
        lab["simulation"]["collision_approximation"] = "convex_hull"

        result = validate_replay_comparison(
            self._result("isaac-gym"),
            lab,
            self._states(),
            self._states(),
            expected_steps=120,
            expected_dt_seconds=1.0 / 60.0,
            expected_collision_approximation="convex_decomposition",
            position_threshold_m=0.1,
            rotation_threshold_rad=0.1,
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["collision_strategy_matches"])

    def test_common_quality_uses_paper_thresholds_for_every_object(self):
        states = self._states()
        states[60:, 1, 0] = 0.11

        quality = summarize_replay_quality(
            states,
            ("root", "child"),
            evaluation_step=60,
            position_threshold_m=0.1,
            rotation_threshold_rad=0.1,
        )

        self.assertFalse(quality["scene_stable"])
        self.assertEqual(quality["stable_object_count"], 1)
        self.assertTrue(quality["objects"]["root"]["stable"])
        self.assertFalse(quality["objects"]["child"]["stable"])


if __name__ == "__main__":
    unittest.main()

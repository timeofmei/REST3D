import sys
import unittest
from types import SimpleNamespace

import numpy as np

from rest3d.optim.cem import CEMOptimizer


def _config(**overrides):
    values = {
        "act_dim": 6,
        "cem_pop_size": 8,
        "cem_elite_frac": 0.25,
        "cem_iters_joint": 3,
        "cem_iters_subtree": 3,
        "cem_update_mode": "cem",
        "keep_best": False,
        "update_use_only_best": False,
        "std_update_mode": "topk_std",
        "decay_std_rate": 0.95,
        "reward_threshold": -0.01,
        "init_trans_x_std": 1.0,
        "init_trans_y_std": 1.0,
        "init_trans_z_std": 1.0,
        "init_rot_roll_std": 1.0,
        "init_rot_pitch_std": 1.0,
        "init_rot_yaw_std": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CEMOptimizerTest(unittest.TestCase):
    def test_module_does_not_import_simulator_runtime(self):
        self.assertNotIn("isaacgym", sys.modules)
        self.assertNotIn("isaaclab", sys.modules)

    def test_seeded_sampling_is_deterministic_for_multiple_objects(self):
        first = CEMOptimizer(_config(), n_objects=2, seed=17)
        second = CEMOptimizer(_config(), n_objects=2, seed=17)

        self.assertEqual(first.act_dim, 12)
        np.testing.assert_allclose(first.sample(), second.sample())

    def test_cem_update_uses_highest_reward_elites(self):
        optimizer = CEMOptimizer(_config(), seed=1)
        samples = np.arange(48, dtype=np.float64).reshape(8, 6)
        rewards = np.arange(8, dtype=np.float64)

        optimizer.update(samples, rewards)

        np.testing.assert_allclose(optimizer.mean, samples[[6, 7]].mean(axis=0))
        best, mean = optimizer.get_final_action()
        np.testing.assert_allclose(best, samples[7])
        np.testing.assert_allclose(mean, optimizer.mean)

    def test_icem_warm_start_carries_elites_once(self):
        optimizer = CEMOptimizer(_config(), seed=3)
        samples = optimizer.sample()
        rewards = np.arange(8, dtype=np.float64)
        optimizer.update(samples, rewards)
        elites = optimizer._last_elites.copy()

        optimizer.warm_start("icem")
        carried = optimizer.sample()

        np.testing.assert_allclose(carried[-len(elites) :], elites)
        self.assertIsNone(optimizer._prev_elites)

    def test_update_rejects_non_finite_reward(self):
        optimizer = CEMOptimizer(_config(), seed=5)
        rewards = np.zeros(8)
        rewards[2] = np.nan
        with self.assertRaises(ValueError):
            optimizer.update(optimizer.sample(), rewards)

    def test_nes_default_sigma_rate_is_usable(self):
        optimizer = CEMOptimizer(
            _config(cem_update_mode="nes", nes_lr_sigma=None), seed=7
        )
        samples = optimizer.sample()
        optimizer.update(samples, -np.square(samples).sum(axis=1))
        self.assertTrue(np.isfinite(optimizer.mean).all())
        self.assertTrue(np.isfinite(optimizer.std).all())


if __name__ == "__main__":
    unittest.main()

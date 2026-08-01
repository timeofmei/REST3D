import unittest

import numpy as np

from rest3d.sim.stability import (
    evaluate_replay_stability,
    quaternion_geodesic_distance_wxyz,
)


class StabilityTest(unittest.TestCase):
    def test_quaternion_distance_uses_shortest_so3_angle(self):
        identity = np.array([[1.0, 0.0, 0.0, 0.0]])
        same_rotation = -identity
        half_turn = np.array([[0.0, 1.0, 0.0, 0.0]])

        np.testing.assert_allclose(
            quaternion_geodesic_distance_wxyz(identity, same_rotation), 0.0
        )
        np.testing.assert_allclose(
            quaternion_geodesic_distance_wxyz(identity, half_turn), np.pi
        )

    def test_evaluation_thresholds_and_velocity_windows(self):
        states = np.zeros((71, 2, 13), dtype=np.float64)
        states[:, :, 3] = 1.0
        states[1:11, 0, 7] = 2.0
        states[61:, 0, 7] = 0.25
        states[60:, 0, 0] = 0.1
        states[60:, 1, 0] = 0.100001
        angle = 0.100001
        states[60:, 0, 3] = np.cos(angle / 2.0)
        states[60:, 0, 5] = np.sin(angle / 2.0)

        metrics = evaluate_replay_stability(states, evaluation_step=60)

        self.assertFalse(metrics.stable[0])
        self.assertFalse(metrics.stable[1])
        self.assertAlmostEqual(metrics.displacement_at_evaluation_m[0], 0.1)
        self.assertAlmostEqual(metrics.rotation_at_evaluation_rad[0], angle)
        self.assertAlmostEqual(metrics.early_max_linear_speed_m_s[0], 2.0)
        self.assertAlmostEqual(metrics.terminal_mean_linear_speed_m_s[0], 0.25)

    def test_exact_motion_threshold_is_stable(self):
        states = np.zeros((61, 1, 13), dtype=np.float64)
        states[:, :, 3] = 1.0
        states[60, 0, 0] = 0.1
        angle = 0.1
        states[60, 0, 3] = np.cos(angle / 2.0)
        states[60, 0, 6] = np.sin(angle / 2.0)

        metrics = evaluate_replay_stability(states)

        self.assertTrue(metrics.scene_stable)

    def test_rejects_unavailable_evaluation_frame(self):
        states = np.zeros((60, 1, 13), dtype=np.float64)
        states[:, :, 3] = 1.0
        with self.assertRaises(ValueError):
            evaluate_replay_stability(states, evaluation_step=60)


if __name__ == "__main__":
    unittest.main()

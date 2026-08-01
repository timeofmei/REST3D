import unittest

import torch

from rest3d.sim.local_cem import (
    LocalCEMEnergyWeights,
    apply_pose_deltas_wxyz,
    evaluate_local_cem_energy,
)


class LocalCEMTest(unittest.TestCase):
    def test_pose_delta_batch_uses_wxyz_axis_angle(self):
        reference = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
        samples = torch.zeros(2, 1, 6)
        samples[0, 0, :3] = torch.tensor([0.5, -0.5, 1.0])
        samples[1, 0, 5] = torch.pi

        poses = apply_pose_deltas_wxyz(reference, samples)

        torch.testing.assert_close(poses[0, 0, :3], torch.tensor([1.5, 1.5, 4.0]))
        torch.testing.assert_close(
            poses[1, 0, 3:7].abs(), torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6, rtol=0.0
        )

    def test_energy_components_cover_motion_layout_velocity_and_penetration(self):
        reference = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        placed = torch.zeros(1, 1, 13)
        placed[..., 3] = 1.0
        early = placed.clone()
        early[..., 7] = 2.0
        settled = placed.clone()
        settled[..., 0] = 0.1
        angle = torch.tensor(0.2)
        settled[..., 3] = torch.cos(angle / 2.0)
        settled[..., 5] = torch.sin(angle / 2.0)
        weights = LocalCEMEnergyWeights(
            pose_stability=1.0,
            rotation_stability=1.0,
            pose_layout=1.0,
            rotation_layout=1.0,
            velocity=1.0,
            placement_penetration=1.0,
            settled_penetration=1.0,
        )

        result = evaluate_local_cem_energy(
            placed,
            early,
            settled,
            reference,
            placement_penetration=torch.tensor([0.3]),
            settled_penetration=torch.tensor([0.4]),
            weights=weights,
        )

        torch.testing.assert_close(result["pose_stability"], torch.tensor([0.1]))
        torch.testing.assert_close(result["rotation_stability"], torch.tensor([0.2]), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(result["pose_layout"], torch.tensor([0.1]))
        torch.testing.assert_close(result["rotation_layout"], torch.tensor([0.2]), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(result["velocity"], torch.tensor([2.0]))
        torch.testing.assert_close(result["energy"], torch.tensor([3.3]), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(result["reward"], torch.tensor([-3.3]), atol=1e-6, rtol=0.0)

    def test_centroid_offset_is_rotated_for_layout(self):
        reference = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        states = torch.zeros(1, 1, 13)
        states[..., 3] = 1.0
        settled = states.clone()
        settled[..., 3] = 0.0
        settled[..., 6] = 1.0

        result = evaluate_local_cem_energy(
            states,
            states,
            settled,
            reference,
            centroid_offsets=torch.tensor([[1.0, 0.0, 0.0]]),
        )

        torch.testing.assert_close(result["pose_layout"], torch.tensor([2.0]))


if __name__ == "__main__":
    unittest.main()

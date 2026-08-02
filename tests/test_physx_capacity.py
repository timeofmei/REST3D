import unittest

from rest3d.sim.physx_capacity import (
    ISAAC_LAB_DEFAULT_RIGID_PATCH_COUNT,
    gpu_rigid_patch_capacity,
)


class PhysxCapacityTest(unittest.TestCase):
    def test_small_scenes_keep_the_isaac_lab_default(self):
        self.assertEqual(
            gpu_rigid_patch_capacity(16, 6),
            ISAAC_LAB_DEFAULT_RIGID_PATCH_COUNT,
        )

    def test_2048_by_six_scene_scales_to_262144_patches(self):
        self.assertEqual(gpu_rigid_patch_capacity(2048, 6), 2**18)

    def test_capacity_is_generic_and_monotonic(self):
        smaller = gpu_rigid_patch_capacity(1024, 8)
        larger = gpu_rigid_patch_capacity(2048, 8)
        self.assertGreaterEqual(larger, smaller)
        self.assertEqual(larger & (larger - 1), 0)

    def test_invalid_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_envs"):
            gpu_rigid_patch_capacity(0, 6)
        with self.assertRaisesRegex(ValueError, "rigid_bodies_per_env"):
            gpu_rigid_patch_capacity(1, 0)


if __name__ == "__main__":
    unittest.main()

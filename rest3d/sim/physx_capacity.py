"""Generic GPU PhysX buffer sizing policies for batched rigid-body scenes."""

from __future__ import annotations


ISAAC_LAB_DEFAULT_RIGID_PATCH_COUNT = 5 * 2**15
RIGID_PATCHES_PER_BODY_BUDGET = 16


def gpu_rigid_patch_capacity(num_envs: int, rigid_bodies_per_env: int) -> int:
    """Return a power-of-two patch capacity for a batched rigid-body scene.

    GPU PhysX buffers cannot grow dynamically.  Isaac Lab's default is enough
    for small smoke tests but can overflow when thousands of replicated scenes
    use convex-decomposition collision shapes.  The estimate is deliberately
    based only on environment and rigid-body counts, never scene object names.
    """

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if rigid_bodies_per_env < 1:
        raise ValueError("rigid_bodies_per_env must be positive")
    estimate = num_envs * rigid_bodies_per_env * RIGID_PATCHES_PER_BODY_BUDGET
    rounded = 1 << (estimate - 1).bit_length()
    return max(ISAAC_LAB_DEFAULT_RIGID_PATCH_COUNT, rounded)

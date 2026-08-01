"""Backend-neutral simulation data structures and coordinate transforms."""

from .replay_scene import (
    ReplayObjectSpec,
    ReplaySceneSpec,
    lab_states_to_rest,
    load_replay_scene,
    rest_poses_to_lab,
)
from .stability import (
    StabilityMetrics,
    evaluate_replay_stability,
    quaternion_geodesic_distance_wxyz,
)
from .local_cem import (
    LocalCEMEnergyWeights,
    apply_pose_deltas_wxyz,
    apply_pose_deltas_about_centroids_wxyz,
    apply_group_member_pose_deltas_wxyz,
    evaluate_local_cem_energy,
)
from .local_groups import (
    LocalGroupPlan,
    LocalGroupSpec,
    ObjectBounds,
    build_local_group_plan,
)
from .physics_assets import (
    PhysicsAssetPolicy,
    PhysicsAssetProperties,
    analyze_physics_asset,
    write_physics_urdf,
)

__all__ = [
    "ReplayObjectSpec",
    "ReplaySceneSpec",
    "StabilityMetrics",
    "LocalCEMEnergyWeights",
    "LocalGroupPlan",
    "LocalGroupSpec",
    "ObjectBounds",
    "PhysicsAssetPolicy",
    "PhysicsAssetProperties",
    "analyze_physics_asset",
    "apply_pose_deltas_wxyz",
    "apply_pose_deltas_about_centroids_wxyz",
    "apply_group_member_pose_deltas_wxyz",
    "build_local_group_plan",
    "evaluate_local_cem_energy",
    "evaluate_replay_stability",
    "lab_states_to_rest",
    "load_replay_scene",
    "rest_poses_to_lab",
    "write_physics_urdf",
    "quaternion_geodesic_distance_wxyz",
]

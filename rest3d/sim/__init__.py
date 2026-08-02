"""Backend-neutral simulation data structures and coordinate transforms."""

from .replay_scene import (
    ReplayObjectSpec,
    ReplaySceneSpec,
    lab_states_to_rest,
    load_replay_scene,
    rest_poses_to_lab,
    rest_states_to_lab,
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
from .global_cem import (
    GlobalEntityPlan,
    GlobalEntitySpec,
    build_global_entity_plan,
    evaluate_convex_hull_intersections_wxyz,
    evaluate_global_cem_energy,
    expand_global_root_samples_wxyz,
    validate_global_object_sets,
)
from .local_groups import (
    LocalGroupPlan,
    LocalGroupSpec,
    ObjectBounds,
    build_local_group_plan,
)
from .local_results import (
    legacy_local_group_payload,
    load_scene_states,
    local_group_execution_order,
    make_initial_scene_states,
    merge_group_candidate_states,
    validate_scene_states,
)
from .physics_assets import (
    PhysicsAssetPolicy,
    PhysicsAssetProperties,
    analyze_physics_asset,
    write_physics_urdf,
)
from .physx_capacity import gpu_rigid_patch_capacity

__all__ = [
    "ReplayObjectSpec",
    "ReplaySceneSpec",
    "StabilityMetrics",
    "GlobalEntityPlan",
    "GlobalEntitySpec",
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
    "build_global_entity_plan",
    "evaluate_convex_hull_intersections_wxyz",
    "legacy_local_group_payload",
    "load_scene_states",
    "local_group_execution_order",
    "make_initial_scene_states",
    "merge_group_candidate_states",
    "validate_scene_states",
    "evaluate_local_cem_energy",
    "evaluate_global_cem_energy",
    "evaluate_replay_stability",
    "gpu_rigid_patch_capacity",
    "lab_states_to_rest",
    "load_replay_scene",
    "rest_poses_to_lab",
    "rest_states_to_lab",
    "expand_global_root_samples_wxyz",
    "validate_global_object_sets",
    "write_physics_urdf",
    "quaternion_geodesic_distance_wxyz",
]

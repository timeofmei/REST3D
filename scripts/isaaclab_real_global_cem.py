"""Run full-scene global CEM with root-only sampling on Isaac Lab GPU PhysX."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaaclab.app import AppLauncher


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-assets", type=Path, required=True)
    parser.add_argument("--global-entities", type=Path, required=True)
    parser.add_argument("--initial-states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--cem-iters", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--early-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--translation-std-m", type=float, default=0.02)
    parser.add_argument("--rotation-std-rad", type=float, default=0.03)
    parser.add_argument("--contact-data-capacity-per-pair", type=int, default=128)
    parser.add_argument(
        "--collision-approximation",
        choices=("convex_hull", "convex_decomposition"),
        default="convex_decomposition",
    )
    parser.add_argument("--lambda-pose-stability", type=float, default=1.0)
    parser.add_argument("--lambda-rotation-stability", type=float, default=1.0)
    parser.add_argument("--lambda-pose-layout", type=float, default=6.0)
    parser.add_argument("--lambda-rotation-layout", type=float, default=1.0)
    parser.add_argument("--lambda-velocity", type=float, default=1.0)
    parser.add_argument("--lambda-placement-penetration", type=float, default=0.5)
    parser.add_argument("--lambda-settled-penetration", type=float, default=0.5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 4:
        parser.error("--num-envs must be at least 4")
    if args.cem_iters < 1:
        parser.error("--cem-iters must be positive")
    if not 1 <= args.early_steps < args.settle_steps:
        parser.error("--early-steps must be between 1 and --settle-steps - 1")
    if args.physics_dt <= 0.0:
        parser.error("--physics-dt must be positive")
    if args.translation_std_m <= 0.0 or args.rotation_std_rad <= 0.0:
        parser.error("CEM standard deviations must be positive")
    if args.contact_data_capacity_per_pair < 1:
        parser.error("--contact-data-capacity-per-pair must be positive")
    weights = (
        args.lambda_pose_stability,
        args.lambda_rotation_stability,
        args.lambda_pose_layout,
        args.lambda_rotation_layout,
        args.lambda_velocity,
        args.lambda_placement_penetration,
        args.lambda_settled_penetration,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in weights):
        parser.error("global energy weights must be finite and non-negative")
    if not args.headless:
        parser.error("the real global CEM integration requires --headless")
    if not str(args.device).startswith("cuda"):
        parser.error("the real global CEM integration requires --device cuda:N")

    args.physics_assets = args.physics_assets.expanduser().resolve(strict=True)
    args.global_entities = args.global_entities.expanduser().resolve(strict=True)
    args.initial_states = args.initial_states.expanduser().resolve(strict=True)
    with args.physics_assets.open("r", encoding="utf-8") as file:
        physics_assets = json.load(file)
    with args.global_entities.open("r", encoding="utf-8") as file:
        global_entities = json.load(file)
    with args.initial_states.open("r", encoding="utf-8") as file:
        initial_payload = json.load(file)
    plan_payload = global_entities.get("plan")
    if not isinstance(plan_payload, dict):
        parser.error("--global-entities must contain a plan object")
    scene_names = plan_payload.get("scene_names")
    if not isinstance(scene_names, list) or not scene_names:
        parser.error("global entity plan must contain scene names")
    if set(physics_assets.get("objects", {})) != set(scene_names):
        parser.error("physics assets and global entity plan object sets differ")
    initial_states = initial_payload.get("states_rest_wxyz")
    if not isinstance(initial_states, dict) or set(initial_states) != set(scene_names):
        parser.error("initial-state object set must exactly match the scene object set")
    for name in scene_names:
        values = np.asarray(initial_states[name], dtype=np.float64)
        if values.shape != (13,) or not np.isfinite(values).all():
            parser.error("initial state must contain 13 finite values: %s" % name)
        if np.linalg.norm(values[3:7]) < 1.0e-12:
            parser.error("initial state quaternion must be nonzero: %s" % name)
        initial_states[name] = values.tolist()
        initial_states[name][7:13] = [0.0] * 6
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.kit_args = "%s --/log/file=%s" % (
        args.kit_args,
        args.output_dir / "kit.log",
    )
    return args, physics_assets, plan_payload, initial_states


ARGS, PHYSICS_ASSETS, PLAN_PAYLOAD, INITIAL_STATES = _parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import (  # noqa: E402
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from rest3d.optim.cem import CEMOptimizer  # noqa: E402
from rest3d.sim.global_cem import (  # noqa: E402
    GlobalEntityPlan,
    evaluate_convex_hull_intersections_wxyz,
    evaluate_global_cem_energy,
    expand_global_root_samples_wxyz,
    validate_global_object_sets,
)
from rest3d.sim.local_cem import (  # noqa: E402
    LocalCEMEnergyWeights,
    normalize_quaternion_wxyz,
    quaternion_conjugate_wxyz,
    quaternion_geodesic_distance_wxyz,
    quaternion_multiply_wxyz,
    quaternion_rotate_wxyz,
)
from rest3d.sim.replay_scene import (  # noqa: E402
    LAB_TO_REST_QUAT_WXYZ,
    LAB_TO_REST_ROTATION,
    REST_TO_LAB_QUAT_WXYZ,
    rest_states_to_lab,
)
from rest3d.utils.mesh import load_trimesh_any  # noqa: E402


PLAN = GlobalEntityPlan.from_dict(PLAN_PAYLOAD)
ALL_NAMES = list(PLAN.scene_names)
SAMPLED_NAMES = list(PLAN.sampled_entity_names)
FIXED_NAMES = list(PLAN.fixed_names)
GROUND_FILTER_NAME = "__ground__"


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _gpu_memory_used_mib():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(completed.stdout.splitlines()[0].strip())
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


def _scene_cfg():
    rigid_objects = {}
    lab_rotation = tuple(float(value) for value in REST_TO_LAB_QUAT_WXYZ)
    manifest_fixed = {
        name for name, record in PHYSICS_ASSETS["objects"].items() if record["fixed"]
    }
    if manifest_fixed != set(FIXED_NAMES):
        raise ValueError("physics manifest and global plan fixed sets differ")
    for index, name in enumerate(ALL_NAMES):
        record = PHYSICS_ASSETS["objects"][name]
        derived_urdf = Path(record["derived_urdf"])
        if not derived_urdf.is_file():
            raise FileNotFoundError("derived URDF is missing: %s" % derived_urdf)
        kinematic = name in set(FIXED_NAMES)
        rigid_objects[name] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Body_%04d" % index,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(derived_urdf),
                usd_dir=str(
                    ARGS.output_dir / "converted_assets" / ("object_%04d" % index)
                ),
                usd_file_name="asset.usd",
                force_usd_conversion=False,
                make_instanceable=False,
                fix_base=False,
                merge_fixed_joints=True,
                joint_drive=None,
                collider_type=ARGS.collision_approximation,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=kinematic,
                    disable_gravity=kinematic,
                    linear_damping=0.3,
                    angular_damping=0.3,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=lab_rotation,
            ),
        )
    cfg = InteractiveSceneCfg(
        num_envs=ARGS.num_envs,
        env_spacing=2.0,
        replicate_physics=True,
        filter_collisions=True,
    )
    cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    cfg.objects = RigidObjectCollectionCfg(rigid_objects=rigid_objects)
    return cfg


def _cem_config():
    return SimpleNamespace(
        act_dim=6,
        cem_pop_size=ARGS.num_envs,
        cem_elite_frac=0.25,
        cem_iters_joint=ARGS.cem_iters,
        cem_iters_subtree=ARGS.cem_iters,
        cem_update_mode="cem",
        # CEMOptimizer already retains the all-time best action separately.
        # Re-inserting it into a two-member elite set can duplicate a zero-action
        # optimum and collapse both the mean and variance after one iteration.
        keep_best=False,
        update_use_only_best=False,
        std_update_mode="topk_std",
        decay_std_rate=0.95,
        reward_threshold=0.0,
        cem_final_best=True,
        cem_seed=ARGS.seed,
        init_trans_x_mean=0.0,
        init_trans_y_mean=0.0,
        init_trans_z_mean=0.0,
        init_rot_roll_mean=0.0,
        init_rot_pitch_mean=0.0,
        init_rot_yaw_mean=0.0,
        init_trans_x_std=ARGS.translation_std_m,
        init_trans_y_std=ARGS.translation_std_m,
        init_trans_z_std=ARGS.translation_std_m,
        init_rot_roll_std=ARGS.rotation_std_rad,
        init_rot_pitch_std=ARGS.rotation_std_rad,
        init_rot_yaw_std=ARGS.rotation_std_rad,
    )


def _local_states(collection, origins):
    states = collection.data.object_state_w.clone()
    states[..., :3] -= origins.unsqueeze(1)
    return states


def _lab_states_to_rest_tensor(states):
    """Convert Lab states on-device so scored geometry equals exported geometry."""

    rotation = torch.as_tensor(
        LAB_TO_REST_ROTATION, device=states.device, dtype=states.dtype
    )
    quaternion = torch.as_tensor(
        LAB_TO_REST_QUAT_WXYZ, device=states.device, dtype=states.dtype
    ).expand(*states.shape[:-1], 4)
    result = states.clone()
    result[..., :3] = states[..., :3] @ rotation.T
    result[..., 3:7] = normalize_quaternion_wxyz(
        quaternion_multiply_wxyz(quaternion, states[..., 3:7])
    )
    result[..., 7:10] = states[..., 7:10] @ rotation.T
    result[..., 10:13] = states[..., 10:13] @ rotation.T
    return result


def _create_contact_views(collection):
    active_index = {name: index for index, name in enumerate(ALL_NAMES)}
    views = []
    filters = []
    for name in ALL_NAMES:
        sensor_index = active_index[name]
        filter_names = [candidate for candidate in ALL_NAMES if candidate != name]
        filter_paths = [
            "/World/envs/env_*/Body_%04d/base" % active_index[candidate]
            for candidate in filter_names
        ]
        filter_names.append(GROUND_FILTER_NAME)
        filter_paths.append("/World/Ground/GroundPlane/CollisionPlane")
        view = collection._physics_sim_view.create_rigid_contact_view(
            "/World/envs/env_*/Body_%04d/base" % sensor_index,
            filter_patterns=filter_paths,
            max_contact_data_count=(
                ARGS.num_envs
                * len(filter_paths)
                * ARGS.contact_data_capacity_per_pair
            ),
        )
        if view.sensor_count != ARGS.num_envs or view.filter_count != len(filter_paths):
            raise RuntimeError("global contact view did not resolve all batched pairs")
        views.append(view)
        filters.append(filter_names)
    return views, filters


def _contact_snapshot(contact_views, dt):
    per_object_penetration = torch.zeros(
        ARGS.num_envs, len(ALL_NAMES), device=ARGS.device
    )
    per_object_force = torch.zeros_like(per_object_penetration)
    saturated = False
    pair_contact_counts = []
    for object_index, view in enumerate(contact_views):
        pair_force = view.get_contact_force_matrix(dt=dt).reshape(
            ARGS.num_envs, view.filter_count, 3
        )
        per_object_force[:, object_index] = torch.linalg.vector_norm(
            pair_force, dim=-1
        ).max(dim=1).values
        _, _, _, separations, counts, starts = view.get_contact_data(dt=dt)
        counts_np = counts.detach().cpu().numpy().astype(np.int64).reshape(
            ARGS.num_envs, view.filter_count
        )
        starts_np = starts.detach().cpu().numpy().astype(np.int64).reshape(
            ARGS.num_envs, view.filter_count
        )
        pair_contact_counts.append(counts_np)
        saturated = saturated or int(counts_np.sum()) >= view.max_contact_data_count
        for env_index in range(ARGS.num_envs):
            for filter_index in range(view.filter_count):
                count = int(counts_np[env_index, filter_index])
                if count:
                    start = int(starts_np[env_index, filter_index])
                    depth = torch.clamp_min(
                        -separations[start : start + count].reshape(-1), 0.0
                    ).max()
                    per_object_penetration[env_index, object_index] = torch.maximum(
                        per_object_penetration[env_index, object_index], depth
                    )
    return (
        per_object_penetration.sum(dim=1),
        per_object_penetration,
        per_object_force.max(dim=1).values,
        saturated,
        pair_contact_counts,
    )


def _load_convex_hulls():
    hulls = []
    for name in ALL_NAMES:
        source_path = Path(PHYSICS_ASSETS["objects"][name]["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError("source mesh is missing: %s" % source_path)
        mesh = load_trimesh_any(str(source_path))
        # Float64 reduces boundary-contact roundoff on the authoritative CUDA
        # path.  The PhysX state pipeline remains float32 CUDA; only the small
        # pairwise geometric predicate is promoted for robust boolean results.
        vertices = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 3:
            raise ValueError("object has no valid convex hull: %s" % name)
        hulls.append(
            torch.as_tensor(vertices, device=ARGS.device, dtype=torch.float64)
        )
    return hulls


def _maximum_entity_relative_pose_errors(reference_states, candidates):
    index_by_name = {name: index for index, name in enumerate(ALL_NAMES)}
    maximum_position = torch.zeros(
        (), device=candidates.device, dtype=candidates.dtype
    )
    maximum_rotation = torch.zeros_like(maximum_position)
    for entity in PLAN.entities:
        root_index = index_by_name[entity.root_name]
        member_indices = torch.tensor(
            [index_by_name[name] for name in entity.member_names],
            dtype=torch.long,
            device=candidates.device,
        )
        member_count = len(entity.member_names)
        reference_root_quaternion = normalize_quaternion_wxyz(
            reference_states[root_index, 3:7]
        )
        inverse_reference_root = quaternion_conjugate_wxyz(
            reference_root_quaternion
        )
        reference_local_position = quaternion_rotate_wxyz(
            inverse_reference_root.unsqueeze(0).expand(member_count, -1),
            reference_states[member_indices, :3]
            - reference_states[root_index, :3].unsqueeze(0),
        )
        reference_local_rotation = normalize_quaternion_wxyz(
            quaternion_multiply_wxyz(
                inverse_reference_root.unsqueeze(0).expand(member_count, -1),
                normalize_quaternion_wxyz(
                    reference_states[member_indices, 3:7]
                ),
            )
        )
        candidate_root_quaternion = normalize_quaternion_wxyz(
            candidates[:, root_index, 3:7]
        )
        inverse_candidate_root = quaternion_conjugate_wxyz(
            candidate_root_quaternion
        )
        candidate_local_position = quaternion_rotate_wxyz(
            inverse_candidate_root.unsqueeze(1).expand(-1, member_count, -1),
            candidates[:, member_indices, :3]
            - candidates[:, root_index, :3].unsqueeze(1),
        )
        candidate_local_rotation = normalize_quaternion_wxyz(
            quaternion_multiply_wxyz(
                inverse_candidate_root.unsqueeze(1).expand(-1, member_count, -1),
                normalize_quaternion_wxyz(candidates[:, member_indices, 3:7]),
            )
        )
        maximum_position = torch.maximum(
            maximum_position,
            torch.linalg.vector_norm(
                candidate_local_position - reference_local_position.unsqueeze(0),
                dim=-1,
            ).max(),
        )
        maximum_rotation = torch.maximum(
            maximum_rotation,
            quaternion_geodesic_distance_wxyz(
                candidate_local_rotation,
                reference_local_rotation.unsqueeze(0).expand(
                    candidates.shape[0], -1, -1
                ),
            ).max(),
        )
    return float(maximum_position.item()), float(maximum_rotation.item())


def _write_state_json(path, states_rest):
    payload = {
        "states_rest_wxyz": {
            name: states_rest[index].tolist()
            for index, name in enumerate(ALL_NAMES)
        }
    }
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _run():
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    total_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=ARGS.physics_dt, device=ARGS.device)
    )
    physics_context = sim._physics_context
    scene = InteractiveScene(_scene_cfg())
    sim.reset()
    collection = scene["objects"]
    if not isinstance(collection, RigidObjectCollection):
        raise RuntimeError("global scene did not create a rigid object collection")
    if collection.object_names != ALL_NAMES:
        raise RuntimeError("collection object order differs from the global plan")
    validate_global_object_sets(
        PLAN,
        simulated_names=collection.object_names,
        evaluated_names=ALL_NAMES,
    )
    dt = sim.get_physics_dt()
    contact_views, contact_filters = _create_contact_views(collection)
    convex_hulls = _load_convex_hulls()

    initial_rest = np.asarray([INITIAL_STATES[name] for name in ALL_NAMES])
    initial_lab = torch.as_tensor(
        rest_states_to_lab(initial_rest), dtype=torch.float32, device=ARGS.device
    )
    initial_lab[..., 7:13] = 0.0
    reference_states = initial_lab.clone()
    zero_expanded = expand_global_root_samples_wxyz(
        reference_states,
        PLAN,
        torch.zeros(
            1, len(SAMPLED_NAMES), 6, device=ARGS.device, dtype=torch.float32
        ),
    )
    zero_pose_error = float(
        torch.max(torch.abs(zero_expanded[0, :, :7] - reference_states[:, :7])).item()
    )
    centroid_offsets = torch.tensor(
        [PHYSICS_ASSETS["objects"][name]["center_of_mass_m"] for name in ALL_NAMES],
        dtype=torch.float32,
        device=ARGS.device,
    )
    actual_masses = collection.data.default_mass[0].reshape(-1)
    actual_inertias = collection.data.default_inertia[0].reshape(len(ALL_NAMES), 3, 3)
    actual_coms = collection.data.object_com_pose_b[0].clone()
    expected_masses = torch.tensor(
        [PHYSICS_ASSETS["objects"][name]["mass_kg"] for name in ALL_NAMES],
        dtype=torch.float32,
        device=actual_masses.device,
    )
    mass_error = torch.abs(actual_masses - expected_masses)
    weights = LocalCEMEnergyWeights(
        pose_stability=ARGS.lambda_pose_stability,
        rotation_stability=ARGS.lambda_rotation_stability,
        pose_layout=ARGS.lambda_pose_layout,
        rotation_layout=ARGS.lambda_rotation_layout,
        velocity=ARGS.lambda_velocity,
        placement_penetration=ARGS.lambda_placement_penetration,
        settled_penetration=ARGS.lambda_settled_penetration,
    )
    cem = CEMOptimizer(_cem_config(), n_objects=len(SAMPLED_NAMES))
    initial_mean = cem.mean.copy()
    initial_std = cem.std.copy()
    iteration_records = []
    all_time_best_rewards = []
    best_placed_state = None
    best_settled_state = None
    best_placed_rest_state = None
    best_settled_rest_state = None
    best_energy_record = None
    best_per_object_record = None
    last_state = None
    last_force = None
    last_placement_intersections = None
    last_settled_intersections = None
    last_placement_pair_matrix = None
    last_settled_pair_matrix = None
    contact_capacity_saturated = False
    maximum_fixed_position_error = 0.0
    maximum_fixed_rotation_error = 0.0
    maximum_relative_position_error = 0.0
    maximum_relative_rotation_error = 0.0
    maximum_energy_reconciliation_error = 0.0
    all_energy_shapes_complete = True
    gpu_samples = [_gpu_memory_used_mib()]
    fixed_indices = torch.tensor(
        [ALL_NAMES.index(name) for name in FIXED_NAMES],
        dtype=torch.long,
        device=ARGS.device,
    )

    simulation_started = time.perf_counter()
    for iteration in range(ARGS.cem_iters):
        samples = cem.sample()
        if iteration == 0:
            samples[0] = 0.0
        sample_tensor = torch.as_tensor(
            samples, dtype=torch.float32, device=ARGS.device
        ).reshape(ARGS.num_envs, len(SAMPLED_NAMES), 6)
        candidate_local = expand_global_root_samples_wxyz(
            reference_states, PLAN, sample_tensor
        )
        relative_position_error, relative_rotation_error = (
            _maximum_entity_relative_pose_errors(reference_states, candidate_local)
        )
        maximum_relative_position_error = max(
            maximum_relative_position_error, relative_position_error
        )
        maximum_relative_rotation_error = max(
            maximum_relative_rotation_error, relative_rotation_error
        )
        candidate_world = candidate_local.clone()
        candidate_world[..., :3] += scene.env_origins.unsqueeze(1)
        collection.write_object_state_to_sim(candidate_world)
        collection.reset()
        sim.forward()
        collection.update(0.0)
        placed = _local_states(collection, scene.env_origins)
        placed_rest = _lab_states_to_rest_tensor(placed)
        (
            placement_contact_penetration,
            _,
            _,
            saturated,
            _,
        ) = _contact_snapshot(contact_views, dt)
        contact_capacity_saturated = contact_capacity_saturated or saturated
        placement_geometry = evaluate_convex_hull_intersections_wxyz(
            placed_rest.to(dtype=torch.float64), convex_hulls
        )

        early = None
        for step in range(ARGS.settle_steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            if step + 1 == ARGS.early_steps:
                early = _local_states(collection, scene.env_origins)
        if early is None:
            raise RuntimeError("early global state was not sampled")
        settled = _local_states(collection, scene.env_origins)
        settled_rest = _lab_states_to_rest_tensor(settled)
        (
            settled_contact_penetration,
            _,
            contact_force,
            saturated,
            contact_counts,
        ) = _contact_snapshot(contact_views, dt)
        contact_capacity_saturated = contact_capacity_saturated or saturated
        settled_geometry = evaluate_convex_hull_intersections_wxyz(
            settled_rest.to(dtype=torch.float64), convex_hulls
        )
        energy = evaluate_global_cem_energy(
            PLAN,
            ALL_NAMES,
            placed,
            early,
            settled,
            reference_states[:, :7],
            centroid_offsets=centroid_offsets,
            placement_penetration=placement_geometry["total"],
            settled_penetration=settled_geometry["total"],
            placement_penetration_by_object=placement_geometry["per_object"],
            settled_penetration_by_object=settled_geometry["per_object"],
            weights=weights,
        )
        per_object = energy["per_object"]
        all_energy_shapes_complete = all_energy_shapes_complete and all(
            tuple(values.shape) == (ARGS.num_envs, len(ALL_NAMES))
            for values in per_object.values()
        )
        for component_name, per_object_values in per_object.items():
            aggregate_values = energy[component_name]
            reconciliation_error = torch.max(
                torch.abs(per_object_values.sum(dim=1) - aggregate_values)
            )
            maximum_energy_reconciliation_error = max(
                maximum_energy_reconciliation_error,
                float(reconciliation_error.item()),
            )
        rewards = energy["reward"].detach().cpu().numpy()
        previous_best_reward = float(cem._best_reward)
        cem.update(samples, rewards)
        best_index = int(np.argmax(rewards))
        if float(rewards[best_index]) > previous_best_reward:
            best_placed_state = placed[best_index].detach().cpu().numpy().copy()
            best_settled_state = settled[best_index].detach().cpu().numpy().copy()
            best_placed_rest_state = (
                placed_rest[best_index].detach().cpu().numpy().copy()
            )
            best_settled_rest_state = (
                settled_rest[best_index].detach().cpu().numpy().copy()
            )
            best_energy_record = {
                name: float(values[best_index].item())
                for name, values in energy.items()
                if name not in {"reward", "per_object"}
            }
            best_per_object_record = {
                name: values[best_index].detach().cpu().numpy().copy()
                for name, values in per_object.items()
            }
        if len(FIXED_NAMES):
            fixed_position_error = torch.linalg.vector_norm(
                settled[:, fixed_indices, :3] - placed[:, fixed_indices, :3], dim=-1
            ).max()
            fixed_rotation_error = quaternion_geodesic_distance_wxyz(
                settled[:, fixed_indices, 3:7], placed[:, fixed_indices, 3:7]
            ).max()
            maximum_fixed_position_error = max(
                maximum_fixed_position_error, float(fixed_position_error.item())
            )
            maximum_fixed_rotation_error = max(
                maximum_fixed_rotation_error, float(fixed_rotation_error.item())
            )
        all_time_best_rewards.append(float(cem._best_reward))
        entity_energy = {}
        for entity in PLAN.entities:
            indices = [ALL_NAMES.index(name) for name in entity.member_names]
            entity_energy[entity.root_name] = float(
                per_object["energy"][best_index, indices].sum().item()
            )
        iteration_records.append(
            {
                "iteration": iteration + 1,
                "best_environment": best_index,
                "current_best_reward": float(rewards[best_index]),
                "all_time_best_reward": float(cem._best_reward),
                "reward_mean": float(rewards.mean()),
                "zero_candidate_reward": float(rewards[0]) if iteration == 0 else None,
                "best_placement_intersection_count": float(
                    placement_geometry["total"][best_index].item()
                ),
                "best_settled_intersection_count": float(
                    settled_geometry["total"][best_index].item()
                ),
                "best_placement_contact_penetration_m": float(
                    placement_contact_penetration[best_index].item()
                ),
                "best_settled_contact_penetration_m": float(
                    settled_contact_penetration[best_index].item()
                ),
                "best_contact_force_n": float(contact_force[best_index].item()),
                "best_energy_components": {
                    name: float(values[best_index].item())
                    for name, values in energy.items()
                    if name not in {"reward", "per_object"}
                },
                "best_entity_energy": entity_energy,
                "contact_pair_counts": [
                    counts[best_index].tolist() for counts in contact_counts
                ],
            }
        )
        last_state = settled
        last_force = contact_force
        last_placement_intersections = placement_geometry["total"]
        last_settled_intersections = settled_geometry["total"]
        last_placement_pair_matrix = placement_geometry["pair_matrix"]
        last_settled_pair_matrix = settled_geometry["pair_matrix"]
        gpu_samples.append(_gpu_memory_used_mib())

    torch.cuda.synchronize()
    simulation_seconds = time.perf_counter() - simulation_started
    if last_state is None or last_force is None:
        raise RuntimeError("global CEM produced no simulation state")
    if (
        best_placed_state is None
        or best_settled_state is None
        or best_energy_record is None
        or best_per_object_record is None
        or best_placed_rest_state is None
        or best_settled_rest_state is None
    ):
        raise RuntimeError("global CEM did not retain an all-time best full-scene state")
    cuda_probe_value = float((last_state.square().sum() + last_force.square().sum()).item())
    finite_gpu_samples = [sample for sample in gpu_samples if sample is not None]
    best_export_placed_geometry = evaluate_convex_hull_intersections_wxyz(
        torch.as_tensor(
            best_placed_rest_state, device=ARGS.device, dtype=torch.float64
        ).unsqueeze(0),
        convex_hulls,
    )
    best_export_settled_geometry = evaluate_convex_hull_intersections_wxyz(
        torch.as_tensor(
            best_settled_rest_state, device=ARGS.device, dtype=torch.float64
        ).unsqueeze(0),
        convex_hulls,
    )
    best_export_placed_count = float(best_export_placed_geometry["total"].item())
    best_export_settled_count = float(best_export_settled_geometry["total"].item())
    checks = {
        "physics_uses_gpu_sim": bool(physics_context.use_gpu_sim),
        "physics_uses_gpu_pipeline": bool(physics_context.use_gpu_pipeline),
        "physics_broadphase_is_gpu": physics_context.get_broadphase_type() == "GPU",
        "simulated_object_set_matches_scene": set(collection.object_names)
        == set(ALL_NAMES),
        "evaluated_object_set_matches_scene": all_energy_shapes_complete,
        "sampled_entity_set_is_scene_subset": set(SAMPLED_NAMES) <= set(ALL_NAMES),
        "only_global_entities_are_sampled": len(cem.mean)
        == len(SAMPLED_NAMES) * 6,
        "all_scene_objects_in_every_candidate": tuple(last_state.shape)
        == (ARGS.num_envs, len(ALL_NAMES), 13),
        "all_objects_have_convex_hulls": len(convex_hulls) == len(ALL_NAMES)
        and all(vertices.shape[0] >= 4 for vertices in convex_hulls),
        "all_objects_have_contact_views": len(contact_views) == len(ALL_NAMES),
        "all_contact_views_include_ground_and_other_objects": all(
            len(names) == len(ALL_NAMES) for names in contact_filters
        ),
        "state_tensor_is_cuda": last_state.device.type == "cuda",
        "contact_tensor_is_cuda": last_force.device.type == "cuda",
        "states_are_finite": bool(torch.isfinite(last_state).all()),
        "mass_properties_match_manifest": float(mass_error.max().item()) < 1.0e-4,
        "masses_are_finite_positive": bool(
            torch.isfinite(actual_masses).all() and (actual_masses > 0.0).all()
        ),
        "inertias_are_finite_positive": bool(
            torch.isfinite(actual_inertias).all()
            and (torch.linalg.eigvalsh(actual_inertias) > 0.0).all()
        ),
        "contact_capacity_not_saturated": not contact_capacity_saturated,
        "fixed_objects_remained_kinematic": (
            maximum_fixed_position_error < 1.0e-5
            and maximum_fixed_rotation_error < 1.0e-5
        ),
        "zero_sample_preserves_reference_poses": zero_pose_error < 1.0e-5,
        "hierarchy_relative_positions_preserved": maximum_relative_position_error
        < 1.0e-4,
        "hierarchy_relative_rotations_preserved": maximum_relative_rotation_error
        < 1.0e-4,
        "all_object_energy_reconciles_with_global": (
            maximum_energy_reconciliation_error < 1.0e-4
        ),
        "gjk_intersection_tensors_are_cuda": (
            last_placement_intersections is not None
            and last_settled_intersections is not None
            and last_placement_intersections.device.type == "cuda"
            and last_settled_intersections.device.type == "cuda"
        ),
        "exported_best_geometry_matches_scored_energy": (
            best_export_placed_count
            == float(best_energy_record["placement_penetration"])
            and best_export_settled_count
            == float(best_energy_record["settled_penetration"])
        ),
        "zero_action_baseline_was_evaluated": bool(
            np.isfinite(iteration_records[0]["zero_candidate_reward"])
        ),
        "cem_distribution_updated": not (
            np.allclose(cem.mean, initial_mean)
            and np.allclose(cem.std, initial_std)
        ),
        "cem_best_reward_is_monotonic": all(
            later >= earlier
            for earlier, later in zip(all_time_best_rewards, all_time_best_rewards[1:])
        ),
        "cem_best_action_exists": cem._best_action is not None,
        "best_full_scene_states_exist": best_placed_state.shape
        == (len(ALL_NAMES), 13),
        "cuda_probe_is_finite": bool(np.isfinite(cuda_probe_value)),
        "torch_has_sm120": "sm_120" in torch.cuda.get_arch_list(),
    }

    best_placed_rest = best_placed_rest_state
    best_settled_rest = best_settled_rest_state
    _write_state_json(
        ARGS.output_dir / "best_global_candidate_states.json", best_placed_rest
    )
    _write_state_json(
        ARGS.output_dir / "best_global_settled_states.json", best_settled_rest
    )
    np.savez_compressed(
        ARGS.output_dir / "real_global_cem_metrics.npz",
        scene_names=np.asarray(ALL_NAMES),
        sampled_names=np.asarray(SAMPLED_NAMES),
        initial_mean=initial_mean,
        initial_std=initial_std,
        final_mean=cem.mean,
        final_std=cem.std,
        best_action=cem._best_action,
        best_reward=np.asarray(cem._best_reward),
        final_iteration_states_lab=last_state.detach().cpu().numpy(),
        placement_intersection_count=last_placement_intersections.detach()
        .cpu()
        .numpy(),
        settled_intersection_count=last_settled_intersections.detach()
        .cpu()
        .numpy(),
        placement_intersection_pair_matrix=last_placement_pair_matrix.detach()
        .cpu()
        .numpy(),
        settled_intersection_pair_matrix=last_settled_pair_matrix.detach()
        .cpu()
        .numpy(),
        best_placed_states_rest=best_placed_rest,
        best_settled_states_rest=best_settled_rest,
        actual_masses=actual_masses.detach().cpu().numpy(),
        actual_inertias=actual_inertias.detach().cpu().numpy(),
        actual_coms=actual_coms.detach().cpu().numpy(),
        iteration_best_rewards=np.asarray(all_time_best_rewards),
    )
    per_object_results = {
        name: {
            component: float(best_per_object_record[component][index])
            for component in best_per_object_record
        }
        for index, name in enumerate(ALL_NAMES)
    }
    entity_results = {
        entity.root_name: {
            "member_names": list(entity.member_names),
            "energy": float(
                sum(
                    per_object_results[name]["energy"]
                    for name in entity.member_names
                )
            ),
        }
        for entity in PLAN.entities
    }
    best_candidate = {
        "sampled_names": SAMPLED_NAMES,
        "simulated_names": ALL_NAMES,
        "evaluated_names": ALL_NAMES,
        "best_action": cem._best_action.tolist(),
        "best_reward": float(cem._best_reward),
        "energy_components": best_energy_record,
        "per_object_energy": per_object_results,
        "per_entity_energy": entity_results,
        "candidate_states": str(
            ARGS.output_dir / "best_global_candidate_states.json"
        ),
        "settled_states": str(ARGS.output_dir / "best_global_settled_states.json"),
    }
    with (ARGS.output_dir / "best_global_candidate.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(best_candidate, file, indent=2)
        file.write("\n")

    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "versions": {
            "python": platform.python_version(),
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab_package": _package_version("isaaclab"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
        "inputs": {
            "physics_assets": str(ARGS.physics_assets),
            "global_entities": str(ARGS.global_entities),
            "initial_states": str(ARGS.initial_states),
        },
        "plan": PLAN.to_dict(),
        "simulation": {
            "num_envs": ARGS.num_envs,
            "scene_object_count_per_env": len(ALL_NAMES),
            "sampled_entity_count": len(SAMPLED_NAMES),
            "state_tensor_device": str(last_state.device),
            "state_tensor_shape": list(last_state.shape),
            "contact_tensor_device": str(last_force.device),
            "steps_per_iteration": ARGS.settle_steps,
            "cem_iterations": ARGS.cem_iters,
            "total_physics_steps": ARGS.settle_steps * ARGS.cem_iters,
            "simulation_seconds": simulation_seconds,
            "physics_steps_per_second": (
                ARGS.settle_steps * ARGS.cem_iters / simulation_seconds
            ),
            "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "system_gpu_memory_used_mib_peak": max(finite_gpu_samples)
            if finite_gpu_samples
            else None,
            "cuda_probe_value": cuda_probe_value,
            "collision_approximation": ARGS.collision_approximation,
            "maximum_hierarchy_relative_position_error_m": (
                maximum_relative_position_error
            ),
            "maximum_hierarchy_relative_rotation_error_rad": (
                maximum_relative_rotation_error
            ),
            "maximum_energy_reconciliation_error": (
                maximum_energy_reconciliation_error
            ),
            "maximum_fixed_position_error_m": maximum_fixed_position_error,
            "maximum_fixed_rotation_error_rad": maximum_fixed_rotation_error,
        },
        "physics_assets": {
            name: {
                "expected_mass_kg": float(expected_masses[index].item()),
                "actual_mass_kg": float(actual_masses[index].item()),
                "mass_error_kg": float(mass_error[index].item()),
                "actual_center_of_mass_pose_wxyz": actual_coms[index].tolist(),
                "actual_inertia_kg_m2": actual_inertias[index].tolist(),
            }
            for index, name in enumerate(ALL_NAMES)
        },
        "contacts": {
            "sensor_names": ALL_NAMES,
            "filter_names": contact_filters,
        },
        "geometry": {
            "penetration_metric": "pairwise-convex-hull-gjk-intersection-count",
            "coordinate_frame": "rest3d-y-up-export-frame",
            "tensor_dtype": "torch.float64",
            "best_exported_placement_intersection_count": (
                best_export_placed_count
            ),
            "best_exported_settled_intersection_count": (
                best_export_settled_count
            ),
            "convex_hull_vertex_counts": {
                name: int(convex_hulls[index].shape[0])
                for index, name in enumerate(ALL_NAMES)
            },
        },
        "energy": {
            "weights": {
                "pose_stability": weights.pose_stability,
                "rotation_stability": weights.rotation_stability,
                "pose_layout": weights.pose_layout,
                "rotation_layout": weights.rotation_layout,
                "velocity": weights.velocity,
                "placement_penetration": weights.placement_penetration,
                "settled_penetration": weights.settled_penetration,
            },
            "best_components": best_energy_record,
            "best_per_object": per_object_results,
            "best_per_entity": entity_results,
        },
        "cem": {
            "seed": ARGS.seed,
            "population": ARGS.num_envs,
            "sampled_names": SAMPLED_NAMES,
            "action_dimension": cem.act_dim,
            "elite_count": cem.n_elite,
            "best_action": cem._best_action.tolist(),
            "best_reward": float(cem._best_reward),
            "iterations": iteration_records,
        },
        "outputs": {
            "candidate_states": str(
                ARGS.output_dir / "best_global_candidate_states.json"
            ),
            "settled_states": str(
                ARGS.output_dir / "best_global_settled_states.json"
            ),
            "candidate": str(ARGS.output_dir / "best_global_candidate.json"),
            "metrics": str(ARGS.output_dir / "real_global_cem_metrics.npz"),
        },
        "total_seconds": time.perf_counter() - total_started,
    }
    with (ARGS.output_dir / "real_global_cem_results.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    if not result["passed"]:
        raise RuntimeError("real global CEM checks failed: %s" % checks)
    return result


def main():
    try:
        result = _run()
        print(json.dumps(result, indent=2))
    except BaseException as error:
        traceback.print_exc()
        failure = {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            with (ARGS.output_dir / "real_global_cem_failure.json").open(
                "x", encoding="utf-8"
            ) as file:
                json.dump(failure, file, indent=2)
                file.write("\n")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

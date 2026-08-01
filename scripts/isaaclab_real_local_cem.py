"""Run one scene-tree local group through batched Isaac Lab GPU CEM."""

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
    parser.add_argument("--local-groups", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--cem-iters", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--early-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--translation-std-m", type=float, default=0.03)
    parser.add_argument("--rotation-std-rad", type=float, default=0.04)
    parser.add_argument("--contact-data-capacity-per-pair", type=int, default=64)
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
    if not args.headless:
        parser.error("the real local CEM integration requires --headless")
    if not str(args.device).startswith("cuda"):
        parser.error("the real local CEM integration requires --device cuda:N")
    args.physics_assets = args.physics_assets.expanduser().resolve(strict=True)
    args.local_groups = args.local_groups.expanduser().resolve(strict=True)
    with args.physics_assets.open("r", encoding="utf-8") as file:
        physics_assets = json.load(file)
    with args.local_groups.open("r", encoding="utf-8") as file:
        local_groups = json.load(file)
    groups = local_groups["groups"]
    if not 0 <= args.group_index < len(groups):
        parser.error("--group-index is outside the local group plan")
    if set(physics_assets["objects"]) != set(local_groups["scene_names"]):
        parser.error("physics asset and local group object sets differ")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir = output_dir
    args.kit_args = f"{args.kit_args} --/log/file={output_dir / 'kit.log'}".strip()
    return args, physics_assets, local_groups, groups[args.group_index]


ARGS, PHYSICS_ASSETS, LOCAL_GROUPS, GROUP = _parse_args()
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
from rest3d.sim.local_cem import (  # noqa: E402
    LocalCEMEnergyWeights,
    apply_group_member_pose_deltas_wxyz,
    evaluate_local_cem_energy,
)
from rest3d.sim.replay_scene import (  # noqa: E402
    REST_TO_LAB_QUAT_WXYZ,
    lab_states_to_rest,
)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _gpu_memory_used_mib() -> int | None:
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


def _active_names() -> list[str]:
    included = set(GROUP["member_names"]) | set(GROUP["context_names"])
    return [name for name in LOCAL_GROUPS["scene_names"] if name in included]


ACTIVE_NAMES = _active_names()
MEMBER_NAMES = list(GROUP["member_names"])
SAMPLED_NAMES = list(GROUP["direct_child_names"])
DYNAMIC_NAMES = [name for name in MEMBER_NAMES if name != GROUP["root_name"]]


def _scene_cfg() -> InteractiveSceneCfg:
    rigid_objects = {}
    lab_rotation = tuple(float(value) for value in REST_TO_LAB_QUAT_WXYZ)
    for index, name in enumerate(ACTIVE_NAMES):
        record = PHYSICS_ASSETS["objects"][name]
        derived_urdf = Path(record["derived_urdf"])
        if not derived_urdf.is_file():
            raise FileNotFoundError(f"derived URDF is missing: {derived_urdf}")
        kinematic = name == GROUP["root_name"] or name in GROUP["context_names"]
        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Body_{index:04d}",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(derived_urdf),
                usd_dir=str(ARGS.output_dir / "converted_assets" / f"object_{index:04d}"),
                usd_file_name="asset.usd",
                force_usd_conversion=False,
                make_instanceable=False,
                fix_base=False,
                merge_fixed_joints=True,
                joint_drive=None,
                collider_type="convex_decomposition",
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


def _cem_config() -> SimpleNamespace:
    return SimpleNamespace(
        act_dim=6,
        cem_pop_size=ARGS.num_envs,
        cem_elite_frac=0.25,
        cem_iters_joint=ARGS.cem_iters,
        cem_iters_subtree=ARGS.cem_iters,
        cem_update_mode="cem",
        keep_best=True,
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


def _local_states(collection: RigidObjectCollection, origins: torch.Tensor) -> torch.Tensor:
    states = collection.data.object_state_w.clone()
    states[..., :3] -= origins.unsqueeze(1)
    return states


def _group_pose_indices(collection: RigidObjectCollection):
    active_index = {name: index for index, name in enumerate(collection.object_names)}
    member_indices = torch.tensor(
        [active_index[name] for name in MEMBER_NAMES], device=ARGS.device, dtype=torch.long
    )
    sampled_member_indices = torch.tensor(
        [MEMBER_NAMES.index(name) for name in SAMPLED_NAMES],
        device=ARGS.device,
        dtype=torch.long,
    )
    owner_by_name = {name: index for index, name in enumerate(SAMPLED_NAMES)}
    for descendant, owner in GROUP.get("driven_descendants", []):
        owner_by_name[descendant] = owner_by_name[owner]
    owner_sample_indices = torch.tensor(
        [owner_by_name.get(name, -1) for name in MEMBER_NAMES],
        device=ARGS.device,
        dtype=torch.long,
    )
    dynamic_member_indices = torch.tensor(
        [MEMBER_NAMES.index(name) for name in DYNAMIC_NAMES],
        device=ARGS.device,
        dtype=torch.long,
    )
    dynamic_active_indices = torch.tensor(
        [active_index[name] for name in DYNAMIC_NAMES],
        device=ARGS.device,
        dtype=torch.long,
    )
    kinematic_active_indices = torch.tensor(
        [
            active_index[name]
            for name in ACTIVE_NAMES
            if name == GROUP["root_name"] or name in GROUP["context_names"]
        ],
        device=ARGS.device,
        dtype=torch.long,
    )
    return (
        active_index,
        member_indices,
        sampled_member_indices,
        owner_sample_indices,
        dynamic_member_indices,
        dynamic_active_indices,
        kinematic_active_indices,
    )


def _create_contact_views(collection, active_index):
    views = []
    filters = []
    for name in DYNAMIC_NAMES:
        sensor_index = active_index[name]
        filter_names = [candidate for candidate in ACTIVE_NAMES if candidate != name]
        filter_paths = [
            f"/World/envs/env_*/Body_{active_index[candidate]:04d}/base"
            for candidate in filter_names
        ]
        view = collection._physics_sim_view.create_rigid_contact_view(
            f"/World/envs/env_*/Body_{sensor_index:04d}/base",
            filter_patterns=filter_paths,
            max_contact_data_count=(
                ARGS.num_envs
                * max(1, len(filter_paths))
                * ARGS.contact_data_capacity_per_pair
            ),
        )
        if view.sensor_count != ARGS.num_envs or view.filter_count != len(filter_paths):
            raise RuntimeError("contact view did not resolve the expected batched object pairs")
        views.append(view)
        filters.append(filter_names)
    return views, filters


def _contact_snapshot(contact_views, dt):
    penetration = torch.zeros(ARGS.num_envs, device=ARGS.device)
    maximum_force = torch.zeros(ARGS.num_envs, device=ARGS.device)
    saturated = False
    pair_contact_counts = []
    for view in contact_views:
        pair_force = view.get_contact_force_matrix(dt=dt).reshape(
            ARGS.num_envs, view.filter_count, 3
        )
        maximum_force = torch.maximum(
            maximum_force,
            torch.linalg.vector_norm(pair_force, dim=-1).max(dim=1).values,
        )
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
                    penetration[env_index] = torch.maximum(
                        penetration[env_index], depth
                    )
    return penetration, maximum_force, saturated, pair_contact_counts


def _run() -> dict:
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    total_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    sim = SimulationContext(sim_utils.SimulationCfg(dt=ARGS.physics_dt, device=ARGS.device))
    physics_context = sim._physics_context
    scene = InteractiveScene(_scene_cfg())
    sim.reset()
    collection: RigidObjectCollection = scene["objects"]
    dt = sim.get_physics_dt()
    if collection.object_names != ACTIVE_NAMES:
        raise RuntimeError("collection object order differs from the local group plan")
    (
        active_index,
        member_indices,
        sampled_member_indices,
        owner_sample_indices,
        dynamic_member_indices,
        dynamic_active_indices,
        kinematic_active_indices,
    ) = _group_pose_indices(collection)
    contact_views, contact_filters = _create_contact_views(collection, active_index)

    default_local = collection.data.default_object_state.clone()
    reference_member_poses = default_local[0, member_indices, :7]
    centroid_offsets = torch.tensor(
        [PHYSICS_ASSETS["objects"][name]["center_of_mass_m"] for name in MEMBER_NAMES],
        dtype=torch.float32,
        device=ARGS.device,
    )
    actual_masses = collection.data.default_mass[0].reshape(-1)
    actual_inertias = collection.data.default_inertia[0].reshape(len(ACTIVE_NAMES), 3, 3)
    actual_coms = collection.data.object_com_pose_b[0].clone()
    expected_masses = torch.tensor(
        [PHYSICS_ASSETS["objects"][name]["mass_kg"] for name in ACTIVE_NAMES],
        dtype=torch.float32,
        device=actual_masses.device,
    )
    mass_error = torch.abs(actual_masses - expected_masses)

    cem = CEMOptimizer(_cem_config(), n_objects=len(SAMPLED_NAMES))
    initial_mean = cem.mean.copy()
    initial_std = cem.std.copy()
    weights = LocalCEMEnergyWeights()
    iteration_records = []
    all_time_best_rewards = []
    last_state = None
    last_force = None
    last_placement_penetration = None
    last_settled_penetration = None
    contact_capacity_saturated = False
    maximum_kinematic_position_error = 0.0
    maximum_kinematic_rotation_error = 0.0
    best_placed_state = None
    best_settled_state = None
    best_energy_record = None
    gpu_samples = [_gpu_memory_used_mib()]

    simulation_started = time.perf_counter()
    for iteration in range(ARGS.cem_iters):
        samples = cem.sample()
        if iteration == 0:
            samples[0] = 0.0
        sample_tensor = torch.as_tensor(
            samples, dtype=torch.float32, device=ARGS.device
        ).reshape(ARGS.num_envs, len(SAMPLED_NAMES), 6)
        member_poses = apply_group_member_pose_deltas_wxyz(
            reference_member_poses,
            centroid_offsets,
            sample_tensor,
            sampled_member_indices,
            owner_sample_indices,
        )
        candidate_states = default_local.clone()
        candidate_states[..., :3] += scene.env_origins.unsqueeze(1)
        candidate_states[:, member_indices, :7] = member_poses
        candidate_states[:, member_indices, :3] += scene.env_origins.unsqueeze(1)
        candidate_states[..., 7:] = 0.0
        collection.write_object_state_to_sim(candidate_states)
        collection.reset()
        sim.forward()
        collection.update(0.0)
        placed = _local_states(collection, scene.env_origins)
        placement_penetration, _, saturated, _ = _contact_snapshot(contact_views, dt)
        contact_capacity_saturated = contact_capacity_saturated or saturated

        early = None
        for step in range(ARGS.settle_steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            if step + 1 == ARGS.early_steps:
                early = _local_states(collection, scene.env_origins)
        if early is None:
            raise RuntimeError("early local group state was not sampled")
        settled = _local_states(collection, scene.env_origins)
        settled_penetration, contact_force, saturated, contact_counts = _contact_snapshot(
            contact_views, dt
        )
        contact_capacity_saturated = contact_capacity_saturated or saturated
        energy = evaluate_local_cem_energy(
            placed[:, dynamic_active_indices],
            early[:, dynamic_active_indices],
            settled[:, dynamic_active_indices],
            reference_member_poses[dynamic_member_indices],
            centroid_offsets=centroid_offsets[dynamic_member_indices],
            placement_penetration=placement_penetration,
            settled_penetration=settled_penetration,
            weights=weights,
        )
        rewards = energy["reward"].detach().cpu().numpy()
        previous_best_reward = float(cem._best_reward)
        cem.update(samples, rewards)
        best_index = int(np.argmax(rewards))
        if float(rewards[best_index]) > previous_best_reward:
            best_placed_state = placed[best_index].detach().cpu().numpy().copy()
            best_settled_state = settled[best_index].detach().cpu().numpy().copy()
            best_energy_record = {
                name: float(values[best_index].item())
                for name, values in energy.items()
                if name != "reward"
            }
        kinematic_state = settled[:, kinematic_active_indices]
        kinematic_reference = default_local[:, kinematic_active_indices]
        position_error = torch.linalg.vector_norm(
            kinematic_state[..., :3] - kinematic_reference[..., :3], dim=-1
        ).max()
        state_quaternion = kinematic_state[..., 3:7]
        reference_quaternion = kinematic_reference[..., 3:7]
        quaternion_alignment = (
            torch.abs(torch.sum(state_quaternion * reference_quaternion, dim=-1))
            / (
                torch.linalg.vector_norm(state_quaternion, dim=-1)
                * torch.linalg.vector_norm(reference_quaternion, dim=-1)
            ).clamp_min(1.0e-12)
        ).clamp(0.0, 1.0)
        rotation_error = (2.0 * torch.acos(quaternion_alignment)).max()
        maximum_kinematic_position_error = max(
            maximum_kinematic_position_error, float(position_error.item())
        )
        maximum_kinematic_rotation_error = max(
            maximum_kinematic_rotation_error, float(rotation_error.item())
        )
        all_time_best_rewards.append(float(cem._best_reward))
        iteration_records.append(
            {
                "iteration": iteration + 1,
                "best_environment": best_index,
                "current_best_reward": float(rewards[best_index]),
                "all_time_best_reward": float(cem._best_reward),
                "reward_mean": float(rewards.mean()),
                "zero_candidate_reward": float(rewards[0]) if iteration == 0 else None,
                "best_placement_penetration_m": float(
                    placement_penetration[best_index].item()
                ),
                "best_settled_penetration_m": float(
                    settled_penetration[best_index].item()
                ),
                "best_contact_force_n": float(contact_force[best_index].item()),
                "best_energy_components": {
                    name: float(values[best_index].item())
                    for name, values in energy.items()
                    if name != "reward"
                },
                "contact_pair_counts": [counts[best_index].tolist() for counts in contact_counts],
            }
        )
        last_state = settled
        last_force = contact_force
        last_placement_penetration = placement_penetration
        last_settled_penetration = settled_penetration
        gpu_samples.append(_gpu_memory_used_mib())

    torch.cuda.synchronize()
    simulation_seconds = time.perf_counter() - simulation_started
    if last_state is None or last_force is None:
        raise RuntimeError("local CEM produced no simulation state")
    if best_placed_state is None or best_settled_state is None or best_energy_record is None:
        raise RuntimeError("local CEM did not retain an all-time best candidate state")
    cuda_probe_value = float((last_state.square().sum() + last_force.square().sum()).item())
    finite_gpu_samples = [sample for sample in gpu_samples if sample is not None]
    checks = {
        "physics_uses_gpu_sim": bool(physics_context.use_gpu_sim),
        "physics_uses_gpu_pipeline": bool(physics_context.use_gpu_pipeline),
        "physics_broadphase_is_gpu": physics_context.get_broadphase_type() == "GPU",
        "all_planned_objects_loaded": collection.object_names == ACTIVE_NAMES,
        "all_group_members_loaded": set(MEMBER_NAMES) <= set(collection.object_names),
        "all_context_objects_loaded": set(GROUP["context_names"]) <= set(collection.object_names),
        "state_tensor_is_cuda": last_state.device.type == "cuda",
        "contact_tensor_is_cuda": last_force.device.type == "cuda",
        "state_shape_is_complete": tuple(last_state.shape)
        == (ARGS.num_envs, len(ACTIVE_NAMES), 13),
        "states_are_finite": bool(torch.isfinite(last_state).all()),
        "mass_properties_match_manifest": float(mass_error.max().item()) < 1.0e-4,
        "mass_properties_are_not_uniform": int(torch.unique(actual_masses).numel()) > 1,
        "inertias_are_finite_positive": bool(
            torch.isfinite(actual_inertias).all()
            and (torch.linalg.eigvalsh(actual_inertias) > 0.0).all()
        ),
        "contact_capacity_not_saturated": not contact_capacity_saturated,
        "kinematic_root_and_context_remained_fixed": (
            maximum_kinematic_position_error < 1.0e-5
            and maximum_kinematic_rotation_error < 1.0e-5
        ),
        "zero_action_baseline_was_evaluated": bool(
            np.isfinite(iteration_records[0]["zero_candidate_reward"])
        ),
        "cem_distribution_updated": not np.allclose(cem.mean, initial_mean),
        "cem_best_reward_is_monotonic": all(
            later >= earlier
            for earlier, later in zip(all_time_best_rewards, all_time_best_rewards[1:])
        ),
        "cem_best_action_exists": cem._best_action is not None,
        "cem_best_candidate_state_exists": best_placed_state is not None,
        "cuda_probe_is_finite": bool(np.isfinite(cuda_probe_value)),
        "torch_has_sm120": "sm_120" in torch.cuda.get_arch_list(),
    }
    np.savez_compressed(
        ARGS.output_dir / "real_local_cem_metrics.npz",
        active_names=np.asarray(ACTIVE_NAMES),
        member_names=np.asarray(MEMBER_NAMES),
        sampled_names=np.asarray(SAMPLED_NAMES),
        initial_mean=initial_mean,
        initial_std=initial_std,
        final_mean=cem.mean,
        final_std=cem.std,
        best_action=cem._best_action,
        best_reward=np.asarray(cem._best_reward),
        settled_states=last_state.detach().cpu().numpy(),
        placement_penetration=last_placement_penetration.detach().cpu().numpy(),
        settled_penetration=last_settled_penetration.detach().cpu().numpy(),
        actual_masses=actual_masses.detach().cpu().numpy(),
        actual_inertias=actual_inertias.detach().cpu().numpy(),
        actual_coms=actual_coms.detach().cpu().numpy(),
        iteration_best_rewards=np.asarray(all_time_best_rewards),
        best_placed_states_lab=best_placed_state,
        best_settled_states_lab=best_settled_state,
        best_placed_states_rest=lab_states_to_rest(best_placed_state),
        best_settled_states_rest=lab_states_to_rest(best_settled_state),
    )
    best_placed_rest = lab_states_to_rest(best_placed_state)
    best_settled_rest = lab_states_to_rest(best_settled_state)
    best_candidate = {
        "group_index": GROUP["index"],
        "root_name": GROUP["root_name"],
        "sampled_names": SAMPLED_NAMES,
        "active_names": ACTIVE_NAMES,
        "best_action": cem._best_action.tolist(),
        "best_reward": float(cem._best_reward),
        "energy_components": best_energy_record,
        "placed_states_rest_wxyz": {
            name: best_placed_rest[index].tolist()
            for index, name in enumerate(ACTIVE_NAMES)
        },
        "settled_states_rest_wxyz": {
            name: best_settled_rest[index].tolist()
            for index, name in enumerate(ACTIVE_NAMES)
        },
    }
    with (ARGS.output_dir / "best_local_group_candidate.json").open(
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
        "group": GROUP,
        "simulation": {
            "num_envs": ARGS.num_envs,
            "active_names": ACTIVE_NAMES,
            "active_object_count_per_env": len(ACTIVE_NAMES),
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
            "maximum_kinematic_position_error_m": maximum_kinematic_position_error,
            "maximum_kinematic_rotation_error_rad": maximum_kinematic_rotation_error,
        },
        "physics_assets": {
            name: {
                "expected_mass_kg": float(expected_masses[index].item()),
                "actual_mass_kg": float(actual_masses[index].item()),
                "mass_error_kg": float(mass_error[index].item()),
                "actual_center_of_mass_pose_wxyz": actual_coms[index].tolist(),
                "actual_inertia_kg_m2": actual_inertias[index].tolist(),
            }
            for index, name in enumerate(ACTIVE_NAMES)
        },
        "contacts": {
            "sensor_names": DYNAMIC_NAMES,
            "filter_names": contact_filters,
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
        "total_seconds": time.perf_counter() - total_started,
    }
    with (ARGS.output_dir / "real_local_cem_results.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    if not result["passed"]:
        raise RuntimeError(f"real local CEM checks failed: {checks}")
    return result


def main() -> int:
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
            with (ARGS.output_dir / "real_local_cem_failure.json").open(
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

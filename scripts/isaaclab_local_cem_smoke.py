"""Run a minimal batched local-group CEM loop on Isaac Lab GPU PhysX."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--early-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
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
    if not args.headless:
        parser.error("the local CEM smoke test requires --headless")
    if not str(args.device).startswith("cuda"):
        parser.error("the local CEM smoke test requires --device cuda:N")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir = output_dir
    args.kit_args = f"{args.kit_args} --/log/file={output_dir / 'kit.log'}".strip()
    return args


ARGS = _parse_args()
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
from isaaclab.utils import configclass  # noqa: E402

from rest3d.optim.cem import CEMOptimizer  # noqa: E402
from rest3d.sim.local_cem import (  # noqa: E402
    LocalCEMEnergyWeights,
    apply_pose_deltas_wxyz,
    evaluate_local_cem_energy,
    quaternion_rotate_wxyz,
)


@configclass
class LocalCEMSmokeSceneCfg(InteractiveSceneCfg):
    """One generic support/child group replicated across candidate environments."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    objects = RigidObjectCollectionCfg(
        rigid_objects={
            "support_root": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SupportRoot",
                spawn=sim_utils.CuboidCfg(
                    size=(0.8, 0.8, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                        solver_position_iteration_count=16,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
            ),
            "movable_child": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MovableChild",
                spawn=sim_utils.CuboidCfg(
                    size=(0.15, 0.15, 0.15),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.3,
                        angular_damping=0.3,
                        max_depenetration_velocity=1.0,
                        solver_position_iteration_count=16,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.175)),
            ),
        }
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
        init_trans_x_std=0.06,
        init_trans_y_std=0.06,
        init_trans_z_std=0.025,
        init_rot_roll_std=0.08,
        init_rot_pitch_std=0.08,
        init_rot_yaw_std=0.08,
    )


def _local_states(collection: RigidObjectCollection, origins: torch.Tensor) -> torch.Tensor:
    states = collection.data.object_state_w.clone()
    states[..., :3] -= origins.unsqueeze(1)
    return states


def _placement_penetration(child_poses: torch.Tensor) -> torch.Tensor:
    half_extent = 0.075
    corners = torch.tensor(
        [
            [x, y, z]
            for x in (-half_extent, half_extent)
            for y in (-half_extent, half_extent)
            for z in (-half_extent, half_extent)
        ],
        device=child_poses.device,
        dtype=child_poses.dtype,
    )
    quaternions = child_poses[:, 0, 3:7].unsqueeze(1).expand(-1, 8, -1)
    rotated = quaternion_rotate_wxyz(
        quaternions,
        corners.unsqueeze(0).expand(child_poses.shape[0], -1, -1),
    )
    child_bottom = child_poses[:, 0, 2] + rotated[..., 2].min(dim=1).values
    support_top = 0.1
    return torch.clamp_min(support_top - child_bottom, 0.0)


def _settled_contact_metrics(contact_view, dt: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
    pair_force = contact_view.get_contact_force_matrix(dt=dt).reshape(
        ARGS.num_envs, contact_view.filter_count, 3
    )
    force_norm = torch.linalg.vector_norm(pair_force[:, 0], dim=-1)
    _, _, _, separations, counts, starts = contact_view.get_contact_data(dt=dt)
    counts_np = counts.detach().cpu().numpy().astype(np.int64).reshape(ARGS.num_envs, -1)
    starts_np = starts.detach().cpu().numpy().astype(np.int64).reshape(ARGS.num_envs, -1)
    penetration = torch.zeros(ARGS.num_envs, device=force_norm.device)
    for env_index in range(ARGS.num_envs):
        count = int(counts_np[env_index, 0])
        if count:
            start = int(starts_np[env_index, 0])
            penetration[env_index] = torch.clamp_min(
                -separations[start : start + count].reshape(-1), 0.0
            ).max()
    saturated = int(counts_np.sum()) >= contact_view.max_contact_data_count
    return penetration, force_norm, saturated


def _run() -> dict:
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    total_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    sim_cfg = sim_utils.SimulationCfg(dt=ARGS.physics_dt, device=ARGS.device)
    sim = SimulationContext(sim_cfg)
    physics_context = sim._physics_context
    scene_cfg = LocalCEMSmokeSceneCfg(
        num_envs=ARGS.num_envs,
        env_spacing=1.5,
        replicate_physics=True,
        filter_collisions=True,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    collection: RigidObjectCollection = scene["objects"]
    dt = sim.get_physics_dt()
    object_names = list(collection.object_names)
    root_index = object_names.index("support_root")
    child_index = object_names.index("movable_child")

    contact_view = collection._physics_sim_view.create_rigid_contact_view(
        "/World/envs/env_*/MovableChild",
        filter_patterns=["/World/envs/env_*/SupportRoot"],
        max_contact_data_count=ARGS.num_envs * 32,
    )
    if contact_view.sensor_count != ARGS.num_envs or contact_view.filter_count != 1:
        raise RuntimeError(
            "batched contact view did not resolve one child/support pair per environment"
        )

    default_local = collection.data.default_object_state.clone()
    reference_child_pose = default_local[0, child_index : child_index + 1, :7]
    weights = LocalCEMEnergyWeights(
        pose_stability=1.0,
        rotation_stability=1.0,
        pose_layout=6.0,
        rotation_layout=1.0,
        velocity=1.0,
        placement_penetration=4.0,
        settled_penetration=4.0,
    )
    cem = CEMOptimizer(_cem_config(), n_objects=1)
    initial_mean = cem.mean.copy()
    initial_std = cem.std.copy()
    iteration_records: list[dict] = []
    best_rewards: list[float] = []
    last_state = None
    last_contact_force = None
    maximum_root_displacement = 0.0
    gpu_memory_samples = [_gpu_memory_used_mib()]

    simulation_started = time.perf_counter()
    for iteration in range(ARGS.cem_iters):
        samples = cem.sample()
        sample_tensor = torch.as_tensor(
            samples,
            dtype=torch.float32,
            device=ARGS.device,
        ).view(ARGS.num_envs, 1, 6)
        child_poses = apply_pose_deltas_wxyz(reference_child_pose, sample_tensor)

        candidate_states = default_local.clone()
        candidate_states[..., :3] += scene.env_origins.unsqueeze(1)
        candidate_states[:, child_index : child_index + 1, :7] = child_poses
        candidate_states[:, child_index, :3] += scene.env_origins
        candidate_states[..., 7:] = 0.0
        collection.write_object_state_to_sim(candidate_states)
        collection.reset()
        sim.forward()
        collection.update(0.0)
        placed = _local_states(collection, scene.env_origins)
        placement_penetration = _placement_penetration(
            placed[:, child_index : child_index + 1, :7]
        )

        early = None
        for step in range(ARGS.settle_steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            if step + 1 == ARGS.early_steps:
                early = _local_states(collection, scene.env_origins)
        if early is None:
            raise RuntimeError("early state was not sampled")
        settled = _local_states(collection, scene.env_origins)
        settled_penetration, contact_force, saturated = _settled_contact_metrics(
            contact_view, dt
        )
        if saturated:
            raise RuntimeError("batched contact data capacity was saturated")

        energy = evaluate_local_cem_energy(
            placed[:, child_index : child_index + 1],
            early[:, child_index : child_index + 1],
            settled[:, child_index : child_index + 1],
            reference_child_pose,
            placement_penetration=placement_penetration,
            settled_penetration=settled_penetration,
            weights=weights,
        )
        rewards = energy["reward"].detach().cpu().numpy()
        cem.update(samples, rewards)
        best_index = int(np.argmax(rewards))
        best_rewards.append(float(cem._best_reward))
        root_displacement = torch.linalg.vector_norm(
            settled[:, root_index, :3] - default_local[:, root_index, :3], dim=-1
        ).max()
        maximum_root_displacement = max(
            maximum_root_displacement, float(root_displacement.item())
        )
        iteration_records.append(
            {
                "iteration": iteration + 1,
                "best_environment": best_index,
                "current_best_reward": float(rewards[best_index]),
                "all_time_best_reward": float(cem._best_reward),
                "reward_mean": float(rewards.mean()),
                "mean_norm": float(np.linalg.norm(cem.mean)),
                "std_mean": float(cem.std.mean()),
                "best_energy_components": {
                    name: float(values[best_index].item())
                    for name, values in energy.items()
                    if name != "reward"
                },
            }
        )
        last_state = settled
        last_contact_force = contact_force
        gpu_memory_samples.append(_gpu_memory_used_mib())

    torch.cuda.synchronize()
    simulation_seconds = time.perf_counter() - simulation_started
    if last_state is None or last_contact_force is None:
        raise RuntimeError("CEM produced no simulation state")
    cuda_probe = last_state.square().sum() + last_contact_force.square().sum()
    cuda_probe_value = float(cuda_probe.item())
    gpu_memory_samples.append(_gpu_memory_used_mib())
    valid_gpu_memory = [value for value in gpu_memory_samples if value is not None]
    checks = {
        "physics_uses_gpu_sim": bool(physics_context.use_gpu_sim),
        "physics_uses_gpu_pipeline": bool(physics_context.use_gpu_pipeline),
        "physics_broadphase_is_gpu": physics_context.get_broadphase_type() == "GPU",
        "state_tensor_is_cuda": last_state.device.type == "cuda",
        "contact_tensor_is_cuda": last_contact_force.device.type == "cuda",
        "state_shape_is_complete": tuple(last_state.shape)
        == (ARGS.num_envs, 2, 13),
        "all_states_are_finite": bool(torch.isfinite(last_state).all()),
        "all_rewards_are_finite": all(
            np.isfinite(record["current_best_reward"]) for record in iteration_records
        ),
        "root_remained_fixed": maximum_root_displacement < 1.0e-5,
        "cem_distribution_updated": not np.allclose(cem.mean, initial_mean),
        "cem_best_reward_is_monotonic": all(
            later >= earlier for earlier, later in zip(best_rewards, best_rewards[1:])
        ),
        "cem_best_action_exists": cem._best_action is not None,
        "cuda_probe_is_finite": bool(np.isfinite(cuda_probe_value)),
        "torch_has_sm120": "sm_120" in torch.cuda.get_arch_list(),
    }
    np.savez_compressed(
        ARGS.output_dir / "local_cem_metrics.npz",
        initial_mean=initial_mean,
        initial_std=initial_std,
        final_mean=cem.mean,
        final_std=cem.std,
        best_action=cem._best_action,
        best_reward=np.asarray(cem._best_reward),
        settled_states=last_state.detach().cpu().numpy(),
        iteration_best_rewards=np.asarray(best_rewards),
    )
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
        "simulation": {
            "num_envs": ARGS.num_envs,
            "object_names": object_names,
            "object_count_per_env": len(object_names),
            "state_tensor_device": str(last_state.device),
            "state_tensor_shape": list(last_state.shape),
            "contact_tensor_device": str(last_contact_force.device),
            "steps_per_iteration": ARGS.settle_steps,
            "cem_iterations": ARGS.cem_iters,
            "total_physics_steps": ARGS.settle_steps * ARGS.cem_iters,
            "simulation_seconds": simulation_seconds,
            "physics_steps_per_second": (
                ARGS.settle_steps * ARGS.cem_iters / simulation_seconds
            ),
            "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "system_gpu_memory_used_mib_peak": max(valid_gpu_memory)
            if valid_gpu_memory
            else None,
            "maximum_root_displacement_m": maximum_root_displacement,
            "cuda_probe_value": cuda_probe_value,
        },
        "cem": {
            "seed": ARGS.seed,
            "population": ARGS.num_envs,
            "elite_count": cem.n_elite,
            "initial_mean": initial_mean.tolist(),
            "initial_std": initial_std.tolist(),
            "final_mean": cem.mean.tolist(),
            "final_std": cem.std.tolist(),
            "best_action": cem._best_action.tolist(),
            "best_reward": float(cem._best_reward),
            "iterations": iteration_records,
        },
        "total_seconds": time.perf_counter() - total_started,
    }
    with (ARGS.output_dir / "local_cem_results.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    if not result["passed"]:
        raise RuntimeError(f"local CEM smoke checks failed: {checks}")
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
            with (ARGS.output_dir / "local_cem_failure.json").open(
                "x", encoding="utf-8"
            ) as file:
                json.dump(failure, file, indent=2)
                file.write("\n")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(1)
    else:
        # The WSL headless Kit cleanup path can segfault after all GPU work and
        # result writes have completed.  This entry point owns its subprocess,
        # so flush the reproducibility evidence and exit without unsafe cleanup.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

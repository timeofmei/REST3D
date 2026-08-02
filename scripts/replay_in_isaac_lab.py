"""Replay a validated REST3D scene with Isaac Lab in headless GPU mode."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

from rest3d.sim.replay_scene import (
    lab_states_to_rest,
    load_replay_scene,
    rest_states_to_lab,
)
from rest3d.sim.local_results import load_scene_states
from rest3d.sim.stability import (
    evaluate_replay_stability,
    quaternion_geodesic_distance_wxyz,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Replay a REST3D scene with Isaac Lab")
    parser.add_argument("--scene-tree", type=Path, required=True)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Read-only scene containing obj_files and, unless overridden, urdf_files",
    )
    parser.add_argument(
        "--urdf-dir",
        type=Path,
        help="Optional derived physics URDF directory; geometry remains in --scene-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New result directory; the command fails if it already exists",
    )
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument(
        "--initial-states",
        type=Path,
        help="Optional full-scene REST3D WXYZ state JSON, such as D local output",
    )
    parser.add_argument(
        "--preserve-initial-velocities",
        action="store_true",
        help="Use velocities from --initial-states instead of starting from rest",
    )
    parser.add_argument(
        "--stability-evaluation-steps",
        type=int,
        default=60,
        help="Compare every object with frame zero at this physics step",
    )
    parser.add_argument("--position-stability-threshold", type=float, default=0.1)
    parser.add_argument("--rotation-stability-threshold", type=float, default=0.1)
    parser.add_argument(
        "--expected-object-count",
        type=int,
        help="Fail if the validated scene does not contain this many objects",
    )
    parser.add_argument(
        "--require-stable",
        action="store_true",
        help="Return failure when any object exceeds a stability threshold",
    )
    parser.add_argument(
        "--contact-data-capacity-per-object",
        type=int,
        default=1024,
        help="Maximum detailed PhysX contact records retained per object and frame",
    )
    parser.add_argument(
        "--state-only-benchmark",
        action="store_true",
        help="Collect root states only so backend throughput instrumentation matches Isaac Gym",
    )
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument(
        "--collision-approximation",
        choices=("convex_hull", "convex_decomposition"),
        default="convex_hull",
    )
    parser.add_argument("--linear-damping", type=float, default=0.3)
    parser.add_argument("--angular-damping", type=float, default=0.3)
    parser.add_argument("--max-depenetration-velocity", type=float, default=1.0)
    parser.add_argument("--num-position-iterations", type=int, default=16)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.settle_steps < 1:
        parser.error("--settle-steps must be positive")
    if not 1 <= args.stability_evaluation_steps <= args.settle_steps:
        parser.error(
            "--stability-evaluation-steps must be between 1 and --settle-steps"
        )
    if args.position_stability_threshold <= 0.0:
        parser.error("--position-stability-threshold must be positive")
    if args.rotation_stability_threshold <= 0.0:
        parser.error("--rotation-stability-threshold must be positive")
    if args.expected_object_count is not None and args.expected_object_count < 1:
        parser.error("--expected-object-count must be positive")
    if args.contact_data_capacity_per_object < 1:
        parser.error("--contact-data-capacity-per-object must be positive")
    if args.physics_dt <= 0.0:
        parser.error("--physics-dt must be positive")
    if not args.headless:
        parser.error("Phase B currently requires --headless because WSL rendering is not validated")
    if not str(args.device).startswith("cuda"):
        parser.error("Isaac Lab replay requires --device cuda:N")

    scene_dir = args.scene_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == scene_dir or scene_dir in output_dir.parents:
        parser.error("--output-dir must be outside the read-only --scene-dir")
    output_dir.mkdir(parents=True, exist_ok=False)
    args.scene_dir = scene_dir
    if args.urdf_dir is not None:
        args.urdf_dir = args.urdf_dir.expanduser().resolve(strict=True)
        if not args.urdf_dir.is_dir():
            parser.error(f"--urdf-dir is not a directory: {args.urdf_dir}")
    args.output_dir = output_dir
    args.scene_tree = args.scene_tree.expanduser().resolve(strict=True)
    if args.initial_states is not None:
        args.initial_states = args.initial_states.expanduser().resolve(strict=True)

    kit_log_arg = f"--/log/file={output_dir / 'kit.log'}"
    args.kit_args = f"{args.kit_args} {kit_log_arg}".strip()
    return args


ARGS = _parse_args()
SCENE = load_replay_scene(
    ARGS.scene_tree,
    ARGS.scene_dir,
    urdf_dir_override=ARGS.urdf_dir,
)
if ARGS.initial_states is None:
    INITIAL_STATES_REST = np.zeros((len(SCENE.objects), 13), dtype=np.float64)
    INITIAL_STATES_REST[:, 3] = 1.0
else:
    _initial_states_by_name = load_scene_states(
        ARGS.initial_states,
        SCENE.names,
        zero_velocities=not ARGS.preserve_initial_velocities,
    )
    INITIAL_STATES_REST = np.asarray(
        [_initial_states_by_name[name] for name in SCENE.names], dtype=np.float64
    )
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvrtc_version() -> tuple[int, int]:
    library = ctypes.CDLL("libnvrtc.so.12")
    major = ctypes.c_int()
    minor = ctypes.c_int()
    result = library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    if result != 0:
        raise RuntimeError(f"nvrtcVersion failed with code {result}")
    return major.value, minor.value


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


def _nvidia_environment() -> dict[str, str | int | None]:
    result: dict[str, str | int | None] = {
        "gpu": None,
        "driver": None,
        "memory_total_mib": None,
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        fields = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        result.update(
            gpu=fields[0],
            driver=fields[1],
            memory_total_mib=int(fields[2]),
        )
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        pass
    return result


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("isaaclab_replay")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(ARGS.output_dir / "replay.log", mode="x")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _spawn_objects(logger: logging.Logger) -> dict[str, RigidObject]:
    objects: dict[str, RigidObject] = {}
    lab_poses = rest_states_to_lab(INITIAL_STATES_REST)[:, :7]

    for index, (spec, pose) in enumerate(zip(SCENE.objects, lab_poses, strict=True)):
        converted_dir = ARGS.output_dir / "converted_assets" / f"object_{index:04d}"
        spawn_cfg = sim_utils.UrdfFileCfg(
            asset_path=str(spec.urdf_path),
            usd_dir=str(converted_dir),
            usd_file_name="asset.usd",
            force_usd_conversion=False,
            make_instanceable=False,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
            collider_type=ARGS.collision_approximation,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=spec.fixed,
                disable_gravity=spec.fixed,
                linear_damping=ARGS.linear_damping,
                angular_damping=ARGS.angular_damping,
                max_depenetration_velocity=ARGS.max_depenetration_velocity,
                solver_position_iteration_count=ARGS.num_position_iterations,
            ),
        )
        cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Object_{index:04d}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(value) for value in pose[:3]),
                rot=tuple(float(value) for value in pose[3:7]),
            ),
        )
        objects[spec.name] = RigidObject(cfg=cfg)
        logger.info(
            "loaded object %d/%d name=%s fixed=%s",
            index + 1,
            len(SCENE.objects),
            spec.name,
            spec.fixed,
        )
    return objects


def _collect_states(objects: dict[str, RigidObject]) -> torch.Tensor:
    return torch.cat([objects[name].data.root_state_w.clone() for name in SCENE.names], dim=0)


def _initial_target_states_lab() -> np.ndarray:
    return rest_states_to_lab(INITIAL_STATES_REST)


def _write_initial_states(objects: dict[str, RigidObject]) -> None:
    states = torch.tensor(
        _initial_target_states_lab(),
        dtype=torch.float32,
        device=ARGS.device,
    )
    for index, name in enumerate(SCENE.names):
        objects[name].write_root_pose_to_sim(states[index : index + 1, :7])
        objects[name].write_root_velocity_to_sim(states[index : index + 1, 7:])
        objects[name].reset()


def _collision_prim_paths_below(prim_path: str) -> tuple[str, ...]:
    """Return collision prim paths below a known scene namespace."""

    root = get_current_stage().GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"collision namespace does not exist: {prim_path}")
    pending = [root]
    collision_paths: list[str] = []
    while pending:
        prim = pending.pop()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(prim.GetPath().pathString)
        pending.extend(prim.GetChildren())
    return tuple(sorted(collision_paths))


def _inspect_collision_geometry(objects: dict[str, RigidObject]) -> dict[str, dict]:
    geometry: dict[str, dict] = {}
    for name in SCENE.names:
        obj = objects[name]
        masses = obj.root_physx_view.get_masses().detach().cpu().numpy().reshape(-1)
        inertias = (
            obj.root_physx_view.get_inertias().detach().cpu().numpy().reshape(-1, 9)
        )
        centers_of_mass = (
            obj.root_physx_view.get_coms().detach().cpu().numpy().reshape(-1, 7)
        )
        geometry[name] = {
            "body_count": int(obj.num_bodies),
            "shape_count": int(obj.root_physx_view.max_shapes),
            "body_paths": [str(path) for path in obj.root_physx_view.prim_paths],
            "mass_kg": float(masses[0]),
            "inertia_matrix_kg_m2": inertias[0].reshape(3, 3).tolist(),
            "center_of_mass_pose_xyzw": centers_of_mass[0].tolist(),
        }
    return geometry


def _create_contact_views(
    objects: dict[str, RigidObject], collision_geometry: dict[str, dict]
) -> tuple[dict[str, object], dict[str, list[str]]]:
    ground_collision_paths = list(_collision_prim_paths_below("/World/Ground"))
    if not ground_collision_paths:
        raise RuntimeError("ground plane has no collision geometry")

    body_paths = {
        name: collision_geometry[name]["body_paths"][0] for name in SCENE.names
    }
    contact_views: dict[str, object] = {}
    filters_by_name: dict[str, list[str]] = {}
    physics_sim_view = next(iter(objects.values()))._physics_sim_view
    for name in SCENE.names:
        filter_paths = [
            body_paths[other_name] for other_name in SCENE.names if other_name != name
        ] + ground_collision_paths
        view = physics_sim_view.create_rigid_contact_view(
            body_paths[name],
            filter_patterns=filter_paths,
            max_contact_data_count=ARGS.contact_data_capacity_per_object,
        )
        if view.sensor_count != 1:
            raise RuntimeError(
                f"contact view for {name} resolved {view.sensor_count} sensor bodies"
            )
        if view.filter_count != len(filter_paths):
            raise RuntimeError(
                f"contact view for {name} resolved {view.filter_count} filters; "
                f"expected {len(filter_paths)}"
            )
        contact_views[name] = view
        filters_by_name[name] = filter_paths
    return contact_views, filters_by_name


def _collect_contact_snapshot(
    contact_views: dict[str, object], dt: float
) -> tuple[
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    force_norms: list[torch.Tensor] = []
    contact_counts = np.zeros(len(SCENE.objects), dtype=np.int64)
    minimum_separation = np.full(len(SCENE.objects), np.nan, dtype=np.float64)
    capacity_saturated = np.zeros(len(SCENE.objects), dtype=bool)
    filter_count = next(iter(contact_views.values())).filter_count
    pair_force_norms: list[torch.Tensor] = []
    pair_contact_counts = np.zeros(
        (len(SCENE.objects), filter_count), dtype=np.int64
    )
    pair_minimum_separation = np.full(
        (len(SCENE.objects), filter_count), np.nan, dtype=np.float64
    )
    for index, name in enumerate(SCENE.names):
        view = contact_views[name]
        net_force = view.get_net_contact_forces(dt=dt)
        force_norms.append(torch.linalg.vector_norm(net_force, dim=-1).max())
        pair_force = view.get_contact_force_matrix(dt=dt).reshape(
            view.sensor_count, view.filter_count, 3
        )
        pair_force_norms.append(torch.linalg.vector_norm(pair_force[0], dim=-1))
        _, _, _, separations, counts, start_indices = view.get_contact_data(dt=dt)
        separations_np = separations.detach().cpu().numpy().reshape(-1)
        counts_np = counts.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        starts_np = (
            start_indices.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        )
        contact_counts[index] = int(counts_np.sum())
        pair_contact_counts[index] = counts_np
        capacity_saturated[index] = (
            contact_counts[index] >= view.max_contact_data_count
        )
        active_separations: list[np.ndarray] = []
        for filter_index, (start, count) in enumerate(
            zip(starts_np, counts_np, strict=True)
        ):
            if count > 0:
                values = separations_np[start : start + count]
                active_separations.append(values)
                pair_minimum_separation[index, filter_index] = float(np.min(values))
        if active_separations:
            minimum_separation[index] = float(
                min(np.min(values) for values in active_separations)
            )
    return (
        torch.stack(force_norms),
        contact_counts,
        minimum_separation,
        capacity_saturated,
        torch.stack(pair_force_norms),
        pair_contact_counts,
        pair_minimum_separation,
    )


def _run() -> dict:
    logger = _configure_logger()
    logger.info("scene_dir=%s", SCENE.scene_dir)
    logger.info("output_dir=%s", ARGS.output_dir)
    logger.info(
        "objects=%d fixed=%d movable=%d names=%s",
        len(SCENE.objects),
        len(SCENE.fixed_names),
        len(SCENE.movable_names),
        list(SCENE.names),
    )
    torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()

    sim_cfg = sim_utils.SimulationCfg(dt=ARGS.physics_dt, device=ARGS.device)
    sim = SimulationContext(sim_cfg)
    physics_context = sim._physics_context
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    asset_started = time.perf_counter()
    objects = _spawn_objects(logger)
    asset_load_seconds = time.perf_counter() - asset_started
    if tuple(objects) != SCENE.names:
        raise RuntimeError("loaded object set/order differs from the validated scene spec")
    if (
        ARGS.expected_object_count is not None
        and len(objects) != ARGS.expected_object_count
    ):
        raise RuntimeError(
            f"validated scene contains {len(objects)} objects; "
            f"expected {ARGS.expected_object_count}"
        )

    sim.reset()
    dt = sim.get_physics_dt()
    _write_initial_states(objects)
    sim.forward()
    for obj in objects.values():
        obj.update(0.0)

    collision_geometry = _inspect_collision_geometry(objects)
    if ARGS.state_only_benchmark:
        contact_views = {}
        contact_filters = {name: [] for name in SCENE.names}
    else:
        contact_views, contact_filters = _create_contact_views(
            objects, collision_geometry
        )

    state_frames_lab = [_collect_states(objects)]
    if ARGS.state_only_benchmark:
        contact_force_frames = []
        contact_count_frames = []
        contact_separation_frames = []
        contact_capacity_frames = []
        pair_contact_force_frames = []
        pair_contact_count_frames = []
        pair_contact_separation_frames = []
    else:
        initial_contact = _collect_contact_snapshot(contact_views, dt)
        contact_force_frames = [initial_contact[0]]
        contact_count_frames = [initial_contact[1]]
        contact_separation_frames = [initial_contact[2]]
        contact_capacity_frames = [initial_contact[3]]
        pair_contact_force_frames = [initial_contact[4]]
        pair_contact_count_frames = [initial_contact[5]]
        pair_contact_separation_frames = [initial_contact[6]]
    gpu_memory_samples = [_gpu_memory_used_mib()]
    simulation_started = time.perf_counter()
    for step in range(ARGS.settle_steps):
        for obj in objects.values():
            obj.write_data_to_sim()
        sim.step(render=False)
        for obj in objects.values():
            obj.update(dt)
        state_frames_lab.append(_collect_states(objects))
        if not ARGS.state_only_benchmark:
            contact_snapshot = _collect_contact_snapshot(contact_views, dt)
            contact_force_frames.append(contact_snapshot[0])
            contact_count_frames.append(contact_snapshot[1])
            contact_separation_frames.append(contact_snapshot[2])
            contact_capacity_frames.append(contact_snapshot[3])
            pair_contact_force_frames.append(contact_snapshot[4])
            pair_contact_count_frames.append(contact_snapshot[5])
            pair_contact_separation_frames.append(contact_snapshot[6])
        if (step + 1) % 10 == 0:
            gpu_memory_samples.append(_gpu_memory_used_mib())
    torch.cuda.synchronize()
    simulation_seconds = time.perf_counter() - simulation_started
    gpu_memory_samples.append(_gpu_memory_used_mib())
    nvrtc_version = _nvrtc_version()

    states_lab_tensor = torch.stack(state_frames_lab, dim=0)
    states_lab = states_lab_tensor.detach().cpu().numpy()
    states_rest = lab_states_to_rest(states_lab)
    if ARGS.state_only_benchmark:
        contact_force_tensor = None
        pair_contact_force_tensor = None
        contact_force_norms = np.zeros(states_lab.shape[:2], dtype=np.float32)
        contact_counts = np.zeros(states_lab.shape[:2], dtype=np.int64)
        contact_minimum_separation = np.full(
            states_lab.shape[:2], np.nan, dtype=np.float64
        )
        contact_capacity_saturated = np.zeros(states_lab.shape[:2], dtype=bool)
        pair_shape = (states_lab.shape[0], states_lab.shape[1], 0)
        pair_contact_force_norms = np.zeros(pair_shape, dtype=np.float32)
        pair_contact_counts = np.zeros(pair_shape, dtype=np.int64)
        pair_contact_minimum_separation = np.full(
            pair_shape, np.nan, dtype=np.float64
        )
    else:
        contact_force_tensor = torch.stack(contact_force_frames, dim=0)
        pair_contact_force_tensor = torch.stack(pair_contact_force_frames, dim=0)
        contact_force_norms = contact_force_tensor.detach().cpu().numpy()
        contact_counts = np.stack(contact_count_frames, axis=0)
        contact_minimum_separation = np.stack(contact_separation_frames, axis=0)
        contact_capacity_saturated = np.stack(contact_capacity_frames, axis=0)
        pair_contact_force_norms = pair_contact_force_tensor.detach().cpu().numpy()
        pair_contact_counts = np.stack(pair_contact_count_frames, axis=0)
        pair_contact_minimum_separation = np.stack(
            pair_contact_separation_frames, axis=0
        )
    stability = evaluate_replay_stability(
        states_rest,
        evaluation_step=ARGS.stability_evaluation_steps,
        position_threshold_m=ARGS.position_stability_threshold,
        rotation_threshold_rad=ARGS.rotation_stability_threshold,
    )
    target_rest = INITIAL_STATES_REST
    initial_position_error = np.linalg.norm(states_rest[0, :, :3] - target_rest[:, :3], axis=1)
    initial_quat_error = quaternion_geodesic_distance_wxyz(
        states_rest[0, :, 3:7], target_rest[:, 3:7]
    )

    minimum_contact_separation = np.array(
        [
            np.nanmin(contact_minimum_separation[:, index])
            if np.isfinite(contact_minimum_separation[:, index]).any()
            else np.nan
            for index in range(len(SCENE.objects))
        ]
    )
    maximum_penetration_depth = np.where(
        np.isfinite(minimum_contact_separation),
        np.maximum(0.0, -minimum_contact_separation),
        0.0,
    )
    valid_gpu_memory = [value for value in gpu_memory_samples if value is not None]
    nvidia_environment = _nvidia_environment()
    checks = {
        "all_scene_objects_loaded": len(objects) == len(SCENE.objects),
        "expected_object_count_matches": ARGS.expected_object_count is None
        or len(objects) == ARGS.expected_object_count,
        "one_rigid_body_per_object": all(
            values["body_count"] == 1 for values in collision_geometry.values()
        ),
        "all_objects_have_collision_shapes": all(
            values["shape_count"] > 0 for values in collision_geometry.values()
        ),
        "physics_uses_gpu_sim": physics_context.use_gpu_sim,
        "physics_uses_gpu_pipeline": physics_context.use_gpu_pipeline,
        "physics_broadphase_is_gpu": physics_context.get_broadphase_type() == "GPU",
        "state_tensor_is_cuda": states_lab_tensor.device.type == "cuda",
        "state_shape_is_complete": states_lab.shape
        == (ARGS.settle_steps + 1, len(SCENE.objects), 13),
        "initial_position_round_trip": float(initial_position_error.max()) < 1.0e-5,
        "initial_quaternion_round_trip": float(initial_quat_error.max()) < 1.0e-5,
        "states_are_finite": bool(np.isfinite(states_lab).all()),
        "torch_has_sm120": "sm_120" in torch.cuda.get_arch_list(),
        "nvrtc_supports_sm120": nvrtc_version >= (12, 8),
        "all_objects_evaluated_for_stability": stability.stable.shape
        == (len(SCENE.objects),),
    }
    if ARGS.state_only_benchmark:
        checks["state_only_benchmark_collection"] = not contact_views
    else:
        checks.update(
            contact_tensor_is_cuda=contact_force_tensor.device.type == "cuda",
            all_objects_have_contact_views=tuple(contact_views) == SCENE.names,
            contact_data_capacity_not_saturated=not bool(
                contact_capacity_saturated.any()
            ),
        )
    if ARGS.require_stable:
        checks["scene_stable_at_evaluation_step"] = stability.scene_stable

    np.save(ARGS.output_dir / "replay_states_lab.npy", states_lab)
    np.save(ARGS.output_dir / "replay_states_rest.npy", states_rest)
    np.savez_compressed(
        ARGS.output_dir / "stability_metrics.npz",
        object_names=np.asarray(SCENE.names),
        fixed=np.asarray([spec.fixed for spec in SCENE.objects]),
        evaluation_step=np.asarray(stability.evaluation_step),
        position_threshold_m=np.asarray(stability.position_threshold_m),
        rotation_threshold_rad=np.asarray(stability.rotation_threshold_rad),
        data_collection=np.asarray(
            "root_states_each_step"
            if ARGS.state_only_benchmark
            else "root_states_and_detailed_contacts_each_step"
        ),
        stable=stability.stable,
        displacement_at_evaluation_m=stability.displacement_at_evaluation_m,
        rotation_at_evaluation_rad=stability.rotation_at_evaluation_rad,
        final_displacement_m=stability.final_displacement_m,
        final_rotation_rad=stability.final_rotation_rad,
        maximum_excursion_m=stability.maximum_excursion_m,
        maximum_rotation_excursion_rad=stability.maximum_rotation_excursion_rad,
        early_max_linear_speed_m_s=stability.early_max_linear_speed_m_s,
        early_max_angular_speed_rad_s=stability.early_max_angular_speed_rad_s,
        terminal_max_linear_speed_m_s=stability.terminal_max_linear_speed_m_s,
        terminal_max_angular_speed_rad_s=stability.terminal_max_angular_speed_rad_s,
        terminal_mean_linear_speed_m_s=stability.terminal_mean_linear_speed_m_s,
        terminal_mean_angular_speed_rad_s=stability.terminal_mean_angular_speed_rad_s,
        contact_force_norm_n=contact_force_norms,
        contact_count=contact_counts,
        contact_minimum_separation_m=contact_minimum_separation,
        contact_capacity_saturated=contact_capacity_saturated,
        contact_filter_paths=np.asarray(
            [contact_filters[name] for name in SCENE.names]
        ),
        pair_contact_force_norm_n=pair_contact_force_norms,
        pair_contact_count=pair_contact_counts,
        pair_contact_minimum_separation_m=pair_contact_minimum_separation,
    )

    object_results: dict[str, dict] = {}
    for index, name in enumerate(SCENE.names):
        unstable_reasons: list[str] = []
        if (
            stability.displacement_at_evaluation_m[index]
            > stability.position_threshold_m
        ):
            unstable_reasons.append("translation")
        if (
            stability.rotation_at_evaluation_rad[index]
            > stability.rotation_threshold_rad
        ):
            unstable_reasons.append("rotation")
        min_separation = minimum_contact_separation[index]
        pair_results: list[dict] = []
        for filter_index, filter_path in enumerate(contact_filters[name]):
            pair_separation_series = pair_contact_minimum_separation[
                :, index, filter_index
            ]
            pair_min_separation = (
                float(np.nanmin(pair_separation_series))
                if np.isfinite(pair_separation_series).any()
                else None
            )
            pair_results.append(
                {
                    "filter_path": filter_path,
                    "evaluation_count": int(
                        pair_contact_counts[
                            stability.evaluation_step, index, filter_index
                        ]
                    ),
                    "evaluation_force_n": float(
                        pair_contact_force_norms[
                            stability.evaluation_step, index, filter_index
                        ]
                    ),
                    "maximum_force_n": float(
                        pair_contact_force_norms[:, index, filter_index].max()
                    ),
                    "frames_with_contact": int(
                        np.count_nonzero(
                            pair_contact_counts[:, index, filter_index] > 0
                        )
                    ),
                    "minimum_separation_m": pair_min_separation,
                    "maximum_penetration_depth_m": max(
                        0.0, -pair_min_separation
                    )
                    if pair_min_separation is not None
                    else 0.0,
                }
            )
        object_results[name] = {
            "fixed": SCENE.objects[index].fixed,
            "collision": {
                **collision_geometry[name],
                "contact_filter_paths": contact_filters[name],
            },
            "initial_state_rest": states_rest[0, index].tolist(),
            "evaluation_state_rest": states_rest[
                stability.evaluation_step, index
            ].tolist(),
            "final_state_rest": states_rest[-1, index].tolist(),
            "stable": bool(stability.stable[index]),
            "unstable_reasons": unstable_reasons,
            "displacement_at_evaluation_m": float(
                stability.displacement_at_evaluation_m[index]
            ),
            "rotation_at_evaluation_rad": float(
                stability.rotation_at_evaluation_rad[index]
            ),
            "final_displacement_m": float(stability.final_displacement_m[index]),
            "final_rotation_rad": float(stability.final_rotation_rad[index]),
            "maximum_excursion_m": float(stability.maximum_excursion_m[index]),
            "maximum_rotation_excursion_rad": float(
                stability.maximum_rotation_excursion_rad[index]
            ),
            "early_max_linear_speed_m_s": float(
                stability.early_max_linear_speed_m_s[index]
            ),
            "early_max_angular_speed_rad_s": float(
                stability.early_max_angular_speed_rad_s[index]
            ),
            "terminal_max_linear_speed_m_s": float(
                stability.terminal_max_linear_speed_m_s[index]
            ),
            "terminal_max_angular_speed_rad_s": float(
                stability.terminal_max_angular_speed_rad_s[index]
            ),
            "terminal_mean_linear_speed_m_s": float(
                stability.terminal_mean_linear_speed_m_s[index]
            ),
            "terminal_mean_angular_speed_rad_s": float(
                stability.terminal_mean_angular_speed_rad_s[index]
            ),
            "contact": {
                "collected": not ARGS.state_only_benchmark,
                "evaluation_count": int(
                    contact_counts[stability.evaluation_step, index]
                ),
                "evaluation_net_force_n": float(
                    contact_force_norms[stability.evaluation_step, index]
                ),
                "maximum_net_force_n": float(contact_force_norms[:, index].max()),
                "frames_with_contact": int(
                    np.count_nonzero(contact_counts[:, index] > 0)
                ),
                "minimum_separation_m": float(min_separation)
                if np.isfinite(min_separation)
                else None,
                "maximum_penetration_depth_m": float(
                    maximum_penetration_depth[index]
                ),
                "capacity_saturated": bool(
                    contact_capacity_saturated[:, index].any()
                ),
                "pairs": pair_results,
            },
        }
        logger.info(
            "stability name=%s stable=%s displacement=%.6f m rotation=%.6f rad "
            "terminal_linear=%.6f m/s terminal_angular=%.6f rad/s penetration=%.6f m",
            name,
            stability.stable[index],
            stability.displacement_at_evaluation_m[index],
            stability.rotation_at_evaluation_rad[index],
            stability.terminal_mean_linear_speed_m_s[index],
            stability.terminal_mean_angular_speed_rad_s[index],
            maximum_penetration_depth[index],
        )

    result = {
        "passed": all(checks.values()),
        "runtime_passed": all(
            value
            for key, value in checks.items()
            if key != "scene_stable_at_evaluation_step"
        ),
        "backend": "isaac-lab",
        "scene_stable": stability.scene_stable,
        "checks": checks,
        "versions": {
            "python": platform.python_version(),
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab_package": _package_version("isaaclab"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvrtc": ".".join(str(value) for value in nvrtc_version),
            "driver": nvidia_environment["driver"],
        },
        "environment": {
            "platform": platform.platform(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "gpu": nvidia_environment["gpu"] or torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_memory_total_mib": nvidia_environment["memory_total_mib"],
        },
        "scene": {
            "scene_tree": str(SCENE.scene_tree_path),
            "scene_dir": str(SCENE.scene_dir),
            "urdf_dir": str(ARGS.urdf_dir) if ARGS.urdf_dir is not None else None,
            "object_names": list(SCENE.names),
            "fixed_names": list(SCENE.fixed_names),
            "movable_names": list(SCENE.movable_names),
            "object_count": len(SCENE.objects),
            "bounds_min_rest": list(SCENE.bounds_min_rest),
            "bounds_max_rest": list(SCENE.bounds_max_rest),
            "initial_states": str(ARGS.initial_states)
            if ARGS.initial_states is not None
            else None,
            "initial_velocities_preserved": ARGS.preserve_initial_velocities,
        },
        "simulation": {
            "device": str(sim.device),
            "physics_uses_gpu_sim": physics_context.use_gpu_sim,
            "physics_uses_gpu_pipeline": physics_context.use_gpu_pipeline,
            "physics_broadphase_type": physics_context.get_broadphase_type(),
            "state_tensor_device": str(states_lab_tensor.device),
            "state_tensor_shape": list(states_lab.shape),
            "contact_tensor_device": str(contact_force_tensor.device)
            if contact_force_tensor is not None
            else None,
            "contact_tensor_shape": list(contact_force_tensor.shape)
            if contact_force_tensor is not None
            else None,
            "pair_contact_tensor_device": str(pair_contact_force_tensor.device)
            if pair_contact_force_tensor is not None
            else None,
            "pair_contact_tensor_shape": list(pair_contact_force_tensor.shape)
            if pair_contact_force_tensor is not None
            else None,
            "steps": ARGS.settle_steps,
            "dt_seconds": dt,
            "startup_seconds": asset_started - total_started,
            "asset_load_seconds": asset_load_seconds,
            "simulation_seconds": simulation_seconds,
            "steps_per_second": ARGS.settle_steps / simulation_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "system_gpu_memory_used_mib_peak": max(valid_gpu_memory) if valid_gpu_memory else None,
            "collision_approximation": ARGS.collision_approximation,
            "data_collection": "root_states_each_step"
            if ARGS.state_only_benchmark
            else "root_states_and_detailed_contacts_each_step",
        },
        "stability": {
            "evaluation_step": stability.evaluation_step,
            "evaluation_time_seconds": stability.evaluation_step * dt,
            "position_threshold_m": stability.position_threshold_m,
            "rotation_threshold_rad": stability.rotation_threshold_rad,
            "require_stable": ARGS.require_stable,
            "scene_stable": stability.scene_stable,
            "stable_object_count": int(np.count_nonzero(stability.stable)),
            "unstable_object_names": [
                name
                for index, name in enumerate(SCENE.names)
                if not stability.stable[index]
            ],
            "metrics_file": str(ARGS.output_dir / "stability_metrics.npz"),
        },
        "objects": object_results,
    }
    with (ARGS.output_dir / "replay_results.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    logger.info("checks=%s", checks)
    logger.info("result=%s", ARGS.output_dir / "replay_results.json")
    if not result["passed"]:
        raise RuntimeError(f"Isaac Lab replay checks failed: {checks}")
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
            with (ARGS.output_dir / "replay_failure.json").open("x", encoding="utf-8") as file:
                json.dump(failure, file, indent=2)
                file.write("\n")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        # Immediate process exit preserves a non-zero code; Kit's WSL cleanup
        # path can otherwise terminate first and mask the original exception.
        os._exit(1)
    else:
        SIMULATION_APP.close(wait_for_replicator=False, skip_cleanup=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

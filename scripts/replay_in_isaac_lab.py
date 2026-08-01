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
    rest_poses_to_lab,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Replay a REST3D scene with Isaac Lab")
    parser.add_argument("--scene-tree", type=Path, required=True)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Read-only Stage 3 scene containing obj_files and urdf_files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New result directory; the command fails if it already exists",
    )
    parser.add_argument("--settle-steps", type=int, default=120)
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
    args.output_dir = output_dir
    args.scene_tree = args.scene_tree.expanduser().resolve(strict=True)

    kit_log_arg = f"--/log/file={output_dir / 'kit.log'}"
    args.kit_args = f"{args.kit_args} {kit_log_arg}".strip()
    return args


ARGS = _parse_args()
SCENE = load_replay_scene(ARGS.scene_tree, ARGS.scene_dir)
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


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
    rest_identity_poses = np.zeros((len(SCENE.objects), 7), dtype=np.float64)
    rest_identity_poses[:, 3] = 1.0
    lab_poses = rest_poses_to_lab(rest_identity_poses)

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
    rest_states = np.zeros((len(SCENE.objects), 13), dtype=np.float64)
    rest_states[:, 3] = 1.0
    lab_poses = rest_poses_to_lab(rest_states[:, :7])
    states = np.zeros_like(rest_states)
    states[:, :7] = lab_poses
    return states


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

    sim.reset()
    dt = sim.get_physics_dt()
    _write_initial_states(objects)
    sim.forward()
    for obj in objects.values():
        obj.update(0.0)

    state_frames_lab = [_collect_states(objects)]
    gpu_memory_samples = [_gpu_memory_used_mib()]
    simulation_started = time.perf_counter()
    for step in range(ARGS.settle_steps):
        for obj in objects.values():
            obj.write_data_to_sim()
        sim.step(render=False)
        for obj in objects.values():
            obj.update(dt)
        state_frames_lab.append(_collect_states(objects))
        if (step + 1) % 10 == 0:
            gpu_memory_samples.append(_gpu_memory_used_mib())
    torch.cuda.synchronize()
    simulation_seconds = time.perf_counter() - simulation_started
    gpu_memory_samples.append(_gpu_memory_used_mib())
    nvrtc_version = _nvrtc_version()

    states_lab_tensor = torch.stack(state_frames_lab, dim=0)
    states_lab = states_lab_tensor.detach().cpu().numpy()
    states_rest = lab_states_to_rest(states_lab)
    target_rest = np.zeros((len(SCENE.objects), 13), dtype=np.float64)
    target_rest[:, 3] = 1.0
    initial_position_error = np.linalg.norm(states_rest[0, :, :3] - target_rest[:, :3], axis=1)
    initial_quat_alignment = np.abs(
        np.sum(states_rest[0, :, 3:7] * target_rest[:, 3:7], axis=1)
    )
    initial_quat_error = 2.0 * np.arccos(np.clip(initial_quat_alignment, 0.0, 1.0))

    displacement = np.linalg.norm(states_rest[-1, :, :3] - states_rest[0, :, :3], axis=1)
    final_speed = np.linalg.norm(states_rest[-1, :, 7:10], axis=1)
    valid_gpu_memory = [value for value in gpu_memory_samples if value is not None]
    checks = {
        "all_scene_objects_loaded": len(objects) == len(SCENE.objects),
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
    }

    np.save(ARGS.output_dir / "replay_states_lab.npy", states_lab)
    np.save(ARGS.output_dir / "replay_states_rest.npy", states_rest)
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "versions": {
            "python": platform.python_version(),
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab_package": _package_version("isaaclab"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvrtc": ".".join(str(value) for value in nvrtc_version),
        },
        "environment": {
            "platform": platform.platform(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "scene": {
            "scene_tree": str(SCENE.scene_tree_path),
            "scene_dir": str(SCENE.scene_dir),
            "object_names": list(SCENE.names),
            "fixed_names": list(SCENE.fixed_names),
            "movable_names": list(SCENE.movable_names),
            "object_count": len(SCENE.objects),
            "bounds_min_rest": list(SCENE.bounds_min_rest),
            "bounds_max_rest": list(SCENE.bounds_max_rest),
        },
        "simulation": {
            "device": str(sim.device),
            "physics_uses_gpu_sim": physics_context.use_gpu_sim,
            "physics_uses_gpu_pipeline": physics_context.use_gpu_pipeline,
            "physics_broadphase_type": physics_context.get_broadphase_type(),
            "state_tensor_device": str(states_lab_tensor.device),
            "state_tensor_shape": list(states_lab.shape),
            "steps": ARGS.settle_steps,
            "dt_seconds": dt,
            "asset_load_seconds": asset_load_seconds,
            "simulation_seconds": simulation_seconds,
            "steps_per_second": ARGS.settle_steps / simulation_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "system_gpu_memory_used_mib_peak": max(valid_gpu_memory) if valid_gpu_memory else None,
            "collision_approximation": ARGS.collision_approximation,
        },
        "objects": {
            name: {
                "fixed": SCENE.objects[index].fixed,
                "initial_state_rest": states_rest[0, index].tolist(),
                "final_state_rest": states_rest[-1, index].tolist(),
                "displacement_m": float(displacement[index]),
                "final_linear_speed_m_s": float(final_speed[index]),
            }
            for index, name in enumerate(SCENE.names)
        },
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

"""Run a finite Isaac Lab GPU PhysX and CUDA tensor smoke test."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> tuple[argparse.Namespace, Path]:
    parser = argparse.ArgumentParser(
        description="Verify Isaac Lab GPU PhysX and the CUDA tensor pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for logs and results; the command fails if it already exists.",
    )
    parser.add_argument("--steps", type=int, default=120)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 60:
        parser.error("--steps must be at least 60")
    if not str(args.device).startswith("cuda"):
        parser.error("this smoke test requires --device cuda:N")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    return args, output_dir


ARGS, OUTPUT_DIR = _parse_args()
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


def _gpu_memory_used_mib() -> int | None:
    """Return system-wide GPU 0 memory use for a coarse PhysX-inclusive peak."""
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
    logger = logging.getLogger("isaaclab_smoke")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(OUTPUT_DIR / "smoke.log", mode="x")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _spawn_scene() -> RigidObject:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    cube_cfg = RigidObjectCfg(
        prim_path="/World/SmokeCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    return RigidObject(cfg=cube_cfg)


def _run() -> dict:
    logger = _configure_logger()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=ARGS.device)
    sim = SimulationContext(sim_cfg)
    physics_context = sim._physics_context
    cube = _spawn_scene()
    sim.reset()

    dt = sim.get_physics_dt()
    cube.update(dt)
    initial_state = cube.data.root_state_w.clone()
    gpu_memory_samples_mib = [_gpu_memory_used_mib()]
    logger.info("initial root state: %s", initial_state)

    for step in range(ARGS.steps):
        cube.write_data_to_sim()
        sim.step(render=False)
        cube.update(dt)
        if (step + 1) % 10 == 0:
            gpu_memory_samples_mib.append(_gpu_memory_used_mib())

    final_state = cube.data.root_state_w.clone()
    displacement = torch.linalg.vector_norm(final_state[:, :3] - initial_state[:, :3], dim=1)
    state_gram = final_state[:, :3] @ final_state[:, :3].T
    state_cuda_result = displacement.sum() + state_gram.sum()
    torch.cuda.synchronize()
    gpu_memory_samples_mib.append(_gpu_memory_used_mib())
    valid_gpu_memory_samples = [
        sample for sample in gpu_memory_samples_mib if sample is not None
    ]

    elapsed = time.perf_counter() - started
    initial_z = float(initial_state[0, 2].item())
    final_z = float(final_state[0, 2].item())
    final_linear_speed = float(torch.linalg.vector_norm(final_state[0, 7:10]).item())

    checks = {
        "sim_device_is_cuda": str(sim.device).startswith("cuda"),
        "physics_uses_gpu_sim": physics_context.use_gpu_sim,
        "physics_uses_gpu_pipeline": physics_context.use_gpu_pipeline,
        "physics_broadphase_is_gpu": physics_context.get_broadphase_type() == "GPU",
        "state_tensor_is_cuda": final_state.device.type == "cuda",
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_has_sm120": "sm_120" in torch.cuda.get_arch_list(),
        "cube_fell": final_z < initial_z - 0.5,
        "cube_resting_on_ground": 0.08 <= final_z <= 0.14,
        "cuda_result_finite": bool(torch.isfinite(state_cuda_result).item()),
    }

    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "versions": {
            "python": platform.python_version(),
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab_package": _package_version("isaaclab"),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "torchaudio": _package_version("torchaudio"),
            "cuda_runtime": torch.version.cuda,
        },
        "system": {
            "platform": platform.platform(),
            "wsl": "microsoft" in platform.release().lower(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_arch_list": torch.cuda.get_arch_list(),
        },
        "simulation": {
            "device": str(sim.device),
            "physics_uses_gpu_sim": physics_context.use_gpu_sim,
            "physics_uses_gpu_pipeline": physics_context.use_gpu_pipeline,
            "physics_broadphase_type": physics_context.get_broadphase_type(),
            "state_tensor_device": str(final_state.device),
            "state_tensor_shape": list(final_state.shape),
            "steps": ARGS.steps,
            "dt_seconds": dt,
            "elapsed_seconds": elapsed,
            "steps_per_second": ARGS.steps / elapsed,
            "initial_state": initial_state.detach().cpu().tolist(),
            "final_state": final_state.detach().cpu().tolist(),
            "initial_z": initial_z,
            "final_z": final_z,
            "displacement_m": displacement.detach().cpu().tolist(),
            "final_linear_speed_m_s": final_linear_speed,
            "state_cuda_result": float(state_cuda_result.item()),
            "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "system_gpu_memory_used_mib_samples": valid_gpu_memory_samples,
            "system_gpu_memory_used_mib_peak": (
                max(valid_gpu_memory_samples) if valid_gpu_memory_samples else None
            ),
        },
        "environment": {
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        },
    }

    result_path = OUTPUT_DIR / "smoke_results.json"
    with result_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")

    logger.info("final root state: %s", final_state)
    logger.info("checks: %s", checks)
    logger.info("results: %s", result_path)
    if not result["passed"]:
        raise RuntimeError(f"Isaac Lab smoke test failed: {checks}")
    return result


def main() -> int:
    try:
        result = _run()
        print(json.dumps(result, indent=2))
        return 0
    finally:
        # Full Kit cleanup can hang in headless WSL when Vulkan graphics device
        # enumeration is unavailable even though CUDA PhysX has completed.
        SIMULATION_APP.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())

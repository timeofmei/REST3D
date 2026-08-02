"""Load and validate backend-neutral REST3D replay trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rest3d.sim.replay_scene import ReplaySceneSpec, load_replay_scene


@dataclass(frozen=True)
class ReplayTrajectory:
    """A validated replay and the scene assets needed to visualize it."""

    replay_dir: Path
    results_path: Path
    states_path: Path
    scene: ReplaySceneSpec
    states_rest: np.ndarray
    dt_seconds: float
    results: dict

    @property
    def names(self) -> tuple[str, ...]:
        return self.scene.names

    @property
    def frame_count(self) -> int:
        return int(self.states_rest.shape[0])


def load_replay_trajectory(replay_dir: str | Path) -> ReplayTrajectory:
    """Load a replay without importing either simulator backend.

    The saved REST trajectory uses ``[x, y, z, qw, qx, qy, qz, ...]`` and
    applies directly to the world-frame Stage 2 meshes.  Object order is
    checked against the common scene loader before any viewer is started.
    """

    replay_dir = Path(replay_dir).expanduser().resolve(strict=True)
    if not replay_dir.is_dir():
        raise NotADirectoryError(f"replay path is not a directory: {replay_dir}")

    results_path = replay_dir / "replay_results.json"
    states_path = replay_dir / "replay_states_rest.npy"
    if not results_path.is_file():
        raise FileNotFoundError(f"replay result is missing: {results_path}")
    if not states_path.is_file():
        raise FileNotFoundError(f"REST replay trajectory is missing: {states_path}")

    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)
    scene_record = results.get("scene")
    if not isinstance(scene_record, dict):
        raise ValueError("replay_results.json is missing the scene record")

    required_scene_fields = ("scene_tree", "scene_dir", "object_names")
    missing = [field for field in required_scene_fields if field not in scene_record]
    if missing:
        raise ValueError(f"replay scene record is missing fields: {missing}")

    expected_names = tuple(scene_record["object_names"])
    if not expected_names or len(set(expected_names)) != len(expected_names):
        raise ValueError("replay object names must be non-empty and unique")
    urdf_dir = scene_record.get("urdf_dir")
    scene = load_replay_scene(
        scene_record["scene_tree"],
        scene_record["scene_dir"],
        urdf_dir_override=urdf_dir,
    )
    if scene.names != expected_names:
        raise ValueError(
            "replay object order differs from the validated scene: "
            f"{expected_names!r} != {scene.names!r}"
        )

    states = np.load(states_path, allow_pickle=False)
    expected_shape_suffix = (len(expected_names), 13)
    if states.ndim != 3 or tuple(states.shape[1:]) != expected_shape_suffix:
        raise ValueError(
            "replay states must have shape [frames, objects, 13]; "
            f"got {states.shape}, expected [frames, {len(expected_names)}, 13]"
        )
    if states.shape[0] < 1:
        raise ValueError("replay trajectory contains no frames")
    if not np.isfinite(states).all():
        raise ValueError("replay trajectory contains non-finite values")

    states = np.asarray(states, dtype=np.float64).copy()
    quaternion_norms = np.linalg.norm(states[..., 3:7], axis=-1, keepdims=True)
    if np.any(quaternion_norms < 1.0e-12):
        raise ValueError("replay trajectory contains a zero-length quaternion")
    states[..., 3:7] /= quaternion_norms

    simulation = results.get("simulation", {})
    dt_seconds = float(simulation.get("dt_seconds", 1.0 / 60.0))
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError(f"invalid replay time step: {dt_seconds}")

    return ReplayTrajectory(
        replay_dir=replay_dir,
        results_path=results_path,
        states_path=states_path,
        scene=scene,
        states_rest=states,
        dt_seconds=dt_seconds,
        results=results,
    )


def trajectory_world_bounds(
    states_rest: np.ndarray, object_bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact union of transformed object AABB corners over all frames."""

    states = np.asarray(states_rest, dtype=np.float64)
    bounds = np.asarray(object_bounds, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] != 13:
        raise ValueError(f"states must have shape [frames, objects, 13], got {states.shape}")
    if bounds.shape != (states.shape[1], 2, 3):
        raise ValueError(
            f"object bounds must have shape [{states.shape[1]}, 2, 3], got {bounds.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(bounds).all():
        raise ValueError("states and object bounds must be finite")

    corner_selectors = np.array(
        [
            [x, y, z]
            for x in (0, 1)
            for y in (0, 1)
            for z in (0, 1)
        ],
        dtype=np.int64,
    )
    corners = np.empty((states.shape[1], 8, 3), dtype=np.float64)
    for object_index in range(states.shape[1]):
        corners[object_index] = bounds[
            object_index, corner_selectors, np.arange(3)[None, :]
        ]

    quaternions = states[..., 3:7]
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("states contain a zero-length quaternion")
    qw, qx, qy, qz = np.moveaxis(quaternions / norms, -1, 0)
    rotations = np.empty(states.shape[:2] + (3, 3), dtype=np.float64)
    rotations[..., 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    rotations[..., 0, 1] = 2.0 * (qx * qy - qz * qw)
    rotations[..., 0, 2] = 2.0 * (qx * qz + qy * qw)
    rotations[..., 1, 0] = 2.0 * (qx * qy + qz * qw)
    rotations[..., 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    rotations[..., 1, 2] = 2.0 * (qy * qz - qx * qw)
    rotations[..., 2, 0] = 2.0 * (qx * qz - qy * qw)
    rotations[..., 2, 1] = 2.0 * (qy * qz + qx * qw)
    rotations[..., 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)

    transformed = np.einsum("foij,okj->foki", rotations, corners)
    transformed += states[..., None, :3]
    return transformed.min(axis=(0, 1, 2)), transformed.max(axis=(0, 1, 2))

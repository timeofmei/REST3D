#!/usr/bin/env python3
"""View a saved REST3D replay in the browser with Viser.

Standalone implementation that supports both replay formats:

- Isaac Lab replay directory: contains ``replay_results.json`` and
  ``replay_states_rest.npy`` (frames x objects x 13, wxyz) plus the Stage-2
  OBJ meshes referenced by the results.
- Isaac Gym variant directory: contains ``replay_states.npy``
  (frames x objects x 7, xyzw) and an ``obj_files/`` folder with the
  world-frame meshes (e.g. ``.../stage3/global_scene``).

It does not import the Isaac backends or any other rest3d module, so it can
be run with a plain Python that has ``viser``, ``trimesh`` and ``numpy``.

Usage::

    python scripts/view_replay_viser.py output/.../replay_v1 --port 8080
    python scripts/view_replay_viser.py output/.../stage3/global_scene --port 8080

Then open http://localhost:8080 and press Play.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import viser

LOGGER = logging.getLogger("view_replay_viser")

STATE_COLS = 13  # [x, y, z, qw, qx, qy, qz, ...]


@dataclass(frozen=True)
class ReplayData:
    """Normalized replay: [frames, objects, 13] states with wxyz quaternions."""

    names: tuple[str, ...]
    states: np.ndarray
    mesh_dir: Path
    default_fps: int
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a saved REST3D replay without rerunning physics"
    )
    parser.add_argument(
        "replay_dir",
        type=Path,
        help=(
            "Replay directory (Isaac Lab: replay_results.json + "
            "replay_states_rest.npy; Isaac Gym: replay_states.npy + obj_files/)"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--fps",
        type=int,
        help="Initial playback FPS (default: the recorded physics frequency)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=None,
        help="Absolute initial camera distance in meters (overrides --camera-distance-scale)",
    )
    parser.add_argument("--camera-distance-scale", type=float, default=1.8)
    parser.add_argument("--camera-pitch-deg", type=float, default=25.0)
    parser.add_argument("--camera-azimuth-deg", type=float, default=270.0)
    parser.add_argument("--object-load-delay", type=float, default=0.0)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument(
        "--load-walls",
        action="store_true",
        help="Also load wall_* objects (Isaac Gym format only)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.fps is not None and not 1 <= args.fps <= 120:
        parser.error("--fps must be between 1 and 120")
    if args.start_frame < 0:
        parser.error("--start-frame cannot be negative")
    if args.camera_distance_scale <= 0.0:
        parser.error("--camera-distance-scale must be positive")
    if args.camera_distance is not None and args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    if args.object_load_delay < 0.0:
        parser.error("--object-load-delay cannot be negative")
    return args


def load_isaac_replay(replay_dir: Path, *, load_walls: bool = False) -> ReplayData:
    """Load the Isaac Lab replay format."""

    del load_walls  # Isaac Lab results already contain the full object list.
    replay_dir = Path(replay_dir).expanduser().resolve(strict=True)
    results_path = replay_dir / "replay_results.json"
    states_path = replay_dir / "replay_states_rest.npy"
    if not results_path.is_file() or not states_path.is_file():
        raise FileNotFoundError(
            f"missing replay_results.json or replay_states_rest.npy in {replay_dir}"
        )

    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    scene = results.get("scene")
    if not isinstance(scene, dict) or not scene.get("object_names"):
        raise ValueError("replay_results.json is missing a valid scene.object_names")

    names = tuple(scene["object_names"])
    states = np.load(states_path, allow_pickle=False).astype(np.float64)
    if states.ndim != 3 or states.shape[1] != len(names) or states.shape[2] != STATE_COLS:
        raise ValueError(
            f"replay states must have shape [frames, {len(names)}, {STATE_COLS}]; "
            f"got {states.shape}"
        )
    norms = np.linalg.norm(states[..., 3:7], axis=-1, keepdims=True)
    states[..., 3:7] /= np.where(norms < 1.0e-12, 1.0, norms)

    dt_seconds = float(results.get("simulation", {}).get("dt_seconds", 1.0 / 60.0))
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError(f"invalid replay time step: {dt_seconds}")
    default_fps = max(1, min(120, round(1.0 / dt_seconds)))
    mesh_dir = Path(results["scene"]["scene_dir"]).expanduser().resolve(strict=True)
    return ReplayData(
        names=names,
        states=states,
        mesh_dir=mesh_dir,
        default_fps=default_fps,
        source="isaac-lab",
    )


def load_gym_replay(variant_dir: Path, *, load_walls: bool = False) -> ReplayData:
    """Load the Isaac Gym variant format (replay_states.npy + obj_files/)."""

    variant_dir = Path(variant_dir).expanduser().resolve(strict=True)
    states_path = variant_dir / "replay_states.npy"
    obj_dir = variant_dir / "obj_files"
    if not states_path.is_file() or not obj_dir.is_dir():
        raise FileNotFoundError(
            f"Isaac Gym replay needs replay_states.npy and obj_files/ in {variant_dir}"
        )

    states = np.load(states_path, allow_pickle=False).astype(np.float64)
    if states.ndim != 3 or states.shape[2] != 7:
        raise ValueError(
            f"gym replay states must have shape [frames, objects, 7]; got {states.shape}"
        )

    names = sorted(path.stem for path in obj_dir.iterdir() if path.suffix.lower() == ".obj")
    if not load_walls:
        names = [name for name in names if not name.startswith("wall_")]
    if not names:
        raise ValueError(f"no usable OBJ assets in {obj_dir}")
    if states.shape[1] != len(names):
        raise ValueError(
            f"gym replay has {states.shape[1]} states but {len(names)} OBJ assets"
        )

    # Gym states store the quaternion as xyzw; viser uses wxyz.
    normalized = np.zeros((states.shape[0], len(names), STATE_COLS), dtype=np.float64)
    normalized[:, :, :3] = states[:, :, :3]
    normalized[:, :, 3:7] = states[:, :, 3:7][:, :, [3, 0, 1, 2]]
    norms = np.linalg.norm(normalized[..., 3:7], axis=-1, keepdims=True)
    normalized[..., 3:7] /= np.where(norms < 1.0e-12, 1.0, norms)

    return ReplayData(
        names=tuple(names),
        states=normalized,
        mesh_dir=variant_dir,
        default_fps=30,
        source="isaac-gym",
    )


def load_replay_data(path: Path, *, load_walls: bool = False) -> ReplayData:
    """Auto-detect the replay format and load a normalized replay."""

    path = Path(path).expanduser().resolve(strict=True)
    if (path / "replay_results.json").is_file() and (
        path / "replay_states_rest.npy"
    ).is_file():
        return load_isaac_replay(path, load_walls=load_walls)
    if (path / "replay_states.npy").is_file():
        return load_gym_replay(path, load_walls=load_walls)
    raise FileNotFoundError(
        f"no recognized replay data in {path} "
        "(expected replay_results.json + replay_states_rest.npy, "
        "or replay_states.npy + obj_files/)"
    )


def load_meshes(scene_dir: Path, names: tuple[str, ...]) -> dict[str, trimesh.Trimesh]:
    """Map each scene object to its OBJ mesh inside ``obj_files``."""

    obj_dir = scene_dir / "obj_files"
    if not obj_dir.is_dir():
        raise FileNotFoundError(f"missing obj directory: {obj_dir}")

    meshes: dict[str, trimesh.Trimesh] = {}
    for name in names:
        candidates = [
            path
            for path in obj_dir.iterdir()
            if path.suffix.lower() == ".obj" and path.stem.endswith(name)
        ]
        if not candidates:
            raise FileNotFoundError(f"no OBJ asset for scene object {name!r} in {obj_dir}")
        path = max(candidates, key=lambda candidate: len(candidate.stem))
        loaded = trimesh.load(path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError(f"mesh scene has no geometry: {path}")
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"unsupported mesh type for {path}: {type(loaded)}")
        if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
            raise ValueError(f"mesh has no usable geometry: {path}")
        meshes[name] = loaded
    return meshes


def world_bounds(
    states: np.ndarray, object_bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Union of transformed object AABB corners over all frames."""

    corner_selectors = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.int64,
    )
    corners = object_bounds[:, corner_selectors, np.arange(3)[None, :]]

    qw, qx, qy, qz = np.moveaxis(states[..., 3:7], -1, 0)
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


def camera_pose(
    center: np.ndarray,
    radius: float,
    *,
    distance_scale: float,
    pitch_deg: float,
    azimuth_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    distance = max(0.5, distance_scale * radius)
    pitch = np.radians(pitch_deg)
    azimuth = np.radians(azimuth_deg)
    horizontal = distance * np.cos(pitch)
    position = (
        float(center[0] + horizontal * np.cos(azimuth)),
        float(center[1] + distance * np.sin(pitch)),
        float(center[2] + horizontal * np.sin(azimuth)),
    )
    return position, tuple(float(value) for value in center)


def run_viewer(args: argparse.Namespace) -> None:
    replay = load_replay_data(args.replay_dir, load_walls=args.load_walls)
    names = replay.names
    states = replay.states
    if args.start_frame >= len(states):
        raise ValueError(
            f"--start-frame {args.start_frame} exceeds final frame {len(states) - 1}"
        )

    meshes = load_meshes(replay.mesh_dir, names)
    object_bounds = np.stack([np.asarray(mesh.bounds, dtype=np.float64) for mesh in meshes.values()])
    bounds_min, bounds_max = world_bounds(states, object_bounds)
    center = 0.5 * (bounds_min + bounds_max)
    radius = max(0.5, float(np.linalg.norm(bounds_max - bounds_min)) * 0.5)
    camera_position, camera_target = camera_pose(
        center,
        radius,
        distance_scale=args.camera_distance_scale,
        pitch_deg=args.camera_pitch_deg,
        azimuth_deg=args.camera_azimuth_deg,
    )
    if args.camera_distance is not None:
        # Keep the same direction from the scene center, at an absolute distance.
        direction = np.asarray(camera_position) - np.asarray(camera_target)
        direction /= max(np.linalg.norm(direction), 1.0e-9)
        camera_position = tuple(
            float(value) for value in np.asarray(camera_target) + direction * args.camera_distance
        )
    camera_distance = float(np.linalg.norm(np.asarray(camera_position) - np.asarray(camera_target)))
    playback_fps = args.fps or replay.default_fps

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    # Set the authoritative initial camera before any client connects;
    # otherwise the client's built-in default camera can override our pose
    # after connection.
    server.initial_camera.position = camera_position
    server.initial_camera.look_at = camera_target
    if not args.no_grid:
        server.scene.add_grid(
            "/ground",
            width=max(2.0, radius * 6.0),
            height=max(2.0, radius * 6.0),
            plane="xz",
            cell_size=0.25,
            section_size=1.0,
        )

    handles = {}
    for index, name in enumerate(names):
        handles[name] = server.scene.add_mesh_trimesh(
            name=f"/objects/object_{index:04d}",
            mesh=meshes[name],
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        LOGGER.info("loaded object %d/%d: %s", index + 1, len(meshes), name)
        if args.object_load_delay:
            time.sleep(args.object_load_delay)

    frame_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=len(states) - 1,
        step=1,
        initial_value=args.start_frame,
    )
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=120, step=1, initial_value=playback_fps
    )
    play_control = server.gui.add_checkbox("▶ Play", initial_value=False)

    def set_frame(frame_index: int) -> None:
        frame = states[int(frame_index)]
        with server.atomic():
            for object_index, name in enumerate(names):
                state = frame[object_index]
                handles[name].position = tuple(float(value) for value in state[:3])
                handles[name].wxyz = tuple(float(value) for value in state[3:7])

    def set_camera(client) -> None:
        client.camera.position = camera_position
        client.camera.look_at = camera_target

    def apply_camera_to_all() -> None:
        for client in server.get_clients().values():
            try:
                set_camera(client)
            except Exception:
                LOGGER.warning("failed to apply camera to a client", exc_info=True)

    @server.on_client_connect
    def _on_client_connect(client) -> None:
        set_camera(client)
        LOGGER.info(
            "client connected; initial camera applied (distance=%.3f m)",
            camera_distance,
        )

    @frame_slider.on_update
    def _on_frame_update(_) -> None:
        set_frame(int(frame_slider.value))

    reset_button = server.gui.add_button(
        "Reset camera",
        hint="Re-apply the initial camera pose",
        order=10.0,
    )

    @reset_button.on_click
    def _on_reset_click(_) -> None:
        apply_camera_to_all()
        LOGGER.info("camera reset requested by client")

    set_frame(args.start_frame)
    LOGGER.info("replay=%s format=%s", args.replay_dir, replay.source)
    LOGGER.info("frames=%d objects=%d fps=%d", len(states), len(meshes), playback_fps)
    LOGGER.info(
        "initial camera: position=%s target=%s distance=%.3f m",
        tuple(round(float(value), 3) for value in camera_position),
        tuple(round(float(value), 3) for value in camera_target),
        camera_distance,
    )
    LOGGER.info("open http://localhost:%d and press Play", server.get_port())

    try:
        while True:
            if play_control.value:
                next_frame = int(frame_slider.value) + 1
                if next_frame >= len(states):
                    frame_slider.value = 0
                    set_frame(0)
                    play_control.value = False
                    continue
                frame_slider.value = next_frame
                set_frame(next_frame)
                time.sleep(1.0 / max(1, int(fps_slider.value)))
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        LOGGER.info("viewer stopped")
    finally:
        server.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_viewer(parse_args())


if __name__ == "__main__":
    main()

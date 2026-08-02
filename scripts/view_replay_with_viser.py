#!/usr/bin/env python3
"""Interactively view a saved REST3D replay in a browser with Viser."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from rest3d.sim.replay_trajectory import (
    ReplayTrajectory,
    load_replay_trajectory,
    trajectory_world_bounds,
)


LOGGER = logging.getLogger("viser_replay")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a saved Isaac Gym or Isaac Lab replay without rerunning physics"
    )
    parser.add_argument(
        "replay_dir",
        type=Path,
        help="Replay directory containing replay_results.json and replay_states_rest.npy",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--fps",
        type=int,
        help="Initial playback FPS (default: the recorded physics frequency)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.8)
    parser.add_argument("--camera-pitch-deg", type=float, default=25.0)
    parser.add_argument("--camera-azimuth-deg", type=float, default=270.0)
    parser.add_argument("--object-load-delay", type=float, default=0.0)
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.fps is not None and not 1 <= args.fps <= 120:
        parser.error("--fps must be between 1 and 120")
    if args.start_frame < 0:
        parser.error("--start-frame cannot be negative")
    if args.camera_distance_scale <= 0.0:
        parser.error("--camera-distance-scale must be positive")
    if args.object_load_delay < 0.0:
        parser.error("--object-load-delay cannot be negative")
    return args


def _load_meshes(
    trajectory: ReplayTrajectory,
) -> tuple[dict[str, trimesh.Trimesh], np.ndarray]:
    meshes: dict[str, trimesh.Trimesh] = {}
    bounds: list[np.ndarray] = []
    for spec in trajectory.scene.objects:
        loaded = trimesh.load(spec.obj_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError(f"mesh scene has no geometry: {spec.obj_path}")
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"unsupported mesh type for {spec.obj_path}: {type(loaded)}")
        if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
            raise ValueError(f"mesh has no usable geometry: {spec.obj_path}")
        meshes[spec.name] = loaded
        bounds.append(np.asarray(loaded.bounds, dtype=np.float64))
    return meshes, np.stack(bounds)


def _camera_pose(
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
    trajectory = load_replay_trajectory(args.replay_dir)
    if args.start_frame >= trajectory.frame_count:
        raise ValueError(
            f"--start-frame {args.start_frame} exceeds final frame "
            f"{trajectory.frame_count - 1}"
        )
    meshes, object_bounds = _load_meshes(trajectory)
    bounds_min, bounds_max = trajectory_world_bounds(
        trajectory.states_rest, object_bounds
    )
    center = 0.5 * (bounds_min + bounds_max)
    radius = max(0.5, float(np.linalg.norm(bounds_max - bounds_min)) * 0.5)
    camera_position, camera_target = _camera_pose(
        center,
        radius,
        distance_scale=args.camera_distance_scale,
        pitch_deg=args.camera_pitch_deg,
        azimuth_deg=args.camera_azimuth_deg,
    )
    playback_fps = args.fps or max(1, min(120, round(1.0 / trajectory.dt_seconds)))

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    if not args.no_grid:
        server.scene.add_grid(
            "/ground",
            width=max(10.0, radius * 6.0),
            height=max(10.0, radius * 6.0),
            plane="xz",
            cell_size=0.25,
            section_size=1.0,
        )

    handles = {}
    for index, name in enumerate(trajectory.names):
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
        max=trajectory.frame_count - 1,
        step=1,
        initial_value=args.start_frame,
    )
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=120, step=1, initial_value=playback_fps
    )
    play_control = server.gui.add_checkbox("▶ Play", initial_value=False)

    def set_frame(frame_index: int) -> None:
        frame_index = int(frame_index)
        states = trajectory.states_rest[frame_index]
        with server.atomic():
            for object_index, name in enumerate(trajectory.names):
                state = states[object_index]
                handles[name].position = tuple(float(value) for value in state[:3])
                handles[name].wxyz = tuple(float(value) for value in state[3:7])

    def set_camera(client) -> None:
        client.camera.position = camera_position
        client.camera.look_at = camera_target

    @server.on_client_connect
    def _on_client_connect(client) -> None:
        set_camera(client)

    @frame_slider.on_update
    def _on_frame_update(_) -> None:
        set_frame(int(frame_slider.value))

    set_frame(args.start_frame)
    LOGGER.info("replay=%s", trajectory.replay_dir)
    LOGGER.info("frames=%d objects=%d fps=%d", trajectory.frame_count, len(meshes), playback_fps)
    LOGGER.info("open http://localhost:%d and press Play", server.get_port())

    try:
        while True:
            if play_control.value:
                next_frame = int(frame_slider.value) + 1
                if next_frame >= trajectory.frame_count:
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
    run_viewer(_parse_args())


if __name__ == "__main__":
    main()

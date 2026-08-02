#!/usr/bin/env python3
"""Render saved REST3D scenes or replay frames for visual inspection."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Pyrender must select EGL before importing PyOpenGL on headless WSL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image

from rest3d.sim.replay_trajectory import load_replay_trajectory


@dataclass(frozen=True)
class RenderObject:
    name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray


VIEW_DIRECTIONS = {
    "front": np.array([0.0, 0.15, -1.0]),
    "front-high": np.array([0.0, 0.75, -1.0]),
    "front-right": np.array([0.75, 0.45, -1.0]),
    "front-left": np.array([-0.75, 0.45, -1.0]),
    "right": np.array([1.0, 0.15, 0.0]),
    "left": np.array([-1.0, 0.15, 0.0]),
    "back": np.array([0.0, 0.15, 1.0]),
    "top": np.array([0.0, 1.0, -0.01]),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed camera views without starting a simulator or browser"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--replay-dir",
        type=Path,
        help="Replay containing replay_results.json and replay_states_rest.npy",
    )
    source.add_argument(
        "--scene-dir",
        type=Path,
        help="Scene directory containing obj_files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for PNG snapshots; an existing path is rejected",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Replay frame to render; negative indices count from the end",
    )
    parser.add_argument(
        "--views",
        default="front-high,front-right,right,left",
        help=f"Comma-separated views chosen from: {','.join(VIEW_DIRECTIONS)}",
    )
    parser.add_argument(
        "--focus",
        default="",
        help="Comma-separated case-insensitive name fragments used to frame a close-up",
    )
    parser.add_argument(
        "--only-focus",
        action="store_true",
        help="Render only objects matched by --focus, in addition to framing them",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--yfov-deg", type=float, default=42.0)
    parser.add_argument("--distance-scale", type=float, default=1.35)
    parser.add_argument(
        "--ground-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a reference plane and grid at world y=0",
    )
    return parser.parse_args()


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene has no geometry: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices):
        raise ValueError(f"mesh has no usable vertices: {path}")
    return loaded


def _load_objects(args: argparse.Namespace) -> tuple[list[RenderObject], int | None]:
    if args.replay_dir is not None:
        trajectory = load_replay_trajectory(args.replay_dir)
        frame = args.frame
        if frame < 0:
            frame += trajectory.frame_count
        if not 0 <= frame < trajectory.frame_count:
            raise ValueError(
                f"frame {args.frame} is outside a {trajectory.frame_count}-frame replay"
            )
        objects = []
        for object_index, spec in enumerate(trajectory.scene.objects):
            state = trajectory.states_rest[frame, object_index]
            transform = trimesh.transformations.quaternion_matrix(state[3:7])
            transform[:3, 3] = state[:3]
            objects.append(RenderObject(spec.name, _as_mesh(spec.obj_path), transform))
        return objects, frame

    scene_dir = args.scene_dir.expanduser().resolve(strict=True)
    obj_dir = scene_dir / "obj_files"
    if not obj_dir.is_dir():
        raise NotADirectoryError(f"scene obj_files directory is missing: {obj_dir}")
    paths = sorted(obj_dir.glob("*.obj"))
    if not paths:
        raise FileNotFoundError(f"no OBJ meshes found in {obj_dir}")
    return [
        RenderObject(path.stem, _as_mesh(path), np.eye(4, dtype=np.float64))
        for path in paths
    ], None


def _transformed_bounds(obj: RenderObject) -> np.ndarray:
    corners = trimesh.bounds.corners(obj.mesh.bounds)
    return trimesh.transform_points(corners, obj.transform)


def _framing_bounds(
    objects: list[RenderObject], focus_fragments: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    framed = objects
    if focus_fragments:
        framed = [
            obj
            for obj in objects
            if any(fragment in obj.name.lower() for fragment in focus_fragments)
        ]
        if not framed:
            raise ValueError(f"no object names matched --focus {focus_fragments!r}")
    points = np.concatenate([_transformed_bounds(obj) for obj in framed], axis=0)
    return points.min(axis=0), points.max(axis=0)


def _camera_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    camera_z = position - target
    camera_z /= np.linalg.norm(camera_z)
    reference_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(camera_z, reference_up))) > 0.995:
        reference_up = np.array([0.0, 0.0, -1.0])
    camera_x = np.cross(reference_up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((camera_x, camera_y, camera_z))
    pose[:3, 3] = position
    return pose


def _render_view(
    objects: list[RenderObject],
    *,
    center: np.ndarray,
    radius: float,
    direction: np.ndarray,
    width: int,
    height: int,
    yfov_deg: float,
    distance_scale: float,
    ground_grid: bool,
) -> np.ndarray:
    scene = pyrender.Scene(
        bg_color=np.array([250, 250, 250, 255], dtype=np.uint8),
        ambient_light=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    if ground_grid:
        half_extent = max(0.5, 2.0 * radius)
        floor = trimesh.creation.box(
            extents=(2.0 * half_extent, 0.002, 2.0 * half_extent)
        )
        floor.apply_translation((center[0], -0.0015, center[2]))
        floor.visual.vertex_colors = np.tile(
            np.array([232, 232, 232, 255], dtype=np.uint8),
            (len(floor.vertices), 1),
        )
        scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False), name="ground")

        line_width = max(0.001, 0.004 * radius)
        grid_lines = []
        for offset in np.linspace(-half_extent, half_extent, 11):
            x_line = trimesh.creation.box(
                extents=(line_width, 0.001, 2.0 * half_extent)
            )
            x_line.apply_translation(
                (center[0] + offset, 0.0005, center[2])
            )
            z_line = trimesh.creation.box(
                extents=(2.0 * half_extent, 0.001, line_width)
            )
            z_line.apply_translation(
                (center[0], 0.0005, center[2] + offset)
            )
            grid_lines.extend((x_line, z_line))
        grid = trimesh.util.concatenate(grid_lines)
        grid.visual.vertex_colors = np.tile(
            np.array([175, 175, 175, 255], dtype=np.uint8),
            (len(grid.vertices), 1),
        )
        scene.add(pyrender.Mesh.from_trimesh(grid, smooth=False), name="ground-grid")

    for obj in objects:
        scene.add(
            pyrender.Mesh.from_trimesh(obj.mesh, smooth=False),
            pose=obj.transform,
            name=obj.name,
        )

    normalized_direction = direction / np.linalg.norm(direction)
    aspect = width / height
    yfov = np.radians(yfov_deg)
    minimum_distance = radius / max(np.sin(yfov * 0.5), 1.0e-6)
    if aspect < 1.0:
        minimum_distance /= aspect
    position = center + normalized_direction * minimum_distance * distance_scale
    pose = _camera_pose(position, center)
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect), pose=pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=pose
    )

    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return color


def main() -> None:
    args = _parse_args()
    if args.width < 64 or args.height < 64:
        raise ValueError("snapshot dimensions must both be at least 64 pixels")
    if not 1.0 < args.yfov_deg < 170.0:
        raise ValueError("--yfov-deg must be between 1 and 170")
    if args.distance_scale <= 1.0:
        raise ValueError("--distance-scale must be greater than 1")

    views = tuple(part.strip() for part in args.views.split(",") if part.strip())
    unknown = sorted(set(views) - set(VIEW_DIRECTIONS))
    if not views or unknown:
        raise ValueError(f"invalid --views selection; unknown views: {unknown}")
    focus = tuple(part.strip().lower() for part in args.focus.split(",") if part.strip())

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    objects, frame = _load_objects(args)
    bounds_min, bounds_max = _framing_bounds(objects, focus)
    if args.only_focus:
        if not focus:
            raise ValueError("--only-focus requires at least one --focus fragment")
        objects = [
            obj
            for obj in objects
            if any(fragment in obj.name.lower() for fragment in focus)
        ]
    center = 0.5 * (bounds_min + bounds_max)
    radius = max(0.05, 0.5 * float(np.linalg.norm(bounds_max - bounds_min)))

    frame_label = "scene" if frame is None else f"frame_{frame:04d}"
    for view in views:
        image = _render_view(
            objects,
            center=center,
            radius=radius,
            direction=VIEW_DIRECTIONS[view],
            width=args.width,
            height=args.height,
            yfov_deg=args.yfov_deg,
            distance_scale=args.distance_scale,
            ground_grid=args.ground_grid,
        )
        output_path = output_dir / f"{frame_label}_{view}.png"
        Image.fromarray(image).save(output_path)
        print(output_path)


if __name__ == "__main__":
    main()

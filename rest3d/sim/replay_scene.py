"""Backend-neutral replay scene discovery and REST3D/Isaac Lab transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rest3d.utils.mesh import load_trimesh_any


# Right-handed +90 degree rotation around X: REST3D Y-up -> Isaac Lab Z-up.
REST_TO_LAB_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
LAB_TO_REST_ROTATION = REST_TO_LAB_ROTATION.T
REST_TO_LAB_QUAT_WXYZ = np.array(
    [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float64
)
LAB_TO_REST_QUAT_WXYZ = np.array(
    [np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0], dtype=np.float64
)


@dataclass(frozen=True)
class ReplayObjectSpec:
    """One rigid object and its backend-neutral replay metadata."""

    name: str
    obj_path: Path
    urdf_path: Path
    fixed: bool


@dataclass(frozen=True)
class ReplaySceneSpec:
    """Validated input scene shared by replay backends."""

    scene_tree_path: Path
    scene_dir: Path
    objects: tuple[ReplayObjectSpec, ...]
    bounds_min_rest: tuple[float, float, float]
    bounds_max_rest: tuple[float, float, float]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objects)

    @property
    def fixed_names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objects if obj.fixed)

    @property
    def movable_names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objects if not obj.fixed)


def _read_scene_tree(
    path: Path,
) -> tuple[set[str], set[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8") as file:
        tree = json.load(file)

    roots = set(tree.get("roots", []))
    nodes = set(tree.get("nodes", []))
    node_info: dict[str, dict[str, str]] = {}
    parent_of: dict[str, str] = {}
    for edge in tree.get("edges", []):
        child = edge["child"]
        parent = edge["parent"]
        if child in parent_of:
            raise ValueError(f"scene node has multiple parents: {child}")
        parent_of[child] = parent
        node_info[child] = {
            "parent": parent,
            "relation": edge["relation"],
            "type": edge["type"],
        }

    declared = roots | nodes
    missing_nodes = set(parent_of) - declared
    if missing_nodes:
        raise ValueError(f"scene tree edges contain undeclared children: {sorted(missing_nodes)}")
    unknown_parents = set(parent_of.values()) - declared
    if unknown_parents:
        raise ValueError(f"scene tree edges contain undeclared parents: {sorted(unknown_parents)}")
    orphan_nodes = nodes - set(parent_of)
    if orphan_nodes:
        raise ValueError(f"scene tree nodes have no parent edge: {sorted(orphan_nodes)}")

    for node in parent_of:
        visited: set[str] = set()
        current = node
        while current in parent_of:
            if current in visited:
                raise ValueError(f"cycle detected at scene node: {current}")
            visited.add(current)
            current = parent_of[current]
    return declared, roots, node_info


def _is_fixed(node: str, roots: set[str], node_info: dict[str, dict[str, str]]) -> bool:
    if node in roots:
        return True
    info = node_info[node]
    return (
        info["type"] == "fixed"
        or info["relation"] in {"attach", "hang"}
    )


def load_replay_scene(scene_tree_path: str | Path, scene_dir: str | Path) -> ReplaySceneSpec:
    """Load and strictly validate matching OBJ/URDF assets without modifying them."""

    scene_tree_path = Path(scene_tree_path).expanduser().resolve(strict=True)
    scene_dir = Path(scene_dir).expanduser().resolve(strict=True)
    obj_dir = scene_dir / "obj_files"
    urdf_dir = scene_dir / "urdf_files"
    if not obj_dir.is_dir():
        raise FileNotFoundError(f"OBJ directory not found: {obj_dir}")
    if not urdf_dir.is_dir():
        raise FileNotFoundError(f"URDF directory not found: {urdf_dir}")

    declared, roots, node_info = _read_scene_tree(scene_tree_path)
    obj_by_name = {path.stem: path for path in sorted(obj_dir.glob("*.obj"))}
    urdf_by_name = {path.stem: path for path in sorted(urdf_dir.glob("*.urdf"))}
    if not obj_by_name:
        raise ValueError(f"no OBJ assets found in: {obj_dir}")

    unknown = set(obj_by_name) - declared
    if unknown:
        raise ValueError(f"OBJ assets are not declared in the scene tree: {sorted(unknown)}")
    missing_urdf = set(obj_by_name) - set(urdf_by_name)
    if missing_urdf:
        raise FileNotFoundError(f"URDF assets missing for: {sorted(missing_urdf)}")
    extra_urdf = set(urdf_by_name) - set(obj_by_name)
    if extra_urdf:
        raise ValueError(f"URDF assets have no matching OBJ: {sorted(extra_urdf)}")

    objects: list[ReplayObjectSpec] = []
    bounds_min: list[np.ndarray] = []
    bounds_max: list[np.ndarray] = []
    for name in sorted(obj_by_name):
        mesh = load_trimesh_any(str(obj_by_name[name]))
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"mesh has no usable geometry: {obj_by_name[name]}")
        bounds_min.append(np.asarray(mesh.bounds[0], dtype=np.float64))
        bounds_max.append(np.asarray(mesh.bounds[1], dtype=np.float64))
        objects.append(
            ReplayObjectSpec(
                name=name,
                obj_path=obj_by_name[name],
                urdf_path=urdf_by_name[name],
                fixed=_is_fixed(name, roots, node_info),
            )
        )

    scene_min = np.min(np.stack(bounds_min), axis=0)
    scene_max = np.max(np.stack(bounds_max), axis=0)
    return ReplaySceneSpec(
        scene_tree_path=scene_tree_path,
        scene_dir=scene_dir,
        objects=tuple(objects),
        bounds_min_rest=tuple(float(value) for value in scene_min),
        bounds_max_rest=tuple(float(value) for value in scene_max),
    )


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("zero-length quaternion cannot be normalized")
    return quaternions / norm


def rest_poses_to_lab(poses_wxyz: np.ndarray) -> np.ndarray:
    """Map REST3D Y-up poses `[x,y,z,qw,qx,qy,qz]` to Isaac Lab Z-up."""

    poses = np.asarray(poses_wxyz, dtype=np.float64)
    if poses.shape[-1] != 7:
        raise ValueError(f"expected poses with final dimension 7, got {poses.shape}")
    result = poses.copy()
    result[..., :3] = poses[..., :3] @ REST_TO_LAB_ROTATION.T
    result[..., 3:7] = _normalize_quaternions(
        _quat_multiply_wxyz(REST_TO_LAB_QUAT_WXYZ, poses[..., 3:7])
    )
    return result


def lab_states_to_rest(states_wxyz: np.ndarray) -> np.ndarray:
    """Map Isaac Lab Z-up root states to REST3D Y-up coordinates."""

    states = np.asarray(states_wxyz, dtype=np.float64)
    if states.shape[-1] != 13:
        raise ValueError(f"expected root states with final dimension 13, got {states.shape}")
    result = states.copy()
    result[..., :3] = states[..., :3] @ LAB_TO_REST_ROTATION.T
    result[..., 3:7] = _normalize_quaternions(
        _quat_multiply_wxyz(LAB_TO_REST_QUAT_WXYZ, states[..., 3:7])
    )
    result[..., 7:10] = states[..., 7:10] @ LAB_TO_REST_ROTATION.T
    result[..., 10:13] = states[..., 10:13] @ LAB_TO_REST_ROTATION.T
    return result

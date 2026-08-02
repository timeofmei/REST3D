import trimesh
import numpy as np
import os
import torch
from copy import deepcopy
from collections import defaultdict



def compute_rigid_transform(src, dst):
    """
    Solve the rigid transform (R, t) such that ``dst ~= R @ src + t``,
    using SVD / Procrustes. The reflection component is removed so R is a
    proper rotation (det = +1).
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    return R, t


def save_obj_with_rigid_transform(input_obj_path, output_obj_path, R, t):
    """
    Re-emit an OBJ file with each vertex position mapped by ``R @ xyz + t``.
    All other lines (vertex colors, faces, normals, etc.) pass through unchanged.
    """
    with open(input_obj_path, 'r') as fin, open(output_obj_path, 'w') as fout:
        for line in fin:
            if line.startswith('v '):
                parts = line.split()
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                new_xyz = R @ xyz + t
                extra = parts[4:]  # vertex colors (r g b) etc.
                if extra:
                    fout.write(f"v {new_xyz[0]} {new_xyz[1]} {new_xyz[2]} {' '.join(extra)}\n")
                else:
                    fout.write(f"v {new_xyz[0]} {new_xyz[1]} {new_xyz[2]}\n")
            else:
                fout.write(line)


def rotation_matrix_from_vectors(vec1, vec2):
    """Rotation matrix that maps unit-vector ``vec1`` onto ``vec2`` (Rodrigues)."""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-10:  # already aligned
        return np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2)) 


def axes_to_cylinder_mesh(axes, centroid, up_axis_idx=None, radius=0.02, up_len=8.0, other_len=2.0):
    """
    axes: (3,3) columns are x/y/z axes (e.g., bbox_info["axes"])
    centroid: (3,) world coords
    up_axis_idx: which axis to highlight as up (green + longer). If None, auto-pick closest to world Y.
    return: trimesh.Trimesh (concatenated cylinders)
    """
    axes = np.asarray(axes, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)

    if up_axis_idx is None:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up_axis_idx = int(np.argmax(np.abs(axes.T @ y_axis)))

    mesh_axes = []
    axis_colors = [
        [255, 0, 0, 255],  # one non-up axis
        [0, 0, 255, 255],  # the other non-up axis
    ]
    color_idx = 0

    for i in range(3):
        direction = axes[:, i]
        n = np.linalg.norm(direction) + 1e-12
        direction = direction / n

        if i == up_axis_idx:
            axis_len = float(up_len)
            axis_color = [0, 255, 0, 255]  # up green
        else:
            axis_len = float(other_len)
            axis_color = axis_colors[color_idx]
            color_idx += 1

        start = centroid
        end = centroid + direction * axis_len

        cyl = trimesh.creation.cylinder(radius=radius, segment=[start, end])
        cyl.visual.vertex_colors = np.tile(axis_color, (len(cyl.vertices), 1))
        mesh_axes.append(cyl)

    return trimesh.util.concatenate(mesh_axes)


def aabb_corners_from_points(xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: (N,3)
    return corners: (8,3) in the same frame as xyz
    """
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    x0, y0, z0 = mins
    x1, y1, z1 = maxs

    corners = torch.stack([
        torch.stack([x0, y0, z0]),
        torch.stack([x1, y0, z0]),
        torch.stack([x1, y1, z0]),
        torch.stack([x0, y1, z0]),
        torch.stack([x0, y0, z1]),
        torch.stack([x1, y0, z1]),
        torch.stack([x1, y1, z1]),
        torch.stack([x0, y1, z1]),
    ], dim=0)
    return corners


def get_object_bboxes_from_outputs(*outputs, in_place=False):
    """
    Compute per-object 3D AABB + axes (from pose rotation)
    """
    from sam3d_objects.utils.visualization import SceneVisualizer
    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    bbox_list = []
    axes_list = []


    for output in outputs:
        xyz_local = output["gaussian"][0].get_xyz.clone()  # (N,3) local/gaussian frame
        # ---- AABB in local frame (8 corners) ----
        bbox_local = aabb_corners_from_points(xyz_local)  # (8,3)

        # ---- move gaussian to scene frame ----
        PC_bbox = SceneVisualizer.object_pointcloud(
        points_local=bbox_local.unsqueeze(0),     # (1,8,3)
        quat_l2c=output["rotation"],
        trans_l2c=output["translation"],
        scale_l2c=output["scale"],
        )
        bbox_scene = PC_bbox.points_list()[0]  # (8,3) torch tensor
        bbox_center = bbox_scene.mean(dim=0)  # (3,) torch tensor

        # ---- Extract axes from OBB vertices using PCA ----
        # Convert to numpy for PCA
        bbox_np = bbox_scene.detach().cpu().numpy().astype(np.float64)
        centroid_np = bbox_center.detach().cpu().numpy().astype(np.float64)

        # Center the vertices
        centered = bbox_np - centroid_np

        # Compute covariance matrix
        cov = centered.T @ centered / len(bbox_np)

        # Eigen decomposition to get principal axes
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalues (largest first)
        idx = eigenvalues.argsort()[::-1]
        axes_scene = eigenvectors[:, idx]  # (3,3) columns are the three axes

        # Ensure right-handed coordinate system
        if np.linalg.det(axes_scene) < 0:
            axes_scene[:, 2] *= -1  
        bbox_list.append({
            "corners": bbox_scene.detach().cpu().numpy(),
            "center": bbox_center.detach().cpu().numpy(),
        })
        axes_list.append(axes_scene)

    return bbox_list, axes_list


def align_bbox_to_y_up_from_axes(mesh, bbox, axes, y_axis=np.array([0.0, 1.0, 0.0])):
    """
    Align mesh to y-up using provided axes (3x3).

    Args:
        mesh: trimesh object
        bbox: bbox_corner (8,3) array-like, corners of the bbox in current frame; bbox center
        axes: (3,3) array-like, columns are x/y/z axes in current frame

    Returns:
        mesh: up-axis fixed mesh
        bbox: up-axis fixed bbox
        axes: up-axis fixed axes
    """
    axes = np.asarray(axes, dtype=np.float64)  # (3,3)

    # choose which provided axis is "up"
    up_axis = _robust_up_direction(mesh, axes, y_axis)

    # rotate up_axis -> y_axis
    rotation_to_y = rotation_matrix_from_vectors(up_axis, y_axis)

    # apply rotation around centroid
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices = (mesh.vertices - centroid) @ rotation_to_y.T + centroid
    bbox_corners = bbox["corners"]
    bbox["corners"] = (bbox_corners - centroid) @ rotation_to_y.T + centroid
    axes = rotation_to_y @ axes 

    return mesh, bbox, axes


def _dominant_face_direction(mesh, min_area_fraction=0.12, merge_cos=0.94):
    """Area-weighted normal of the mesh's largest flat region, or None.

    Returns None when no single flat region covers enough of the surface
    (e.g. spheres or heavily curved objects).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return None
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(cross, axis=1)
    lengths[lengths == 0.0] = 1.0
    normals = cross / lengths[:, None]
    areas = 0.5 * lengths
    total = float(areas.sum())
    if total <= 0.0:
        return None

    # Quantize directions into clusters and accumulate their areas (O(F)).
    rounded = np.round(normals, 1)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    cluster_area = np.bincount(inverse, weights=areas)
    cluster_sum = np.stack(
        [
            np.bincount(inverse, weights=normals[:, 0]),
            np.bincount(inverse, weights=normals[:, 1]),
            np.bincount(inverse, weights=normals[:, 2]),
        ],
        axis=1,
    )
    cluster_norm = np.linalg.norm(cluster_sum, axis=1)
    cluster_norm[cluster_norm == 0.0] = 1.0
    cluster_dir = cluster_sum / cluster_norm[:, None]

    # Greedily merge nearby clusters into the largest flat region.
    used = np.zeros(len(cluster_area), dtype=bool)
    best_area = 0.0
    best_dir = None
    while True:
        available = np.where(~used)[0]
        if len(available) == 0:
            break
        seed = available[int(np.argmax(cluster_area[available]))]
        used[seed] = True
        region_area = cluster_area[seed]
        region_dir = cluster_dir[seed].copy()
        while True:
            candidates = np.where(
                ~used & (np.abs(cluster_dir @ region_dir) >= merge_cos)
            )[0]
            if len(candidates) == 0:
                break
            pick = candidates[int(np.argmax(cluster_area[candidates]))]
            used[pick] = True
            region_area = region_area + cluster_area[pick]
            region_dir = region_dir * region_area + cluster_dir[pick] * cluster_area[pick]
            region_dir = region_dir / (np.linalg.norm(region_dir) + 1e-12)
        if region_area > best_area:
            best_area = region_area
            best_dir = region_dir
    if best_dir is None or best_area / total < min_area_fraction:
        return None
    return best_dir


def _robust_up_direction(mesh, axes, y_axis, side_surface_cos=0.5):
    """Estimate the object's up direction from its own geometry.

    The OBB axes returned by the reconstruction are reliable as a basis, but
    selecting the up axis as the one closest to the *current* frame's y is not:
    the reconstruction frame is camera-relative, so the heuristic regularly
    picks a horizontal axis for objects with a dominant flat top (tables,
    shoes, boards).  Prefer the area-weighted dominant face normal; fall back
    to the OBB-y heuristic for objects without a dominant flat region.

    In the scene-upright (coarse-aligned) frame the two cases are told apart
    by the dominant plane's inclination: a horizontal top surface points
    nearly straight up, while a *side* surface (e.g. a seated person's back)
    is the largest flat region of some objects yet is nearly horizontal, so a
    dominant direction far from the vertical must not be rotated to become
    the up axis.
    """
    direction = _dominant_face_direction(mesh)
    if direction is not None and direction @ y_axis < 0.0:
        direction = -direction
    if direction is not None and abs(float(direction @ y_axis)) >= side_surface_cos:
        return direction
    # no dominant plane, or the dominant plane is a side surface: rely on the
    # OBB axis closest to the (scene-upright) frame's y
    dots = np.abs(np.asarray(axes, dtype=np.float64).T @ y_axis)
    return axes[:, int(np.argmax(dots))]


def get_scene_up_axis_from_ref_axes(axes_ref, meshes_ref=None, y_axis=np.array([0.0, 1.0, 0.0]), weighted=True, eps=1e-12):
    """
    Estimate a common scene up-axis from reference object axes (e.g., floors).

    Args:
        axes_ref: list of (3,3) arrays, each column is an OBB axis in world frame
        meshes_ref: list of trimesh meshes corresponding to axes_ref (optional)
        y_axis: world up reference (default [0,1,0])
        weighted: whether to weight by mesh size
        eps: numerical epsilon

    Returns:
        scene_up_axis: (3,) unit vector
    """
    up_axes = []
    weights = []

    for i, A in enumerate(axes_ref):
        A = np.asarray(A, dtype=np.float64)

        # prefer the mesh's own dominant flat region over the camera-relative
        # OBB-y proximity heuristic (see _robust_up_direction)
        up = None
        if meshes_ref is not None and i < len(meshes_ref):
            up = _dominant_face_direction(meshes_ref[i])
        if up is None:
            dots = np.abs(A.T @ y_axis)          # (3,)
            up_idx = int(np.argmax(dots))
            up = A[:, up_idx]

        # align sign to +y
        if np.dot(up, y_axis) < 0:
            up = -up

        up = up / (np.linalg.norm(up) + eps)
        up_axes.append(up)

        if weighted and meshes_ref is not None:
            # weight by mesh size (AABB volume as a proxy)
            ext = np.asarray(meshes_ref[i].bounding_box.extents, dtype=np.float64)
            w = float(np.prod(ext))
            weights.append(max(w, eps))
        else:
            weights.append(1.0)

    U = np.stack(up_axes, axis=0)          # (N,3)
    w = np.asarray(weights, dtype=np.float64)  # (N,)

    scene_up_axis = (U * w[:, None]).sum(axis=0) / w.sum()
    scene_up_axis = scene_up_axis / (np.linalg.norm(scene_up_axis) + eps)

    return scene_up_axis


def y_align(ori_results, object_names, coarse2fine=True, up_axis='y'):
    """
    Fix the meshes orientation to make them Y-up right using the predicted bounding boxes.
    """
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    meshes, meshes_ref, bboxes, axes, axes_ref = ori_results['meshes'], ori_results['meshes_ref'], ori_results['bboxes'], ori_results['axes'], ori_results['axes_ref']
    coarse_y_align_meshes = meshes.copy()
    coarse_y_align_bboxes = bboxes.copy()
    coarse_y_align_axes = axes.copy()
    if coarse2fine:
        # scene centroid from ref mesh vertices
        all_ref_vertices = np.vstack([m.vertices for m in meshes_ref])
        scene_centroid = all_ref_vertices.mean(axis=0)
        scene_up_axis = get_scene_up_axis_from_ref_axes(axes_ref, meshes_ref, y_axis)

        # rotation to y-up
        scene_rotation = rotation_matrix_from_vectors(scene_up_axis, y_axis)

        coarse_y_align_meshes, coarse_y_align_bboxes, coarse_y_align_axes = [], [], []
        # apply global rotation to all meshes, bboxes, axes
        for mesh, bbox, axis in zip(meshes, bboxes, axes):
            _mesh = mesh.copy()
            _mesh.vertices = (_mesh.vertices - scene_centroid) @ scene_rotation.T + scene_centroid
            _bbox = bbox.copy()
            _bbox["corners"] = (_bbox["corners"] - scene_centroid) @ scene_rotation.T + scene_centroid
            _axes = scene_rotation @ axis
            coarse_y_align_meshes.append(_mesh)
            coarse_y_align_bboxes.append(_bbox)
            coarse_y_align_axes.append(_axes)


    y_align_meshes, y_align_bboxes, y_align_axes = [], [], []
    for mesh, bbox, axes, objname in zip(coarse_y_align_meshes, coarse_y_align_bboxes, coarse_y_align_axes, object_names):
        if "pillow" in objname.lower():
            y_align_meshes.append(mesh)
            y_align_bboxes.append(bbox)
            y_align_axes.append(axes)
            continue
        y_align_mesh, y_align_bbox, y_align_axis = align_bbox_to_y_up_from_axes(
            mesh, bbox, axes
        )
        y_align_meshes.append(y_align_mesh)
        y_align_bboxes.append(y_align_bbox)
        y_align_axes.append(y_align_axis)

    y_align_results = {
        "meshes": y_align_meshes,
        "bboxes": y_align_bboxes,
        "axes": y_align_axes
    }
    return y_align_results



def read_obj_vertices(filepath):
    """Read all vertices from an OBJ file. Returns (vertices, raw_lines)."""
    vertices = []
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            lines.append(line)
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return vertices, lines


def get_y_bounds(vertices):
    """Return (min_y, max_y) over the given vertex list."""
    if not vertices:
        return 0, 0
    y_values = [v[1] for v in vertices]
    return min(y_values), max(y_values)


def write_obj_with_y_offset(filepath, lines, y_offset):
    """Re-emit an OBJ file with ``y_offset`` added to every vertex's Y. Other
    columns (vertex colors etc.) are preserved verbatim."""
    with open(filepath, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                extra = parts[4:]  # vertex colors (r g b) etc.
                if extra:
                    f.write(f"v {x} {y + y_offset} {z} {' '.join(extra)}\n")
                else:
                    f.write(f"v {x} {y + y_offset} {z}\n")
            else:
                f.write(line)


def load_trimesh_any(path):
    """Load any mesh file via trimesh, collapsing Scene into a single Trimesh."""
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        if len(m.geometry) == 0:
            raise ValueError(f"Empty scene: {path}")
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def compute_mesh_info(obj_path):
    """Return (centroid, half_extents, aabb_min, aabb_max) for a mesh file."""
    mesh = load_trimesh_any(obj_path)
    amin, amax = mesh.bounds
    centroid = (amin + amax) * 0.5
    half_ext = (amax - amin) * 0.5
    return (centroid.astype(np.float64), half_ext.astype(np.float64),
            amin.astype(np.float64), amax.astype(np.float64))

"""Stage-2 entry with the mix3r mesh backend (mix3r-direct branch, see
doc/2026-8-16.md).

Differences from the sam3d backend (scripts/infer_scene3d.py):

  - meshes come from cached mix3r multi-view runs (one normalized mesh per
    object) instead of single-view SAM3D;
  - metric scale and world pose come from the option-d solution
    (rest3d/utils/d_route.py over the wand calibration) instead of the
    SAM3D bbox/axes pipeline: up = support-plane normal, yaw anchored on
    the footprint long axis with a 4-candidate silhouette check (up sign x
    180 deg flip), scale s = h_real / h_mesh per object;
  - the support plane becomes y = 0 (canonical frame), so the scene is
    metric with true horizontal placement;
  - Y assembly and URDF generation reuse the existing code paths
    (place_to_ground, generate_urdf_files), so the output tree is
    drop-in compatible with Stage 3.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import cv2
import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rest3d.utils.calibration import load_wand_calibration  # noqa: E402
from rest3d.utils.log import attach_file_handler, get_logger  # noqa: E402
from rest3d.utils.postprocess import place_to_ground  # noqa: E402
from rest3d.utils.urdf import generate_urdf_files  # noqa: E402

logger = get_logger("stage2")

DEFAULTS = {
    "dataset": "data/0810/2026-08-10_14-32-37",
    "calibration": "data/0810/calibration_results-20-0-21.json",
    "d_solution": "output/mix3r_direct/2026-08-16_v1/d_solution.json",
    "scene_tree": "output/mix3r_direct/2026-08-16_v1/stage1/scene_tree_filtered.json",
    "mix3r_run": "output/mix3r_new/2026-08-13_143237_mix3r_views49_80_v1",
    "output_folder": "output/mix3r_direct",
    "output_name": "2026-08-16_v1",
    "silhouette_views": "cam_77,cam_76,cam_74",
    "silhouette_downsample": 4,
}


def rotation_aligning(source, target):
    """Minimal rotation taking unit vector ``source`` to ``target``."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(source @ target)
    if cosine < -1.0 + 1e-9:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + K + (K @ K) / (1.0 + cosine)


def largest_component(mesh):
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1
    )
    biggest = max(components, key=len)
    if len(components) == 1:
        return mesh, 1.0
    kept = mesh.submesh([biggest], append=True, repair=False)
    return kept, len(biggest) / len(mesh.faces)


def silhouette_iou(vertices_world, camera, mask_path, downsample):
    stride = max(1, len(vertices_world) // 4000)
    pixels = camera.project_world(vertices_world[::stride])
    points = pixels.reshape(-1, 1, 2).astype(np.float32)
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points)
    height = camera.height // downsample
    width = camera.width // downsample
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, [(hull.astype(np.float32) / downsample).astype(np.int32)], 1)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 128
    mask_small = mask[::downsample, ::downsample]
    intersection = np.logical_and(canvas > 0, mask_small).sum()
    union = np.logical_or(canvas > 0, mask_small).sum()
    return float(intersection) / max(int(union), 1)


def mask_centroid_pixels(mask_dir, view):
    mask = np.asarray(Image.open(os.path.join(mask_dir, f"{view}.png")).convert("L")) > 128
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def refine_inplane_offset(world, long_axis, normal, cameras, views, mask_dir):
    """Least-squares in-plane translation aligning the projected mesh
    centroid with the mask centroid across the silhouette views. The
    contact point is only a median of bottom-ray hits; this trims the
    residual horizontal bias before the scene is written."""
    e2 = np.cross(normal, long_axis)
    basis = np.stack([long_axis, e2], axis=1)

    stride = max(1, len(world) // 2000)
    targets = np.concatenate(
        [mask_centroid_pixels(mask_dir, view) for view in views]
    )

    def residuals(delta):
        moved = world[::stride] + basis @ delta
        projected = []
        for view in views:
            pixels = cameras[view].project_world(moved)
            projected.append(pixels.mean(axis=0))
        return np.concatenate(projected) - targets

    r0 = residuals(np.zeros(2))
    jacobian = np.zeros((len(r0), 2))
    for axis in range(2):
        step = np.zeros(2)
        step[axis] = 0.01
        jacobian[:, axis] = (residuals(step) - r0) / 0.01
    delta, *_ = np.linalg.lstsq(jacobian, -r0, rcond=None)
    norm = float(np.linalg.norm(delta))
    if norm > 0.08:  # keep the refinement local; >8 cm indicates a deeper issue
        delta *= 0.08 / norm
    return world + (basis @ delta), delta.tolist(), float(np.linalg.norm(residuals(delta)))


def solve_object_pose(mesh, item, plane_point, plane_normal, cameras, views, mask_dir, downsample):
    """Pick orientation (up sign x 180 deg flip) by silhouette IoU, return
    the world-space (wand frame) vertices and the scale."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centered = vertices - vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    h1, h2, up_object = vt[0], vt[1], vt[2]

    normal = plane_normal / np.linalg.norm(plane_normal)
    long_axis = np.asarray(item["footprint_long_axis"], dtype=np.float64)
    long_axis -= (long_axis @ normal) * normal
    long_axis /= np.linalg.norm(long_axis)
    contact = np.asarray(item["contact"], dtype=np.float64)

    candidates = []
    for up_sign in (1.0, -1.0):
        for long_sign in (1.0, -1.0):
            column_long = long_sign * long_axis
            column_up = up_sign * normal
            column_mid = np.cross(column_up, column_long)
            column_mid /= np.linalg.norm(column_mid)
            E = np.stack([column_long, column_mid, column_up], axis=1)
            M = np.stack([h1, h2, up_object])
            rotation = E @ M.T

            along_up = vertices @ (rotation.T @ normal)
            h_mesh = float(along_up.max() - along_up.min())
            scale = float(item["h_real_m"]) / h_mesh
            rotated = (rotation @ (scale * vertices).T).T
            # Anchor the object standing on the plane at the contact: the
            # bottom vertex (0.1 percentile, robust to stragglers) is placed
            # exactly on the contact point. mix3r meshes are object-centred,
            # so the mean vertex carries no height information.
            along_normal = rotated @ normal
            bottom_offset = contact - rotated[
                np.argmin(np.abs(along_normal - np.percentile(along_normal, 0.1)))
            ]
            world = rotated + bottom_offset

            ious = []
            for view in views:
                camera = cameras[view]
                mask_path = os.path.join(mask_dir, f"{view}.png")
                ious.append(silhouette_iou(world, camera, mask_path, downsample))
            candidates.append(
                {
                    "up_sign": up_sign,
                    "long_sign": long_sign,
                    "scale": scale,
                    "h_mesh": h_mesh,
                    "world": world,
                    "ious": ious,
                    "score": float(np.mean(ious)),
                }
            )
    best = max(candidates, key=lambda c: c["score"])
    best["world"], refinement_delta, refinement_residual = refine_inplane_offset(
        best["world"], long_axis, normal, cameras, views, mask_dir
    )
    best["ious"] = [
        silhouette_iou(best["world"], cameras[view], os.path.join(mask_dir, f"{view}.png"), downsample)
        for view in views
    ]
    best["score"] = float(np.mean(best["ious"]))
    best["refinement_delta"] = refinement_delta
    best["refinement_residual_px"] = refinement_residual
    return best, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--calibration", default=DEFAULTS["calibration"])
    parser.add_argument("--d_solution", default=DEFAULTS["d_solution"])
    parser.add_argument("--scene_tree", default=DEFAULTS["scene_tree"])
    parser.add_argument("--mix3r_run", default=DEFAULTS["mix3r_run"],
                        help="mix3r run directory with out_<object>/sample_mesh_perviewbias.ply")
    parser.add_argument("--output_folder", default=DEFAULTS["output_folder"])
    parser.add_argument("--output_name", default=DEFAULTS["output_name"])
    parser.add_argument("--silhouette_views", default=DEFAULTS["silhouette_views"])
    parser.add_argument("--silhouette_downsample", type=int, default=DEFAULTS["silhouette_downsample"])
    args = parser.parse_args()

    with open(args.d_solution) as f:
        solution = json.load(f)
    plane_point = np.asarray(solution["plane_point"], dtype=np.float64)
    plane_normal = np.asarray(solution["plane_normal"], dtype=np.float64)
    plane_normal /= np.linalg.norm(plane_normal)

    with open(args.scene_tree) as f:
        tree = json.load(f)
    object_names = tree["nodes"]

    cameras = load_wand_calibration(args.calibration)
    views = [v.strip() for v in args.silhouette_views.split(",") if v.strip()]
    masks_root = os.path.join(args.dataset, "masks")

    output_dir = os.path.join(args.output_folder, args.output_name, "stage2")
    raw_dir = os.path.join(output_dir, "scene_raw", "obj_files")
    y_align_dir = os.path.join(output_dir, "scene_y_align", "obj_files")
    canon_dir = os.path.join(output_dir, "scene_canon", "obj_files")
    urdf_dir = os.path.join(output_dir, "scene_canon", "urdf_files")
    for d in (raw_dir, y_align_dir, canon_dir, urdf_dir):
        os.makedirs(d, exist_ok=True)

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_handler = attach_file_handler(logger, os.path.join(output_dir, f"stage2_log_{_ts}.txt"))
    logger.info(f"mix3r-direct stage2: {len(object_names)} objects, dataset={args.dataset}")

    # Canonical frame: support plane -> y = 0, plane normal -> +Y.
    plane_to_y = rotation_aligning(plane_normal, np.array([0.0, 1.0, 0.0]))

    metrics = {"plane_point": plane_point.tolist(), "plane_normal": plane_normal.tolist(), "objects": {}}
    for object_id in object_names:
        item = solution["objects"][object_id]
        mesh_path = os.path.join(args.mix3r_run, f"out_{object_id}", "sample_mesh_perviewbias.ply")
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"mix3r mesh not found: {mesh_path}")
        mesh = trimesh.load(mesh_path, process=False)
        mesh, kept_fraction = largest_component(mesh)
        mesh.export(os.path.join(raw_dir, f"{object_id}.obj"))

        best, candidates = solve_object_pose(
            mesh, item, plane_point, plane_normal, cameras, views,
            os.path.join(masks_root, object_id), args.silhouette_downsample,
        )
        canonical = (best["world"] - plane_point) @ plane_to_y.T
        out_mesh = trimesh.Trimesh(
            vertices=canonical,
            faces=mesh.faces,
            vertex_colors=mesh.visual.vertex_colors,
            process=False,
        )
        out_mesh.export(os.path.join(y_align_dir, f"scene_y_align_{object_id}.obj"))

        metrics["objects"][object_id] = {
            "scale": round(best["scale"], 5),
            "h_mesh": round(best["h_mesh"], 5),
            "h_real_m": item["h_real_m"],
            "up_sign": best["up_sign"],
            "long_sign": best["long_sign"],
            "silhouette_ious": [round(v, 3) for v in best["ious"]],
            "refinement_delta_m": [round(v, 4) for v in best["refinement_delta"]],
            "refinement_residual_px": round(best["refinement_residual_px"], 1),
            "candidate_scores": [round(c["score"], 3) for c in candidates],
            "largest_component_fraction": round(kept_fraction, 6),
        }
        logger.info(
            f"{object_id}: s={best['scale']:.4f} h_mesh={best['h_mesh']:.3f} "
            f"h_real={item['h_real_m']:.3f} IoU={np.round(best['ious'], 3).tolist()} "
            f"(candidates {np.round([c['score'] for c in candidates], 3).tolist()})"
        )
        if best["score"] < 0.3:
            logger.warning(f"{object_id}: best silhouette IoU {best['score']:.3f} is low; orientation uncertain")

    with open(os.path.join(output_dir, "mix3r_direct_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    tree_dst = os.path.join(output_dir, "scene_tree.json")
    shutil.copy(args.scene_tree, tree_dst)
    place_to_ground(y_align_dir, canon_dir, tree_dst)
    generate_urdf_files(object_names, urdf_dir, prefix="scene_canon_")

    logger.info(f"stage2 mix3r-direct complete: {output_dir}")
    logger.removeHandler(log_handler)
    log_handler.close()


if __name__ == "__main__":
    main()

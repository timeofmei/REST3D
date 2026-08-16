"""Option-d metric solver (mix3r-direct): support plane, per-object contact
points, metric heights and yaw directions from calibrated multi-view masks.

All geometry is metric (metres) in the wand-calibration world frame. The
scale/pose consumers (phase C) receive, per object:

  - ``contact``   support-plane contact point (median of per-view
                  ray/plane intersections);
  - ``h_real``    metric height, median over views of the "mask top ray
                  crossed with the vertical line through the contact"
                  measurement;
  - ``yaw_dir``   horizontal direction from contact to the triangulated
                  mask-centroid point, projected onto the support plane.

Known failure modes (inherited from doc/2026-8-11.md): the top-edge
measurement overestimates flat-topped objects seen at grazing angles, and
occluded/nested contact points bias the plane fit; per-view spread is
reported so both are visible.
"""

import numpy as np


def mask_bottom_pixel(mask):
    """Boolean mask -> representative bottom-centre pixel (x, y).

    Uses the bottom ~1% band of the mask (min 2 px) so a single noisy row
    cannot dominate.
    """
    ys, xs = np.nonzero(mask)
    band = max(2, int(round(0.01 * (ys.max() - ys.min()))))
    selected = ys >= ys.max() - band
    return float(np.median(xs[selected])), float(np.median(ys[selected]))


def mask_top_pixel(mask):
    ys, xs = np.nonzero(mask)
    band = max(2, int(round(0.01 * (ys.max() - ys.min()))))
    selected = ys <= ys.min() + band
    return float(np.median(xs[selected])), float(np.median(ys[selected]))


def mask_centroid_pixel(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def triangulate_rays(origins, directions):
    """Least-squares point minimizing squared perpendicular ray distances."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for origin, direction in zip(origins, directions):
        projector = np.eye(3) - np.outer(direction, direction)
        A += projector
        b += projector @ origin
    return np.linalg.solve(A, b)


def fit_plane(points):
    """SVD plane fit -> (point_on_plane, unit_normal, rms_residual)."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    distances = (points - centroid) @ normal
    rms = float(np.sqrt(np.mean(distances**2)))
    return centroid, normal, rms


def ray_plane_intersections(origins, directions, plane_point, plane_normal):
    """Batched ray/plane intersections; returns NaN for parallel/behind rays."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    denominators = directions @ plane_normal
    t = ((plane_point - origins) @ plane_normal) / denominators
    t[np.abs(denominators) < 1e-12] = np.nan
    t[t < 0] = np.nan
    return origins + t[:, None] * directions


def ray_vertical_line_height(origin, direction, contact, up):
    """Height above the plane where the ray crosses the vertical line
    through ``contact`` along ``up`` (least squares over the 3 equations)."""
    A = np.stack([direction, -up], axis=1)
    solution, *_ = np.linalg.lstsq(A, contact - origin, rcond=None)
    return float(solution[1])


def solve_scene(views, cameras):
    """Solve support plane + per-object D quantities.

    ``views``: {object_id: [(camera_id, bottom_xy, top_xy, centroid_xy), ...]}
    ``cameras``: {camera_id: WandCamera}

    Height estimator: the top-edge rays of a box-like object pass through
    different (near/far) edges whose heights are equal, so the ray-bundle
    triangulation point sits at the common top height; ``h_real`` is its
    signed distance to the support plane. Per-view sensitivity comes from
    leave-one-out re-triangulation (jackknife).

    Contact: median of per-view bottom-ray/plane intersections. Different
    views see different footprint edges, so the intersection scatter is the
    object footprint size, not noise — reported as ``footprint_radius_m``.

    Plane stability: bottom intersections are split even/odd by view index,
    each half refits a plane; the normal angle and offset between halves is
    the honest consistency check (intersections with one plane are coplanar
    by construction, so a single-plane RMS would always be zero).
    """
    rays = {}
    for object_id, entries in views.items():
        per_view = []
        for camera_id, bottom_xy, top_xy, centroid_xy in entries:
            camera = cameras[camera_id]
            bottom = camera.rays_world(np.array([bottom_xy]))
            top = camera.rays_world(np.array([top_xy]))
            centroid = camera.rays_world(np.array([centroid_xy]))
            per_view.append(
                {
                    "camera_id": camera_id,
                    "bottom": (bottom[0][0], bottom[1][0]),
                    "top": (top[0][0], top[1][0]),
                    "centroid": (centroid[0][0], centroid[1][0]),
                }
            )
        rays[object_id] = per_view

    # Rough contact points by ray-bundle triangulation (no plane needed).
    rough_contacts = {
        object_id: triangulate_rays(
            [v["bottom"][0] for v in per_view],
            [v["bottom"][1] for v in per_view],
        )
        for object_id, per_view in rays.items()
    }

    def refit_plane(view_selector):
        # Initialize from this subset's own triangulated contacts so split
        # fits are independent (a shared initialization would bias both
        # halves toward the same fixed point and fake stability).
        subset_contacts = []
        for per_view in rays.values():
            chosen = [v for v in per_view if view_selector(v)]
            subset_contacts.append(
                triangulate_rays(
                    [v["bottom"][0] for v in chosen],
                    [v["bottom"][1] for v in chosen],
                )
            )
        point, normal, _ = fit_plane(subset_contacts)
        for _ in range(3):
            selected = []
            for per_view in rays.values():
                chosen = [v for v in per_view if view_selector(v)]
                origins = np.array([v["bottom"][0] for v in chosen])
                directions = np.array([v["bottom"][1] for v in chosen])
                intersections = ray_plane_intersections(
                    origins, directions, point, normal
                )
                selected.append(intersections)
            point, normal, _ = fit_plane(
                np.concatenate([p[~np.isnan(p).any(axis=1)] for p in selected])
            )
        return point, normal

    plane_point, plane_normal = refit_plane(lambda v: True)

    # Orient "up" toward the cameras (they look down onto the support plane).
    camera_mean = np.mean([c.center_world for c in cameras.values()], axis=0)
    if plane_normal @ (camera_mean - plane_point) < 0:
        plane_normal = -plane_normal

    # Honest plane stability: even/odd view split.
    even_point, even_normal = refit_plane(lambda v: int(v["camera_id"][4:]) % 2 == 0)
    if even_normal @ plane_normal < 0:
        even_normal = -even_normal
    plane_half_angle_deg = float(
        np.degrees(np.arccos(abs(plane_normal @ even_normal)))
    )
    plane_half_offset_m = float(
        np.abs((even_point - plane_point) @ plane_normal)
    )

    results = {
        "plane_point": plane_point,
        "plane_normal": plane_normal,
        "plane_half_angle_deg": plane_half_angle_deg,
        "plane_half_offset_m": plane_half_offset_m,
        "objects": {},
    }
    for object_id, per_view in rays.items():
        origins = np.array([v["bottom"][0] for v in per_view])
        directions = np.array([v["bottom"][1] for v in per_view])
        contacts = ray_plane_intersections(
            origins, directions, plane_point, plane_normal
        )
        valid = ~np.isnan(contacts).any(axis=1)
        contact_median = np.median(contacts[valid], axis=0)
        distances = np.linalg.norm(contacts[valid] - contact_median, axis=1)
        footprint_radius = float(np.percentile(distances, 90))

        # In-plane footprint principal axes (per-view bottom intersections
        # sweep the footprint outline). The long axis anchors the mesh yaw.
        in_plane = contacts[valid] - plane_point
        horizontal = in_plane - np.outer(in_plane @ plane_normal, plane_normal)
        _, _, footprint_vt = np.linalg.svd(horizontal, full_matrices=False)
        long_axis = footprint_vt[0]
        long_axis -= (long_axis @ plane_normal) * plane_normal
        long_axis /= np.linalg.norm(long_axis)

        top_origins = [v["top"][0] for v in per_view]
        top_directions = [v["top"][1] for v in per_view]
        top_point = triangulate_rays(top_origins, top_directions)
        h_real = float((top_point - plane_point) @ plane_normal)
        h_jackknife = []
        for index in range(len(per_view)):
            remaining_o = [o for i, o in enumerate(top_origins) if i != index]
            remaining_d = [d for i, d in enumerate(top_directions) if i != index]
            if len(remaining_o) < 3:
                continue
            partial = triangulate_rays(remaining_o, remaining_d)
            h_jackknife.append(float((partial - plane_point) @ plane_normal))

        centroid_point = triangulate_rays(
            [v["centroid"][0] for v in per_view],
            [v["centroid"][1] for v in per_view],
        )
        yaw = centroid_point - contact_median
        yaw = yaw - (yaw @ plane_normal) * plane_normal
        yaw_norm = float(np.linalg.norm(yaw))
        yaw_dir = yaw / yaw_norm if yaw_norm > 1e-9 else np.zeros(3)

        results["objects"][object_id] = {
            "views": [v["camera_id"] for v in per_view],
            "n_views": len(per_view),
            "contact": contact_median,
            "footprint_radius_m": footprint_radius,
            "footprint_long_axis": long_axis,
            "h_real_m": h_real,
            "h_jackknife_min_m": float(min(h_jackknife)),
            "h_jackknife_max_m": float(max(h_jackknife)),
            "centroid": centroid_point,
            "yaw_dir": yaw_dir,
        }
    return results

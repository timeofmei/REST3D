"""Wand-calibration loader for the 0810 capture rig.

Parses the capture system's calibration JSON (one entry per camera with
fx/fy/cx/cy, radial/tangential distortion k1..k4/p1/p2, and a world2Cam
pose as quaternion + translation) into per-camera metric objects.

Unit convention: the file stores translations in millimetres; this module
converts to metres at load time and every value it returns (camera centers,
rays, triangulated points) is metric. Pixel intrinsics are in pixels and
unchanged.
"""

import json

import cv2
import numpy as np


class WandCamera:
    """One calibrated camera with metric world pose and ray helpers."""

    def __init__(self, camera_id, width, height, K, dist, q_wxyz, t_world2cam_m):
        self.id = camera_id
        self.width = width
        self.height = height
        self.K = K
        # OpenCV plumb-line order: [k1, k2, p1, p2, k3, k4]; cv2 requires the
        # vector length to be one of 4/5/8/12/14, so pad to 8 with zeros.
        dist = np.asarray(dist, dtype=np.float64).ravel()
        self.dist = np.concatenate([dist, np.zeros(max(0, 8 - len(dist)))])
        self.R_world2cam = quaternion_to_matrix(q_wxyz)
        self.t_world2cam = t_world2cam_m
        self.R_cam2world = self.R_world2cam.T
        self.center_world = -self.R_cam2world @ self.t_world2cam

    def undistort_pixels(self, pixels):
        """Pixels (N,2) -> ideal normalized camera coords (N,2)."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(pixels, self.K, self.dist)
        return normalized.reshape(-1, 2)

    def rays_world(self, pixels):
        """Pixels (N,2) -> (origins (N,3), unit directions (N,3)) in world frame."""
        normalized = self.undistort_pixels(pixels)
        dirs_cam = np.hstack([normalized, np.ones((len(normalized), 1))])
        dirs_world = dirs_cam @ self.R_cam2world.T
        dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)
        origins = np.broadcast_to(self.center_world, dirs_world.shape)
        return origins, dirs_world

    def project_world(self, points_world):
        """World points (N,3) -> distorted pixels (N,2)."""
        points_cam = (
            np.asarray(points_world, dtype=np.float64) @ self.R_world2cam.T
            + self.t_world2cam
        )
        projected, _ = cv2.projectPoints(
            points_cam.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            self.K,
            self.dist,
        )
        return projected.reshape(-1, 2)


def quaternion_to_matrix(q_wxyz):
    """Hamilton quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = [float(v) for v in q_wxyz]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_wand_calibration(path):
    """Load the capture calibration JSON -> {camera_id: WandCamera} (metric)."""
    with open(path) as stream:
        data = json.load(stream)

    cameras = {}
    for entry in data["calibrations"]:
        K = np.array(
            [
                [entry["fx"], 0.0, entry["cx"]],
                [0.0, entry["fy"], entry["cy"]],
                [0.0, 0.0, 1.0],
            ]
        )
        dist = np.array(
            [
                entry["k1"],
                entry["k2"],
                entry["p1"],
                entry["p2"],
                entry["k3"],
                entry["k4"],
            ]
        )
        pose = entry["world2Cam"]
        q = (pose["qw"], pose["qx"], pose["qy"], pose["qz"])
        # File translations are millimetres; convert once, here.
        t_m = 1e-3 * np.array([pose["x"], pose["y"], pose["z"]])
        cameras[entry["cameraSN"]] = WandCamera(
            camera_id=entry["cameraSN"],
            width=entry["imageSize"]["w"],
            height=entry["imageSize"]["h"],
            K=K,
            dist=dist,
            q_wxyz=q,
            t_world2cam_m=t_m,
        )
    return cameras

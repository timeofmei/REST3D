"""Backend-neutral pose batching and energy terms for local-group CEM."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalCEMEnergyWeights:
    pose_stability: float = 1.0
    rotation_stability: float = 1.0
    pose_layout: float = 6.0
    rotation_layout: float = 1.0
    velocity: float = 1.0
    placement_penetration: float = 0.5
    settled_penetration: float = 0.5

    @classmethod
    def from_config(cls, config):
        return cls(
            pose_stability=float(config.lambda_pose_stab),
            rotation_stability=float(config.lambda_rot_stab),
            pose_layout=float(config.lambda_pose_layout),
            rotation_layout=float(config.lambda_rot_layout),
            velocity=float(config.lambda_vel),
            placement_penetration=float(config.lambda_place_geo_pen),
            settled_penetration=float(config.lambda_settled_geo_pen),
        )


def normalize_quaternion_wxyz(quaternion):
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion final dimension must be 4")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool(torch.any(norm < 1.0e-12)):
        raise ValueError("zero-length quaternion cannot be normalized")
    return quaternion / norm


def quaternion_multiply_wxyz(left, right):
    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("quaternion operands must have matching [..., 4] shapes")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_conjugate_wxyz(quaternion):
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion final dimension must be 4")
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def axis_angle_to_quaternion_wxyz(axis_angle):
    if axis_angle.shape[-1] != 3:
        raise ValueError("axis-angle final dimension must be 3")
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    small = angle < 1.0e-6
    scale = torch.where(
        small,
        0.5 - angle.square() / 48.0,
        torch.sin(half_angle) / torch.clamp_min(angle, 1.0e-12),
    )
    return normalize_quaternion_wxyz(
        torch.cat((torch.cos(half_angle), axis_angle * scale), dim=-1)
    )


def quaternion_geodesic_distance_wxyz(reference, candidate):
    if reference.shape != candidate.shape or reference.shape[-1] != 4:
        raise ValueError("quaternion arrays must have matching [..., 4] shapes")
    reference = normalize_quaternion_wxyz(reference)
    candidate = normalize_quaternion_wxyz(candidate)
    alignment = torch.sum(reference * candidate, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(alignment)


def quaternion_rotate_wxyz(quaternion, vector):
    if quaternion.shape[:-1] != vector.shape[:-1] or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector batch shapes must match")
    quaternion = normalize_quaternion_wxyz(quaternion)
    scalar = quaternion[..., :1]
    imaginary = quaternion[..., 1:]
    first_cross = torch.linalg.cross(imaginary, vector, dim=-1)
    return vector + 2.0 * (
        scalar * first_cross
        + torch.linalg.cross(imaginary, first_cross, dim=-1)
    )


def apply_pose_deltas_wxyz(reference_poses, samples):
    """Apply `[dx,dy,dz, axis_angle_xyz]` samples to WXYZ poses."""

    if reference_poses.ndim != 2 or reference_poses.shape[-1] != 7:
        raise ValueError("reference_poses must have shape [objects, 7]")
    if samples.ndim != 3 or samples.shape[1:] != (reference_poses.shape[0], 6):
        raise ValueError(
            "samples must have shape [environments, objects, 6] matching references"
        )
    if samples.device != reference_poses.device:
        raise ValueError("samples and reference poses must share a device")
    reference = reference_poses.unsqueeze(0).expand(samples.shape[0], -1, -1)
    position = reference[..., :3] + samples[..., :3]
    delta_quaternion = axis_angle_to_quaternion_wxyz(samples[..., 3:6])
    orientation = normalize_quaternion_wxyz(
        quaternion_multiply_wxyz(reference[..., 3:7], delta_quaternion)
    )
    return torch.cat((position, orientation), dim=-1)


def apply_pose_deltas_about_centroids_wxyz(
    reference_root_poses, centroid_offsets, samples
):
    """Apply pose deltas while keeping each mesh centroid as its rotation pivot."""

    if reference_root_poses.ndim != 2 or reference_root_poses.shape[-1] != 7:
        raise ValueError("reference_root_poses must have shape [objects, 7]")
    if centroid_offsets.shape != (reference_root_poses.shape[0], 3):
        raise ValueError("centroid_offsets must have shape [objects, 3]")
    if not (
        reference_root_poses.device == centroid_offsets.device == samples.device
    ):
        raise ValueError("reference poses, centroids, and samples must share a device")
    updated = apply_pose_deltas_wxyz(reference_root_poses, samples)
    reference = reference_root_poses.unsqueeze(0).expand(samples.shape[0], -1, -1)
    offsets = centroid_offsets.unsqueeze(0).expand(samples.shape[0], -1, -1)
    reference_centroids = reference[..., :3] + quaternion_rotate_wxyz(
        reference[..., 3:7], offsets
    )
    updated_centroids = reference_centroids + samples[..., :3]
    root_positions = updated_centroids - quaternion_rotate_wxyz(
        updated[..., 3:7], offsets
    )
    return torch.cat((root_positions, updated[..., 3:7]), dim=-1)


def apply_group_member_pose_deltas_wxyz(
    reference_member_poses,
    centroid_offsets,
    samples,
    sampled_member_indices,
    owner_sample_indices,
):
    """Move sampled children and all of their descendants as rigid subtrees."""

    member_count = reference_member_poses.shape[0]
    if reference_member_poses.shape != (member_count, 7):
        raise ValueError("reference_member_poses must have shape [members, 7]")
    if centroid_offsets.shape != (member_count, 3):
        raise ValueError("centroid_offsets must have shape [members, 3]")
    if samples.ndim != 3 or samples.shape[-1] != 6:
        raise ValueError("samples must have shape [environments, sampled_children, 6]")
    if sampled_member_indices.shape != (samples.shape[1],):
        raise ValueError("sampled_member_indices must identify every sampled child")
    if owner_sample_indices.shape != (member_count,):
        raise ValueError("owner_sample_indices must identify every member's sampled owner")
    if not (
        reference_member_poses.device
        == centroid_offsets.device
        == samples.device
        == sampled_member_indices.device
        == owner_sample_indices.device
    ):
        raise ValueError("all group pose tensors must share a device")
    owner_values = owner_sample_indices.detach().cpu().tolist()
    if any(owner < -1 or owner >= samples.shape[1] for owner in owner_values):
        raise ValueError("owner sample indices are out of range")

    owner_reference = reference_member_poses[sampled_member_indices]
    owner_centroids = centroid_offsets[sampled_member_indices]
    owner_updated = apply_pose_deltas_about_centroids_wxyz(
        owner_reference, owner_centroids, samples
    )
    reference_owner_quaternion = owner_reference[:, 3:7].unsqueeze(0).expand(
        samples.shape[0], -1, -1
    )
    world_delta_quaternion = normalize_quaternion_wxyz(
        quaternion_multiply_wxyz(
            owner_updated[..., 3:7],
            quaternion_conjugate_wxyz(reference_owner_quaternion),
        )
    )
    owner_reference_centroid = owner_reference[:, :3] + quaternion_rotate_wxyz(
        owner_reference[:, 3:7], owner_centroids
    )
    owner_updated_centroid = owner_reference_centroid.unsqueeze(0) + samples[..., :3]

    result = reference_member_poses.unsqueeze(0).expand(
        samples.shape[0], -1, -1
    ).clone()
    for member_index, owner_index in enumerate(owner_values):
        if owner_index < 0:
            continue
        delta_quaternion = world_delta_quaternion[:, owner_index]
        reference_position = reference_member_poses[member_index, :3].unsqueeze(0).expand(
            samples.shape[0], -1
        )
        relative_position = (
            reference_position
            - owner_reference_centroid[owner_index].unsqueeze(0)
        )
        result[:, member_index, :3] = quaternion_rotate_wxyz(
            delta_quaternion, relative_position
        ) + owner_updated_centroid[:, owner_index]
        reference_quaternion = reference_member_poses[
            member_index, 3:7
        ].unsqueeze(0).expand(samples.shape[0], -1)
        result[:, member_index, 3:7] = normalize_quaternion_wxyz(
            quaternion_multiply_wxyz(delta_quaternion, reference_quaternion)
        )
    return result


def _validate_state_batches(placed, early, settled, reference_poses):
    expected = placed.shape
    if placed.ndim != 3 or placed.shape[-1] != 13:
        raise ValueError("placed states must have shape [environments, objects, 13]")
    if early.shape != expected or settled.shape != expected:
        raise ValueError("placed, early, and settled states must have matching shapes")
    if reference_poses.shape != (placed.shape[1], 7):
        raise ValueError("reference poses must have shape [objects, 7]")
    if not (placed.device == early.device == settled.device == reference_poses.device):
        raise ValueError("all energy tensors must share a device")
    if not bool(
        torch.isfinite(placed).all()
        and torch.isfinite(early).all()
        and torch.isfinite(settled).all()
        and torch.isfinite(reference_poses).all()
    ):
        raise ValueError("energy inputs must be finite")


def evaluate_local_cem_energy(
    placed_states,
    early_states,
    settled_states,
    reference_poses,
    *,
    centroid_offsets=None,
    placement_penetration=None,
    settled_penetration=None,
    weights=None,
):
    """Compute the existing local CEM energy terms for every environment."""

    _validate_state_batches(placed_states, early_states, settled_states, reference_poses)
    weights = LocalCEMEnergyWeights() if weights is None else weights
    environment_count, object_count = placed_states.shape[:2]
    device = placed_states.device
    dtype = placed_states.dtype
    if centroid_offsets is None:
        centroid_offsets = torch.zeros(object_count, 3, device=device, dtype=dtype)
    if centroid_offsets.shape != (object_count, 3) or centroid_offsets.device != device:
        raise ValueError("centroid_offsets must have shape [objects, 3] on the state device")

    reference = reference_poses.unsqueeze(0).expand(environment_count, -1, -1)
    offset = centroid_offsets.unsqueeze(0).expand(environment_count, -1, -1)
    placed_points = placed_states[..., :3] + quaternion_rotate_wxyz(
        placed_states[..., 3:7], offset
    )
    settled_points = settled_states[..., :3] + quaternion_rotate_wxyz(
        settled_states[..., 3:7], offset
    )
    reference_points = reference[..., :3] + quaternion_rotate_wxyz(
        reference[..., 3:7], offset
    )

    components = {
        "pose_stability": torch.linalg.vector_norm(
            settled_points - placed_points, dim=-1
        ).sum(dim=1),
        "rotation_stability": quaternion_geodesic_distance_wxyz(
            settled_states[..., 3:7], placed_states[..., 3:7]
        ).sum(dim=1),
        "pose_layout": torch.linalg.vector_norm(
            settled_points - reference_points, dim=-1
        ).sum(dim=1),
        "rotation_layout": quaternion_geodesic_distance_wxyz(
            settled_states[..., 3:7], reference[..., 3:7]
        ).sum(dim=1),
        "velocity": torch.linalg.vector_norm(
            early_states[..., 7:10], dim=-1
        ).sum(dim=1),
    }
    for name, value in (
        ("placement_penetration", placement_penetration),
        ("settled_penetration", settled_penetration),
    ):
        if value is None:
            value = torch.zeros(environment_count, device=device, dtype=dtype)
        if value.shape != (environment_count,) or value.device != device:
            raise ValueError(f"{name} must have shape [environments] on the state device")
        components[name] = value

    energy = (
        weights.pose_stability * components["pose_stability"]
        + weights.rotation_stability * components["rotation_stability"]
        + weights.pose_layout * components["pose_layout"]
        + weights.rotation_layout * components["rotation_layout"]
        + weights.velocity * components["velocity"]
        + weights.placement_penetration * components["placement_penetration"]
        + weights.settled_penetration * components["settled_penetration"]
    )
    components["energy"] = energy
    components["reward"] = -energy
    return components

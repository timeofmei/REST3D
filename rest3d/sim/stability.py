"""Backend-neutral rigid-body stability metrics for replay state sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityMetrics:
    """Per-object motion metrics evaluated against one reference frame."""

    evaluation_step: int
    position_threshold_m: float
    rotation_threshold_rad: float
    displacement_at_evaluation_m: np.ndarray
    rotation_at_evaluation_rad: np.ndarray
    final_displacement_m: np.ndarray
    final_rotation_rad: np.ndarray
    maximum_excursion_m: np.ndarray
    maximum_rotation_excursion_rad: np.ndarray
    early_max_linear_speed_m_s: np.ndarray
    early_max_angular_speed_rad_s: np.ndarray
    terminal_max_linear_speed_m_s: np.ndarray
    terminal_max_angular_speed_rad_s: np.ndarray
    terminal_mean_linear_speed_m_s: np.ndarray
    terminal_mean_angular_speed_rad_s: np.ndarray
    stable: np.ndarray

    @property
    def scene_stable(self) -> bool:
        return bool(np.all(self.stable))


def quaternion_geodesic_distance_wxyz(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Return the shortest SO(3) angle between WXYZ quaternions in radians."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.shape[-1] != 4:
        raise ValueError(
            "quaternion arrays must have matching shapes with final dimension 4, "
            f"got {reference.shape} and {candidate.shape}"
        )
    reference_norm = np.linalg.norm(reference, axis=-1, keepdims=True)
    candidate_norm = np.linalg.norm(candidate, axis=-1, keepdims=True)
    if np.any(reference_norm < 1.0e-12) or np.any(candidate_norm < 1.0e-12):
        raise ValueError("zero-length quaternion cannot be compared")
    alignment = np.sum(
        (reference / reference_norm) * (candidate / candidate_norm), axis=-1
    )
    return 2.0 * np.arccos(np.clip(np.abs(alignment), 0.0, 1.0))


def evaluate_replay_stability(
    states_rest: np.ndarray,
    *,
    evaluation_step: int = 60,
    position_threshold_m: float = 0.1,
    rotation_threshold_rad: float = 0.1,
    early_window_steps: int = 10,
    terminal_window_steps: int = 10,
) -> StabilityMetrics:
    """Evaluate every object against frame zero using the paper's motion rule."""

    states = np.asarray(states_rest, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] != 13:
        raise ValueError(f"expected states shaped [frames, objects, 13], got {states.shape}")
    if states.shape[1] < 1:
        raise ValueError("at least one object is required")
    if not np.isfinite(states).all():
        raise ValueError("states contain non-finite values")
    if evaluation_step < 1 or evaluation_step >= states.shape[0]:
        raise ValueError(
            f"evaluation_step must be in [1, {states.shape[0] - 1}], got {evaluation_step}"
        )
    if position_threshold_m <= 0.0 or rotation_threshold_rad <= 0.0:
        raise ValueError("stability thresholds must be positive")
    if early_window_steps < 1 or terminal_window_steps < 1:
        raise ValueError("velocity windows must be positive")

    reference = states[0]
    evaluation = states[evaluation_step]
    final = states[-1]
    displacement_at_evaluation = np.linalg.norm(
        evaluation[:, :3] - reference[:, :3], axis=-1
    )
    rotation_at_evaluation = quaternion_geodesic_distance_wxyz(
        reference[:, 3:7], evaluation[:, 3:7]
    )
    final_displacement = np.linalg.norm(final[:, :3] - reference[:, :3], axis=-1)
    final_rotation = quaternion_geodesic_distance_wxyz(
        reference[:, 3:7], final[:, 3:7]
    )

    through_evaluation = states[1 : evaluation_step + 1]
    excursion = np.linalg.norm(
        through_evaluation[:, :, :3] - reference[None, :, :3], axis=-1
    )
    rotation_excursion = quaternion_geodesic_distance_wxyz(
        np.broadcast_to(reference[None, :, 3:7], through_evaluation[:, :, 3:7].shape),
        through_evaluation[:, :, 3:7],
    )

    early_stop = min(1 + early_window_steps, states.shape[0])
    early = states[1:early_stop]
    terminal_count = min(terminal_window_steps, states.shape[0] - 1)
    terminal = states[-terminal_count:]
    early_linear_speed = np.linalg.norm(early[:, :, 7:10], axis=-1)
    early_angular_speed = np.linalg.norm(early[:, :, 10:13], axis=-1)
    terminal_linear_speed = np.linalg.norm(terminal[:, :, 7:10], axis=-1)
    terminal_angular_speed = np.linalg.norm(terminal[:, :, 10:13], axis=-1)

    stable = (displacement_at_evaluation <= position_threshold_m) & (
        rotation_at_evaluation <= rotation_threshold_rad
    )
    return StabilityMetrics(
        evaluation_step=evaluation_step,
        position_threshold_m=position_threshold_m,
        rotation_threshold_rad=rotation_threshold_rad,
        displacement_at_evaluation_m=displacement_at_evaluation,
        rotation_at_evaluation_rad=rotation_at_evaluation,
        final_displacement_m=final_displacement,
        final_rotation_rad=final_rotation,
        maximum_excursion_m=np.max(excursion, axis=0),
        maximum_rotation_excursion_rad=np.max(rotation_excursion, axis=0),
        early_max_linear_speed_m_s=np.max(early_linear_speed, axis=0),
        early_max_angular_speed_rad_s=np.max(early_angular_speed, axis=0),
        terminal_max_linear_speed_m_s=np.max(terminal_linear_speed, axis=0),
        terminal_max_angular_speed_rad_s=np.max(terminal_angular_speed, axis=0),
        terminal_mean_linear_speed_m_s=np.mean(terminal_linear_speed, axis=0),
        terminal_mean_angular_speed_rad_s=np.mean(terminal_angular_speed, axis=0),
        stable=stable,
    )

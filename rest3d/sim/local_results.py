"""Backend-neutral state chaining and legacy local-group result export."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def make_initial_scene_states(
    scene_names: Sequence[str], *, base_dy: float
) -> dict[str, list[float]]:
    """Create identity REST3D WXYZ root states above the ground plane."""

    if not np.isfinite(base_dy):
        raise ValueError("base_dy must be finite")
    if len(set(scene_names)) != len(scene_names):
        raise ValueError("scene names must be unique")
    return {
        name: [0.0, float(base_dy), 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6
        for name in scene_names
    }


def validate_scene_states(
    states: Mapping[str, Sequence[float]], scene_names: Sequence[str]
) -> None:
    expected = set(scene_names)
    if set(states) != expected:
        raise ValueError("state object set must exactly match the scene object set")
    for name in scene_names:
        state = np.asarray(states[name], dtype=np.float64)
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError(f"state must contain 13 finite values: {name}")
        if np.linalg.norm(state[3:7]) < 1.0e-12:
            raise ValueError(f"state quaternion must be nonzero: {name}")


def merge_group_candidate_states(
    current_states: Mapping[str, Sequence[float]],
    candidate: Mapping,
    member_names: Sequence[str],
) -> dict[str, list[float]]:
    """Update only group members from one retained Isaac Lab candidate."""

    merged = {name: list(values) for name, values in current_states.items()}
    settled = candidate.get("settled_states_rest_wxyz", {})
    missing = set(member_names) - set(settled)
    if missing:
        raise ValueError(f"candidate is missing group members: {sorted(missing)}")
    for name in member_names:
        state = np.asarray(settled[name], dtype=np.float64)
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError(f"candidate state must contain 13 finite values: {name}")
        merged[name] = state.tolist()
    validate_scene_states(merged, list(current_states))
    return merged


def local_group_execution_order(groups: Sequence[Mapping]) -> list[int]:
    """Return a deterministic child-before-parent order for nested groups."""

    indices = [int(group["index"]) for group in groups]
    if len(set(indices)) != len(indices):
        raise ValueError("local group indices must be unique")
    by_index = {int(group["index"]): group for group in groups}
    dependencies = {index: set() for index in indices}
    for child_index, child in by_index.items():
        child_root = child["root_name"]
        for parent_index, parent in by_index.items():
            if child_index == parent_index:
                continue
            if child_root in set(parent["member_names"]) - {parent["root_name"]}:
                dependencies[parent_index].add(child_index)

    order: list[int] = []
    remaining = set(indices)
    while remaining:
        ready = sorted(index for index in remaining if not dependencies[index] & remaining)
        if not ready:
            raise ValueError("local group dependencies contain a cycle")
        order.extend(ready)
        remaining.difference_update(ready)
    return order


def legacy_local_group_payload(
    group: Mapping,
    final_states: Mapping[str, Sequence[float]],
    *,
    base_dy: float,
) -> dict:
    """Convert final WXYZ states to the XYZW schema consumed by stable_scene."""

    names = [group["root_name"], *group["direct_child_names"]]
    missing = set(names) - set(final_states)
    if missing:
        raise ValueError(f"final states are missing local group objects: {sorted(missing)}")
    objects = {}
    for name in names:
        state = np.asarray(final_states[name], dtype=np.float64)
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError(f"final state must contain 13 finite values: {name}")
        w, x, y, z = state[3:7]
        objects[name] = {
            "pos": state[:3].tolist(),
            "rot": [float(x), float(y), float(z), float(w)],
            "lin_vel": state[7:10].tolist(),
        }
    return {"base_dy": float(base_dy), "objects": objects}

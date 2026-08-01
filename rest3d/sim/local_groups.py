"""Backend-neutral local-group discovery and nearby collision context."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ObjectBounds:
    minimum: Tuple[float, float, float]
    maximum: Tuple[float, float, float]

    def arrays(self):
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("object bounds must contain two xyz vectors")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("object bounds must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("object bounds must have positive extents")
        return minimum, maximum


@dataclass(frozen=True)
class LocalGroupSpec:
    index: int
    root_name: str
    direct_child_names: Tuple[str, ...]
    member_names: Tuple[str, ...]
    driven_descendants: Tuple[Tuple[str, str], ...]
    context_names: Tuple[str, ...]
    interaction_margin_m: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LocalGroupPlan:
    scene_names: Tuple[str, ...]
    groups: Tuple[LocalGroupSpec, ...]

    def to_dict(self) -> dict:
        return {
            "scene_names": list(self.scene_names),
            "group_count": len(self.groups),
            "groups": [group.to_dict() for group in self.groups],
        }


def _aabb_distance(first: ObjectBounds, second: ObjectBounds) -> float:
    first_min, first_max = first.arrays()
    second_min, second_max = second.arrays()
    gap = np.maximum(np.maximum(first_min - second_max, second_min - first_max), 0.0)
    return float(np.linalg.norm(gap))


def _union_bounds(names: Iterable[str], bounds: Mapping[str, ObjectBounds]) -> ObjectBounds:
    arrays = [bounds[name].arrays() for name in names]
    if not arrays:
        raise ValueError("cannot combine an empty object set")
    return ObjectBounds(
        minimum=tuple(float(value) for value in np.min([item[0] for item in arrays], axis=0)),
        maximum=tuple(float(value) for value in np.max([item[1] for item in arrays], axis=0)),
    )


def build_local_group_plan(
    scene_tree_path: Union[str, Path],
    *,
    scene_names: Sequence[str],
    fixed_names: Iterable[str],
    object_bounds: Mapping[str, ObjectBounds],
    interaction_margin_m: float = 0.15,
) -> LocalGroupPlan:
    """Find local groups and geometry-nearby context without using object names."""

    if not np.isfinite(interaction_margin_m) or interaction_margin_m < 0.0:
        raise ValueError("interaction margin must be finite and non-negative")
    path = Path(scene_tree_path).expanduser().resolve(strict=True)
    with path.open("r", encoding="utf-8") as file:
        tree = json.load(file)
    ordered_names = tuple(scene_names)
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("scene object names must be unique")
    scene_set = set(ordered_names)
    if set(object_bounds) != scene_set:
        raise ValueError("object bounds must exactly match the scene object set")
    fixed_set = set(fixed_names)
    if not fixed_set <= scene_set:
        raise ValueError("fixed objects must belong to the scene object set")

    declared_nodes = set(tree.get("nodes", []))
    if scene_set != declared_nodes:
        raise ValueError("scene assets must exactly match declared non-root scene nodes")
    children: Dict[str, list] = {}
    parent_of: Dict[str, str] = {}
    for edge in tree.get("edges", []):
        child = edge["child"]
        parent = edge["parent"]
        if child in parent_of:
            raise ValueError("scene node has multiple parents: %s" % child)
        parent_of[child] = parent
        children.setdefault(parent, []).append(child)

    order = {name: index for index, name in enumerate(ordered_names)}

    def descendants(root_name):
        result = []
        pending = list(children.get(root_name, []))
        while pending:
            current = pending.pop(0)
            if current in scene_set:
                result.append(current)
            pending[0:0] = children.get(current, [])
        return result

    groups = []
    for root_name in ordered_names:
        direct_movable = [
            name
            for name in children.get(root_name, [])
            if name in scene_set and name not in fixed_set
        ]
        if root_name in fixed_set or not direct_movable:
            continue
        members = [root_name] + descendants(root_name)
        members = sorted(set(members), key=order.__getitem__)
        driven_descendants = []
        for direct_child in direct_movable:
            for descendant in descendants(direct_child):
                if descendant in members:
                    driven_descendants.append((descendant, direct_child))
        driven_descendants.sort(key=lambda pair: order[pair[0]])
        group_bounds = _union_bounds(members, object_bounds)
        context = []
        for candidate in ordered_names:
            if candidate in members:
                continue
            if _aabb_distance(group_bounds, object_bounds[candidate]) <= interaction_margin_m:
                context.append(candidate)
        ancestor = parent_of.get(root_name)
        while ancestor is not None:
            if ancestor in scene_set and ancestor not in members and ancestor not in context:
                context.append(ancestor)
            ancestor = parent_of.get(ancestor)
        context.sort(key=order.__getitem__)
        groups.append(
            LocalGroupSpec(
                index=len(groups),
                root_name=root_name,
                direct_child_names=tuple(sorted(direct_movable, key=order.__getitem__)),
                member_names=tuple(members),
                driven_descendants=tuple(driven_descendants),
                context_names=tuple(context),
                interaction_margin_m=float(interaction_margin_m),
            )
        )
    return LocalGroupPlan(scene_names=ordered_names, groups=tuple(groups))

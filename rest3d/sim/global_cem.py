"""Backend-neutral hierarchy expansion and object-set gates for global CEM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Union

import torch

from rest3d.utils.coll_det import gjk_batch

from .local_cem import (
    LocalCEMEnergyWeights,
    apply_pose_deltas_wxyz,
    evaluate_local_cem_energy,
    normalize_quaternion_wxyz,
    quaternion_conjugate_wxyz,
    quaternion_multiply_wxyz,
    quaternion_geodesic_distance_wxyz,
    quaternion_rotate_wxyz,
)


@dataclass(frozen=True)
class GlobalEntitySpec:
    """One sampled scene object and every descendant moved with that root."""

    index: int
    root_name: str
    member_names: Tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GlobalEntityPlan:
    """Deterministic partition used to sample roots and simulate all objects."""

    scene_names: Tuple[str, ...]
    fixed_names: Tuple[str, ...]
    static_names: Tuple[str, ...]
    owner_entity_indices: Tuple[int, ...]
    entities: Tuple[GlobalEntitySpec, ...]

    @property
    def sampled_entity_names(self) -> Tuple[str, ...]:
        return tuple(entity.root_name for entity in self.entities)

    def to_dict(self) -> dict:
        return {
            "scene_names": list(self.scene_names),
            "fixed_names": list(self.fixed_names),
            "static_names": list(self.static_names),
            "owner_entity_indices": list(self.owner_entity_indices),
            "sampled_entity_names": list(self.sampled_entity_names),
            "entity_count": len(self.entities),
            "entities": [entity.to_dict() for entity in self.entities],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "GlobalEntityPlan":
        """Load and validate the JSON representation used by Stage E jobs."""

        try:
            scene_names = tuple(payload["scene_names"])
            fixed_names = tuple(payload["fixed_names"])
            static_names = tuple(payload["static_names"])
            owner_entity_indices = tuple(
                int(value) for value in payload["owner_entity_indices"]
            )
            entities = tuple(
                GlobalEntitySpec(
                    index=int(record["index"]),
                    root_name=record["root_name"],
                    member_names=tuple(record["member_names"]),
                )
                for record in payload["entities"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid global entity plan payload") from error
        if len(set(scene_names)) != len(scene_names):
            raise ValueError("global entity plan scene names must be unique")
        scene_set = set(scene_names)
        if len(set(fixed_names)) != len(fixed_names):
            raise ValueError("global entity plan fixed names must be unique")
        if len(set(static_names)) != len(static_names):
            raise ValueError("global entity plan static names must be unique")
        if not set(fixed_names) <= scene_set or not set(static_names) <= scene_set:
            raise ValueError("global entity plan fixed/static names must belong to the scene")
        if len(owner_entity_indices) != len(scene_names):
            raise ValueError("global entity owner map must match the scene object count")
        if tuple(entity.index for entity in entities) != tuple(range(len(entities))):
            raise ValueError("global entity indices must be contiguous and ordered")
        owned_names = set()
        for entity in entities:
            if entity.root_name not in scene_set or not entity.member_names:
                raise ValueError("global entity root/member set is invalid")
            if len(set(entity.member_names)) != len(entity.member_names):
                raise ValueError("global entity member names must be unique")
            if entity.root_name not in set(entity.member_names):
                raise ValueError("global entity root must belong to its member set")
            if not set(entity.member_names) <= scene_set:
                raise ValueError("global entity members must belong to the scene")
            duplicate = owned_names & set(entity.member_names)
            if duplicate:
                raise ValueError("global entity member ownership overlaps")
            owned_names.update(entity.member_names)
        if owned_names & set(static_names):
            raise ValueError("owned and static global object sets must be disjoint")
        if owned_names | set(static_names) != scene_set:
            raise ValueError("owned and static global object sets must cover the scene")
        for object_index, owner_index in enumerate(owner_entity_indices):
            name = scene_names[object_index]
            if owner_index < -1 or owner_index >= len(entities):
                raise ValueError("global entity owner index is out of range")
            if owner_index == -1 and name not in set(static_names):
                raise ValueError("unowned global objects must be declared static")
            if owner_index >= 0 and name not in set(entities[owner_index].member_names):
                raise ValueError("global entity owner map disagrees with member sets")
        for entity in entities:
            for name in entity.member_names:
                if owner_entity_indices[scene_names.index(name)] != entity.index:
                    raise ValueError("global entity member is assigned to the wrong owner")
        plan = cls(
            scene_names=scene_names,
            fixed_names=fixed_names,
            static_names=static_names,
            owner_entity_indices=owner_entity_indices,
            entities=entities,
        )
        declared_sampled = payload.get("sampled_entity_names")
        if declared_sampled is not None and tuple(declared_sampled) != plan.sampled_entity_names:
            raise ValueError("global entity sampled-name summary is inconsistent")
        return plan


def build_global_entity_plan(
    scene_tree_path: Union[str, Path],
    *,
    scene_names: Sequence[str],
    fixed_names: Iterable[str] = (),
) -> GlobalEntityPlan:
    """Partition a scene into sampled movable roots and their full subtrees.

    A movable object is sampled only when it has no movable ancestor. Every
    descendant of that root is still assigned to the entity and will therefore
    be placed in every candidate environment. Fixed objects outside a sampled
    subtree remain static but are retained in the full scene object set.
    """

    path = Path(scene_tree_path).expanduser().resolve(strict=True)
    with path.open("r", encoding="utf-8") as file:
        tree = json.load(file)
    ordered_names = tuple(scene_names)
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("scene object names must be unique")
    scene_set = set(ordered_names)
    declared_nodes = set(tree.get("nodes", []))
    declared_roots = set(tree.get("roots", []))
    if scene_set != declared_nodes:
        raise ValueError("scene objects must exactly match declared non-root nodes")
    if declared_nodes & declared_roots:
        raise ValueError("scene roots and non-root nodes must be disjoint")

    fixed_set = set(fixed_names)
    if not fixed_set <= scene_set:
        raise ValueError("fixed objects must belong to the scene object set")
    movable_set = scene_set - fixed_set
    parent_of = {}
    children = {}
    for edge in tree.get("edges", []):
        child = edge["child"]
        parent = edge["parent"]
        if child not in scene_set:
            raise ValueError("scene edge child is not a declared object: %s" % child)
        if parent not in scene_set and parent not in declared_roots:
            raise ValueError("scene edge parent is undeclared: %s" % parent)
        if child in parent_of:
            raise ValueError("scene object has multiple parents: %s" % child)
        parent_of[child] = parent
        children.setdefault(parent, []).append(child)
    orphaned = scene_set - set(parent_of)
    if orphaned:
        raise ValueError("scene objects have no parent edge: %s" % sorted(orphaned))

    for name in ordered_names:
        visited = set()
        current = name
        while current in parent_of:
            if current in visited:
                raise ValueError("scene tree contains a cycle at: %s" % current)
            visited.add(current)
            current = parent_of[current]

    order = {name: index for index, name in enumerate(ordered_names)}

    def ancestors(name):
        result = []
        current = parent_of.get(name)
        while current in scene_set:
            result.append(current)
            current = parent_of.get(current)
        return result

    def descendants(name):
        result = []
        pending = list(children.get(name, []))
        while pending:
            current = pending.pop(0)
            if current in scene_set:
                result.append(current)
                pending[0:0] = children.get(current, [])
        return result

    sampled_roots = [
        name
        for name in ordered_names
        if name in movable_set
        and not any(ancestor in movable_set for ancestor in ancestors(name))
    ]
    owner_by_name = {}
    entities = []
    for root_name in sampled_roots:
        members = sorted(
            set([root_name, *descendants(root_name)]), key=order.__getitem__
        )
        entity_index = len(entities)
        for name in members:
            if name in owner_by_name:
                raise ValueError(
                    "scene object belongs to multiple global entities: %s" % name
                )
            owner_by_name[name] = entity_index
        entities.append(
            GlobalEntitySpec(
                index=entity_index,
                root_name=root_name,
                member_names=tuple(members),
            )
        )
    unowned_movable = movable_set - set(owner_by_name)
    if unowned_movable:
        raise ValueError(
            "movable scene objects have no sampled entity owner: %s"
            % sorted(unowned_movable)
        )
    static_names = tuple(name for name in ordered_names if name not in owner_by_name)
    owner_indices = tuple(owner_by_name.get(name, -1) for name in ordered_names)
    plan = GlobalEntityPlan(
        scene_names=ordered_names,
        fixed_names=tuple(name for name in ordered_names if name in fixed_set),
        static_names=static_names,
        owner_entity_indices=owner_indices,
        entities=tuple(entities),
    )
    if not set(plan.sampled_entity_names) <= scene_set:
        raise AssertionError("sampled global entities must belong to the scene")
    return plan


def validate_global_object_sets(
    plan: GlobalEntityPlan,
    *,
    simulated_names: Sequence[str],
    evaluated_names: Sequence[str],
) -> None:
    """Enforce the paper-required full-scene simulator and energy domains."""

    expected = set(plan.scene_names)
    simulated = tuple(simulated_names)
    evaluated = tuple(evaluated_names)
    if len(set(simulated)) != len(simulated):
        raise ValueError("simulated object names must be unique")
    if len(set(evaluated)) != len(evaluated):
        raise ValueError("evaluated object names must be unique")
    if set(simulated) != expected:
        raise ValueError("simulated object set must exactly match the scene object set")
    if set(evaluated) != expected:
        raise ValueError("evaluated object set must exactly match the scene object set")
    if not set(plan.sampled_entity_names) <= expected:
        raise ValueError("sampled entity set must be a subset of the scene object set")


def evaluate_convex_hull_intersections_wxyz(
    states: torch.Tensor,
    convex_hull_vertices: Sequence[torch.Tensor],
    excluded_pairs: Iterable[Tuple[int, int]] = (),
) -> Mapping[str, torch.Tensor]:
    """Count all pairwise convex-hull intersections for every environment.

    This implements the paper's geometric penetration term with the existing
    batched GJK predicate.  Each intersecting unordered pair contributes one to
    ``total`` and one half to each endpoint in ``per_object``, so summing the
    per-object attribution reproduces the global count without double-counting.
    """

    if states.ndim != 3 or states.shape[-1] != 13:
        raise ValueError("states must have shape [environments, objects, 13]")
    environment_count, object_count = states.shape[:2]
    if len(convex_hull_vertices) != object_count:
        raise ValueError("one convex hull is required for every scene object")
    if not bool(torch.isfinite(states).all()):
        raise ValueError("intersection states must be finite")

    world_hulls = []
    for object_index, vertices in enumerate(convex_hull_vertices):
        if not isinstance(vertices, torch.Tensor) or vertices.ndim != 2:
            raise ValueError("convex hull vertices must be rank-two tensors")
        if vertices.shape[0] < 4 or vertices.shape[1] != 3:
            raise ValueError("each convex hull must contain at least four xyz vertices")
        if vertices.device != states.device or vertices.dtype != states.dtype:
            raise ValueError("convex hulls and states must share device and dtype")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("convex hull vertices must be finite")
        vertex_count = vertices.shape[0]
        orientations = states[:, object_index, 3:7].unsqueeze(1).expand(
            -1, vertex_count, -1
        )
        local_vertices = vertices.unsqueeze(0).expand(environment_count, -1, -1)
        world_hulls.append(
            quaternion_rotate_wxyz(orientations, local_vertices)
            + states[:, object_index, :3].unsqueeze(1)
        )

    excluded = {tuple(sorted((int(first), int(second)))) for first, second in excluded_pairs}
    if any(
        first == second or first < 0 or second >= object_count
        for first, second in excluded
    ):
        raise ValueError("excluded intersection pair indices are invalid")
    total = torch.zeros(environment_count, device=states.device, dtype=states.dtype)
    per_object = torch.zeros(
        environment_count, object_count, device=states.device, dtype=states.dtype
    )
    pair_matrix = torch.zeros(
        environment_count,
        object_count,
        object_count,
        device=states.device,
        dtype=states.dtype,
    )
    for first in range(object_count):
        for second in range(first + 1, object_count):
            if (first, second) in excluded:
                continue
            intersects = gjk_batch(
                world_hulls[first],
                world_hulls[second],
                nonconverged_is_collision=False,
            ).to(dtype=states.dtype)
            total += intersects
            per_object[:, first] += 0.5 * intersects
            per_object[:, second] += 0.5 * intersects
            pair_matrix[:, first, second] = intersects
            pair_matrix[:, second, first] = intersects
    return {
        "total": total,
        "per_object": per_object,
        "pair_matrix": pair_matrix,
    }


def expand_global_root_samples_wxyz(
    reference_states: torch.Tensor,
    plan: GlobalEntityPlan,
    samples: torch.Tensor,
) -> torch.Tensor:
    """Expand sampled root 6-DoF deltas into full-scene WXYZ root states.

    Child transforms are recomputed as
    ``T_child_world = T_candidate_root_world @ T_child_local``. The returned
    object axis always follows ``plan.scene_names`` and includes static objects.
    """

    object_count = len(plan.scene_names)
    entity_count = len(plan.entities)
    if reference_states.shape != (object_count, 13):
        raise ValueError("reference_states must have shape [scene objects, 13]")
    if samples.ndim != 3 or samples.shape[1:] != (entity_count, 6):
        raise ValueError("samples must have shape [environments, sampled entities, 6]")
    if reference_states.device != samples.device:
        raise ValueError("reference states and samples must share a device")
    if reference_states.dtype != samples.dtype:
        raise ValueError("reference states and samples must share a dtype")
    if not bool(
        torch.isfinite(reference_states).all() and torch.isfinite(samples).all()
    ):
        raise ValueError("reference states and samples must be finite")

    index_by_name = {name: index for index, name in enumerate(plan.scene_names)}
    root_indices = torch.tensor(
        [index_by_name[entity.root_name] for entity in plan.entities],
        dtype=torch.long,
        device=reference_states.device,
    )
    candidate_root_poses = apply_pose_deltas_wxyz(
        reference_states[root_indices, :7], samples
    )
    environment_count = samples.shape[0]
    result = reference_states.unsqueeze(0).expand(
        environment_count, -1, -1
    ).clone()
    result[..., 7:13] = 0.0

    for entity in plan.entities:
        member_indices = torch.tensor(
            [index_by_name[name] for name in entity.member_names],
            dtype=torch.long,
            device=reference_states.device,
        )
        reference_root = reference_states[index_by_name[entity.root_name], :7]
        reference_members = reference_states[member_indices, :7]
        inverse_root_quaternion = quaternion_conjugate_wxyz(
            normalize_quaternion_wxyz(reference_root[3:7])
        )
        member_count = len(entity.member_names)
        inverse_root_batch = inverse_root_quaternion.unsqueeze(0).expand(
            member_count, -1
        )
        local_positions = quaternion_rotate_wxyz(
            inverse_root_batch,
            reference_members[:, :3] - reference_root[:3].unsqueeze(0),
        )
        local_quaternions = normalize_quaternion_wxyz(
            quaternion_multiply_wxyz(
                inverse_root_batch,
                normalize_quaternion_wxyz(reference_members[:, 3:7]),
            )
        )

        candidate_root = candidate_root_poses[:, entity.index]
        candidate_quaternion = candidate_root[:, 3:7].unsqueeze(1).expand(
            -1, member_count, -1
        )
        local_position_batch = local_positions.unsqueeze(0).expand(
            environment_count, -1, -1
        )
        local_quaternion_batch = local_quaternions.unsqueeze(0).expand(
            environment_count, -1, -1
        )
        result[:, member_indices, :3] = (
            candidate_root[:, :3].unsqueeze(1)
            + quaternion_rotate_wxyz(
                candidate_quaternion, local_position_batch
            )
        )
        result[:, member_indices, 3:7] = normalize_quaternion_wxyz(
            quaternion_multiply_wxyz(
                candidate_quaternion, local_quaternion_batch
            )
        )
    return result


def evaluate_global_cem_energy(
    plan: GlobalEntityPlan,
    evaluated_names: Sequence[str],
    placed_states: torch.Tensor,
    early_states: torch.Tensor,
    settled_states: torch.Tensor,
    reference_poses: torch.Tensor,
    *,
    centroid_offsets=None,
    placement_penetration=None,
    settled_penetration=None,
    placement_penetration_by_object=None,
    settled_penetration_by_object=None,
    weights=None,
) -> Mapping:
    """Evaluate every scene object with the shared physical energy terms."""

    if tuple(evaluated_names) != plan.scene_names:
        raise ValueError("global energy object order must match the scene plan")
    if placed_states.ndim != 3 or placed_states.shape[1] != len(plan.scene_names):
        raise ValueError("global energy must receive every scene object")
    weights = LocalCEMEnergyWeights() if weights is None else weights
    result = evaluate_local_cem_energy(
        placed_states,
        early_states,
        settled_states,
        reference_poses,
        centroid_offsets=centroid_offsets,
        placement_penetration=placement_penetration,
        settled_penetration=settled_penetration,
        weights=weights,
    )
    environment_count, object_count = placed_states.shape[:2]
    device = placed_states.device
    dtype = placed_states.dtype
    if centroid_offsets is None:
        centroid_offsets = torch.zeros(object_count, 3, device=device, dtype=dtype)
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
    zeros = torch.zeros(
        environment_count, object_count, device=device, dtype=dtype
    )
    for name, value in (
        ("placement_penetration_by_object", placement_penetration_by_object),
        ("settled_penetration_by_object", settled_penetration_by_object),
    ):
        if value is not None and (
            value.shape != (environment_count, object_count)
            or value.device != device
        ):
            raise ValueError(
                "%s must have shape [environments, scene objects] on the state device"
                % name
            )
    per_object = {
        "pose_stability": torch.linalg.vector_norm(
            settled_points - placed_points, dim=-1
        ),
        "rotation_stability": quaternion_geodesic_distance_wxyz(
            settled_states[..., 3:7], placed_states[..., 3:7]
        ),
        "pose_layout": torch.linalg.vector_norm(
            settled_points - reference_points, dim=-1
        ),
        "rotation_layout": quaternion_geodesic_distance_wxyz(
            settled_states[..., 3:7], reference[..., 3:7]
        ),
        "velocity": torch.linalg.vector_norm(early_states[..., 7:10], dim=-1),
        "placement_penetration": placement_penetration_by_object
        if placement_penetration_by_object is not None
        else zeros,
        "settled_penetration": settled_penetration_by_object
        if settled_penetration_by_object is not None
        else zeros,
    }
    per_object["energy"] = (
        weights.pose_stability * per_object["pose_stability"]
        + weights.rotation_stability * per_object["rotation_stability"]
        + weights.pose_layout * per_object["pose_layout"]
        + weights.rotation_layout * per_object["rotation_layout"]
        + weights.velocity * per_object["velocity"]
        + weights.placement_penetration * per_object["placement_penetration"]
        + weights.settled_penetration * per_object["settled_penetration"]
    )
    result["per_object"] = per_object
    return result

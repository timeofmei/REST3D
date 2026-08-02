"""Resolve Stage 2 vertical placement from explicit scene-support semantics."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from rest3d.utils.mesh import read_obj_vertices


_LOGGER = logging.getLogger("stage2")
ROOT_SUPPORTS = frozenset({"floor", "wall", "ceiling", "floor-wall"})
PRESERVE_RELATIONS = frozenset({"inside", "supported-by"})


def _bounds(vertices: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    if points.size == 0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def _xz_overlap_ratio(
    child_min: np.ndarray,
    child_max: np.ndarray,
    parent_min: np.ndarray,
    parent_max: np.ndarray,
) -> float:
    overlap_min = np.maximum(child_min[[0, 2]], parent_min[[0, 2]])
    overlap_max = np.minimum(child_max[[0, 2]], parent_max[[0, 2]])
    overlap_area = float(np.prod(np.maximum(0.0, overlap_max - overlap_min)))
    child_area = float(np.prod(np.maximum(0.0, child_max[[0, 2]] - child_min[[0, 2]])))
    return overlap_area / child_area if child_area > 0.0 else 0.0


def _is_complex_legacy_on(
    child_min_y: float,
    parent_min_y: float,
    parent_max_y: float,
    *,
    min_overlap_m: float,
    min_parent_overlap_fraction: float,
) -> tuple[bool, float, float]:
    """Detect an ``on`` edge that cannot mean bottom-to-top stacking.

    A child that substantially straddles the parent's vertical extent is a
    partial or multi-contact support case. Moving its global lowest point above
    the parent's global highest point destroys the reconstructed relationship.
    Both an absolute and scale-relative threshold are required so ordinary
    reconstruction noise around a tabletop still snaps to the top surface.
    """

    penetration = max(0.0, parent_max_y - child_min_y)
    parent_height = max(0.0, parent_max_y - parent_min_y)
    threshold = max(min_overlap_m, min_parent_overlap_fraction * parent_height)
    return penetration > threshold, penetration, threshold


def _record(
    records: dict[str, dict[str, Any]],
    *,
    name: str,
    parent: str,
    relation: str,
    declared_type: str,
    placement_mode: str,
    source_min: np.ndarray,
    source_max: np.ndarray,
    y_offset: float,
    details: dict[str, Any] | None = None,
) -> None:
    output_min = source_min.copy()
    output_max = source_max.copy()
    output_min[1] += y_offset
    output_max[1] += y_offset
    records[name] = {
        "parent": parent,
        "relation": relation,
        "declared_type": declared_type,
        "placement_mode": placement_mode,
        "source_bounds_min_m": source_min.tolist(),
        "source_bounds_max_m": source_max.tolist(),
        "output_bounds_min_m": output_min.tolist(),
        "output_bounds_max_m": output_max.tolist(),
        "translation_m": [0.0, float(y_offset), 0.0],
        "rotation_changed": False,
        **(details or {}),
    }


def _write_obj_with_translation(
    output_path: Path, lines: list[str], translation: np.ndarray
) -> None:
    """Write an OBJ after a pure translation while preserving vertex colors."""

    with output_path.open("w", encoding="utf-8") as file_obj:
        for line in lines:
            if not line.startswith("v "):
                file_obj.write(line)
                continue
            parts = line.split()
            position = np.asarray(parts[1:4], dtype=np.float64) + translation
            extra = parts[4:]
            suffix = f" {' '.join(extra)}" if extra else ""
            file_obj.write(
                f"v {position[0]} {position[1]} {position[2]}{suffix}\n"
            )


def _descendants_excluding_subtrees(
    root: str,
    children_map: dict[str, list[str]],
    excluded_roots: set[str],
) -> list[str]:
    descendants: list[str] = []
    queue = deque(children_map.get(root, []))
    while queue:
        name = queue.popleft()
        if name in excluded_roots:
            continue
        descendants.append(name)
        queue.extend(children_map.get(name, []))
    return descendants


def _as_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene has no geometry: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError(f"mesh has no usable faces: {path}")
    return loaded


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    target = quantile * float(cumulative[-1])
    index = min(np.searchsorted(cumulative, target), len(values) - 1)
    return float(sorted_values[index])


def _partial_support_surface_gap(
    *,
    child_path: Path,
    parent_path: Path,
    child_translation: np.ndarray,
    parent_translation: np.ndarray,
    normal_y_threshold: float,
    contact_quantile: float,
    maximum_gap_m: float,
) -> tuple[float | None, dict[str, Any]]:
    """Estimate the first substantial child-bottom/parent-top surface gap.

    Global extrema cannot describe partial support: a seated figure's lowest
    point is a shoe and a chair's highest point is usually its backrest.  This
    instead pairs locally overlapping downward- and upward-facing triangles,
    then uses a low area-weighted gap quantile to ignore isolated near-misses.
    """

    child_mesh = _as_trimesh(child_path)
    parent_mesh = _as_trimesh(parent_path)
    child_mask = child_mesh.face_normals[:, 1] <= -normal_y_threshold
    parent_mask = parent_mesh.face_normals[:, 1] >= normal_y_threshold
    if not np.any(child_mask) or not np.any(parent_mask):
        return None, {"reason": "insufficient-horizontal-surfaces"}

    child_centers = child_mesh.triangles_center[child_mask] + child_translation
    parent_centers = parent_mesh.triangles_center[parent_mask] + parent_translation
    child_areas = (
        child_mesh.area_faces[child_mask]
        * -child_mesh.face_normals[child_mask, 1]
    )
    parent_areas = (
        parent_mesh.area_faces[parent_mask]
        * parent_mesh.face_normals[parent_mask, 1]
    )
    child_valid = child_areas > 1.0e-10
    parent_valid = parent_areas > 1.0e-10
    child_centers = child_centers[child_valid]
    child_areas = child_areas[child_valid]
    parent_centers = parent_centers[parent_valid]
    if not len(child_centers) or not len(parent_centers):
        return None, {"reason": "insufficient-nondegenerate-surfaces"}

    parent_xz_extent = np.ptp(parent_centers[:, [0, 2]], axis=0)
    parent_xz_diagonal = float(np.linalg.norm(parent_xz_extent))
    dense_mesh_radius = float(np.clip(0.04 * parent_xz_diagonal, 0.008, 0.025))
    sparse_face_radius = float(np.sqrt(np.quantile(parent_areas[parent_valid], 0.9)))
    search_radius = max(
        dense_mesh_radius,
        min(0.25 * parent_xz_diagonal, sparse_face_radius),
    )
    neighbor_count = min(24, len(parent_centers))
    _, indices = cKDTree(parent_centers[:, [0, 2]]).query(
        child_centers[:, [0, 2]],
        k=neighbor_count,
        distance_upper_bound=search_radius,
        workers=-1,
    )
    if neighbor_count == 1:
        indices = indices[:, None]

    valid_neighbor = indices < len(parent_centers)
    parent_y = np.full(indices.shape, -np.inf, dtype=np.float64)
    parent_y[valid_neighbor] = parent_centers[indices[valid_neighbor], 1]
    below_child = valid_neighbor & (
        parent_y <= child_centers[:, 1, None] + 0.002
    )
    parent_y[~below_child] = -np.inf
    highest_parent_y = parent_y.max(axis=1)
    matched = np.isfinite(highest_parent_y)
    gaps = child_centers[matched, 1] - highest_parent_y[matched]
    weights = child_areas[matched]
    usable = (gaps >= 0.0) & (gaps <= maximum_gap_m)
    gaps = gaps[usable]
    weights = weights[usable]

    child_footprint_area = float(
        np.prod(np.maximum(0.0, np.ptp(child_mesh.vertices[:, [0, 2]], axis=0)))
    )
    matched_area = float(weights.sum())
    minimum_evidence_area = max(1.0e-5, 0.005 * child_footprint_area)
    if not len(gaps) or matched_area < minimum_evidence_area:
        return None, {
            "reason": "insufficient-overlapping-surface-area",
            "matched_projected_area_m2": matched_area,
            "minimum_projected_area_m2": minimum_evidence_area,
            "search_radius_m": search_radius,
        }

    gap = _weighted_quantile(gaps, weights, contact_quantile)
    return gap, {
        "method": "local-horizontal-face-correspondence",
        "estimated_gap_m": gap,
        "contact_quantile": contact_quantile,
        "matched_projected_area_m2": matched_area,
        "minimum_projected_area_m2": minimum_evidence_area,
        "search_radius_m": search_radius,
        "normal_y_threshold": normal_y_threshold,
    }


def place_to_ground(
    output_obj_y_align_dir: str | os.PathLike[str],
    output_obj_canon_dir: str | os.PathLike[str],
    scene_tree_path: str | os.PathLike[str],
    ceiling_height_threshold: float = 1.8,
    *,
    report_path: str | os.PathLike[str] | None = None,
    resolved_scene_tree_path: str | os.PathLike[str] | None = None,
    clearance_m: float = 0.005,
    complex_overlap_m: float = 0.05,
    complex_parent_overlap_fraction: float = 0.25,
    partial_support_min_xz_overlap_ratio: float = 0.5,
    partial_support_max_parent_shift_m: float = 0.35,
    partial_support_max_child_drop_m: float = 0.2,
    partial_support_contact_quantile: float = 0.05,
    partial_support_surface_normal_y: float = 0.65,
) -> float:
    """Translate y-aligned meshes vertically according to support semantics.

    ``on`` is reserved for simple bottom-to-top stacking. ``inside`` and
    ``supported-by`` preserve the reconstructed parent/child vertical pose.
    Legacy ``on`` edges that substantially straddle their parent's vertical
    extent are conservatively treated as partial support as well.

    The input directory is read-only. Canonical meshes and the optional JSON
    diagnostics report are written below caller-selected output paths.
    """

    input_dir = Path(output_obj_y_align_dir).expanduser().resolve(strict=True)
    output_dir = Path(output_obj_canon_dir).expanduser().resolve()
    tree_path = Path(scene_tree_path).expanduser().resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("\n%s", "=" * 60)
    _LOGGER.info("Resolve vertical placement from scene-support semantics...")
    _LOGGER.info("%s\n", "=" * 60)

    with tree_path.open(encoding="utf-8") as file_obj:
        scene_tree = json.load(file_obj)

    parent_map: dict[str, str] = {}
    relation_map: dict[str, str] = {}
    type_map: dict[str, str] = {}
    physics_role_map: dict[str, str] = {}
    children_map: dict[str, list[str]] = defaultdict(list)
    for edge in scene_tree.get("edges", []):
        child = edge["child"]
        parent = edge["parent"]
        if parent.startswith("the_floor"):
            parent = "floor"
        parent_map[child] = parent
        relation_map[child] = edge.get("relation", "on")
        type_map[child] = edge.get("type", "movable")
        physics_role_map[child] = edge.get("physics_role", "")
        children_map[parent].append(child)

    all_nodes: set[str] = set()
    queue = deque(scene_tree.get("roots", []))
    while queue:
        parent = queue.popleft()
        for child in children_map.get(parent, []):
            if child not in all_nodes:
                all_nodes.add(child)
                queue.append(child)

    objects: dict[str, dict[str, Any]] = {}
    missing_meshes: list[str] = []
    for name in sorted(all_nodes):
        if name.startswith("the_floor"):
            continue
        path = input_dir / f"scene_y_align_{name}.obj"
        if not path.is_file():
            missing_meshes.append(name)
            continue
        vertices, lines = read_obj_vertices(str(path))
        minimum, maximum = _bounds(vertices)
        objects[name] = {
            "path": path,
            "vertices": vertices,
            "lines": lines,
            "minimum": minimum,
            "maximum": maximum,
            "center": np.asarray(vertices, dtype=np.float64).mean(axis=0),
        }

    offsets: dict[str, float] = {}
    translations: dict[str, np.ndarray] = {}
    output_max_y: dict[str, float] = {}
    processed: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    floor_offsets: list[float] = []

    def write_object(
        name: str,
        y_offset: float,
        placement_mode: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        data = objects[name]
        output_path = output_dir / f"scene_canon_{name}.obj"
        translation = np.array([0.0, y_offset, 0.0], dtype=np.float64)
        _write_obj_with_translation(output_path, data["lines"], translation)
        offsets[name] = float(y_offset)
        translations[name] = translation
        output_max_y[name] = float(data["maximum"][1] + y_offset)
        _record(
            records,
            name=name,
            parent=parent_map.get(name, "floor"),
            relation=relation_map.get(name, "on"),
            declared_type=type_map.get(name, "movable"),
            placement_mode=placement_mode,
            source_min=data["minimum"],
            source_max=data["maximum"],
            y_offset=y_offset,
            details=details,
        )
        processed.add(name)

    for root in ("floor", "floor-wall"):
        for name in children_map.get(root, []):
            if name not in objects:
                processed.add(name)
                continue
            y_offset = -float(objects[name]["minimum"][1])
            write_object(name, y_offset, f"{root}-contact")
            floor_offsets.append(y_offset)
            _LOGGER.info("  %s: %s contact, y_offset=%.4f", name, root, y_offset)

    significant_offsets = [value for value in floor_offsets if value > 0.05]
    reference_offset = float(np.median(significant_offsets or floor_offsets or [0.0]))
    _LOGGER.info("  Reference scene offset: %.4f", reference_offset)

    for name in children_map.get("wall", []):
        if name not in objects:
            processed.add(name)
            continue
        minimum = objects[name]["minimum"]
        y_offset = reference_offset + max(0.0, -float(minimum[1] + reference_offset))
        write_object(name, y_offset, "wall-relative")

    ceiling_children = set(children_map.get("ceiling", []))
    processed.update(ceiling_children)

    queue = deque()
    for name in tuple(processed):
        queue.extend(child for child in children_map.get(name, []) if child not in processed)

    while queue:
        name = queue.popleft()
        if name in processed:
            continue
        if name not in objects:
            processed.add(name)
            _LOGGER.warning("  %s: y-aligned OBJ not found, skipping", name)
            continue

        data = objects[name]
        minimum = data["minimum"]
        maximum = data["maximum"]
        parent = parent_map.get(name, "floor")
        relation = relation_map.get(name, "on")
        details: dict[str, Any] = {}

        if parent not in offsets or parent not in objects:
            y_offset = -float(minimum[1])
            mode = "fallback-floor-contact"
        elif relation == "hang":
            y_offset = output_max_y[parent] + clearance_m - float(maximum[1])
            mode = "hang-top-aligned"
        else:
            parent_data = objects[parent]
            parent_minimum = parent_data["minimum"]
            parent_maximum = parent_data["maximum"]
            parent_offset = offsets[parent]
            complex_legacy_on, penetration, complex_threshold = _is_complex_legacy_on(
                float(minimum[1]),
                float(parent_minimum[1]),
                float(parent_maximum[1]),
                min_overlap_m=complex_overlap_m,
                min_parent_overlap_fraction=complex_parent_overlap_fraction,
            )
            details = {
                "parent_source_bounds_min_m": parent_minimum.tolist(),
                "parent_source_bounds_max_m": parent_maximum.tolist(),
                "source_xz_child_overlap_ratio": _xz_overlap_ratio(
                    minimum, maximum, parent_minimum, parent_maximum
                ),
                "source_vertical_top_penetration_m": float(penetration),
                "complex_support_threshold_m": float(complex_threshold),
            }
            preserve_relative = relation in PRESERVE_RELATIONS or (
                relation == "on" and complex_legacy_on
            )
            if preserve_relative:
                y_offset = parent_offset
                if minimum[1] + y_offset < 0.0:
                    y_offset += -float(minimum[1] + y_offset)
                    mode = "preserve-relative-floor-clamped"
                elif relation == "on":
                    mode = "preserve-relative-auto"
                else:
                    mode = "preserve-relative-explicit"
                details["relative_min_y_before_m"] = float(
                    minimum[1] - parent_minimum[1]
                )
                details["relative_min_y_after_m"] = float(
                    minimum[1] + y_offset - (parent_minimum[1] + parent_offset)
                )
            else:
                target_min_y = output_max_y[parent] + clearance_m
                y_offset = target_min_y - float(minimum[1])
                mode = "top-surface"

        write_object(name, y_offset, mode, details)
        _LOGGER.info(
            "  %s: parent=%s relation=%s mode=%s y=[%.4f, %.4f]",
            name,
            parent,
            relation,
            mode,
            minimum[1] + y_offset,
            maximum[1] + y_offset,
        )
        queue.extend(child for child in children_map.get(name, []) if child not in processed)

    non_ceiling_max = max(
        (value for name, value in output_max_y.items() if name not in ceiling_children),
        default=0.0,
    )
    ceiling_height = max(non_ceiling_max, ceiling_height_threshold)

    for name in sorted(ceiling_children):
        if name not in objects:
            continue
        minimum = objects[name]["minimum"]
        maximum = objects[name]["maximum"]
        y_offset = reference_offset + max(0.0, -float(minimum[1] + reference_offset))
        if maximum[1] + y_offset > ceiling_height:
            y_offset = max(ceiling_height - float(maximum[1]), -float(minimum[1]))
        write_object(name, y_offset, "ceiling-relative")

    # A partially supported kinematic child is a reconstructed world-space
    # anchor, not a payload that should be snapped onto a parent's global AABB
    # top. If the source reconstruction only places the parent under a small
    # edge of that anchor, move the parent horizontally beneath the supported
    # object. Ordinary descendants follow the parent, while anchored child
    # subtrees retain their world pose.
    partial_children_by_parent: dict[str, list[str]] = defaultdict(list)
    for child, record in records.items():
        partial_support = record["relation"] == "supported-by" or (
            record["relation"] == "on"
            and record["placement_mode"].startswith("preserve-relative")
        )
        explicit_role = physics_role_map.get(child, "")
        is_kinematic_anchor = explicit_role == "kinematic" or (
            not explicit_role and partial_support
        )
        if is_kinematic_anchor:
            parent = record["parent"]
            if parent in records and parent in objects:
                partial_children_by_parent[parent].append(child)

    for parent, anchor_children in partial_children_by_parent.items():
        current_parent_min = objects[parent]["minimum"] + translations[parent]
        current_parent_max = objects[parent]["maximum"] + translations[parent]
        overlap_before = {
            child: _xz_overlap_ratio(
                objects[child]["minimum"] + translations[child],
                objects[child]["maximum"] + translations[child],
                current_parent_min,
                current_parent_max,
            )
            for child in anchor_children
            if child in objects and child in translations
        }
        if not overlap_before:
            continue

        shift = np.zeros(3, dtype=np.float64)
        if min(overlap_before.values()) < partial_support_min_xz_overlap_ratio:
            child_centers = np.stack(
                [
                    objects[child]["center"] + translations[child]
                    for child in overlap_before
                ]
            )
            parent_center = objects[parent]["center"] + translations[parent]
            shift[[0, 2]] = (
                child_centers[:, [0, 2]].mean(axis=0) - parent_center[[0, 2]]
            )
            horizontal_norm = float(np.linalg.norm(shift[[0, 2]]))
            if horizontal_norm > partial_support_max_parent_shift_m:
                shift *= partial_support_max_parent_shift_m / horizontal_norm

        def overlap_after_fraction(fraction: float) -> dict[str, float]:
            candidate_min = current_parent_min + fraction * shift
            candidate_max = current_parent_max + fraction * shift
            return {
                child: _xz_overlap_ratio(
                    objects[child]["minimum"] + translations[child],
                    objects[child]["maximum"] + translations[child],
                    candidate_min,
                    candidate_max,
                )
                for child in overlap_before
            }

        # Preserve as much of the reconstructed parent pose as possible. Search
        # only along the center-alignment direction and stop as soon as every
        # anchor reaches the requested footprint coverage.
        full_shift_overlap = overlap_after_fraction(1.0)
        if (
            np.any(shift)
            and min(full_shift_overlap.values())
            >= partial_support_min_xz_overlap_ratio
        ):
            lower = 0.0
            upper = 1.0
            for _ in range(40):
                middle = 0.5 * (lower + upper)
                if (
                    min(overlap_after_fraction(middle).values())
                    >= partial_support_min_xz_overlap_ratio
                ):
                    upper = middle
                else:
                    lower = middle
            shift *= upper

        excluded_roots = set(overlap_before)
        moved_names = [parent] + _descendants_excluding_subtrees(
            parent, children_map, excluded_roots
        )
        moved_names = [
            name for name in moved_names if name in objects and name in translations
        ]
        for moved_name in moved_names:
            translations[moved_name] = translations[moved_name] + shift
            _write_obj_with_translation(
                output_dir / f"scene_canon_{moved_name}.obj",
                objects[moved_name]["lines"],
                translations[moved_name],
            )
            record = records[moved_name]
            record["translation_m"] = translations[moved_name].tolist()
            record["output_bounds_min_m"] = (
                objects[moved_name]["minimum"] + translations[moved_name]
            ).tolist()
            record["output_bounds_max_m"] = (
                objects[moved_name]["maximum"] + translations[moved_name]
            ).tolist()

        adjusted_parent_min = objects[parent]["minimum"] + translations[parent]
        adjusted_parent_max = objects[parent]["maximum"] + translations[parent]
        overlap_after = {
            child: _xz_overlap_ratio(
                objects[child]["minimum"] + translations[child],
                objects[child]["maximum"] + translations[child],
                adjusted_parent_min,
                adjusted_parent_max,
            )
            for child in overlap_before
        }
        records[parent]["partial_support_alignment"] = {
            "anchor_children": sorted(overlap_before),
            "horizontal_shift_m": shift.tolist(),
            "moved_objects": moved_names,
            "xz_overlap_ratio_before": overlap_before,
            "xz_overlap_ratio_after": overlap_after,
            "minimum_required_xz_overlap_ratio": partial_support_min_xz_overlap_ratio,
        }
        for child in overlap_before:
            records[child]["output_xz_child_overlap_ratio"] = overlap_after[child]
            records[child]["support_parent_horizontal_shift_m"] = shift.tolist()
        _LOGGER.info(
            "  %s: align partial support under %s, shift=[%.4f, %.4f] m",
            parent,
            sorted(overlap_before),
            shift[0],
            shift[2],
        )

        # Once the support is horizontally beneath the anchor, close the local
        # support-surface gap by moving the anchored child subtree.  The parent
        # remains floor-contact, and the anchor is never dropped through the
        # floor or farther than the conservative configured limit.
        for child in overlap_before:
            gap, contact_details = _partial_support_surface_gap(
                child_path=objects[child]["path"],
                parent_path=objects[parent]["path"],
                child_translation=translations[child],
                parent_translation=translations[parent],
                normal_y_threshold=partial_support_surface_normal_y,
                contact_quantile=partial_support_contact_quantile,
                maximum_gap_m=partial_support_max_child_drop_m + clearance_m,
            )
            records[child]["partial_support_contact"] = contact_details
            if gap is None or gap <= clearance_m:
                continue

            requested_drop = min(
                gap - clearance_m, partial_support_max_child_drop_m
            )
            available_floor_clearance = max(
                0.0,
                float(objects[child]["minimum"][1] + translations[child][1]),
            )
            applied_drop = min(requested_drop, available_floor_clearance)
            vertical_shift = np.array([0.0, -applied_drop, 0.0], dtype=np.float64)
            moved_anchor_names = [child] + _descendants_excluding_subtrees(
                child, children_map, set()
            )
            moved_anchor_names = [
                name
                for name in moved_anchor_names
                if name in objects and name in translations
            ]
            for moved_name in moved_anchor_names:
                translations[moved_name] = translations[moved_name] + vertical_shift
                _write_obj_with_translation(
                    output_dir / f"scene_canon_{moved_name}.obj",
                    objects[moved_name]["lines"],
                    translations[moved_name],
                )
                record = records[moved_name]
                record["translation_m"] = translations[moved_name].tolist()
                record["output_bounds_min_m"] = (
                    objects[moved_name]["minimum"] + translations[moved_name]
                ).tolist()
                record["output_bounds_max_m"] = (
                    objects[moved_name]["maximum"] + translations[moved_name]
                ).tolist()
            contact_details.update(
                {
                    "target_clearance_m": clearance_m,
                    "requested_child_drop_m": requested_drop,
                    "applied_child_drop_m": applied_drop,
                    "floor_limited": applied_drop < requested_drop,
                    "moved_objects": moved_anchor_names,
                }
            )
            _LOGGER.info(
                "  %s: close local support gap above %s, drop=%.4f m",
                child,
                parent,
                applied_drop,
            )

    report = {
        "schema_version": 1,
        "scene_tree": str(tree_path),
        "input_obj_dir": str(input_dir),
        "output_obj_dir": str(output_dir),
        "parameters": {
            "clearance_m": clearance_m,
            "complex_overlap_m": complex_overlap_m,
            "complex_parent_overlap_fraction": complex_parent_overlap_fraction,
            "partial_support_min_xz_overlap_ratio": partial_support_min_xz_overlap_ratio,
            "partial_support_max_parent_shift_m": partial_support_max_parent_shift_m,
            "partial_support_max_child_drop_m": partial_support_max_child_drop_m,
            "partial_support_contact_quantile": partial_support_contact_quantile,
            "partial_support_surface_normal_y": partial_support_surface_normal_y,
            "ceiling_height_threshold_m": ceiling_height_threshold,
        },
        "reference_scene_y_offset_m": reference_offset,
        "ceiling_height_m": ceiling_height,
        "missing_meshes": sorted(missing_meshes),
        "objects": records,
    }

    if resolved_scene_tree_path is not None:
        resolved_tree = Path(resolved_scene_tree_path).expanduser().resolve()
        for edge in scene_tree.get("edges", []):
            record = records.get(edge.get("child"))
            if record is None:
                continue
            mode = record["placement_mode"]
            edge["support_mode"] = (
                "preserve-relative" if mode.startswith("preserve-relative") else mode
            )
            partial_support = edge.get("relation") == "supported-by" or (
                edge.get("relation") == "on"
                and mode.startswith("preserve-relative")
            )
            if "physics_role" not in edge:
                if partial_support:
                    edge["physics_role"] = "kinematic"
                elif edge.get("type") == "fixed" or edge.get("relation") in {
                    "attach",
                    "hang",
                    "on-attach",
                }:
                    edge["physics_role"] = "fixed"
                else:
                    edge["physics_role"] = "dynamic"
            if "collision_policy" not in edge:
                edge["collision_policy"] = (
                    "kinematic-isolated"
                    if edge["physics_role"] == "kinematic" and partial_support
                    else "default"
                )
            record["support_mode"] = edge["support_mode"]
            record["physics_role"] = edge["physics_role"]
            record["collision_policy"] = edge["collision_policy"]
        resolved_tree.parent.mkdir(parents=True, exist_ok=True)
        with resolved_tree.open("w", encoding="utf-8") as file_obj:
            json.dump(scene_tree, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")
        report["resolved_scene_tree"] = str(resolved_tree)
    if report_path is not None:
        resolved_report = Path(report_path).expanduser().resolve()
        resolved_report.parent.mkdir(parents=True, exist_ok=True)
        with resolved_report.open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")
        _LOGGER.info("  Support diagnostics: %s", resolved_report)

    _LOGGER.info("  Processed %d objects -> %s", len(records), output_dir)
    return float(ceiling_height)

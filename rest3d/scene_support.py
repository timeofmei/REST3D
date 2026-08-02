"""Resolve Stage 2 vertical placement from explicit scene-support semantics."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from rest3d.utils.mesh import read_obj_vertices, write_obj_with_y_offset


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
    children_map: dict[str, list[str]] = defaultdict(list)
    for edge in scene_tree.get("edges", []):
        child = edge["child"]
        parent = edge["parent"]
        if parent.startswith("the_floor"):
            parent = "floor"
        parent_map[child] = parent
        relation_map[child] = edge.get("relation", "on")
        type_map[child] = edge.get("type", "movable")
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
            "vertices": vertices,
            "lines": lines,
            "minimum": minimum,
            "maximum": maximum,
        }

    offsets: dict[str, float] = {}
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
        write_obj_with_y_offset(str(output_path), data["lines"], y_offset)
        offsets[name] = float(y_offset)
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

    report = {
        "schema_version": 1,
        "scene_tree": str(tree_path),
        "input_obj_dir": str(input_dir),
        "output_obj_dir": str(output_dir),
        "parameters": {
            "clearance_m": clearance_m,
            "complex_overlap_m": complex_overlap_m,
            "complex_parent_overlap_fraction": complex_parent_overlap_fraction,
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
            if "physics_role" not in edge:
                partial_support = edge.get("relation") == "supported-by" or (
                    edge.get("relation") == "on"
                    and mode.startswith("preserve-relative")
                )
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
                    "support-lineage-only"
                    if edge["physics_role"] == "kinematic"
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

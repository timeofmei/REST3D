"""Generic rigid-body mass properties derived from reconstructed meshes."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from rest3d.utils.mesh import load_trimesh_any


@dataclass(frozen=True)
class PhysicsAssetPolicy:
    """Geometry-only defaults used when no material metadata is available."""

    nominal_density_kg_m3: float = 700.0
    minimum_mass_kg: float = 0.02
    maximum_mass_kg: float = 100.0
    fallback_solid_fraction: float = 0.15
    # Reconstructed surfaces often describe a thin watertight shell rather than
    # the bulk material of the physical object.  Keep a conservative, generic
    # lower bound so shell volume does not produce near-zero mass and inertia.
    minimum_bounding_box_fill_fraction: float = 0.30
    minimum_extent_m: float = 1.0e-5

    def validate(self) -> None:
        if not np.isfinite(self.nominal_density_kg_m3) or self.nominal_density_kg_m3 <= 0.0:
            raise ValueError("nominal density must be finite and positive")
        if not np.isfinite(self.minimum_mass_kg) or self.minimum_mass_kg <= 0.0:
            raise ValueError("minimum mass must be finite and positive")
        if (
            not np.isfinite(self.maximum_mass_kg)
            or self.maximum_mass_kg < self.minimum_mass_kg
        ):
            raise ValueError("maximum mass must be finite and at least the minimum mass")
        if not 0.0 < self.fallback_solid_fraction <= 1.0:
            raise ValueError("fallback solid fraction must be in (0, 1]")
        if not 0.0 <= self.minimum_bounding_box_fill_fraction <= 1.0:
            raise ValueError("minimum bounding-box fill fraction must be in [0, 1]")
        if not np.isfinite(self.minimum_extent_m) or self.minimum_extent_m <= 0.0:
            raise ValueError("minimum extent must be finite and positive")


@dataclass(frozen=True)
class PhysicsAssetProperties:
    """Mass properties and diagnostics for one mesh in its link frame."""

    source_path: str
    vertex_count: int
    face_count: int
    watertight: bool
    bounds_min_m: Tuple[float, float, float]
    bounds_max_m: Tuple[float, float, float]
    extents_m: Tuple[float, float, float]
    bounding_box_volume_m3: float
    raw_volume_m3: float
    raw_bounding_box_fill_fraction: float
    minimum_bounding_box_fill_fraction: float
    volume_was_floored: bool
    signed_mesh_volume_m3: float
    effective_volume_m3: float
    volume_source: str
    nominal_density_kg_m3: float
    unclamped_mass_kg: float
    mass_kg: float
    mass_was_clamped: bool
    center_of_mass_m: Tuple[float, float, float]
    inertia_kg_m2: Tuple[Tuple[float, float, float], ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _box_inertia(mass: float, extents: np.ndarray) -> np.ndarray:
    x, y, z = extents
    return np.diag(
        [
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        ]
    )


def _regularize_inertia(inertia: np.ndarray, mass: float, extents: np.ndarray) -> np.ndarray:
    inertia = 0.5 * (np.asarray(inertia, dtype=np.float64) + np.asarray(inertia).T)
    if inertia.shape != (3, 3) or not np.isfinite(inertia).all():
        return _box_inertia(mass, extents)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    minimum_moment = max(1.0e-10, mass * float(np.max(extents)) ** 2 * 1.0e-7)
    eigenvalues = np.maximum(eigenvalues, minimum_moment)
    regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    if not np.isfinite(regularized).all():
        return _box_inertia(mass, extents)
    return 0.5 * (regularized + regularized.T)


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    """Indent an ElementTree on both Python 3.8 and newer interpreters."""

    prefix = "\n" + level * "  "
    child_prefix = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_prefix
        for child in element:
            _indent_xml(child, level + 1)
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = prefix
    if level and (not element.tail or not element.tail.strip()):
        element.tail = prefix


def analyze_physics_asset(
    mesh_path: Union[str, Path], policy: PhysicsAssetPolicy = PhysicsAssetPolicy()
) -> PhysicsAssetProperties:
    """Derive generic rigid-body properties without modifying the source mesh."""

    policy.validate()
    path = Path(mesh_path).expanduser().resolve(strict=True)
    mesh = load_trimesh_any(str(path)).copy()
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError("mesh must contain at least four vertices and four faces: %s" % path)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    if not np.isfinite(bounds).all() or np.any(extents < policy.minimum_extent_m):
        raise ValueError("mesh bounds are degenerate or non-finite: %s" % path)

    signed_volume = float(mesh.volume)
    watertight = bool(mesh.is_watertight)
    use_mesh_volume = watertight and np.isfinite(signed_volume) and abs(signed_volume) > 0.0
    if use_mesh_volume:
        if signed_volume < 0.0:
            mesh.invert()
            signed_volume = float(mesh.volume)
        mass_properties = mesh.mass_properties
        reference_volume = float(mass_properties.volume)
        raw_volume = reference_volume
        volume_source = "watertight_mesh"
    else:
        hull = mesh.convex_hull
        mass_properties = hull.mass_properties
        reference_volume = float(mass_properties.volume)
        raw_volume = reference_volume * policy.fallback_solid_fraction
        volume_source = "convex_hull_fraction"
    if not np.isfinite(reference_volume) or reference_volume <= 0.0:
        raise ValueError("mesh has no positive mass-property volume: %s" % path)

    bounding_box_volume = float(np.prod(extents))
    minimum_volume = (
        bounding_box_volume * policy.minimum_bounding_box_fill_fraction
    )
    effective_volume = max(raw_volume, minimum_volume)
    volume_was_floored = effective_volume > raw_volume and not np.isclose(
        effective_volume, raw_volume
    )
    if volume_was_floored:
        volume_source += "_with_bbox_floor"
    unclamped_mass = effective_volume * policy.nominal_density_kg_m3
    mass = float(np.clip(unclamped_mass, policy.minimum_mass_kg, policy.maximum_mass_kg))
    mass_scale = mass / reference_volume
    center_mass = np.asarray(mass_properties.center_mass, dtype=np.float64)
    inertia = np.asarray(mass_properties.inertia, dtype=np.float64) * mass_scale
    inertia = _regularize_inertia(inertia, mass, extents)
    if center_mass.shape != (3,) or not np.isfinite(center_mass).all():
        center_mass = 0.5 * (bounds[0] + bounds[1])

    return PhysicsAssetProperties(
        source_path=str(path),
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        watertight=watertight,
        bounds_min_m=tuple(float(value) for value in bounds[0]),
        bounds_max_m=tuple(float(value) for value in bounds[1]),
        extents_m=tuple(float(value) for value in extents),
        bounding_box_volume_m3=bounding_box_volume,
        raw_volume_m3=float(raw_volume),
        raw_bounding_box_fill_fraction=float(raw_volume / bounding_box_volume),
        minimum_bounding_box_fill_fraction=policy.minimum_bounding_box_fill_fraction,
        volume_was_floored=volume_was_floored,
        signed_mesh_volume_m3=signed_volume,
        effective_volume_m3=float(effective_volume),
        volume_source=volume_source,
        nominal_density_kg_m3=policy.nominal_density_kg_m3,
        unclamped_mass_kg=float(unclamped_mass),
        mass_kg=mass,
        mass_was_clamped=not np.isclose(mass, unclamped_mass),
        center_of_mass_m=tuple(float(value) for value in center_mass),
        inertia_kg_m2=tuple(
            tuple(float(value) for value in row) for row in inertia
        ),
    )


def write_physics_urdf(
    output_path: Union[str, Path],
    *,
    robot_name: str,
    mesh_path: Union[str, Path],
    properties: PhysicsAssetProperties,
) -> Path:
    """Write a derived URDF with explicit mesh-based properties, refusing overwrite."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite derived URDF: %s" % output)
    source_mesh = Path(mesh_path).expanduser().resolve(strict=True)
    mesh_reference = os.path.relpath(str(source_mesh), str(output.parent))

    robot = ET.Element("robot", {"name": robot_name})
    link = ET.SubElement(robot, "link", {"name": "base"})
    for section_name in ("visual", "collision"):
        section = ET.SubElement(link, section_name)
        geometry = ET.SubElement(section, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {"filename": mesh_reference, "scale": "1 1 1"},
        )
    inertial = ET.SubElement(link, "inertial")
    center = properties.center_of_mass_m
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": "%.12g %.12g %.12g" % center, "rpy": "0 0 0"},
    )
    ET.SubElement(inertial, "mass", {"value": "%.12g" % properties.mass_kg})
    inertia = properties.inertia_kg_m2
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "%.12g" % inertia[0][0],
            "ixy": "%.12g" % inertia[0][1],
            "ixz": "%.12g" % inertia[0][2],
            "iyy": "%.12g" % inertia[1][1],
            "iyz": "%.12g" % inertia[1][2],
            "izz": "%.12g" % inertia[2][2],
        },
    )
    _indent_xml(robot)
    tree = ET.ElementTree(robot)
    with output.open("xb") as file:
        tree.write(file, encoding="utf-8", xml_declaration=True)
    return output

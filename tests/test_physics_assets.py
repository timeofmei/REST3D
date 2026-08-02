import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from rest3d.sim.physics_assets import (
    PhysicsAssetPolicy,
    analyze_physics_asset,
    sanitize_urdf_name,
    write_physics_urdf,
)


class PhysicsAssetTest(unittest.TestCase):
    def test_watertight_box_has_analytic_mass_center_and_inertia(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.obj"
            trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(path)
            properties = analyze_physics_asset(
                path,
                PhysicsAssetPolicy(
                    nominal_density_kg_m3=10.0,
                    minimum_mass_kg=0.01,
                    maximum_mass_kg=1000.0,
                ),
            )

        self.assertEqual(properties.volume_source, "watertight_mesh")
        self.assertAlmostEqual(properties.effective_volume_m3, 6.0)
        self.assertAlmostEqual(properties.mass_kg, 60.0)
        np.testing.assert_allclose(properties.center_of_mass_m, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(
            properties.inertia_kg_m2,
            np.diag([65.0, 50.0, 25.0]),
            atol=1e-10,
        )

    def test_open_mesh_uses_recorded_convex_hull_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.obj"
            mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            mesh.update_faces(np.arange(len(mesh.faces) - 1))
            mesh.export(path)
            properties = analyze_physics_asset(
                path,
                PhysicsAssetPolicy(
                    nominal_density_kg_m3=20.0,
                    minimum_mass_kg=0.01,
                    maximum_mass_kg=1000.0,
                    fallback_solid_fraction=0.25,
                    minimum_bounding_box_fill_fraction=0.0,
                ),
            )

        self.assertFalse(properties.watertight)
        self.assertEqual(properties.volume_source, "convex_hull_fraction")
        self.assertAlmostEqual(properties.effective_volume_m3, 0.25)
        self.assertAlmostEqual(properties.mass_kg, 5.0)

    def test_mass_clamp_scales_inertia_and_derived_urdf_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "small.obj"
            trimesh.creation.box(extents=(0.01, 0.02, 0.03)).export(mesh_path)
            properties = analyze_physics_asset(
                mesh_path,
                PhysicsAssetPolicy(minimum_mass_kg=0.5, maximum_mass_kg=1.0),
            )
            urdf_path = write_physics_urdf(
                root / "derived" / "small.urdf",
                robot_name="generic_body",
                mesh_path=mesh_path,
                properties=properties,
            )
            root_xml = ET.parse(urdf_path).getroot()

            self.assertTrue(properties.mass_was_clamped)
            self.assertAlmostEqual(properties.mass_kg, 0.5)
            self.assertEqual(root_xml.find("./link/inertial/mass").attrib["value"], "0.5")
            inertia = np.asarray(properties.inertia_kg_m2)
            self.assertTrue(np.all(np.linalg.eigvalsh(inertia) > 0.0))
            with self.assertRaises(FileExistsError):
                write_physics_urdf(
                    urdf_path,
                    robot_name="generic_body",
                    mesh_path=mesh_path,
                    properties=properties,
                )

    def test_thin_watertight_shell_uses_generic_bounding_box_volume_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shell.obj"
            outer = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            inner = trimesh.creation.box(extents=(0.99, 0.99, 0.99))
            inner.invert()
            trimesh.util.concatenate((outer, inner)).export(path)
            properties = analyze_physics_asset(
                path,
                PhysicsAssetPolicy(
                    nominal_density_kg_m3=10.0,
                    minimum_mass_kg=0.01,
                    maximum_mass_kg=1000.0,
                    minimum_bounding_box_fill_fraction=0.15,
                ),
            )

        self.assertTrue(properties.watertight)
        self.assertTrue(properties.volume_was_floored)
        self.assertEqual(properties.volume_source, "watertight_mesh_with_bbox_floor")
        self.assertAlmostEqual(properties.effective_volume_m3, 0.15)
        self.assertAlmostEqual(properties.mass_kg, 1.5)


    def test_hyphenated_names_are_sanitized_and_mesh_copy_is_referenced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "scene_canon_black_U-shaped_object_000.obj"
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
            properties = analyze_physics_asset(
                mesh_path,
                PhysicsAssetPolicy(minimum_mass_kg=0.01, maximum_mass_kg=1000.0),
            )
            urdf_path = write_physics_urdf(
                root / "derived" / "black_U-shaped_object_000.urdf",
                robot_name="black_U-shaped_object_000",
                mesh_path=mesh_path,
                properties=properties,
            )
            root_xml = ET.parse(urdf_path).getroot()
            mesh_copy = root / "derived" / "meshes" / "scene_canon_black_U_shaped_object_000.obj"

            self.assertEqual(sanitize_urdf_name("black_U-shaped_object_000"), "black_U_shaped_object_000")
            self.assertEqual(root_xml.attrib["name"], "black_U_shaped_object_000")
            self.assertTrue(mesh_copy.is_file())
            self.assertEqual(
                root_xml.find("./link/visual/geometry/mesh").attrib["filename"],
                os.path.relpath(str(mesh_copy), str(urdf_path.parent)),
            )


if __name__ == "__main__":
    unittest.main()

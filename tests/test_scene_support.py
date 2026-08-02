import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from rest3d.scene_support import place_to_ground


class SceneSupportTest(unittest.TestCase):
    def _write_obj(
        self,
        directory: Path,
        name: str,
        ys: tuple[float, ...],
        *,
        x_offset: float = 0.0,
    ) -> None:
        vertices = "".join(
            f"v {index % 2 + x_offset} {y} {index // 2}\n"
            for index, y in enumerate(ys)
        )
        (directory / f"scene_y_align_{name}.obj").write_text(
            vertices + "f 1 2 3\n", encoding="utf-8"
        )

    def _run(
        self,
        relation: str,
        child_ys: tuple[float, ...],
        *,
        child_x_offset: float = 0.0,
        child_physics_role: str | None = None,
    ):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        self._write_obj(source, "support", (-1.0, -1.0, 0.0, 0.0))
        self._write_obj(source, "child", child_ys, x_offset=child_x_offset)
        tree = root / "scene_tree.json"
        child_edge = {
            "child": "child",
            "parent": "support",
            "relation": relation,
            "type": "movable",
        }
        if child_physics_role is not None:
            child_edge["physics_role"] = child_physics_role
        tree.write_text(
            json.dumps(
                {
                    "roots": ["floor", "wall", "ceiling", "floor-wall"],
                    "nodes": ["support", "child"],
                    "edges": [
                        {
                            "child": "support",
                            "parent": "floor",
                            "relation": "on",
                            "type": "movable",
                        },
                        child_edge,
                    ],
                }
            ),
            encoding="utf-8",
        )
        report = root / "report.json"
        resolved_tree = root / "resolved_scene_tree.json"
        place_to_ground(
            source,
            output,
            tree,
            report_path=report,
            resolved_scene_tree_path=resolved_tree,
        )
        return (
            temporary,
            json.loads(report.read_text(encoding="utf-8")),
            json.loads(resolved_tree.read_text(encoding="utf-8")),
        )

    def _write_mesh(self, directory: Path, name: str, mesh: trimesh.Trimesh) -> None:
        mesh.export(directory / f"scene_y_align_{name}.obj")

    def test_simple_on_relation_snaps_child_to_parent_top(self):
        temporary, report, tree = self._run("on", (-0.02, -0.02, 0.2, 0.2))
        with temporary:
            child = report["objects"]["child"]
            self.assertEqual(child["placement_mode"], "top-surface")
            self.assertAlmostEqual(child["output_bounds_min_m"][1], 1.005)
            self.assertEqual(tree["edges"][1]["physics_role"], "dynamic")

    def test_deep_legacy_on_overlap_preserves_relative_pose(self):
        temporary, report, tree = self._run("on", (-0.7, -0.7, 0.3, 0.3))
        with temporary:
            child = report["objects"]["child"]
            self.assertEqual(child["placement_mode"], "preserve-relative-auto")
            self.assertAlmostEqual(child["output_bounds_min_m"][1], 0.3)
            self.assertAlmostEqual(
                child["relative_min_y_before_m"], child["relative_min_y_after_m"]
            )
            self.assertEqual(tree["edges"][1]["support_mode"], "preserve-relative")
            self.assertEqual(tree["edges"][1]["physics_role"], "kinematic")
            self.assertEqual(
                tree["edges"][1]["collision_policy"], "kinematic-isolated"
            )

    def test_supported_by_explicitly_preserves_relative_pose(self):
        temporary, report, tree = self._run(
            "supported-by", (-0.02, -0.02, 0.2, 0.2)
        )
        with temporary:
            child = report["objects"]["child"]
            self.assertEqual(child["placement_mode"], "preserve-relative-explicit")
            self.assertAlmostEqual(child["output_bounds_min_m"][1], 0.98)
            self.assertFalse(child["rotation_changed"])
            self.assertEqual(tree["edges"][1]["physics_role"], "kinematic")
            self.assertEqual(
                tree["edges"][1]["collision_policy"], "kinematic-isolated"
            )

    def test_inside_preserves_height_without_forcing_kinematic_role(self):
        temporary, report, tree = self._run(
            "inside", (-0.7, -0.7, 0.3, 0.3)
        )
        with temporary:
            child = report["objects"]["child"]
            self.assertEqual(child["placement_mode"], "preserve-relative-explicit")
            self.assertEqual(tree["edges"][1]["physics_role"], "dynamic")
            self.assertEqual(tree["edges"][1]["collision_policy"], "default")

    def test_partial_support_moves_parent_under_world_space_anchor(self):
        temporary, report, _ = self._run(
            "supported-by",
            (-0.7, -0.7, 0.3, 0.3),
            child_x_offset=0.6,
        )
        with temporary:
            support = report["objects"]["support"]
            child = report["objects"]["child"]
            alignment = support["partial_support_alignment"]
            self.assertAlmostEqual(alignment["horizontal_shift_m"][0], 0.1)
            self.assertEqual(alignment["moved_objects"], ["support"])
            self.assertGreater(
                alignment["xz_overlap_ratio_after"]["child"],
                alignment["xz_overlap_ratio_before"]["child"],
            )
            self.assertEqual(child["translation_m"][0], 0.0)

    def test_explicit_dynamic_partial_support_is_not_a_world_space_anchor(self):
        temporary, report, tree = self._run(
            "supported-by",
            (-0.7, -0.7, 0.3, 0.3),
            child_x_offset=0.6,
            child_physics_role="dynamic",
        )
        with temporary:
            self.assertNotIn(
                "partial_support_alignment", report["objects"]["support"]
            )
            self.assertEqual(tree["edges"][1]["physics_role"], "dynamic")
            self.assertEqual(tree["edges"][1]["collision_policy"], "default")

    def test_partial_support_closes_local_surface_gap_not_global_extrema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()

            seat = trimesh.creation.box(extents=(0.2, 0.05, 0.2))
            seat.apply_translation((0.0, -0.225, 0.0))
            leg = trimesh.creation.box(extents=(0.02, 1.0, 0.02))
            leg.apply_translation((0.18, -0.5, 0.18))
            self._write_mesh(
                source, "support", trimesh.util.concatenate((seat, leg))
            )

            seated_body = trimesh.creation.box(extents=(0.2, 0.4, 0.2))
            seated_body.apply_translation((0.0, 0.1, 0.0))
            foot = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
            foot.apply_translation((0.0, -0.675, 0.0))
            self._write_mesh(
                source, "child", trimesh.util.concatenate((seated_body, foot))
            )

            tree = root / "scene_tree.json"
            tree.write_text(
                json.dumps(
                    {
                        "roots": ["floor", "wall", "ceiling", "floor-wall"],
                        "nodes": ["support", "child"],
                        "edges": [
                            {
                                "child": "support",
                                "parent": "floor",
                                "relation": "on",
                                "type": "movable",
                            },
                            {
                                "child": "child",
                                "parent": "support",
                                "relation": "on",
                                "type": "movable",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report_path = root / "report.json"
            place_to_ground(source, output, tree, report_path=report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

            child = report["objects"]["child"]
            contact = child["partial_support_contact"]
            self.assertEqual(
                contact["method"], "local-horizontal-face-correspondence"
            )
            self.assertAlmostEqual(contact["estimated_gap_m"], 0.1, places=6)
            self.assertAlmostEqual(
                contact["applied_child_drop_m"], 0.095, places=6
            )
            self.assertAlmostEqual(child["output_bounds_min_m"][1], 0.205, places=6)
            self.assertTrue(np.isclose(child["translation_m"][1], 0.905))


if __name__ == "__main__":
    unittest.main()

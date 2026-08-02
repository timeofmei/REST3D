import json
import tempfile
import unittest
from pathlib import Path

from rest3d.scene_support import place_to_ground


class SceneSupportTest(unittest.TestCase):
    def _write_obj(self, directory: Path, name: str, ys: tuple[float, ...]) -> None:
        vertices = "".join(
            f"v {index % 2} {y} {index // 2}\n" for index, y in enumerate(ys)
        )
        (directory / f"scene_y_align_{name}.obj").write_text(
            vertices + "f 1 2 3\n", encoding="utf-8"
        )

    def _run(self, relation: str, child_ys: tuple[float, ...]):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        self._write_obj(source, "support", (-1.0, -1.0, 0.0, 0.0))
        self._write_obj(source, "child", child_ys)
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
                            "relation": relation,
                            "type": "movable",
                        },
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

    def test_inside_preserves_height_without_forcing_kinematic_role(self):
        temporary, report, tree = self._run(
            "inside", (-0.7, -0.7, 0.3, 0.3)
        )
        with temporary:
            child = report["objects"]["child"]
            self.assertEqual(child["placement_mode"], "preserve-relative-explicit")
            self.assertEqual(tree["edges"][1]["physics_role"], "dynamic")
            self.assertEqual(tree["edges"][1]["collision_policy"], "default")


if __name__ == "__main__":
    unittest.main()

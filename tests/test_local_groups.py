import json
import tempfile
import unittest
from pathlib import Path

from rest3d.sim.local_groups import ObjectBounds, build_local_group_plan


class LocalGroupPlanTest(unittest.TestCase):
    def test_descendants_and_nearby_context_are_included_without_name_rules(self):
        tree = {
            "roots": ["world_root"],
            "nodes": ["support_a", "child_a", "nested_a", "nearby_b", "far_c"],
            "edges": [
                {"child": "support_a", "parent": "world_root", "relation": "on", "type": "movable"},
                {"child": "child_a", "parent": "support_a", "relation": "on", "type": "movable"},
                {"child": "nested_a", "parent": "child_a", "relation": "on", "type": "movable"},
                {"child": "nearby_b", "parent": "world_root", "relation": "on", "type": "movable"},
                {"child": "far_c", "parent": "world_root", "relation": "on", "type": "movable"},
            ],
        }
        names = ("support_a", "child_a", "nested_a", "nearby_b", "far_c")
        bounds = {
            "support_a": ObjectBounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "child_a": ObjectBounds((0.2, 1.0, 0.2), (0.8, 1.4, 0.8)),
            "nested_a": ObjectBounds((0.3, 1.4, 0.3), (0.7, 1.6, 0.7)),
            "nearby_b": ObjectBounds((1.05, 0.0, 0.0), (1.5, 1.0, 1.0)),
            "far_c": ObjectBounds((4.0, 0.0, 0.0), (5.0, 1.0, 1.0)),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.json"
            path.write_text(json.dumps(tree), encoding="utf-8")
            plan = build_local_group_plan(
                path,
                scene_names=names,
                fixed_names=(),
                object_bounds=bounds,
                interaction_margin_m=0.1,
            )

        self.assertEqual(len(plan.groups), 2)
        outer = plan.groups[0]
        self.assertEqual(outer.root_name, "support_a")
        self.assertEqual(outer.direct_child_names, ("child_a",))
        self.assertEqual(outer.member_names, ("support_a", "child_a", "nested_a"))
        self.assertEqual(outer.driven_descendants, (("nested_a", "child_a"),))
        self.assertEqual(outer.context_names, ("nearby_b",))
        inner = plan.groups[1]
        self.assertEqual(inner.root_name, "child_a")
        self.assertEqual(inner.member_names, ("child_a", "nested_a"))
        self.assertIn("support_a", inner.context_names)

    def test_bounds_must_exactly_match_scene_objects(self):
        tree = {
            "roots": ["world_root"],
            "nodes": ["body_a"],
            "edges": [
                {"child": "body_a", "parent": "world_root", "relation": "on", "type": "movable"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.json"
            path.write_text(json.dumps(tree), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_local_group_plan(
                    path,
                    scene_names=("body_a",),
                    fixed_names=(),
                    object_bounds={},
                )


if __name__ == "__main__":
    unittest.main()

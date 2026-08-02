import json
import tempfile
import unittest
from pathlib import Path

from rest3d.sim.local_results import (
    legacy_local_group_payload,
    load_scene_states,
    local_group_execution_order,
    make_initial_scene_states,
    merge_group_candidate_states,
)


class LocalResultsTest(unittest.TestCase):
    def test_load_scene_states_validates_names_and_clears_velocities(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "states.json"
            states = make_initial_scene_states(["first", "second"], base_dy=0.0)
            states["first"][7:13] = [1, 2, 3, 4, 5, 6]
            path.write_text(json.dumps({"states_rest_wxyz": states}), encoding="utf-8")
            loaded = load_scene_states(path, ["first", "second"])
            self.assertEqual(loaded["first"][7:13], [0.0] * 6)
            preserved = load_scene_states(
                path, ["first", "second"], zero_velocities=False
            )
            self.assertEqual(preserved["first"][7:13], [1, 2, 3, 4, 5, 6])

    def test_nested_groups_run_child_before_parent(self):
        groups = [
            {
                "index": 4,
                "root_name": "outer_support",
                "member_names": ["outer_support", "inner_support", "item"],
            },
            {
                "index": 2,
                "root_name": "inner_support",
                "member_names": ["inner_support", "item"],
            },
        ]
        self.assertEqual(local_group_execution_order(groups), [2, 4])

    def test_merge_changes_members_only_and_exports_xyzw(self):
        states = make_initial_scene_states(
            ["support", "item", "context"], base_dy=0.002
        )
        candidate = {
            "settled_states_rest_wxyz": {
                "support": [1, 2, 3, 0.5, 0.1, 0.2, 0.3, 4, 5, 6, 0, 0, 0],
                "item": [7, 8, 9, 0.6, 0.4, 0.3, 0.2, 1, 2, 3, 0, 0, 0],
                "context": [9] * 13,
            }
        }
        merged = merge_group_candidate_states(
            states, candidate, ["support", "item"]
        )
        self.assertEqual(merged["context"], states["context"])
        group = {
            "root_name": "support",
            "direct_child_names": ["item"],
        }
        payload = legacy_local_group_payload(group, merged, base_dy=0.002)
        self.assertEqual(payload["objects"]["support"]["rot"], [0.1, 0.2, 0.3, 0.5])
        self.assertEqual(payload["objects"]["item"]["lin_vel"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()

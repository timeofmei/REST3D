import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from rest3d.sim.global_cem import (
    GlobalEntityPlan,
    build_global_entity_plan,
    evaluate_convex_hull_intersections_wxyz,
    evaluate_global_cem_energy,
    expand_global_root_samples_wxyz,
    validate_global_object_sets,
)


class GlobalCEMTest(unittest.TestCase):
    SCENE_NAMES = (
        "group_root",
        "child_one",
        "child_two",
        "nested_child",
        "independent",
    )

    def _write_tree(self, root: Path) -> Path:
        path = root / "scene_tree.json"
        path.write_text(
            json.dumps(
                {
                    "roots": ["ground"],
                    "nodes": list(self.SCENE_NAMES),
                    "edges": [
                        {
                            "child": "group_root",
                            "parent": "ground",
                            "relation": "on",
                            "type": "movable",
                        },
                        {
                            "child": "child_one",
                            "parent": "group_root",
                            "relation": "on",
                            "type": "movable",
                        },
                        {
                            "child": "child_two",
                            "parent": "group_root",
                            "relation": "on",
                            "type": "movable",
                        },
                        {
                            "child": "nested_child",
                            "parent": "child_one",
                            "relation": "on",
                            "type": "movable",
                        },
                        {
                            "child": "independent",
                            "parent": "ground",
                            "relation": "on",
                            "type": "movable",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _reference_states(self) -> torch.Tensor:
        states = torch.zeros(len(self.SCENE_NAMES), 13, dtype=torch.float64)
        states[:, 3] = 1.0
        states[:, :3] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=torch.float64,
        )
        return states

    def test_only_roots_are_sampled_but_all_descendants_are_expanded(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_global_entity_plan(
                self._write_tree(Path(temporary)),
                scene_names=self.SCENE_NAMES,
            )

        self.assertEqual(plan.sampled_entity_names, ("group_root", "independent"))
        self.assertEqual(
            plan.entities[0].member_names,
            ("group_root", "child_one", "child_two", "nested_child"),
        )
        self.assertEqual(plan.entities[1].member_names, ("independent",))
        self.assertEqual(GlobalEntityPlan.from_dict(plan.to_dict()), plan)
        invalid_payload = plan.to_dict()
        invalid_payload["owner_entity_indices"][
            self.SCENE_NAMES.index("nested_child")
        ] = -1
        invalid_payload["static_names"].append("nested_child")
        with self.assertRaisesRegex(ValueError, "owned and static"):
            GlobalEntityPlan.from_dict(invalid_payload)

        samples = torch.zeros(2, 2, 6, dtype=torch.float64)
        samples[1, 0, :3] = torch.tensor([0.0, 0.0, 1.0])
        samples[1, 0, 5] = math.pi / 2.0
        samples[1, 1, 0] = 0.5
        expanded = expand_global_root_samples_wxyz(
            self._reference_states(), plan, samples
        )

        torch.testing.assert_close(expanded[0], self._reference_states())
        torch.testing.assert_close(
            expanded[1, :, :3],
            torch.tensor(
                [
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 0.0, 3.0],
                ],
                dtype=torch.float64,
            ),
        )
        expected_group_quaternion = torch.tensor(
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=torch.float64
        )
        torch.testing.assert_close(
            expanded[1, :4, 3:7],
            expected_group_quaternion.unsqueeze(0).expand(4, -1),
        )
        torch.testing.assert_close(
            expanded[1, 4, 3:7],
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        self.assertEqual(tuple(expanded.shape), (2, len(self.SCENE_NAMES), 13))
        self.assertTrue(
            torch.equal(
                expanded[..., 7:13], torch.zeros_like(expanded[..., 7:13])
            )
        )

    def test_every_object_is_required_by_simulation_and_energy(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_global_entity_plan(
                self._write_tree(Path(temporary)),
                scene_names=self.SCENE_NAMES,
            )
        validate_global_object_sets(
            plan,
            simulated_names=self.SCENE_NAMES,
            evaluated_names=self.SCENE_NAMES,
        )
        with self.assertRaisesRegex(ValueError, "simulated object set"):
            validate_global_object_sets(
                plan,
                simulated_names=self.SCENE_NAMES[:-1],
                evaluated_names=self.SCENE_NAMES,
            )
        with self.assertRaisesRegex(ValueError, "evaluated object set"):
            validate_global_object_sets(
                plan,
                simulated_names=self.SCENE_NAMES,
                evaluated_names=self.SCENE_NAMES[:-1],
            )

        reference = self._reference_states()
        placed = reference.unsqueeze(0).clone()
        early = placed.clone()
        settled = placed.clone()
        baseline = evaluate_global_cem_energy(
            plan,
            self.SCENE_NAMES,
            placed,
            early,
            settled,
            reference[:, :7],
        )
        settled[:, self.SCENE_NAMES.index("nested_child"), 0] += 0.25
        disturbed = evaluate_global_cem_energy(
            plan,
            self.SCENE_NAMES,
            placed,
            early,
            settled,
            reference[:, :7],
        )
        self.assertEqual(float(baseline["energy"].item()), 0.0)
        self.assertGreater(
            float(disturbed["energy"].item()),
            float(baseline["energy"].item()),
        )
        nested_index = self.SCENE_NAMES.index("nested_child")
        self.assertGreater(
            float(disturbed["per_object"]["energy"][0, nested_index].item()),
            0.0,
        )
        unaffected = disturbed["per_object"]["energy"][0].clone()
        unaffected[nested_index] = 0.0
        self.assertEqual(float(unaffected.sum().item()), 0.0)

    def test_static_objects_remain_in_every_candidate_without_being_sampled(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_global_entity_plan(
                self._write_tree(Path(temporary)),
                scene_names=self.SCENE_NAMES,
                fixed_names=("independent",),
            )
        self.assertEqual(plan.sampled_entity_names, ("group_root",))
        self.assertEqual(plan.static_names, ("independent",))
        samples = torch.zeros(2, 1, 6, dtype=torch.float64)
        samples[1, 0, :3] = torch.tensor([0.25, -0.5, 0.75])
        expanded = expand_global_root_samples_wxyz(
            self._reference_states(), plan, samples
        )
        independent_index = self.SCENE_NAMES.index("independent")
        torch.testing.assert_close(
            expanded[:, independent_index],
            self._reference_states()[independent_index].unsqueeze(0).expand(2, -1),
        )
        self.assertEqual(tuple(expanded.shape), (2, len(self.SCENE_NAMES), 13))

    def test_pairwise_gjk_counts_every_object_without_double_counting(self):
        cube = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float64,
        )
        states = torch.zeros(2, 3, 13, dtype=torch.float64)
        states[..., 3] = 1.0
        states[:, 1, 0] = torch.tensor([0.25, 3.0], dtype=torch.float64)
        states[:, 2, 0] = 5.0
        intersections = evaluate_convex_hull_intersections_wxyz(
            states, [cube, cube.clone(), cube.clone()]
        )

        torch.testing.assert_close(
            intersections["total"], torch.tensor([1.0, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            intersections["per_object"],
            torch.tensor(
                [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
            ),
        )
        torch.testing.assert_close(
            intersections["per_object"].sum(dim=1), intersections["total"]
        )
        self.assertEqual(tuple(intersections["pair_matrix"].shape), (2, 3, 3))
        self.assertEqual(float(intersections["pair_matrix"][0, 0, 1].item()), 1.0)
        self.assertEqual(float(intersections["pair_matrix"][0, 1, 0].item()), 1.0)


if __name__ == "__main__":
    unittest.main()

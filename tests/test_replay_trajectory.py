import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rest3d.sim.replay_trajectory import (
    load_replay_trajectory,
    trajectory_world_bounds,
)


class ReplayTrajectoryTest(unittest.TestCase):
    def _write_replay(self, root: Path, *, state_shape=(3, 2, 13)) -> Path:
        scene_dir = root / "stage2" / "scene_canon"
        obj_dir = scene_dir / "obj_files"
        urdf_dir = scene_dir / "urdf_files"
        replay_dir = root / "replay"
        obj_dir.mkdir(parents=True)
        urdf_dir.mkdir()
        replay_dir.mkdir()
        tree_path = root / "stage2" / "scene_tree.json"
        tree_path.write_text(
            json.dumps(
                {
                    "roots": ["alpha"],
                    "nodes": ["beta"],
                    "edges": [
                        {
                            "child": "beta",
                            "parent": "alpha",
                            "relation": "on",
                            "type": "movable",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        urdf = "<robot name='asset'><link name='base'/></robot>\n"
        for name in ("alpha", "beta"):
            (obj_dir / f"scene_canon_{name}.obj").write_text(obj, encoding="utf-8")
            (urdf_dir / f"scene_canon_{name}.urdf").write_text(
                urdf, encoding="utf-8"
            )

        results = {
            "passed": True,
            "scene": {
                "scene_tree": str(tree_path),
                "scene_dir": str(scene_dir),
                "urdf_dir": str(urdf_dir),
                "object_names": ["alpha", "beta"],
            },
            "simulation": {"dt_seconds": 0.02},
            "stability": {"evaluation_step": 2, "scene_stable": True},
        }
        (replay_dir / "replay_results.json").write_text(
            json.dumps(results), encoding="utf-8"
        )
        states = np.zeros(state_shape, dtype=np.float32)
        if len(state_shape) == 3 and state_shape[-1] >= 7:
            states[..., 3] = 2.0
        np.save(replay_dir / "replay_states_rest.npy", states)
        return replay_dir

    def test_loads_common_scene_and_normalizes_wxyz_states(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = self._write_replay(Path(directory))
            trajectory = load_replay_trajectory(replay_dir)

        self.assertEqual(trajectory.names, ("alpha", "beta"))
        self.assertEqual(trajectory.frame_count, 3)
        self.assertEqual(trajectory.dt_seconds, 0.02)
        np.testing.assert_allclose(trajectory.states_rest[..., 3], 1.0)
        np.testing.assert_allclose(trajectory.states_rest[..., 4:7], 0.0)

    def test_invalid_state_shape_is_rejected_before_viewer_start(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = self._write_replay(
                Path(directory), state_shape=(3, 2, 12)
            )
            with self.assertRaisesRegex(ValueError, "frames, objects, 13"):
                load_replay_trajectory(replay_dir)

    def test_transformed_bounds_cover_every_replay_frame(self):
        states = np.zeros((2, 1, 13), dtype=np.float64)
        states[0, 0, 3] = 1.0
        states[1, 0, :3] = [10.0, 0.0, 0.0]
        states[1, 0, 5] = 1.0  # 180 degrees around REST3D +Y.
        bounds = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]])

        minimum, maximum = trajectory_world_bounds(states, bounds)

        np.testing.assert_allclose(minimum, [0.0, 0.0, -3.0], atol=1.0e-12)
        np.testing.assert_allclose(maximum, [10.0, 2.0, 3.0], atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()

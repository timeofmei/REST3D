import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rest3d.sim.replay_scene import (
    lab_states_to_rest,
    load_replay_scene,
    rest_poses_to_lab,
    rest_states_to_lab,
)


class ReplaySceneTest(unittest.TestCase):
    def _write_scene(self, root: Path) -> tuple[Path, Path]:
        scene_dir = root / "scene"
        obj_dir = scene_dir / "obj_files"
        urdf_dir = scene_dir / "urdf_files"
        obj_dir.mkdir(parents=True)
        urdf_dir.mkdir()
        tree_path = root / "scene_tree.json"
        tree_path.write_text(
            json.dumps(
                {
                    "roots": ["ground"],
                    "nodes": ["movable_asset", "fixed_asset"],
                    "edges": [
                        {
                            "child": "movable_asset",
                            "parent": "ground",
                            "relation": "on",
                            "type": "movable",
                        },
                        {
                            "child": "fixed_asset",
                            "parent": "ground",
                            "relation": "attach",
                            "type": "fixed",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        urdf = "<robot name='asset'><link name='base'/></robot>\n"
        for name in ("movable_asset", "fixed_asset"):
            (obj_dir / f"{name}.obj").write_text(obj, encoding="utf-8")
            (urdf_dir / f"{name}.urdf").write_text(urdf, encoding="utf-8")
        return tree_path, scene_dir

    def test_scene_assets_are_strict_and_ordered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tree_path, scene_dir = self._write_scene(Path(temp_dir))
            scene = load_replay_scene(tree_path, scene_dir)
            self.assertEqual(scene.names, ("fixed_asset", "movable_asset"))
            self.assertEqual(scene.fixed_names, ("fixed_asset",))
            self.assertEqual(scene.movable_names, ("movable_asset",))

    def test_missing_urdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tree_path, scene_dir = self._write_scene(Path(temp_dir))
            (scene_dir / "urdf_files" / "movable_asset.urdf").unlink()
            with self.assertRaises(FileNotFoundError):
                load_replay_scene(tree_path, scene_dir)

    def test_undeclared_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tree_path, scene_dir = self._write_scene(Path(temp_dir))
            obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            (scene_dir / "obj_files" / "undeclared.obj").write_text(obj, encoding="utf-8")
            (scene_dir / "urdf_files" / "undeclared.urdf").write_text(
                "<robot name='undeclared'><link name='base'/></robot>\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_replay_scene(tree_path, scene_dir)

    def test_common_prefix_and_optional_extra_urdf_are_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_dir = root / "scene"
            obj_dir = scene_dir / "obj_files"
            urdf_dir = scene_dir / "urdf_files"
            obj_dir.mkdir(parents=True)
            urdf_dir.mkdir()
            tree_path = root / "scene_tree.json"
            tree_path.write_text(
                json.dumps(
                    {
                        "roots": ["support"],
                        "nodes": ["item"],
                        "edges": [
                            {
                                "child": "item",
                                "parent": "support",
                                "relation": "on",
                                "type": "movable",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (obj_dir / "reconstruction_item.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
            (urdf_dir / "reconstruction_item.urdf").write_text(
                "<robot/>", encoding="utf-8"
            )
            (urdf_dir / "support.urdf").write_text("<robot/>", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no matching OBJ"):
                load_replay_scene(tree_path, scene_dir)
            scene = load_replay_scene(
                tree_path, scene_dir, allow_extra_urdf=True
            )

            self.assertEqual(scene.names, ("item",))
            self.assertEqual(scene.asset_prefix, "reconstruction_")
            self.assertEqual(scene.objects[0].obj_path.name, "reconstruction_item.obj")

    def test_y_up_z_up_pose_and_state_round_trip(self):
        poses_rest = np.array(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64
        )
        poses_lab = rest_poses_to_lab(poses_rest)
        np.testing.assert_allclose(poses_lab[0, :3], [1.0, -3.0, 2.0], atol=1e-12)

        states_lab = np.zeros((1, 13), dtype=np.float64)
        states_lab[:, :7] = poses_lab
        states_lab[:, 7:10] = [1.0, -3.0, 2.0]
        states_lab[:, 10:13] = [-4.0, -6.0, 5.0]
        states_rest = lab_states_to_rest(states_lab)
        np.testing.assert_allclose(states_rest[0, :7], poses_rest[0], atol=1e-12)
        np.testing.assert_allclose(states_rest[0, 7:10], [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(states_rest[0, 10:13], [-4.0, 5.0, 6.0], atol=1e-12)
        np.testing.assert_allclose(rest_states_to_lab(states_rest), states_lab, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

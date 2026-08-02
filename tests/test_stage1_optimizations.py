import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from rest3d.utils import vlm
from scripts import infer_scenetree as stage1


class VlmImageEncodingTests(unittest.TestCase):
    def tearDown(self):
        vlm.set_vlm_image_max_edge(1536)
        vlm.get_vlm_stats(reset=True)

    def test_path_images_are_resized_and_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "large.png"
            Image.new("RGB", (800, 400), "navy").save(image_path)

            vlm.set_vlm_image_max_edge(400)
            vlm.get_vlm_stats(reset=True)
            first = vlm.image_to_png_bytes(image_path)
            second = vlm.image_to_png_bytes(image_path)

            with tempfile.NamedTemporaryFile(suffix=".png") as encoded:
                encoded.write(first)
                encoded.flush()
                with Image.open(encoded.name) as resized:
                    self.assertEqual(resized.size, (400, 200))
            self.assertEqual(first, second)
            stats = vlm.get_vlm_stats()
            self.assertEqual(stats["file_cache_misses"], 1)
            self.assertEqual(stats["file_cache_hits"], 1)

    def test_zero_max_edge_preserves_original_resolution(self):
        image = Image.new("RGB", (640, 320), "white")
        vlm.set_vlm_image_max_edge(0)
        encoded = vlm.image_to_png_bytes(image)
        with tempfile.NamedTemporaryFile(suffix=".png") as output:
            output.write(encoded)
            output.flush()
            with Image.open(output.name) as decoded:
                self.assertEqual(decoded.size, (640, 320))


class SceneTreeBatchTests(unittest.TestCase):
    def test_supported_by_alias_gets_kinematic_physics_role(self):
        objects = ["posed_body", "support"]
        parents = ["floor", *objects]
        response = json.dumps([{
            "child": "posed_body",
            "parent": "support",
            "relation": "supported_by",
            "type": "movable",
        }])
        edge = stage1._parse_scene_tree_batch_response(
            response, objects, parents
        )["posed_body"]
        self.assertEqual(edge["relation"], "supported-by")
        self.assertEqual(edge["physics_role"], "kinematic")

    def test_batch_json_is_validated_and_canonicalized(self):
        objects = ["desk_000", "keyboard_000"]
        parents = ["floor", "wall", *objects]
        response = """```json
[
  {"child":"DESK_000","parent":"floor","relation":"on","type":"movable"},
  {"child":"keyboard_000","parent":"desk_000","relation":"on","type":"movable"},
  {"child":"unknown","parent":"floor","relation":"on","type":"movable"}
]
```"""
        edges = stage1._parse_scene_tree_batch_response(response, objects, parents)
        self.assertEqual(set(edges), set(objects))
        self.assertEqual(edges["desk_000"]["child"], "desk_000")
        self.assertEqual(edges["keyboard_000"]["parent"], "desk_000")

    def test_explicit_on_the_parent_hint_overrides_visual_ambiguity(self):
        objects = [
            "keyboard_on_the_desk_000",
            "person_sitting_behind_the_desk_000",
            "white_desk_with_black_modesty_panel_000",
        ]
        parents = ["floor", "wall", *objects]
        response = json.dumps([{
            "child": "keyboard_on_the_desk_000",
            "parent": "person_sitting_behind_the_desk_000",
            "relation": "on",
            "type": "movable",
        }])
        edges = stage1._parse_scene_tree_batch_response(response, objects, parents)
        edge = edges["keyboard_on_the_desk_000"]
        self.assertEqual(edge["parent"], "white_desk_with_black_modesty_panel_000")
        self.assertEqual(edge["relation"], "on")

    def test_contact_sheets_pack_six_overlays_each(self):
        objects = [f"object_{index:03d}" for index in range(7)]
        obj_data = {
            name: {"overlay": Image.new("RGB", (300, 500), "white")}
            for name in objects
        }
        sheets = stage1._make_scene_tree_contact_sheets(objects, obj_data)
        self.assertEqual(len(sheets), 2)
        self.assertEqual(sheets[0].size, (1152, 1092))

    def test_compact_overlay_highlights_only_mask_pixels(self):
        import numpy as np

        scene = Image.new("RGB", (4, 4), (100, 100, 100))
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        overlay = stage1._make_compact_mask_overlay(scene, mask)
        self.assertEqual(overlay.getpixel((0, 0)), (100, 100, 100))
        self.assertNotEqual(overlay.getpixel((1, 1)), (100, 100, 100))

    def test_batch_query_uses_one_vlm_call_for_multiple_objects(self):
        objects = ["desk_000", "keyboard_000"]
        obj_data = {
            name: {"overlay": Image.new("RGB", (300, 500), "white")}
            for name in objects
        }
        response = json.dumps([
            {"child": "desk_000", "parent": "floor", "relation": "on", "type": "movable"},
            {"child": "keyboard_000", "parent": "desk_000", "relation": "on", "type": "movable"},
        ])
        with mock.patch.object(
            stage1, "generate_vlm_response", return_value=response
        ) as generate:
            edges = stage1._query_scene_tree_batch(
                "image.png", objects, obj_data, ["floor", "wall", *objects]
            )

        generate.assert_called_once()
        content = generate.call_args.args[0][0]["content"]
        uploaded_images = [item for item in content if item["type"] == "image"]
        self.assertEqual(len(uploaded_images), 2)  # source image + one contact sheet
        self.assertEqual(set(edges), set(objects))

    def test_fingerprint_changes_when_a_mask_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            image_path.write_bytes(b"image")
            mask_path.write_bytes(b"mask-one")
            first = stage1._scene_tree_fingerprint(
                str(image_path), [str(mask_path)], "gemini", True
            )
            mask_path.write_bytes(b"mask-two")
            second = stage1._scene_tree_fingerprint(
                str(image_path), [str(mask_path)], "gemini", True
            )
            self.assertNotEqual(first, second)


class Stage1CacheTests(unittest.TestCase):
    def test_floor_cache_uses_the_floor_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir)
            agent_dir = stage_dir / "segment_agent_out"
            mask_dir = stage_dir / "segemented_obj"
            agent_dir.mkdir()
            mask_dir.mkdir()
            (stage_dir / "scene_object_lists.txt").write_text("chair\n")
            (agent_dir / "image_chair_agent_gemini_pred.json").write_text("{}")
            (agent_dir / "image_the_floor_agent_gemini_pred.json").write_text("{}")
            (mask_dir / "chair_000.png").write_bytes(b"mask")
            (mask_dir / "the_floor.png").write_bytes(b"floor")

            objects, cached = stage1.load_cached_results(
                str(stage_dir), str(agent_dir), "image", "gemini"
            )
            self.assertEqual(objects, ["chair", "the floor"])
            self.assertEqual(cached, {"chair", "the floor"})

    def test_sam_model_is_initialized_only_on_first_service_call(self):
        args = argparse.Namespace(vlm_backend="gemini")
        fake_model = object()
        fake_processor = object()
        with mock.patch.object(
            stage1, "build_sam3_image_model", return_value=fake_model
        ) as build_model, mock.patch.object(
            stage1, "Sam3Processor", return_value=fake_processor
        ) as build_processor, mock.patch.object(
            stage1, "call_sam_service_orig", return_value="result.json"
        ) as original_service:
            _, _, service = stage1.build_agent_components(args)
            build_model.assert_not_called()
            service(image_path="image.png", text_prompt="chair")
            service(image_path="image.png", text_prompt="desk")

        build_model.assert_called_once()
        build_processor.assert_called_once_with(fake_model, confidence_threshold=0.5)
        self.assertEqual(original_service.call_count, 2)
        self.assertIs(original_service.call_args.kwargs["sam3_processor"], fake_processor)


if __name__ == "__main__":
    unittest.main()

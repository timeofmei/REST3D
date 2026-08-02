#!/usr/bin/env python3
"""Rebuild Stage 2 support placement without modifying the source run."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rest3d.scene_support import place_to_ground
from rest3d.utils.urdf import generate_urdf_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_stage2.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    source_tree = source / "scene_tree.json"
    source_y_align = source / "scene_y_align" / "obj_files"
    if not source_tree.is_file() or not source_y_align.is_dir():
        raise FileNotFoundError(
            f"source Stage 2 must contain scene_tree.json and scene_y_align/obj_files: {source}"
        )

    stage2 = output / "stage2"
    obj_dir = stage2 / "scene_canon" / "obj_files"
    urdf_dir = stage2 / "scene_canon" / "urdf_files"
    obj_dir.mkdir(parents=True)
    urdf_dir.mkdir(parents=True)
    tree_path = stage2 / "scene_tree.json"
    shutil.copy2(source_tree, tree_path)

    place_to_ground(
        source_y_align,
        obj_dir,
        tree_path,
        report_path=stage2 / "support_diagnostics.json",
        resolved_scene_tree_path=tree_path,
    )
    object_names = sorted(
        path.stem.removeprefix("scene_canon_") for path in obj_dir.glob("*.obj")
    )
    generate_urdf_files(object_names, str(urdf_dir), prefix="scene_canon_")


if __name__ == "__main__":
    main()

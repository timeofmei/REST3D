#!/usr/bin/env python3
"""Validate a Stage G run and write its standard evidence files."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from rest3d.sim.stage3_regression import validate_stage3_regression


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_hashes(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split("  ", 1)
        records[relative_path] = digest
    return records


def _write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _git(repo_root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve(strict=True)
    pipeline_run = output_dir / "pipeline_run"
    stage3_dir = pipeline_run / "stage3"
    replay_dir = output_dir / "final_replay"
    comparison_dir = output_dir / "backend_comparison" / "comparison"

    fixture = _read_json(pipeline_run / "stage2" / "regression_fixture.json")
    physics = _read_json(stage3_dir / "physics_assets" / "physics_assets.json")
    local_plan = _read_json(stage3_dir / "local_group_plan" / "local_groups.json")
    pipeline = _read_json(stage3_dir / "stage3_pipeline_results.json")
    global_result = _read_json(stage3_dir / "global_cem" / "real_global_cem_results.json")
    replay = _read_json(replay_dir / "replay_results.json")
    states = np.load(replay_dir / "replay_states_rest.npy")
    comparison = _read_json(comparison_dir / "metrics.json")
    metrics = validate_stage3_regression(
        expected_names=fixture["object_names"],
        stage2_hashes_before=_read_hashes(output_dir / "stage2_before.sha256"),
        stage2_hashes_after=_read_hashes(output_dir / "stage2_after.sha256"),
        physics_assets=physics,
        local_plan=local_plan,
        pipeline_result=pipeline,
        global_result=global_result,
        replay_result=replay,
        replay_state_shape=states.shape,
        comparison_result=comparison,
    )
    metrics["artifacts"] = {
        "pipeline": str(stage3_dir / "stage3_pipeline_results.json"),
        "global_cem": str(
            stage3_dir / "global_cem" / "real_global_cem_results.json"
        ),
        "final_replay": str(replay_dir / "replay_results.json"),
        "backend_comparison": str(comparison_dir / "metrics.json"),
    }
    environment = {
        "git": _git(Path(__file__).resolve().parents[1]),
        "platform": platform.platform(),
        "isaac_lab": {
            "versions": global_result.get("versions"),
            "environment": global_result.get("environment"),
        },
        "isaac_gym": comparison.get("software_and_hardware", {}).get(
            "isaac_gym"
        ),
        "device_contract": {
            "isaac_gym": "GPU PhysX + CPU tensor pipeline",
            "isaac_lab": "GPU PhysX + GPU tensor pipeline",
        },
    }
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "metrics.json", metrics)
    np.save(output_dir / "states_initial.npy", states[0], allow_pickle=False)
    np.save(output_dir / "states_final.npy", states[-1], allow_pickle=False)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

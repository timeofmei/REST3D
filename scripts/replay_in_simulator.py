"""Dispatch scene replay to an isolated Isaac Gym or Isaac Lab process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--backend",
        choices=("isaac-gym", "isaac-lab"),
        default="isaac-gym",
    )
    args, backend_args = parser.parse_known_args()

    script_dir = Path(__file__).resolve().parent
    backend_script = {
        "isaac-gym": script_dir / "replay_in_isaac_gym.py",
        "isaac-lab": script_dir / "replay_in_isaac_lab.py",
    }[args.backend]
    os.execv(
        sys.executable,
        [sys.executable, str(backend_script), *backend_args],
    )


if __name__ == "__main__":
    main()

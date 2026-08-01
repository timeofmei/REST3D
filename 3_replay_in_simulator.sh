#!/usr/bin/env bash
# Replay an optimized scene with the selected simulator backend.
#   - Loads stable scene from output/<image_name>/stage3/global_scene or global_scene_w_walls
#   - Settles objects under gravity and records an MP4 （output/<image_name>/stage3/global_scene/replay*.mp4）
#   - Optionally opens a viser 3D browser viewer (--viser)
#   - Note that without --viser, Isaac Gym does not display object textures by default
#
# Usage examples:
#   bash 3_replay_in_simulator.sh --output-dir output/foreground_run1


INPUT=data/my_capture_frame145_centered_meters/images/cam_22.png
OUTPUT_DIR=""
REPLAY_OUTPUT_DIR=""
BACKEND="isaac-gym"
SETTLE_STEPS=120

usage() {
    cat <<'EOF'
Usage: bash 3_replay_in_simulator.sh [options]

Options:
  --input PATH       Input image used to derive the default run name
  --output-dir DIR   Read this exact pipeline run directory
  --replay-output-dir DIR
                     Write replay results to this new directory (required)
  --backend NAME     isaac-gym (default) or isaac-lab
  --settle-steps N   Physics steps (default: 120)
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 ]] || { echo "ERROR: --input requires a path" >&2; exit 2; }
            INPUT=$2
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --output-dir requires a path" >&2; exit 2; }
            OUTPUT_DIR=$2
            shift 2
            ;;
        --replay-output-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --replay-output-dir requires a path" >&2; exit 2; }
            REPLAY_OUTPUT_DIR=$2
            shift 2
            ;;
        --backend)
            [[ $# -ge 2 ]] || { echo "ERROR: --backend requires a value" >&2; exit 2; }
            BACKEND=$2
            shift 2
            ;;
        --settle-steps)
            [[ $# -ge 2 ]] || { echo "ERROR: --settle-steps requires a value" >&2; exit 2; }
            SETTLE_STEPS=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

STEM=$(basename "${INPUT%.*}")
if [[ -n "$OUTPUT_DIR" ]]; then
    RUN_DIR=${OUTPUT_DIR%/}
    [[ -n "$RUN_DIR" ]] || { echo "ERROR: --output-dir cannot be /" >&2; exit 2; }
else
    RUN_DIR="output/${STEM}"
fi

printf 'Run directory: %s\n' "$RUN_DIR"

case "$BACKEND" in
    isaac-gym|isaac-lab) ;;
    *) echo "ERROR: --backend must be isaac-gym or isaac-lab" >&2; exit 2 ;;
esac
[[ -n "$REPLAY_OUTPUT_DIR" ]] || {
    echo "ERROR: --replay-output-dir is required to prevent overwriting input results" >&2
    exit 2
}
[[ ! -e "$REPLAY_OUTPUT_DIR" ]] || {
    echo "ERROR: replay output path already exists: $REPLAY_OUTPUT_DIR" >&2
    exit 1
}

SCENE_TREE="${RUN_DIR}/stage2/scene_tree.json"
SCENE_DIR="${RUN_DIR}/stage3/global_scene"

if [[ "$BACKEND" == "isaac-lab" ]]; then
    exec bash scripts/run_isaaclab_replay.sh \
        "$SCENE_TREE" "$SCENE_DIR" "$REPLAY_OUTPUT_DIR" \
        --settle-steps "$SETTLE_STEPS"
fi

[[ -n "${CONDA_PREFIX:-}" ]] || {
    echo "ERROR: activate the Python 3.8 gym environment for the isaac-gym backend" >&2
    exit 1
}
ISAACGYM_LIBPYTHON="${CONDA_PREFIX}/lib/libpython3.8.so.1.0"
if [[ ! -f "${ISAACGYM_LIBPYTHON}" ]]; then
    echo "ERROR: ${ISAACGYM_LIBPYTHON} not found; activate the Python 3.8 gym environment." >&2
    exit 1
fi

ISAACGYM_DRIVER_LIB=""
if [[ -d /usr/lib/wsl/lib ]]; then
    ISAACGYM_DRIVER_LIB="/usr/lib/wsl/lib"
fi

PYTHONPATH="$(pwd):${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH="${ISAACGYM_DRIVER_LIB}" \
LD_PRELOAD="${ISAACGYM_LIBPYTHON}" \
python scripts/replay_in_simulator.py \
    --backend isaac-gym \
    --scene-tree "$SCENE_TREE" \
    --scene-dir "$SCENE_DIR" \
    --output-dir "$REPLAY_OUTPUT_DIR" \
    --settle_steps "$SETTLE_STEPS" --headless --no_save_video

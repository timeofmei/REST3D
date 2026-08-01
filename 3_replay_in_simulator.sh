#!/usr/bin/env bash
# Replay optimized scene in Isaac Gym.
#   - Loads stable scene from output/<image_name>/stage3/global_scene or global_scene_w_walls
#   - Settles objects under gravity and records an MP4 （output/<image_name>/stage3/global_scene/replay*.mp4）
#   - Optionally opens a viser 3D browser viewer (--viser)
#   - Note that without --viser, Isaac Gym does not display object textures by default


INPUT=data/my_capture_frame145_centered_meters/images/cam_22.png
STEM=$(basename "${INPUT%.*}")

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
    --scene_tree "output/${STEM}/stage2/scene_tree.json" \
    --output_dir "output/${STEM}/stage3/global_scene" \
    --settle_steps 120 --viser

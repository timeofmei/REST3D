#!/usr/bin/env bash
# Run to stabilize a 3D scene inferred from a single image (scene_canon):
#  
#
# Output:
#   output/<image_name>/stage3/   (stabilized 3D scene)


INPUT=data/my_capture_frame145_centered_meters/images/cam_22.png
STEM=$(basename "${INPUT%.*}")

ISAACGYM_LIBPYTHON="${CONDA_PREFIX}/lib/libpython3.8.so.1.0"
if [[ ! -f "${ISAACGYM_LIBPYTHON}" ]]; then
    echo "ERROR: ${ISAACGYM_LIBPYTHON} not found; activate the Python 3.8 gym environment." >&2
    exit 1
fi

# Limit the legacy Isaac Gym loader workaround to this Python process.  Adding
# the whole Conda lib directory to LD_LIBRARY_PATH pollutes subsequently
# activated environments and can make Bash load gym's libtinfo/libstdc++.
ISAACGYM_DRIVER_LIB=""
if [[ -d /usr/lib/wsl/lib ]]; then
    ISAACGYM_DRIVER_LIB="/usr/lib/wsl/lib"
fi

PYTHONPATH="$(pwd):${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH="${ISAACGYM_DRIVER_LIB}" \
LD_PRELOAD="${ISAACGYM_LIBPYTHON}" \
python scripts/stable_scene.py \
    --scene_dir  "output/${STEM}/stage2" \
    --output_dir "output/${STEM}/stage3"

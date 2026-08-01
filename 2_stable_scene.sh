#!/usr/bin/env bash
# Run to stabilize a 3D scene inferred from a single image (scene_canon):
#  
#
# Output:
#   output/<image_name>/stage3/   (stabilized 3D scene)
#
# Usage examples:
#   bash 2_stable_scene.sh --output-dir output/foreground_run1


INPUT=data/my_capture_frame145_centered_meters/images/cam_22.png
OUTPUT_DIR=""

usage() {
    cat <<'EOF'
Usage: bash 2_stable_scene.sh [options]

Options:
  --input PATH       Input image used to derive the default run name
  --output-dir DIR   Use this exact run directory
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
    --scene_dir  "${RUN_DIR}/stage2" \
    --output_dir "${RUN_DIR}/stage3"

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
BACKEND="isaac-gym"
LOCAL_GROUPS_ONLY=false
GYM_ARGS=()
ISAACLAB_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash 2_stable_scene.sh [options]

Options:
  --input PATH       Input image used to derive the default run name
  --output-dir DIR   Use this exact run directory
  --backend NAME     isaac-gym (default) or isaac-lab
  --local-groups-only
                     Stop Isaac Lab after local CEM instead of continuing to global CEM
  --cem-pop-size N   Override parallel candidate environments (default: 2048)
  --cem-iters-subtree N
                     Override local CEM iterations (default: 15)
  --cem-iters-joint N
                     Override global CEM iterations (default: 15)
  --cem-seed N       Fix the local CEM random seed
  --total-settle-steps N
  --vel-settle-steps N
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
        --backend)
            [[ $# -ge 2 ]] || { echo "ERROR: --backend requires a value" >&2; exit 2; }
            BACKEND=$2
            shift 2
            ;;
        --local-groups-only)
            LOCAL_GROUPS_ONLY=true
            shift
            ;;
        --cem-pop-size)
            [[ $# -ge 2 ]] || { echo "ERROR: --cem-pop-size requires a value" >&2; exit 2; }
            GYM_ARGS+=("--cem-pop-size" "$2")
            ISAACLAB_ARGS+=("--num-envs" "$2")
            shift 2
            ;;
        --cem-iters-subtree)
            [[ $# -ge 2 ]] || { echo "ERROR: --cem-iters-subtree requires a value" >&2; exit 2; }
            GYM_ARGS+=("--cem-iters-subtree" "$2")
            ISAACLAB_ARGS+=("--cem-iters" "$2")
            shift 2
            ;;
        --cem-iters-joint)
            [[ $# -ge 2 ]] || { echo "ERROR: --cem-iters-joint requires a value" >&2; exit 2; }
            GYM_ARGS+=("--cem-iters-joint" "$2")
            ISAACLAB_ARGS+=("--global-cem-iters" "$2")
            shift 2
            ;;
        --cem-seed)
            [[ $# -ge 2 ]] || { echo "ERROR: --cem-seed requires a value" >&2; exit 2; }
            GYM_ARGS+=("--cem-seed" "$2")
            ISAACLAB_ARGS+=("--seed" "$2")
            shift 2
            ;;
        --total-settle-steps)
            [[ $# -ge 2 ]] || { echo "ERROR: --total-settle-steps requires a value" >&2; exit 2; }
            GYM_ARGS+=("--total-settle-steps" "$2")
            ISAACLAB_ARGS+=("--settle-steps" "$2")
            shift 2
            ;;
        --vel-settle-steps)
            [[ $# -ge 2 ]] || { echo "ERROR: --vel-settle-steps requires a value" >&2; exit 2; }
            GYM_ARGS+=("--vel-settle-steps" "$2")
            ISAACLAB_ARGS+=("--early-steps" "$2")
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

case "$BACKEND" in
    isaac-gym|isaac-lab) ;;
    *) echo "ERROR: --backend must be isaac-gym or isaac-lab" >&2; exit 2 ;;
esac

STEM=$(basename "${INPUT%.*}")
if [[ -n "$OUTPUT_DIR" ]]; then
    RUN_DIR=${OUTPUT_DIR%/}
    [[ -n "$RUN_DIR" ]] || { echo "ERROR: --output-dir cannot be /" >&2; exit 2; }
else
    RUN_DIR="output/${STEM}"
fi

printf 'Run directory: %s\n' "$RUN_DIR"

if [[ "$BACKEND" == "isaac-lab" ]]; then
    if [[ "$LOCAL_GROUPS_ONLY" != true ]]; then
        ISAACLAB_ARGS+=("--run-global")
    fi
    bash scripts/run_isaaclab_local_groups.sh \
        "${RUN_DIR}/stage2" \
        "${RUN_DIR}/stage3" \
        "${ISAACLAB_ARGS[@]}"
    exit $?
fi

if [[ "$LOCAL_GROUPS_ONLY" == true ]]; then
    echo "ERROR: --local-groups-only is available only with --backend isaac-lab." >&2
    exit 2
fi

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
    --output_dir "${RUN_DIR}/stage3" \
    "${GYM_ARGS[@]}"

#!/usr/bin/env bash
# Run to infer a 3D scene from a single image:
#  
# Before running, export your VLM API key in the shell:
#   export GEMINI_API_KEY=...
#
# Output:
#   output/<image_name>/stage1/   (scene tree, masks)
#   output/<image_name>/stage2/   (3D scene: scene_canon)
#
# Usage examples:
#   bash 1_infer_scenecanon.sh --output-dir output/foreground_run1

INPUT=data/my_capture_frame145_centered_meters/images/cam_22.png
OUTPUT_DIR=""
VLM_MAX_IMAGE_EDGE=1536
STAGE1_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash 1_infer_scenecanon.sh [options]

Options:
  --input PATH       Input image (default: the INPUT value in this script)
  --output-dir DIR   Write to this exact run directory
  --vlm-max-image-edge N
                     Longest image edge uploaded to the VLM (default: 1536;
                     SAM3 still uses the original resolution; 0 disables)
  --force-scene-tree Rebuild the scene tree instead of using its fingerprint cache
  --legacy-scene-tree-per-object
                     Use the old one-VLM-request-per-mask scene-tree path
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
        --vlm-max-image-edge)
            [[ $# -ge 2 ]] || { echo "ERROR: --vlm-max-image-edge requires a number" >&2; exit 2; }
            VLM_MAX_IMAGE_EDGE=$2
            shift 2
            ;;
        --force-scene-tree)
            STAGE1_EXTRA_ARGS+=(--force_scene_tree)
            shift
            ;;
        --legacy-scene-tree-per-object)
            STAGE1_EXTRA_ARGS+=(--legacy_scene_tree_per_object)
            shift
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
OUTPUT_ROOT=$(dirname -- "$RUN_DIR")
OUTPUT_NAME=$(basename -- "$RUN_DIR")

printf 'Input: %s\nRun directory: %s\n' "$INPUT" "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 python scripts/infer_scenetree.py \
    --image_folder "$INPUT" \
    --output_folder "$OUTPUT_ROOT" \
    --output_name "$OUTPUT_NAME" \
    --vlm_max_image_edge "$VLM_MAX_IMAGE_EDGE" \
    "${STAGE1_EXTRA_ARGS[@]}" \
    || { printf "\033[31mERROR: stage1 failed; stage2 will NOT be run.\033[0m\n" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=0 python scripts/infer_scene3d.py \
    --image_folder "$INPUT" \
    --output_folder "$OUTPUT_ROOT" \
    --output_name "$OUTPUT_NAME" \
    || { printf "\033[31mERROR: stage2 failed.\033[0m\n" >&2; exit 1; }

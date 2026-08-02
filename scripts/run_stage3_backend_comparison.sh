#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 SCENE_TREE SCENE_DIRECTORY NEW_OUTPUT_DIRECTORY [options]" >&2
    exit 2
fi

scene_tree="$(realpath "$1")"
scene_dir="$(realpath "$2")"
output_dir="$(realpath -m "$3")"
shift 3

steps=120
early_step=15
evaluation_step=60
position_threshold=0.1
rotation_threshold=0.1
seed=211
vhacd_max_hulls=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) steps="$2"; shift 2 ;;
        --early-step) early_step="$2"; shift 2 ;;
        --evaluation-step) evaluation_step="$2"; shift 2 ;;
        --position-threshold) position_threshold="$2"; shift 2 ;;
        --rotation-threshold) rotation_threshold="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --vhacd-max-hulls) vhacd_max_hulls="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
gym_python="${GYM_PYTHON:-/home/yangyankun/miniconda3/envs/gym/bin/python}"
gym_libpython="${GYM_LIBPYTHON:-/home/yangyankun/miniconda3/envs/gym/lib/libpython3.8.so.1.0}"
isaaclab_python="${ISAACLAB_PYTHON:-/home/yangyankun/miniconda3/envs/isaaclab/bin/python}"
wsl_driver_lib_dir="${WSL_DRIVER_LIB_DIR:-/usr/lib/wsl/lib}"
nvrtc_library="${ISAACLAB_NVRTC_LIBRARY:-/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12}"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output path: ${output_dir}" >&2
    exit 1
fi
if [[ ! -f "${scene_tree}" || ! -d "${scene_dir}/obj_files" || ! -d "${scene_dir}/urdf_files" ]]; then
    echo "The read-only scene tree and scene OBJ/URDF directories must exist" >&2
    exit 1
fi
for executable in "${gym_python}" "${isaaclab_python}"; do
    [[ -x "${executable}" ]] || { echo "Python is not executable: ${executable}" >&2; exit 1; }
done
for library in "${gym_libpython}" "${wsl_driver_lib_dir}/libcuda.so" "${nvrtc_library}"; do
    [[ -f "${library}" ]] || { echo "Required runtime library is missing: ${library}" >&2; exit 1; }
done

mkdir -p "${output_dir}"
gym_output="${output_dir}/isaac_gym"
lab_output="${output_dir}/isaac_lab"
report_output="${output_dir}/comparison"

PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
CONDA_PREFIX="$(dirname "$(dirname "${gym_python}")")" \
LD_LIBRARY_PATH="${wsl_driver_lib_dir}" \
LD_PRELOAD="${gym_libpython}" \
"${gym_python}" "${repo_root}/scripts/replay_in_simulator.py" \
    --backend isaac-gym \
    --scene-tree "${scene_tree}" \
    --scene-dir "${scene_dir}" \
    --output-dir "${gym_output}" \
    --settle-steps "${steps}" \
    --stability-evaluation-steps "${evaluation_step}" \
    --position-stability-threshold "${position_threshold}" \
    --rotation-stability-threshold "${rotation_threshold}" \
    --vhacd-max-hulls "${vhacd_max_hulls}" \
    --headless --no_save_video \
    2>&1 | tee "${output_dir}/isaac_gym.stdout.log"

bash "${repo_root}/scripts/run_isaaclab_replay.sh" \
    "${scene_tree}" "${scene_dir}" "${lab_output}" \
    --settle-steps "${steps}" \
    --stability-evaluation-steps "${evaluation_step}" \
    --position-stability-threshold "${position_threshold}" \
    --rotation-stability-threshold "${rotation_threshold}" \
    --collision-approximation convex_decomposition \
    --state-only-benchmark \
    2>&1 | tee "${output_dir}/isaac_lab.stdout.log"

nvrtc_dir="$(dirname "${nvrtc_library}")"
PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
CONDA_PREFIX="$(dirname "$(dirname "${isaaclab_python}")")" \
LD_LIBRARY_PATH="${nvrtc_dir}:${wsl_driver_lib_dir}" \
LD_PRELOAD="${nvrtc_library}" \
"${isaaclab_python}" "${repo_root}/scripts/compare_isaac_backends.py" \
    --scene-tree "${scene_tree}" \
    --scene-dir "${scene_dir}" \
    --isaac-gym-result-dir "${gym_output}" \
    --isaac-lab-result-dir "${lab_output}" \
    --output-dir "${report_output}" \
    --steps "${steps}" \
    --early-step "${early_step}" \
    --evaluation-step "${evaluation_step}" \
    --position-threshold "${position_threshold}" \
    --rotation-threshold "${rotation_threshold}" \
    --collision-approximation convex_decomposition \
    --seed "${seed}" \
    2>&1 | tee "${output_dir}/comparison.stdout.log"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 PHYSICS_ASSETS_JSON LOCAL_GROUPS_JSON GROUP_INDEX NEW_OUTPUT_DIRECTORY [extra args]" >&2
    exit 2
fi

physics_assets="$(realpath "$1")"
local_groups="$(realpath "$2")"
group_index="$3"
output_dir="$(realpath -m "$4")"
shift 4

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
isaaclab_python="${ISAACLAB_PYTHON:-/home/yangyankun/miniconda3/envs/isaaclab/bin/python}"
wsl_driver_lib_dir="${WSL_DRIVER_LIB_DIR:-/usr/lib/wsl/lib}"
nvrtc_library="${ISAACLAB_NVRTC_LIBRARY:-/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12}"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output path: ${output_dir}" >&2
    exit 1
fi
if [[ ! -f "${physics_assets}" || ! -f "${local_groups}" ]]; then
    echo "Physics asset and local group manifests must both exist" >&2
    exit 1
fi
if [[ ! -x "${isaaclab_python}" ]]; then
    echo "Isaac Lab Python is not executable: ${isaaclab_python}" >&2
    exit 1
fi
if [[ ! -f "${wsl_driver_lib_dir}/libcuda.so" ]]; then
    echo "WSL CUDA driver library was not found: ${wsl_driver_lib_dir}/libcuda.so" >&2
    exit 1
fi
if [[ ! -f "${nvrtc_library}" ]]; then
    echo "CUDA 12.8 NVRTC library was not found: ${nvrtc_library}" >&2
    exit 1
fi

driver_library_path="$(dirname "${nvrtc_library}"):${wsl_driver_lib_dir}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    driver_library_path="${driver_library_path}:${LD_LIBRARY_PATH}"
fi
preload_libraries="${nvrtc_library}"
if [[ -n "${LD_PRELOAD:-}" ]]; then
    preload_libraries="${preload_libraries}:${LD_PRELOAD}"
fi

OMNI_KIT_ACCEPT_EULA=YES \
CONDA_PREFIX="$(dirname "$(dirname "${isaaclab_python}")")" \
PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
LD_LIBRARY_PATH="${driver_library_path}" \
LD_PRELOAD="${preload_libraries}" \
"${isaaclab_python}" "${repo_root}/scripts/isaaclab_real_local_cem.py" \
    --physics-assets "${physics_assets}" \
    --local-groups "${local_groups}" \
    --group-index "${group_index}" \
    --output-dir "${output_dir}" \
    --headless \
    --device cuda:0 \
    "$@"

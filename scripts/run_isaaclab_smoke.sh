#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
output_dir="$(realpath -m "$1")"
isaaclab_python="${ISAACLAB_PYTHON:-/home/yangyankun/miniconda3/envs/isaaclab/bin/python}"
wsl_driver_lib_dir="${WSL_DRIVER_LIB_DIR:-/usr/lib/wsl/lib}"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output path: ${output_dir}" >&2
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

driver_library_path="${wsl_driver_lib_dir}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    driver_library_path="${driver_library_path}:${LD_LIBRARY_PATH}"
fi

OMNI_KIT_ACCEPT_EULA=YES \
CONDA_PREFIX="$(dirname "$(dirname "${isaaclab_python}")")" \
LD_LIBRARY_PATH="${driver_library_path}" \
"${isaaclab_python}" "${repo_root}/scripts/isaaclab_smoke_test.py" \
    --headless \
    --device cuda:0 \
    --steps 120 \
    --output-dir "${output_dir}" \
    --kit_args="--/log/file=${output_dir}/kit.log"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 PHYSICS_ASSETS_JSON GLOBAL_ENTITIES_JSON INITIAL_STATES_JSON NEW_OUTPUT_DIRECTORY [extra args]" >&2
    exit 2
fi

physics_assets="$(realpath "$1")"
global_entities="$(realpath "$2")"
initial_states="$(realpath "$3")"
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
if [[ ! -f "${physics_assets}" || ! -f "${global_entities}" || ! -f "${initial_states}" ]]; then
    echo "Physics assets, global entities, and initial states must all exist" >&2
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

set +e
OMNI_KIT_ACCEPT_EULA=YES \
CONDA_PREFIX="$(dirname "$(dirname "${isaaclab_python}")")" \
PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
LD_LIBRARY_PATH="${driver_library_path}" \
LD_PRELOAD="${preload_libraries}" \
"${isaaclab_python}" "${repo_root}/scripts/isaaclab_real_global_cem.py" \
    --physics-assets "${physics_assets}" \
    --global-entities "${global_entities}" \
    --initial-states "${initial_states}" \
    --output-dir "${output_dir}" \
    --headless \
    --device cuda:0 \
    "$@"
global_status=$?
set -e

if [[ -f "${output_dir}/real_global_cem_failure.json" ]]; then
    echo "Isaac Lab global CEM recorded a failure: ${output_dir}/real_global_cem_failure.json" >&2
    if [[ ${global_status} -eq 0 ]]; then
        exit 1
    fi
fi
if [[ ! -f "${output_dir}/real_global_cem_results.json" ]]; then
    echo "Isaac Lab global CEM did not produce real_global_cem_results.json: ${output_dir}" >&2
    if [[ ${global_status} -eq 0 ]]; then
        exit 1
    fi
fi
exit "${global_status}"

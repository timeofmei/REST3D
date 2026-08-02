#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 REPLAY_DIRECTORY [viewer args]" >&2
    exit 2
fi

replay_dir=$(realpath "$1")
shift
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

if [[ ! -f "${replay_dir}/replay_results.json" || ! -f "${replay_dir}/replay_states_rest.npy" ]]; then
    echo "Replay directory must contain replay_results.json and replay_states_rest.npy: ${replay_dir}" >&2
    exit 1
fi

if [[ -n "${VISER_PYTHON:-}" ]]; then
    viewer_python=${VISER_PYTHON}
elif current_python=$(command -v python 2>/dev/null) && \
        "${current_python}" -c 'import numpy, trimesh, viser' >/dev/null 2>&1; then
    viewer_python=${current_python}
elif command -v conda >/dev/null 2>&1; then
    conda_base=$(conda info --base)
    viewer_python="${conda_base}/envs/isaaclab/bin/python"
else
    viewer_python=""
fi

if [[ -z "${viewer_python:-}" || ! -x "${viewer_python}" ]]; then
    echo "Viser Python is unavailable; set VISER_PYTHON to an environment containing viser and trimesh." >&2
    exit 1
fi
if ! "${viewer_python}" -c 'import numpy, trimesh, viser' >/dev/null 2>&1; then
    echo "${viewer_python} cannot import numpy, trimesh, and viser." >&2
    echo "Install the documented constrained Viser version in isaaclab or set VISER_PYTHON explicitly." >&2
    exit 1
fi

PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
exec "${viewer_python}" "${repo_root}/scripts/view_replay_with_viser.py" \
    "${replay_dir}" "$@"

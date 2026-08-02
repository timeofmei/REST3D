#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
output_dir="$(realpath -m "$1")"
rest3d_python="${REST3D_PYTHON:-/home/yangyankun/miniconda3/envs/rest3d/bin/python}"
gym_python="${GYM_PYTHON:-/home/yangyankun/miniconda3/envs/gym/bin/python}"
isaaclab_python="${ISAACLAB_PYTHON:-/home/yangyankun/miniconda3/envs/isaaclab/bin/python}"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output path: ${output_dir}" >&2
    exit 1
fi
for executable in "${rest3d_python}" "${gym_python}" "${isaaclab_python}"; do
    [[ -x "${executable}" ]] || {
        echo "Python is not executable: ${executable}" >&2
        exit 1
    }
done

mkdir -p "${output_dir}"
printf '%q %q\n' "$0" "$1" > "${output_dir}/command.txt"
exec > >(tee "${output_dir}/run.log") 2>&1
cd "${repo_root}"

echo "[stage-g] pure Python regression"
"${rest3d_python}" -m pytest -q tests

python_sources=(
    rest3d/sim/backend_comparison.py
    rest3d/sim/global_cem.py
    rest3d/sim/local_cem.py
    rest3d/sim/local_groups.py
    rest3d/sim/physics_assets.py
    rest3d/sim/replay_scene.py
    rest3d/sim/stability.py
    rest3d/sim/stage3_regression.py
    scripts/compare_isaac_backends.py
    scripts/create_stage3_regression_scene.py
    scripts/isaaclab_local_group_pipeline.py
    scripts/isaaclab_real_global_cem.py
    scripts/isaaclab_real_local_cem.py
    scripts/replay_in_isaac_gym.py
    scripts/replay_in_isaac_lab.py
    scripts/replay_in_simulator.py
    scripts/summarize_stage3_regression.py
)
for executable in "${rest3d_python}" "${gym_python}" "${isaaclab_python}"; do
    "${executable}" -m py_compile "${python_sources[@]}"
done

shell_sources=(
    2_stable_scene.sh
    3_replay_in_simulator.sh
    scripts/run_isaaclab_local_cem_smoke.sh
    scripts/run_isaaclab_local_groups.sh
    scripts/run_isaaclab_real_global_cem.sh
    scripts/run_isaaclab_real_local_cem.sh
    scripts/run_isaaclab_replay.sh
    scripts/run_isaaclab_smoke.sh
    scripts/run_stage3_backend_comparison.sh
    scripts/run_stage3_regression.sh
)
bash -n "${shell_sources[@]}"

pipeline_run="${output_dir}/pipeline_run"
stage2_dir="${pipeline_run}/stage2"
echo "[stage-g] create generic Stage 2 fixture"
"${rest3d_python}" scripts/create_stage3_regression_scene.py \
    "${stage2_dir}" \
    --object-names amber_base_713 cobalt_piece_829 violet_peer_947

find "${stage2_dir}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${output_dir}/stage2_before.sha256"

echo "[stage-g] complete Isaac Lab local and global CEM"
bash 2_stable_scene.sh \
    --output-dir "${pipeline_run}" \
    --backend isaac-lab \
    --cem-pop-size 4 \
    --cem-iters-subtree 1 \
    --cem-iters-joint 1 \
    --cem-seed 701 \
    --total-settle-steps 20 \
    --vel-settle-steps 5

echo "[stage-g] final 60-frame Isaac Lab replay"
bash 3_replay_in_simulator.sh \
    --output-dir "${pipeline_run}" \
    --backend isaac-lab \
    --replay-output-dir "${output_dir}/final_replay" \
    --settle-steps 60 \
    --require-stable

echo "[stage-g] matched Isaac Gym and Isaac Lab replay"
bash scripts/run_stage3_backend_comparison.sh \
    "${stage2_dir}/scene_tree.json" \
    "${stage2_dir}/scene_canon" \
    "${output_dir}/backend_comparison" \
    --steps 60 \
    --early-step 10 \
    --evaluation-step 30 \
    --position-threshold 0.1 \
    --rotation-threshold 0.1 \
    --seed 709 \
    --vhacd-max-hulls 8

find "${stage2_dir}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${output_dir}/stage2_after.sha256"

echo "[stage-g] validate standard artifacts and contracts"
PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
"${isaaclab_python}" scripts/summarize_stage3_regression.py "${output_dir}"

echo "[stage-g] PASS: ${output_dir}/metrics.json"

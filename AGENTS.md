# Repository Agent Guidelines

## Scope

These instructions apply to the entire repository.

## Development baseline

- All future development is based on the current branch (`main`, which was
  reset to `fbbcc0e` and merged with the y-align orientation fix; see
  `doc/2026-8-2.md` 问题 2 for the full history).
- Do not consult, reference, or merge from the archive branches
  (`backup/main-at-fbf240d`, `fix/scene-support-semantics`,
  `experiment/isaac-lab-migration`, or the pre-reset `origin/main` history)
  unless the user explicitly mentions them. They exist only to preserve the
  discarded Isaac Lab migration / scene-support work. They are mirrored to
  `origin/` and treated as read-only archives; never advance or delete them
  without explicit user instruction.
- The current `main` contains no Isaac Lab code. All `isaaclab`/`isaac-lab`
  references in `doc/` and `environments/` are stale archive material from
  the discarded work; they describe scripts and workflows that do not exist
  in this codebase and must not be treated as current capabilities.

## Project environments

- Use the `rest3d` Conda environment for scene understanding and reconstruction
  (Stages 1 and 2). It uses Python 3.11, PyTorch with CUDA 12.8, and must support
  the RTX 5090 (`sm_120`).
- The Gemini VLM API key used by Stage 1 lives in the repository-local `.env`
  file; load it with `set -a; source .env; set +a` before running `1_infer_scenecanon.sh`.
  This is a local-repository-specific loading convention.
- Use the `gym` Conda environment for Isaac Gym Stage 3 stabilization,
  optimization, and replay. Isaac Gym Preview 4 uses Python 3.8 and an older
  PyTorch build; on RTX 5090 it must use GPU PhysX with the CPU tensor pipeline
  unless that environment's PyTorch build explicitly supports `sm_120`.
- Do not persist the `gym` environment's library directory in
  `LD_LIBRARY_PATH`. Scope the WSL driver path and `libpython3.8` preload to the
  Isaac Gym process so other shells and Conda environments are not polluted.

## Command style

- Always activate the target Conda environment with `conda activate <env>`
  before running environment-specific commands. Do not invoke binaries via
  absolute paths such as `~/miniconda3/envs/<env>/bin/python`.
- Replay and replay visualization (`scripts/view_replay_viser.py`) must run in
  the `gym` environment. Do not use or maintain compatibility with the
  `isaaclab` environment's viser 0.2.11.

## Editing and verification

- Preserve unrelated user changes and do not overwrite hard-coded input paths
  or experiment settings unless the requested task requires it.
- Use the smallest relevant verification for each change. At minimum, run
  `bash -n` for edited shell scripts and `python -m py_compile` for edited Python
  entry points.
- For CUDA compatibility changes, verify an actual CUDA operation; a successful
  import alone is insufficient.
- Keep installation documentation consistent with the versions and execution
  modes that have been tested in the relevant Conda environments.

## Git policy

- Unless the user explicitly asks for a commit, do not commit any changes.
- Never use a fast-forward merge or `--ff-only` when merging one branch into
  another. Use `git merge --no-ff` to create an explicit merge commit and
  preserve the visible branch topology, unless the user explicitly requests a
  different merge strategy.
- When the user explicitly requests a commit, write a detailed commit message
  in English. Use a concise English subject and an English body that explains
  the motivation, the important implementation details, and the verification
  performed.
- Do not push, amend, rebase, or otherwise rewrite Git history unless the user
  explicitly requests that operation.

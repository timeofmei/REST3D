# Repository Agent Guidelines

## Scope

These instructions apply to the entire repository.

## Project environments

- Use the `rest3d` Conda environment for scene understanding and reconstruction
  (Stages 1 and 2). It uses Python 3.11, PyTorch with CUDA 12.8, and must support
  the RTX 5090 (`sm_120`).
- Use the `gym` Conda environment for Isaac Gym stabilization and replay
  (Stage 3). Isaac Gym Preview 4 uses Python 3.8 and an older PyTorch build.
- On RTX 5090, Stage 3 must use GPU PhysX with the CPU tensor pipeline unless
  the installed PyTorch build explicitly supports `sm_120`.
- Do not persist the `gym` environment's library directory in
  `LD_LIBRARY_PATH`. Scope the WSL driver path and `libpython3.8` preload to the
  Isaac Gym process so other shells and Conda environments are not polluted.

## Editing and verification

- Preserve unrelated user changes and do not overwrite hard-coded input paths
  or experiment settings unless the requested task requires it.
- Use the smallest relevant verification for each change. At minimum, run
  `bash -n` for edited shell scripts and `python -m py_compile` for edited Python
  entry points.
- For CUDA compatibility changes, verify an actual CUDA operation; a successful
  import alone is insufficient.
- Keep installation documentation consistent with the versions and execution
  modes that have been tested in the two Conda environments.

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

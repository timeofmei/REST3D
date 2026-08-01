# Isaac Lab environment for REST3D Stage 3

This environment is independent from the existing `rest3d` and `gym` Conda
environments. The tested A–G baseline is:

- Isaac Lab source tag `v2.3.2` (`37ddf626871758333d6ed89cf64ad702aef127d0`);
- Isaac Sim `5.1.0.0`;
- Python `3.11.15`;
- PyTorch `2.7.0+cu128`, torchvision `0.22.0+cu128`, and torchaudio
  `2.7.0+cu128`;
- CUDA runtime `12.8` supplied by the PyTorch wheels;
- RTX 5090 compute capability 12.0 (`sm_120`).

The installed package metadata reports `isaaclab==0.54.2`; the authoritative
Isaac Lab source identity for this editable install is the `v2.3.2` Git tag and
commit above. Other observed transitive versions include NumPy `1.26.0`, trimesh
`4.5.1`, and `warp-lang==1.15.0`.

The version pair follows the official
[Isaac Lab dependency table](https://github.com/isaac-sim/IsaacLab#isaac-sim-version-dependency)
and the
[Isaac Lab 2.3.2 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html).
Isaac Sim 5.1 is no longer supported by NVIDIA, and WSL is not in its official
Linux support matrix. This combination is therefore a tested migration baseline,
not a claim of general WSL support.

## Installation used for Phase A

```bash
conda create -n isaaclab python=3.11.15 pip -y
conda activate isaaclab

pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

pip install --upgrade \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install \
  filelock==3.13.1 fsspec==2024.6.1 networkx==3.3 wheel==0.41.2 \
  setuptools==80.9.0 psutil==5.9.8 typing_extensions==4.12.2 onnx==1.18.0
pip install flatdict==4.0.1 --no-build-isolation

git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git /home/yangyankun/IsaacLab-2.3.2
pip install -e /home/yangyankun/IsaacLab-2.3.2/source/isaaclab
```

The Isaac Lab package metadata pins Starlette 0.49.1, while Isaac Sim 5.1 pins
FastAPI 0.115.7, whose metadata requires Starlette below 0.46. These exact
upstream constraints have no intersection, so `pip check` reports that one
known conflict. Phase A does not use the FastAPI path; the actual Kit launch,
CUDA PhysX step, CUDA state tensor, and PyTorch CUDA operation were tested.

## WSL process environment

On this machine, `/usr/lib/wsl/lib/libcuda.so` is not found by an unmodified
process even though `libcuda.so.1` is registered. Isaac PhysX then reports a
null CUDA handle and falls back to software. Scope the WSL driver directory to
the Isaac Lab process only:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib python your_isaaclab_entry.py
```

Do not export this setting in a shell profile. The smoke wrapper applies it only
to its child process and preserves an existing `LD_LIBRARY_PATH` after the WSL
driver directory.

## Reproduce the GPU smoke test

The output path must not already exist:

```bash
bash scripts/run_isaaclab_smoke.sh \
  output/isaaclab_migration/smoke_20260801_reproduction_v1
```

The script checks GPU simulation, the GPU tensor pipeline, GPU broadphase,
CUDA root-state tensors, `sm_120` support, a state-derived CUDA operation, and
the final resting pose of a falling rigid body. It writes `kit.log`,
`smoke.log`, and `smoke_results.json` into the new output directory.

The current WSL Vulkan loader enumerates only llvmpipe, so Kit logs a graphics
device enumeration error. Headless CUDA PhysX and tensor operations work, but
rendering is not validated and must remain disabled until the Vulkan path is
fixed or tested on an officially supported host.

After installation, verify the exact source checkout and package/runtime versions:

```bash
git -C /home/yangyankun/IsaacLab-2.3.2 rev-parse HEAD
git -C /home/yangyankun/IsaacLab-2.3.2 describe --tags --always --dirty

conda activate isaaclab
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
python -c 'import importlib.metadata as m; print(m.version("isaacsim"), m.version("isaaclab"))'
```

The version commands are diagnostic only. A successful import is not the GPU
acceptance test; run the smoke command below and require all checks in
`smoke_results.json` to be true.

## NVRTC for runtime-generated PyTorch kernels

Isaac Sim can load the CUDA 12.6 NVRTC library before PyTorch. Precompiled
PyTorch CUDA kernels still work, but a runtime-generated kernel then fails on
the RTX 5090 with `invalid value for --gpu-architecture`. The tested machine
already has CUDA Toolkit 12.8, so replay scopes its NVRTC library to the child
process:

```bash
LD_PRELOAD=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/lib/wsl/lib \
python your_isaaclab_entry.py
```

`scripts/run_isaaclab_replay.sh` applies this automatically. It does not
persist either variable or modify the existing Conda environments. Override
`ISAACLAB_NVRTC_LIBRARY` only when using another verified CUDA 12.8 location.

For the complete Stage 3 CLI, final 60-frame validation, old Gym regression,
output fields, and the full Stage G headless regression, continue with
[the Stage 3 guide](../doc/isaac-lab-stage3.md).

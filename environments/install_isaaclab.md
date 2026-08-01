# Isaac Lab environment for REST3D Stage 3

This environment is independent from the existing `rest3d` and `gym` Conda
environments. The tested Phase A baseline is:

- Isaac Lab source tag `v2.3.2` (`37ddf626871758333d6ed89cf64ad702aef127d0`);
- Isaac Sim `5.1.0.0`;
- Python `3.11.15`;
- PyTorch `2.7.0+cu128`, torchvision `0.22.0+cu128`, and torchaudio
  `2.7.0+cu128`;
- CUDA runtime `12.8` supplied by the PyTorch wheels;
- RTX 5090 compute capability 12.0 (`sm_120`).

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

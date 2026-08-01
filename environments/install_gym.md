# `gym` Environment Setup

Tested with Isaac Gym Preview 4, Python 3.8, PyTorch 2.2.2 and both Ampere and
Blackwell (RTX 50-series) GPUs. Used for physics simulation and scene
stabilization (Stage 3).

> Isaac Gym requires Python 3.8 and is incompatible with the `rest3d` environment (Python 3.11). A separate environment is necessary.

**1. Create the conda environment:**
```bash
conda create -n gym python=3.8
conda activate gym
```

**2. Install PyTorch:**
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

**3. Install Isaac Gym:**

Download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym), extract, then install:
```bash
cd /path/to/isaacgym/python
pip install -e .
```

On WSL2, do **not** add the whole Conda `lib` directory to a persistent
`LD_LIBRARY_PATH`. It can make Bash and programs from subsequently activated
environments load the `gym` environment's `libtinfo`/`libstdc++`. The project
launchers instead preload only `libpython3.8` and expose `/usr/lib/wsl/lib` to
the Isaac Gym process.

**4. Install Python dependencies:**
```bash
pip install rl-games==1.1.4 termcolor==1.1.0 tensorboard==2.14.0 protobuf==3.20.0 \
  trimesh==3.23.5 smplx==0.1.28 scipy==1.9.1 numpy==1.24.4 \
  wandb pyyaml tqdm joblib viser==1.0.24 opencv-python==4.6.0.66
```

**5. Verify:**
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
LD_PRELOAD="$CONDA_PREFIX/lib/libpython3.8.so.1.0" \
python -c "import isaacgym, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### RTX 50-series behavior

The newest PyTorch wheel available for Isaac Gym Preview 4's Python 3.8 does
not contain `sm_120` kernels. REST3D detects this automatically and selects:

- GPU PhysX for simulation;
- the CPU tensor pipeline for exchanging actor state with Python.

This is intentional: physics still runs on the 5090, while PyTorch never tries
to launch an unsupported CUDA kernel. Do not force `use_gpu_pipeline=True` in
this environment.

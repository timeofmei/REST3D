# `gym` Environment Setup

Tested on CUDA 12.1, Python 3.8, PyTorch 2.2.2. Used for physics simulation and scene stabilization (Stage 3).

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

For WSL2, configure the environment so Isaac Gym can find both
`libpython3.8.so` and the WSL NVIDIA driver library:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
vim $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Add the following line to `env_vars.sh`:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Then reactivate the environment:

```bash
conda deactivate
conda activate gym
```

**4. Install Python dependencies:**
```bash
pip install rl-games==1.1.4 termcolor==1.1.0 tensorboard==2.14.0 protobuf==3.20.0 \
  trimesh==3.23.5 smplx==0.1.28 scipy==1.9.1 numpy==1.24.4 \
  wandb pyyaml tqdm joblib viser==1.0.24 opencv-python==4.6.0.66
```

**5. Verify:**
```bash
python -c "import isaacgym, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

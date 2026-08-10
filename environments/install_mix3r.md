# `mix3r` Environment Setup

Used as the mix3r object-mesh generation backend for Stage 2 (alternative /
parallel to SAM-3D-Objects). Created and verified on 2026-08-10 on an RTX 5090
(`sm_120`) with WSL. All versions below are the ones that were actually tested;
do not upgrade them blindly (see the compatibility notes at the end).

> Please first clone the repo and set `REST3D_ROOT` as described in the main [README](../README.md):
> ```bash
> export REST3D_ROOT="/path/to/REST3D"
> ```

## 1. Create the conda environment

Same runtime as `rest3d`: Python 3.11 + PyTorch 2.7.1+cu128.

```bash
conda create -n mix3r python=3.11 -y
conda activate mix3r
```

## 2. Install PyTorch and core dependencies

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install xformers==0.0.31          # requires torch==2.7.1 exactly
pip install spconv-cu121==2.3.8       # TRELLIS sparse backend (default 'spconv')
pip install kaolin==0.18.0 \
  --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html
```

## 3. Compile the two CUDA extensions (need CUDA Toolkit 12.8)

Both are not on PyPI and must be built from source against the cu128 toolkit.
Compile with the 12.8 toolkit so the extensions support `sm_120`:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

**nvdiffrast** (NVlabs, used by the TRELLIS renderers):

```bash
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
```

**diff_gaussian_rasterization_mipsplatting** — the mip-splatting fork of
diff-gaussian-rasterization, renamed to the module name mix3r imports
(`diff_gaussian_rasterization_mipsplatting`). The upstream mip-splatting repo
renamed it to `diff_gaussian_rasterization`; the `_mipsplatting` name only
lives in setup.py/package dir, the C++ side uses the `TORCH_EXTENSION_NAME`
macro so renaming is sufficient:

```bash
cd /tmp
git clone https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting/submodules/diff-gaussian-rasterization
mv diff_gaussian_rasterization diff_gaussian_rasterization_mipsplatting
sed -i 's/diff_gaussian_rasterization/diff_gaussian_rasterization_mipsplatting/g' setup.py
pip install --no-build-isolation .
```

> On this machine the built source is kept at
> `/home/yangyankun/checkpoints/src/diff-gaussian-rasterization-mipsplatting/`
> (contains the Inria non-commercial LICENSE).

## 4. Install mix3r's runtime dependencies

```bash
pip install --no-build-isolation git+https://github.com/jsnln/cvcg_utils
pip install git+https://github.com/EasternJournalist/utils3d   # NOT the PyPI package!
pip install numpy==1.26.0
pip install opencv-python==4.9.0.80 pillow==11.3.0 trimesh rich easydict \
  einops loguru safetensors tqdm libigl plyfile imageio huggingface_hub
```

⚠️ Version-critical points:

- **`utils3d` must come from `EasternJournalist/utils3d`** — the PyPI package
  (0.1.x) has no `utils3d.numpy` submodule and fails at the very last step
  with `module 'utils3d' has no attribute 'numpy'`.
- **`numpy` must stay at 1.26.x**: `opencv-python 4.9.0.80` is built against
  the numpy 1.x ABI. Installing new packages may pull numpy 2.x and break
  `import cv2` (`_ARRAY_API not found`); re-pin with
  `pip install numpy==1.26.0` afterwards.
- `cvcg_utils` pulls in `igl`/`plyfile`/`imageio`; `imageio[ffmpeg]` is only
  needed for the demo `render.mp4` and is optional for REST3D.

## 5. Download the checkpoints

Weights are not in git. Suggested layout (mirrors `SAM3D_OBJECTS_ROOT`):

```bash
mkdir -p /path/to/checkpoints
cd /path/to/checkpoints
git clone https://www.modelscope.cn/jsnln00/mix3r.git        # v1+v2 safetensors, ~13 GiB
huggingface-cli download microsoft/TRELLIS-image-large \
  --local-dir /path/to/checkpoints/TRELLIS-image-large
```

- `mix3r_v1.safetensors` / `mix3r_v2.safetensors`: the model checkpoints
  (V2 recommended for in-the-wild images).
- `TRELLIS-image-large` **must be a local directory**: `Pipeline.from_pretrained`
  loads `ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors` via a raw
  `load_file(f'{trellis_path}/ckpts/...')` path, which fails if `trellis_path`
  is a HuggingFace repo id.
- DINOv2 and Pi3 weights download automatically into the HF / torch hub caches
  on first run.

## 6. Verify

Import smoke test (a successful import alone is not enough — run a real
inference):

```bash
python - <<'EOF'
import os, sys
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# Blackwell fix: xformers 0.0.31 dispatches to its bundled flash-attn-3
# (Hopper sm_90) kernels first, which fail to launch on sm_120.
import xformers.ops.fmha.dispatch as _d; _d._set_use_fa3(False)
sys.path.insert(0, "$REST3D_ROOT/third_party/mix3r")
import torch
from mix3r_model.pipelines.pipeline_original import Pipeline
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0))
EOF
```

Then run a full inference, e.g. the penguin asset from the submodule:

```bash
cd "$REST3D_ROOT/third_party/mix3r"
python - <<'EOF'
import os, sys, time
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import xformers.ops.fmha.dispatch as _d; _d._set_use_fa3(False)
from cvcg_utils.image import read_rgba
from mix3r_model.pipelines.pipeline_original import Pipeline

pipeline = Pipeline.from_pretrained(
    mix3r_path='/path/to/checkpoints/mix3r/mix3r_v2.safetensors',
    trellis_path='/path/to/checkpoints/TRELLIS-image-large')
pipeline.cuda()
images = [read_rgba(f'assets/penguin/{i:03d}.png') for i in range(7)]
t0 = time.time()
pipeline.run(images, 'out_penguin_v2', seed=None, recenter=False)
print('done in', time.time() - t0, 's')
EOF
```

Reference timings on RTX 5090 D v2: model load ~50 s, `pipeline.run` ~52 s for
7 × 900×1600 images. Outputs land in `out_penguin_v2/` (textured mesh
`sample_mesh_perviewbias.ply`, `out_gs.ply`, per-view point clouds, `slat.npz`).

## Runtime requirements (every invocation)

Two categories: **required for correctness** (omitting them errors out or silently
breaks results) and **optional performance settings**. Once integrated into
REST3D, these will be encapsulated in `rest3d/models/build_mix3r.py` (applied
automatically at process start); when running manually, set them as below.

**Required (every invocation, correctness):**

- `ATTN_BACKEND=xformers`: mix3r defaults to the `flash_attn` backend, which
  raises ImportError at import time when flash-attn is not installed;
- `OPENCV_IO_ENABLE_OPENEXR=1`: **must be set before `import cv2`** (the
  pipeline writes EXR intermediates);
- `xformers.ops.fmha.dispatch._set_use_fa3(False)`: mandatory on sm_120
  (Blackwell), otherwise the first sampling step fails with a CUDA error; on
  other GPUs it can be omitted (dispatch falls back to flash2/cutlass
  automatically), but disabling it unconditionally is harmless;
- the mix3r repo root must be on `sys.path` (mix3r is not pip-installable; it
  is consumed as source).

**Optional (performance only, no effect on correctness):**

- `SPCONV_ALGO=native`: skips spconv's startup algorithm benchmarking
  (`auto` also works but benchmarks first; upstream recommends `native` when
  running only once).

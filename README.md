# REST3D: Reconstructing Physically Stable 3D Scenes from a Single Image

<p align="center">
  <a href="https://shirleymaxx.github.io/">Xiaoxuan Ma</a>&emsp;
  <a href="https://jiashunwang.github.io/">Jiashun Wang</a>&emsp;
  <a href="https://nicolasugrinovic.github.io/">Nicolás Ugrinovic</a>&emsp;
  <a href="https://yehonathanlitman.github.io/">Yehonathan Litman</a>&emsp;
  <a href="https://kriskitani.github.io/">Kris Kitani</a>
</p>

<p align="center">
  Carnegie Mellon University
</p>
  

<p align="center"><a href="https://arxiv.org/abs/2605.30338"><img src="https://img.shields.io/badge/arXiv-REST3D-d55c5c?logo=arxiv&style=flat" alt="arXiv"></a> &nbsp;<a href="https://shirleymaxx.github.io/REST3D/"><img src="https://img.shields.io/badge/Project-Page-6bbf59?logo=googlechrome&style=flat" alt="Project Page"></a> &nbsp;<a href="https://shirleymaxx.github.io/REST3D/#interactive"><img src="https://img.shields.io/badge/Interactive-3D-f2c14e?logo=unity&style=flat" alt="Interactive 3D"></a></p>

<h3 align="center">
  ⚡️ TL;DR: From a single casual image to a visually consistent and physically stable interactive 3D scene.
</h3>
<p align="center">
  <img src="assets/teaser.gif" alt="REST3D teaser" width="100%">
</p>


## News

🚩 **2026.06**: Released the code.

**2026.05**: Released the arXiv paper and project page.


## 🛠️ Installation

Please follow [INSTALL.md](INSTALL.md) for detailed installation instructions.
Two Conda environments are used:

- `rest3d` — scene reconstruction (Stages 1–2)
- `gym` — Isaac Gym stabilization and replay (Stage 3)

After installing both environments, activate them as needed:

```bash
conda activate rest3d   # Stages 1–2
conda deactivate
conda activate gym      # Stage 3
```

## 🚀 Quick start
Follow the steps below to reconstruct a physically stable 3D scene from a single image and interactively inspect object stability in physics simulator.

> Remember to set your API key before running (`GEMINI_API_KEY` — see [INSTALL.md](INSTALL.md)).

**Step 1. Infer 3D scene from a single image**

```bash
conda activate rest3d
# Point Stage 2 at the sam-3d-objects checkout with its downloaded
# checkpoints (see INSTALL.md / environments/install_rest3d.md):
export SAM3D_OBJECTS_ROOT=/path/to/sam-3d-objects
# Optional: if HuggingFace is unreachable, force offline use of the
# locally cached MoGe/SAM3D models:
export HF_HUB_OFFLINE=1
bash 1_infer_scenecanon.sh \
  --input data/xxx.png \
  --output-dir output/runxxx
```
Outputs are saved to `output/runxxx/stage2/scene_canon/`.

> Stage 1 additionally loads the SAM3 model checkpoint from
> `$HOME/sam3/checkpoints/sam3.pt` (path is hard-coded in
> `scripts/infer_scenetree.py`); make sure the file exists there before
> running.

**(Optional) Custom prompt / object list**

To bypass VLM-generated scene object lists, specify your own objects. Before
running Step 1, create `stage1/scene_object_lists.txt` under the run directory,
with one object name per line:

```bash
mkdir -p output/runxxx/stage1
cat > output/runxxx/stage1/scene_object_lists.txt <<'EOF'
table with patterned reddish-brown cloth
black and white sneaker
white shoelace
black U-shaped object
black blocky object
EOF
```

When this file exists, the script uses the listed object names directly (and
appends `the floor` in memory), skipping the VLM object-list generation; all
subsequent segmentation, reconstruction, and scene-tree steps are based on
this list. If the file is absent, the default VLM-generated flow runs instead.

**Step 2. Stabilize the scene**

```bash
conda activate gym
bash 2_stable_scene.sh \
  --input data/xxx.png \
  --output-dir output/runxxx
```
Outputs are saved to `output/runxxx/stage3/`:
- `global_scene/` — physically stable scene
- `global_scene_w_walls/` — (optional) additionally fits walls to the scene and adjusts wall-attached object positions accordingly

**🤗 Visualize and interact with the physically stable scene**

```bash
conda activate gym
bash 3_replay_in_simulator.sh \
  --input data/xxx.png \
  --output-dir output/runxxx
```
The replay settles the scene under gravity, records an MP4 into
`output/runxxx/stage3/global_scene/`, and opens a Viser 3D browser viewer.
Follow the printed URL (e.g. `http://localhost:8082`) to interactively inspect
the physics simulation settling process.

**Running the physics replay without the browser viewer**

The shell wrapper hard-codes `--viser`, so the physics + MP4 recording part can
be run alone by invoking the Python entry point directly (no `--viser`):

```bash
conda activate gym
PYTHONPATH="$(pwd):${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/usr/lib/wsl/lib LD_PRELOAD="${CONDA_PREFIX}/lib/libpython3.8.so.1.0" \
python scripts/replay_in_simulator.py \
  --scene_tree output/runxxx/stage2/scene_tree.json \
  --output_dir output/runxxx/stage3/global_scene \
  --settle_steps 120
```

Note: in this version the Viser viewer is integrated into the replay process
(physics only advances once a browser connects); there is no standalone
viewer-only command.


## Citation
If you find this work useful, please cite:
```bibtex
@article{ma2026rest3d,
  title     = {REST3D: Reconstructing Physically Stable 3D Scenes from a Single Image},
  author    = {Ma, Xiaoxuan and Wang, Jiashun and Ugrinovic, Nicol\'{a}s and Litman, Yehonathan and Kitani, Kris},
  booktitle = {arXiv preprint arXiv:2605.30338},
  year      = {2026}
}
```

## Acknowledgements

This repository builds upon the following excellent open-source projects: [SAM 3](https://github.com/facebookresearch/sam3), [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), and [Isaac Gym](https://developer.nvidia.com/isaac-gym).

## License

This project is released under the [CC&nbsp;BY-NC&nbsp;4.0](https://creativecommons.org/licenses/by-nc/4.0/).
See [`LICENSE`](LICENSE) for details.

**Non-commercial use only.** For commercial licensing, please contact the [author](mailto:xiaoxuam@andrew.cmu.edu). Please note that it also relies on external libraries, which may be subject to their own licenses and terms of use.

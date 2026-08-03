# RUNS 记录

## 2026-08-02 cam_40 gym 后端全流程（y-align 修复后）

运行目录：`output/gym_backend/cam_40_run1`
输入图像：`data/shoes/images/cam_40_frame0.png`
后端：Isaac Gym（Preview 4，`gym` conda 环境，Python 3.8 / PyTorch 2.2.2+cu121）
代码状态：`main`（fbbcc0e 基线）+ y-align 修复（`rest3d/utils/mesh.py` 几何上方向，
commit `321d32b` / merge `76a891f`）；本次运行是在修复提交前用修复代码验证的。
GPU 说明：RTX 5090（sm_120）上 PyTorch 不支持 GPU tensor pipeline，
实际使用 GPU PhysX + CPU tensor pipeline（日志可见）。

### 环境准备

```bash
conda activate gym
export GEMINI_API_KEY=...   # stage1 VLM（gemini）需要
# Isaac Gym 进程需要预加载 libpython3.8（wrapper 脚本已处理，不要全局设置）
```

### 命令（按顺序）

1. Stage 1 + Stage 2（19:44–19:51；stage2 于 21:44 用 y-align 修复重跑）：

```bash
bash 1_infer_scenecanon.sh --input data/shoes/images/cam_40_frame0.png \
  --output-dir output/gym_backend/cam_40_run1
```

等价于：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_scenetree.py \
  --image_folder data/shoes/images/cam_40_frame0.png \
  --output_folder output/gym_backend --output_name cam_40_run1
# y-align 修复后重跑 stage2（读取缓存 raw mesh，重新规范化）：
CUDA_VISIBLE_DEVICES=0 python scripts/infer_scene3d.py \
  --image_folder data/shoes/images/cam_40_frame0.png \
  --output_folder output/gym_backend --output_name cam_40_run1
```

2. Stage 3 CEM 物理稳定化（21:51–22:19，全部参数用默认）：

```bash
bash 2_stable_scene.sh --output-dir output/gym_backend/cam_40_run1
```

等价于：

```bash
PYTHONPATH="$(pwd):${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD="${CONDA_PREFIX}/lib/libpython3.8.so.1.0" \
python scripts/stable_scene.py \
  --scene_dir  output/gym_backend/cam_40_run1/stage2 \
  --output_dir output/gym_backend/cam_40_run1/stage3
```

3. Isaac Gym 回放 + viser 录制（22:19 完成）：

```bash
bash 3_replay_in_simulator.sh --output-dir output/gym_backend/cam_40_run1
```

等价于（wrapper 硬编码 `--viser`）：

```bash
PYTHONPATH="$(pwd):${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD="${CONDA_PREFIX}/lib/libpython3.8.so.1.0" \
python scripts/replay_in_simulator.py \
  --scene_tree  output/gym_backend/cam_40_run1/stage2/scene_tree.json \
  --output_dir  output/gym_backend/cam_40_run1/stage3/global_scene \
  --settle_steps 120 --viser
```

### 参数

Stage 2（y-align，21:44 重跑）：
- 6 个缓存 mesh（5 物体 + floor 参考），重跑 y-align 后全部 upright（cos_y≈1.0）
- 桌子落地面：y 从 [-0.2758, 0.0539] → [0, 0.3297]；4 个物体上桌面（y≈0.3347 起）
- ceiling 高度 1.8 m，非 ceiling 场景最大 Y = 0.3877

Stage 3 CEM（`StableSceneCfg` 默认值）：
- `cem_pop_size=2048`、`cem_episodes=2`、`cem_iters_subtree/joint=15`
- `cem_elite_frac=0.025`、`total_settle_steps=60`、`vel_settle_steps=15`
- V-HACD 开启（table 8 hulls / 其余 16 hulls）、`cem_warm_start=prev_mean_std`

Replay：
- `settle_steps=120`、视频 `fps=30`（默认）、viser 相机默认参数

### 结果

- Stage 3 CEM：local_group best reward **-0.7152**（stab=0.206, layout=0.414,
  vel=0.096, pen=0）；global best reward **-0.0246**（stab=0.004, layout=0.019,
  vel=0.001, pen=0）
- 对比 y-align 修复前：全局 reward 从 -0.56 → -0.0246；局部组从 -11.24（pen=1）
  → -0.72（pen=0）
- Replay：`stage3/global_scene/replay_states.npy`（120 帧 × 5 物体 × 7）+
  `stage3/global_scene/replay_viser.mp4`，物理稳定落盘
- 产物：`stage2/scene_canon/`、`stage3/local_groups/`、`stage3/global_scene/`
  （obj_files + urdf_files）、`stage3/log/`

### 备注

- `stage3/global_scene_w_walls/` 尝试失败（21:55 日志仅 73 字节，无回放产物），
  本次成功结果以 `global_scene` 为准。
- 查看回放：`python scripts/view_replay_viser.py output/gym_backend/cam_40_run1/stage3/global_scene`
  （需 `gym` 环境，浏览器打开 http://localhost:8080）。

## 2026-08-03 cam_74 gym 后端全流程（stage1 提速 + 人工编辑物体清单）

运行目录：`output/gym_backend/cam_74_run1`
输入图像：`data/shoes/images/cam_74_frame0.png`
后端：Stage 1/2 用 `rest3d` 环境；Stage 3 / Replay 用 Isaac Gym（`gym` 环境）
代码状态：`main`（fbbcc0e + y-align 修复）+ stage1 提速（commit `a5eebdb` 的
`rest3d/utils/vlm.py`、`scripts/infer_scenetree.py`，仅代码，未含 .sh/.md/test）

### 命令（按顺序）

```bash
conda activate rest3d
set -a; source .env; set +a
bash 1_infer_scenecanon.sh --input data/shoes/images/cam_74_frame0.png \
  --output-dir output/gym_backend/cam_74_run1

conda activate gym
bash 2_stable_scene.sh --output-dir output/gym_backend/cam_74_run1
bash 3_replay_in_simulator.sh --output-dir output/gym_backend/cam_74_run1
```

### 过程与参数

- Stage1 跑了两轮：首轮 VLM 自动清单 **50.0s**（14:09–14:16）；人工编辑
  `stage1/scene_object_lists.txt` 后重跑 **579.2s**（14:19–14:28，
  VLM 14 请求 / 22 图 / 65.19 MiB / 549.1s）
- 物体清单（人工版）：table / black and white sneaker / white shoelace /
  black U-shaped object / black blocky object + floor
- Stage2：6 物体 SAM3D 重建 + y-align 全部 upright；桌面 y=0.5442，
  4 物体上桌（y≥0.5492），ceiling 1.8 m
- Stage3 CEM 默认参数：`cem_pop_size=2048`、`cem_episodes=2`、`cem_iters=15`、
  `total_settle_steps=60`、`vel_settle_steps=15`、V-HACD 开
- Replay：`settle_steps=120`、视频 `fps=30`

### 结果

- local_group：ALL-TIME BEST reward **-0.4144**（stab=0.098, layout=0.220,
  vel=0.097, pen=0），iter=9
- global：ALL-TIME BEST reward **-0.0155**（stab=0.000, layout=0.012,
  vel=0.003, pen=0），iter=13
- Replay：`stage3/global_scene/replay_states.npy`（120×5×7）+
  `stage3/global_scene/replay_viser.mp4`（518 KB）
- 耗时：stage1 ~9.7 min + stage2 ~1.3 min + stage3 ~6 min + replay ~3 min

### 备注

- 首次 viser 录制因浏览器标签页切到后台卡死（`get_render` 无限等待），
  kill 后保持标签页前台重跑成功（15:15–15:16）。
- 查看回放：`python scripts/view_replay_viser.py output/gym_backend/cam_74_run1/stage3/global_scene`

## 2026-08-02 cam22_support_contact_v2 isaac-lab 后端（归档分支）

运行目录：`output/fix_scene_support/cam22_support_contact_v2`
输入场景：cam22（坐姿人物 + 办公椅 + 桌子 + 键盘/鼠标/钳子，6 物体）
后端：Isaac Lab（Isaac Sim 5.1）；代码为**归档分支**产物（`rest3d.sim` 流程，
main 已无此代码，但 replay 数据可用查看脚本直接播放）

### 说明

- 本目录无 stage1（复用之前 cam22 的 stage1 结果），从 stage2 开始（08-02 16:55）
- stage3 分 `pipeline_v1`（CEM 物理优化）与 `replay_v1_long`（300 步长回放）
- 具体执行命令在归档分支上完成，未在本目录留档；结果文件完整

### 参数（`pipeline_v1/stage3_pipeline_results.json`）

- local：4 envs × 1 iter；global：4 envs × 1 iter；`sampled_entity_count=2`；
  `global_seed=18`
- 环境版本：Python 3.11.15 / Isaac Sim 5.1.0.0 / Isaac Lab 0.54.2 /
  torch 2.7.0+cu128 / CUDA 12.8 / driver 591.86

### 结果

- stage3 pipeline：`passed=True`，`global_best_reward=-1.645`，
  `total_seconds=30.6s`，`global_peak_gpu_memory_mib=3652`
- replay_v1_long：`passed=True`、`scene_stable=True`、301 帧 × 6 物体、dt=1/60
  - 稳定性判定：评估步 240、4.0 s、位移阈值 0.1 m / 旋转 0.1 rad，
    6/6 稳定，`unstable_object_names=[]`
  - 仿真：total 8.9 s（sim 3.77 s，79.6 steps/s），
    `system_gpu_memory_used_mib_peak=3560`
- 产物：`stage3/pipeline_v1/`（CEM 结果、physics_assets、local_groups）、
  `stage3/replay_v1_long/`（replay_results.json、replay_states_rest.npy、
  stability_metrics.npz）、`visual_audit/`

### 备注

- 查看回放：`python scripts/view_replay_viser.py output/fix_scene_support/cam22_support_contact_v2/stage3/replay_v1_long`
- 显存峰值（3.5–3.6 GB）仅为 replay/CEM 阶段记录；gym 流程（cam_40/cam_74）
  日志无显存采样

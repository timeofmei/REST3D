# Isaac Lab Stage 3 使用与回归指南

本文说明如何在不修改旧 `gym` 环境的前提下，使用 Isaac Lab 后端运行 REST3D
Stage 3。迁移的设计和逐阶段证据见
[《REST3D Stage 3：Isaac Lab 迁移计划》](./2026-8-1-Issac-Lab.md)，环境创建细节见
[Isaac Lab 独立环境安装说明](../environments/install_isaaclab.md)。

## 1. 已验证的运行组合

| 组件 | 已验证版本 |
| --- | --- |
| Isaac Lab 源码 | `v2.3.2`，commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Isaac Sim | `5.1.0.0` |
| Python | `3.11.15` |
| PyTorch | `2.7.0+cu128` |
| CUDA runtime / NVRTC | `12.8 / 12.8` |
| Viser / WebSockets | `0.2.11 / 12.0` |
| GPU / driver | RTX 5090 D v2，compute capability 12.0；实测驱动 `591.86` |
| 主机 | WSL2，headless 物理模式 |

这是本机实际通过的迁移基线，不代表 NVIDIA 官方支持 WSL。Isaac Sim 5.1 已停止
维护；后续升级必须重新运行本文的完整回归，不能仅凭 import 成功判定兼容。

旧后端仍使用独立 `gym` 环境的 Isaac Gym Preview 4。RTX 5090 上它保持 GPU PhysX，
但因其 PyTorch wheel 不支持 `sm_120`，root-state tensor pipeline 必须在 CPU。新后端
在独立 `isaaclab` 环境中使用 GPU PhysX、GPU broadphase 和 CUDA tensor pipeline。

## 2. 后端入口

必须从仓库根目录运行。两个顶层入口都接受同一 backend 名称：

```text
--backend isaac-gym
--backend isaac-lab
```

`isaac-gym` 是兼容默认值；`isaac-lab` 使用新链路。不要在同一个 Python 进程中同时
导入两个 runtime，仓库 wrapper 会分别启动相应环境。

### 从新图片运行完整 Stage 3

Stage 1–2 仍在 `rest3d` 环境完成。用户只需在原流程中指定要提取的对象；Stage 3 从
`stage2/scene_tree.json` 和 `stage2/scene_canon/` 自动发现对象，并从几何统一推导
质量、惯量和碰撞资产，不需要编写逐对象物理配置。

假设 Stage 1–2 的新结果位于 `output/NEW_IMAGE_RUN/stage2/`，且
`output/NEW_IMAGE_RUN/stage3/` 尚不存在：

```bash
conda activate isaaclab
bash 2_stable_scene.sh \
  --output-dir output/NEW_IMAGE_RUN \
  --backend isaac-lab
```

这条命令依次完成通用物理资产生成、局部组 CEM、全对象全局 CEM，并输出
`stage3/stage3_pipeline_results.json`。全局 CEM 只采样组根 6-DoF，但每个候选都包含
全部后代和独立对象，所有对象共同进入 PhysX 与稳定、布局、速度、GJK 穿插能量。

调试局部阶段时可以显式停止在 D：

```bash
bash 2_stable_scene.sh \
  --output-dir output/NEW_IMAGE_RUN \
  --backend isaac-lab \
  --local-groups-only \
  --cem-pop-size 16 \
  --cem-iters-subtree 2
```

`stage3/` 必须不存在；调试运行应使用另一个完整的新 run 目录，不能复用正式结果。

### 最终 60 帧验证

最终 replay 必须写入新的、尚不存在的目录：

```bash
bash 3_replay_in_simulator.sh \
  --output-dir output/NEW_IMAGE_RUN \
  --backend isaac-lab \
  --replay-output-dir output/isaaclab_migration/NEW_FINAL_REPLAY \
  --settle-steps 120 \
  --require-stable \
  --viser
```

Isaac Lab 入口会自动读取 `stage3/physics_assets/urdf_files/` 和
`stage3/global_cem/best_global_candidate_states.json`。报告在第 60 帧用论文门槛逐对象
判断：位移超过 `0.1 m` 或旋转超过 `0.1 rad` 即不稳定。`replay_results.json` 中的
`scene_stable`、`stable_object_count`、`unstable_object_names` 和 `objects` 是最终
判断依据。

`--viser` 会在物理完成并保存 `replay_states_rest.npy` 后，使用独立浏览器前端显示
真实轨迹。打开 `http://localhost:8080`，右上角采用原作者的 `Frame`、`FPS`、
`▶ Play` 三项 replay 控件；播放到末帧后会回到第 0 帧并自动停止。关闭时在启动命令
的终端按 `Ctrl-C`。Viser 进程只读取
OBJ、结果 JSON 和轨迹，不导入 Isaac Gym，也不会重新计算或改变 Isaac Lab 物理。

已经完成 replay 时无需再次运行物理，可直接查看现有目录：

```bash
bash scripts/run_viser_replay.sh \
  output/isaaclab_migration/EXISTING_REPLAY \
  --port 8080 --fps 30
```

wrapper 优先使用当前环境，然后查找独立 `isaaclab` 环境。这里固定使用 Viser
`0.2.11`：它是最后一个允许 Isaac Sim 5.1 强制要求的 `websockets==12.0` 的版本；
Viser `0.2.12` 及以后要求 `websockets>=13.1`，不能装入该 Isaac Lab 基线。安装时还
必须锁住 NumPy 和 typing-extensions，精确命令见独立环境安装说明。其他兼容的独立
Python 可用 `VISER_PYTHON=/path/to/python` 显式选择；不要让 pip 自动升级这些核心包。

### 旧 Isaac Gym 回归

旧后端仍可用，并且现在同时支持无前缀的旧 `global_scene` 和 Stage 2 常见的带前缀
资产文件名：

```bash
conda activate gym
bash 3_replay_in_simulator.sh \
  --output-dir output/OLD_OR_GYM_RUN \
  --backend isaac-gym \
  --replay-output-dir output/isaaclab_migration/NEW_GYM_REPLAY \
  --settle-steps 120
```

旧 Gym 的 `libpython3.8` 和 WSL driver 路径只由子进程临时设置；不要把 gym 的
`lib/` 永久加入 `LD_LIBRARY_PATH`。

## 3. 分层可复现命令

以下所有 `NEW_*` 路径都必须不存在：

```bash
# GPU PhysX + CUDA tensor smoke
bash scripts/run_isaaclab_smoke.sh \
  output/isaaclab_migration/NEW_SMOKE

# 任意完整场景 replay
bash scripts/run_isaaclab_replay.sh \
  output/NEW_IMAGE_RUN/stage2/scene_tree.json \
  output/NEW_IMAGE_RUN/stage2/scene_canon \
  output/isaaclab_migration/NEW_REPLAY \
  --urdf-dir output/NEW_IMAGE_RUN/stage3/physics_assets/urdf_files \
  --initial-states output/NEW_IMAGE_RUN/stage3/global_cem/best_global_candidate_states.json \
  --collision-approximation convex_decomposition \
  --settle-steps 120 \
  --stability-evaluation-steps 60 \
  --position-stability-threshold 0.1 \
  --rotation-stability-threshold 0.1 \
  --require-stable

# 完整局部组；加 --run-global 会继续全局 CEM
bash scripts/run_isaaclab_local_groups.sh \
  output/NEW_IMAGE_RUN/stage2 \
  output/isaaclab_migration/NEW_STAGE3 \
  --num-envs 64 --cem-iters 15 --run-global

# 两后端同输入 replay 与公共 CUDA 全对象能量比较
bash scripts/run_stage3_backend_comparison.sh \
  output/NEW_IMAGE_RUN/stage2/scene_tree.json \
  output/NEW_IMAGE_RUN/stage2/scene_canon \
  output/isaaclab_migration/NEW_BACKEND_COMPARISON \
  --steps 120 --early-step 15 --evaluation-step 60
```

`2_stable_scene.sh --backend isaac-lab` 的正式默认搜索规模与原作者
`StableSceneCfg` 一致，为局部/全局各 2048 个并行候选、15 次迭代。上面的 64 环境命令
是显式的小规模开发检查，不是正式默认值。2048 环境下全场景 convex decomposition
会超过 Isaac Lab 默认的 GPU PhysX rigid-patch 容量；后端现在按环境数和每环境刚体数
自动使用通用的 2 次幂容量，并把容量及是否发生 overflow 写入结果门禁。

在 RTX 5090/WSL 上完成的 2048×15 六对象完整 `重跑2` 结果位于
`output/isaaclab_migration/cam22_highspec_2048x15_v2`：局部和全局均通过，全局
state/contact tensor 均为 `cuda:0`，GPU PhysX patch 容量为 262144，日志中没有 PhysX
error，峰值显存 13862 MiB，总耗时 142.6 秒。

局部/全局 runtime 的底层脚本也可独立调用，但需要物理资产、组计划和状态 JSON；
正常使用优先选择 `2_stable_scene.sh` 或 `run_isaaclab_local_groups.sh`，避免手工拼接
不一致的中间输入。

## 4. 一条命令的 Stage G 回归

下列命令会创建任意名称的三对象 Stage 2 输入，并实际运行：纯 Python 测试、三个
Python 环境的编译检查、shell 语法检查、4 环境局部 CEM、4 环境全局 CEM、60 帧
最终 Lab replay，以及 Gym/Lab 同输入比较。它会校验输入哈希保持不变。

```bash
bash scripts/run_stage3_regression.sh \
  output/isaaclab_migration/NEW_STAGE_G_REGRESSION
```

正式实测结果位于：

```text
output/isaaclab_migration/stage_g_generic_regression_v4/
```

该次 `51 passed`，G 汇总的 14 项门禁和双后端比较的 19 项门禁全部通过；局部组数
为 1、全局采样实体数为 2、并行环境数为 4，全局仿真 `0.343 s`、峰值显存
`4088 MiB`，最终 Lab replay 在强制稳定门禁下 `3/3` 通过。旧 Gym 为 GPU PhysX +
CPU tensor，Lab 为 GPU PhysX + GPU tensor，公共能量实际在 `cuda:0` 计算。

## 5. 输出和日志契约

所有 wrapper 都拒绝覆盖既有输出。物理运行至少在其结果 JSON 中记录对象集合、版本、
设备、tensor shape、稳定阈值、逐对象指标、耗时和显存。主要入口如下：

| 运行 | 主要结果 |
| --- | --- |
| smoke | `smoke_results.json`、`smoke.log`、`kit.log` |
| 完整 Stage 3 | `stage3_pipeline_results.json`、`local_group_pipeline_results.json`、`global_cem/real_global_cem_results.json` |
| replay | `replay_results.json`、`replay_states_rest.npy`、`stability_metrics.npz`、`replay.log`、`kit.log` |
| 后端比较 | `comparison/metrics.json`、两后端轨迹和 stdout 日志 |
| Stage G 回归 | `command.txt`、`environment.json`、`run.log`、`metrics.json`、`states_initial.npy`、`states_final.npy` |

`environment.json` 记录 Git、主机、版本和设备；`metrics.json` 记录门禁、对象集合、
并行环境、耗时、峰值显存和稳定结果。失败目录不会被删除或复用，诊断应从对应
`run.log`、`*.stdout.log`、`kit.log` 和 `*_failure.json` 开始。

## 6. Headless 与 WSL 已知限制

这里的 headless 指“不打开窗口、不渲染画面，只运行物理和 tensor 计算”。它适合
服务器、WSL 和自动回归；它不等于 CPU 模式。本机 headless 日志仍会报告 Vulkan、
renderer、GLFW 或 X Server 不可用，因为 WSL 当前只能枚举 llvmpipe。这些图形告警
不代表 PhysX 回退：必须以结果 JSON 中的 GPU simulation、GPU broadphase、CUDA
state/contact tensor、`sm_120` 实算门禁为准。

当前已验证的是 headless 物理链路，没有验证 Isaac Sim GUI、RTX 渲染或相机视频。
保存的物理轨迹可以由 Viser 浏览器交互查看；这不等同于 Isaac Sim renderer，也不
依赖 Vulkan。若需要 RTX 画面或 Isaac Sim 原生 viewer，应先在 NVIDIA 官方支持的
原生 Linux/Windows 主机验证 Vulkan/显示路径；不要为此直接修改驱动、系统 Vulkan、
Docker daemon 或现有 Conda 环境。

# REST3D Stage 3：Isaac Lab 迁移计划

日期：2026-08-01
状态：A–B 阶段已完成；C 阶段验证实现已完成，但实际场景的通用凸分解验证未通过；
D 阶段已启动，公共 CEM 与合成批量 GPU smoke 已完成，真实局部组资产迁移尚未完成；
E–G 阶段尚未开始。

## 1. 目标与边界

目标是在不移除 Isaac Gym Preview 4 后端的前提下，为 REST3D Stage 3 增加
Isaac Sim + Isaac Lab 后端，使 RTX 5090 能运行 GPU PhysX 和完整 GPU tensor
pipeline，并修正全局 CEM 未让组内子对象共同参与物理评估的问题。

实施必须满足以下边界：

- 只在 `/home/yangyankun/REST3D-isaac-lab` worktree 和
  `experiment/isaac-lab-migration` 分支工作；
- 不读取后写回、不切换、不修改 `/home/yangyankun/REST3D` 主工作树；
- 保留 `rest3d` 和 `gym` Conda 环境原状，Isaac Lab 使用独立环境；
- 未经明确要求不提交、推送、合并、rebase、amend 或改写历史；
- `output/cam_22_foreground_run1` 只作为输入；任何测试都必须指定全新输出目录，
  且程序应拒绝覆盖已存在的非空结果目录；
- 实现不能包含当前场景的具体对象名，也不能按对象名称字符串猜测质量、惯量、
  碰撞体或绑定关系；
- 不以“一律关闭父子碰撞”作为关系处理策略；承托关系中的对象仍必须正常碰撞；
- 遇到驱动修改、系统级大规模安装、数据删除、额外权限或宿主机迁移需求时停止，
  先汇报证据和所需操作。

## 2. 开始前检查结果

已完整阅读仓库根目录 `AGENTS.md`，并执行只读的 worktree/Git 检查：

| 检查项 | 结果 |
| --- | --- |
| `pwd -P` | `/home/yangyankun/REST3D-isaac-lab` |
| 当前分支 | `experiment/isaac-lab-migration` |
| 当前 HEAD | `a5eebdb` — `Speed up Stage 1 scene inference` |
| 主工作树 | `/home/yangyankun/REST3D`，分支 `main`，未触碰 |
| 迁移 worktree | `/home/yangyankun/REST3D-isaac-lab`，目标分支正确 |
| 检查时状态 | 计划文件 `doc/2026-8-1-Issac-Lab.md` 为既有未跟踪空文件；无其他改动 |
| 最近提交 | `a5eebdb`、`fbbcc0e`、`57560a1`、`fe4bcb5`、`90c495a` |

后续每个阶段开始前都要再次检查路径、分支和 `git status --short --branch`。若路径或
分支不符合上述值，立即停止，不自动切换分支。

计划编写时没有执行 GPU、Vulkan、Isaac Sim 或 Isaac Lab 命令。用户随后明确要求
开始 A 阶段；实际执行结果记录在第 6 节的“A 阶段执行结果”中。

## 3. 官方兼容性调研与版本决策门

### 3.1 已确认的官方信息

- Isaac Lab 官方兼容表说明 `v2.3.x` 支持 Isaac Sim 4.5、5.0、5.1；Isaac Lab
  3.0 对应 Isaac Sim 6.0/6.0.1：
  [Isaac Lab version dependency](https://github.com/isaac-sim/IsaacLab#isaac-sim-version-dependency)。
- Isaac Lab `v2.3.2` 是 2.x 的最后一个非 beta 版本；当前 3.0 发布仍明确标为
  beta，并提示可能有破坏性变更和性能回归：
  [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases)。
- Isaac Sim 5.0/5.1 系列采用 Python 3.11；Isaac Lab 2.3 的官方发布说明记录
  Isaac Sim 5.0 使用 PyTorch 2.7.0+cu128，并明确包含 Blackwell 支持：
  [Isaac Lab release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0)。
- Isaac Sim 5.1 的要求页包含 Blackwell GPU，并给出当时测试的 Linux 驱动版本，
  但该文档现在同时将 5.1 标记为不再支持：
  [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。
- Isaac Sim 6.0.1 当前 pip 文档要求 Python 3.12，并示例安装 PyTorch 2.11.0
  cu128/cu130：
  [Isaac Sim Python installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)。
- 官方对 Isaac Lab 的常规推荐安装方式是 pip 安装 Isaac Sim、源码安装 Isaac Lab；
  容器是 headless Linux 部署的备选：
  [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index)。
- Isaac Lab 2.3.2 的 `SimulationCfg.device` 支持 `cuda:N`，其仿真数据接口使用
  PyTorch tensor：
  [Isaac Lab simulation API](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.html)。
- 官方支持矩阵列出原生 Ubuntu 和 Windows，没有把 WSL 列为受支持 OS；最新容器
  文档还明确说明 Windows host（包括 WSL）不受支持。因此 WSL 运行只能视为待实测
  路径，不能在完成 A 阶段前宣称受官方支持：
  [Isaac Sim container installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)。
- REST3D 论文 §3.3.2 明确说明：全局候选只更新组根，但每个候选应把全部组内
  子对象放入 simulator，与其余对象联合评估；子对象姿态由新组根和局部相对姿态
  组合得到：
  [REST3D §3.3.2](https://arxiv.org/html/2605.30338#S3.SS3.SSS2)。

### 3.2 当前版本矛盾

截至 2026-08-01，官方资料中没有一个同时满足“Isaac Lab 非 beta”和“其配套
Isaac Sim 仍在官方支持期”的无争议组合：

| 候选 | 优点 | 风险 |
| --- | --- | --- |
| Isaac Lab 2.3.2 + Isaac Sim 5.1 + Python 3.11 + PyTorch cu128 | Isaac Lab 最后一个非 beta 2.x；API 稳定；Blackwell 路径明确 | Isaac Sim 5.1 已停止支持 |
| Isaac Lab 3.0 beta2 patch1 + Isaac Sim 6.0.1 + Python 3.12 | Isaac Sim 当前受支持；新架构面向后续版本 | Isaac Lab 仍为 beta；数据从 Torch 转向 Warp，迁移范围和回归风险明显更大 |

因此暂定将 `Isaac Lab 2.3.2 + Isaac Sim 5.1 + Python 3.11 + 官方 cu128
PyTorch` 作为低风险基线候选，但不在计划阶段锁死具体 wheel。A 阶段先从各 tag 的
官方 lock/依赖文件记录精确版本，再分别进行兼容性检查；满足下列决策条件后才锁定：

1. 能在当前 WSL headless 启动；
2. RTX 5090 上 GPU PhysX 实际步进；
3. simulator 状态数据位于 `cuda:0`；
4. 状态数据参与的真实 PyTorch CUDA 运算成功；
5. 无需修改宿主驱动或污染现有环境；
6. API 足以实现批量刚体、root state 写回和读取。

如果 2.3.2/5.1 满足全部条件，先用它完成迁移，之后单列 3.x 升级工作。如果它因
已知缺陷或 WSL 图形栈无法运行，再评估 3.0 beta/6.0.1；不能仅因为版本号较新就
接受 beta。若两者在 WSL 都被阻塞，保留完整日志并停止，向用户提出官方支持的
原生 Ubuntu 或 Windows 路径。由于官方已说明 Windows/WSL host 不支持容器，不能
把 WSL Docker 当作默认补救方案。

## 4. 已发现的现有实现问题

当前 Stage 3 的仿真代码直接依赖 Isaac Gym：

- `rest3d/models/cem_opt.py` 顶层导入 `isaacgym`，同时包含通用 CEM、资产加载、
  仿真创建、批量环境和 tensor 操作；
- `scripts/replay_in_simulator.py` 也直接构建 Isaac Gym sim、asset 和 actor；
- `2_stable_scene.sh` 与 `3_replay_in_simulator.sh` 无条件要求 Python 3.8
  `libpython` preload，无法承载第二后端；
- `scripts/stable_scene.py::global_scene_optimize()` 把 `actor_filter` 设置为组根和
  非组成员单体，导致组内子对象没有被创建为全局候选环境的 actor；
- 当前全局稳定性、速度和几何穿插项只遍历 `entity_names`，因此即使导出阶段能由
  组根重建子对象位姿，子对象仍没有进入 PhysX 或最终能量；
- 资产碰撞参数中已有依据对象名称关键字选择 V-HACD hull 数量的逻辑。这类策略
  不能扩展到任意新图片，应替换为几何复杂度、尺寸、体积和场景语义驱动的规则；
- 当前脚本把“读取已有 Stage 3 结果”和“写 replay 结果”放在同一目录语义中，
  容易覆盖已有结果。新接口必须分离 `--scene-dir`/`--state-dir` 与 `--output-dir`。

## 5. 目标架构

### 5.1 公共数据与后端边界

先从现有代码提取后端无关的数据结构，不复制整套业务逻辑：

- `SceneSpec`：对象、场景树、支持关系、fixed/movable 属性、资产路径、尺度和参考
  位姿；
- `RigidAssetSpec`：视觉网格、碰撞近似、质量、质心、惯量、材质和刚体属性；
- `SceneStateBatch`：统一的 `[num_envs, num_objects, 13]` root state 视图，包含
  position、quaternion、linear velocity 和 angular velocity；
- `SimulationBackend`：创建/销毁仿真、加载资产、创建批量环境、写入 root state、
  物理步进、刷新状态、读取接触数据和设备信息；
- `PhysicsEvaluation`：稳定性、速度、穿插和布局项，只消费公共 tensor/数组，不
  直接导入 Isaac Gym 或 Isaac Lab。

建议新增后端模块（实际路径在 B 阶段根据最小改动原则确认）：

```text
rest3d/sim/
  types.py
  scene_spec.py
  backend.py
  isaac_gym_backend.py
  isaac_lab_backend.py
```

Isaac Gym 要求在 Torch 前导入，Isaac Lab 要求先启动 `AppLauncher` 再导入部分
模块，因此两个后端必须延迟导入，入口层不能同时顶层导入两套 runtime。必要时用
各自的薄入口进程隔离环境，但业务数据和评估公式仍共享。

### 5.2 坐标与 quaternion 约定

现有 Stage 3 是 Y-up；Isaac Sim/Lab 常见示例是 Z-up。迁移时不在业务代码各处
临时换轴，而是在 backend 边界定义唯一、可逆的坐标变换，并记录：

- 世界 up 轴和重力方向；
- 长度单位（必须为米）；
- mesh 本地坐标与 actor/root 坐标；
- quaternion 分量顺序和乘法方向；
- angular velocity 单位。

Isaac Lab 2.x 和 3.x 的 quaternion/data API 有变更，适配器必须绑定选定版本并用
单位 quaternion、90° 单轴旋转和位姿往返测试验证，禁止靠观察 replay 猜测。

### 5.3 通用物理策略

- fixed/movable 由场景树的 `type`、`relation` 和根节点语义决定，不由对象名决定；
- `on`/承托关系必须保留父子碰撞；普通相邻对象也默认碰撞；
- 只有明确表达刚性附着的 `attach`/`hang` 关系才允许使用 fixed joint、kinematic
  约束或精确的 pair filter，且策略需记录在场景关系中；
- 不使用全局 parent-child collision disable；
- collision approximation 根据 mesh 是否封闭、面数、凹度、尺寸比例和动态/静态
  类型选择 convex hull、convex decomposition 或适用的 SDF/mesh collider；
- 密度采用通用默认值或未来的类别元数据，质量由有效体积推导并设置上下限；非封闭
  网格使用可复现的包围体回退；
- 质心和惯量优先由经过清理的 collision mesh 计算，退化轴采用数值下限，所有
  fallback 写入日志；
- 人物等非刚体重建在当前阶段仍按通用刚体资产处理，除非输入语义明确标为固定或
  附着；不能仅凭对象名特殊冻结。

### 5.4 CLI 与输出安全

两个入口都增加：

```text
--backend isaac-gym
--backend isaac-lab
--output-dir <explicit-new-directory>
```

原则：

- `isaac-gym` 保持当前默认行为和 `gym` 环境兼容；
- `isaac-lab` 只在新 `isaaclab` 环境运行；
- shell wrapper 只在 `isaac-gym` 分支设置 WSL driver path 和 Python 3.8 preload；
- 任何 backend 都不得持久修改 `LD_LIBRARY_PATH`；
- replay 输入与输出参数分离；
- 测试命令必须显式给出新目录，程序以原子方式创建并在已存在时失败；
- 日志、状态 tensor、视频和指标只写到该次新目录；
- 不复制或修改只读输入目录中的任何文件。

## 6. 分阶段实施计划

### A. 独立环境与 RTX 5090 GPU smoke test

目标：在不修改 `rest3d`、`gym` 或宿主驱动的条件下，确定可用版本组合并证明完整
GPU 链路。

步骤：

1. 重新核验 worktree、分支和 Git 状态。
2. 只读记录 WSL/Ubuntu/kernel、glibc、GPU、驱动、显存、CUDA driver、Vulkan
   枚举、磁盘空间和现有 Conda 环境列表。
3. 对照所选 tag 的官方依赖文件生成精确版本清单；不复用现有环境。
4. 创建独立 `isaaclab` Conda 环境。优先使用官方推荐的 Isaac Sim pip + Isaac Lab
   固定 tag 源码安装方式；不写入全局 shell profile。
5. 运行官方 compatibility checker 和最小 headless 启动；保存完整 Kit 日志。
6. 新增最小 smoke 脚本：创建地面和一个简单动态刚体，在 `cuda:0` 上运行至少
   60 个真实 PhysX step。
7. 从 simulator 读取真实 root-state tensor，确认 `device == cuda:0`；使用该状态
   执行至少一个真实 PyTorch CUDA 运算（如位移范数和矩阵运算），同步并检查结果。
8. 记录 `torch.cuda.get_arch_list()`、GPU compute capability、sim device、tensor
   device、tensor shape、首末位姿、运行耗时、并行环境数、峰值 Torch 显存和进程
   峰值显存。
9. 关闭 app 后检查退出码和日志中的 GPU/PhysX 错误，不能把成功 import 当作通过。

建议的新输出目录示例：

```text
output/isaaclab_migration/smoke_20260801_v1
```

该目录在运行前必须不存在。A 阶段验收条件是：headless 启动成功、刚体受重力落地、
GPU PhysX 有明确配置/日志证据、root state 位于 CUDA、基于该状态的 PyTorch CUDA
kernel 成功、结果日志可复现。任一项失败均不进入 B 阶段。

#### A 阶段执行结果（2026-08-01）

A 阶段已完成，B 阶段没有启动。所有安装均位于新的 Conda 环境 `isaaclab`，没有
修改 `rest3d` 或 `gym` 环境，也没有修改宿主驱动或系统 Vulkan 配置。

锁定并实测的版本如下：

| 项目 | 实测值 |
| --- | --- |
| Isaac Lab | 源码 tag `v2.3.2`，commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Isaac Sim | `5.1.0.0`，Kit `107.3.3` |
| Python | `3.11.15` |
| PyTorch | `2.7.0+cu128` |
| torchvision / torchaudio | `0.22.0+cu128` / `2.7.0+cu128` |
| CUDA runtime | `12.8` |
| GPU | `NVIDIA GeForce RTX 5090 D v2`，compute capability `12.0` |
| Windows NVIDIA driver | `591.86` |
| WSL | Ubuntu 22.04.5，kernel `6.18.33.2-microsoft-standard-WSL2` |

官方 compatibility checker 在 headless 模式退出码为 0，结果为 `PASSED`。第一次
实际 PhysX smoke 在未设置 `LD_LIBRARY_PATH` 时记录到 `omni.physx handle on CUDA
lib is (nil)`，随后 GPU solver 和 GPU broadphase 都明确回退到 software，并在 300
秒后超时。该失败被保留为证据，没有覆盖后重跑。

原因是 WSL 的 `/usr/lib/wsl/lib/libcuda.so` 没有被默认动态库查找路径找到；直接用
`ctypes.CDLL("libcuda.so")` 可复现失败，而只对进程设置
`LD_LIBRARY_PATH=/usr/lib/wsl/lib` 后加载成功。最终 smoke 只对 Isaac Lab 子进程
临时注入该路径，没有写入 shell profile。成功运行的 Kit 日志显示：

```text
omni.physx handle on CUDA lib is 0x...
Using CUDA device ordinal 0.
```

同一日志中没有 `GPU solver pipeline failed` 或 `GPU Bp pipeline failed`。运行时配置
和数值结果为：

| 指标 | 结果 |
| --- | ---: |
| headless 退出码 | `0` |
| 并行环境 / 动态刚体 | `1 / 1` |
| PhysX simulation / tensor pipeline / broadphase | `GPU / GPU / GPU` |
| root-state tensor | shape `[1, 13]`，device `cuda:0` |
| PyTorch architecture | 包含 `sm_120` 和 `compute_120` |
| 真实物理步数 | `120`，`dt = 1/60 s` |
| 刚体初始 / 最终高度 | `0.991825 m / 0.100000 m` |
| 最终线速度 | `0.000245 m/s` |
| 状态驱动 CUDA 结果 | `0.901825`，有限值 |
| smoke 内计时 | `2.935 s`，`40.885 steps/s` |
| PyTorch allocator 峰值 | `8,531,968 bytes` |
| 整卡显存采样峰值 | `4,182 MiB` |

成功结果位于：

```text
output/isaaclab_migration/smoke_20260801_v1/gpu_physx_verified_v2/
```

其中 `smoke_results.json` 是机器可读结果，`smoke.log` 是状态摘要，`kit.log` 保存
PhysX/CUDA 证据。兼容性与两轮失败日志也保留在同一 smoke 根目录下。可复现入口为
`scripts/run_isaaclab_smoke.sh`；独立安装与命令见
`environments/install_isaaclab.md`。

当前限制：WSL 中 `vulkaninfo` 只枚举到 llvmpipe，Kit 的 graphics foundation 因此
报告无法创建 Vulkan GPU device。该错误没有阻止 headless CUDA PhysX 和 CUDA
tensor smoke 通过，但渲染尚未通过验证。在不修改驱动或系统 Vulkan 配置的前提下，
B 阶段只能先做无渲染 headless replay；若任务需要视频或相机渲染，应先单独解决或
在官方支持的宿主环境验证 Vulkan 路径。

### B. Isaac Lab replay backend

目标：先建立单环境、全对象、可回放的最小完整链路，不移植 CEM。

步骤：

1. 抽取公共场景读取、对象列表、物理属性和最终状态格式。
2. 给 replay CLI 增加 backend 选择，并保留原 Isaac Gym 入口行为。
3. 实现 Isaac Lab app 生命周期、ground、资产加载、actor 创建、root state 写入、
   step、state 读取和资源释放。
4. 在 backend 边界完成坐标/up-axis/quaternion 转换和往返测试。
5. 所有场景对象都必须进入 simulator；固定边界与六个目标对象分别计数并写入日志。
6. 先用合成的两个刚体场景测试，再读取实际 Stage 3 pose 进行 replay。
7. 视频为可选项；headless 状态验证不能依赖 GUI 或视频成功。

实际场景入口参数如下：

```bash
conda activate isaaclab
python scripts/replay_in_simulator.py \
  --backend isaac-lab \
  --scene-tree output/cam_22_foreground_run1/stage2/scene_tree.json \
  --scene-dir output/cam_22_foreground_run1/stage3/global_scene \
  --output-dir output/isaaclab_migration/cam22_replay_v1 \
  --headless --settle-steps 60
```

`output/cam_22_foreground_run1` 全程只读，示例输出目录必须在运行前不存在。

#### B 阶段执行结果（2026-08-01）

B 阶段已完成，C 阶段没有启动。replay 入口现在通过 `--backend isaac-gym` 或
`--backend isaac-lab` 启动隔离的后端进程，避免两套 runtime 的导入顺序和 Python
版本冲突。两个后端共享严格验证的 replay scene spec；输入和输出已拆分为
`--scene-dir` 与 `--output-dir`，输出目录存在或位于只读输入目录内时直接失败。

公共输入层完成了以下工作：

- OBJ、URDF 与 scene tree 对象集合严格匹配，缺失或未声明资产不再静默跳过；
- fixed/movable 只依据 scene tree 的 `type` 和 `attach`/`hang` 关系，不依据对象名；
- REST3D Y-up 与 Isaac Lab Z-up 使用固定右手坐标变换；位置、四元数、线速度和角
  速度均有往返测试；
- fixed 资产在 Isaac Lab 中使用保留碰撞的 kinematic rigid body，不关闭与子对象
  或其他对象的碰撞；
- 每次 replay 保存完整的初末/逐帧 root state、对象集合、device、耗时和显存数据。

实现过程中发现 PyTorch cu128 wheel 目录同时包含 NVRTC 12.6 主库与 12.8 alternate
库。Kit 在 Torch 之前加载 12.6 后，`RigidObjectData` 的首个 `sm_120` JIT kernel 会
报 `invalid value for --gpu-architecture`。当前机器已有 CUDA Toolkit 12.8，因此
wrapper 只对 Isaac Lab replay 子进程预加载
`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12`；没有修改 Conda 环境、
动态链接器配置或 shell profile。修复后真实 NVRTC CUDA kernel 与 replay 均通过。

合成场景验证包含一个固定碰撞支撑体和一个动态刚体：

| 指标 | 结果 |
| --- | ---: |
| 对象 / 固定 / 可移动 | `2 / 1 / 1` |
| 状态 tensor | `[121, 2, 13]`，`cuda:0` |
| 动态体位移 | `0.300000 m`，落到支撑面后静止 |
| 固定体位移 / 最终速度 | `0 m / 0 m/s` |
| 120 步耗时 | `0.928 s`，`129.249 steps/s` |
| 整卡显存采样峰值 | `3949 MiB` |

只读实际输入 `output/cam_22_foreground_run1/stage3/global_scene` 的 replay 结果为：

| 指标 | 结果 |
| --- | ---: |
| OBJ / URDF / 已加载对象 | `6 / 6 / 6` |
| 状态 tensor | `[121, 6, 13]`，`cuda:0` |
| GPU simulation / tensor pipeline / broadphase | `GPU / GPU / GPU` |
| 资产转换耗时 | `2.894 s` |
| 120 步耗时 | `1.024 s`，`117.142 steps/s` |
| 总耗时 | `6.157 s` |
| 整卡显存采样峰值 | `3949 MiB` |
| 逐对象平移范围 | `0.00517–0.03670 m` |

结果目录为：

```text
output/isaaclab_migration/cam22_replay_phase_b_v1/
```

Kit 日志记录非空 CUDA handle、CUDA ordinal 0 和 PhysX tensor context device 0，且
没有 GPU solver/broadphase 回退。当前使用通用 `convex_hull` 作为 B 阶段最小 replay
近似；该次平移结果不能替代 C 阶段的逐对象旋转、碰撞质量和 60 帧稳定性验证。

旧 Isaac Gym 后端也用同一合成输入实际回归：GPU PhysX 启用，因旧 PyTorch 不支持
`sm_120` 而按既有策略使用 CPU tensor pipeline，120 步正常完成。两个后端的结果
均写入各自的新目录，实际输入目录在测试时间内没有新增或修改文件。

可复现命令：

```bash
bash scripts/run_isaaclab_replay.sh \
  output/cam_22_foreground_run1/stage2/scene_tree.json \
  output/cam_22_foreground_run1/stage3/global_scene \
  output/isaaclab_migration/cam22_replay_phase_b_reproduction_v1 \
  --settle-steps 120
```

### C. 六对象完整场景稳定性验证

目标：用实际场景确认全部对象联合步进并产出逐对象稳定性结果。

步骤：

1. 对加载成功、缺失和跳过的资产做严格集合校验，缺一项即失败，不能静默跳过。
2. 初始状态写入后读取回验，确认对象数量、位姿和碰撞体数量。
3. 在同一环境让六个目标对象连同固定边界共同运行至少 60 帧。
4. 对每个对象计算初末位置差和 SO(3) geodesic rotation 差。
5. 按论文/任务标准判定：60 帧后位移大于 `0.1 m` 或旋转大于 `0.1 rad` 即不稳定。
6. 同时记录早期/末端线速度、角速度、接触或穿插指标，定位翻倒和飞散来源。
7. 输出机器可读 JSON/NPZ 和人类可读日志；任何视频只是辅助证据。

该阶段不通过时先诊断资产尺度、坐标、collision、质量/惯量和关系表达，不直接调
CEM 参数掩盖问题。

#### C 阶段当前执行结果（2026-08-01）

阶段 B 已提交为 `7f70ccb`（`Add Isaac Lab replay backend`）。随后启动 C 阶段；
C 阶段实现最终提交为 `a841725`（`Add full-scene stability validation`）。当前实现增加了
以下通用验证能力：

- `--stability-evaluation-steps` 明确指定相对第 0 帧的评估步，默认 `60`；
- `--position-stability-threshold` 和 `--rotation-stability-threshold` 默认分别为
  `0.1 m`、`0.1 rad`，等于阈值仍稳定，超过任一阈值即不稳定；
- 四元数使用符号不变的 SO(3) geodesic 最短角距离，不用欧拉角差；
- `--expected-object-count` 可对当前实验显式要求六对象，但实现不包含对象名或固定
  对象数量；
- 每个对象回验刚体数量、collision shape 数、质量、惯量和质心；
- 启用 GPU PhysX contact view，逐帧记录总/成对接触力、接触点数、最小 separation
  和最大穿插深度；状态和接触 tensor 都必须位于 CUDA；
- JSON 保存逐对象结果和人类可读原因，`stability_metrics.npz` 保存完整数值数组；
  `--require-stable` 使任一对象不稳定时进程返回失败。

后端无关稳定性函数已有阈值边界、四元数双覆盖、速度窗口和帧数错误测试。合成场景
中的动态体刻意下落 `0.300000 m`：运行时检查全部通过、固定体稳定、动态体按规则
判为不稳定，证明验证器不会把“仿真成功”误当成“场景稳定”。合成结果位于：

```text
output/isaaclab_migration/synthetic_replay_phase_c_v1/
```

实际六对象场景先以 B 阶段的单凸包近似运行。它得到 6/6 稳定，但人物与桌子之间的
最大穿插约 `0.03425 m`，说明单凸包结果过于乐观，不能作为 C 阶段最终结论。随后
使用不依赖对象名称的 `convex_decomposition` 重跑，所有对象各生成 16 个 collision
shape，结果如下：

| 指标 | 结果 |
| --- | ---: |
| 对象数 / 稳定数 | `6 / 5` |
| 状态 tensor | `[61, 6, 13]`，`cuda:0` |
| 成对接触 tensor | `[61, 6, 6]`，`cuda:0` |
| GPU simulation / tensor pipeline / broadphase | `GPU / GPU / GPU` |
| 不稳定对象第 60 步位移 | `0.140452 m` |
| 不稳定对象第 60 步旋转 | `0.437530 rad` |
| 资产转换耗时 | `2.894 s` |
| 60 步仿真耗时 | `0.796 s`，`75.393 steps/s`（详细接触采样开启） |
| 总耗时 | `5.664 s` |
| 整卡显存采样峰值 | `4014 MiB` |

失败对象为场景树中“人物位于椅子上”的子对象。成对接触数据把主要初始冲突定位为
人物—桌子（最大穿插 `0.027419 m`，接触存在 57/61 帧），其次为人物—椅子
（`0.006433 m`）。这只是结果诊断，代码没有使用这些对象名做分支或禁碰撞。

同时，输入 URDF 对所有几何尺度差异很大的对象都使用完全相同的 `1.0 kg` 质量和
对角 `0.1 kg·m²` 惯量。这是现有资产生成的占位物理属性，不能被视为满足通用质量/
惯量策略。当前结论是：C 阶段验证链路已建立并真实报告失败；实际凸分解场景尚未
通过稳定性标准，下一步应从通用质量/惯量生成、初始关系约束和碰撞体质量继续修正，
不能调宽阈值或提前进入 CEM。

正式失败证据保存在全新目录：

```text
output/isaaclab_migration/cam22_stability_phase_c_convex_decomposition_v2/
```

可复现命令（输出目录必须在执行前不存在）：

```bash
bash scripts/run_isaaclab_replay.sh \
  output/cam_22_foreground_run1/stage2/scene_tree.json \
  output/cam_22_foreground_run1/stage3/global_scene \
  output/isaaclab_migration/cam22_stability_phase_c_reproduction_v1 \
  --settle-steps 60 \
  --stability-evaluation-steps 60 \
  --expected-object-count 6 \
  --position-stability-threshold 0.1 \
  --rotation-stability-threshold 0.1 \
  --collision-approximation convex_decomposition \
  --require-stable
```

WSL headless 日志仍会报告 Vulkan/renderer 无设备；该路径没有图形输出。独立的
PhysX context、state/contact CUDA tensor、NVRTC 12.8 和 GPU broadphase 检查均通过，
因此这些 renderer 消息不表示本次 GPU 物理计算回退到 CPU。

### D. 局部组 CEM 移植

目标：复用公共 CEM 和能量函数，用 Isaac Lab 的批量环境与 CUDA state tensor
完成局部组优化。

步骤：

1. 把 `CEMOptimizer` 从 Isaac Gym 模块依赖中分离。
2. 实现批量环境布局、对象索引表、批量 root state set/get 和 reset。
3. 先以较小 population 验证采样、写回、settle、reward 和 elite update，再扩大。
4. 局部组根、直接子对象及必要的可移动承托祖先按通用 scene tree 规则进入仿真。
5. 对稳定性、旋转、速度、placement penetration、settled penetration 和 layout 各项
   做 shape/device/数值一致性测试。
6. 固定随机种子，保存最优候选和每轮摘要，避免只凭最终 mesh 判断。

#### D 阶段当前进展（2026-08-01）

在提交 C 阶段后已经启动 D，但尚未进入 E。当前完成的是局部 CEM 的公共逻辑与
Isaac Lab 合成批量链路，不是实际 Stage 3 局部组迁移完成：

- 将纯 NumPy `CEMOptimizer` 从含 Isaac Gym 顶层依赖的模块中抽离到
  `rest3d/optim/cem.py`；旧 Isaac Gym 调用点继续导入同一实现，不复制业务逻辑；
- 保留 CEM、NES、warm start 和 iCEM elite carry，并补充维度、有限值、标准差等
  输入校验；`cem_seed` 和旧后端的 `--cem-seed` 提供确定性采样；
- 新增后端无关的 WXYZ 四元数 6-DoF 位姿组合及局部能量计算，覆盖 pose/rotation
  stability、pose/rotation layout、早期速度、placement penetration 和 settled
  penetration；
- 合成 smoke 使用 `RigidObjectCollection` 在每个候选环境中放置一个通用固定承托体
  和一个动态子刚体，执行批量 root-state 写入/读回、真实 GPU PhysX 步进、CUDA
  contact tensor、能量计算和 elite update；fixture 名称只描述物理角色，不引用实际
  测试场景对象名；
- 公共 CEM 和局部能量已有 9 个单元测试；仓库测试发现共 27 项通过；旧 `gym`
  Python 3.8 进程在局部 preload 下可导入抽离后的同一 `CEMOptimizer`。

权威 smoke 使用 32 个并行候选、3 轮 CEM、每轮 60 个 PhysX step，结果为：

| 指标 | 结果 |
| --- | ---: |
| 每环境刚体数 / 并行环境数 | `2 / 32` |
| state tensor | `[32, 2, 13]`，`cuda:0` |
| contact tensor | `cuda:0` |
| GPU simulation / tensor pipeline / broadphase | `GPU / GPU / GPU` |
| 总 PhysX steps | `180` |
| 仿真耗时 / 吞吐 | `1.067 s / 168.684 steps/s` |
| 整卡显存采样峰值 | `4013 MiB` |
| 固定组根最大位移 | `0.0 m` |
| 每轮最佳 reward | `-0.250543, -0.150859, -0.072671` |

结果和机器可读指标位于：

```text
output/isaaclab_migration/local_cem_phase_d_smoke_v4/
```

可复现命令（输出目录必须在执行前不存在）：

```bash
bash scripts/run_isaaclab_local_cem_smoke.sh \
  output/isaaclab_migration/local_cem_phase_d_reproduction_v1 \
  --num-envs 32 \
  --cem-iters 3 \
  --settle-steps 60 \
  --early-steps 15 \
  --seed 17
```

当前 D 阶段仍缺少：按 scene tree 选择真实局部组、把实际 URDF/mesh/collision 资产
批量装入候选环境、使用真实几何计算 placement penetration、保存实际局部组最优结果
并与旧后端做同输入对比。C 阶段实际凸分解场景的 `5/6` 稳定失败也仍然存在；本次
合成 smoke 不代表该问题已经修复，不能据此进入 E 阶段。

### E. 符合论文的全对象全局 CEM

目标：全局 CEM 只采样高层 entity/组根的 6-DoF，但每个候选环境都实例化并模拟
全部组内子对象。

每个候选的必需流程：

1. 为组根或独立对象采样 6-DoF delta。
2. 用 `T_child_world = T_root_world @ T_child_local` 递归组合全部后代的初始位姿。
3. 将组根、全部后代、其余独立对象和固定边界一同写入该候选环境。
4. 不把物理 group 当成一个合并刚体；除非场景语义明确要求刚性附着，各子对象都
   保持独立刚体和真实碰撞。
5. 运行 PhysX 后读取全部对象的状态和速度。
6. `E_stab`、`E_vel`、`E_pen` 和 `E_layout` 对全部参与对象求值；报告中同时给出
   group/entity 汇总和逐对象明细。
7. CEM distribution 仍只更新高层 entity/组根的变量，子对象不是额外采样维度。

必须加入结构性断言：

```text
simulated_object_set == expected_scene_object_set
evaluated_object_set == expected_scene_object_set
sampled_entity_set <= simulated_object_set
```

并建立一个与对象名称无关的合成层级测试：一个根、两个子对象、一个独立对象；
移动根后验证子对象初始位姿正确组合，并验证任意子对象受碰撞扰动都会改变候选能量。

### F. Isaac Gym / Isaac Lab 结果与性能对比

目标：确保旧后端可用，并量化新后端的正确性和收益。

统一记录：

| 字段 | 内容 |
| --- | --- |
| software | backend、Isaac Sim/Lab/Gym、Python、PyTorch、CUDA、driver |
| hardware | GPU 名称、compute capability、总显存 |
| workload | 对象数、固定对象数、并行环境数、步数、CEM population/iterations |
| device | PhysX device、pipeline device、state tensor device |
| performance | 启动、资产加载、每轮/总耗时、steps/s、峰值显存 |
| quality | 每对象位移/旋转/速度、穿插、稳定对象数、全场景稳定结论 |

对比使用相同输入、physics dt、步数、collision 策略和随机种子。Isaac Gym 在 RTX
5090 上预期仍是 GPU PhysX + CPU tensor pipeline；Isaac Lab 必须是 GPU PhysX +
GPU tensor。数值不要求逐 bit 相同，但对象集合、阈值定义和能量项必须一致。

### G. 文档与回归测试

目标：提供从环境创建到实际场景复现的完整说明，并防止旧后端退化。

交付项：

- 独立 Isaac Lab 安装文档和精确版本锁定；
- smoke、replay、局部 CEM、全局 CEM、最终验证的可复现命令；
- WSL/headless 的实际支持结论和已知限制，不把未测试组合写成支持；
- 两个 backend 的 CLI 文档、输出目录规则和日志字段；
- 纯 Python 的 scene tree、位姿组合、坐标转换、稳定阈值和集合断言测试；
- 可用时的小规模 headless integration test；
- Isaac Gym 回归命令；
- 实际场景结果和性能对比报告。

## 7. 分阶段门禁与验证矩阵

| 阶段 | 最小验证 | 进入下一阶段的条件 |
| --- | --- | --- |
| A | compatibility checker、真实 PhysX step、真实 Torch CUDA op | GPU PhysX + CUDA state tensor 全部成立 |
| B | 合成 replay、实际资产加载、state round-trip | Isaac Lab 可完整加载并 replay 全对象 |
| C | 实际六对象 60 帧逐对象报告 | 对象集合完整，稳定性判据可复现 |
| D | 小 population 局部 CEM | reward/device/批量 state 正确，无 CPU 意外回退 |
| E | 层级合成测试 + 实际全局 CEM | 全部子对象进入仿真和全部能量项 |
| F | 两后端同输入对比 | 旧 Gym 可用，新 Lab GPU 链路成立 |
| G | 文档命令与回归测试 | 新图片无需手写对象级物理配置 |

每次 Python 修改至少运行 `python -m py_compile`；shell 修改至少运行 `bash -n`。
纯逻辑使用单元测试，CUDA/PhysX 改动必须运行真实 GPU 操作。高风险集成步骤先用小
环境数验证，再逐步扩大，避免一次性分配 2048 个复杂场景导致显存耗尽。

## 8. 输出目录与证据格式

建议所有迁移实验统一放在新的命名空间：

```text
output/isaaclab_migration/
  smoke_20260801_v1/
  cam22_replay_v1/
  cam22_joint_scene_v1/
  cam22_local_cem_v1/
  cam22_global_cem_v1/
  comparison_v1/
```

这些只是计划中的显式示例；正式运行前逐一确认目录不存在。每次结果至少包含：

```text
command.txt
environment.json
run.log
metrics.json
states_initial.npy
states_final.npy
```

`environment.json` 记录 Git commit、dirty status、版本、设备和 WSL 信息；
`metrics.json` 记录对象集合、逐对象指标、全局稳定结论、耗时和显存。日志不得包含
凭据、token 或无关环境变量。

## 9. 停止条件

出现以下任一情况时不自行扩大操作范围：

- 需要更新/降级 Windows 或 Linux NVIDIA 驱动；
- 需要安装系统级 Vulkan、X server、Docker daemon 或 NVIDIA Container Toolkit；
- 需要删除已有 Conda 环境、Isaac/Omniverse cache 或 output 数据；
- WSL 无法提供 Isaac Sim 所需的 Vulkan/CUDA/图形能力；
- 只有官方不支持的 WSL 容器路径可能继续；
- 需要修改主工作树、合并分支或改写 Git 历史；
- 测试输出目录已存在且无法保证不覆盖。

届时应提供失败命令、完整日志位置、驱动/GPU/版本信息、已尝试的官方方案和下一步
所需权限，由用户决定是否继续。

## 10. 完成定义

- `/home/yangyankun/REST3D` 主工作树无变化；
- Isaac Gym backend 在原 `gym` 环境继续可用；
- Isaac Lab backend 在独立环境的 RTX 5090 上实际运行 GPU PhysX、GPU state
  tensor 和真实 PyTorch CUDA 运算；
- replay、局部 CEM、全局 CEM 和最终验证共享场景输入与评估逻辑；
- 全局 CEM 只采样组根，但全部后代进入每个候选的 PhysX 与四类能量项；
- 碰撞、质量和惯量采用通用几何/语义策略，不含当前场景对象名硬编码；
- 新图片仍只需指定提取对象，不要求手写逐对象物理配置；
- 实际场景在新目录得到逐对象 60 帧稳定性报告；
- 提供可复现命令、版本、设备、耗时、并行环境数、峰值显存和两后端对比；
- 不覆盖任何既有输出。

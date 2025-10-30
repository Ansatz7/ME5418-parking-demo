# Parking Environment Demo / 泊车环境演示包

**English** | [中文](#中文指南)

Note: Compared to earlier submissions, this repository has been refactored to make running, testing, and evaluation simpler (clear Quick Start, one‑command evaluation, and split NN/Gym checks).

## English Guide

[This repository](https://github.com/Ansatz7/ME5418-parking-demo) packages the self-contained parking environment used for coursework demos. Below is a quick navigation list; feel free to jump straight to the section you need.

### Table of Contents

1. [Quick Start](#quick-start)
2. [Gym Demo Module](#gym-demo-module)
3. [Neural Network Module](#neural-network-module)
4. [Agent Learning Module](#agent-learning-module)

### Quick Start

The default workflow uses `pip install -e .` so users can get the demo running in minutes. An optional `environment.yml` is provided for those who prefer Conda/Mamba.

#### Environment Setup

**Method A (recommended): pip install -e .**

```bash
# ensure mamba is available (replace mamba with conda if preferred)
conda install -n base -c conda-forge mamba

# create a clean Python 3.10 environment
mamba create -n parking-rl python=3.10 -y
mamba activate parking-rl

# upgrade tooling and install the package in editable mode
python -m pip install --upgrade pip
pip install -e .

# smoke tests
parking-gym-demo --mode random --episodes 1 --max-steps 400
parking-gym-demo --mode manual --episodes 1 --max-steps 400

# leave the environment when finished
mamba deactivate
```

**Method B: environment.yml + Mamba**

```bash
mamba env create -f environment.yml -n parking-rl-yml
mamba activate parking-rl-yml
parking-gym-demo --mode random --episodes 1 --max-steps 400
mamba deactivate
```

*All commands also work if you substitute `mamba` with `conda`. The new dependency list includes `PyQt5`, so the GUI tuner works out of the box.*

#### Neural Network Quick Check (this week’s focus)

- For full details (architecture, export and visualization), see:
  [Neural Network Module](#neural-network-module)

- Single step (T=1):

  ```bash
  python -m parking_project_submission.neural_network_demo
  ```

- Longer sequence (e.g., T=8):

  ```bash
  python -m parking_project_submission.neural_network_demo --seq-len 8
  ```

- Optional ONNX export for Netron (install onnx first: `pip install onnx`):

  ```bash
  python -m parking_project_submission.neural_network_demo --export-onnx artifacts/model_lstm.onnx
  ```

- One-shot NN smoke tests script:

  ```bash
  bash scripts/quick_test_nn.sh
  ```

- Optional tools for visualization/summaries (install on demand):

  ```bash
  python -m pip install onnx onnxsim netron torchinfo
  ```

#### Gym Quick Check

- Headless random rollout (per-step logs with `--per-step`):

  ```bash
  parking-gym-demo --mode random --episodes 1 --max-steps 400 --no-visualize --per-step
  ```

- Manual mode (keyboard). Use `--summary` for a one-line episode summary (no per-step logs):

  ```bash
  parking-gym-demo --mode manual --episodes 1 --max-steps 400 --summary
  ```

- Generate randomized config and visualize:

  ```bash
  python -m parking_project_submission.parking_env.generate_training_config \
    --out parking_project_submission/configs/train_quick.json --seed 123
  parking-gym-demo --mode random --episodes 1 --max-steps 200 \
    --config parking_project_submission/configs/train_quick.json
  ```

- One-shot Gym smoke tests script:

  ```bash
  bash scripts/quick_test_gym.sh
  ```

#### One-Command Evaluation

For a quick evaluation, just run the single entry file without any arguments:

```bash
bash run_submission.sh
```

This performs:
- A short, visualized random environment rollout with per-step logging; and
- A short neural-network demo (Lidar+Residual+LSTM, sequence T=4).

#### Helper Scripts

- Create or update the environment and install the package in editable mode (auto-detects mamba/conda):

  ```bash
  bash scripts/setup_env.sh
  ```

- Run non-interactive smoke tests (both NN and Gym):

  ```bash
  bash scripts/quick_test.sh
  ```

  This runs Gym quick checks, two neural-network demos (T=1 and T=8), and
  optionally exports an ONNX model if the `onnx` package is installed.

- Clean build metadata and caches (optional):

  ```bash
  bash scripts/clean.sh
  ```

#### Repository Layout

```
parking_project_submission/
├── configs/                    # default + sample configs (bilingual notes)
├── gym_demo.py                 # CLI entry (manual/random modes)
├── modules/                    # JSON helpers, workflows, networks
├── parking_env/                # Gymnasium env, GUI tuner, config generator
├── agent_learning.py           # placeholder for future PPO pipeline
└── neural_network_demo.py      # Lidar+Residual+LSTM demo (forward/sample/onnx)

requirements.txt
setup.py
environment.yml
```

### Gym Demo Module

This section covers everything related to the parking environment and its supporting utilities.

#### Environment Overview

- **Implementation:** `parking_project_submission/parking_env/env.py`
- **Observation space:** 11 core vehicle features (pose, velocity, steering) plus lidar ranges (`ray_angles`) for a total of `obs_dim` entries. Gaussian noise can be toggled at runtime.
- **Action space:** 2D continuous vector `[longitudinal_accel, steering_accel]`, clipped to the vehicle limits.
- **Reward shaping:** Distance, heading, velocity, smoothness, step cost, collision penalty, and success bonus. Termination occurs on success, collision, boundary exit, or timeout.
- **Rendering:** Matplotlib overlays vehicle pose, ray distances, last action, and per-term reward breakdown.

Run the CLI demo via:

```bash
parking-gym-demo --mode random --episodes 1 --max-steps 400
parking-gym-demo --mode manual --episodes 1 --max-steps 400
```

Use `--config path/to/config.json`, `--no-visualize`, and `--quiet` as needed.

#### Assist Model Tuner GUI

- **Script:** `parking_project_submission/parking_env/assist_model_tuner.py`
- **Purpose:** Tune steering and velocity assist gains with live plots, then optionally write changes back to JSON.
- **Usage:**

```bash
python -m parking_project_submission.parking_env.assist_model_tuner \
  --config parking_project_submission/configs/demo_default.json --sync
```

Check the “sync to JSON” box (or pass `--sync`) to persist slider updates back to the config file. This requires a Qt-capable environment; `PyQt5` ships with the default requirements.

#### Random Scenario Generator

- **Script:** `parking_project_submission/parking_env/generate_training_config.py`
- **Description:** Produces map variations while keeping dynamics consistent with `DEFAULT_CONFIG`.
- **Sample command:**

```bash
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_random.json --seed 123
```

Outputs include randomized spawn regions, slot placement, and obstacle layouts.

### Neural Network Module

- **Location:** `parking_project_submission/modules/networks.py`, `parking_project_submission/neural_network_demo.py`
- **Backbone:** Lidar+Residual+LSTM
  - Inputs: split observation into `[B, T, 11]` base features (MLP→128) and `[B, T, N]` lidar distances (1D-CNN+GAP→FC→128)
  - Fusion: concat → residual refiner (Linear→LN→ReLU→Linear + skip)
  - Temporal core: LSTM(hidden=128) → policy/value heads
- **Outputs:** Gaussian policy N(μ, σ) with clamped log_std and value V(s).

CLI usage (no training; forward, sample, toy loss, backward):

```bash
# default: T=1, CPU
python -m parking_project_submission.neural_network_demo

# custom sequence length
python -m parking_project_submission.neural_network_demo --seq-len 8

# optional: export ONNX for Netron (install onnx first: pip install onnx)
python -m parking_project_submission.neural_network_demo --export-onnx artifacts/model_lstm.onnx
```

Optional helpers: `pip install onnx` for export, `pip install netron` to view `.onnx`, `pip install torchinfo` for richer summaries.

Note: This submission runs entirely on CPU to maximize portability across machines. GPU instructions and flags are intentionally omitted; a GPU-enabled variant will be provided later.

- **Next steps:** Integrate PPO training loop (clip loss, value loss, entropy), GAE advantages, optimizer schedule and masks for variable-length sequences.

#### Model Visualization (ONNX + Netron)

- Export ONNX from the demo (already supported by `--export-onnx`):

  ```bash
  python -m parking_project_submission.neural_network_demo \
    --seq-len 4 --export-onnx artifacts/model_lstm.onnx
  ```

- Simplify the graph for cleaner visualization (optional):

  ```bash
  python -m pip install onnx onnxsim
  python -m onnxsim artifacts/model_lstm.onnx artifacts/model_lstm_simple.onnx
  ```

- Visualize in Netron:

  - Website: https://netron.app (drag-and-drop `artifacts/model_lstm_simple.onnx`)
  - Local app (optional): `pip install netron && netron artifacts/model_lstm_simple.onnx`

Tips: using a fixed small sequence length (e.g., `--seq-len 1`) reduces dynamic-shape nodes; running `onnxsim` further folds shape ops for a shorter graph.

### Agent Learning Module

- **Location:** `parking_project_submission/agent_learning.py`
- **Current status:** CLI stub outlining the planned PPO workflow (argument parsing, placeholders).
- **Roadmap:** Next iteration will replace the stub with the actual training routine, tests, and logging hooks. Documentation will be updated alongside that release.

---

## 中文指南

[English](#english-guide) | **中文**

说明：相较此前提交版本，本仓库已做结构性重构，重点优化了运行与评估流程（更清晰的 Quick Start、单文件一键评测、NN/Gym 分离验证等）。

[本仓库](https://github.com/Ansatz7/ME5418-parking-demo)为智能泊车演示项目的独立提交包，下面提供快速导航，方便直接跳到所需章节。

### 目录

1. [快速上手](#快速上手)
2. [Gym 模块](#gym-模块)
3. [神经网络模块](#神经网络模块)
4. [Agent Learning 模块](#agent-learning-模块)

### 快速上手

推荐采用 `pip install -e .`，保证使用者或读者几分钟内即可运行。若偏好 Conda/Mamba，也提供 `environment.yml`。

#### 环境搭建

**方法 A（推荐）：pip install -e .**

```bash
# 若已安装 mamba 可跳过此行；如习惯 conda，可将 mamba 替换为 conda
conda install -n base -c conda-forge mamba

# 创建 Python 3.10 环境
mamba create -n parking-rl python=3.10 -y
mamba activate parking-rl

# 升级 pip 并以开发模式安装本项目
python -m pip install --upgrade pip
pip install -e .

# 快速验证
parking-gym-demo --mode random --episodes 1 --max-steps 400
parking-gym-demo --mode manual --episodes 1 --max-steps 400

# 完成后退出环境
mamba deactivate
```

**方法 B：environment.yml + Mamba**

```bash
mamba env create -f environment.yml -n parking-rl-yml
mamba activate parking-rl-yml
parking-gym-demo --mode random --episodes 1 --max-steps 400
mamba deactivate
```

*依赖列表已包含 `PyQt5`，助力调参 GUI 默认可用；若只使用命令行，也可自行改为精简安装。*

#### 神经网络快速验证（本周重点）

- 需要更完整的设计、导出与可视化说明，请查看：
  [神经网络模块](#神经网络模块)

- 单步（T=1）：

  ```bash
  python -m parking_project_submission.neural_network_demo
  ```

- 多步序列（例如 T=8）：

  ```bash
  python -m parking_project_submission.neural_network_demo --seq-len 8
  ```

- 可选导出 ONNX 供 Netron 可视化（先安装 onnx：`pip install onnx`）：

  ```bash
  python -m parking_project_submission.neural_network_demo --export-onnx artifacts/model_lstm.onnx
  ```

- 一键 NN 自检脚本：

  ```bash
  bash scripts/quick_test_nn.sh
  ```

- 可选工具（按需安装，用于可视化/结构摘要）：

  ```bash
  python -m pip install onnx onnxsim netron torchinfo
  ```

#### Gym 快速验证

- 无界面随机回放（需要逐步日志可加 `--per-step`）：

  ```bash
  parking-gym-demo --mode random --episodes 1 --max-steps 400 --no-visualize --per-step
  ```

- 手动模式（键盘）。若只需回合摘要（不打印逐步日志），可加 `--summary`：

  ```bash
  parking-gym-demo --mode manual --episodes 1 --max-steps 400 --summary
  ```

- 随机可视化：

  ```bash
  python -m parking_project_submission.parking_env.generate_training_config \
    --out parking_project_submission/configs/train_quick.json --seed 123
  parking-gym-demo --mode random --episodes 1 --max-steps 200 \
    --config parking_project_submission/configs/train_quick.json
  ```

- 一键 Gym 自检脚本：

  ```bash
  bash scripts/quick_test_gym.sh
  ```

#### 一键评测（无参数）

评测时，只需运行一个入口脚本，无需任何参数：

```bash
bash run_submission.sh
```

此脚本会执行：
- 一段短程、带可视化的随机环境回放（逐步打印）；
- 一次简短的神经网络演示（Lidar+Residual+LSTM，序列长度 T=4）。

#### 辅助脚本

- 一键创建/更新环境并安装（自动检测 mamba/conda）：

  ```bash
  bash scripts/setup_env.sh
  ```

- 快速自检（无 UI 的随机回放 + 随机场景再跑一遍）：

  ```bash
  bash scripts/quick_test.sh
  ```

- 清理构建产物与缓存（可选）：

  ```bash
  bash scripts/clean.sh
  ```

#### 仓库结构

```
parking_project_submission/
├── configs/                    # 默认配置与示例场景（含双语注释）
├── gym_demo.py                 # 命令行入口（随机 / 手动模式）
├── modules/                    # JSON 工具、工作流、网络结构
├── parking_env/                # 环境实现、调参 GUI、配置生成器
├── agent_learning.py           # PPO 训练入口预留脚本
└── neural_network_demo.py      # Lidar+Residual+LSTM 示例

requirements.txt
setup.py
environment.yml
```

### Gym 模块

此部分涵盖泊车环境本体及其辅助工具。

#### 环境概览

- **实现路径：** `parking_project_submission/parking_env/env.py`
- **观测空间：** 车辆姿态、速度、转向等 11 个核心特征，加上若干激光雷达距离，默认叠加可配置高斯噪声。
- **动作空间：** 连续动作 `[纵向加速度, 转向加速度]`，根据车辆极限自动裁剪。
- **奖励设计：** 距离、朝向、速度、平滑度、时间步成本、碰撞惩罚、成功奖励；成功 / 碰撞 / 越界 / 超时触发终止。
- **渲染信息：** Matplotlib 视图显示车辆姿态、雷达读数、上一动作、奖励拆分等。

命令行示例：

```bash
parking-gym-demo --mode random --episodes 1 --max-steps 400
parking-gym-demo --mode manual --episodes 1 --max-steps 400
```

支持 `--config path/to/config.json`、`--no-visualize`、`--quiet` 等参数。

#### 助力模型调参 GUI

- **脚本：** `parking_project_submission/parking_env/assist_model_tuner.py`
- **作用：** 调整方向盘回正与纵向阻尼参数，实时可视化曲线，可同步回 JSON。
- **使用方式：**

```bash
python -m parking_project_submission.parking_env.assist_model_tuner \
  --config parking_project_submission/configs/demo_default.json --sync
```

勾选界面右下角的 “sync to JSON”（或命令行直接加 `--sync`）即可把滑块的数值写回配置文件。Linux 无显示环境时可设置 `QT_QPA_PLATFORM=offscreen`。

#### 随机场景生成器

- **脚本：** `parking_project_submission/parking_env/generate_training_config.py`
- **说明：** 在保持动力学参数不变的前提下，随机化出生区域、车位位置、障碍物等场景元素。
- **示例命令：**

```bash
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_random.json --seed 123
```

生成的 JSON 可直接供训练或评估脚本加载。

### 神经网络模块

- **关键文件：** `parking_project_submission/modules/networks.py`、`parking_project_submission/neural_network_demo.py`
- **时序骨干：** Lidar+Residual+LSTM
  - 输入：将观测拆成 `[B, T, 11]` 基础特征（MLP→128）与 `[B, T, N]` LiDAR 距离（1D-CNN+自适应池化→FC→128）
  - 融合：concat → 残差精炼块（Linear→LN→ReLU→Linear 与残差相加）
  - 时序层：LSTM(hidden=128) → 策略/价值头
- **输出：** 高斯策略 N(μ, σ)（log_std 有界）与状态值 V(s)。

命令行用法（不训练，仅前向、采样、玩具损失、反向）：

```bash
# 默认：T=1，CPU
python -m parking_project_submission.neural_network_demo

# 指定序列长度
python -m parking_project_submission.neural_network_demo --seq-len 8

# 可选：导出 ONNX（先安装 onnx：pip install onnx）
python -m parking_project_submission.neural_network_demo --export-onnx artifacts/model_lstm.onnx
```

可选工具：`pip install onnx` 导出、`pip install netron` 浏览 `.onnx`、`pip install torchinfo` 打印更详细结构摘要。

说明：当前提交版仅使用 CPU 运行，最大化跨机器的可复现性；GPU 参数与说明已暂时移除，后续会提供 GPU 适配版本。

- **后续计划：** 集成 PPO（clip/value/entropy）、GAE 优势、优化器与变长序列 mask 等，完成训练部分后更新本节说明。

#### 模型可视化（ONNX + Netron）

- 从演示脚本导出 ONNX（已内置开关）：

  ```bash
  python -m parking_project_submission.neural_network_demo \
    --seq-len 4 --export-onnx artifacts/model_lstm.onnx
  ```

- 简化计算图（可选，使图更短更清晰）：

  ```bash
  python -m pip install onnx onnxsim
  python -m onnxsim artifacts/model_lstm.onnx artifacts/model_lstm_simple.onnx
  ```

- 在 Netron 中打开：

  - 网页版：https://netron.app （拖入 `artifacts/model_lstm_simple.onnx`）
  - 本地应用（可选）：`pip install netron && netron artifacts/model_lstm_simple.onnx`

提示：导出时使用较小的序列（如 `--seq-len 1`）可减少动态形状节点；`onnxsim` 会折叠形状相关算子，得到更紧凑的示意图。

### Agent Learning 模块

- **文件：** `parking_project_submission/agent_learning.py`
- **当前状态：** 训练入口的 CLI 占位脚本，仅校验参数并输出计划流程。
- **下一步：** 替换为真实的 PPO 训练循环，加入日志与测试后将完善本章节说明。

祝调参顺利，泊车顺滑！ 🚗

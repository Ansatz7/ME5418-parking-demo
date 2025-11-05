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

Before you begin
- Open a terminal and cd into the project root (this repository folder; typically `ME5418-parking-demo` if cloned from GitHub):
  - Linux (Ubuntu 22.04 example): `cd /path/to/ME5418-parking-demo`
- Commands below are shown for Bash on Linux.
- Prefer Mamba for speed; if you like Conda, skip installing Mamba and replace every `mamba` command with `conda`.

**Method A (recommended): pip install -e .**

```bash
# Choose one package manager
# - If using Mamba (recommended): ensure installed and initialized, then use `mamba ...` below
# - If you prefer Conda: skip installing Mamba and replace `mamba` with `conda` in all commands

# (Mamba-only) Ensure mamba is installed and shell-initialized
conda install -n base -c conda-forge mamba -y
eval "$(mamba shell hook --shell bash)"

# Create a clean Python 3.10 environment and activate it
mamba create -n parking-rl python=3.10 -y
mamba activate parking-rl

# Install the package in editable mode
pip install -e .

# Run smoke tests (keep env active afterward for more tests)
echo "Running random demo..."
parking-gym-demo --mode random --episodes 1 --max-steps 10 --no-visualize
echo "Running manual demo (close window to finish)..."
parking-gym-demo --mode manual --episodes 1 --max-steps 400

# Leave the environment active for further experiments.
# When completely done, deactivate with:  mamba deactivate   (or: conda deactivate)
```

**Method B: environment.yml (Mamba or Conda)**

```bash
# Choose one package manager
# - Mamba (recommended): ensure installed/initialized
# - Conda: replace `mamba` with `conda` in the commands below

# (Mamba-only) Ensure mamba is installed and shell-initialized
conda install -n base -c conda-forge mamba -y
eval "$(mamba shell hook --shell bash)"

# Create environment from yml file
mamba env create -f environment.yml
mamba activate parking-rl

# Test installation (keep env active for more tests)
parking-gym-demo --mode random --episodes 1 --max-steps 10 --no-visualize

# Deactivate later when completely done:
# mamba deactivate   (or: conda deactivate)
```

<!-- removed duplicate Method B block to avoid confusion -->

*All commands also work if you substitute `mamba` with `conda`. The dependency list includes `PyQt5`, so the GUI tuner works out of the box.*

#### Learning Agent Quick Start (this week’s focus)

- Minimal PPO training (stateless, T=1; CPU-only):

  ```bash
  python -m parking_project_submission.agent_learning --total-steps 200000 --rollout-len 2048
  ```

- Recurrent PPO training (uses LSTM memory + BPTT):

  ```bash
  python -m parking_project_submission.ppo_recurrent --total-steps 200000 \
    --rollout-len 2048 --chunk-len 256 --epochs 4
  ```

- Evaluate a saved checkpoint (uses policy mean, deterministic):

  ```bash
  python -m parking_project_submission.agent_learning --eval --checkpoint artifacts/ppo_minimal.pt --eval-episodes 5
  ```

- Visualize the trained policy (policy mode in the Gym demo):

  ```bash
  parking-gym-demo --mode policy --episodes 1 --max-steps 400 \
    --checkpoint artifacts/ppo_minimal.pt
  
  # extras
  parking-gym-demo --mode policy --stochastic --checkpoint artifacts/ppo_minimal.pt
  parking-gym-demo --mode policy --checkpoint artifacts/ppo_minimal.pt --record artifacts/rollout.mp4
  ```

See details: [Agent Learning Module](#agent-learning-module)

#### Neural Network Quick Check

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

- Note: to run a trained policy, see Learning Agent Quick Start above.

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
├── configs/                    # Default + sample configs (bilingual notes)
├── gym_demo.py                 # CLI entry (random/manual/policy, --summary/--per-step/--record)
├── modules/
│   ├── __init__.py
│   ├── utils.py               # JSON helpers
│   ├── workflow.py            # Demo runners + keyboard controller
│   └── networks.py            # Lidar+Residual+LSTM backbone (this week’s focus)
├── parking_env/
│   ├── __init__.py
│   ├── env.py                 # Environment (observations, rewards, rendering)
│   ├── assist_model_tuner.py  # Qt/Matplotlib assist tuner (optional)
│   └── generate_training_config.py # Randomized scene generator
├── agent_learning.py          # Minimal PPO trainer (stateless T=1)
├── ppo_recurrent.py           # Recurrent PPO trainer (LSTM + BPTT)
└── neural_network_demo.py     # NN demo (forward/sample/backward/ONNX)

scripts/
├── quick_test.sh              # One-shot: NN + Gym checks (non-interactive)
├── quick_test_nn.sh           # NN-only quick checks (T=1/T=8, optional ONNX)
├── quick_test_gym.sh          # Gym-only quick checks
├── setup_env.sh               # Create env + pip install -e .
└── clean.sh                   # Clean caches + artifacts/*.onnx

requirements.txt               # Core Python deps (CPU-only)
setup.py                       # Packaging (editable install)
environment.yml                # Optional conda/mamba wrapper
```

Key files (focus this week):
- `parking_project_submission/modules/networks.py`: Lidar+Residual+LSTM actor–critic.
  - Encoders: base MLP (11→128, LayerNorm+ReLU) and lidar 1D-CNN (+GAP→FC→128).
  - Fusion: `fusion_refiner` residual block (Linear→LN→ReLU→Linear + skip).
  - Temporal: LSTM(256→128); heads for Gaussian policy (clamped log_std) and value.
  - Helpers: `initial_state`, `split_flat_obs`; CPU-only.
- `parking_project_submission/neural_network_demo.py`: Forward/sample/backward demo; flags `--seq-len`, `--export-onnx`.
- `parking_project_submission/gym_demo.py`: Random/manual; `--summary` (manual one-line), `--per-step` (random per-step logs).
- `parking_project_submission/parking_env/env.py`: Environment core (rays, dynamics, rewards, rendering overlays).
- `parking_project_submission/ppo_recurrent.py`: Recurrent PPO trainer (LSTM + BPTT).

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

- **Location:** `parking_project_submission/agent_learning.py` (minimal PPO) and `parking_project_submission/ppo_recurrent.py` (recurrent PPO)
- **Current status:** Minimal PPO (stateless T=1) and Recurrent PPO (uses LSTM with BPTT) are available.

Usage (training and evaluation)
- Train on CPU (stateless PPO):
  - `python -m parking_project_submission.agent_learning --total-steps 200000 --rollout-len 2048`
- Train on CPU (recurrent PPO with memory + BPTT):
  - `python -m parking_project_submission.ppo_recurrent --total-steps 200000 --rollout-len 2048 --chunk-len 256 --epochs 4`
- Evaluate a checkpoint deterministically (policy mean):
  - `python -m parking_project_submission.agent_learning --eval --checkpoint artifacts/ppo_minimal.pt --eval-episodes 5`
- Visualize the trained policy in the Gym demo (renders a driving episode):
  - Deterministic: `parking-gym-demo --mode policy --checkpoint artifacts/ppo_minimal.pt`
  - Stochastic: `parking-gym-demo --mode policy --stochastic --checkpoint artifacts/ppo_minimal.pt`
  - Record to mp4 (requires `pip install imageio imageio-ffmpeg`): `parking-gym-demo --mode policy --checkpoint artifacts/ppo_minimal.pt --record artifacts/rollout.mp4`

Notes
- This minimal trainer uses the environment’s default config to keep things simple. A future update will wire `--config` for training to match the Gym demo configuration pipeline.
- Policy mode currently uses mean action by default; add `--stochastic` for sampling.
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

开始之前
- 打开终端并切换到项目根目录（本仓库文件夹；若从 GitHub 克隆，默认目录名为 `ME5418-parking-demo`）：
  - Linux（Ubuntu 22.04 示例）：`cd /path/to/ME5418-parking-demo`
- 下方命令以 Linux 的 Bash 为例。
- 推荐使用 Mamba；若更习惯 Conda，可跳过安装 Mamba，并把所有 `mamba` 命令替换为 `conda`。

**方法 A (推荐): pip install -e .**

```bash
# 选择一个包管理器
# - 若使用 Mamba（推荐）：确保已安装并初始化；下文使用 `mamba ...`
# - 若偏好 Conda：可跳过安装 Mamba，把命令中的 `mamba` 全部改为 `conda`

# （仅 Mamba）确保 mamba 已安装并完成 shell 初始化
conda install -n base -c conda-forge mamba -y
eval "$(mamba shell hook --shell bash)"

# 创建并激活一个干净的 Python 3.10 环境
mamba create -n parking-rl python=3.10 -y
mamba activate parking-rl

# 以可编辑模式安装本包
pip install -e .

# 运行冒烟测试（后续还要继续测试，环境不要退出）
echo "正在运行随机策略演示..."
parking-gym-demo --mode random --episodes 1 --max-steps 10 --no-visualize
echo "正在运行手动模式演示 (关闭窗口后结束)..."
parking-gym-demo --mode manual --episodes 1 --max-steps 400

# 冒烟后请保持环境激活，便于继续验证。
# 若完全结束所有工作再退出：mamba deactivate（或：conda deactivate）
```

**方法 B: environment.yml（Mamba 或 Conda）**

```bash
# 选择一个包管理器
# - Mamba（推荐）：确保已安装并完成初始化
# - Conda：将下方命令中的 `mamba` 替换为 `conda`

# （仅 Mamba）确保 mamba 已安装并完成 shell 初始化
conda install -n base -c conda-forge mamba -y
eval "$(mamba shell hook --shell bash)"

# 从 yml 文件创建环境
mamba env create -f environment.yml
mamba activate parking-rl

# 测试安装（保持环境激活，方便继续调试）
parking-gym-demo --mode random --episodes 1 --max-steps 10 --no-visualize

# 完全结束后再退出：
# mamba deactivate（或：conda deactivate）
```

<!-- 删除重复的 方法 B 小节以避免歧义 -->

*依赖列表已包含 `PyQt5`，助力调参 GUI 默认可用；若只使用命令行，也可自行改为精简安装。*

#### Learning Agent 快速上手（本周重点）

- 最小 PPO 训练（忽略时序记忆，T=1；仅 CPU）：

  ```bash
  python -m parking_project_submission.agent_learning --total-steps 200000 --rollout-len 2048
  ```

- 时序版 PPO 训练（使用 LSTM 记忆 + BPTT）：

  ```bash
  python -m parking_project_submission.ppo_recurrent --total-steps 200000 \
    --rollout-len 2048 --chunk-len 256 --epochs 4
  ```

- 评测已保存的模型（用策略均值，确定性）：

  ```bash
  python -m parking_project_submission.agent_learning --eval --checkpoint artifacts/ppo_minimal.pt --eval-episodes 5
  ```

- 在 Gym 演示器中可视化策略（policy 模式）：

  ```bash
  parking-gym-demo --mode policy --episodes 1 --max-steps 400 \
    --checkpoint artifacts/ppo_minimal.pt

  # 进阶：随机采样与视频录制（需安装 imageio 与 imageio-ffmpeg）
  parking-gym-demo --mode policy --stochastic --checkpoint artifacts/ppo_minimal.pt
  parking-gym-demo --mode policy --checkpoint artifacts/ppo_minimal.pt --record artifacts/rollout.mp4
  ```

详见： [Agent Learning 模块](#agent-learning-模块)

#### 神经网络快速验证

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

- 注：如需运行已训练策略，请参考上面的 “Learning Agent 快速上手”。

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
├── configs/                    # 默认/示例配置（含双语注释）
├── gym_demo.py                 # 命令行入口（随机/手动/策略，支持 --summary/--per-step/--record）
├── modules/
│   ├── __init__.py
│   ├── utils.py               # JSON 读写工具
│   ├── workflow.py            # 演示运行器 + 键盘控制器
│   └── networks.py            # Lidar+Residual+LSTM 主干（本周重点）
├── parking_env/
│   ├── __init__.py
│   ├── env.py                 # 环境核心（观测/奖励/渲染）
│   ├── assist_model_tuner.py  # 助力调参 GUI（可选）
│   └── generate_training_config.py # 随机场景生成器
├── agent_learning.py          # 最小 PPO 训练器（T=1，忽略记忆）
├── ppo_recurrent.py           # 时序 PPO 训练器（LSTM + BPTT）
└── neural_network_demo.py     # 神经网络示例（前向/采样/反传/ONNX）

scripts/
├── quick_test.sh              # 一键自检：NN + Gym（非交互）
├── quick_test_nn.sh           # 仅 NN 自检（T=1/T=8，可选导出 ONNX）
├── quick_test_gym.sh          # 仅 Gym 自检
├── setup_env.sh               # 创建环境 + pip install -e .
└── clean.sh                   # 清理缓存 + artifacts/*.onnx

requirements.txt               # 核心依赖（CPU 版本）
setup.py                       # 包装配置（可编辑安装）
environment.yml                # 可选的 conda/mamba 包装
```

关键文件（本周重点）：
- `parking_project_submission/modules/networks.py`：实现 Lidar+Residual+LSTM 架构。
  - 编码器：基础特征 MLP（11→128，LayerNorm+ReLU）与 LiDAR 1D-CNN（+自适应池化→FC→128）。
  - 融合：`fusion_refiner` 残差块（Linear→LN→ReLU→Linear，与残差相加）。
  - 时序层：LSTM(256→128)，策略（高斯，log_std 有界）与价值头。
  - 工具：`initial_state`、`split_flat_obs`；默认 CPU 运行。
- `parking_project_submission/neural_network_demo.py`：前向/采样/反传演示；支持 `--seq-len`、`--export-onnx`。
- `parking_project_submission/gym_demo.py`：随机/手动演示；`--summary`（手动仅摘要）、`--per-step`（随机逐步日志）。
- `parking_project_submission/parking_env/env.py`：环境核心（激光射线、动力学、奖励、渲染叠层）。
- `parking_project_submission/ppo_recurrent.py`：时序 PPO 训练器（LSTM + BPTT）。

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

- **文件：** `parking_project_submission/agent_learning.py`（最小版 PPO）与 `parking_project_submission/ppo_recurrent.py`（时序版 PPO）
- **当前状态：** 已提供最小版（T=1）与时序版（携带 LSTM 记忆 + BPTT）两种训练器。

使用方式（训练/评测/可视化）
- 训练（CPU，最小版）：
  - `python -m parking_project_submission.agent_learning --total-steps 200000 --rollout-len 2048`
- 训练（CPU，时序版，带记忆 + BPTT）：
  - `python -m parking_project_submission.ppo_recurrent --total-steps 200000 --rollout-len 2048 --chunk-len 256 --epochs 4`
- 评测（确定性，均值动作）：
  - `python -m parking_project_submission.agent_learning --eval --checkpoint artifacts/ppo_minimal.pt --eval-episodes 5`
- 在 Gym 演示器中可视化（策略模式）：
  - 均值动作：`parking-gym-demo --mode policy --checkpoint artifacts/ppo_minimal.pt`
  - 随机采样：`parking-gym-demo --mode policy --stochastic --checkpoint artifacts/ppo_minimal.pt`
  - 录制 mp4（需 `pip install imageio imageio-ffmpeg`）：`parking-gym-demo --mode policy --checkpoint artifacts/ppo_minimal.pt --record artifacts/rollout.mp4`

说明
- 最小训练器为简化演示，当前使用环境默认配置；后续会接入 `--config` 以与 Gym 演示的配置生成保持一致。
- 策略模式默认使用均值动作；如需更有随机性的表现，增加 `--stochastic` 即可。

祝调参顺利，泊车顺滑！ 🚗

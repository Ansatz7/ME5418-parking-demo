# Parking Environment Demo / 泊车环境演示包

**English** | [中文](#中文指南)

## English Guide

This repository packages the self-contained parking environment used for coursework demos. Below is a quick navigation list; feel free to jump straight to the section you need.

### Table of Contents

1. [Quick Start](#quick-start)
2. [Gym Demo Module](#gym-demo-module)
3. [Neural Network Module](#neural-network-module)
4. [Agent Learning Module](#agent-learning-module)

### Quick Start

The default workflow uses `pip install -e .` so graders can get the demo running in minutes. An optional `environment.yml` is provided for those who prefer Conda/Mamba.

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

#### Repository Layout

```
parking_project_submission/
├── configs/                    # default + sample configs (bilingual notes)
├── gym_demo.py                 # CLI entry (manual/random modes)
├── modules/                    # JSON helpers, workflows, networks
├── parking_env/                # Gymnasium env, GUI tuner, config generator
├── agent_learning.py           # placeholder for future PPO pipeline
└── neural_network_demo.py      # GRU actor-critic demo

requirements.txt
setup.py
environment.yml
```

#### Submission Checklist

- Create a fresh environment (Method A or B) and run `pip install -e .`.
- Execute `parking-gym-demo --mode random` and `--mode manual` to verify rendering & logging.
- Launch the config generator and assist tuner (see [Gym Demo Module](#gym-demo-module)) to confirm read/write access.
- Render this README and ensure every command works as documented.

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
- **Current status:** Recurrent actor-critic (GRU) skeleton implemented; demo script shows a single inference + backward pass using `ParkingEnv`.
- **Next steps:** Integrate PPO training loop, add richer feature encoders, and document hyperparameters. This section will be expanded once the training pipeline lands.

### Agent Learning Module

- **Location:** `parking_project_submission/agent_learning.py`
- **Current status:** CLI stub outlining the planned PPO workflow (argument parsing, placeholders).
- **Roadmap:** Next iteration will replace the stub with the actual training routine, tests, and logging hooks. Documentation will be updated alongside that release.

---

## 中文指南

[English](#english-guide) | **中文**

本仓库为智能泊车演示项目的独立提交包，下面提供快速导航，方便直接跳到所需章节。

### 目录

1. [快速上手](#快速上手)
2. [Gym 模块](#gym-模块)
3. [神经网络模块](#神经网络模块)
4. [Agent Learning 模块](#agent-learning-模块)

### 快速上手

推荐采用 `pip install -e .`，保证助教或阅卷人几分钟内即可运行。若偏好 Conda/Mamba，也提供 `environment.yml`。

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

#### 仓库结构

```
parking_project_submission/
├── configs/                    # 默认配置与示例场景（含双语注释）
├── gym_demo.py                 # 命令行入口（随机 / 手动模式）
├── modules/                    # JSON 工具、工作流、网络结构
├── parking_env/                # 环境实现、调参 GUI、配置生成器
├── agent_learning.py           # PPO 训练入口预留脚本
└── neural_network_demo.py      # GRU Actor-Critic 示例

requirements.txt
setup.py
environment.yml
```

#### 提交前自检

- 按上述步骤在全新环境中完成安装。
- 分别运行 `parking-gym-demo --mode random` 与 `--mode manual` 确认渲染、日志无误。
- 根据 [Gym 模块](#gym-模块) 的说明尝试调参 GUI 与随机地图生成器。
- 渲染 README，逐条核对命令、参数说明与实际表现是否一致。

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
- **当前状态：** 实现了 GRU 架构的 Actor-Critic 骨架，并提供示例脚本演示一次前向推理与反向传播。
- **后续计划：** 集成 PPO 训练流程、扩展特征编码、补充超参数说明。模块文档将在完成训练脚本后同步更新。

### Agent Learning 模块

- **文件：** `parking_project_submission/agent_learning.py`
- **当前状态：** 训练入口的 CLI 占位脚本，仅校验参数并输出计划流程。
- **下一步：** 替换为真实的 PPO 训练循环，加入日志与测试后将完善本章节说明。

祝调参顺利，泊车顺滑！ 🚗

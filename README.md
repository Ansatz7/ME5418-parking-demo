# Parking Environment Demo / 泊车环境演示包

**English** | [中文](#中文指南)

## English Guide

This repository packages the self-contained parking environment used for coursework demos. The default workflow uses `pip install -e .`, with an optional `environment.yml` for teams who prefer Conda/Mamba.

### Environment Setup

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

# smoke test (random policy)
parking-gym-demo --mode random --episodes 1 --max-steps 400

# optional: manual driving
parking-gym-demo --mode manual --episodes 1 --max-steps 400

# leave the environment when finished
mamba deactivate
```

*All commands also work if you substitute `mamba` with `conda`; Mamba simply resolves dependencies faster.*

**Method B: environment.yml + Mamba**

```bash
mamba env create -f environment.yml -n parking-rl-yml
mamba activate parking-rl-yml
parking-gym-demo --mode random --episodes 1 --max-steps 400
mamba deactivate
```

### Repository Layout

```
parking_project_submission/
├── configs/                    # default + sample configs (bilingual notes)
├── gym_demo.py                 # CLI entry (manual/random modes)
├── modules/                    # JSON helpers, workflows, networks
├── parking_env/                # Gymnasium env, GUI tuner, config generator
├── agent_learning.py           # placeholder for future PPO pipeline
└── neural_network_demo.py      # GRU actor-critic demo

requirements.txt                # pip dependencies
setup.py                        # enables `pip install -e .`
environment.yml                 # optional Conda/Mamba wrapper
```

### Core Scripts

- `parking_project_submission.gym_demo` — runs random/manual demos with CLI flags.
- `parking_project_submission.parking_env.generate_training_config` — generates randomized map configs aligned with `DEFAULT_CONFIG`.
- `parking_project_submission.parking_env.assist_model_tuner` — Qt/Matplotlib GUI for steering & velocity assist tuning (`QT_QPA_PLATFORM=offscreen` on headless servers).

Common commands:

```bash
# generate a new scenario
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_random.json

# tune steering/velocity assists and sync back to JSON
python -m parking_project_submission.parking_env.assist_model_tuner \
  --config parking_project_submission/configs/demo_default.json --sync
```

### GPU / Deep RL Notes

- Current demo depends only on CPU-friendly packages (Gymnasium 0.29.1, NumPy 1.23.5, Matplotlib 3.7.1, PyTorch 2.2.2).
- Grading hardware: CUDA 12.4 + RTX 4090. When adding PPO/LSTM training, document the exact PyTorch wheel (e.g., `torch==2.2.2+cu124`).
- Training scripts will arrive in later iterations; their dependencies will be listed separately.

### Submission Checklist

1. Recreate the project in a fresh environment (`pip install -e .` or `environment.yml`).
2. Run `parking-gym-demo --mode random` and `--mode manual` to verify rendering and logging.
3. Execute the config generator and assist tuner to confirm read/write access.
4. Render the README and ensure every command works as documented.

---

## 中文指南

[English](#english-guide) | **中文**

本仓库提供独立可提交的泊车环境演示包。默认安装方式为 `pip install -e .`，也提供 `environment.yml` 以便 Conda/Mamba 用户快速部署。

### 环境搭建

**方法 A（推荐）：pip install -e .**

```bash
# 先确保已安装 mamba（如果更习惯 conda，可把下方指令中的 mamba 全替换为 conda）
conda install -n base -c conda-forge mamba

# 创建干净的 Python 3.10 环境
mamba create -n parking-rl python=3.10 -y
mamba activate parking-rl

# 升级 pip 并以开发模式安装本项目
python -m pip install --upgrade pip
pip install -e .

# 烟雾测试：随机策略
parking-gym-demo --mode random --episodes 1 --max-steps 400

# 可选：测试手动驾驶
parking-gym-demo --mode manual --episodes 1 --max-steps 400

# 完成后退出环境
mamba deactivate
```

*以上命令若将 `mamba` 替换为 `conda` 亦可运行，使用 Mamba 仅是为了加速依赖解析。*

**方法 B：environment.yml + Mamba**

```bash
mamba env create -f environment.yml -n parking-rl-yml
mamba activate parking-rl-yml
parking-gym-demo --mode random --episodes 1 --max-steps 400
mamba deactivate
```

### 仓库结构

```
parking_project_submission/
├── configs/                    # 默认配置与示例场景（含双语注释）
├── gym_demo.py                 # 命令行入口（随机/手动模式）
├── modules/                    # JSON 工具、工作流、网络结构
├── parking_env/                # 环境实现、调参 GUI、配置生成器
├── agent_learning.py           # 预留的 PPO 训练入口
└── neural_network_demo.py      # GRU Actor-Critic 网络示例

requirements.txt
setup.py
environment.yml
```

### 关键脚本

- `parking_project_submission.gym_demo` — 随机/手动模式的统一入口，所有参数由 CLI 指定。
- `parking_project_submission.parking_env.generate_training_config` — 在保持动力学一致的前提下随机生成场景配置。
- `parking_project_submission.parking_env.assist_model_tuner` — Qt/Matplotlib GUI 调整方向盘与纵向助力（无显示环境可设置 `QT_QPA_PLATFORM=offscreen`）。

常用命令：

```bash
# 生成训练配置
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_random.json

# 启动助力调参 GUI 并同步回 JSON
python -m parking_project_submission.parking_env.assist_model_tuner \
  --config parking_project_submission/configs/demo_default.json --sync
```

### GPU / 深度强化学习提示

- 当前示例仅依赖 CPU 版本的 Gymnasium 0.29.1、NumPy 1.23.5、Matplotlib 3.7.1 与 PyTorch 2.2.2。
- 评分机器提供 **CUDA 12.4** 与 **RTX 4090**。若后续加入 PPO/LSTM 训练，请注明具体可用的 PyTorch 轮子（如 `torch==2.2.2+cu124`）。
- 训练脚本稍后补充，会单独列出新增依赖。

### 提交前自检

1. 在全新环境中按 README 步骤完成安装（`pip install -e .` 或 `environment.yml`）。
2. 分别运行 `parking-gym-demo --mode random` 与 `--mode manual`，确认渲染与日志正常。
3. 执行配置生成器与助力调参 GUI，确保具备读写权限。
4. 查看渲染后的 README，逐条核对命令是否可复现。

祝调参顺利，泊车顺滑！ 🚗

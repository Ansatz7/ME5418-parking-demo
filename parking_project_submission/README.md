# parking_project_submission

独立提交包所在目录，包含以下子模块：

- `parking_env/`：环境定义及辅助脚本（`env.py`、`assist_model_tuner.py`、`generate_training_config.py`）。
- `modules/`：演示工作流与通用工具（`workflow.py`、`utils.py` 等）。
- `gym_demo.py`：命令行入口，支持随机策略与手动驾驶。
- `configs/`：默认配置与示例训练配置（`demo_default.json`、`train_001.json`）。
- `agent_learning.py`、`neural_network_demo.py`：后续算法/网络演示脚本。

## 安装与运行

### 方式 A：纯 pip（推荐给助教）
```bash
python -m venv .venv
source .venv/bin/activate          # Windows 使用 .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .

# 随机策略演示
parking-gym-demo --mode random --episodes 1

# 手动驾驶演示
parking-gym-demo --mode manual --episodes 1
```
默认会读取 `parking_project_submission/configs/demo_default.json`，如需自定义配置可通过
`--config path/to/config.json` 指定。

不带任何参数运行 `parking-gym-demo` 时，会直接进入手动模式，单回合步数为 `max_steps=4000`。

### 方式 B：uv / mamba（可选）
- `uv pip install -e .`：在任意虚拟环境中执行，速度快。
- `mamba env create -f ../environment.yml`：如果喜欢 Conda/Mamba，可用提供的 `environment.yml`。

> 提交前务必在全新环境或另一台机器上照 README 步骤完整跑一遍，确认流程无误。

## 其他脚本

```bash
# 生成训练配置示例
python -m parking_project_submission.parking_env.generate_training_config \
  --out parking_project_submission/configs/train_auto.json

# 打开助力模型调参 GUI
python -m parking_project_submission.parking_env.assist_model_tuner \
  --config parking_project_submission/configs/demo_default.json --sync
```

如需在 GPU 上训练，请根据目标机器的 CUDA 版本选择合适的 PyTorch 轮子。提交时务必说明
所需的版本（测试机提供 CUDA 12.4 / NVIDIA RTX 4090）。

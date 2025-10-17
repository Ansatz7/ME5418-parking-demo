
# 停车环境演示项目（Parking Environment Demo）

这是一个**模块化的停车环境仿真项目**，可用于交互式演示、辅助转向控制调试等任务。  
整个流程基于 Conda 环境 (`parking-rl`) 构建，通过 Jupyter Notebook 来统一调用各个脚本。

---

## 环境配置

- Conda 版本：**25.7.0**（Anaconda / Miniconda 发行版）
- 环境定义文件：[`environment.yml`](environment.yml)

在新机器上重新创建环境：

```bash
conda env create -f environment.yml
conda activate parking-rl
python -m ipykernel install --user --name parking-rl --display-name "python3 (parking-rl)"
````

如果环境已存在：

```bash
conda env update -f environment.yml --prune
conda activate parking-rl
```

系统需支持 Qt 图形界面（环境已包含 `pyqt>=5.15`）。
若在无界面的 Linux 环境中运行，可设置：

```bash
export QT_QPA_PLATFORM=offscreen
```

或使用 X11 转发。

该环境自带 Jupyter Notebook，可直接打开并运行。
打开 `ParkingEnv_Demo.ipynb` 时请选择内核：
**`python3 (parking-rl)`**，确保在正确环境中运行。

---

## 项目结构

```
parking_project/
├── generate_training_config.py   # 生成随机训练配置文件
├── main.py                       # 命令行入口（手动/随机停车模式）
├── parking_gym.py                # Gymnasium 环境定义与辅助函数
├── assist_model_tuner.py         # Qt + Matplotlib 辅助控制调节界面
├── generated_configs/
│   ├── notebook_override.json    # 由 Notebook 实时修改的配置文件
│   └── train_001.json            # 由命令行生成的示例配置
└── ParkingEnv_Demo.ipynb         # Notebook 前端，整合所有功能

README.md                         # 本文件
environment.yml                   # Conda 环境定义文件
```

项目采用模块化设计：

* JSON 配置文件记录场景与参数；
* Notebook 负责高层控制与交互；
* 各脚本负责命令行演示与调参工具。

---

## 默认场景参数

* 时间步长：`dt = 0.1 s`，每回合最大步数 `max_steps = 4000`
* 场地大小：60 m × 60 m，车辆初始区域 `[-6, 6] × [-6, 6]`
* 九束激光雷达：角度 `[-135, -90, -60, -30, 30, 60, 90, 135, 0]`°，最大测距 12 m
* 转向范围：±60°，角速度 ±30°/s
* 转向辅助参数：`Kp=52.5`, `Kd=8.75`, 死区 0.03 rad/s²
* 纵向阻尼参数：`K=2.45`，速度死区 0.03 m/s
* 默认启用辅助控制
* 观测噪声：σ=0.005（可通过 `env.unwrapped.set_observation_noise(enabled=False)` 关闭）

这些默认值在 `parking_gym.DEFAULT_CONFIG` 中定义。
Notebook 使用的 `generated_configs/notebook_override.json` 与其保持同步，可安全修改。

---

## 使用方法

### 1. 启动 Notebook

```bash
cd parking_project
jupyter notebook ParkingEnv_Demo.ipynb
```

Notebook 会引导你完成：

1. 设置运行参数（回合数、步数、模式等）
2. 选择或编辑配置文件
3. 启动辅助模型调节器 (`assist_model_tuner.py`)
4. 启动停车演示 (`main.py`)
5. 查看环境使用技巧（状态/动作空间、噪声控制等）

---

### 2. 命令行方式调节辅助模型

```bash
conda run -n parking-rl python assist_model_tuner.py \
  --config generated_configs/notebook_override.json \
  --angle0 20 --rate0 0 --steps 200 --sync
```

滑块可实时调节转向回正与速度阻尼；
加上 `--sync` 参数时，调整结果会自动写回 JSON 文件。

---

### 3. 命令行停车演示

```bash
conda run -n parking-rl python main.py \
  --mode manual --episodes 1 --max-steps 4000 \
  --config generated_configs/notebook_override.json
```

可选参数：

* `--mode random`：使用随机控制器
* `--sleep-scale <float>`：调整动画播放速度

---

### 4. 生成训练配置文件

```bash
conda run -n parking-rl python generate_training_config.py \
  --out generated_configs/train_001.json
```

此命令会基于默认配置随机生成新场景（车位、障碍物等），
不会修改控制参数，确保调试一致性。

---

## 开发者提示

* 推荐 Python 版本：**3.8**（兼容 Gymnasium 0.29）
* 图形界面依赖 Matplotlib 的 `QtAgg` 后端
* JSON 文件支持中英文注释并保持 UTF-8 编码
* `generate_training_config.py` 仅随机场景部分，控制参数保持一致
* 若在 Notebook 外修改配置文件，请启用 `--sync` 保持同步

---

**祝你调参顺利，停车无忧！🚗**

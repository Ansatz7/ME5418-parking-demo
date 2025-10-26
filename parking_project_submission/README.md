# parking_project_submission

该目录用于打包提交一个独立可运行的泊车环境 Demo。核心模块位于
`parking_project_submission/parking_project`，包含：

- `parking_gym.py`：Gymnasium 环境定义。
- `main.py`：命令行演示入口，支持手动与随机策略模式。
- `generate_training_config.py`：随机生成训练配置的工具脚本。
- `assist_model_tuner.py`：Qt/Matplotlib 辅助模型调参 GUI。

示例用法：

```bash
# 随机策略演示
python -m parking_project_submission.parking_project.main --mode random --episodes 1

# 生成训练配置
python -m parking_project_submission.parking_project.generate_training_config \
  --out parking_project_submission/parking_project/generated_configs/train_auto.json
```

后续我们会在此分支上逐步完善提交流程所需的依赖、脚本与文档。

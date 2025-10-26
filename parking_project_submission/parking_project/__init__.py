"""打包提交用的停车环境模块。

暴露核心的 `ParkingEnv` 环境类以及默认配置 `DEFAULT_CONFIG`，方便外部脚本
通过 `parking_project_submission.parking_project` 导入。
"""

from .parking_gym import ParkingEnv, DEFAULT_CONFIG

__all__ = ["ParkingEnv", "DEFAULT_CONFIG"]

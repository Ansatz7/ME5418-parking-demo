"""Parking environment package exposed to submission scripts.

提供 `ParkingEnv` 环境类与默认配置 `DEFAULT_CONFIG` 的统一导入口。
"""

from .env import ParkingEnv, DEFAULT_CONFIG

__all__ = ["ParkingEnv", "DEFAULT_CONFIG"]

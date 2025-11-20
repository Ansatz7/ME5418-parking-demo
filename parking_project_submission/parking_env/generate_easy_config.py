"""Generate an easy ("baby level") ParkingEnv configuration.

Features:
- Vehicle spawns ANYWHERE in the map with ANY rotation.
- Parking slot spawns RELATIVE to the vehicle.
- Curriculum Distribution (Corrected):
    - 5%:  Instant Victory (Inside slot) - 免费午餐
    - 15%: Front (Close) - 前向 1.0~2.0m
    - 30%: Behind (Very Close) - 后向 0.5~0.75m
    - 30%: Behind (Close) - 后向 0.75~1.0m
    - 20%: Behind (Medium) - 后向 1.0~2.5m
- Obstacles: Spawn relative to the parking slot (left/right sides) to guide the agent.
- ONLY exports map-related configs.
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt

from .env import DEFAULT_CONFIG, ParkingEnv
from parking_project_submission.modules.utils import write_json


def sample_easy_config(seed: Optional[int] = None) -> Dict:
    rng = random.Random(seed)
    # 使用 deepcopy 确保不修改全局配置，且只修改地图相关参数
    full_config: Dict = deepcopy(DEFAULT_CONFIG)

    full_config["rng_seed"] = rng.randint(0, 1_000_000)
    field_size = float(full_config["field_size"])

    # 1. 出生区域：全图随机
    # 满足“出生点不必在中央”的要求，增加场景多样性
    safe_margin = 4.0
    limit = field_size / 2.0 - safe_margin
    full_config["spawn_region"] = [-limit, limit, -limit, limit]

    # 2. 车位设置：相对生成模式
    slot_cfg = full_config["parking_slot"].copy()
    slot_cfg["relative_placement"] = True 

    # --- 概率分布逻辑 (已更正注释以匹配代码) ---
    prob = rng.random() # 生成 0.0 到 1.0 之间的随机数

    if prob < 0.05:
        # [5% 概率] 出生即胜利 (Instant Victory)
        # 放在车身中心 -0.2 ~ 0.2 米处
        # 作用：让 Critic 快速学会“终点状态”的高价值
        slot_cfg["offset_x_range"] = (-0.2, 0.2)
        slot_cfg["offset_y_range"] = (-0.1, 0.1)

    elif prob < 0.20:
        # [15% 概率] 前方一点点 (Front, Close)
        # 放在车头前方 1.0 ~ 2.0 米
        # 作用：学会简单的油门控制
        slot_cfg["offset_x_range"] = (1.0, 2.0)
        slot_cfg["offset_y_range"] = (-0.2, 0.2)

    elif prob < 0.50:
        # [30% 概率] 后方非常近 (Behind, Very Close)
        # 放在车尾后方 0.5 ~ 0.75 米
        # 作用：倒车入库的极简入门，几乎只要直退
        slot_cfg["offset_x_range"] = (-0.75, -0.5)
        slot_cfg["offset_y_range"] = (-0.2, 0.2)

    elif prob < 0.80:
        # [30% 概率] 后方稍近 (Behind, Close)
        # 放在车尾后方 0.75 ~ 1.0 米
        # 作用：稍微增加一点距离感
        slot_cfg["offset_x_range"] = (-1.0, -0.75)
        slot_cfg["offset_y_range"] = (-0.2, 0.2)

    else:
        # [20% 概率] 后方中等距离 (Behind, Medium)
        # 放在车尾后方 1.0 ~ 2.5 米
        # 作用：主力训练区间，距离适中
        slot_cfg["offset_x_range"] = (-2.5, -1.0)
        slot_cfg["offset_y_range"] = (-0.2, 0.2)

    # 统一设置：角度偏差极小，降低难度，专注练习距离控制
    slot_cfg["orientation_range"] = (-0.1, 0.1)
    
    full_config["parking_slot"] = slot_cfg

    # 3. 静态障碍物设置：在车位两侧生成
    static_cfg = full_config["static_obstacles"].copy()
    static_cfg["count"] = 2           # 左右各放一些，形成通道
    static_cfg["relative_to_slot"] = True  # <--- 新增开关：相对于车位生成
    static_cfg["side_distance"] = 3.5      # <--- 新增参数：离车位中心的横向距离 (比如左右3.5米处)
    static_cfg["x_random"] = 2.0           # <--- 新增参数：前后位置的随机范围
    
    # size_range 也可以稍微调整，不需要太大
    static_cfg["size_range"] = (0.5, 1.0)
    
    full_config["static_obstacles"] = static_cfg

    # 动态障碍物：保留一个龟速的，离远点
    dynamic_cfg = full_config["dynamic_obstacles"].copy()
    dynamic_cfg["count"] = 1
    dynamic_cfg["min_distance"] = 6.0 
    dynamic_cfg["speed_range"] = (0.1, 0.2)
    full_config["dynamic_obstacles"] = dynamic_cfg

    # 只保留地图相关的字段
    map_keys = [
        "rng_seed",
        "field_size",
        "spawn_region",
        "parking_slot",
        "static_obstacles",
        "dynamic_obstacles"
    ]
    
    map_only_config = {k: full_config[k] for k in map_keys if k in full_config}
    
    return map_only_config


def save_preview(config: Dict, img_path: Path) -> None:
    """Render the generated config to a PNG file (headless)."""
    print(f"Generating preview image at {img_path} ...")
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    _orig_show = plt.show
    _orig_pause = plt.pause
    plt.show = lambda *args, **kwargs: None
    plt.pause = lambda *args, **kwargs: None

    try:
        env = ParkingEnv(config=config)
        env.reset()
        env.render()
        if env.fig:
            env.fig.savefig(str(img_path), dpi=100, bbox_inches='tight')
        env.close()
    except Exception as e:
        print(f"Warning: Failed to save preview image: {e}")
    finally:
        plt.show = _orig_show
        plt.pause = _orig_pause
        matplotlib.use(original_backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EASY ParkingEnv configs.")
    default_out = Path("parking_project_submission/configs/train_easy.json")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = sample_easy_config(args.seed)
    write_json(args.out, config)
    print(f"Wrote EASY config to {args.out}")

    if not args.no_preview:
        img_path = args.out.with_suffix(".png")
        save_preview(config, img_path)


if __name__ == "__main__":
    main()
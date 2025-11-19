"""Generate an easy ("baby level") ParkingEnv configuration.

Features:
- Vehicle spawns ANYWHERE in the map with ANY rotation.
- Parking slot spawns RELATIVE to the vehicle (always close behind).
- Obstacles are present but sparse and slow.
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
    config: Dict = deepcopy(DEFAULT_CONFIG)

    config["rng_seed"] = rng.randint(0, 1_000_000)
    field_size = float(config["field_size"])

    # 1. 出生区域：全图随机！ (满足你的要求：不必在中央，角度任意)
    # 我们留一点边距(4m)防止车直接生在墙里
    safe_margin = 4.0
    limit = field_size / 2.0 - safe_margin
    config["spawn_region"] = [-limit, limit, -limit, limit]

    # 2. 车位设置：开启“相对生成模式” (Relative Placement)
    slot_cfg = config["parking_slot"].copy()
    slot_cfg["relative_placement"] = True  # <--- 关键开关：告诉env.py使用相对坐标

    # 相对坐标设定 (相对于车身)：
    # X: 在车后方 3.0 到 5.0 米处 (倒车入库的完美起手式)
    slot_cfg["offset_x_range"] = (-5.0, -3.0)
    # Y: 左右偏移很小 (0.5米内)，几乎正对
    slot_cfg["offset_y_range"] = (-0.5, 0.5)
    # Angle: 角度偏差极小 (+/- 5度)，几乎平行
    slot_cfg["orientation_range"] = (-0.1, 0.1)
    
    config["parking_slot"] = slot_cfg

    # 3. 障碍物：有，但很简单
    # 静态：1个，离车至少5米远
    static_cfg = config["static_obstacles"].copy()
    static_cfg["count"] = 1
    static_cfg["min_distance"] = 5.0
    config["static_obstacles"] = static_cfg

    # 动态：1个，离车至少6米远，龟速
    dynamic_cfg = config["dynamic_obstacles"].copy()
    dynamic_cfg["count"] = 1
    dynamic_cfg["min_distance"] = 6.0
    dynamic_cfg["speed_range"] = (0.2, 0.5) # 很慢
    config["dynamic_obstacles"] = dynamic_cfg

    return config


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
"""Utilities for producing randomized ParkingEnv configuration files.

Only map-related fields (spawn region, parking slot placement, obstacle
layouts) are randomized. Timing parameters, lidar layout, and assist-model
settings stay at their defaults so that other tooling (GUI, jupyter notebook cells)
remains the single source of truth for those values.
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

from parking_gym import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Utility helpers 工具函数
# ---------------------------------------------------------------------------
def _sample_range(center: float, span: float) -> Tuple[float, float]:
    """Return a bounded interval with rounding, given center and total span.

    根据中心值与区间长度生成带四舍五入的范围 (lo, hi)。
    """
    half = span / 2.0
    lo = center - half
    hi = center + half
    return (round(lo, 2), round(hi, 2))


# ---------------------------------------------------------------------------
# Core sampler 采样核心逻辑
# ---------------------------------------------------------------------------
def sample_training_config(seed: Optional[int] = None) -> Dict:
    """Create a randomized map config derived from DEFAULT_CONFIG.

    基于 DEFAULT_CONFIG 随机化场景相关字段，返回新的配置字典。
    """
    rng = random.Random(seed)
    config: Dict = deepcopy(DEFAULT_CONFIG)

    config["rng_seed"] = rng.randint(0, 1_000_000)

    field_size = float(config["field_size"])

    # Spawn region around the field center with moderate variation.
    spawn_span_x = field_size * rng.uniform(0.24, 0.36)
    spawn_span_y = field_size * rng.uniform(0.22, 0.34)
    spawn_center_x = rng.uniform(-field_size * 0.12, field_size * 0.12)
    spawn_center_y = rng.uniform(-field_size * 0.12, field_size * 0.12)
    config["spawn_region"] = [
        round(spawn_center_x - spawn_span_x / 2.0, 2),
        round(spawn_center_x + spawn_span_x / 2.0, 2),
        round(spawn_center_y - spawn_span_y / 2.0, 2),
        round(spawn_center_y + spawn_span_y / 2.0, 2),
    ]

    # Parking slot geometry and placement.
    slot_cfg = config["parking_slot"].copy()
    slot_cfg["length"] = round(slot_cfg["length"] + rng.uniform(-0.4, 0.6), 2)
    slot_cfg["width"] = round(slot_cfg["width"] + rng.uniform(-0.3, 0.4), 2)

    offset_x_center = rng.uniform(-field_size * 0.36, -field_size * 0.18)
    offset_x_span = field_size * rng.uniform(0.10, 0.18)
    slot_cfg["offset_x_range"] = _sample_range(offset_x_center, offset_x_span)

    offset_y_center = rng.uniform(-field_size * 0.28, field_size * 0.28)
    offset_y_span = field_size * rng.uniform(0.14, 0.24)
    slot_cfg["offset_y_range"] = _sample_range(offset_y_center, offset_y_span)

    angle_span = round(rng.uniform(6.0, 18.0), 2)
    slot_cfg["orientation_range"] = (-angle_span, angle_span)
    config["parking_slot"] = slot_cfg

    # Static obstacle clutter.
    static_cfg = config["static_obstacles"].copy()
    static_cfg["count"] = rng.randint(1, 4)
    static_cfg["size_range"] = (
        round(rng.uniform(0.8, 1.5), 2),
        round(rng.uniform(1.7, 3.0), 2),
    )
    static_cfg["min_distance"] = round(rng.uniform(1.5, 3.2), 2)
    static_cfg["seed"] = rng.randint(0, 10_000)
    config["static_obstacles"] = static_cfg

    # Dynamic obstacle behaviour (count/radius/paths) is part of the scene.
    dynamic_cfg = config["dynamic_obstacles"].copy()
    dynamic_cfg["count"] = rng.randint(0, 2)
    dynamic_cfg["radius"] = round(rng.uniform(0.8, 1.4), 2)
    dynamic_cfg["speed_range"] = (
        round(rng.uniform(0.4, 0.8), 2),
        round(rng.uniform(0.85, 1.25), 2),
    )
    dynamic_cfg["behavior"] = rng.choice(["goal_driven", "random_walk", "patrol"])
    dynamic_cfg["min_distance"] = round(rng.uniform(3.5, 5.5), 2)
    dynamic_cfg["heading_noise"] = round(rng.uniform(10.0, 18.0), 2)
    config["dynamic_obstacles"] = dynamic_cfg

    # Leave reward, vehicle, and lidar settings untouched for external tooling.
    return config


# ---------------------------------------------------------------------------
# CLI plumbing 命令行接口
# ---------------------------------------------------------------------------
def write_config(config: Dict, path: Path) -> None:
    """Serialize config to JSON with UTF-8 and trailing newline.

    以 UTF-8 写出 JSON 并保留结尾换行，便于版本管理对比。
    """
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for the generator utility.

    解析命令行选项，控制生成文件的输出路径与随机种子。
    """
    parser = argparse.ArgumentParser(description="Generate randomized ParkingEnv configs.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic generation.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point that generates and writes a config file.

    命令行入口：生成配置并写入磁盘文件。
    """
    args = parse_args()
    # Deterministic when --seed supplied; otherwise rely on fresh RNG jitter.
    # 当传入 --seed 时结果可复现，否则每次随机生成全新场景。
    config = sample_training_config(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_config(config, args.out)
    print(f"Wrote config to {args.out}")


if __name__ == "__main__":
    main()

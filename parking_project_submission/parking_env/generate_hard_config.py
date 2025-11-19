"""Generate a hard-difficulty ParkingEnv configuration.

Hard configs are designed to be significantly more challenging than the
default demo:

- spawn region is further from the parking slot,
- parking slot orientation has a wider random range,
- more static obstacles with larger size variation,
- one or more dynamic obstacles with non-trivial motion.

As with the other generators, vehicle dynamics, rewards and lidar layout
remain identical to ``DEFAULT_CONFIG``.
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from .env import DEFAULT_CONFIG
from parking_project_submission.modules.utils import write_json


def sample_hard_config(seed: Optional[int] = None) -> Dict:
    rng = random.Random(seed)
    config: Dict = deepcopy(DEFAULT_CONFIG)

    field_size = float(config["field_size"])

    # Spawn region: further away and larger area.
    span_x = field_size * rng.uniform(0.30, 0.45)
    span_y = field_size * rng.uniform(0.25, 0.40)
    center_x = rng.uniform(-field_size * 0.20, field_size * 0.20)
    center_y = rng.uniform(-field_size * 0.20, field_size * 0.20)
    config["spawn_region"] = [
        round(center_x - span_x / 2.0, 2),
        round(center_x + span_x / 2.0, 2),
        round(center_y - span_y / 2.0, 2),
        round(center_y + span_y / 2.0, 2),
    ]

    # Parking slot: more variation in pose and size.
    slot_cfg = config["parking_slot"].copy()
    slot_cfg["length"] = round(slot_cfg["length"] + rng.uniform(-0.5, 0.8), 2)
    slot_cfg["width"] = round(slot_cfg["width"] + rng.uniform(-0.4, 0.6), 2)

    offset_x_center = rng.uniform(-field_size * 0.40, -field_size * 0.20)
    offset_x_span = field_size * rng.uniform(0.14, 0.22)
    slot_cfg["offset_x_range"] = (
        round(offset_x_center - offset_x_span / 2.0, 2),
        round(offset_x_center + offset_x_span / 2.0, 2),
    )

    offset_y_center = rng.uniform(-field_size * 0.30, field_size * 0.30)
    offset_y_span = field_size * rng.uniform(0.18, 0.28)
    slot_cfg["offset_y_range"] = (
        round(offset_y_center - offset_y_span / 2.0, 2),
        round(offset_y_center + offset_y_span / 2.0, 2),
    )

    angle_span = round(rng.uniform(10.0, 25.0), 2)
    slot_cfg["orientation_range"] = (-angle_span, angle_span)
    config["parking_slot"] = slot_cfg

    # Static obstacles: more clutter, larger boxes.
    static_cfg = config["static_obstacles"].copy()
    static_cfg["count"] = rng.randint(3, 6)
    static_cfg["size_range"] = (
        round(rng.uniform(1.0, 2.0), 2),
        round(rng.uniform(2.0, 3.2), 2),
    )
    static_cfg["min_distance"] = round(rng.uniform(1.5, 3.0), 2)
    static_cfg["seed"] = rng.randint(0, 10_000)
    config["static_obstacles"] = static_cfg

    # Dynamic obstacles: enable 1–2 moving discs with varied behaviour.
    dynamic_cfg = config["dynamic_obstacles"].copy()
    dynamic_cfg["count"] = rng.randint(1, 2)
    dynamic_cfg["radius"] = round(rng.uniform(0.8, 1.4), 2)
    dynamic_cfg["speed_range"] = (
        round(rng.uniform(0.5, 1.0), 2),
        round(rng.uniform(1.0, 1.4), 2),
    )
    dynamic_cfg["behavior"] = rng.choice(["goal_driven", "random_walk", "patrol"])
    dynamic_cfg["min_distance"] = round(rng.uniform(3.5, 5.5), 2)
    dynamic_cfg["heading_noise"] = round(rng.uniform(10.0, 20.0), 2)
    config["dynamic_obstacles"] = dynamic_cfg

    config["rng_seed"] = rng.randint(0, 1_000_000)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HARD ParkingEnv configs.")
    default_out = Path("parking_project_submission/configs/train_hard.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output JSON file path (default: {default_out}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = sample_hard_config(args.seed)
    write_json(args.out, config)
    print(f"Wrote HARD config to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()


"""Generate a medium-difficulty ParkingEnv configuration.

Medium configs modestly increase the difficulty relative to the default demo:

- spawn region is moderately far from the slot,
- parking slot orientation has a small random range,
- a small number of static obstacles are present,
- dynamic obstacles are disabled (keeps interactions simpler).

Vehicle dynamics, rewards and lidar layout remain identical to
``DEFAULT_CONFIG`` for compatibility with other tooling.
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from .env import DEFAULT_CONFIG
from parking_project_submission.modules.utils import write_json


def sample_medium_config(seed: Optional[int] = None) -> Dict:
    rng = random.Random(seed)
    config: Dict = deepcopy(DEFAULT_CONFIG)

    field_size = float(config["field_size"])

    # Spawn region: centred but larger than EASY, smaller than HARD.
    span_x = field_size * rng.uniform(0.25, 0.35)
    span_y = field_size * rng.uniform(0.20, 0.30)
    center_x = rng.uniform(-field_size * 0.10, field_size * 0.10)
    center_y = rng.uniform(-field_size * 0.10, field_size * 0.10)
    config["spawn_region"] = [
        round(center_x - span_x / 2.0, 2),
        round(center_x + span_x / 2.0, 2),
        round(center_y - span_y / 2.0, 2),
        round(center_y + span_y / 2.0, 2),
    ]

    # Parking slot: slight randomisation in size and orientation.
    slot_cfg = config["parking_slot"].copy()
    slot_cfg["length"] = round(slot_cfg["length"] + rng.uniform(-0.2, 0.4), 2)
    slot_cfg["width"] = round(slot_cfg["width"] + rng.uniform(-0.2, 0.3), 2)

    offset_x_center = rng.uniform(-field_size * 0.30, -field_size * 0.16)
    offset_x_span = field_size * rng.uniform(0.08, 0.14)
    slot_cfg["offset_x_range"] = (
        round(offset_x_center - offset_x_span / 2.0, 2),
        round(offset_x_center + offset_x_span / 2.0, 2),
    )

    offset_y_center = rng.uniform(-field_size * 0.22, field_size * 0.22)
    offset_y_span = field_size * rng.uniform(0.10, 0.20)
    slot_cfg["offset_y_range"] = (
        round(offset_y_center - offset_y_span / 2.0, 2),
        round(offset_y_center + offset_y_span / 2.0, 2),
    )

    angle_span = round(rng.uniform(4.0, 10.0), 2)
    slot_cfg["orientation_range"] = (-angle_span, angle_span)
    config["parking_slot"] = slot_cfg

    # Static obstacles: a small number of moderate-sized boxes.
    static_cfg = config["static_obstacles"].copy()
    static_cfg["count"] = rng.randint(1, 3)
    static_cfg["size_range"] = (
        round(rng.uniform(0.8, 1.4), 2),
        round(rng.uniform(1.6, 2.8), 2),
    )
    static_cfg["min_distance"] = round(rng.uniform(1.8, 3.0), 2)
    static_cfg["seed"] = rng.randint(0, 10_000)
    config["static_obstacles"] = static_cfg

    # Dynamic obstacles: disabled at medium level to focus on geometry.
    dynamic_cfg = config["dynamic_obstacles"].copy()
    dynamic_cfg["count"] = 0
    config["dynamic_obstacles"] = dynamic_cfg

    config["rng_seed"] = rng.randint(0, 1_000_000)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MEDIUM ParkingEnv configs.")
    default_out = Path("parking_project_submission/configs/train_medium.json")
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
    config = sample_medium_config(args.seed)
    write_json(args.out, config)
    print(f"Wrote MEDIUM config to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()


"""Utilities for producing randomized ParkingEnv configuration files.

All angle fields in the produced configs are expressed in degrees. The gym
environment converts them to radians internally, so downstream consumers should
also supply overrides in degrees.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from parking_gym import DEFAULT_CONFIG


def _sample_range(center: float, spread: float) -> Tuple[float, float]:
    half = spread / 2.0
    lo = center - half
    hi = center + half
    return (round(lo, 2), round(hi, 2))


def _jitter_angles(base: List[float], rng: random.Random, jitter_deg: float = 6.0) -> List[float]:
    return [angle + rng.uniform(-jitter_deg, jitter_deg) for angle in base]


def sample_training_config(seed: Optional[int] = None) -> Dict:
    rng = random.Random(seed)
    config: Dict = {}

    # Base timing and global geometry.
    config["dt"] = round(rng.uniform(0.07, 0.13), 3)
    field_size = rng.uniform(18.0, 32.0)
    config["field_size"] = round(field_size, 2)
    config["ray_max_range"] = round(field_size * rng.uniform(0.4, 0.7), 2)

    jittered_angles = _jitter_angles(DEFAULT_CONFIG["ray_angles"], rng)
    config["ray_angles"] = jittered_angles

    spawn_half_x = field_size * rng.uniform(0.22, 0.38)
    spawn_half_y = field_size * rng.uniform(0.22, 0.38)
    spawn_center_x = rng.uniform(-field_size * 0.15, field_size * 0.15)
    spawn_center_y = rng.uniform(-field_size * 0.15, field_size * 0.15)
    config["spawn_region"] = [
        round(spawn_center_x - spawn_half_x, 2),
        round(spawn_center_x + spawn_half_x, 2),
        round(spawn_center_y - spawn_half_y, 2),
        round(spawn_center_y + spawn_half_y, 2),
    ]

    # Vehicle geometry: sample dimensions while keeping kinematics reasonable.
    vehicle_cfg = DEFAULT_CONFIG["vehicle"].copy()
    vehicle_cfg.update(
        {
            "length": round(rng.uniform(3.6, 4.6), 2),
            "width": round(rng.uniform(1.7, 2.1), 2),
            "wheel_base": round(rng.uniform(2.2, 2.8), 2),
            "max_speed": round(rng.uniform(2.2, 3.8), 2),
            "max_reverse_speed": round(rng.uniform(-2.8, -1.6), 2),
            "max_steering_angle": round(rng.uniform(35.0, 50.0), 2),
            "max_steering_rate": round(rng.uniform(50.0, 75.0), 2),
            "steering_damping": round(rng.uniform(0.4, 0.8), 2),
            "enable_steering_assist": rng.choice([True, False]),
        }
    )
    config["vehicle"] = vehicle_cfg

    # Parking slot placement.
    slot_cfg = DEFAULT_CONFIG["parking_slot"].copy()
    slot_cfg.update(
        {
            "length": round(vehicle_cfg["length"] + rng.uniform(0.6, 1.7), 2),
            "width": round(vehicle_cfg["width"] + rng.uniform(0.3, 1.1), 2),
            "offset_x_range": _sample_range(
                rng.uniform(-field_size * 0.45, -field_size * 0.2),
                field_size * rng.uniform(0.08, 0.22),
            ),
            "offset_y_range": _sample_range(
                rng.uniform(-field_size * 0.25, field_size * 0.25),
                field_size * rng.uniform(0.18, 0.3),
            ),
            "orientation_range": (
                -(angle_span := round(rng.uniform(4.0, 15.0), 2)),
                angle_span,
            ),
        }
    )
    config["parking_slot"] = slot_cfg

    # Static obstacles for clutter.
    static_cfg = DEFAULT_CONFIG["static_obstacles"].copy()
    static_cfg.update(
        {
            "count": rng.randint(1, 5),
            "size_range": (
                round(rng.uniform(0.6, 1.4), 2),
                round(rng.uniform(1.6, 3.2), 2),
            ),
            "min_distance": round(rng.uniform(1.3, 2.8), 2),
            "seed": rng.randint(0, 10_000),
        }
    )
    config["static_obstacles"] = static_cfg

    # Dynamic obstacles with varied behaviour.
    dynamic_cfg = DEFAULT_CONFIG["dynamic_obstacles"].copy()
    dynamic_cfg.update(
        {
            "count": rng.randint(0, 3),
            "radius": round(rng.uniform(0.7, 1.5), 2),
            "speed_range": (
                round(rng.uniform(0.35, 0.85), 2),
                round(rng.uniform(0.9, 1.5), 2),
            ),
            "behavior": rng.choice(["goal_driven", "random_walk", "patrol"]),
            "min_distance": round(rng.uniform(3.5, 5.0), 2),
            "heading_noise": round(rng.uniform(10.0, 20.0), 2),
        }
    )
    config["dynamic_obstacles"] = dynamic_cfg

    # Reward shaping tweaks stay close to defaults but allow limited variation.
    reward_cfg = DEFAULT_CONFIG["reward"].copy()
    reward_cfg.update(
        {
            "distance_scale": round(rng.uniform(1.0, 2.0), 2),
            "heading_scale": round(rng.uniform(0.35, 0.75), 2),
            "collision": round(rng.uniform(-170.0, -80.0), 2),
            "success": round(rng.uniform(110.0, 170.0), 2),
            "smoothness": round(rng.uniform(0.02, 0.08), 3),
            "step_cost": round(rng.uniform(0.1, 0.25), 3),
            "velocity_penalty": round(rng.uniform(0.2, 0.45), 3),
        }
    )
    config["reward"] = reward_cfg

    success_cfg = DEFAULT_CONFIG["success_thresholds"].copy()
    success_cfg.update(
        {
            "position": round(rng.uniform(0.25, 0.55), 3),
            "orientation": round(rng.uniform(3.5, 8.0), 2),
            "speed": round(rng.uniform(0.18, 0.45), 3),
            "steering": round(rng.uniform(3.5, 6.5), 2),
        }
    )
    config["success_thresholds"] = success_cfg

    config["rng_seed"] = rng.randint(0, 1_000_000)
    return config


def write_config(config: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    config = sample_training_config(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_config(config, args.out)
    print(f"Wrote config to {args.out}")


if __name__ == "__main__":
    main()

"""Generate an easy ("baby level") ParkingEnv configuration.

The easy config simplifies the scene for curriculum learning:
- Spawn region and parking slot are very close ("face-to-face").
- Orientation is perfectly aligned (0 deg).
- Obstacles are present but sparse, far away, and slow-moving.

Also supports generating a preview image of the map layout.
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

# Use non-interactive backend for headless rendering
import matplotlib
import matplotlib.pyplot as plt

from .env import DEFAULT_CONFIG, ParkingEnv
from parking_project_submission.modules.utils import write_json


def sample_easy_config(seed: Optional[int] = None) -> Dict:
    """Return an easy-level config derived from :data:`DEFAULT_CONFIG`.

    Difficulty adjustments:
    - Distance: Very close (1.5m - 2.5m behind vehicle).
    - Angle: Perfectly aligned (0 deg).
    - Obstacles: 1 static (far), 1 dynamic (slow & far).
    """
    rng = random.Random(seed)
    config: Dict = deepcopy(DEFAULT_CONFIG)

    # 1. RNG Seed for the environment
    config["rng_seed"] = rng.randint(0, 1_000_000)

    field_size = float(config["field_size"])

    # 2. Spawn region: Small area around the center
    span = field_size * 0.10
    config["spawn_region"] = [
        -span / 2.0,
        span / 2.0,
        -span / 2.0,
        span / 2.0,
    ]

    # 3. Parking slot: "Face-to-face" / "Back-to-back" setup
    slot_cfg = config["parking_slot"].copy()
    # Offset X: -2.5m to -1.5m (very close behind the car)
    slot_cfg["offset_x_range"] = (-2.5, -1.5)
    # Offset Y: Small variation (-0.5m to 0.5m) to require slight steering
    slot_cfg["offset_y_range"] = (-0.5, 0.5)
    # Orientation: Perfectly aligned (0 deg)
    slot_cfg["orientation_range"] = (0.0, 0.0)
    config["parking_slot"] = slot_cfg

    # 4. Static obstacles: Keep 1, but far away
    static_cfg = config["static_obstacles"].copy()
    static_cfg["count"] = 1
    static_cfg["min_distance"] = 4.0  # Push them away from the goal
    static_cfg["seed"] = rng.randint(0, 10_000)
    config["static_obstacles"] = static_cfg

    # 5. Dynamic obstacles: Keep 1, but very slow and far
    dynamic_cfg = config["dynamic_obstacles"].copy()
    dynamic_cfg["count"] = 1
    dynamic_cfg["radius"] = round(rng.uniform(0.8, 1.2), 2)
    # Speed: 0.1 - 0.3 m/s (Turtle speed)
    dynamic_cfg["speed_range"] = (0.1, 0.3)
    dynamic_cfg["behavior"] = "random_walk" # Simple behavior
    dynamic_cfg["min_distance"] = 5.0       # Keep safe distance
    config["dynamic_obstacles"] = dynamic_cfg

    return config


def save_preview(config: Dict, img_path: Path) -> None:
    """Render the generated config to a PNG file (headless mode)."""
    print(f"Generating preview image at {img_path} ...")
    
    # Switch to Agg backend to avoid GUI windows popping up
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    # Monkey-patch plt.show and plt.pause to prevent blocking/errors
    _orig_show = plt.show
    _orig_pause = plt.pause
    plt.show = lambda *args, **kwargs: None
    plt.pause = lambda *args, **kwargs: None

    try:
        env = ParkingEnv(config=config)
        env.reset()  # Generate actual positions based on config ranges
        env.render() # Draw the map using matplotlib
        
        # Save figure
        if env.fig:
            env.fig.savefig(str(img_path), dpi=100, bbox_inches='tight')
        
        env.close()
    except Exception as e:
        print(f"Warning: Failed to save preview image: {e}")
    finally:
        # Restore original matplotlib state
        plt.show = _orig_show
        plt.pause = _orig_pause
        matplotlib.use(original_backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EASY ParkingEnv configs.")
    default_out = Path("parking_project_submission/configs/train_easy.json")
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
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable automatic preview image generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Generate configuration
    config = sample_easy_config(args.seed)
    
    # Write JSON
    write_json(args.out, config)
    print(f"Wrote EASY config to {args.out}")

    # Generate Preview Image (Default: ON)
    if not args.no_preview:
        img_path = args.out.with_suffix(".png")
        save_preview(config, img_path)


if __name__ == "__main__":
    main()
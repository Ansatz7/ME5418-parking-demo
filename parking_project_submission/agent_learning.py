"""Placeholder training entry point.

This script outlines the structure that will host the PPO/actor-critic
training routine in subsequent iterations. For now it validates arguments and
explains the expected workflow so evaluators understand the submission layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parking agent training stub")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON overrides for the training environment.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (placeholder value).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed applied to environment and network initialisation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print("Parking agent training is not yet implemented in this submission.")
    print("Configuration preview:")
    print(f"  config override: {args.config or 'default demo config'}")
    print(f"  episodes: {args.episodes}")
    print(f"  seed: {args.seed}")
    print("TODO: integrate PPO/actor-critic training loop in future iteration.")


if __name__ == "__main__":
    main()

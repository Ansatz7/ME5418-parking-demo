"""Single-entry demo script for the parking environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from parking_project_submission.modules import DemoOptions, run_demo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parking environment demo runner (manual/random/policy)."
    )
    parser.add_argument("--mode", choices=["random", "manual", "policy"], default="manual")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument(
        "--sleep-scale",
        type=float,
        default=0.0,
        help="Animation slowdown factor; leave at 0 for fastest headless runs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config path overriding the default demo settings.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable matplotlib rendering (useful for automated smoke tests).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step logging for less verbose console output.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Manual mode: suppress per-step logs and print one-line episode summary.",
    )
    parser.add_argument(
        "--per-step",
        action="store_true",
        help="Random mode: print per-step logs in addition to the final episode summary.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Policy mode: path to model checkpoint (defaults to artifacts/ppo_agent.pt)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Policy mode: use stochastic sampling instead of deterministic mean.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="Policy mode: save a video to this path (requires imageio)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    options = DemoOptions(
        mode=args.mode,
        episodes=args.episodes,
        max_steps=args.max_steps,
        sleep_scale=args.sleep_scale,
        config_path=args.config,
        visualize=not args.no_visualize,
        verbose=not args.quiet,
        summary=args.summary,
        per_step=args.per_step,
        policy_checkpoint=args.checkpoint,
        stochastic=args.stochastic,
        record_path=args.record,
    )
    run_demo(options)


if __name__ == "__main__":
    main()

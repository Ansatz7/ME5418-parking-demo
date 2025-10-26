"""Demonstration of the recurrent actor-critic network used in the project."""

from __future__ import annotations

import argparse
from typing import Sequence

import torch

from parking_project_submission.modules import RecurrentActorCritic
from parking_project_submission.parking_env import ParkingEnv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural network demo for ParkingEnv")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on (default cpu).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env = ParkingEnv()
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        model = RecurrentActorCritic(obs_dim, action_dim).to(args.device)
        hidden = model.initial_state().to(args.device)

        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
        obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # [T=1, B=1, obs_dim]

        output = model(obs_tensor, hidden)
        action_sample = output.action_dist.sample()
        log_prob = output.action_dist.log_prob(action_sample).sum()
        value = output.value

        # Simple loss = negative log-likelihood + value baseline
        loss = -log_prob + value.mean()
        loss.backward()

        print("=== Neural Network Demo ===")
        print(f"Observation dim: {obs_dim}, action dim: {action_dim}")
        print(f"Sampled action: {action_sample.detach().cpu().numpy()}")
        print(f"Value estimate: {value.detach().cpu().numpy()}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        print(f"Gradient norm after backward(): {grad_norm:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

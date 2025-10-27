"""Neural network demo supporting GRU or Lidar+LSTM backbones.

Two options for temporal core while sharing the same input split (base 11-dim
features + lidar N beams):

- --arch gru   -> CNN(lidar)+MLP(base) -> concat -> GRU -> policy/value
- --arch lstm  -> CNN(lidar)+MLP(base) -> concat -> LSTM -> policy/value

This script runs a single forward pass, samples an action, computes a simple
toy loss and backpropagates to verify gradients and shapes.
"""

from __future__ import annotations

import argparse
from typing import Sequence
from pathlib import Path

import torch

from parking_project_submission.modules import (
    RecurrentActorCriticLidar,
    RecurrentActorCriticLidarGRU,
)
from parking_project_submission.parking_env import ParkingEnv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural network demo for ParkingEnv")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on (default cpu).",
    )
    parser.add_argument(
        "--arch",
        choices=["gru", "lstm"],
        default="gru",
        help="Choose temporal core: GRU or LSTM (both share lidar+base encoders).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Sequence length T for the demo forward pass.",
    )
    parser.add_argument(
        "--export-onnx",
        type=Path,
        help="Optional ONNX export path; when provided, exports the chosen arch.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env = ParkingEnv()
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        base_dim = 11
        lidar_dim = obs_dim - base_dim

        if args.arch == "lstm":
            model = RecurrentActorCriticLidar(base_dim=base_dim, action_dim=action_dim).to(args.device)
            h, c = model.initial_state(batch_size=1, device=torch.device(args.device))
        else:
            model = RecurrentActorCriticLidarGRU(base_dim=base_dim, action_dim=action_dim).to(args.device)
            h = model.initial_state(batch_size=1, device=torch.device(args.device))

        # Build a dummy sequence of T observations from repeated reset states
        T = max(1, int(args.seq_len))
        obs_list = []
        for _ in range(T):
            ob, _ = env.reset()
            obs_list.append(torch.tensor(ob, dtype=torch.float32))
        flat_obs = torch.stack(obs_list, dim=0).unsqueeze(0).to(args.device)  # [B=1, T, obs_dim]
        base_obs = flat_obs[..., :base_dim]
        lidar_obs = flat_obs[..., base_dim:]

        if args.arch == "lstm":
            out = model(base_obs, lidar_obs, (h, c))
        else:
            out = model(base_obs, lidar_obs, h)

        # Sample last step action for logging
        action_sample = out.action_dist.sample()  # [B, T, A]
        last_action = action_sample[:, -1]
        last_value = out.value[:, -1]

        # Simple toy loss on the last step
        log_prob = out.action_dist.log_prob(action_sample).sum(dim=-1)  # [B, T]
        loss = -log_prob.mean() + out.value.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        print("=== Neural Network Demo ===")
        print(f"Arch: {args.arch} | base_dim={base_dim}, lidar_dim={lidar_dim}, action_dim={action_dim}")
        print(f"Sampled action (last step): {last_action.detach().cpu().numpy()}")
        print(f"Value estimate (last step): {last_value.detach().cpu().numpy()}")
        print(f"Gradient norm after backward(): {float(grad_norm):.4f}")

        # Optional ONNX export (pure tensor I/O for compatibility)
        if args.export_onnx is not None:
            out_path = args.export_onnx
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.arch == "lstm":
                class _LSTMExport(torch.nn.Module):
                    def __init__(self, inner: RecurrentActorCriticLidar) -> None:
                        super().__init__()
                        self.inner = inner

                    def forward(self, base_obs, lidar_obs, h, c):
                        out = self.inner(base_obs, lidar_obs, (h, c))
                        mean = out.action_dist.mean
                        h_next, c_next = out.next_state
                        return mean, out.value, h_next, c_next

                wrapper = _LSTMExport(model).to(args.device)
                torch.onnx.export(
                    wrapper,
                    (base_obs, lidar_obs, h, c),
                    str(out_path),
                    input_names=["base_obs", "lidar_obs", "h", "c"],
                    output_names=["action_mean", "value", "next_h", "next_c"],
                    opset_version=17,
                    dynamic_axes={"base_obs": {1: "T"}, "lidar_obs": {1: "T"}},
                )
            else:
                class _GRUExport(torch.nn.Module):
                    def __init__(self, inner: RecurrentActorCriticLidarGRU) -> None:
                        super().__init__()
                        self.inner = inner

                    def forward(self, base_obs, lidar_obs, h):
                        out = self.inner(base_obs, lidar_obs, h)
                        mean = out.action_dist.mean
                        return mean, out.value, out.next_state

                wrapper = _GRUExport(model).to(args.device)
                torch.onnx.export(
                    wrapper,
                    (base_obs, lidar_obs, h),
                    str(out_path),
                    input_names=["base_obs", "lidar_obs", "h"],
                    output_names=["action_mean", "value", "next_h"],
                    opset_version=17,
                    dynamic_axes={"base_obs": {1: "T"}, "lidar_obs": {1: "T"}},
                )
            print(f"Exported ONNX to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

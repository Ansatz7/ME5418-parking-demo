"""Inspect and visualize the project neural networks without training.

Usage examples:

  # Lidar+LSTM architecture: print summary, run a forward pass, export ONNX
  python -m parking_project_submission.neural_network_inspect \
    --arch lidar --export-onnx artifacts/model_lidar.onnx

  # GRU baseline: print summary and ONNX
  python -m parking_project_submission.neural_network_inspect \
    --arch gru --export-onnx artifacts/model_gru.onnx

This script has no extra dependencies; if `torchinfo` or `torchsummary` is
available it will print a richer layer-wise summary automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from parking_project_submission.parking_env import ParkingEnv
from parking_project_submission.modules import (
    RecurrentActorCritic,
    RecurrentActorCriticLidar,
)


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _maybe_layer_summary(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    # Try torchinfo or torchsummary if present; fall back to param count.
    try:
        from torchinfo import summary  # type: ignore

        summary(model, input_data=inputs, verbose=2, col_names=("input_size", "output_size", "num_params"))
        return
    except Exception:
        pass
    try:
        from torchsummary import summary as ts  # type: ignore

        # torchsummary expects (channels, H, W) style; not ideal for RNNs, so just print param count.
        raise RuntimeError  # force fallback below for consistent output
    except Exception:
        print(f"Model parameters: {_param_count(model):,}")


def build_and_inspect(arch: str, device: str, export_onnx: Optional[Path]) -> None:
    env = ParkingEnv()
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        if arch == "gru":
            model = RecurrentActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
            # Inputs: [T, B, obs_dim], hidden [1, B, H]
            T, B = 1, 1
            dummy_obs = torch.zeros(T, B, obs_dim, dtype=torch.float32, device=device)
            hidden = model.initial_state(batch_size=B).to(device)

            print("Architecture: GRU actor-critic (obs->MLP->GRU->heads)")
            _maybe_layer_summary(model, (dummy_obs, hidden))

            out = model(dummy_obs, hidden)
            print("Policy mean shape:", out.action_dist.mean.shape)
            print("Value shape:", out.value.shape)

            if export_onnx is not None:
                # Wrap to return pure tensors (ONNX cannot export custom dataclasses)
                class _GRUExport(torch.nn.Module):
                    def __init__(self, inner: RecurrentActorCritic) -> None:
                        super().__init__()
                        self.inner = inner

                    def forward(self, obs: torch.Tensor, h: torch.Tensor):
                        out = self.inner(obs, h)
                        mean = out.action_dist.mean
                        return mean, out.value, out.next_state

                wrapper = _GRUExport(model)
                export_onnx.parent.mkdir(parents=True, exist_ok=True)
                torch.onnx.export(
                    wrapper,
                    (dummy_obs, hidden),
                    str(export_onnx),
                    input_names=["obs", "h"],
                    output_names=["action_mean", "value", "next_h"],
                    opset_version=17,
                    dynamic_axes={"obs": {0: "T", 1: "B"}},
                )
                print(f"Exported ONNX to {export_onnx}")

        else:  # lidar
            base_dim = 11
            n_rays = obs_dim - base_dim
            model = RecurrentActorCriticLidar(base_dim=base_dim, action_dim=action_dim).to(device)
            # Inputs: [B, T, base_dim], [B, T, n_rays], (h,c)
            B, T = 1, 1
            base = torch.zeros(B, T, base_dim, dtype=torch.float32, device=device)
            lidar = torch.zeros(B, T, n_rays, dtype=torch.float32, device=device)
            h0, c0 = model.initial_state(batch_size=B, device=torch.device(device))

            print("Architecture: Lidar(CNN)+Base(MLP)->concat->LSTM->heads")
            _maybe_layer_summary(model, (base, lidar, (h0, c0)))

            out = model(base, lidar, (h0, c0))
            print("Policy mean shape:", out.action_dist.mean.shape)
            print("Value shape:", out.value.shape)

            if export_onnx is not None:
                # Wrap to present (h, c) as two separate tensor inputs/outputs
                class _LSTMExport(torch.nn.Module):
                    def __init__(self, inner: RecurrentActorCriticLidar) -> None:
                        super().__init__()
                        self.inner = inner

                    def forward(
                        self,
                        base_obs: torch.Tensor,
                        lidar_obs: torch.Tensor,
                        h: torch.Tensor,
                        c: torch.Tensor,
                    ):
                        out = self.inner(base_obs, lidar_obs, (h, c))
                        mean = out.action_dist.mean
                        h_next, c_next = out.next_state
                        return mean, out.value, h_next, c_next

                wrapper = _LSTMExport(model)
                export_onnx.parent.mkdir(parents=True, exist_ok=True)
                torch.onnx.export(
                    wrapper,
                    (base, lidar, h0, c0),
                    str(export_onnx),
                    input_names=["base_obs", "lidar_obs", "h", "c"],
                    output_names=["action_mean", "value", "next_h", "next_c"],
                    opset_version=17,
                    dynamic_axes={"base_obs": {1: "T"}, "lidar_obs": {1: "T"}},
                )
                print(f"Exported ONNX to {export_onnx}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect/visualize parking project neural networks")
    p.add_argument("--arch", choices=["gru", "lidar"], default="lidar", help="Network to inspect")
    p.add_argument("--device", default="cpu")
    p.add_argument("--export-onnx", type=Path, help="Optional ONNX export path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_and_inspect(args.arch, args.device, args.export_onnx)


if __name__ == "__main__":
    main()

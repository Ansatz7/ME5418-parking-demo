"""Neural network demo for the Lidar+LSTM backbone.

This architecture is now fixed:
- Lidar+LSTM -> CNN(lidar)+MLP(base) -> concat -> ResidualBlock -> LSTM -> policy/value

This script runs a single forward pass, samples an action, computes a simple
toy loss and backpropagates to verify gradients and shapes.
"""

from __future__ import annotations

import argparse
from typing import Sequence
from pathlib import Path

import torch

# --- 修改：只导入 LSTM 版本的网络 ---
from parking_project_submission.modules import RecurrentActorCriticLidar
from parking_project_submission.parking_env import ParkingEnv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural network demo for ParkingEnv (CPU-only)")
    # --- 删除：--arch 参数 ---
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Sequence length T for the demo forward pass.",
    )
    parser.add_argument(
        "--export-onnx",
        type=Path,
        help="Optional ONNX export path; when provided, exports the model.",
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

        # --- 修改：直接实例化 LSTM 模型，删除 GRU 的 if/else ---
        # CPU-only: always place tensors and modules on CPU for portability
        device = torch.device("cpu")
        model = RecurrentActorCriticLidar(base_dim=base_dim, action_dim=action_dim).to(device)
        h, c = model.initial_state(batch_size=1, device=device)

        # Build a dummy sequence of T observations from repeated reset states
        T = max(1, int(args.seq_len))
        obs_list = []
        for _ in range(T):
            ob, _ = env.reset()
            obs_list.append(torch.tensor(ob, dtype=torch.float32))
        flat_obs = torch.stack(obs_list, dim=0).unsqueeze(0)  # [B=1, T, obs_dim]
        
        # --- 修改：使用新的 forward_from_flat_obs 帮助函数 ---
        # (注: RecurrentActorCriticLidar 已经有了 forward_from_flat_obs)
        # 我们可以直接用它，或者像以前一样手动拆分
        
        # 为了清晰，我们保留手动拆分，并删除 if/else
        base_obs = flat_obs[..., :base_dim]
        lidar_obs = flat_obs[..., base_dim:]

        out = model(base_obs, lidar_obs, (h, c))
        # --- 修改结束 ---

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
        # --- 修改：硬编码 Arch ---
        print(f"Arch: Lidar+Residual+LSTM | base_dim={base_dim}, lidar_dim={lidar_dim}, action_dim={action_dim}")
        print(f"Sampled action (last step): {last_action.detach().cpu().numpy()}")
        print(f"Value estimate (last step): {last_value.detach().cpu().numpy()}")
        print(f"Gradient norm after backward(): {float(grad_norm):.4f}")

        # Optional ONNX export (pure tensor I/O for compatibility)
        if args.export_onnx is not None:
            out_path = args.export_onnx
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # --- 修改：删除 GRU 的 if/else，只保留 LSTM 导出逻辑 ---
            class _LSTMExport(torch.nn.Module):
                def __init__(self, inner: RecurrentActorCriticLidar) -> None:
                    super().__init__()
                    self.inner = inner

                def forward(self, base_obs, lidar_obs, h, c):
                    # (注: 导出时我们调用原始的 forward)
                    out = self.inner(base_obs, lidar_obs, (h, c))
                    mean = out.action_dist.mean
                    h_next, c_next = out.next_state
                    return mean, out.value, h_next, c_next

            wrapper = _LSTMExport(model)
            torch.onnx.export(
                wrapper,
                (base_obs, lidar_obs, h, c),
                str(out_path),
                input_names=["base_obs", "lidar_obs", "h", "c"],
                output_names=["action_mean", "value", "next_h", "next_c"],
                opset_version=17,
                dynamic_axes={"base_obs": {1: "T"}, "lidar_obs": {1: "T"}},
            )
            print(f"Exported ONNX to {out_path}")
            # --- 修改结束 ---
    finally:
        env.close()


if __name__ == "__main__":
    main()

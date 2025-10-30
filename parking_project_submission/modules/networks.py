"""Neural network building blocks used across submission demos.

Currently this module exposes a lightweight recurrent actor-critic skeleton
(Lidar+LSTM+ResidualBlock) that mirrors the architecture discussed in class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class ActorCriticOutputLSTM:
    """Container for LSTM-based actor-critic outputs."""

    action_dist: Normal
    value: torch.Tensor
    next_state: Tuple[torch.Tensor, torch.Tensor]


class RecurrentActorCriticLidar(nn.Module):
    """Two-branch recurrent actor-critic with CNN lidar encoder + MLP base encoder.

    This module follows the proposed architecture:
    - lidar distances (1 x N) -> 1D CNN + global average pooling -> 128-d embedding
    - base state (base_dim=11) -> MLP -> 128-d embedding
    - concat -> ResidualBlock(256d) -> LSTM (hidden_dim) -> policy/value heads

    By default it expects batch-first tensors in forward(); helper methods are
    provided to initialise hidden state and to split flat observations coming
    from ParkingEnv into base/lidar parts.
    """

    def __init__(
        self,
        *,
        base_dim: int = 11,
        action_dim: int = 2,
        hidden_dim: int = 128,
        lidar_channels: int = 128,
    ) -> None:
        super().__init__()
        self.base_dim = int(base_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        # Lidar encoder: [B*T, 1, N] -> [B*T, C, 1] -> [B*T, C]
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, lidar_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(lidar_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_channels, 128),
            nn.ReLU(inplace=True),
        )

        # Base-state encoder: [B*T, base_dim] -> 128
        self.base_fc = nn.Sequential(
            nn.Linear(self.base_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        # --- 新增：残差精炼块 (Residual Refinement Block) ---
        # 这个块将处理 128(base) + 128(lidar) = 256 维的融合特征
        self.fusion_refiner_F = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256), # 使用 LayerNorm 保持稳定
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )
        # 最后的激活函数，在 F(x) + x 之后使用
        self.fusion_refiner_activation = nn.ReLU(inplace=True)
        # --- 新增结束 ---

        # Temporal model
        self.lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_dim, batch_first=True)

        # Heads
        self.policy_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.policy_logstd = nn.Linear(self.hidden_dim, self.action_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised (h, c) for a single-layer LSTM."""

        dev = device if device is not None else next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=dev)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=dev)
        return h0, c0

    @staticmethod
    def split_flat_obs(obs: torch.Tensor, base_dim: int = 11, batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a flat observation vector into base and lidar components.

        Accepts shapes [B, obs_dim] or [B, T, obs_dim] (when batch_first=True).
        Returns tensors with shapes [B, T, base_dim] and [B, T, N_rays].
        """

        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [B, 1, obs_dim]
        if obs.dim() != 3:
            raise ValueError("Expected obs with shape [B, obs_dim] or [B, T, obs_dim].")
        base = obs[..., :base_dim]
        lidar = obs[..., base_dim:]
        return base, lidar

    def _encode(self, base_obs_bt: torch.Tensor, lidar_obs_bt: torch.Tensor) -> torch.Tensor:
        """Encode base and lidar branches; inputs are [B*T, ...]."""

        z_base = self.base_fc(base_obs_bt)  # [B*T, 128]
        x_lidar = self.lidar_conv(lidar_obs_bt)  # [B*T, C, 1]
        x_lidar = x_lidar.squeeze(-1)  # [B*T, C]
        z_lidar = self.lidar_fc(x_lidar)  # [B*T, 128]
        return torch.cat([z_base, z_lidar], dim=1)  # [B*T, 256]

    def forward(
        self,
        base_obs: torch.Tensor,
        lidar_obs: torch.Tensor,
        rnn_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> ActorCriticOutputLSTM:
        """Forward pass.

        Parameters
        ----------
        base_obs: [B, T, base_dim] or [B, base_dim]
        lidar_obs: [B, T, N_rays] or [B, N_rays]
        rnn_hidden: optional (h, c) where each is [1, B, hidden]
        """

        if base_obs.dim() == 2:
            base_obs = base_obs.unsqueeze(1)
        if lidar_obs.dim() == 2:
            lidar_obs = lidar_obs.unsqueeze(1)
        if base_obs.size(0) != lidar_obs.size(0) or base_obs.size(1) != lidar_obs.size(1):
            raise ValueError("base_obs and lidar_obs must share batch and time dimensions.")

        B, T, _ = base_obs.shape
        base_bt = base_obs.contiguous().view(B * T, -1)
        lidar_bt = lidar_obs.contiguous().view(B * T, 1, -1)

        z_bt = self._encode(base_bt, lidar_bt)  # [B*T, 256]

        # --- 修改：应用残差块 (Apply Residual Block) ---
        # x = z_bt (保底通道)
        # F(x) = self.fusion_refiner_F(z_bt) (精炼通道)
        # 输出 = ReLU( F(x) + x )
        z_refined_bt = self.fusion_refiner_activation( self.fusion_refiner_F(z_bt) + z_bt )
        # --- 修改结束 ---
        
        z = z_refined_bt.view(B, T, -1)  # [B, T, 256]

        if rnn_hidden is None:
            h0, c0 = self.initial_state(batch_size=B, device=z.device)
        else:
            h0, c0 = rnn_hidden

        lstm_out, next_state = self.lstm(z, (h0, c0))  # [B, T, H]
        mean = self.policy_mean(lstm_out)
        log_std = self.policy_logstd(lstm_out).clamp(min=-5.0, max=2.0)
        std = torch.exp(log_std)
        dist = Normal(loc=mean, scale=std)
        value = self.value_head(lstm_out).squeeze(-1)  # [B, T]

        return ActorCriticOutputLSTM(action_dist=dist, value=value, next_state=next_state)

    def forward_from_flat_obs(
        self,
        flat_obs: torch.Tensor,
        rnn_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> ActorCriticOutputLSTM:
        """Convenience wrapper to accept flat observations and split internally.

        `flat_obs` supports [B, obs_dim] or [B, T, obs_dim].
        """

        base_obs, lidar_obs = self.split_flat_obs(flat_obs, base_dim=self.base_dim, batch_first=True)
        return self.forward(base_obs, lidar_obs, rnn_hidden)


# --- 删除：RecurrentActorCriticLidarGRU 整个类 ---
# (已删除)
"""Neural network building blocks used across submission demos.

Currently this module exposes a lightweight recurrent actor-critic skeleton
that mirrors the architecture discussed in class. 训练脚本在后续批次会复用
这一实现，因此这里提供一个易于理解的 PyTorch 版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class ActorCriticOutput:
    """Container returned by :class:`RecurrentActorCritic` forward pass."""

    action_dist: Normal
    value: torch.Tensor
    next_state: torch.Tensor


class RecurrentActorCritic(nn.Module):
    """Minimal gated recurrent actor-critic network.

    Parameters
    ----------
    obs_dim:
        Flattened observation dimension coming from ``ParkingEnv``.
    action_dim:
        Size of the continuous action vector (2 in our environment).
    hidden_dim:
        Number of hidden units used for the GRU and the projection heads.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Log-std parameterised as a free vector (broadcast over time steps).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Return the zero-initialised recurrent state."""

        return torch.zeros(1, batch_size, self.gru.hidden_size)

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> ActorCriticOutput:
        """Run a forward pass.

        Parameters
        ----------
        obs:
            Tensor of shape ``[T, B, obs_dim]`` containing a sequence of
            observations (``T`` time steps, ``B`` batch size).
        state:
            Recurrent hidden state with shape ``[1, B, hidden_dim]`` produced by
            :meth:`initial_state` or a previous call.
        """

        encoded = self.obs_encoder(obs)
        gru_out, next_state = self.gru(encoded, state)
        features = gru_out[-1]

        mean = self.policy_head(features)
        std = torch.exp(self.log_std).expand_as(mean)
        action_dist = Normal(loc=mean, scale=std)
        value = self.value_head(features).squeeze(-1)

        return ActorCriticOutput(action_dist=action_dist, value=value, next_state=next_state)

"""Minimal PPO training entry (CPU‑only).

EN: This file implements a smallest-possible PPO training loop that treats
each time step independently (T=1) although the network contains an LSTM.
This helps verify the full learning pipeline end-to-end before upgrading to
true recurrent PPO with sequence buffers.

ZH: 本文件实现一个“最小可跑”的 PPO 训练循环。尽管网络里含有 LSTM，
这里先把时间步按 T=1 处理（每步相互独立，等价忽略记忆），目的是先让
训练流程端到端跑通，之后再升级到“真正的时序版” PPO（带序列缓存）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import optim

from parking_project_submission.parking_env import ParkingEnv
from parking_project_submission.modules import RecurrentActorCriticLidar


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """EN: Parse CLI arguments for minimal PPO training/eval.
    ZH: 解析用于最小 PPO 训练/评测的命令行参数。
    """

    parser = argparse.ArgumentParser(description="Parking agent minimal PPO (CPU-only)")

    # General / 通用
    parser.add_argument("--config", type=Path, help="Optional JSON env override file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env/model")

    # Train / 训练参数
    parser.add_argument("--total-steps", type=int, default=200_000, help="Total env steps")
    parser.add_argument("--rollout-len", type=int, default=2048, help="Steps per update")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coef")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coef")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Grad clip norm")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("artifacts/ppo_minimal.pt"),
        help="Model checkpoint path (will be created)",
    )

    # Eval / 评测参数
    parser.add_argument("--eval", action="store_true", help="Eval only, no training")
    parser.add_argument("--checkpoint", type=Path, help="Path to load for eval")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Eval episodes")

    return parser.parse_args(argv)


# -----------------------------
# Helper computations / 辅助计算
# -----------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    lam: float,
    last_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """EN: Compute GAE advantages and returns for a single rollout.
    ZH: 计算一次 rollout 的 GAE 优势与回报。
    """

    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    ret = values + adv
    return adv, ret


def ppo_update(
    model: RecurrentActorCriticLidar,
    optimizer: optim.Optimizer,
    obs_batch: np.ndarray,
    act_batch: np.ndarray,
    old_logp_batch: np.ndarray,
    adv_batch: np.ndarray,
    ret_batch: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
) -> None:
    """EN: One PPO update over several epochs of minibatches.
    ZH: 在若干个 epoch 上分批完成一次 PPO 参数更新。
    """

    device = next(model.parameters()).device
    N = obs_batch.shape[0]

    # Advantage normalization / 优势归一化
    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

    for _ in range(epochs):
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            j = idx[start : start + batch_size]
            ob = torch.tensor(obs_batch[j], dtype=torch.float32, device=device)
            act = torch.tensor(act_batch[j], dtype=torch.float32, device=device)
            old_logp = torch.tensor(old_logp_batch[j], dtype=torch.float32, device=device)
            adv = torch.tensor(adv_batch[j], dtype=torch.float32, device=device)
            ret = torch.tensor(ret_batch[j], dtype=torch.float32, device=device)

            # Forward with T=1 / 单步前向
            out = model.forward_from_flat_obs(ob)
            dist, value = out.action_dist, out.value[:, -1]
            logp = dist.log_prob(act.unsqueeze(1)).sum(-1).squeeze(1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (ret - value).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def train_minimal(args: argparse.Namespace) -> None:
    """EN: Minimal single-env PPO trainer (ignores LSTM memory; T=1).
    ZH: 最小单环境 PPO 训练器（忽略 LSTM 记忆；按 T=1 处理）。
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = ParkingEnv() if args.config is None else ParkingEnv(config=None)
    # Note: ParkingEnv merges defaults internally; external overrides can be
    # loaded via workflow helpers in a fuller trainer. For minimality we use
    # the default config here. / 最小实现直接使用默认配置。

    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device("cpu")
    model = RecurrentActorCriticLidar(base_dim=11, action_dim=act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    total_steps = int(args.total_steps)
    rollout_len = int(args.rollout_len)

    step_count = 0
    while step_count < total_steps:
        # Buffers for one rollout / 一次 rollout 的缓存
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        for _ in range(rollout_len):
            ob_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = model.forward_from_flat_obs(ob_t)  # T=1 forward
            dist, value = out.action_dist, out.value[:, -1]
            action = dist.sample()[0, -1].detach().numpy()

            logp = (
                dist.log_prob(torch.tensor(action, dtype=torch.float32).view(1, 1, -1))
                .sum(-1)[0, -1]
                .item()
            )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(value.item())
            done_buf.append(done)

            obs = next_obs
            step_count += 1
            if done:
                obs, _ = env.reset()
            if step_count >= total_steps:
                break

        # Bootstrap value for last step / 末步的引导价值
        if done_buf[-1]:
            last_value = 0.0
        else:
            last_value = (
                model.forward_from_flat_obs(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                .value[:, -1]
                .item()
            )

        adv, ret = compute_gae(
            np.asarray(rew_buf, dtype=np.float32),
            np.asarray(val_buf, dtype=np.float32),
            np.asarray(done_buf, dtype=np.bool_),
            gamma=float(args.gamma),
            lam=float(args.gae_lambda),
            last_value=float(last_value),
        )

        ppo_update(
            model,
            optimizer,
            np.asarray(obs_buf, dtype=np.float32),
            np.asarray(act_buf, dtype=np.float32),
            np.asarray(logp_buf, dtype=np.float32),
            adv,
            ret,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            clip_range=float(args.clip_range),
            vf_coef=float(args.vf_coef),
            ent_coef=float(args.ent_coef),
            max_grad_norm=float(args.max_grad_norm),
        )

        ep_finished = int(np.sum(done_buf))
        avg_reward = float(np.mean(rew_buf))
        print(
            f"[PPO] Update done: steps {len(obs_buf)} (episodes {ep_finished}), "
            f"total {step_count}, avg step reward {avg_reward:.3f}",
            flush=True,
        )

    # Save / 保存
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(args.save_path))
    print(f"Saved checkpoint to {args.save_path}")
    env.close()


def evaluate(args: argparse.Namespace) -> None:
    """EN: Deterministic evaluation using policy mean.
    ZH: 使用策略均值进行确定性评测。
    """

    ckpt = args.checkpoint or args.save_path
    if not ckpt or not ckpt.exists():
        raise FileNotFoundError("Checkpoint not found; pass --checkpoint or train first.")

    env = ParkingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = RecurrentActorCriticLidar(base_dim=11, action_dim=act_dim)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    returns = []
    for ep in range(int(args.eval_episodes)):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < env.config.get("max_steps", 4000):
            with torch.no_grad():
                out = model.forward_from_flat_obs(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = out.action_dist.mean[0, -1].numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
        print(f"Eval episode {ep+1}: return={total:.2f}, steps={steps}")
    print(f"Eval average return over {len(returns)} episodes: {np.mean(returns):.2f}")
    env.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.eval:
        evaluate(args)
    else:
        train_minimal(args)


if __name__ == "__main__":
    main()

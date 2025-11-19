"""Recurrent PPO trainer (CPU-only) for the ParkingEnv.

EN: This script upgrades the minimal PPO into a proper recurrent PPO (PPO-RNN)
that trains the LSTM memory. It collects rollouts while carrying the LSTM
hidden state across time, resets it on episode boundaries, and performs
backpropagation through time (BPTT) on contiguous sequence chunks.

ZH: 本脚本实现“时序版” PPO（PPO-RNN）。采样阶段在时间上携带 LSTM 的
隐状态；回合结束时重置隐状态；学习阶段按连续序列块做前向与 BPTT，
从而真正训练网络的记忆能力。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple, Dict

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter  # <--- 新增

from parking_project_submission.parking_env import ParkingEnv
from parking_project_submission.modules.workflow import resolve_config
from parking_project_submission.modules import RecurrentActorCriticLidar


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """EN: Parse CLI arguments for recurrent PPO training/eval.
    ZH: 解析用于时序 PPO 训练/评测的命令行参数。
    """

    p = argparse.ArgumentParser(description="Recurrent PPO (LSTM) for ParkingEnv (CPU-only)")
    # General / 通用
    p.add_argument("--seed", type=int, default=42, help="Random seed for env/model")
    p.add_argument("--config", type=Path, help="Optional JSON env override file")
    p.add_argument("--log-dir", type=Path, default=Path("runs"), help="TensorBoard log directory") # <--- 新增
    # Train / 训练
    p.add_argument("--total-steps", type=int, default=200_000, help="Total env steps")
    p.add_argument("--rollout-len", type=int, default=2048, help="Steps per update (T)")
    p.add_argument("--chunk-len", type=int, default=256, help="BPTT chunk length (<= T)")
    p.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    p.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coef")
    p.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coef")
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="Grad clip norm")
    p.add_argument(
        "--save-path",
        type=Path,
        default=Path("artifacts/ppo_agent.pt"),
        help="Model checkpoint path (will be created)",
    )
    # Eval / 评测
    p.add_argument("--eval", action="store_true", help="Eval only, no training")
    p.add_argument("--checkpoint", type=Path, help="Path to load for eval")
    p.add_argument("--eval-episodes", type=int, default=5, help="Eval episodes")
    return p.parse_args(argv)


# -----------------------------
# GAE with dones mask / 含回合掩码的 GAE
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


# -----------------------------
# PPO update on contiguous chunks / 连续序列块上的 PPO 更新
# -----------------------------

def ppo_update_recurrent(
    model: RecurrentActorCriticLidar,
    optimizer: optim.Optimizer,
    obs_seq: np.ndarray,  # [T, obs_dim]
    act_seq: np.ndarray,  # [T, A]
    old_logp_seq: np.ndarray,  # [T]
    adv_seq: np.ndarray,  # [T]
    ret_seq: np.ndarray,  # [T]
    h_seq: torch.Tensor,  # [T, 1, 1, H] hidden BEFORE step t
    c_seq: torch.Tensor,  # [T, 1, 1, H]
    *,
    epochs: int,
    chunk_len: int,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
) -> Dict[str, float]: # <--- 修改：返回 metrics
    """EN: Perform PPO updates by iterating over contiguous chunks.
    ZH: 按时间顺序、连续切片做多轮 PPO 更新，适配 LSTM 的 BPTT。
    """

    device = next(model.parameters()).device
    T = obs_seq.shape[0]

    # Advantage normalization across the whole rollout / 跨整个 rollout 做优势归一化
    adv_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

    def iter_chunks():
        start = 0
        while start < T:
            end = min(T, start + chunk_len)
            yield start, end
            start = end

    # Metrics trackers
    clip_losses = []
    value_losses = []
    entropy_losses = []
    total_losses = []

    for _ in range(epochs):
        # No shuffle! Preserve temporal order / 不打乱，保持时间顺序
        for s, e in iter_chunks():
            L = e - s  # chunk length
            # Tensorise chunk / 转为张量
            ob = torch.tensor(obs_seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)  # [1, L, obs]
            act = torch.tensor(act_seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)  # [1, L, A]
            old_logp = torch.tensor(old_logp_seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)  # [1, L]
            adv = torch.tensor(adv_seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)  # [1, L]
            ret = torch.tensor(ret_seq[s:e], dtype=torch.float32, device=device).unsqueeze(0)  # [1, L]

            # Initial hidden state for this chunk / 本片段起始隐状态
            h0 = h_seq[s].to(device)  # [1, 1, H]
            c0 = c_seq[s].to(device)  # [1, 1, H]

            out = model.forward_from_flat_obs(ob, (h0, c0))
            dist, value = out.action_dist, out.value  # [1, L, A], [1, L]
            logp = dist.log_prob(act).sum(-1)  # [1, L]
            entropy = dist.entropy().sum(-1).mean()  # scalar

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

            # Record metrics
            clip_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
            total_losses.append(loss.item())

    return {
        "policy_loss": np.mean(clip_losses),
        "value_loss": np.mean(value_losses),
        "entropy": np.mean(entropy_losses),
        "loss": np.mean(total_losses),
    }


def train_recurrent(args: argparse.Namespace) -> None:
    """EN: Single-env recurrent PPO trainer with BPTT.
    ZH: 单环境时序 PPO 训练器（带 BPTT）。
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(args.log_dir)) # <--- 初始化 Writer
    print(f"TensorBoard logging to: {args.log_dir}")
    print(f"Run 'tensorboard --logdir {args.log_dir}' to visualize.")

    # Build environment with optional overrides / 按需加载配置覆盖
    env_cfg = resolve_config(args.config)
    env = ParkingEnv(config=env_cfg)
    obs, _ = env.reset()
    act_dim = env.action_space.shape[0]

    device = torch.device("cpu")
    model = RecurrentActorCriticLidar(base_dim=11, action_dim=act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    # Optional resume from checkpoint / 可选：从 checkpoint 继续训练
    start_step = 0
    if args.checkpoint is not None and Path(args.checkpoint).exists():
        payload = torch.load(str(args.checkpoint), map_location="cpu")
        if isinstance(payload, dict) and "model_state" in payload:
            model.load_state_dict(payload["model_state"]) 
            opt_state = payload.get("optimizer_state")
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception:
                    pass
            start_step = int(payload.get("step_count", 0))
        else:
            # Backward compatibility: file is a raw state_dict
            model.load_state_dict(payload)

    total_steps = int(args.total_steps)
    rollout_len = int(args.rollout_len)
    chunk_len = max(1, int(args.chunk_len))

    step_count = int(start_step)
    while step_count < total_steps:
        # Buffers for a full rollout / 一次完整 rollout 的缓存
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        # Hidden state at each step BEFORE processing step t / 每个时间步开始时的隐状态
        h_list, c_list = [], []  # each element: [1, 1, H]

        # LSTM hidden state carried over time / 时间上传递的隐状态
        h, c = model.initial_state(batch_size=1, device=device)

        for t in range(rollout_len):
            # Record pre-step hidden for BPTT / 记录当前步的起始隐状态
            h_list.append(h.detach())
            c_list.append(c.detach())

            ob_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, obs]
            out = model.forward_from_flat_obs(ob_t, (h, c))  # T=1 forward with memory
            dist = out.action_dist
            value = out.value[:, -1]
            h, c = out.next_state  # carry memory to next step / 传递隐状态

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
                # Reset env and memory on episode end / 回合结束重置环境与隐状态
                obs, _ = env.reset()
                h, c = model.initial_state(batch_size=1, device=device)
            if step_count >= total_steps:
                break

        # Bootstrap last value if rollout does not end with done / 如未终止，用最后状态估值引导
        if done_buf[-1]:
            last_value = 0.0
        else:
            last_value = (
                model.forward_from_flat_obs(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), (h, c))
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

        # Convert hidden lists to tensors of shape [T, 1, 1, H]
        h_seq = torch.stack(h_list, dim=0)
        c_seq = torch.stack(c_list, dim=0)

        metrics = ppo_update_recurrent( # <--- 捕获 metrics
            model,
            optimizer,
            np.asarray(obs_buf, dtype=np.float32),
            np.asarray(act_buf, dtype=np.float32),
            np.asarray(logp_buf, dtype=np.float32),
            adv,
            ret,
            h_seq,
            c_seq,
            epochs=int(args.epochs),
            chunk_len=int(chunk_len),
            clip_range=float(args.clip_range),
            vf_coef=float(args.vf_coef),
            ent_coef=float(args.ent_coef),
            max_grad_norm=float(args.max_grad_norm),
        )

        ep_finished = int(np.sum(done_buf))
        avg_reward = float(np.mean(rew_buf))

        # --- Logging to TensorBoard --- <--- 新增日志记录
        writer.add_scalar("Rollout/AvgReward", avg_reward, step_count)
        writer.add_scalar("Rollout/EpisodesFinished", ep_finished, step_count)
        writer.add_scalar("Train/PolicyLoss", metrics["policy_loss"], step_count)
        writer.add_scalar("Train/ValueLoss", metrics["value_loss"], step_count)
        writer.add_scalar("Train/Entropy", metrics["entropy"], step_count)
        writer.add_scalar("Train/TotalLoss", metrics["loss"], step_count)
        # ------------------------------

        print(
            f"[PPO-RNN] Update: steps {len(obs_buf)} (eps {ep_finished}), total {step_count}, "
            f"rew {avg_reward:.3f}, loss {metrics['loss']:.3f}, ent {metrics['entropy']:.3f}",
            flush=True,
        )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save rich checkpoint for future resume / 保存包含优化器与步数的信息
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step_count": step_count,
        },
        str(args.save_path),
    )
    print(f"Saved recurrent checkpoint to {args.save_path}")
    writer.close() # <--- 关闭 Writer
    env.close()


def evaluate(args: argparse.Namespace) -> None:
    """EN: Deterministic evaluation using policy mean.
    ZH: 使用策略均值进行确定性评测。
    """

    ckpt = args.checkpoint or args.save_path
    if not ckpt or not ckpt.exists():
        raise FileNotFoundError("Checkpoint not found; pass --checkpoint or train first.")

    env_cfg = resolve_config(args.config)
    env = ParkingEnv(config=env_cfg)
    act_dim = env.action_space.shape[0]
    model = RecurrentActorCriticLidar(base_dim=11, action_dim=act_dim)
    payload = torch.load(str(ckpt), map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        model.load_state_dict(payload["model_state"])  # rich checkpoint
    else:
        model.load_state_dict(payload)  # raw state_dict
    model.eval()

    returns = []
    for ep in range(int(args.eval_episodes)):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        h, c = model.initial_state(batch_size=1, device=torch.device("cpu"))
        while not done and steps < env.config.get("max_steps", 4000):
            with torch.no_grad():
                out = model.forward_from_flat_obs(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), (h, c))
                action = out.action_dist.mean[0, -1].numpy()
                h, c = out.next_state
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
            if done:
                break
        returns.append(total)
        print(f"Eval episode {ep+1}: return={total:.2f}, steps={steps}")
    print(f"Eval average return over {len(returns)} episodes: {np.mean(returns):.2f}")
    env.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.eval:
        evaluate(args)
    else:
        train_recurrent(args)


if __name__ == "__main__":
    main()
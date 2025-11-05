"""Reusable workflows for CLI demos.

This module centralises helper routines shared by the submission scripts:

* building a demo-friendly environment configuration,
* loading optional JSON overrides,
* spinning up manual / random rollouts with consistent logging.
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from parking_project_submission.parking_env import DEFAULT_CONFIG, ParkingEnv
from .networks import RecurrentActorCriticLidar
import torch

from .utils import read_json


@dataclass
class DemoOptions:
    """Runtime options shared across demo entry points."""

    mode: str = "random"
    episodes: int = 1
    max_steps: int = 400
    sleep_scale: float = 0.0
    config_path: Optional[Path] = None
    visualize: bool = True
    verbose: bool = True
    summary: bool = False  # print one-line episode summary (manual mode)
    per_step: bool = False  # print per-step logs (random mode)
    policy_checkpoint: Optional[Path] = None  # path for policy mode
    stochastic: bool = False  # policy mode: sample() if True else mean
    record_path: Optional[Path] = None  # policy mode: save video to this path


def build_base_config() -> Dict[str, Any]:
    """Return a copy of the tuned defaults used by the demos."""

    return copy.deepcopy(DEFAULT_CONFIG)


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides into a deep copy of ``base``."""

    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load the baseline config and optionally merge a JSON override."""

    config = build_base_config()
    if config_path is None:
        return config

    try:
        overrides = read_json(config_path)
    except OSError as exc:  # pragma: no cover - user-facing I/O errors
        print(f"Failed to read config file {config_path}: {exc}", file=sys.stderr)
        return config
    return merge_config(config, overrides)


class ManualController:
    """Keyboard-driven controller matching the original demo behaviour."""

    def __init__(self, vehicle_cfg: Optional[Dict[str, Any]] = None) -> None:
        vehicle_cfg = vehicle_cfg or {}
        self._forward_delta = float(vehicle_cfg.get("manual_forward_accel", 1.5))
        self._reverse_delta = float(vehicle_cfg.get("manual_reverse_accel", 2.0))
        self._steering_delta = float(vehicle_cfg.get("manual_steering_accel", 1.0))
        self.running = True
        self._key_state: Dict[str, bool] = {}
        self.longitudinal = 0.0
        self.steering = 0.0

    def attach(self, env: ParkingEnv) -> None:
        if env.fig is None:
            env.render()
        canvas = env.fig.canvas
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.mpl_connect("key_release_event", self._on_key_release)

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if not key:
            return
        self._key_state[key] = True
        if key == "escape":
            self.running = False

    def _on_key_release(self, event) -> None:
        key = (event.key or "").lower()
        if not key:
            return
        self._key_state[key] = False

    def action(self) -> np.ndarray:
        lon = 0.0
        steer = 0.0
        if self._key_state.get("up"):
            lon += self._forward_delta
        if self._key_state.get("down"):
            lon -= self._reverse_delta
        if self._key_state.get("left"):
            steer += self._steering_delta
        if self._key_state.get("right"):
            steer -= self._steering_delta

        self.longitudinal = float(np.clip(lon, -2.0, 2.0))
        self.steering = float(np.clip(steer, -1.5, 1.5))
        return np.array([self.longitudinal, self.steering], dtype=np.float32)


def run_random_demo(options: DemoOptions) -> None:
    """Execute a random-policy rollout with logging."""

    env = ParkingEnv(config=resolve_config(options.config_path))
    try:
        for episode in range(options.episodes):
            obs, info = env.reset()
            total_reward = 0.0
            if options.visualize:
                env.render()
            for step in range(options.max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if options.visualize:
                    env.render()
                    if options.sleep_scale > 0.0:
                        time.sleep(env.dt * options.sleep_scale)
                if options.per_step and options.verbose:
                    print(
                        f"[Random] Ep {episode + 1} Step {step + 1} "
                        f"Reward {reward:.3f} Term {info['terminal_reason']} "
                        f"Dist {info['distance_to_slot']:.2f} "
                        f"Head {np.degrees(info['heading_error']):.1f} deg",
                        flush=True,
                    )
                if terminated or truncated:
                    break
            if options.verbose:
                print(
                    f"[Random Policy] Episode {episode + 1} finished in {step + 1} steps "
                    f"total reward {total_reward:.2f} termination {info['terminal_reason']}",
                    flush=True,
                )
    finally:
        env.close()


def run_manual_demo(options: DemoOptions) -> None:
    """Launch the interactive manual driving demo."""

    env = ParkingEnv(config=resolve_config(options.config_path))
    controller = ManualController(env.vehicle_cfg)
    obs, info = env.reset()
    env.render()
    if env.fig is not None:
        env.fig.set_size_inches(12, 6, forward=True)
    controller.attach(env)

    try:
        for episode in range(options.episodes):
            if episode > 0:
                obs, info = env.reset()
                env.render()
            step = 0
            while controller.running and step < options.max_steps:
                action = controller.action()
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if options.verbose and not options.summary:
                    print(
                        f"Episode {episode + 1} Step {step + 1} "
                        f"Reward {reward:.3f} Termination {info['terminal_reason']} "
                        f"Distance {info['distance_to_slot']:.2f} "
                        f"Heading {np.degrees(info['heading_error']):.1f} deg",
                        flush=True,
                    )
                step += 1
                if options.sleep_scale > 0.0:
                    time.sleep(env.dt * options.sleep_scale)
                if terminated or truncated:
                    break
            if not controller.running:
                break
            if options.summary and options.verbose:
                print(
                    f"[Manual] Episode {episode + 1} finished in {step} steps "
                    f"termination {info['terminal_reason']} dist {info['distance_to_slot']:.2f} "
                    f"heading {np.degrees(info['heading_error']):.1f} deg",
                    flush=True,
                )
    finally:
        env.close()


def run_demo(options: DemoOptions) -> None:
    """Dispatch to the requested demo mode."""

    if options.mode == "manual":
        run_manual_demo(options)
    elif options.mode == "policy":
        run_policy_demo(options)
    else:
        run_random_demo(options)


def run_policy_demo(options: DemoOptions) -> None:
    """Run a deterministic policy rollout using a saved checkpoint.

    Loads `RecurrentActorCriticLidar` and uses the mean action at each step.
    与随机/手动模式一致，支持渲染与逐步/摘要日志。
    """

    ckpt = options.policy_checkpoint
    if ckpt is None:
        ckpt = Path("artifacts/ppo_agent.pt")

    env = ParkingEnv(config=resolve_config(options.config_path))
    try:
        obs, info = env.reset()
        if options.visualize:
            env.render()

        act_dim = env.action_space.shape[0]
        model = RecurrentActorCriticLidar(base_dim=11, action_dim=act_dim)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()

        # Optional video writer / 可选视频写入
        writer = None
        if options.record_path is not None:
            try:
                import imageio
                options.record_path.parent.mkdir(parents=True, exist_ok=True)
                fps = max(1, int(round(1.0 / float(env.dt))))
                writer = imageio.get_writer(str(options.record_path), fps=fps)
                print(f"[Policy] Recording to {options.record_path} at {fps} FPS")
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"Failed to init video writer: {exc}. Proceeding without recording.")
                writer = None

        for episode in range(options.episodes):
            if episode > 0:
                obs, info = env.reset()
                if options.visualize:
                    env.render()
            # Initialize LSTM memory per-episode / 每个回合初始化 LSTM 记忆
            h, c = model.initial_state(batch_size=1, device=torch.device("cpu"))
            total_reward = 0.0
            for step in range(options.max_steps):
                with torch.no_grad():
                    out = model.forward_from_flat_obs(
                        torch.tensor(obs, dtype=torch.float32).unsqueeze(0), (h, c)
                    )
                    if options.stochastic:
                        action = out.action_dist.sample()[0, -1].numpy()
                    else:
                        action = out.action_dist.mean[0, -1].numpy()
                    # Carry memory / 传递记忆
                    h, c = out.next_state
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if options.visualize:
                    env.render()
                    if options.sleep_scale > 0.0:
                        time.sleep(env.dt * options.sleep_scale)
                # Grab frame if recording / 若录制则抓取画面帧
                if writer is not None and env.fig is not None:
                    try:
                        env.fig.canvas.draw()
                        w_px, h_px = env.fig.canvas.get_width_height()
                        frame = np.frombuffer(
                            env.fig.canvas.tostring_rgb(), dtype=np.uint8
                        ).reshape(h_px, w_px, 3)
                        writer.append_data(frame)
                    except Exception:
                        pass
                if options.per_step and options.verbose:
                    print(
                        f"[Policy] Ep {episode + 1} Step {step + 1} "
                        f"Reward {reward:.3f} Term {info['terminal_reason']} "
                        f"Dist {info['distance_to_slot']:.2f} "
                        f"Head {np.degrees(info['heading_error']):.1f} deg",
                        flush=True,
                    )
                if terminated or truncated:
                    # Reset memory together with environment / 回合结束一并重置记忆
                    h, c = model.initial_state(batch_size=1, device=torch.device("cpu"))
                    break
            if options.verbose:
                print(
                    f"[Policy] Episode {episode + 1} finished in {step + 1} steps "
                    f"total reward {total_reward:.2f} termination {info['terminal_reason']}",
                    flush=True,
                )
    finally:
        try:
            if 'writer' in locals() and writer is not None:
                writer.close()
        except Exception:
            pass
        env.close()

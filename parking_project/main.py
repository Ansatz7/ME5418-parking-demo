"""Command-line entry points for ParkingEnv demos.

Provides manual (keyboard-driven) and random policy loops using a shared
configuration pipeline. 方便在命令行运行停车环境的手动和随机演示，复用了同一套配置逻辑。
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from parking_gym import DEFAULT_CONFIG, ParkingEnv


# ---------------------------------------------------------------------------
# Configuration helpers 配置辅助函数
# ---------------------------------------------------------------------------
def build_config(overrides: Optional[Dict] = None) -> Dict:
    """Return the default degree-based demo config, optionally merged with overrides.

    构造基准配置（角度均为度），并与外部覆盖项合并。
    """
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "rng_seed": 42,
            "ray_angles": [
                -135.0,
                -90.0,
                -60.0,
                -30.0,
                30.0,
                60.0,
                90.0,
                135.0,
            ],
        }
    )
    config["static_obstacles"] = config["static_obstacles"].copy()
    config["dynamic_obstacles"] = config["dynamic_obstacles"].copy()
    config["static_obstacles"].update({"count": 2})
    config["dynamic_obstacles"].update({"count": 1, "behavior": "goal_driven"})
    if overrides:
        config = merge_config(config, overrides)
    return config


def merge_config(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge two dictionaries without mutating the inputs.

    递归合并两个字典，且不修改原始数据。
    """
    merged = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = value.copy()
        else:
            merged[key] = value

    for key, value in overrides.items():
        if key not in merged or not isinstance(value, dict):
            merged[key] = value
        else:
            merged[key] = merge_config(merged[key], value)
    return merged


def load_config(path: Path) -> Dict:
    """Read a JSON config file into a dictionary.

    从 JSON 文件读取配置并返回字典。
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a JSON object at the top level.")
        return data


# ---------------------------------------------------------------------------
# Manual controller definition 手动控制器定义
# ---------------------------------------------------------------------------
class ManualController:
    def __init__(self, vehicle_cfg: Optional[Dict] = None) -> None:
        """Keyboard-driven controller using config-defined acceleration deltas.

        基于键盘输入的控制器，所用的加速度增量可从车辆配置中读取。
        """
        self.longitudinal = 0.0
        self.steering = 0.0
        self.running = True
        self._key_state: Dict[str, bool] = {}
        cfg = vehicle_cfg or {}
        self._forward_delta = float(cfg.get("manual_forward_accel", 1.5))
        self._reverse_delta = float(cfg.get("manual_reverse_accel", 2.0))
        self._steering_delta = float(cfg.get("manual_steering_accel", 1.0))

    def attach(self, env: ParkingEnv) -> None:
        if env.fig is None:
            env.render()
        canvas = env.fig.canvas
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.mpl_connect("key_release_event", self._on_key_release)

    def _on_key_press(self, event) -> None:
        if event.key is None:
            return
        key = event.key.lower()
        self._key_state[key] = True
        if key == "escape":
            self.running = False

    def _on_key_release(self, event) -> None:
        if event.key is None:
            return
        key = event.key.lower()
        self._key_state[key] = False

    def action(self) -> np.ndarray:
        """Convert current keyboard state to a continuous action vector.

        将当前按键状态转换为连续动作向量 [lon_accel, steer_accel]。
        """
        lon = 0.0
        steer = 0.0
        if self._key_state.get("up", False):
            lon += self._forward_delta
        if self._key_state.get("down", False):
            lon -= self._reverse_delta
        if self._key_state.get("left", False):
            steer += self._steering_delta
        if self._key_state.get("right", False):
            steer -= self._steering_delta
        self.longitudinal = np.clip(lon, -2.0, 2.0)
        self.steering = np.clip(steer, -1.5, 1.5)
        return np.array([self.longitudinal, self.steering], dtype=np.float32)


# ---------------------------------------------------------------------------
# Demo loops 演示主循环
# ---------------------------------------------------------------------------
def manual_demo(
    episodes: int,
    max_steps: int,
    verbose: bool = True,
    sleep_scale: float = 0.5,
    *,
    config: Optional[Dict] = None,
) -> None:
    """Run the interactive manual-driving loop with keyboard control.

    启动带键盘控制的手动演示循环，可实时渲染车辆状态。
    """
    env = ParkingEnv(config=config or build_config())
    # ManualController keeps per-key accelerations mirrored from vehicle cfg
    # 手动控制器读取车辆配置中的按键加速度增量，保持交互一致。
    controller = ManualController(env.vehicle_cfg)
    obs, info = env.reset()
    env.render()
    if env.fig is not None:
        env.fig.set_size_inches(12, 6, forward=True)
    controller.attach(env)

    for ep in range(episodes):
        if ep > 0:
            obs, info = env.reset()
            env.render()
            if env.fig is not None:
                env.fig.set_size_inches(12, 6, forward=True)
        step = 0
        while controller.running and step < max_steps:
            action = controller.action()  # 人类键盘输入 -> 连续动作 / Human input to continuous action
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if verbose:
                print(
                    f"Episode {ep + 1} Step {step + 1} "
                    f"Reward {reward:.3f} Termination {info['terminal_reason']} "
                    f"Distance {info['distance_to_slot']:.2f} "
                    f"Heading {np.degrees(info['heading_error']):.1f} deg",
                    flush=True,
                )
            step += 1
            if sleep_scale > 0:
                time.sleep(env.dt * sleep_scale)
            if terminated or truncated:
                break
        if not controller.running:
            break
    env.close()


def random_policy_demo(
    episodes: int,
    max_steps: int,
    *,
    visualize: bool = True,
    sleep_scale: float = 0.5,
    config: Optional[Dict] = None,
) -> None:
    """Play episodes with random actions for smoke-testing the environment.

    使用随机策略运行若干回合，便于快速验收环境行为。
    """
    env = ParkingEnv(config=config or build_config())
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            total_reward = 0.0
            if visualize:
                env.render()
                if env.fig is not None:
                    env.fig.set_size_inches(12, 6, forward=True)
            for step in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if visualize:
                    env.render()
                    if sleep_scale > 0:
                        time.sleep(env.dt * sleep_scale)
                if terminated or truncated:
                    break
            print(
                f"[Random Policy] Episode {ep + 1} finished in {step + 1} steps "
                f"Total reward {total_reward:.2f} Termination {info['terminal_reason']}",
                flush=True,
            )
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo launcher.

    解析演示脚本的命令行参数（模式、轮次、配置文件等）。
    """
    parser = argparse.ArgumentParser(description="ParkingEnv demo runner.")
    # mode, episodes, max-steps define the high-level roll-out settings
    # 运行模式、轮次数量、最大步数决定演示循环的宏观参数。
    parser.add_argument("--mode", choices=["manual", "random"], default="random")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--sleep-scale",
        type=float,
        default=0.5,
        help="Scale factor for animation speed (applies to both manual and random runs).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON file with ParkingEnv configuration overrides.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point that resolves config overrides and dispatches demos.

    主入口：合并配置覆盖项后执行手动或随机演示。
    """
    args = parse_args()
    # Start from tuned defaults, then merge any JSON overrides supplied by user
    # 以调好的默认配置为起点，再合并用户提供的 JSON 覆盖值。
    config = build_config()
    if args.config is not None:
        try:
            overrides = load_config(args.config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Failed to load config from {args.config}: {exc}", file=sys.stderr)
            sys.exit(1)
        config = merge_config(config, overrides)
    try:
        if args.mode == "manual":
            manual_demo(
                args.episodes,
                args.max_steps,
                sleep_scale=args.sleep_scale,
                config=config,
            )
        else:
            random_policy_demo(
                args.episodes,
                args.max_steps,
                sleep_scale=args.sleep_scale,
                config=config,
            )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()

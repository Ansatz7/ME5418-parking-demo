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
    else:
        run_random_demo(options)

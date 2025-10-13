"""Interactive steering assist tuner launched via Qt window.

This script reproduces the steering-return simulator that previously lived in the
notebook. It loads vehicle parameters (either from a JSON config or the default
environment config), opens a Qt-backed Matplotlib window with sliders for Kp,
Kd, deadband, and initial state, and (optionally) writes updated values back to
the JSON file while you tune.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, CheckButtons
except Exception as exc:  # pragma: no cover - matplotlib import errors surface quickly
    raise SystemExit(f"Failed to import matplotlib: {exc}")

from parking_gym import DEFAULT_CONFIG


def compute_steering_history(
    kp: float,
    kd: float,
    deadband: float,
    angle0_deg: float,
    rate0_deg: float,
    commanded_accel_deg: float,
    steps: int,
    dt: float,
) -> Dict[str, np.ndarray]:
    """Integrate the steering assist dynamics and return angle/rate traces."""
    angle = np.radians(angle0_deg)
    rate = np.radians(rate0_deg)
    cmd_accel = np.radians(commanded_accel_deg)
    deadband_rad = np.radians(deadband)

    angle_hist = [np.degrees(angle)]
    rate_hist = [np.degrees(rate)]
    delta_hist = [0.0]

    for _ in range(steps):
        assist = 0.0
        if abs(cmd_accel) <= deadband_rad:
            assist = kp * angle + kd * rate
        rate += (cmd_accel - assist) * dt
        angle += rate * dt
        angle_hist.append(np.degrees(angle))
        rate_hist.append(np.degrees(rate))
        delta_hist.append(np.degrees(rate * dt))

    time = np.arange(len(angle_hist)) * dt
    return {
        "time": time,
        "angle_deg": np.array(angle_hist),
        "rate_deg_s": np.array(rate_hist),
        "delta_angle_deg": np.array(delta_hist),
    }

def load_vehicle_config(config_path: Path | None) -> Dict[str, Any]:
    """Load a vehicle configuration dict from JSON or defaults."""
    if config_path is None:
        # Use a shallow copy to avoid mutating DEFAULT_CONFIG in-place.
        return json.loads(json.dumps(DEFAULT_CONFIG))["vehicle"]

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"Config file '{config_path}' not found.")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config file '{config_path}' is not valid JSON: {exc}.")

    if "vehicle" not in cfg:
        raise SystemExit("Config file does not contain a 'vehicle' section.")
    return cfg["vehicle"]


def persist_vehicle_config(
    config_path: Path,
    vehicle_cfg: Dict[str, Any],
    indent: int = 2,
) -> None:
    """Write updated vehicle config back into the JSON file, preserving other keys."""
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            root_cfg = json.load(fh)
    except FileNotFoundError:
        root_cfg = {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config file '{config_path}' became invalid JSON: {exc}.")

    root_cfg.setdefault("vehicle", {})
    root_cfg["vehicle"].update(vehicle_cfg)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(root_cfg, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")


def launch_tuner(args: argparse.Namespace) -> None:
    """Set up the Qt window and sliders, wiring callbacks for updates/sync."""
    try:
        matplotlib.use("QtAgg")
    except Exception as exc:
        raise SystemExit(
            "Failed to activate QtAgg backend. Ensure PyQt/PySide is installed in the "
            f"'{args.conda_env}' environment.\n{exc}"
        )

    config_path = Path(args.config).resolve() if args.config else None
    vehicle_cfg = load_vehicle_config(config_path)

    kp_init = float(vehicle_cfg.get("steering_damping", 0.9))
    kd_init = float(vehicle_cfg.get("steering_rate_damping", 0.0))
    deadband_init = float(vehicle_cfg.get("steering_assist_deadband", 0.0))

    history = compute_steering_history(
        kp_init,
        kd_init,
        deadband_init,
        angle0_deg=args.angle0,
        rate0_deg=args.rate0,
        commanded_accel_deg=0.0,
        steps=args.steps,
        dt=DEFAULT_CONFIG["dt"],
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.subplots_adjust(bottom=0.32, hspace=0.2)
    try:
        fig.canvas.manager.set_window_title("Steering Assist Tuner")
    except Exception:
        pass

    angle_line, = axes[0].plot(history["time"], history["angle_deg"], label="steering angle (deg)")
    axes[0].axhline(0.0, color="k", linewidth=0.8)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    rate_line, = axes[1].plot(history["time"], history["rate_deg_s"], label="steering rate (deg/s)")
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[1].set_ylabel("Rate (deg/s)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    delta_line, = axes[2].step(
        history["time"], history["delta_angle_deg"], where="post", label="Δangle per step (deg)"
    )
    axes[2].axhline(0.0, color="k", linewidth=0.8)
    axes[2].set_ylabel("Δangle (deg)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.4)

    axes[2].set_xlabel("Time (s)")

    slider_color = "#f0f0f0"
    kp_ax = fig.add_axes([0.12, 0.24, 0.68, 0.03], facecolor=slider_color)
    kd_ax = fig.add_axes([0.12, 0.19, 0.68, 0.03], facecolor=slider_color)
    deadband_ax = fig.add_axes([0.12, 0.14, 0.68, 0.03], facecolor=slider_color)
    angle0_ax = fig.add_axes([0.12, 0.09, 0.68, 0.03], facecolor=slider_color)
    rate0_ax = fig.add_axes([0.12, 0.04, 0.68, 0.03], facecolor=slider_color)

    kp_slider = Slider(kp_ax, "Kp", 0.1, 100.0, valinit=kp_init, valstep=0.05)
    kd_slider = Slider(kd_ax, "Kd", 0.0, 10.0, valinit=kd_init, valstep=0.05)
    deadband_slider = Slider(deadband_ax, "deadband", 0.0, 1.0, valinit=deadband_init, valstep=0.01)
    angle0_slider = Slider(angle0_ax, "angle0", 0.0, 45.0, valinit=args.angle0, valstep=1.0)
    rate0_slider = Slider(rate0_ax, "rate0", -10.0, 10.0, valinit=args.rate0, valstep=0.5)

    check_ax = fig.add_axes([0.82, 0.58, 0.14, 0.1])
    check = CheckButtons(check_ax, ["sync to JSON"], [args.sync])
    for text in check.labels:
        text.set_fontsize(9)

    def update(_event: Any) -> None:
        nonlocal history
        history = compute_steering_history(
            kp_slider.val,
            kd_slider.val,
            deadband_slider.val,
            angle0_slider.val,
            rate0_slider.val,
            commanded_accel_deg=0.0,
            steps=args.steps,
            dt=DEFAULT_CONFIG["dt"],
        )
        angle_line.set_data(history["time"], history["angle_deg"])
        rate_line.set_data(history["time"], history["rate_deg_s"])
        delta_line.set_data(history["time"], history["delta_angle_deg"])
        for axis in axes:
            axis.relim()
            axis.autoscale_view()
        fig.canvas.draw_idle()

        if check.get_status()[0] and config_path is not None:
            vehicle_cfg["steering_damping"] = float(kp_slider.val)
            vehicle_cfg["steering_rate_damping"] = float(kd_slider.val)
            vehicle_cfg["steering_assist_deadband"] = float(deadband_slider.val)
            persist_vehicle_config(config_path, vehicle_cfg)

    kp_slider.on_changed(update)
    kd_slider.on_changed(update)
    deadband_slider.on_changed(update)
    angle0_slider.on_changed(update)
    rate0_slider.on_changed(update)

    plt.show()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steering assist tuner GUI")
    parser.add_argument("--config", type=str, help="Path to JSON config to load/update.")
    parser.add_argument("--angle0", type=float, default=20.0, help="Initial steering angle in degrees.")
    parser.add_argument("--rate0", type=float, default=0.0, help="Initial steering rate in deg/s.")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="When enabled, slider updates are written back to the JSON config file.",
    )
    parser.add_argument(
        "--conda-env",
        default="parking-rl",
        help="Name of the conda env (used for error messaging only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    launch_tuner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

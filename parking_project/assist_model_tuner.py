"""Interactive assist-model tuner launched via Qt window.

This script extends the original steering-return simulator by also modeling
longitudinal damping (how quickly the vehicle coasts to a stop once throttle is
released). Sliders let you adjust the steering and velocity assist gains,
deadbands, and initial conditions; when *sync* is enabled the updated
parameters are written back to the JSON config.
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


# ---------------------------------------------------------------------------
# Simulation helpers 仿真辅助函数
# ---------------------------------------------------------------------------
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
    """Integrate steering-assist dynamics and return angle/rate traces.

    积分方向盘回正模型，返回角度与角速度的时间序列。
    """
    angle = np.radians(angle0_deg)
    rate = np.radians(rate0_deg)
    cmd_accel = np.radians(commanded_accel_deg)
    deadband_rad = np.radians(deadband)

    angle_hist = [np.degrees(angle)]
    rate_hist = [np.degrees(rate)]

    for _ in range(steps):
        assist = 0.0
        if abs(cmd_accel) <= deadband_rad:
            assist = kp * angle + kd * rate
        rate += (cmd_accel - assist) * dt
        angle += rate * dt
        angle_hist.append(np.degrees(angle))
        rate_hist.append(np.degrees(rate))

    time = np.arange(len(angle_hist)) * dt
    return {
        "time": time,
        "angle_deg": np.array(angle_hist),
        "rate_deg_s": np.array(rate_hist),
    }


def compute_velocity_history(
    damping: float,
    deadband: float,
    velocity0: float,
    steps: int,
    dt: float,
) -> Dict[str, np.ndarray]:
    """Simulate longitudinal damping when throttle command is released.

    模拟松油门后的纵向阻尼过程，输出速度/加速度轨迹。
    """

    velocity = float(velocity0)
    vel_hist = [velocity]
    accel_hist = []
    assist_hist = []

    for _ in range(steps):
        assist = damping * velocity if deadband >= 0.0 else 0.0
        accel = -assist

        velocity += accel * dt

        assist_hist.append(assist)
        accel_hist.append(accel)
        vel_hist.append(velocity)

    time = np.arange(len(vel_hist)) * dt
    accel_hist = np.array(accel_hist, dtype=float)
    assist_hist = np.array(assist_hist, dtype=float)

    accel_hist = np.append(accel_hist, accel_hist[-1] if accel_hist.size else 0.0)
    assist_hist = np.append(assist_hist, assist_hist[-1] if assist_hist.size else 0.0)

    return {
        "time": time,
        "velocity": np.array(vel_hist),
        "accel": accel_hist,
        "assist": assist_hist,
    }


# ---------------------------------------------------------------------------
# Configuration I/O 配置读写
# ---------------------------------------------------------------------------
def load_vehicle_config(config_path: Path | None) -> Dict[str, Any]:
    """Load a vehicle configuration dict from JSON or defaults.

    从给定 JSON 读取车辆配置，若缺省则返回内置默认值。
    """
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
    """Write updated vehicle config back into the JSON file, preserving other keys.

    将更新后的车辆配置写回 JSON，保持其余键不变。
    """
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
    """Set up the Qt window and sliders, wiring callbacks for updates/sync.

    初始化 Qt 窗口与滑块控件，并注册实时更新与同步回调。
    """
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
    vel_damping_init = float(vehicle_cfg.get("velocity_damping", 0.8))
    vel_deadband_init = float(vehicle_cfg.get("velocity_deadband", 0.05))

    steering_history = compute_steering_history(
        kp_init,
        kd_init,
        deadband_init,
        angle0_deg=args.angle0,
        rate0_deg=args.rate0,
        commanded_accel_deg=0.0,
        steps=args.steps,
        dt=DEFAULT_CONFIG["dt"],
    )

    velocity_history = compute_velocity_history(
        vel_damping_init,
        vel_deadband_init,
        velocity0=args.vel0,
        steps=args.steps,
        dt=DEFAULT_CONFIG["dt"],
    )

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        3,
        2,
        left=0.07,
        right=0.95,
        bottom=0.32,
        top=0.95,
        wspace=0.3,
        hspace=0.28,
    )
    steering_axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    vel_ax = fig.add_subplot(gs[0, 1])
    accel_ax = fig.add_subplot(gs[1, 1], sharex=vel_ax)
    steering_summary_ax = fig.add_subplot(gs[2, 0])
    velocity_summary_ax = fig.add_subplot(gs[2, 1])
    steering_summary_ax.axis("off")
    velocity_summary_ax.axis("off")

    try:
        fig.canvas.manager.set_window_title("Assist Model Tuner")
    except Exception:
        pass

    angle_line, = steering_axes[0].plot(
        steering_history["time"], steering_history["angle_deg"], label="steering angle (deg)"
    )
    steering_axes[0].axhline(0.0, color="k", linewidth=0.8)
    steering_axes[0].set_ylabel("Angle (deg)")
    steering_axes[0].legend()
    steering_axes[0].grid(True, linestyle="--", alpha=0.4)

    rate_line, = steering_axes[1].plot(
        steering_history["time"], steering_history["rate_deg_s"], label="steering rate (deg/s)"
    )
    steering_axes[1].axhline(0.0, color="k", linewidth=0.8)
    steering_axes[1].set_ylabel("Rate (deg/s)")
    steering_axes[1].legend()
    steering_axes[1].grid(True, linestyle="--", alpha=0.4)
    steering_axes[1].set_xlabel("Time (s)")

    vel_line, = vel_ax.plot(
        velocity_history["time"], velocity_history["velocity"], label="velocity (m/s)"
    )
    vel_ax.axhline(0.0, color="k", linewidth=0.8)
    vel_ax.set_ylabel("Velocity (m/s)")
    vel_ax.legend()
    vel_ax.grid(True, linestyle="--", alpha=0.4)

    accel_line, = accel_ax.plot(
        velocity_history["time"], velocity_history["accel"], label="net accel (m/s²)"
    )
    assist_line, = accel_ax.plot(
        velocity_history["time"], velocity_history["assist"],
        linestyle=":",
        label="assist (m/s²)",
    )
    accel_ax.axhline(0.0, color="k", linewidth=0.8)
    accel_ax.set_ylabel("Acceleration (m/s²)")
    accel_ax.legend()
    accel_ax.grid(True, linestyle="--", alpha=0.4)
    accel_ax.set_xlabel("Time (s)")

    def _format_steering_summary(hist: Dict[str, np.ndarray]) -> str:
        angle = hist["angle_deg"]
        rate = hist["rate_deg_s"]
        time = hist["time"]
        threshold = 0.5
        below = np.where(np.abs(angle) <= threshold)[0]
        if below.size:
            settle_time = time[below[0]]
            settle_str = f"Settling time |angle|≤{threshold:.1f}°: {settle_time:.2f} s"
        else:
            settle_str = "Settling time |angle|≤0.5°: > sim horizon"

        return "\n".join(
            [
                f"Final steering angle: {angle[-1]:+.2f}°",
                f"Final steering rate: {rate[-1]:+.2f}°/s",
                settle_str,
                "",
                "Steering assist model:",
                r"  $\alpha_{\mathrm{assist}} = K_p\,\phi + K_d\,\dot{\phi}$",
                r"  $\dot{\phi}_{t+1} = \dot{\phi}_t + (\alpha_{\mathrm{cmd}} - \alpha_{\mathrm{assist}})\,\Delta t$",
                r"  $\phi_{t+1} = \phi_t + \dot{\phi}_{t+1}\,\Delta t$",
            ]
        )

    def _format_velocity_summary(hist: Dict[str, np.ndarray]) -> str:
        velocity = hist["velocity"]
        time = hist["time"]
        threshold = 0.05
        below = np.where(np.abs(velocity) <= threshold)[0]
        if below.size:
            settle_time = time[below[0]]
            settle_str = f"Settling time |v|≤{threshold:.2f}: {settle_time:.2f} s"
        else:
            settle_str = "Settling time |v|≤0.05: > sim horizon"
        return "\n".join(
            [
                f"Final velocity: {velocity[-1]:+.3f} m/s",
                f"Mean assist accel: {hist['assist'].mean():+.3f} m/s²",
                settle_str,
                "",
                "Velocity damping model:",
                r"  $a_{\mathrm{assist}} = K_v\,v$",
                r"  $a_{\mathrm{net}} = -a_{\mathrm{assist}}$",
                r"  $v_{t+1} = v_t + a_{\mathrm{net}}\,\Delta t$",
            ]
        )

    steering_summary_text = steering_summary_ax.text(
        0.0, 1.0, _format_steering_summary(steering_history), va="top", fontsize=9
    )
    velocity_summary_text = velocity_summary_ax.text(
        0.0, 1.0, _format_velocity_summary(velocity_history), va="top", fontsize=9
    )

    slider_color = "#f0f0f0"
    kp_ax = fig.add_axes([0.08, 0.24, 0.34, 0.03], facecolor=slider_color)
    kd_ax = fig.add_axes([0.08, 0.19, 0.34, 0.03], facecolor=slider_color)
    deadband_ax = fig.add_axes([0.08, 0.14, 0.34, 0.03], facecolor=slider_color)
    angle0_ax = fig.add_axes([0.08, 0.09, 0.34, 0.03], facecolor=slider_color)
    rate0_ax = fig.add_axes([0.08, 0.04, 0.34, 0.03], facecolor=slider_color)

    vel_gain_ax = fig.add_axes([0.55, 0.24, 0.34, 0.03], facecolor=slider_color)
    vel_deadband_ax = fig.add_axes([0.55, 0.19, 0.34, 0.03], facecolor=slider_color)
    vel0_ax = fig.add_axes([0.55, 0.14, 0.34, 0.03], facecolor=slider_color)

    kp_slider = Slider(kp_ax, "Kp", 0.1, 100.0, valinit=kp_init, valstep=0.05)
    kd_slider = Slider(kd_ax, "Kd", 0.0, 10.0, valinit=kd_init, valstep=0.05)
    deadband_slider = Slider(deadband_ax, "deadband", 0.0, 1.0, valinit=deadband_init, valstep=0.01)
    angle0_slider = Slider(angle0_ax, "angle0", 0.0, 45.0, valinit=args.angle0, valstep=1.0)
    rate0_slider = Slider(rate0_ax, "rate0", -10.0, 10.0, valinit=args.rate0, valstep=0.5)

    vel_gain_slider = Slider(vel_gain_ax, "Vel gain", 0.0, 3.0, valinit=vel_damping_init, valstep=0.05)
    vel_deadband_slider = Slider(vel_deadband_ax, "Vel deadband", 0.0, 1.0, valinit=vel_deadband_init, valstep=0.01)
    vel0_slider = Slider(vel0_ax, "vel0", -5.0, 5.0, valinit=args.vel0, valstep=0.1)

    check_ax = fig.add_axes([0.77, 0.30, 0.18, 0.08])
    check = CheckButtons(check_ax, ["sync to JSON"], [args.sync])
    for text in check.labels:
        text.set_fontsize(9)

    def update(_event: Any) -> None:
        nonlocal steering_history, velocity_history
        # Re-simulate traces as sliders move; keeps plots & summaries live-updated
        # 滑块变化时重新仿真曲线，保证图形和文字摘要实时刷新。
        steering_history = compute_steering_history(
            kp_slider.val,
            kd_slider.val,
            deadband_slider.val,
            angle0_slider.val,
            rate0_slider.val,
            commanded_accel_deg=0.0,
            steps=args.steps,
            dt=DEFAULT_CONFIG["dt"],
        )
        angle_line.set_data(steering_history["time"], steering_history["angle_deg"])
        rate_line.set_data(steering_history["time"], steering_history["rate_deg_s"])

        velocity_history = compute_velocity_history(
            vel_gain_slider.val,
            vel_deadband_slider.val,
            vel0_slider.val,
            steps=args.steps,
            dt=DEFAULT_CONFIG["dt"],
        )
        vel_line.set_data(velocity_history["time"], velocity_history["velocity"])
        accel_line.set_data(velocity_history["time"], velocity_history["accel"])
        assist_line.set_data(velocity_history["time"], velocity_history["assist"])

        for axis in steering_axes + [vel_ax, accel_ax]:
            axis.relim()
            axis.autoscale_view()

        steering_summary_text.set_text(_format_steering_summary(steering_history))
        velocity_summary_text.set_text(_format_velocity_summary(velocity_history))
        fig.canvas.draw_idle()

        if check.get_status()[0] and config_path is not None:
            # Sync back to JSON so CLI / notebook pick up new assists
            # 若启用同步则写回 JSON，方便 CLI 或 Notebook 使用最新助力参数。
            vehicle_cfg["steering_damping"] = float(kp_slider.val)
            vehicle_cfg["steering_rate_damping"] = float(kd_slider.val)
            vehicle_cfg["steering_assist_deadband"] = float(deadband_slider.val)
            vehicle_cfg["velocity_damping"] = float(vel_gain_slider.val)
            vehicle_cfg["velocity_deadband"] = float(vel_deadband_slider.val)
            persist_vehicle_config(config_path, vehicle_cfg)

    kp_slider.on_changed(update)
    kd_slider.on_changed(update)
    deadband_slider.on_changed(update)
    angle0_slider.on_changed(update)
    rate0_slider.on_changed(update)
    vel_gain_slider.on_changed(update)
    vel_deadband_slider.on_changed(update)
    vel0_slider.on_changed(update)

    plt.show()


# ---------------------------------------------------------------------------
# CLI entry 命令行入口
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the tuner GUI.

    解析调参器所需的命令行参数（配置路径、初始条件等）。
    """
    parser = argparse.ArgumentParser(description="Assist-model tuner GUI")
    parser.add_argument("--config", type=str, help="Path to JSON config to load/update.")
    parser.add_argument("--angle0", type=float, default=20.0, help="Initial steering angle in degrees.")
    parser.add_argument("--rate0", type=float, default=0.0, help="Initial steering rate in deg/s.")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
    parser.add_argument(
        "--vel0",
        type=float,
        default=2.0,
        help="Initial longitudinal speed in m/s when evaluating throttle damping.",
    )
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
    """Program entry point launching the tuner UI.

    程序入口：根据传入参数启动调参界面。
    """
    args = parse_args(argv or sys.argv[1:])
    launch_tuner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

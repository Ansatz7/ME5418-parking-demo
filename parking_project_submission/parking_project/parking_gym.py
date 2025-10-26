"""Gymnasium environment that simulates a parking scenario.

Angles in the public configuration are expressed in degrees; the environment
converts them to radians internally. 模拟智能泊车场景的 Gymnasium 环境，外部配置使用角度制，内部统一换算成弧度。
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - IPython not always available
    display = None


# All angles in DEFAULT_CONFIG (and any user overrides) are expressed in degrees.
# 默认配置（以及外部覆盖项）中的角度均采用度数表示。
DEFAULT_CONFIG: Dict = {
    "dt": 0.1,
    "max_steps": 4000,
    "field_size": 60.0,
    "ray_max_range": 12.0,
    "observation_noise": {
        "enabled": True,
        "std": 0.005,
    },
    "ray_angles": [
        -135.0,
        -90.0,
        -60.0,
        -30.0,
        30.0,
        60.0,
        90.0,
        135.0,
        0.0,
    ],
    "vehicle": {
        "length": 4.0,
        "width": 2.0,
        "wheel_base": 2.5,
        "max_speed": 3.0,
        "max_reverse_speed": -2.0,
        "max_steering_angle": 60.0,
        "max_steering_rate": 30.0,
        "steering_damping": 52.5,
        "steering_rate_damping": 8.75,  # viscous term for steering-rate damping (rad/s^2 per rad/s)
        "steering_assist_deadband": 0.03,  # rad/s^2 tolerance before assist engages
        "velocity_damping": 2.45,
        "velocity_deadband": 0.03,
        "enable_steering_assist": True,
        "manual_forward_accel": 1.5,
        "manual_reverse_accel": 2.0,
        "manual_steering_accel": 10.0,
    },
    "spawn_region": [-6.0, 6.0, -6.0, 6.0],
    "parking_slot": {
        "length": 5.5,
        "width": 2.5,
        "offset_x_range": (-7.0, -3.0),
        "offset_y_range": (-4.0, 4.0),
        "orientation_range": (0.0, 0.0),
    },
    "static_obstacles": {
        "count": 3,
        "size_range": (1.0, 2.5),
        "min_distance": 2.0,
        "seed": None,
    },
    "dynamic_obstacles": {
        "count": 2,
        "radius": 1.0,
        "speed_range": (0.5, 1.0),
        "behavior": "goal_driven",
        "min_distance": 4.0,
        "heading_noise": 15.0,
    },
    "reward": {
        "distance_scale": 1.5,
        "heading_scale": 0.5,
        "collision": -120.0,
        "success": 140.0,
        "smoothness": 0.05,
        "step_cost": 0.2,
        "velocity_penalty": 0.3,
    },
    "success_thresholds": {
        "position": 0.4,
        "orientation": 6.0,
        "speed": 0.3,
        "steering": 5.0,
    },
    "rng_seed": None,
}


class ParkingEnv(gym.Env):
    """Gymnasium-compatible parking environment with lidar-based observations.

    基于 Gymnasium 的泊车环境，观测包含多束激光距离与车辆/车位的相对姿态。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[Dict] = None, render_mode: str = "human") -> None:
        super().__init__()
        self.config = self._merge_config(config)
        self.rng = np.random.default_rng(self.config["rng_seed"])
        self.render_mode = render_mode
        self.vehicle_cfg = self.config["vehicle"]
        self.reward_cfg = self.config["reward"]
        self.success_cfg = self.config["success_thresholds"]
        obs_noise_cfg = self.config.get("observation_noise", {})
        self.obs_noise_enabled = bool(obs_noise_cfg.get("enabled", True))
        self.obs_noise_std = float(obs_noise_cfg.get("std", 0.005))  # σ matches DEFAULT_CONFIG
        if self.obs_noise_std < 0.0:
            raise ValueError("Observation noise std must be non-negative.")

        self.dt = self.config["dt"]
        self.field_size = self.config["field_size"]
        self.half_field = self.field_size / 2.0

        self.num_rays = len(self.config["ray_angles"])
        if self.num_rays <= 0:
            raise ValueError("Config must specify at least one ray angle.")

        self._base_obs_dim = 11  # core features before appending ray distances
        self.obs_dim = self._base_obs_dim + self.num_rays

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-3.0, -1.5], dtype=np.float32),
            high=np.array([2.0, 1.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.vehicle_state: Dict[str, float] = {}
        self.target_slot: Dict[str, float] = {}
        self.static_obstacles: List[Dict] = []
        self.dynamic_obstacles: List[Dict] = []

        self.current_step = 0
        self.last_action = np.zeros(2, dtype=float)
        self.last_reward = 0.0
        self.last_reward_terms: Dict[str, float] = {}

        self.fig = None
        self.ax = None
        self._artists = {}

    def set_observation_noise(
        self,
        *,
        enabled: Optional[bool] = None,
        std: Optional[float] = None,
    ) -> None:
        """Toggle or retune observation noise at runtime.

        运行期间启用/禁用观测噪声，或调整噪声标准差。
        """
        if enabled is not None:
            self.obs_noise_enabled = bool(enabled)
        if std is not None:
            std = float(std)
            if std < 0.0:
                raise ValueError("Observation noise std must be non-negative.")
            self.obs_noise_std = std

    def _merge_config(self, override: Optional[Dict]) -> Dict:
        """Merge user overrides into the default config and convert angles.

        合并用户覆盖项并把所有角度字段转换为弧度供内部使用。
        """
        base_cfg = copy.deepcopy(DEFAULT_CONFIG)
        if override is None:
            return self._convert_angles_to_radians(base_cfg)

        user_cfg = copy.deepcopy(override)
        for key, value in user_cfg.items():
            if key not in base_cfg or not isinstance(value, dict):
                base_cfg[key] = value
            else:
                base_cfg[key].update(value)
        return self._convert_angles_to_radians(base_cfg)

    def _convert_angles_to_radians(self, config: Dict) -> Dict:
        # Convert degree-based configuration values to radians for internal use.
        # 将度数配置转换为弧度，供物理和几何计算使用。
        config["ray_angles"] = [math.radians(float(v)) for v in config["ray_angles"]]

        vehicle_cfg = config.get("vehicle", {})
        if "max_steering_angle" in vehicle_cfg:
            vehicle_cfg["max_steering_angle"] = math.radians(float(vehicle_cfg["max_steering_angle"]))
        if "max_steering_rate" in vehicle_cfg:
            vehicle_cfg["max_steering_rate"] = math.radians(float(vehicle_cfg["max_steering_rate"]))
        if "steering_assist_deadband" in vehicle_cfg:
            vehicle_cfg["steering_assist_deadband"] = math.radians(float(vehicle_cfg["steering_assist_deadband"]))
        if "manual_steering_accel" in vehicle_cfg:
            vehicle_cfg["manual_steering_accel"] = math.radians(float(vehicle_cfg["manual_steering_accel"]))

        slot_cfg = config.get("parking_slot", {})
        if "orientation_range" in slot_cfg:
            slot_cfg["orientation_range"] = tuple(
                math.radians(float(v)) for v in slot_cfg["orientation_range"]
            )

        dyn_cfg = config.get("dynamic_obstacles", {})
        if "heading_noise" in dyn_cfg:
            dyn_cfg["heading_noise"] = math.radians(float(dyn_cfg["heading_noise"]))

        success_cfg = config.get("success_thresholds", {})
        if "orientation" in success_cfg:
            success_cfg["orientation"] = math.radians(float(success_cfg["orientation"]))
        if "steering" in success_cfg:
            success_cfg["steering"] = math.radians(float(success_cfg["steering"]))

        return config

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options is None:
            options = {}
        self.current_step = 0

        self._spawn_vehicle()
        self._generate_parking_slot()
        self._generate_static_obstacles()
        self._generate_dynamic_obstacles()

        observation = self._get_observation()
        info = self._initial_info()
        self.last_action = np.zeros(2, dtype=float)
        self.last_reward = 0.0
        self.last_reward_terms = {}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._update_vehicle(action)
        self._update_dynamic_obstacles()

        observation = self._get_observation()
        reward, info = self._compute_reward()
        terminated, truncated, term_reason = self._check_termination(info)
        info["terminal_reason"] = term_reason

        self.last_action = np.array(action, dtype=float)
        self.last_reward = float(reward)
        self.last_reward_terms = info.get("reward_terms", {}).copy()

        if terminated or truncated:
            info["final_observation"] = observation

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            raise NotImplementedError("Only human render mode is implemented.")

        if self.fig is None:
            plt.ion()
            plt.rcParams.setdefault("keymap.save", [])
            plt.rcParams["keymap.save"] = []  # avoid save dialog on 's' key during manual control
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax.set_xlim(-self.half_field, self.half_field)
            self.ax.set_ylim(-self.half_field, self.half_field)
            self.ax.set_aspect("equal")
            self.ax.grid(True, linestyle="--", alpha=0.3)
            self._artists = {}
            backend = plt.get_backend().lower()
            if display is not None and backend.startswith("module://ipympl"):
                display(self.fig)
            else:
                plt.show(block=False)

        self.ax.cla()
        self.ax.set_xlim(-self.half_field, self.half_field)
        self.ax.set_ylim(-self.half_field, self.half_field)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.3)

        boundary = plt.Rectangle(
            (-self.half_field, -self.half_field),
            self.field_size,
            self.field_size,
            fill=False,
            lw=2.0,
            color="black",
        )
        self.ax.add_patch(boundary)

        if self.target_slot:
            self._draw_parking_slot()
        if self.static_obstacles:
            self._draw_static_obstacles()
        if self.dynamic_obstacles:
            self._draw_dynamic_obstacles()
        if self.vehicle_state:
            self._draw_vehicle()
            self._draw_rays()

        if self.vehicle_state:
            obs = self._get_observation(raw=True)
            sensor_values = obs[-self.num_rays :]
            slot_error = self._vehicle_to_slot_frame()
            ray_angles_deg = [math.degrees(angle) for angle in self.config["ray_angles"]]
            state_dim = self.observation_space.shape[0]
            xr, yr, dtheta = slot_error
            cos_yaw = math.cos(self.vehicle_state['yaw'])
            sin_yaw = math.sin(self.vehicle_state['yaw'])
            cos_dtheta = math.cos(dtheta)
            sin_dtheta = math.sin(dtheta)
            text_lines = [
                f"Step: {self.current_step} (dt={self.dt:.2f}s)",
                f"State ({state_dim} dims):",
                "  Relative slot frame:",
                f"    xr={xr:.2f} m, yr={yr:.2f} m",
                f"    delta yaw={math.degrees(dtheta):+.1f} deg (cos={cos_dtheta:+.2f}, sin={sin_dtheta:+.2f})",
                f"    dv={self.vehicle_state['velocity'] - 0.0:+.2f} m/s (target=0)",
                "  Vehicle dynamics:",
                f"    v={self.vehicle_state['velocity']:.2f} m/s",
                f"    steering angle={math.degrees(self.vehicle_state['steering_angle']):+.1f} deg",
                f"    steering rate={math.degrees(self.vehicle_state['steering_rate']):+.1f} deg/s",
                "  Global pose (for cos/sin encoding):",
                f"    x={self.vehicle_state['x']:.2f} m, y={self.vehicle_state['y']:.2f} m",
                f"    cos(yaw)={cos_yaw:+.2f}, sin(yaw)={sin_yaw:+.2f}",
                f"  Rays ({self.num_rays} beams, normalized 0-1):",
            ]
            text_lines.extend(
                [
                    f"    Ray {idx} ({angle:+.0f} deg): {dist:.2f}"
                    for idx, (angle, dist) in enumerate(zip(ray_angles_deg, sensor_values), start=1)
                ]
            )
            self.ax.text(
                1.02,
                0.99,
                "\n".join(text_lines),
                transform=self.ax.transAxes,
                fontsize=8,
                va="top",
            )

            action_lines = [
                "Action input (lon, steer):",
                f"  lon accel: {self.last_action[0]:+.2f} m/s^2",
                f"  steer accel: {math.degrees(self.last_action[1]):+.1f} deg/s^2",
            ]
            self.ax.text(
                -0.32,
                0.99,
                "\n".join(action_lines),
                transform=self.ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )

            if self.last_reward_terms:
                reward_lines = [
                    "Reward components:",
                    f"  total: {self.last_reward:+.2f}",
                    f"  distance: {self.last_reward_terms.get('distance', 0.0):+.2f}",
                    f"  heading: {self.last_reward_terms.get('heading', 0.0):+.2f}",
                    f"  velocity: {self.last_reward_terms.get('velocity', 0.0):+.2f}",
                    f"  smoothness: {self.last_reward_terms.get('smoothness', 0.0):+.2f}",
                    f"  step: {self.last_reward_terms.get('step', 0.0):+.2f}",
                    f"  collision: {self.last_reward_terms.get('collision', 0.0):+.2f}",
                    f"  success: {self.last_reward_terms.get('success', 0.0):+.2f}",
                ]
            else:
                reward_lines = [
                    "Reward components:",
                    "  total: 0.00",
                ]
            self.ax.text(
                -0.32,
                0.55,
                "\n".join(reward_lines),
                transform=self.ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )

        plt.pause(0.001)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    # Vehicle and environment setup helpers ---------------------------------
    def _spawn_vehicle(self) -> None:
        xmin, xmax, ymin, ymax = self.config["spawn_region"]
        x = self.rng.uniform(xmin, xmax)
        y = self.rng.uniform(ymin, ymax)
        yaw = self.rng.uniform(-math.pi, math.pi)
        velocity = 0.0
        steering_angle = 0.0
        steering_rate = 0.0

        self.vehicle_state = {
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
            "velocity": float(velocity),
            "steering_angle": float(steering_angle),
            "steering_rate": float(steering_rate),
        }

    def _generate_parking_slot(self) -> None:
        slot_cfg = self.config["parking_slot"]
        offset_x = self.rng.uniform(*slot_cfg["offset_x_range"])
        offset_y = self.rng.uniform(*slot_cfg["offset_y_range"])
        yaw = self.rng.uniform(*slot_cfg["orientation_range"])

        center = np.array([offset_x, offset_y], dtype=float)
        base_slot_cfg = DEFAULT_CONFIG["parking_slot"]
        length = float(slot_cfg.get("length", base_slot_cfg["length"]))
        width = float(slot_cfg.get("width", base_slot_cfg["width"]))

        self.target_slot = {
            "center": center,
            "yaw": float(yaw),
            "length": length,
            "width": width,
        }

    def _generate_static_obstacles(self) -> None:
        cfg = self.config["static_obstacles"]
        self.static_obstacles = []
        count = cfg["count"]
        size_min, size_max = cfg["size_range"]
        min_dist = cfg["min_distance"]

        if cfg.get("seed") is not None:
            local_rng = np.random.default_rng(cfg["seed"])
        else:
            local_rng = self.rng

        attempts = 0
        while len(self.static_obstacles) < count and attempts < 200:
            attempts += 1
            width = float(local_rng.uniform(size_min, size_max))
            height = float(local_rng.uniform(size_min, size_max))
            x = float(local_rng.uniform(-self.half_field + width, self.half_field - width))
            y = float(local_rng.uniform(-self.half_field + height, self.half_field - height))

            center = np.array([x, y], dtype=float)
            if self._is_far_from_vehicle(center, min_dist) and self._is_far_from_slot(
                center, min_dist
            ):
                self.static_obstacles.append(
                    {
                        "center": center,
                        "size": (width, height),
                    }
                )

    def _generate_dynamic_obstacles(self) -> None:
        cfg = self.config["dynamic_obstacles"]
        self.dynamic_obstacles = []
        count = cfg["count"]
        radius = cfg["radius"]
        min_dist = cfg["min_distance"]

        for _ in range(count):
            for _ in range(100):
                x = float(self.rng.uniform(-self.half_field + radius, self.half_field - radius))
                y = float(self.rng.uniform(-self.half_field + radius, self.half_field - radius))
                center = np.array([x, y], dtype=float)
                if (
                    self._is_far_from_vehicle(center, min_dist)
                    and self._is_far_from_slot(center, min_dist)
                ):
                    break
            else:
                center = np.array([0.0, 0.0], dtype=float)

            heading = float(self.rng.uniform(-math.pi, math.pi))
            speed = float(self.rng.uniform(*cfg["speed_range"]))
            obstacle = {
                "pos": center,
                "heading": heading,
                "speed": speed,
                "radius": float(radius),
                "behavior": cfg["behavior"],
                "target": self._sample_random_point(),
            }
            self.dynamic_obstacles.append(obstacle)

    # Update functions -------------------------------------------------------
    # 状态更新函数集，负责车辆、动态障碍等的时间推进。
    def _update_vehicle(self, action: np.ndarray) -> None:
        """Integrate vehicle dynamics for one time-step.

        对车辆进行一步积分：接收 [纵向加速度, 转向角加速度]，更新速度、姿态等状态量。
        """
        accel = float(action[0])
        steering_accel = float(action[1])

        vel = self.vehicle_state["velocity"]
        velocity_deadband = self.vehicle_cfg.get("velocity_deadband", 0.0)
        velocity_damping = self.vehicle_cfg.get("velocity_damping", 0.0)
        if abs(accel) <= velocity_deadband and velocity_damping > 0.0:
            accel -= velocity_damping * vel
        vel += accel * self.dt
        vel = np.clip(
            vel,
            self.vehicle_cfg["max_reverse_speed"],
            self.vehicle_cfg["max_speed"],
        )

        steering_rate = self.vehicle_state["steering_rate"]
        steering_rate += steering_accel * self.dt

        assist_deadband = self.vehicle_cfg.get("steering_assist_deadband", 0.0)
        if (
            self.vehicle_cfg.get("enable_steering_assist", False)
            and abs(steering_accel) <= assist_deadband
        ):
            assist_accel = self.vehicle_cfg["steering_damping"] * self.vehicle_state["steering_angle"]
            assist_accel += self.vehicle_cfg.get("steering_rate_damping", 0.0) * steering_rate
            steering_rate -= assist_accel * self.dt

        steering_rate = np.clip(
            steering_rate,
            -self.vehicle_cfg["max_steering_rate"],
            self.vehicle_cfg["max_steering_rate"],
        )

        steering_angle = self.vehicle_state["steering_angle"] + steering_rate * self.dt
        steering_angle = np.clip(
            steering_angle,
            -self.vehicle_cfg["max_steering_angle"],
            self.vehicle_cfg["max_steering_angle"],
        )

        yaw = self.vehicle_state["yaw"]
        wheel_base = self.vehicle_cfg["wheel_base"]
        beta = math.atan(math.tan(steering_angle) / 2.0)

        x = self.vehicle_state["x"]
        y = self.vehicle_state["y"]

        x += vel * math.cos(yaw + beta) * self.dt
        y += vel * math.sin(yaw + beta) * self.dt
        yaw += vel * math.tan(steering_angle) / wheel_base * self.dt
        yaw = self._wrap_angle(yaw)

        self.vehicle_state.update(
            {
                "velocity": vel,
                "steering_rate": steering_rate,
                "steering_angle": steering_angle,
                "yaw": yaw,
                "x": x,
                "y": y,
            }
        )

        self.vehicle_state["velocity"] += self.rng.normal(0.0, 0.02)
        self.vehicle_state["steering_angle"] += self.rng.normal(0.0, math.radians(0.2))

    def _update_dynamic_obstacles(self) -> None:
        if not self.dynamic_obstacles:
            return

        cfg = self.config["dynamic_obstacles"]
        behavior = cfg["behavior"]
        heading_noise = cfg.get("heading_noise", math.radians(10.0))

        for obstacle in self.dynamic_obstacles:
            if behavior == "random_walk":
                obstacle["heading"] += self.rng.normal(0.0, heading_noise)
            elif behavior == "goal_driven":
                target = obstacle["target"]
                direction = target - obstacle["pos"]
                if np.linalg.norm(direction) < 0.5:
                    obstacle["target"] = self._sample_random_point()
                else:
                    obstacle["heading"] = math.atan2(direction[1], direction[0])

            obstacle["heading"] = self._wrap_angle(obstacle["heading"])
            displacement = np.array(
                [
                    math.cos(obstacle["heading"]),
                    math.sin(obstacle["heading"]),
                ],
                dtype=float,
            )
            obstacle["pos"] += displacement * obstacle["speed"] * self.dt

            for i in range(2):
                if obstacle["pos"][i] > self.half_field - obstacle["radius"]:
                    obstacle["pos"][i] = self.half_field - obstacle["radius"]
                    obstacle["heading"] = self._wrap_angle(obstacle["heading"] + math.pi)
                if obstacle["pos"][i] < -self.half_field + obstacle["radius"]:
                    obstacle["pos"][i] = -self.half_field + obstacle["radius"]
                    obstacle["heading"] = self._wrap_angle(obstacle["heading"] + math.pi)

    # Observation & reward ---------------------------------------------------
    # 观测/奖励相关逻辑：组装状态向量、计算奖励与终止条件。
    def _get_observation(self, raw: bool = False) -> np.ndarray:
        x = self.vehicle_state["x"]
        y = self.vehicle_state["y"]
        yaw = self.vehicle_state["yaw"]
        velocity = self.vehicle_state["velocity"]
        steering_angle = self.vehicle_state["steering_angle"]
        steering_rate = self.vehicle_state["steering_rate"]

        rel_slot = self._vehicle_to_slot_frame()
        ray_distances = self._cast_rays()

        obs = np.array(
            [
                x / self.half_field,
                y / self.half_field,
                math.cos(yaw),
                math.sin(yaw),
                velocity,
                steering_angle,
                steering_rate,
                rel_slot[0] / self.field_size,
                rel_slot[1] / self.field_size,
                math.cos(rel_slot[2]),
                math.sin(rel_slot[2]),
                *ray_distances,
            ],
            dtype=np.float32,
        )

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"Observation dimension mismatch: expected {self.obs_dim}, got {obs.shape[0]}"
            )

        if not raw and self.obs_noise_enabled and self.obs_noise_std > 0.0:
            # Apply configurable Gaussian noise to encourage policy robustness
            # 叠加可配置的高斯噪声，提升策略对传感抖动的鲁棒性。
            obs += self.rng.normal(0.0, self.obs_noise_std, size=obs.shape)

        return obs

    def _compute_reward(self) -> Tuple[float, Dict]:
        rel_slot = self._vehicle_to_slot_frame()
        distance = np.linalg.norm(rel_slot[:2])
        heading_error = abs(rel_slot[2])
        velocity = abs(self.vehicle_state["velocity"])
        steering_rate = abs(self.vehicle_state["steering_rate"])
        collision = self._check_collision()
        success = self._check_success(rel_slot, velocity)

        reward = 0.0
        distance_term = -self.reward_cfg["distance_scale"] * distance
        heading_term = -self.reward_cfg["heading_scale"] * heading_error
        velocity_term = -self.reward_cfg["velocity_penalty"] * velocity
        smoothness_term = -self.reward_cfg["smoothness"] * (steering_rate ** 2)
        step_term = -self.reward_cfg["step_cost"]

        collision_term = self.reward_cfg["collision"] if collision else 0.0
        success_term = self.reward_cfg["success"] if success else 0.0

        reward += distance_term
        reward += heading_term
        reward += velocity_term
        reward += smoothness_term
        reward += step_term
        reward += collision_term
        reward += success_term

        info = {
            "distance_to_slot": distance,
            "heading_error": heading_error,
            "collision": collision,
            "success": success,
            "reward_terms": {
                "distance": distance_term,
                "heading": heading_term,
                "velocity": velocity_term,
                "smoothness": smoothness_term,
                "step": step_term,
                "collision": collision_term,
                "success": success_term,
            },
        }
        return reward, info

    def _check_termination(self, info: Dict) -> Tuple[bool, bool, str]:
        if info["collision"]:
            return True, False, "collision"

        if abs(self.vehicle_state["x"]) > self.half_field or abs(self.vehicle_state["y"]) > self.half_field:
            return True, False, "out_of_bounds"

        if info["success"]:
            return True, False, "success"

        if self.current_step >= self.config["max_steps"]:
            return False, True, "timeout"

        return False, False, "running"

    def _initial_info(self) -> Dict:
        rel_slot = self._vehicle_to_slot_frame()
        return {
            "distance_to_slot": float(np.linalg.norm(rel_slot[:2])),
            "heading_error": float(abs(rel_slot[2])),
            "collision": False,
            "success": False,
            "terminal_reason": "reset",
        }

    def _check_collision(self) -> bool:
        vehicle_poly = self._vehicle_polygon()

        for obstacle in self.static_obstacles:
            rect = self._obstacle_rectangle(obstacle)
            if self._polygon_collision(vehicle_poly, rect):
                return True

        for obstacle in self.dynamic_obstacles:
            if self._polygon_circle_collision(vehicle_poly, obstacle["pos"], obstacle["radius"]):
                return True

        return False

    def _check_success(self, rel_slot: np.ndarray, velocity: float) -> bool:
        pos_ok = np.linalg.norm(rel_slot[:2]) < self.success_cfg["position"]
        heading_ok = abs(rel_slot[2]) < self.success_cfg["orientation"]
        steering_ok = abs(self.vehicle_state["steering_angle"]) < self.success_cfg["steering"]
        speed_ok = velocity < self.success_cfg["speed"]
        return pos_ok and heading_ok and steering_ok and speed_ok

    # Geometry helpers -------------------------------------------------------
    # 几何辅助函数：用于碰撞检测与坐标系转换。
    def _vehicle_polygon(self) -> np.ndarray:
        length = self.vehicle_cfg["length"]
        width = self.vehicle_cfg["width"]
        x = self.vehicle_state["x"]
        y = self.vehicle_state["y"]
        yaw = self.vehicle_state["yaw"]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        half_length = length / 2.0
        half_width = width / 2.0

        corners_local = np.array(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ]
        )

        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_world = corners_local @ rotation.T + np.array([x, y])
        return corners_world

    def _obstacle_rectangle(self, obstacle: Dict) -> np.ndarray:
        center = obstacle["center"]
        width, height = obstacle["size"]
        half_w = width / 2.0
        half_h = height / 2.0
        corners = np.array(
            [
                [center[0] - half_w, center[1] - half_h],
                [center[0] + half_w, center[1] - half_h],
                [center[0] + half_w, center[1] + half_h],
                [center[0] - half_w, center[1] + half_h],
            ]
        )
        return corners

    def _polygon_collision(self, poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
        if self._separating_axis_theorem(poly_a, poly_b):
            return False
        return True

    def _polygon_circle_collision(self, poly: np.ndarray, center: np.ndarray, radius: float) -> bool:
        if self._point_in_polygon(center, poly):
            return True
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            if self._distance_point_to_segment(center, p1, p2) <= radius:
                return True
        return False

    def _distance_point_to_segment(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0.0:
            return float(np.linalg.norm(point - seg_start))
        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        projection = seg_start + t * seg_vec
        return float(np.linalg.norm(point - projection))

    def _point_in_polygon(self, point: np.ndarray, poly: np.ndarray) -> bool:
        x, y = point
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _separating_axis_theorem(self, poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
        def _normals(poly: np.ndarray) -> List[np.ndarray]:
            normals = []
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal_norm = np.linalg.norm(normal)
                if normal_norm > 0:
                    normals.append(normal / normal_norm)
            return normals

        def _project(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
            projections = poly @ axis
            return projections.min(), projections.max()

        for axis in _normals(poly_a) + _normals(poly_b):
            min_a, max_a = _project(poly_a, axis)
            min_b, max_b = _project(poly_b, axis)
            if max_a < min_b or max_b < min_a:
                return True
        return False

    def _cast_rays(self) -> List[float]:
        ray_distances = []
        origin = np.mean(self._vehicle_polygon(), axis=0)
        base_yaw = self.vehicle_state["yaw"]
        max_range = self.config["ray_max_range"]

        for angle in self.config["ray_angles"]:
            ray_dir = np.array(
                [math.cos(base_yaw + angle), math.sin(base_yaw + angle)],
                dtype=float,
            )
            distance = self._ray_distance(origin, ray_dir, max_range)
            ray_distances.append(distance / max_range)
        return ray_distances

    def _ray_distance(self, origin: np.ndarray, direction: np.ndarray, max_range: float) -> float:
        min_dist = max_range
        dist_field = self._distance_to_field(origin, direction, max_range)
        min_dist = min(min_dist, dist_field)

        for obstacle in self.static_obstacles:
            dist = self._distance_to_rectangle(origin, direction, obstacle)
            if dist is not None:
                min_dist = min(min_dist, dist)

        for obstacle in self.dynamic_obstacles:
            dist = self._distance_to_circle(origin, direction, obstacle["pos"], obstacle["radius"])
            if dist is not None:
                min_dist = min(min_dist, dist)

        return float(min_dist)

    def _distance_to_field(self, origin: np.ndarray, direction: np.ndarray, max_range: float) -> float:
        t_values = []
        for axis in range(2):
            if abs(direction[axis]) < 1e-6:
                continue
            bound = self.half_field if direction[axis] > 0 else -self.half_field
            t = (bound - origin[axis]) / direction[axis]
            if 0.0 < t <= max_range:
                other = origin[(axis + 1) % 2] + t * direction[(axis + 1) % 2]
                if -self.half_field <= other <= self.half_field:
                    t_values.append(t)
        if t_values:
            return min(t_values)
        return max_range

    def _distance_to_rectangle(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        obstacle: Dict,
    ) -> Optional[float]:
        corners = self._obstacle_rectangle(obstacle)
        min_t = None
        for i in range(len(corners)):
            a = corners[i]
            b = corners[(i + 1) % len(corners)]
            t = self._ray_segment_intersection(origin, direction, a, b)
            if t is not None and (min_t is None or t < min_t):
                min_t = t
        return min_t

    def _distance_to_circle(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> Optional[float]:
        oc = origin - center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        candidates = [t for t in (t1, t2) if t > 0]
        if not candidates:
            return None
        return min(candidates)

    def _ray_segment_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> Optional[float]:
        v1 = origin - p1
        v2 = p2 - p1
        denominator = direction[0] * v2[1] - direction[1] * v2[0]
        if abs(denominator) < 1e-8:
            return None
        t1 = (v2[0] * v1[1] - v2[1] * v1[0]) / denominator
        t2 = (direction[0] * v1[1] - direction[1] * v1[0]) / denominator
        if t1 >= 0 and 0 <= t2 <= 1:
            return t1
        return None

    def _vehicle_to_slot_frame(self) -> np.ndarray:
        vehicle_pos = np.array([self.vehicle_state["x"], self.vehicle_state["y"]])
        slot_center = self.target_slot["center"]
        diff = slot_center - vehicle_pos
        slot_yaw = self.target_slot["yaw"]
        rotation = np.array(
            [
                [math.cos(slot_yaw), math.sin(slot_yaw)],
                [-math.sin(slot_yaw), math.cos(slot_yaw)],
            ]
        )
        local = rotation @ diff
        yaw_error = self._wrap_angle(self.vehicle_state["yaw"] - slot_yaw)
        return np.array([local[0], local[1], yaw_error])

    # Drawing helpers --------------------------------------------------------
    # 绘图辅助函数：负责 Matplotlib 渲染车辆、车位、传感器射线等元素。
    def _draw_vehicle(self) -> None:
        required_keys = {"x", "y", "yaw", "steering_angle"}
        if not required_keys.issubset(self.vehicle_state.keys()):
            return
        poly = self._vehicle_polygon()
        self.ax.fill(poly[:, 0], poly[:, 1], color="#1f77b4", alpha=0.6)

        rear_edge = poly[2:4]
        self.ax.plot(
            rear_edge[:, 0],
            rear_edge[:, 1],
            color="red",
            linewidth=2.0,
        )

        body_center = np.mean(poly, axis=0)

        wheel_base = self.vehicle_cfg["wheel_base"]
        heading = self.vehicle_state["yaw"]
        axle_dir = np.array([math.cos(heading), math.sin(heading)])
        half_wheel_base = wheel_base / 2.0
        rear_axle = body_center - half_wheel_base * axle_dir
        front_axle = body_center + half_wheel_base * axle_dir

        wheel_length = 0.5
        wheel_width = 0.2
        for axle_center, steering in (
            (front_axle, self.vehicle_state["steering_angle"]),
            (rear_axle, 0.0),
        ):
            for offset in (-0.6, 0.6):
                lateral = np.array([-math.sin(heading), math.cos(heading)]) * offset
                wheel_center = axle_center + lateral
                self._draw_wheel(wheel_center, heading + steering, wheel_length, wheel_width)

    def _draw_wheel(
        self,
        center: np.ndarray,
        angle: float,
        length: float,
        width: float,
    ) -> None:
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        corners = np.array(
            [
                [length / 2, width / 2],
                [length / 2, -width / 2],
                [-length / 2, -width / 2],
                [-length / 2, width / 2],
            ]
        )
        world = corners @ rotation.T + center
        self.ax.fill(world[:, 0], world[:, 1], color="black")

    def _draw_parking_slot(self) -> None:
        slot = self.target_slot
        if not slot:
            return

        base_slot_cfg = DEFAULT_CONFIG["parking_slot"]
        length = slot.get("length", base_slot_cfg["length"])
        width = slot.get("width", base_slot_cfg["width"])
        center = slot.get("center")
        rotation = slot.get("yaw")

        if center is None or rotation is None:
            return
        half_length = length / 2.0
        half_width = width / 2.0
        corners_local = np.array(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ]
        )
        rotation_matrix = np.array(
            [
                [math.cos(rotation), -math.sin(rotation)],
                [math.sin(rotation), math.cos(rotation)],
            ]
        )
        corners_world = corners_local @ rotation_matrix.T + center
        slot_patch = plt.Polygon(
            corners_world,
            closed=True,
            fill=False,
            linestyle="--",
            edgecolor="#2ca02c",
            linewidth=2.0,
        )
        self.ax.add_patch(slot_patch)

        rear_edge = corners_world[2:4]
        self.ax.plot(
            rear_edge[:, 0],
            rear_edge[:, 1],
            color="red",
            linewidth=2.0,
        )

    def _draw_static_obstacles(self) -> None:
        for obstacle in self.static_obstacles:
            width, height = obstacle["size"]
            center = obstacle["center"]
            rect = plt.Rectangle(
                (center[0] - width / 2, center[1] - height / 2),
                width,
                height,
                color="#ff7f0e",
                alpha=0.6,
            )
            self.ax.add_patch(rect)

    def _draw_dynamic_obstacles(self) -> None:
        for obstacle in self.dynamic_obstacles:
            patch = plt.Circle(
                obstacle["pos"],
                obstacle["radius"],
                color="#d62728",
                alpha=0.6,
            )
            self.ax.add_patch(patch)

    def _draw_rays(self) -> None:
        origin = np.mean(self._vehicle_polygon(), axis=0)
        base_yaw = self.vehicle_state["yaw"]
        max_range = self.config["ray_max_range"]

        for value, angle in zip(self._cast_rays(), self.config["ray_angles"]):
            length = value * max_range
            direction = np.array([math.cos(base_yaw + angle), math.sin(base_yaw + angle)])
            endpoint = origin + direction * length
            self.ax.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                linestyle="--",
                color="#17becf",
                linewidth=1.2,
            )

    # Utility ----------------------------------------------------------------
    # 工具函数：角度规整、随机采样等通用方法。
    def _wrap_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _is_far_from_vehicle(self, point: np.ndarray, min_distance: float) -> bool:
        vehicle_pos = np.array([self.vehicle_state["x"], self.vehicle_state["y"]])
        return np.linalg.norm(point - vehicle_pos) > min_distance

    def _is_far_from_slot(self, point: np.ndarray, min_distance: float) -> bool:
        return np.linalg.norm(point - self.target_slot["center"]) > min_distance

    def _sample_random_point(self) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(-self.half_field, self.half_field),
                self.rng.uniform(-self.half_field, self.half_field),
            ],
            dtype=float,
        )

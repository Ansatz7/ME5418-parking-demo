# Module Overview for ME5418-parking-demo

This document summarizes the purpose, inputs/outputs, main functions, running instructions, and important notes for each file/module in the repository. It is intended as a developer-facing quick reference.

## Project summary

- Goal: provide an interactive parking simulation environment (Gymnasium) with demo scripts, a Jupyter Notebook frontend, and a Qt-based steering assist tuner.
- Environment: Conda env `parking-rl` (Python 3.8), Matplotlib + PyQt + Gymnasium 0.29.
- Configurations are JSON files in `parking_project/generated_configs`.

---

## Files and modules

### README.md
- Purpose: repository overview, conda env setup, how to run the notebook, CLI examples, and steering tuner usage.
- Notes: README now includes an ipykernel registration command to make the `parking-rl` kernel appear in Jupyter as `python3 (parking-rl)`. Consider adding a WSL/X11/WSLg note for GUI users.

### environment.yml
- Purpose: conda environment spec for `parking-rl`.
- Key dependencies: python=3.8, numpy, matplotlib, ipykernel, ipython, pyqt>=5.15, pip (gymnasium==0.29.1).
- Notes: after creating the environment, run `python -m ipykernel install --user --name parking-rl --display-name "python3 (parking-rl)"` to register the kernel.

---

### `parking_project/main.py`
- Purpose: CLI entry points to run manual (keyboard) or random policy demos.

Main API / functions:
- `build_config(overrides: Optional[Dict]) -> Dict`: builds a demo config from `parking_gym.DEFAULT_CONFIG` and merges overrides.
- `merge_config(base: Dict, overrides: Dict) -> Dict`: recursive non-mutating merge.
- `load_config(path: Path) -> Dict`: load JSON config.
- `ManualController`: binds Matplotlib canvas keyboard events to a continuous action vector `[lon_accel, steer_accel]`.
- `manual_demo(...)` / `random_policy_demo(...)`: run episodes, interact with `ParkingEnv`.
- `parse_args()` / `main()`: CLI plumbing.

Inputs / outputs:
- Input: CLI args, optional JSON config file path.
- Output: Matplotlib Qt window rendering the environment and terminal logs.

Notes:
- GUI backend required (Qt). For headless CI or remote Linux without X server, set `QT_QPA_PLATFORM=offscreen` or forward X.
- Manual controller requires the Matplotlib figure's canvas to receive keyboard events.

---

### `parking_project/parking_gym.py`
- Purpose: Gymnasium `ParkingEnv` implementation. Handles vehicle kinematics, sensing (ray casts), obstacles, reward, termination, and rendering.

Key pieces:
- `DEFAULT_CONFIG`: canonical configuration (angles in degrees).
- `ParkingEnv`:
  - `__init__(config=None, render_mode='human')`
  - `_merge_config`, `_convert_angles_to_radians`
  - `reset(seed=None, options=None)`
  - `step(action)` -> `(observation, reward, terminated, truncated, info)`
  - `render()` / `close()`
  - Internal helpers for geometry, ray casting, collision detection (SAT), dynamic obstacles, reward calculation.

State and action:
- `action_space`: Box shape (2,) -> [lon_accel, steer_accel].
- `observation_space`: 19-dim Box including normalized x,y, cos(yaw), sin(yaw), velocity, steering angle, steering rate, relative slot pos and orientation, and 8 ray distances.

Notes:
- JSON configs must remain degree-based for angles; conversion is automatic inside the env.
- Rendering uses Matplotlib non-blocking show; in notebooks, an interactive backend (e.g., ipympl) is helpful but the notebook currently launches the demo as an external process.

---

### `parking_project/generate_training_config.py`
- Purpose: sample randomized environment/training configs and write them as JSON.

Key functions:
- `sample_training_config(seed=None) -> Dict`
- `write_config(config, path)`
- CLI: `--out` (required), `--seed` (optional)

Notes:
- Output angle fields are degrees.
- Useful for producing multiple varied training scenes.

---

### `parking_project/steering_tuner.py`
- Purpose: Qt-backed Matplotlib tuner GUI for adjusting steering assist parameters (Kp, Kd, deadband) and visualizing the steering dynamics.

Key functions:
- `compute_steering_history(kp, kd, deadband, angle0_deg, rate0_deg, commanded_accel_deg, steps, dt)`
- `load_vehicle_config(config_path)`
- `persist_vehicle_config(config_path, vehicle_cfg)`
- `launch_tuner(args)` — sets QtAgg backend, builds the UI, wires callbacks and optional JSON sync.

Notes:
- Requires Qt-compatible Matplotlib backend (PyQt/PySide).
- When `--sync` is enabled, slider changes are persisted back into the JSON file's `vehicle` section using UTF-8.

---

### `parking_project/ParkingEnv_Demo.ipynb`
- Purpose: notebook front-end that exposes common workflows as runnable cells: set parameters, generate config, launch tuning GUI, and launch CLI demos.

Important cells:
- Parameter cell (episodes, max_steps, mode, sleep_scale).
- Config generation cell (invokes `generate_training_config.py`).
- Config selection cell (points to `generated_configs/notebook_override.json`).
- Steering tuner launcher cell (constructs `conda run -n parking-rl python steering_tuner.py ...` and sets `MPLBACKEND=QtAgg`).
- CLI demo launcher cell (constructs and runs `conda run -n parking-rl python main.py ...`).

Notes:
- Notebook launches external CLI processes for GUI windows — these need a GUI-capable environment (X server or WSLg).
- The notebook itself should use the `python3 (parking-rl)` kernel for inline operations; the external commands will still run via `conda run`.

---

## Suggested next tasks (high-level)
1. Add CI-style unit tests for environment core (reset/step consistency, reward ranges).
2. Add a small example script that runs `main.random_policy_demo` headlessly and records a few episodes to JSON for unit tests.
3. Improve README with a short WSL2/WSLg / X11 troubleshooting section for GUI windows.
4. Add a `scripts/` helper to register the ipykernel automatically after env creation.
5. Convert notebook launchers to use the active kernel process instead of `conda run` where appropriate (safer inside same kernel).

---

If you want, I can open a PR with `docs/MODULE_OVERVIEW.md` added (already created locally), add unit tests, or implement any item from the suggested next tasks. Tell me which one to do next.

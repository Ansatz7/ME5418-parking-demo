"""Interactive Map Designer (Integrated Single Window).

Features:
1. **Control Panel**: Separate window with sliders to adjust parameters in real-time.
2. **Map Window**: Direct interaction on the environment view.
   - Left Click: Place Obstacle.
   - Right Click: Undo last obstacle.
   - Key 'd': Switch to Dynamic Obstacle mode (Click Start -> Click End).
   - Key 's': Switch back to Static Obstacle mode.
3. **Auto Save**: Saves config and preview image on close.

Usage:
    python -m parking_project_submission.parking_env.generate_medium_config
"""

from __future__ import annotations

import argparse
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Try to force a GUI backend that supports interactivity
try:
    matplotlib.use('Qt5Agg')
except:
    pass

from .env import DEFAULT_CONFIG, ParkingEnv
from parking_project_submission.modules.utils import write_json


class MapDesigner:
    def __init__(self, base_config: Dict, output_path: Path, initial_params: Dict):
        self.output_path = output_path
        self.params = initial_params
        
        # Data storage
        self.static_obstacles = []  # List[Dict]
        self.dynamic_obstacles = [] # List[Dict]
        
        # Modes: 'static', 'dynamic_start', 'dynamic_end'
        self.mode = 'static'
        self.temp_dynamic_start = None
        
        # 1. Initialize Environment
        # We build the initial config once
        self.config = self._build_config()
        self.env = ParkingEnv(config=self.config)
        self.env.reset()
        
        # 2. Render once to create the Figure and Axes
        self.env.render()
        
        # 3. HIJACK the environment's figure!
        if self.env.fig is None:
            raise RuntimeError("Environment failed to render figure. backend issue?")
        
        self.fig_map = self.env.fig
        self.ax_map = self.env.ax
        
        # Update Window Title
        try:
            self.fig_map.canvas.manager.set_window_title("Map Designer (Click Here to Place Obstacles)")
        except:
            pass

        # 4. Setup Event Listeners on the Map Window
        self.cid_click = self.fig_map.canvas.mpl_connect('button_press_event', self.on_map_click)
        self.cid_key = self.fig_map.canvas.mpl_connect('key_press_event', self.on_key_press)

        # 5. Create Controls Window (Separate Figure)
        self.setup_controls()

        # 6. Initial Draw
        self.render_scene()
        
        print("\n" + "="*60)
        print(" 🖱️  MAP DESIGNER INSTRUCTIONS")
        print("="*60)
        print(" [Map Window] Left Click:   Place object")
        print(" [Map Window] Right Click:  Undo last object")
        print(" [Map Window] Key 'd':      Dynamic Obstacle Mode (Click A -> Click B)")
        print(" [Map Window] Key 's':      Static Obstacle Mode")
        print(" [Controls]   Sliders:      Adjust Vehicle/Slot/Speed")
        print("="*60 + "\n")
        
        # Block until closed
        plt.ioff()
        plt.show(block=True)
        
        # Save on exit (when plt.show returns)
        self.save()

    def setup_controls(self):
        """Create the slider window."""
        self.fig_ctrl = plt.figure(figsize=(6, 8))
        self.fig_ctrl.canvas.manager.set_window_title("Controls")
        
        # Adjust layout for sliders
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95)
        
        # Define Sliders (Label, Min, Max, Initial)
        self.sliders = {}
        param_defs = [
            ('spawn_x', -15.0, 15.0),
            ('spawn_y', -15.0, 15.0),
            ('spawn_yaw', -180, 180), # 这个参数现在会生效了！
            ('slot_dx', -20.0, 20.0),
            ('slot_dy', -20.0, 20.0),
            ('slot_yaw', -180, 180),
            ('dyn_speed', 0.1, 5.0),
        ]
        
        num_sliders = len(param_defs)
        height = 0.04
        gap = 0.02
        
        for i, (key, vmin, vmax) in enumerate(param_defs):
            y_pos = 0.9 - i * (height + gap)
            ax = self.fig_ctrl.add_axes([0.3, y_pos, 0.55, height])
            
            val = self.params.get(key, 0.0)
            if key == 'dyn_speed' and val == 0: val = 0.3
            
            slider = Slider(ax, key, vmin, vmax, valinit=val)
            slider.on_changed(self.update_params)
            self.sliders[key] = slider

        # Add a Save Button
        ax_save = self.fig_ctrl.add_axes([0.3, 0.05, 0.4, 0.08])
        self.btn_save = Button(ax_save, 'Force Save')
        self.btn_save.on_clicked(lambda x: self.save())

    def update_params(self, val):
        """Callback: Update config and refresh env when slider moves."""
        for key, slider in self.sliders.items():
            self.params[key] = slider.val
        
        self.refresh_env()

    def _build_config(self) -> Dict:
        """Inject current params into a clean config."""
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        
        # 1. Spawn
        eps = 0.01
        cfg["spawn_region"] = [
            self.params['spawn_x'] - eps, self.params['spawn_x'] + eps,
            self.params['spawn_y'] - eps, self.params['spawn_y'] + eps,
        ]
        # --- 新增：写入车头朝向到配置 ---
        cfg["spawn_yaw"] = math.radians(self.params['spawn_yaw'])
        
        # 2. Slot
        cfg["parking_slot"]["relative_placement"] = True
        cfg["parking_slot"]["offset_x_range"] = (self.params['slot_dx'] - eps, self.params['slot_dx'] + eps)
        cfg["parking_slot"]["offset_y_range"] = (self.params['slot_dy'] - eps, self.params['slot_dy'] + eps)
        
        # --- 【修改这里】不要在这里转弧度，直接传度数给环境 ---
        yaw_deg = self.params['slot_yaw'] 
        cfg["parking_slot"]["orientation_range"] = (yaw_deg - eps, yaw_deg + eps)
        # 3. Disable internal random gen
        cfg["static_obstacles"]["count"] = 0
        cfg["dynamic_obstacles"]["count"] = 0
        
        # 4. Inject Manual lists
        cfg["manual_static_obstacles"] = self.static_obstacles
        cfg["manual_dynamic_obstacles"] = self.dynamic_obstacles
        
        return cfg

    def refresh_env(self):
        """Fast reload of environment state."""
        new_config = self._build_config()
        
        # Direct update of env config to avoid closing/reopening window
        self.env.config = self.env._merge_config(new_config)
        
        # Reset environment (now _spawn_vehicle will read spawn_yaw!)
        self.env.reset()
        
        self.render_scene()

    def on_key_press(self, event):
        if event.key == 'd':
            self.mode = 'dynamic_start'
            self.temp_dynamic_start = None
            self.ax_map.set_title(f"MODE: Dynamic Obstacle (Click Start Point)", color='red')
        elif event.key == 's':
            self.mode = 'static'
            self.ax_map.set_title(f"MODE: Static Obstacle", color='black')
        
        self.fig_map.canvas.draw()

    def on_map_click(self, event):
        if event.inaxes != self.env.ax:
            return

        # --- RIGHT CLICK: UNDO ---
        if event.button == 3: 
            if self.mode.startswith('dynamic') and self.dynamic_obstacles:
                self.dynamic_obstacles.pop()
                print("↩️ Removed last dynamic obstacle.")
            elif self.mode == 'static' and self.static_obstacles:
                self.static_obstacles.pop()
                print("↩️ Removed last static obstacle.")
            
            self.mode = 'static' 
            self.refresh_env()
            return

        # --- LEFT CLICK: PLACE ---
        if event.button == 1:
            x, y = float(event.xdata), float(event.ydata)
            
            if self.mode == 'static':
                self.static_obstacles.append({
                    "center": [x, y],
                    "size": [1.0, 1.0]
                })
                print(f"➕ Static Obs at ({x:.1f}, {y:.1f})")
                self.refresh_env()
                
            elif self.mode == 'dynamic_start':
                self.temp_dynamic_start = [x, y]
                self.mode = 'dynamic_end'
                self.ax_map.set_title("MODE: Dynamic (Now Click End Point)", color='blue')
                self.fig_map.canvas.draw()
                
            elif self.mode == 'dynamic_end':
                start = self.temp_dynamic_start
                end = [x, y]
                
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                heading = math.atan2(dy, dx)
                
                self.dynamic_obstacles.append({
                    "pos": start,
                    "target": end,
                    "heading": heading,
                    "speed": self.params['dyn_speed'],
                    "radius": 1.0,
                    "behavior": "goal_driven"
                })
                print(f"➕ Dynamic Obs path created.")
                
                self.mode = 'dynamic_start' 
                self.refresh_env()

    def render_scene(self):
        """Redraw the map."""
        self.ax_map.cla() 
        self.env.render() 
        
        # Draw overlays on top (redundant check for manual obs visual feedback)
        # Since env.render already draws them via config injection, 
        # we mainly just update the title here.
        
        count_s = len(self.static_obstacles)
        count_d = len(self.dynamic_obstacles)
        self.ax_map.set_title(f"Static: {count_s} | Dynamic: {count_d} | 'd': Dynamic Mode | 's': Static Mode")
        
        self.fig_map.canvas.draw()

    def save(self):
        final_config = self._build_config()
        
        # Filter map keys (Added spawn_yaw!)
        map_keys = [
            "rng_seed", "field_size", "spawn_region", "spawn_yaw", # <--- 关键：保存 spawn_yaw
            "parking_slot", "static_obstacles", "dynamic_obstacles", 
            "manual_static_obstacles", "manual_dynamic_obstacles"
        ]
        save_data = {k: final_config[k] for k in map_keys if k in final_config}
        
        write_json(self.output_path, save_data)
        print(f"\n✅ Saved map to: {self.output_path}")
        
        img_path = self.output_path.with_suffix('.png')
        try:
            self.fig_map.savefig(str(img_path), dpi=100, bbox_inches='tight')
            print(f"🖼️  Saved preview: {img_path}")
        except:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Map Designer")
    parser.add_argument("--out", type=Path, default="parking_project_submission/configs/train_medium.json")
    
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--slot-dx", type=float, default=-4.0)
    parser.add_argument("--slot-dy", type=float, default=0.0)
    parser.add_argument("--slot-d-yaw", type=float, default=0.0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    initial_params = {
        'spawn_x': args.spawn_x,
        'spawn_y': args.spawn_y,
        'spawn_yaw': args.heading,
        'slot_dx': args.slot_dx,
        'slot_dy': args.slot_dy,
        'slot_yaw': args.slot_d_yaw,
        'dyn_speed': 0.3,
    }
    
    print("Starting Map Designer... Check for TWO windows.")
    designer = MapDesigner(DEFAULT_CONFIG, args.out, initial_params)

if __name__ == "__main__":
    main()
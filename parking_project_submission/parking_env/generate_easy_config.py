"""Interactive Map Designer for Medium Difficulty.

This script launches a GUI window where you can:
1. Define Vehicle spawn pose (via CLI args).
2. Define Parking Slot relative position (via CLI args).
3. CLICK on the plot to place Static Obstacles manually.
4. Automatically adds one slow dynamic obstacle far away.

Usage:
    python -m parking_project_submission.parking_env.generate_medium_config \
        --spawn-x 0 --spawn-y 0 --heading 0 \
        --slot-dx 5.0 --slot-dy 0.0 --slot-d-yaw 0 \
        --out parking_project_submission/configs/train_medium.json
"""

from __future__ import annotations

import argparse
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

# 尝试设置后端以支持交互，WSL用户可能需要这一步
try:
    matplotlib.use('Qt5Agg')
except:
    pass

from .env import DEFAULT_CONFIG, ParkingEnv
from parking_project_submission.modules.utils import write_json


class MapDesigner:
    def __init__(self, base_config: Dict, output_path: Path):
        self.config = base_config
        self.output_path = output_path
        self.static_obstacles = []
        
        # 1. 初始化环境
        self.env = ParkingEnv(config=self.config)
        self.env.reset()
        
        # 2. 强制渲染一次以创建 env.fig 和 env.ax
        self.env.render()
        
        # 3. 【关键修复】直接使用环境的画板，而不是自己新建一个
        if self.env.fig is None:
            raise RuntimeError("Environment failed to render figure.")
            
        self.fig = self.env.fig
        self.ax = self.env.ax
        
        # 设置窗口标题
        try:
            self.fig.canvas.manager.set_window_title("Map Designer - Left Click to Add Obstacle, Close Window to Save")
        except:
            pass # 某些后端不支持设置标题，忽略
        
        # 绑定点击事件
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 初始渲染
        self.render_scene()
        
        print("\n" + "="*60)
        print(" 🖱️  地图设计器已启动 (INTERACTIVE MODE)")
        print("="*60)
        print(" • [左键点击] 地图任意位置：添加一个静态障碍物")
        print(" • [关闭窗口]：保存当前配置并退出")
        print("="*60 + "\n")
        
        # 4. 【关键修复】开启阻塞模式，防止窗口一闪而过
        # 关闭交互模式，确保 show() 会卡住程序直到窗口关闭
        plt.ioff()
        plt.show(block=True)

    def render_scene(self):
        # 调用环境渲染（它会清空重绘车辆和车位）
        self.env.render()
        
        # 在环境之上绘制我们手动添加的障碍物
        for obs in self.static_obstacles:
            center = obs["center"]
            size = obs["size"]
            # 绘制红色方块代表手动障碍物
            rect = plt.Rectangle(
                (center[0] - size[0]/2, center[1] - size[1]/2),
                size[0], size[1],
                color="red", alpha=0.6, label="Manual"
            )
            self.ax.add_patch(rect)
            # 画个叉标记中心
            self.ax.plot(center[0], center[1], 'rx')

        # 更新标题显示数量
        self.ax.set_title(f"Manual Obstacles: {len(self.static_obstacles)} | Close window to SAVE")
        plt.draw()

    def on_click(self, event):
        # 确保点击在坐标轴内
        if event.inaxes != self.ax:
            return
        
        # 在点击位置添加障碍物
        # 默认大小设为 1.0m x 1.0m
        obstacle = {
            "center": [float(event.xdata), float(event.ydata)],
            "size": [1.0, 1.0], 
        }
        self.static_obstacles.append(obstacle)
        print(f"➕ Added obstacle at ({event.xdata:.2f}, {event.ydata:.2f})")
        
        # 重新渲染
        self.render_scene()

    def save(self):
        """当窗口关闭后被调用"""
        # 将手动障碍物列表保存到配置中
        # 注意：我们需要 env.py 支持 'manual_static_obstacles' 字段
        self.config["manual_static_obstacles"] = self.static_obstacles
        
        # 将原有的随机计数设为0，避免干扰
        self.config["static_obstacles"]["count"] = 0
        
        # 只保留地图相关的 key，避免覆盖 env.py 里的车辆/奖励参数
        map_keys = [
            "rng_seed", "field_size", "spawn_region", "parking_slot", 
            "static_obstacles", "dynamic_obstacles", "manual_static_obstacles"
        ]
        final_config = {k: self.config[k] for k in map_keys if k in self.config}
        
        write_json(self.output_path, final_config)
        print(f"\n✅ Configuration saved to: {self.output_path}")
        print(f"   Contains {len(self.static_obstacles)} manual obstacles.")


def create_base_config(args) -> Dict:
    """Construct the deterministic base map configuration."""
    # 使用 deepcopy 避免修改全局配置
    config = deepcopy(DEFAULT_CONFIG)
    
    # 1. 车辆出生点 (绝对坐标)
    # 通过将范围缩小到一个极小值来固定位置
    epsilon = 0.01
    config["spawn_region"] = [
        args.spawn_x - epsilon, args.spawn_x + epsilon,
        args.spawn_y - epsilon, args.spawn_y + epsilon
    ]
    
    # 2. 车位位置 (相对于车辆)
    config["parking_slot"]["relative_placement"] = True
    
    # X: 前后距离
    config["parking_slot"]["offset_x_range"] = (args.slot_dx - epsilon, args.slot_dx + epsilon)
    # Y: 左右距离
    config["parking_slot"]["offset_y_range"] = (args.slot_dy - epsilon, args.slot_dy + epsilon)
    # Angle: 相对角度
    yaw_rad = math.radians(args.slot_d_yaw)
    config["parking_slot"]["orientation_range"] = (yaw_rad - 0.01, yaw_rad + 0.01)

    # 3. 动态障碍物 (保留一个，放远点，龟速)
    config["dynamic_obstacles"]["count"] = 1
    config["dynamic_obstacles"]["radius"] = 1.0
    config["dynamic_obstacles"]["speed_range"] = (0.1, 0.3)
    config["dynamic_obstacles"]["min_distance"] = 8.0 
    
    return config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Map Designer")
    parser.add_argument("--out", type=Path, default="parking_project_submission/configs/train_medium.json")
    
    # 出生点参数 (Absolute)
    parser.add_argument("--spawn-x", type=float, default=0.0, help="Vehicle spawn X")
    parser.add_argument("--spawn-y", type=float, default=0.0, help="Vehicle spawn Y")
    
    # 车位参数 (Relative to Vehicle)
    parser.add_argument("--slot-dx", type=float, default=-4.0, help="Slot X distance (Front+, Back-)")
    parser.add_argument("--slot-dy", type=float, default=0.0, help="Slot Y distance (Left+, Right-)")
    parser.add_argument("--slot-d-yaw", type=float, default=0.0, help="Slot relative angle (degrees)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    config = create_base_config(args)
    
    # 启动设计器
    designer = MapDesigner(config, args.out)
    
    # 当窗口关闭，plt.show() 返回后，保存数据
    designer.save()

if __name__ == "__main__":
    main()
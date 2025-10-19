#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新的路线可视化方案
- 车辆路线分行显示，每个路线放在方框里
- 节点用方框表示，显示节点名字
- 无人机任务用箭头从节点方框指向回收点
- 不考虑二维坐标，只做直观效果
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import Dict, List, Tuple, Any
import os

class RouteVisualizer:
    """路线可视化器 - 新的直观设计"""
    
    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
        self.colors = {
            'vehicle': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'drone': ['#A8E6CF', '#FFD93D', '#6BCF7F', '#4D96FF', '#FF8A80'],
            'node': '#E8E8E8',
            'text': '#2C3E50'
        }
    
    def visualize_routes(self, state, title: str = "车辆路线与无人机任务", 
                        save_path: str = None):
        """
        可视化车辆路线和无人机任务
        
        Args:
            state: 状态对象或字典，包含车辆路线和无人机任务
            title: 图表标题
            save_path: 保存路径
        """
        # 创建图形 - 使用更大的画布
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(0, 16)  # 进一步增加宽度
        ax.set_ylim(0, 16)  # 进一步增加高度
        ax.axis('off')
        
        # 获取车辆路线和无人机任务
        if hasattr(state, 'vehicle_routes'):
            # 如果是对象，使用属性访问
            vehicle_routes = state.vehicle_routes
            # 从customer_plan中提取无人机任务
            drone_tasks = self._extract_drone_tasks_from_customer_plan(state.customer_plan)
            print(f"调试信息: 车辆路线数量: {len(vehicle_routes)}")
            print(f"调试信息: 无人机任务数量: {len(drone_tasks)}")
            if drone_tasks:
                print(f"调试信息: 无人机任务详情: {drone_tasks}")
        else:
            # 如果是字典，使用get方法
            vehicle_routes = state.get('vehicle_routes', [])
            drone_tasks = state.get('drone_tasks', [])
        
        # 绘制车辆路线
        self._draw_vehicle_routes(ax, vehicle_routes)
        
        # 绘制无人机任务
        self._draw_drone_tasks(ax, vehicle_routes, drone_tasks)
        
        # 设置标题
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        # 添加图例
        self._add_legend(ax)
        
        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        # 调整布局，避免tight_layout警告
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()
    
    def _draw_vehicle_routes(self, ax, vehicle_routes: List[List[int]]):
        """绘制车辆路线"""
        num_vehicles = len(vehicle_routes)
        
        # 计算每行的高度
        row_height = 8.0 / max(num_vehicles, 1)
        
        for i, route in enumerate(vehicle_routes):
            y_pos = 8.5 - (i + 0.5) * row_height
            
            # 绘制车辆标题
            vehicle_color = self.colors['vehicle'][i % len(self.colors['vehicle'])]
            ax.text(0.5, y_pos, f"车辆 {i+1}", fontsize=14, fontweight='bold', 
                   color=vehicle_color, ha='left', va='center')
            
            # 绘制路线方框
            self._draw_route_boxes(ax, route, y_pos, vehicle_color)
    
    def _draw_route_boxes(self, ax, route: List[int], y_pos: float, color: str):
        """绘制单个路线的节点方框（支持换行）"""
        if not route:
            return
        
        # 计算方框位置
        box_width = 0.7
        box_height = 0.5
        spacing = 0.1
        
        # 计算每行能容纳的节点数
        available_width = 7.0  # 从x=2.0到x=9.0
        nodes_per_row = int(available_width / (box_width + spacing))
        
        start_x = 2.0
        start_y = y_pos
        
        for j, node_id in enumerate(route):
            # 计算当前节点应该在哪一行
            row = j // nodes_per_row
            col = j % nodes_per_row
            
            # 计算位置
            x_pos = start_x + col * (box_width + spacing)
            y_pos = start_y - row * (box_height + 0.2)  # 向下换行
            
            # 绘制节点方框
            box = FancyBboxPatch(
                (x_pos, y_pos - box_height/2), box_width, box_height,
                boxstyle="round,pad=0.05",
                facecolor='white',
                edgecolor=color,
                linewidth=2
            )
            ax.add_patch(box)
            
            # 添加节点ID
            ax.text(x_pos + box_width/2, y_pos, str(node_id), 
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   color=color)
            
            # 如果不是最后一个节点，绘制连接线
            if j < len(route) - 1:
                next_row = (j + 1) // nodes_per_row
                next_col = (j + 1) % nodes_per_row
                
                if row == next_row:  # 同一行
                    next_x = start_x + next_col * (box_width + spacing)
                    next_y = start_y - next_row * (box_height + 0.2)
                    
                    # 绘制连接线
                    ax.plot([x_pos + box_width, next_x], [y_pos, next_y], 
                           color=color, linewidth=2, alpha=0.7)
                else:  # 换行，不绘制连接线
                    pass
    
    def _draw_drone_tasks(self, ax, vehicle_routes: List[List[int]], drone_tasks: List[Dict]):
        """绘制无人机任务箭头"""
        if not drone_tasks:
            return
        
        # 计算车辆路线信息
        route_info = self._get_route_info(vehicle_routes)
        
        # 按无人机ID分组任务
        drone_groups = {}
        for task in drone_tasks:
            drone_id = task.get('drone_id', 1)
            if drone_id not in drone_groups:
                drone_groups[drone_id] = []
            drone_groups[drone_id].append(task)
        
        # 为每个无人机分配不同的颜色
        drone_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#A8E6CF', '#FFD93D', '#6BCF7F']
        
        # 在右侧创建任务列表区域
        task_list_x = 12.5
        task_list_y_start = 14
        
        for drone_idx, (drone_id, tasks) in enumerate(drone_groups.items()):
            drone_color = drone_colors[drone_idx % len(drone_colors)]
            
            # 绘制无人机标题
            ax.text(task_list_x, task_list_y_start - drone_idx * 1.8, 
                   f"无人机 {drone_id}", fontsize=12, fontweight='bold', 
                   color=drone_color, ha='left', va='center')
            
            # 绘制该无人机的所有任务
            for task_idx, task in enumerate(tasks):
                launch_node = task.get('launch_node')
                customer_node = task.get('customer_node')
                recovery_node = task.get('recovery_node')
                launch_vehicle = task.get('launch_vehicle', '?')
                recovery_vehicle = task.get('recovery_vehicle', '?')
                customer_id = task.get('customer_id', customer_node)
                
                # 绘制任务箭头（如果节点存在）
                if all([launch_node is not None, customer_node is not None, recovery_node is not None]):
                    launch_pos = self._get_node_position(route_info, launch_node)
                    recovery_pos = self._get_node_position(route_info, recovery_node)
                    
                    if launch_pos and recovery_pos:
                        # 创建箭头
                        arrow = ConnectionPatch(
                            launch_pos, recovery_pos, "data", "data",
                            arrowstyle="->", shrinkA=8, shrinkB=8,
                            mutation_scale=15, fc=drone_color, ec=drone_color,
                            linewidth=2, alpha=0.6
                        )
                        ax.add_patch(arrow)
                        
                        # 在箭头上添加小标签
                        mid_x = (launch_pos[0] + recovery_pos[0]) / 2
                        mid_y = (launch_pos[1] + recovery_pos[1]) / 2
                        
                        ax.text(mid_x, mid_y, f"U{drone_id}", 
                               fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=drone_color, alpha=0.8),
                               color='white', fontweight='bold')
                
                # 在右侧任务列表中显示详细信息
                task_text = f"  客户{customer_id}: {launch_node}→{customer_node}→{recovery_node}"
                if launch_vehicle != recovery_vehicle:
                    task_text += f" (异车: V{launch_vehicle}→V{recovery_vehicle})"
                else:
                    task_text += f" (同车: V{launch_vehicle})"
                
                ax.text(task_list_x, task_list_y_start - drone_idx * 1.8 - 0.4 - task_idx * 0.3, 
                       task_text, fontsize=9, ha='left', va='center',
                       color='#2C3E50')
    
    def _get_route_info(self, vehicle_routes: List[List[int]]) -> Dict[int, Tuple[float, float]]:
        """获取每个节点在路线中的位置信息（支持换行）"""
        route_info = {}
        num_vehicles = len(vehicle_routes)
        row_height = 7.0 / max(num_vehicles, 1)
        
        for i, route in enumerate(vehicle_routes):
            y_pos = 8.5 - (i + 0.5) * row_height
            
            # 计算方框位置（与_draw_route_boxes保持一致）
            box_width = 0.7
            box_height = 0.5
            spacing = 0.1
            
            # 计算每行能容纳的节点数
            available_width = 7.0  # 从x=2.0到x=9.0
            nodes_per_row = int(available_width / (box_width + spacing))
            
            start_x = 2.0
            start_y = y_pos
            
            for j, node_id in enumerate(route):
                # 计算当前节点应该在哪一行
                row = j // nodes_per_row
                col = j % nodes_per_row
                
                # 计算位置
                x_pos = start_x + col * (box_width + spacing) + box_width/2
                y_pos = start_y - row * (box_height + 0.2)  # 向下换行
                
                route_info[node_id] = (x_pos, y_pos)
        
        return route_info
    
    def _get_node_position(self, route_info: Dict[int, Tuple[float, float]], node_id: int) -> Tuple[float, float]:
        """获取节点位置"""
        return route_info.get(node_id, None)
    
    def _extract_drone_tasks_from_customer_plan(self, customer_plan):
        """从customer_plan中提取无人机任务信息"""
        drone_tasks = []
        
        if not customer_plan:
            return drone_tasks
        
        # 遍历customer_plan，提取任务信息
        # 格式：{customer_id: (drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle)}
        for customer_id, task_info in customer_plan.items():
            if isinstance(task_info, (list, tuple)) and len(task_info) >= 6:
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = task_info[:6]
                drone_task = {
                    'drone_id': drone_id,
                    'launch_node': launch_node,
                    'customer_node': customer_node,
                    'recovery_node': recovery_node,
                    'launch_vehicle': launch_vehicle,
                    'recovery_vehicle': recovery_vehicle,
                    'customer_id': customer_id
                }
                drone_tasks.append(drone_task)
        
        return drone_tasks
    
    def _add_legend(self, ax):
        """添加图例"""
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#FF6B6B', linewidth=2, label='车辆路线节点'),
            plt.Line2D([0], [0], color='#A8E6CF', linewidth=2, label='无人机任务箭头'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#A8E6CF', alpha=0.8, label='无人机任务标签'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
                 frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.02, 0.98))
    
    def analyze_routes(self, state):
        """分析路线信息"""
        if hasattr(state, 'vehicle_routes'):
            # 如果是对象，使用属性访问
            vehicle_routes = state.vehicle_routes
            drone_tasks = self._extract_drone_tasks_from_customer_plan(state.customer_plan)
        else:
            # 如果是字典，使用get方法
            vehicle_routes = state.get('vehicle_routes', [])
            drone_tasks = state.get('drone_tasks', [])
        
        print("路线分析:")
        print("="*50)
        
        print(f"车辆数量: {len(vehicle_routes)}")
        for i, route in enumerate(vehicle_routes):
            print(f"车辆 {i+1}: {' → '.join(map(str, route))}")
        
        print(f"\n无人机任务数量: {len(drone_tasks)}")
        for i, task in enumerate(drone_tasks):
            drone_id = task.get('drone_id', i + 1)
            launch = task.get('launch_node', '?')
            customer = task.get('customer_node', '?')
            recovery = task.get('recovery_node', '?')
            print(f"无人机 {drone_id}: {launch} → {customer} → {recovery}")


def create_sample_route_state():
    """创建示例路线状态"""
    return {
        'vehicle_routes': [
            [0, 1, 2, 3],  # 车辆1路线
            [0, 4, 5, 6],  # 车辆2路线
            [0, 7, 8, 9]   # 车辆3路线
        ],
        'drone_tasks': [
            {'drone_id': 1, 'launch_node': 1, 'customer_node': 10, 'recovery_node': 2},
            {'drone_id': 2, 'launch_node': 4, 'customer_node': 11, 'recovery_node': 5},
            {'drone_id': 3, 'launch_node': 7, 'customer_node': 12, 'recovery_node': 8},
            {'drone_id': 4, 'launch_node': 2, 'customer_node': 13, 'recovery_node': 3}
        ]
    }


if __name__ == "__main__":
    # 创建示例状态
    state = create_sample_route_state()
    
    # 创建可视化器
    visualizer = RouteVisualizer(figsize=(16, 10))
    
    # 设置保存路径
    save_path = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\map_test\route_visualization.png"
    
    # 进行可视化
    visualizer.visualize_routes(state, "车辆路线与无人机任务可视化", save_path)
    
    # 分析路线
    visualizer.analyze_routes(state)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆路线和无人机任务可视化工具

该模块提供可视化功能，用于检查车辆路线和无人机任务的整体流程。
支持显示：
1. 车辆路线（地面路径）
2. 无人机任务（空中路径）
3. 节点信息（发射点、客户点、回收点）
4. 无人机ID、发射车辆、回收车辆信息
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VDACVisualizer:
    """VDAC车辆路线和无人机任务可视化器"""
    
    def __init__(self, figsize=(15, 12)):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.colors = {
            'depot': '#FF6B6B',      # 红色 - 仓库
            'customer': '#4ECDC4',   # 青色 - 客户点
            'vtp': '#45B7D1',        # 蓝色 - VTP节点
            'vehicle_route': '#96CEB4',  # 绿色 - 车辆路线
            'uav_route': '#FFEAA7',  # 黄色 - 无人机路线
            'launch': '#DDA0DD',     # 紫色 - 发射点
            'recovery': '#98D8C8',   # 薄荷绿 - 回收点
        }
        
        # 车辆路线颜色（不同车辆使用不同颜色）
        self.vehicle_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        # 无人机颜色
        self.uav_colors = [
            '#FF1744', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5',
            '#2196F3', '#03A9F4', '#00BCD4', '#009688', '#4CAF50'
        ]
    
    def visualize_solution(self, vehicle_routes: List[List[int]], 
                          customer_plan: Dict[int, Tuple], 
                          uav_assignments: Dict[int, List[Tuple]],
                          node_info: Optional[Dict[int, Dict]] = None,
                          title: str = "VDAC车辆路线和无人机任务可视化",
                          save_path: str = None):
        """
        可视化完整的解决方案
        
        Args:
            vehicle_routes: 车辆路线列表，每个元素是一个车辆的路由（节点ID列表）
            customer_plan: 客户计划字典，{customer_id: (drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle)}
            uav_assignments: 无人机分配字典，{drone_id: [(drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle), ...]}
            node_info: 节点信息字典，{node_id: {'x': x, 'y': y, 'type': type, ...}}
            title: 图形标题
            save_path: 保存路径，如果提供则保存图片
        """
        # 如果没有提供节点信息，生成默认的节点坐标
        if node_info is None:
            node_info = self._generate_default_node_info(vehicle_routes, customer_plan)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制节点
        self._draw_nodes(ax, node_info)
        
        # 绘制车辆路线
        self._draw_vehicle_routes(ax, vehicle_routes, node_info)
        
        # 绘制无人机任务
        self._draw_uav_missions(ax, customer_plan, uav_assignments, node_info)
        
        # 设置图形属性
        self._setup_plot(ax, title, node_info)
        
        # 不添加图例，保持图表简洁
        
        plt.tight_layout()
        
        # 保存图片（如果提供了路径）
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
        
        # 打印详细信息
        self._print_detailed_info(vehicle_routes, customer_plan, uav_assignments)
    
    def _generate_default_node_info(self, vehicle_routes: List[List[int]], 
                                   customer_plan: Dict[int, Tuple]) -> Dict[int, Dict]:
        """
        生成默认的节点信息（如果没有提供真实坐标）
        
        Args:
            vehicle_routes: 车辆路线列表
            customer_plan: 客户计划字典
            
        Returns:
            节点信息字典
        """
        # 收集所有节点
        all_nodes = set()
        for route in vehicle_routes:
            all_nodes.update(route)
        all_nodes.update(customer_plan.keys())
        
        # 为每个assignment中的节点添加
        for assignment in customer_plan.values():
            if len(assignment) >= 4:
                all_nodes.update([assignment[1], assignment[2], assignment[3]])  # launch_node, customer_node, recovery_node
        
        # 生成坐标
        node_info = {}
        nodes_list = list(all_nodes)
        
        # 使用圆形布局生成坐标
        n_nodes = len(nodes_list)
        if n_nodes > 0:
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            radius = 10
            
            for i, node_id in enumerate(nodes_list):
                x = radius * np.cos(angles[i])
                y = radius * np.sin(angles[i])
                
                # 确定节点类型和名称
                node_type = 'customer'
                if i == 0:  # 假设第一个节点是仓库
                    node_type = 'depot'
                    node_name = f'仓库-{node_id}'
                elif node_id in [assignment[1] for assignment in customer_plan.values()]:  # 发射节点
                    node_type = 'vtp'
                    node_name = f'发射点-{node_id}'
                elif node_id in [assignment[3] for assignment in customer_plan.values()]:  # 回收节点
                    node_type = 'vtp'
                    node_name = f'回收点-{node_id}'
                elif node_id in customer_plan.keys():  # 客户节点
                    node_type = 'customer'
                    node_name = f'客户-{node_id}'
                else:
                    node_name = f'节点-{node_id}'
                
                node_info[node_id] = {
                    'x': x,
                    'y': y,
                    'type': node_type,
                    'label': node_name,
                    'name': node_name
                }
        
        return node_info
    
    def _draw_nodes(self, ax, node_info: Dict[int, Dict]):
        """
        绘制节点，显示节点ID和类型信息
        
        Args:
            ax: matplotlib轴对象
            node_info: 节点信息字典
        """
        for node_id, info in node_info.items():
            x, y = info['x'], info['y']
            node_type = info.get('type', 'customer')
            node_name = info.get('name', f'Node {node_id}')
            
            # 根据节点类型选择颜色和形状
            if node_type == 'depot':
                color = self.colors['depot']
                size = 500  # 增大仓库节点
                marker = 's'  # 方形
                label = f'仓库\n{node_id}'
            elif node_type == 'vtp':
                color = self.colors['vtp']
                size = 400  # 增大VTP节点
                marker = '^'  # 三角形
                label = f'VTP\n{node_id}'
            else:  # customer
                color = self.colors['customer']
                size = 300  # 增大客户点
                marker = 'o'  # 圆形
                label = f'客户\n{node_id}'
            
            # 绘制节点
            ax.scatter(x, y, c=color, s=size, marker=marker, 
                      edgecolors='black', linewidth=2, alpha=0.8, zorder=5)
            
            # 添加节点标签，显示类型和ID
            ax.annotate(label, (x, y), 
                       xytext=(0, 0), textcoords='offset points', 
                       fontsize=10, fontweight='bold', ha='center', va='center',
                       color='white', zorder=6)
            
            # 在节点旁边添加坐标信息
            ax.annotate(f'({x:.1f}, {y:.1f})', (x, y), 
                       xytext=(25, 25), textcoords='offset points', 
                       fontsize=8, color='black', zorder=6,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    def _draw_vehicle_routes(self, ax, vehicle_routes: List[List[int]], 
                           node_info: Dict[int, Dict]):
        """
        绘制车辆路线，显示清晰的标注
        
        Args:
            ax: matplotlib轴对象
            vehicle_routes: 车辆路线列表
            node_info: 节点信息字典
        """
        for vehicle_id, route in enumerate(vehicle_routes):
            if not route:
                continue
                
            color = self.vehicle_colors[vehicle_id % len(self.vehicle_colors)]
            
            # 绘制路线
            for i in range(len(route) - 1):
                start_node = route[i]
                end_node = route[i + 1]
                
                if start_node in node_info and end_node in node_info:
                    start_x, start_y = node_info[start_node]['x'], node_info[start_node]['y']
                    end_x, end_y = node_info[end_node]['x'], node_info[end_node]['y']
                    
                    # 绘制箭头
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', color=color, 
                                             lw=3, alpha=0.8))
                    
                    # 在箭头中点添加路线段标注
                    mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
                    ax.annotate(f'{start_node}→{end_node}', (mid_x, mid_y), 
                               fontsize=9, color=color, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color))
            
            # 在路线起点添加车辆总览信息
            if route and route[0] in node_info:
                start_x, start_y = node_info[route[0]]['x'], node_info[route[0]]['y']
                route_str = ' → '.join(map(str, route))
                ax.annotate(f'车辆{vehicle_id + 1}: {route_str}', 
                           (start_x, start_y), xytext=(30, 30), 
                           textcoords='offset points', fontsize=9, 
                           color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
    
    def _draw_uav_missions(self, ax, customer_plan: Dict[int, Tuple], 
                          uav_assignments: Dict[int, List[Tuple]], 
                          node_info: Dict[int, Dict]):
        """
        绘制无人机任务
        
        Args:
            ax: matplotlib轴对象
            customer_plan: 客户计划字典
            uav_assignments: 无人机分配字典
            node_info: 节点信息字典
        """
        for customer_id, assignment in customer_plan.items():
            if len(assignment) < 6:
                continue
                
            drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
            
            # 获取无人机颜色
            uav_color = self.uav_colors[drone_id % len(self.uav_colors)]
            
            # 绘制发射到客户的路径
            if launch_node in node_info and customer_node in node_info:
                launch_x, launch_y = node_info[launch_node]['x'], node_info[launch_node]['y']
                customer_x, customer_y = node_info[customer_node]['x'], node_info[customer_node]['y']
                
                # 绘制虚线箭头（空中路径）
                ax.annotate('', xy=(customer_x, customer_y), xytext=(launch_x, launch_y),
                           arrowprops=dict(arrowstyle='->', color=uav_color, 
                                         lw=2, alpha=0.8, linestyle='--'))
                
                # 添加无人机发射路径标签
                mid_x, mid_y = (launch_x + customer_x) / 2, (launch_y + customer_y) / 2
                ax.annotate(f'U{drone_id}发射\n{launch_node}→{customer_node}', (mid_x, mid_y), 
                           fontsize=9, color=uav_color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=uav_color))
            
            # 绘制客户到回收的路径
            if customer_node in node_info and recovery_node in node_info:
                customer_x, customer_y = node_info[customer_node]['x'], node_info[customer_node]['y']
                recovery_x, recovery_y = node_info[recovery_node]['x'], node_info[recovery_node]['y']
                
                # 绘制虚线箭头（空中路径）
                ax.annotate('', xy=(recovery_x, recovery_y), xytext=(customer_x, customer_y),
                           arrowprops=dict(arrowstyle='->', color=uav_color, 
                                         lw=2, alpha=0.8, linestyle='--'))
                
                # 添加无人机回收路径标签
                mid_x, mid_y = (customer_x + recovery_x) / 2, (customer_y + recovery_y) / 2
                ax.annotate(f'U{drone_id}回收\n{customer_node}→{recovery_node}', (mid_x, mid_y), 
                           fontsize=9, color=uav_color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=uav_color))
    
    def _setup_plot(self, ax, title: str, node_info: Dict[int, Dict]):
        """
        设置图形属性
        
        Args:
            ax: matplotlib轴对象
            title: 图形标题
            node_info: 节点信息字典
        """
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X坐标', fontsize=16)
        ax.set_ylabel('Y坐标', fontsize=16)
        
        # 设置坐标轴范围
        if node_info:
            x_coords = [info['x'] for info in node_info.values()]
            y_coords = [info['y'] for info in node_info.values()]
            
            x_margin = (max(x_coords) - min(x_coords)) * 0.1
            y_margin = (max(y_coords) - min(y_coords)) * 0.1
            
            ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
            ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _add_legend(self, ax, vehicle_routes: List[List[int]]):
        """
        添加详细图例
        
        Args:
            ax: matplotlib轴对象
            vehicle_routes: 车辆路线列表
        """
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['depot'], 
                      markersize=15, label='仓库 (方形)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=self.colors['vtp'], 
                      markersize=15, label='VTP节点 (三角形)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['customer'], 
                      markersize=15, label='客户点 (圆形)'),
        ]
        
        # 添加车辆路线图例
        for i, route in enumerate(vehicle_routes):
            if route:
                color = self.vehicle_colors[i % len(self.vehicle_colors)]
                legend_elements.append(
                    plt.Line2D([0], [0], color=color, lw=4, 
                              label=f'车辆{i+1}路线')
                )
        
        # 添加无人机路线图例
        legend_elements.append(
            plt.Line2D([0], [0], color=self.colors['uav_route'], lw=4, linestyle='--', 
                      label='无人机路线 (虚线)')
        )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
    
    def _print_detailed_info(self, vehicle_routes: List[List[int]], 
                           customer_plan: Dict[int, Tuple], 
                           uav_assignments: Dict[int, List[Tuple]]):
        """
        打印详细信息
        
        Args:
            vehicle_routes: 车辆路线列表
            customer_plan: 客户计划字典
            uav_assignments: 无人机分配字典
        """
        print("\n" + "="*80)
        print("详细任务信息")
        print("="*80)
        
        # 打印车辆路线信息
        print(f"\n车辆路线信息 (共{len(vehicle_routes)}辆车):")
        for vehicle_id, route in enumerate(vehicle_routes):
            print(f"  车辆 {vehicle_id + 1}: {' -> '.join(map(str, route))}")
        
        # 打印客户任务信息
        print(f"\n客户任务信息 (共{len(customer_plan)}个客户):")
        for customer_id, assignment in customer_plan.items():
            if len(assignment) >= 6:
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                print(f"  客户 {customer_id}:")
                print(f"    无人机ID: {drone_id}")
                print(f"    发射节点: {launch_node} (车辆 {launch_vehicle})")
                print(f"    客户节点: {customer_node}")
                print(f"    回收节点: {recovery_node} (车辆 {recovery_vehicle})")
                print(f"    任务类型: {'同车任务' if launch_vehicle == recovery_vehicle else '异车任务'}")
        
        # 打印无人机分配信息
        print(f"\n无人机分配信息:")
        for drone_id, assignments in uav_assignments.items():
            print(f"  无人机 {drone_id}: {len(assignments)}个任务")
            for i, assignment in enumerate(assignments):
                if len(assignment) >= 6:
                    _, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                    print(f"    任务 {i+1}: 从车辆{launch_vehicle}的节点{launch_node}发射 -> 服务客户{customer_node} -> 被车辆{recovery_vehicle}的节点{recovery_node}回收")
        
        print("="*80)


def create_sample_data():
    """
    创建示例数据用于测试可视化功能
    
    Returns:
        tuple: (vehicle_routes, customer_plan, uav_assignments)
    """
    # 示例车辆路线
    vehicle_routes = [
        [1, 2, 3, 4, 5],  # 车辆1的路线
        [1, 6, 7, 8, 9],  # 车辆2的路线
        [1, 10, 11, 12]   # 车辆3的路线
    ]
    
    # 示例客户计划
    customer_plan = {
        2: (1, 2, 2, 3, 1, 1),    # 无人机1，从车辆1的节点2发射，服务客户2，被车辆1的节点3回收
        3: (2, 3, 3, 4, 1, 1),    # 无人机2，从车辆1的节点3发射，服务客户3，被车辆1的节点4回收
        6: (1, 6, 6, 7, 2, 2),    # 无人机1，从车辆2的节点6发射，服务客户6，被车辆2的节点7回收
        7: (3, 7, 7, 8, 2, 2),    # 无人机3，从车辆2的节点7发射，服务客户7，被车辆2的节点8回收
        10: (2, 10, 10, 11, 3, 3), # 无人机2，从车辆3的节点10发射，服务客户10，被车辆3的节点11回收
        11: (1, 11, 11, 12, 3, 1), # 无人机1，从车辆3的节点11发射，服务客户11，被车辆1的节点12回收（异车任务）
    }
    
    # 示例无人机分配
    uav_assignments = {
        1: [
            (1, 2, 2, 3, 1, 1),   # 同车任务
            (1, 6, 6, 7, 2, 2),   # 同车任务
            (1, 11, 11, 12, 3, 1) # 异车任务
        ],
        2: [
            (2, 3, 3, 4, 1, 1),   # 同车任务
            (2, 10, 10, 11, 3, 3) # 同车任务
        ],
        3: [
            (3, 7, 7, 8, 2, 2)    # 同车任务
        ]
    }
    
    return vehicle_routes, customer_plan, uav_assignments


def main():
    """
    主函数 - 演示可视化功能
    """
    print("VDAC车辆路线和无人机任务可视化工具")
    print("="*50)
    
    # 创建可视化器
    visualizer = VDACVisualizer()
    
    # 创建示例数据
    vehicle_routes, customer_plan, uav_assignments = create_sample_data()
    
    # 进行可视化
    visualizer.visualize_solution(
        vehicle_routes=vehicle_routes,
        customer_plan=customer_plan,
        uav_assignments=uav_assignments,
        title="VDAC示例：车辆路线和无人机任务"
    )


if __name__ == "__main__":
    main()

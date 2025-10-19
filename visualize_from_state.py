#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从FastMfstspState对象直接可视化

这个工具可以从您的FastMfstspState对象中直接提取数据并进行可视化，
无需手动输入数据格式。
"""

from visualization import VDACVisualizer
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class StateVisualizer:
    """从状态对象直接可视化的工具类"""
    
    def __init__(self, figsize=(15, 12)):
        """
        初始化状态可视化器
        
        Args:
            figsize: 图形大小
        """
        self.visualizer = VDACVisualizer(figsize)
    
    def visualize_state(self, state, title: str = "VDAC状态可视化", save_path: str = None):
        """
        从状态对象直接可视化
        
        Args:
            state: 状态对象（通常是FastMfstspState对象）
            title: 图形标题
            save_path: 保存路径，如果提供则保存图片
        """
        # 检查输入类型
        if hasattr(state, 'dtype') and 'numpy' in str(type(state)):
            raise ValueError(f"检测到numpy数值类型 ({type(state)})，请传入完整的状态对象")
        
        # 检查必要的属性
        required_attrs = ['vehicle_routes', 'customer_plan', 'uav_assignments']
        missing_attrs = [attr for attr in required_attrs if not hasattr(state, attr)]
        if missing_attrs:
            raise ValueError(f"状态对象缺少必要属性: {missing_attrs}")
        
        # 提取数据
        vehicle_routes = state.vehicle_routes
        customer_plan = state.customer_plan
        uav_assignments = state.uav_assignments
        
        # 生成节点信息（如果有node信息的话）
        node_info = None
        if hasattr(state, 'node') and state.node:
            node_info = self._extract_node_info(state.node)
        
        # 进行可视化
        self.visualizer.visualize_solution(
            vehicle_routes=vehicle_routes,
            customer_plan=customer_plan,
            uav_assignments=uav_assignments,
            node_info=node_info,
            title=title,
            save_path=save_path
        )
    
    def _extract_node_info(self, node_dict: Dict) -> Dict[int, Dict]:
        """
        从node字典中提取节点信息，支持node[node].position格式
        
        Args:
            node_dict: 节点字典
            
        Returns:
            节点信息字典
        """
        node_info = {}
        
        for node_id, node_obj in node_dict.items():
            # 优先从position属性获取坐标
            x, y = 0, 0
            
            # 尝试从position属性获取坐标（支持node[node].position格式）
            if hasattr(node_obj, 'position'):
                position = node_obj.position
                if isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 2:
                    x, y = float(position[0]), float(position[1])
                elif hasattr(position, 'x') and hasattr(position, 'y'):
                    x, y = float(position.x), float(position.y)
            
            # 如果没有position属性，尝试其他属性
            if x == 0 and y == 0:
                x = getattr(node_obj, 'x', 0)
                y = getattr(node_obj, 'y', 0)
            
            # 如果还是没有，尝试其他可能的属性名
            if x == 0 and y == 0:
                x = getattr(node_obj, 'latDeg', 0)
                y = getattr(node_obj, 'lonDeg', 0)
            
            # 如果还是没有，尝试map_position
            if x == 0 and y == 0 and hasattr(node_obj, 'map_position'):
                map_pos = node_obj.map_position
                if isinstance(map_pos, (list, tuple, np.ndarray)) and len(map_pos) >= 2:
                    x, y = float(map_pos[0]), float(map_pos[1])
            
            # 确定节点类型
            node_type = 'customer'
            if hasattr(node_obj, 'nodeType'):
                if node_obj.nodeType == 'DEPOT':
                    node_type = 'depot'
                elif 'VTP' in str(node_obj.nodeType):
                    node_type = 'vtp'
            elif hasattr(node_obj, 'type'):
                if node_obj.type == 'DEPOT':
                    node_type = 'depot'
                elif 'VTP' in str(node_obj.type):
                    node_type = 'vtp'
            
            # 生成标签
            label = f'Node {node_id}'
            if hasattr(node_obj, 'label'):
                label = node_obj.label
            elif hasattr(node_obj, 'name'):
                label = node_obj.name
            
            # 添加坐标信息到标签中
            if x != 0 or y != 0:
                label += f' ({x:.2f}, {y:.2f})'
            
            node_info[node_id] = {
                'x': x,
                'y': y,
                'type': node_type,
                'label': label,
                'original_position': getattr(node_obj, 'position', None)
            }
        
        return node_info
    
    def visualize_comparison(self, state1, state2, 
                           title1: str = "状态1", title2: str = "状态2"):
        """
        比较两个状态的可视化
        
        Args:
            state1: 第一个状态对象
            state2: 第二个状态对象
            title1: 第一个状态的标题
            title2: 第二个状态的标题
        """
        import matplotlib.pyplot as plt
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 可视化第一个状态
        plt.sca(ax1)
        self.visualize_state(state1, title1)
        
        # 可视化第二个状态
        plt.sca(ax2)
        self.visualize_state(state2, title2)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_state(self, state):
        """
        分析状态并打印统计信息
        
        Args:
            state: 状态对象
        """
        print("\n" + "="*80)
        print("状态分析报告")
        print("="*80)
        
        # 基本统计
        num_vehicles = len(state.vehicle_routes)
        num_customers = len(state.customer_plan)
        num_uavs = len(state.uav_assignments)
        
        print(f"车辆数量: {num_vehicles}")
        print(f"客户数量: {num_customers}")
        print(f"无人机数量: {num_uavs}")
        
        # 车辆路线统计
        print(f"\n车辆路线统计:")
        for i, route in enumerate(state.vehicle_routes):
            print(f"  车辆 {i+1}: {len(route)}个节点, 路线: {' -> '.join(map(str, route))}")
        
        # 无人机任务统计
        print(f"\n无人机任务统计:")
        for drone_id, assignments in state.uav_assignments.items():
            print(f"  无人机 {drone_id}: {len(assignments)}个任务")
            
            # 统计同车任务和异车任务
            same_vehicle_tasks = 0
            diff_vehicle_tasks = 0
            
            for assignment in assignments:
                if len(assignment) >= 6:
                    launch_vehicle = assignment[4]
                    recovery_vehicle = assignment[5]
                    if launch_vehicle == recovery_vehicle:
                        same_vehicle_tasks += 1
                    else:
                        diff_vehicle_tasks += 1
            
            print(f"    同车任务: {same_vehicle_tasks}, 异车任务: {diff_vehicle_tasks}")
        
        # 成本信息
        if hasattr(state, '_total_cost') and state._total_cost is not None:
            print(f"\n总成本: {state._total_cost:.2f}")
        
        if hasattr(state, 'uav_cost') and state.uav_cost:
            total_uav_cost = sum(state.uav_cost.values())
            print(f"无人机总成本: {total_uav_cost:.2f}")
        
        print("="*80)


def visualize_from_state_example():
    """
    示例：如何从状态对象可视化
    """
    print("从状态对象可视化的示例")
    print("="*50)
    
    # 这里您需要提供您的状态对象
    # 例如：
    # state = your_fast_alns_solver.get_current_state()
    # 或者
    # state = create_fast_initial_state(...)
    
    print("请将您的状态对象传递给StateVisualizer.visualize_state()方法")
    print("示例代码：")
    print("""
    from visualize_from_state import StateVisualizer
    
    # 创建可视化器
    visualizer = StateVisualizer()
    
    # 可视化您的状态
    visualizer.visualize_state(your_state, "您的解决方案")
    
    # 分析状态
    visualizer.analyze_state(your_state)
    """)


if __name__ == "__main__":
    visualize_from_state_example()

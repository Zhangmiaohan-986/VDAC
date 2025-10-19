#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速可视化工具 - 一键可视化您的VDAC状态

这个工具提供了一个简单的函数，您只需要传入状态对象就能立即看到可视化结果。
"""

from visualize_from_state import StateVisualizer
import matplotlib.pyplot as plt

def quick_visualize(state, title="VDAC解决方案", show_analysis=True, figsize=(15, 12), node_info=None):
    """
    快速可视化VDAC状态 - 一键调用函数
    
    Args:
        state: 状态对象或包含必要数据的字典
        title: 图形标题
        show_analysis: 是否显示详细分析
        figsize: 图形大小
        node_info: 节点信息字典，格式为 {node_id: {'x': x, 'y': y, 'z': z, 'type': type}}
    
    Returns:
        StateVisualizer对象，可以用于进一步操作
    """
    print(f"🚀 开始可视化: {title}")
    print("-" * 50)
    
    # 创建可视化器
    visualizer = StateVisualizer(figsize)
    
    try:
        # 如果是状态对象，直接可视化
        if hasattr(state, 'vehicle_routes') and hasattr(state, 'customer_plan') and hasattr(state, 'uav_assignments'):
            print("✓ 检测到状态对象")
            
            # 检查是否有node信息
            if hasattr(state, 'node') and state.node:
                print("✓ 检测到状态对象中的节点信息，将显示实际坐标")
            elif node_info:
                print("✓ 检测到外部传入的节点信息，将显示实际坐标")
                # 将外部节点信息添加到状态对象中
                state.node = node_info
            else:
                print("⚠ 未检测到节点信息，将使用默认坐标布局")
            
            visualizer.visualize_state(state, title)
            
            if show_analysis:
                visualizer.analyze_state(state)
        
        # 如果是字典，尝试提取数据
        elif isinstance(state, dict):
            print("✓ 检测到字典数据，尝试提取...")
            vehicle_routes = state.get('vehicle_routes', [])
            customer_plan = state.get('customer_plan', {})
            uav_assignments = state.get('uav_assignments', {})
            node_info = state.get('node_info', None)
            
            if not vehicle_routes:
                raise ValueError("字典中缺少vehicle_routes数据")
            
            # 检查是否有节点坐标信息
            if node_info:
                print("✓ 检测到节点坐标信息，将显示实际坐标")
            else:
                print("⚠ 未检测到节点坐标信息，将使用默认坐标布局")
            
            # 使用基础可视化器
            from visualization import VDACVisualizer
            base_visualizer = VDACVisualizer(figsize)
            base_visualizer.visualize_solution(
                vehicle_routes=vehicle_routes,
                customer_plan=customer_plan,
                uav_assignments=uav_assignments,
                node_info=node_info,
                title=title
            )
            
            if show_analysis:
                print_analysis_from_dict(state)
        
        # 如果是numpy数值类型，可能是误传
        elif hasattr(state, 'dtype') and 'numpy' in str(type(state)):
            raise ValueError(f"检测到numpy数值类型 ({type(state)})，请传入完整的状态对象或字典数据")
        
        else:
            raise ValueError(f"不支持的数据类型: {type(state)}")
        
        print("✅ 可视化完成！")
        return visualizer
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        print("请检查您的数据格式是否正确")
        
        # 提供诊断信息
        print("\n🔍 数据类型诊断:")
        try:
            from diagnose_data_type import diagnose_data_type
            diagnose_data_type(state, "传入的数据")
            
            # 如果是状态对象，进行详细调试
            if hasattr(state, 'vehicle_routes'):
                print("\n🔍 状态对象详细调试:")
                from debug_state_object import debug_state_object
                debug_state_object(state, "传入的状态对象")
        except ImportError:
            print(f"数据类型: {type(state)}")
            if hasattr(state, 'dtype'):
                print("⚠️  检测到numpy数值类型，请传入完整的状态对象或字典数据")
        
        return None

def print_analysis_from_dict(data):
    """从字典数据打印分析信息"""
    print("\n" + "="*80)
    print("数据分析报告")
    print("="*80)
    
    vehicle_routes = data.get('vehicle_routes', [])
    customer_plan = data.get('customer_plan', {})
    uav_assignments = data.get('uav_assignments', {})
    
    print(f"车辆数量: {len(vehicle_routes)}")
    print(f"客户数量: {len(customer_plan)}")
    print(f"无人机数量: {len(uav_assignments)}")
    
    # 车辆路线统计
    print(f"\n车辆路线统计:")
    for i, route in enumerate(vehicle_routes):
        print(f"  车辆 {i+1}: {len(route)}个节点, 路线: {' -> '.join(map(str, route))}")
    
    # 无人机任务统计
    if uav_assignments:
        print(f"\n无人机任务统计:")
        for drone_id, assignments in uav_assignments.items():
            print(f"  无人机 {drone_id}: {len(assignments)}个任务")
    
    print("="*80)

def create_sample_state():
    """
    创建一个示例状态对象用于测试
    
    Returns:
        状态对象
    """
    # 示例数据
    vehicle_routes = [
        [1, 2, 3, 4, 5],  # 车辆1的路线
        [1, 6, 7, 8, 9],  # 车辆2的路线
        [1, 10, 11, 12]   # 车辆3的路线
    ]
    
    customer_plan = {
        2: (1, 2, 2, 3, 1, 1),    # 客户2：无人机1，从车辆1的节点2发射，服务客户2，被车辆1的节点3回收
        3: (2, 3, 3, 4, 1, 1),    # 客户3：无人机2，从车辆1的节点3发射，服务客户3，被车辆1的节点4回收
        6: (1, 6, 6, 7, 2, 2),    # 客户6：无人机1，从车辆2的节点6发射，服务客户6，被车辆2的节点7回收
        7: (2, 7, 7, 8, 2, 2),    # 客户7：无人机2，从车辆2的节点7发射，服务客户7，被车辆2的节点8回收
        10: (1, 10, 10, 11, 3, 3), # 客户10：无人机1，从车辆3的节点10发射，服务客户10，被车辆3的节点11回收
        11: (2, 11, 11, 12, 3, 3)  # 客户11：无人机2，从车辆3的节点11发射，服务客户11，被车辆3的节点12回收
    }
    
    uav_assignments = {
        1: [
            (1, 2, 2, 3, 1, 1),   # 无人机1的任务1
            (1, 6, 6, 7, 2, 2),   # 无人机1的任务2
            (1, 10, 10, 11, 3, 3) # 无人机1的任务3
        ],
        2: [
            (2, 3, 3, 4, 1, 1),   # 无人机2的任务1
            (2, 7, 7, 8, 2, 2),   # 无人机2的任务2
            (2, 11, 11, 12, 3, 3) # 无人机2的任务3
        ]
    }
    
    # 创建状态对象（简化版本）
    from collections import defaultdict
    
    # 创建符合要求的defaultdict结构
    vehicle_task_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    global_reservation_table = defaultdict(list)
    
    # 创建一个简单的状态对象用于测试
    class SampleState:
        def __init__(self, vehicle_routes, uav_assignments, customer_plan, vehicle_task_data, global_reservation_table):
            self.vehicle_routes = vehicle_routes
            self.uav_assignments = uav_assignments
            self.customer_plan = customer_plan
            self.vehicle_task_data = vehicle_task_data
            self.global_reservation_table = global_reservation_table
            self.total_cost = 100.0
    
    state = SampleState(
        vehicle_routes=vehicle_routes,
        uav_assignments=uav_assignments,
        customer_plan=customer_plan,
        vehicle_task_data=vehicle_task_data,
        global_reservation_table=global_reservation_table
    )
    
    return state

def demo():
    """
    演示如何使用快速可视化工具
    """
    print("🎯 VDAC快速可视化工具演示")
    print("="*60)
    
    # 方法1：使用示例状态对象
    print("\n📋 方法1：使用示例状态对象")
    sample_state = create_sample_state()
    quick_visualize(sample_state, "示例VDAC解决方案")
    
    # 方法2：使用字典数据（包含坐标信息）
    print("\n📋 方法2：使用字典数据（包含坐标信息）")
    sample_data = {
        'vehicle_routes': [
            [1, 2, 3, 4],
            [1, 5, 6, 7]
        ],
        'customer_plan': {
            2: (1, 2, 2, 3, 1, 1),
            3: (2, 3, 3, 4, 1, 1),
            5: (1, 5, 5, 6, 2, 2),
            6: (2, 6, 6, 7, 2, 2)
        },
        'uav_assignments': {
            1: [(1, 2, 2, 3, 1, 1), (1, 5, 5, 6, 2, 2)],
            2: [(2, 3, 3, 4, 1, 1), (2, 6, 6, 7, 2, 2)]
        },
        'node_info': {
            1: {'x': 0, 'y': 0, 'type': 'depot', 'label': '仓库 (0.0, 0.0)'},
            2: {'x': 2, 'y': 1, 'type': 'vtp', 'label': 'VTP节点2 (2.0, 1.0)'},
            3: {'x': 4, 'y': 2, 'type': 'customer', 'label': '客户3 (4.0, 2.0)'},
            4: {'x': 6, 'y': 1, 'type': 'vtp', 'label': 'VTP节点4 (6.0, 1.0)'},
            5: {'x': 2, 'y': -1, 'type': 'vtp', 'label': 'VTP节点5 (2.0, -1.0)'},
            6: {'x': 4, 'y': -2, 'type': 'customer', 'label': '客户6 (4.0, -2.0)'},
            7: {'x': 6, 'y': -1, 'type': 'vtp', 'label': 'VTP节点7 (6.0, -1.0)'}
        }
    }
    quick_visualize(sample_data, "字典数据示例")

def compare_solutions(state1, state2, title1="解决方案1", title2="解决方案2"):
    """
    比较两个解决方案
    
    Args:
        state1: 第一个状态对象
        state2: 第二个状态对象
        title1: 第一个解决方案的标题
        title2: 第二个解决方案的标题
    """
    print(f"🔄 比较解决方案: {title1} vs {title2}")
    print("-" * 60)
    
    visualizer = StateVisualizer()
    visualizer.visualize_comparison(state1, state2, title1, title2)

# 使用示例和说明
if __name__ == "__main__":
    print("""
🎯 VDAC快速可视化工具使用说明
=====================================

这个工具提供了一个超级简单的函数来可视化您的VDAC状态：

1. 基本用法：
   quick_visualize(your_state, "您的标题")

2. 从您的代码中调用：
   from quick_visualize import quick_visualize
   quick_visualize(my_state, "调试方案")

3. 比较两个方案：
   compare_solutions(state1, state2, "方案A", "方案B")

4. 运行演示：
   python quick_visualize.py
""")
    
    # 运行演示
    demo()

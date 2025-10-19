#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VDAC数据可视化使用示例

使用方法：
1. 修改下面的 vehicle_routes, customer_plan, uav_assignments 变量
2. 运行此文件查看可视化结果

数据格式说明：
- vehicle_routes: 车辆路线列表，每个元素是一个车辆的路由（节点ID列表）
- customer_plan: 客户计划字典，{customer_id: (drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle)}
- uav_assignments: 无人机分配字典，{drone_id: [(drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle), ...]}
"""

from visualization import VDACVisualizer

def visualize_my_data():
    """
    可视化您的数据
    请修改下面的数据为您实际的数据
    """
    
    # ==================== 请在这里输入您的数据 ====================
    
    # 车辆路线数据
    # 格式：每个元素是一个车辆的路由（节点ID列表）
    vehicle_routes = [
        [1, 2, 3, 4, 5],  # 车辆1的路线：从节点1 -> 节点2 -> 节点3 -> 节点4 -> 节点5
        [1, 6, 7, 8, 9],  # 车辆2的路线：从节点1 -> 节点6 -> 节点7 -> 节点8 -> 节点9
        [1, 10, 11, 12]   # 车辆3的路线：从节点1 -> 节点10 -> 节点11 -> 节点12
    ]
    
    # 客户计划数据
    # 格式：{customer_id: (drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle)}
    customer_plan = {
        2: (1, 2, 2, 3, 1, 1),    # 客户2：无人机1，从车辆1的节点2发射，服务客户2，被车辆1的节点3回收
        3: (2, 3, 3, 4, 1, 1),    # 客户3：无人机2，从车辆1的节点3发射，服务客户3，被车辆1的节点4回收
        6: (1, 6, 6, 7, 2, 2),    # 客户6：无人机1，从车辆2的节点6发射，服务客户6，被车辆2的节点7回收
        7: (3, 7, 7, 8, 2, 2),    # 客户7：无人机3，从车辆2的节点7发射，服务客户7，被车辆2的节点8回收
        10: (2, 10, 10, 11, 3, 3), # 客户10：无人机2，从车辆3的节点10发射，服务客户10，被车辆3的节点11回收
        11: (1, 11, 11, 12, 3, 1), # 客户11：无人机1，从车辆3的节点11发射，服务客户11，被车辆1的节点12回收（异车任务）
    }
    
    # 无人机分配数据
    # 格式：{drone_id: [(drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle), ...]}
    uav_assignments = {
        1: [
            (1, 2, 2, 3, 1, 1),   # 无人机1的任务1：同车任务
            (1, 6, 6, 7, 2, 2),   # 无人机1的任务2：同车任务
            (1, 11, 11, 12, 3, 1) # 无人机1的任务3：异车任务
        ],
        2: [
            (2, 3, 3, 4, 1, 1),   # 无人机2的任务1：同车任务
            (2, 10, 10, 11, 3, 3) # 无人机2的任务2：同车任务
        ],
        3: [
            (3, 7, 7, 8, 2, 2)    # 无人机3的任务1：同车任务
        ]
    }
    
    # ==================== 数据输入结束 ====================
    
    # 创建可视化器
    visualizer = VDACVisualizer(figsize=(16, 12))
    
    # 进行可视化
    print("开始可视化您的数据...")
    visualizer.visualize_solution(
        vehicle_routes=vehicle_routes,
        customer_plan=customer_plan,
        uav_assignments=uav_assignments,
        title="您的VDAC解决方案可视化"
    )
    
    print("可视化完成！")


def visualize_with_real_coordinates():
    """
    如果您有真实的节点坐标信息，可以使用这个函数
    """
    
    # ==================== 请在这里输入您的数据 ====================
    
    # 车辆路线数据
    vehicle_routes = [
        [1, 2, 3, 4, 5],
        [1, 6, 7, 8, 9],
        [1, 10, 11, 12]
    ]
    
    # 客户计划数据
    customer_plan = {
        2: (1, 2, 2, 3, 1, 1),
        3: (2, 3, 3, 4, 1, 1),
        6: (1, 6, 6, 7, 2, 2),
        7: (3, 7, 7, 8, 2, 2),
        10: (2, 10, 10, 11, 3, 3),
        11: (1, 11, 11, 12, 3, 1),
    }
    
    # 无人机分配数据
    uav_assignments = {
        1: [(1, 2, 2, 3, 1, 1), (1, 6, 6, 7, 2, 2), (1, 11, 11, 12, 3, 1)],
        2: [(2, 3, 3, 4, 1, 1), (2, 10, 10, 11, 3, 3)],
        3: [(3, 7, 7, 8, 2, 2)]
    }
    
    # 节点坐标信息（如果有真实坐标的话）
    node_info = {
        1: {'x': 0, 'y': 0, 'type': 'depot', 'label': '仓库'},
        2: {'x': 2, 'y': 1, 'type': 'vtp', 'label': 'VTP节点2'},
        3: {'x': 4, 'y': 2, 'type': 'customer', 'label': '客户3'},
        4: {'x': 6, 'y': 1, 'type': 'vtp', 'label': 'VTP节点4'},
        5: {'x': 8, 'y': 0, 'type': 'vtp', 'label': 'VTP节点5'},
        6: {'x': 2, 'y': -1, 'type': 'vtp', 'label': 'VTP节点6'},
        7: {'x': 4, 'y': -2, 'type': 'customer', 'label': '客户7'},
        8: {'x': 6, 'y': -1, 'type': 'vtp', 'label': 'VTP节点8'},
        9: {'x': 8, 'y': -2, 'type': 'vtp', 'label': 'VTP节点9'},
        10: {'x': 2, 'y': 3, 'type': 'vtp', 'label': 'VTP节点10'},
        11: {'x': 4, 'y': 4, 'type': 'customer', 'label': '客户11'},
        12: {'x': 6, 'y': 3, 'type': 'vtp', 'label': 'VTP节点12'},
    }
    
    # ==================== 数据输入结束 ====================
    
    # 创建可视化器
    visualizer = VDACVisualizer(figsize=(16, 12))
    
    # 进行可视化（使用真实坐标）
    print("开始可视化您的数据（使用真实坐标）...")
    visualizer.visualize_solution(
        vehicle_routes=vehicle_routes,
        customer_plan=customer_plan,
        uav_assignments=uav_assignments,
        node_info=node_info,  # 使用真实坐标
        title="您的VDAC解决方案可视化（真实坐标）"
    )
    
    print("可视化完成！")


if __name__ == "__main__":
    print("VDAC数据可视化工具")
    print("="*50)
    print("请选择可视化模式：")
    print("1. 使用默认坐标（推荐）")
    print("2. 使用真实坐标")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        visualize_my_data()
    elif choice == "2":
        visualize_with_real_coordinates()
    else:
        print("无效选择，使用默认模式...")
        visualize_my_data()


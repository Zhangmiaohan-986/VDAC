#!/usr/bin/env python
"""
测试rm_empty_node函数的功能
"""

from rm_node_sort_node import rm_empty_node

def test_rm_empty_node():
    """
    测试rm_empty_node函数
    """
    # 模拟客户计划数据
    # 格式: {mission_tuple: assignment_info}
    # mission_tuple = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle)
    customer_plan = {
        (1, 10, 20, 15, 1, 1): "assignment1",  # 无人机1，车辆1发射和回收
        (2, 12, 25, 18, 1, 1): "assignment2",  # 无人机2，车辆1发射和回收
        (3, 30, 35, 32, 2, 2): "assignment3",  # 无人机3，车辆2发射和回收
        (4, 40, 45, 42, 3, 3): "assignment4",  # 无人机4，车辆3发射和回收
    }
    
    # 模拟车辆路线数据
    # 格式: [vehicle_route1, vehicle_route2, ...]
    # 每个route包含: [起点, 中间节点1, 中间节点2, ..., 终点]
    vehicle_route = [
        [1, 10, 11, 12, 15, 16, 18, 20],  # 车辆1的路线
        [2, 30, 31, 32, 35, 36],          # 车辆2的路线
        [3, 40, 41, 42, 45, 46],          # 车辆3的路线
        [4, 50, 51, 52, 53],              # 车辆4的路线（无无人机任务）
    ]
    
    print("原始车辆路线:")
    for i, route in enumerate(vehicle_route):
        print(f"车辆{i+1}: {route}")
    
    print("\n客户计划:")
    for mission_tuple, info in customer_plan.items():
        drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = mission_tuple
        print(f"无人机{drone_id}: 车辆{launch_vehicle}发射({launch_node}) -> 客户{customer_node} -> 车辆{recovery_vehicle}回收({recovery_node})")
    
    # 调用rm_empty_node函数
    filtered_vehicle_route, empty_nodes_by_vehicle = rm_empty_node(customer_plan, vehicle_route)
    
    print("\n=== 处理结果 ===")
    
    print("\n过滤后的车辆路线:")
    for i, route in enumerate(filtered_vehicle_route):
        print(f"车辆{i+1}: {route}")
    
    print("\n各车辆的空节点（无无人机任务的节点）:")
    for vehicle_id, empty_nodes in empty_nodes_by_vehicle.items():
        if empty_nodes:
            print(f"车辆{vehicle_id}: {empty_nodes}")
        else:
            print(f"车辆{vehicle_id}: 无空节点")
    
    print("\n=== 详细分析 ===")
    
    # 详细分析每个车辆的情况
    for i, (original_route, filtered_route) in enumerate(zip(vehicle_route, filtered_vehicle_route)):
        vehicle_id = i + 1
        print(f"\n车辆{vehicle_id}:")
        print(f"  原始路线: {original_route}")
        print(f"  过滤后路线: {filtered_route}")
        
        # 找出该车辆上的无人机任务
        vehicle_missions = []
        for mission_tuple in customer_plan.keys():
            drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = mission_tuple
            if launch_vehicle == vehicle_id and recovery_vehicle == vehicle_id:
                vehicle_missions.append(mission_tuple)
        
        if vehicle_missions:
            print(f"  该车辆上的无人机任务:")
            for mission in vehicle_missions:
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = mission
                print(f"    无人机{drone_id}: 发射节点{launch_node}, 回收节点{recovery_node}")
        else:
            print(f"  该车辆无无人机任务")
        
        if vehicle_id in empty_nodes_by_vehicle and empty_nodes_by_vehicle[vehicle_id]:
            print(f"  移除的空节点: {empty_nodes_by_vehicle[vehicle_id]}")

if __name__ == "__main__":
    test_rm_empty_node() 
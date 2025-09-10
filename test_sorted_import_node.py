#!/usr/bin/env python
"""
测试sorted_import_node函数的功能
"""

from fast_alns_solver import FastMfstspState


def test_sorted_import_node():
    """
    测试sorted_import_node函数
    """
    # 创建测试数据
    vehicle_routes = {
        1: [1, 10, 11, 12, 15, 16, 18, 20],  # 车辆1的路线
        2: [2, 30, 31, 32, 35, 36],          # 车辆2的路线
        3: [3, 40, 41, 42, 45, 46]           # 车辆3的路线
    }
    
    uav_assignments = {}
    customer_plan = {}
    vehicle_task_data = {}
    global_reservation_table = {}
    
    # 创建FastMfstspState对象
    state = FastMfstspState(
        vehicle_routes=vehicle_routes,
        uav_assignments=uav_assignments,
        customer_plan=customer_plan,
        vehicle_task_data=vehicle_task_data,
        global_reservation_table=global_reservation_table,
        total_cost=100,
        init_uav_plan={},
        init_vehicle_plan_time={},
        vehicle={},
        T=[1, 2, 3],
        V=[1, 2, 3, 4, 5, 6],
        veh_distance={}
    )
    
    # 模拟当前计划（正在使用的任务）
    current_plan = {
        'customer1': (1, 10, 20, 15, 1, 1),  # 无人机1，车辆1发射和回收
        'customer2': (2, 12, 25, 18, 1, 1),  # 无人机2，车辆1发射和回收
        'customer3': (3, 30, 35, 32, 2, 2),  # 无人机3，车辆2发射和回收
        'customer4': (4, 40, 45, 42, 3, 3),  # 无人机4，车辆3发射和回收
    }
    
    # 模拟被破坏的计划（被移除的任务）
    destroyed_plan = {
        'customer5': (5, 11, 22, 16, 1, 1),  # 无人机5，车辆1发射和回收（被破坏）
        'customer6': (6, 31, 33, 35, 2, 2),  # 无人机6，车辆2发射和回收（被破坏）
        'customer7': (7, 41, 43, 45, 3, 3),  # 无人机7，车辆3发射和回收（被破坏）
    }
    
    # 调用sorted_import_node函数
    result = state.sorted_import_node(destroyed_plan, current_plan)
    
    # 打印结果
    print("=" * 60)
    print("sorted_import_node函数测试结果")
    print("=" * 60)
    
    for vehicle_id, levels in result.items():
        print(f"\n车辆 {vehicle_id} 的节点分类:")
        print(f"  等级1 (当前使用): {levels['level1']}")
        print(f"  等级2 (被破坏): {levels['level2']}")
        print(f"  等级3 (从未使用): {levels['level3']}")
    
    # 验证结果
    print("\n" + "=" * 60)
    print("结果验证:")
    print("=" * 60)
    
    # 验证车辆1
    vehicle1_result = result[1]
    expected_level1 = [10, 12, 15, 18]  # 当前使用的发射/回收节点
    expected_level2 = [11, 16]          # 被破坏的发射/回收节点
    expected_level3 = [1, 20]           # 从未使用的节点
    
    print(f"车辆1 - 等级1: {vehicle1_result['level1']} (期望: {expected_level1})")
    print(f"车辆1 - 等级2: {vehicle1_result['level2']} (期望: {expected_level2})")
    print(f"车辆1 - 等级3: {vehicle1_result['level3']} (期望: {expected_level3})")
    
    # 验证车辆2
    vehicle2_result = result[2]
    expected_level1 = [30, 32]          # 当前使用的发射/回收节点
    expected_level2 = [31, 35]          # 被破坏的发射/回收节点
    expected_level3 = [2, 36]           # 从未使用的节点
    
    print(f"车辆2 - 等级1: {vehicle2_result['level1']} (期望: {expected_level1})")
    print(f"车辆2 - 等级2: {vehicle2_result['level2']} (期望: {expected_level2})")
    print(f"车辆2 - 等级3: {vehicle2_result['level3']} (期望: {expected_level3})")
    
    # 验证车辆3
    vehicle3_result = result[3]
    expected_level1 = [40, 42]          # 当前使用的发射/回收节点
    expected_level2 = [41, 45]          # 被破坏的发射/回收节点
    expected_level3 = [3, 46]           # 从未使用的节点
    
    print(f"车辆3 - 等级1: {vehicle3_result['level1']} (期望: {expected_level1})")
    print(f"车辆3 - 等级2: {vehicle3_result['level2']} (期望: {expected_level2})")
    print(f"车辆3 - 等级3: {vehicle3_result['level3']} (期望: {expected_level3})")
    
    return result


def test_edge_cases():
    """
    测试边界情况
    """
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)
    
    # 创建测试数据
    vehicle_routes = {
        1: [1, 2, 3, 4, 5]
    }
    
    uav_assignments = {}
    customer_plan = {}
    vehicle_task_data = {}
    global_reservation_table = {}
    
    state = FastMfstspState(
        vehicle_routes=vehicle_routes,
        uav_assignments=uav_assignments,
        customer_plan=customer_plan,
        vehicle_task_data=vehicle_task_data,
        global_reservation_table=global_reservation_table,
        total_cost=100,
        init_uav_plan={},
        init_vehicle_plan_time={},
        vehicle={},
        T=[1],
        V=[1, 2],
        veh_distance={}
    )
    
    # 测试1：空计划
    print("\n测试1：空计划")
    result1 = state.sorted_import_node({}, {})
    print(f"结果: {result1}")
    
    # 测试2：只有当前计划，没有破坏计划
    print("\n测试2：只有当前计划")
    current_plan = {'customer1': (1, 2, 3, 4, 1, 1)}
    result2 = state.sorted_import_node({}, current_plan)
    print(f"结果: {result2}")
    
    # 测试3：只有破坏计划，没有当前计划
    print("\n测试3：只有破坏计划")
    destroyed_plan = {'customer1': (1, 2, 3, 4, 1, 1)}
    result3 = state.sorted_import_node(destroyed_plan, {})
    print(f"结果: {result3}")
    
    # 测试4：节点在多个车辆中使用
    print("\n测试4：节点在多个车辆中使用")
    vehicle_routes_multi = {
        1: [1, 2, 3],
        2: [2, 3, 4]
    }
    
    state_multi = FastMfstspState(
        vehicle_routes=vehicle_routes_multi,
        uav_assignments=uav_assignments,
        customer_plan=customer_plan,
        vehicle_task_data=vehicle_task_data,
        global_reservation_table=global_reservation_table,
        total_cost=100,
        init_uav_plan={},
        init_vehicle_plan_time={},
        vehicle={},
        T=[1, 2],
        V=[1, 2],
        veh_distance={}
    )
    
    current_plan_multi = {
        'customer1': (1, 2, 5, 3, 1, 1),  # 车辆1使用节点2和3
        'customer2': (2, 3, 6, 4, 2, 2)   # 车辆2使用节点3和4
    }
    
    result4 = state_multi.sorted_import_node({}, current_plan_multi)
    print(f"结果: {result4}")


if __name__ == "__main__":
    # 运行主要测试
    test_sorted_import_node()
    
    # 运行边界情况测试
    test_edge_cases() 
import math
import numpy as np
NODE_TYPE_DEPOT	= 0
NODE_TYPE_CUST	= 1

TYPE_TRUCK 		= 1
TYPE_UAV 		= 2

def cal_low_cost(i_vtp,customre,j_vtp,v_id,recv_v_id, uav_id, uav_travel,veh_distance,veh_travel,node,vehicle,vehicle_type,xeee):  # 计算无人机和车辆的成本下限
    """计算无人机成本下限"""
    # 计算无人机成本下限
    # cost = 0
    path = []
    if vehicle_type == TYPE_UAV:
        air_i = node[i_vtp].map_key
        air_j = node[j_vtp].map_key
        per_cost = vehicle[uav_id].per_cost  # 计算无人机运行的单公里配送成本
        # 计算无人机的飞行时间，飞行距离，飞行成本，飞行能耗及飞行路线
        uav_time_ic = uav_travel[uav_id][air_i][customre].totalTime
        uav_distance_ic = uav_travel[uav_id][air_i][customre].totalDistance
        uav_route_ic = uav_travel[uav_id][air_i][customre].path
        uav_cost_ic = uav_distance_ic * per_cost
        path.extend(uav_route_ic)
        uav_time_cj = uav_travel[uav_id][customre][air_j].totalTime
        uav_distance_cj = uav_travel[uav_id][customre][air_j].totalDistance
        uav_route_cj = uav_travel[uav_id][customre][air_j].path
        uav_cost_cj = uav_distance_cj * per_cost
        path.extend(uav_route_cj)
        uav_time_ij = uav_time_ic + uav_time_cj
        uav_distance_ij = uav_distance_ic + uav_distance_cj
        uav_cost_ij = uav_distance_ij * per_cost
        # uav_energy_ij = uav_travel[uav_id][air_i][customre].energy
        return uav_cost_ij, uav_time_ij, path
    elif vehicle_type == TYPE_TRUCK:
        per_cost = vehicle[v_id].per_cost
        veh_distance = veh_distance[v_id][i_vtp][j_vtp]
        veh_cost = veh_distance * per_cost
        veh_time = veh_travel[v_id][i_vtp][j_vtp]
        return veh_cost, veh_time, path

# 计算每种方案成本的下限，没有包含空中无人机的相互避障状况
def calculate_plan_cost(best_plan_cost, vehicle_route, vehicle, vehicle_id, uav_id, veh_distance):
    total_cost = 0
    np_best_plan_cost = np.array(list(best_plan_cost.values()))
    uav_total_cost = np.sum(np_best_plan_cost)
    vehicle_fix_cost = 0
    vehicle_total_route_cost = 0
    # 计算车辆固定成本
    for v_id in vehicle_id:
        fix_cost = vehicle[v_id].fix_cost
        vehicle_fix_cost += fix_cost
    # 计算每辆车的路径成本
    for veh_idx, route in enumerate(vehicle_route):
        veh_id = vehicle_id[veh_idx]
        vehicle_route_cost = calculate_veh_route_cost(route, vehicle[veh_id], veh_distance)
        vehicle_total_route_cost += vehicle_route_cost
    total_cost = uav_total_cost + vehicle_fix_cost + vehicle_total_route_cost
    return total_cost

# 计算车辆的路径成本，按照车辆行驶的距离计算
def calculate_veh_route_cost(route, vehicle, veh_distance):
    veh_cost = 0
    veh_per_cost = vehicle.per_cost  # 车辆行驶的单位成本
    for i in range(len(route)-1):
        veh_cost += veh_distance[vehicle.id][route[i]][route[i+1]] * veh_per_cost
    return veh_cost


def update_calculate_plan_cost(best_uav_plan, vehicle_route, vehicle, vehicle_id, uav_id, veh_distance):
    total_cost = 0
    uav_total_cost = 0
    vehicle_fix_cost = 0
    vehicle_total_route_cost = 0
    # 计算车辆固定成本
    for v_id in vehicle_id:
        fix_cost = vehicle[v_id].fix_cost
        vehicle_fix_cost += fix_cost
    # 计算每辆车的路径成本
    for veh_idx, route in enumerate(vehicle_route):
        veh_id = vehicle_id[veh_idx]
        vehicle_route_cost = calculate_veh_route_cost(route, vehicle[veh_id], veh_distance)
        vehicle_total_route_cost += vehicle_route_cost
    for mission_tuple, mission_dict in best_uav_plan.items():
        uav_id = mission_dict['drone_id']
        uav_cost = mission_dict['uav_cost']
        uav_time = mission_dict['uav_time']
        uav_route = mission_dict['uav_route']
        uav_total_cost += uav_cost
    total_cost = uav_total_cost + vehicle_fix_cost + vehicle_total_route_cost
    return total_cost
    
    

# def sort_customer_plans(customer_costs, plan_y):
#     """
#     对每个客户的成本列表进行排序，并同步调整对应方案的顺序
    
#     :param customer_costs: 客户成本字典 {cid: [cost1, cost2...]}
#     :param plan_y: 原始方案字典 {cid: [plan1, plan2...]}
#     :return: 排序后的 (sorted_costs, sorted_plan_y)
#     """
#     sorted_costs = {}
#     sorted_plan_y = {}
    
#     # 遍历每个客户
#     for cid in customer_costs:
#         # 获取原始成本列表和方案列表
#         original_costs = customer_costs[cid]
#         original_plans = plan_y[cid]
        
#         # 验证数据一致性
#         if len(original_costs) != len(original_plans):
#             raise ValueError(f"客户{cid}的成本列表与方案数量不匹配")
        
#         # 合并成本与方案进行同步排序
#         combined = sorted(zip(original_costs, original_plans),
#                          key=lambda x: x[0])  # 按成本升序
        
#         # 拆分排序结果
#         sorted_cost = [cost for cost, _ in combined]
#         sorted_plan = [plan for _, plan in combined]
        
#         # 存储结果
#         sorted_costs[cid] = sorted_cost
#         sorted_plan_y[cid] = sorted_plan
    
#     return sorted_costs, sorted_plan_y

def sort_customer_plans(customer_costs, plan_y, plan_time, plan_uav_route):
    """
    根据成本对客户的所有方案进行排序，并保持所有相关数据同步
    
    参数:
    customer_costs -- 包含客户成本列表的字典
    plan_y -- 包含客户方案的字典
    plan_time -- 包含客户方案时间的字典
    plan_uav_route -- 包含客户无人机路径的字典
    
    返回:
    排序后的四个字典元组 (customer_costs, plan_y, plan_time, plan_uav_route)
    """
    # 处理每个客户
    for customer in customer_costs.keys():
        # 创建(成本, 索引)对，用于跟踪原始位置
        indexed_costs = [(cost, idx) for idx, cost in enumerate(customer_costs[customer])]
        
        # 按成本从小到大排序
        indexed_costs.sort(key=lambda x: x[0])
        
        # 提取排序后的索引
        sorted_indices = [idx for _, idx in indexed_costs]
        
        # 使用排序后的索引重新排列所有列表
        customer_costs[customer] = [customer_costs[customer][idx] for idx in sorted_indices]
        plan_y[customer] = [plan_y[customer][idx] for idx in sorted_indices]
        plan_time[customer] = [plan_time[customer][idx] for idx in sorted_indices]
        plan_uav_route[customer] = [plan_uav_route[customer][idx] for idx in sorted_indices]
    
    return customer_costs, plan_y, plan_time, plan_uav_route

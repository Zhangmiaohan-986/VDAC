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
    # 计算车辆固定成本,取消当前车辆固定成本
    # for v_id in vehicle_id:
    #     fix_cost = vehicle[v_id].fix_cost
    #     vehicle_fix_cost += fix_cost
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
    

def calculate_window_cost(best_customer_plan,
                          best_plan_cost,
                          veh_arrival_times,
                          vehicle,
                          customer_time_windows_h,
                          early_arrival_cost,
                          late_arrival_cost,
                          uav_fly_time_to_customer,
                          node):
    """
    计算带时间窗口惩罚的总成本。

    Parameters
    ----------
    best_customer_plan : dict
        key -> (uav_id, i_vtp, customer_node, j_vtp, v_id, recv_v_id)。
    best_uav_plan : dict or None
        这里暂时没有用到，占位以兼容你原有接口。
    best_plan_cost : dict
        key -> 原始无人机/车辆成本（比如你的 best_plan_cost）。
    veh_arrival_times : dict
        veh_arrival_times[v_id][node_id] = 卡车到达该 VTP 节点的时间 (小时)。
    vehicle : dict
        vehicle[v_id] / vehicle[uav_id] -> make_vehicle 对象。
    truck_ids : list
        卡车 id 列表，这里暂时没直接用到（你以后可以扩展卡车时间窗惩罚）。
    uav_ids : list
        无人机 id 列表，这里也只是占位。
    veh_distance : dict
        卡车距离矩阵，这里暂时不用。
    customer_time_windows_h : dict
        tw_idx -> {'cust', 'ready_h', 'due_h', 'service_h'}。
    early_arrival_cost : list or tuple
        [penalty_per_hour, penalty_per_minute]，此处只用 penalty_per_hour。
    late_arrival_cost : list or tuple
        [penalty_per_hour, penalty_per_minute]，此处只用 penalty_per_hour。
    uav_fly_time_to_customer : dict or None
        若为 dict：uav_fly_time_to_customer[key] = 无人机从发射 VTP 到客户的飞行时间 (小时)。
        若为 None：默认为所有飞行时间为 0，你可以在外面换成基于 uav_travel 的真实值。

    Returns
    -------
    window_total_cost : float
        所有 key 的 total 成本之和。
    uav_tw_violation_cost : dict
        key -> 时间窗惩罚成本。
    total_cost_dict : dict
        key -> best_plan_cost[key] + uav_tw_violation_cost[key]。
    """
    # 1) 先算每个客户的时间窗惩罚
    uav_tw_violation_cost = compute_uav_tw_violation_cost(
        best_customer_plan=best_customer_plan,
        veh_arrival_times=veh_arrival_times,
        vehicle=vehicle,
        customer_time_windows_h=customer_time_windows_h,
        early_arrival_cost=early_arrival_cost,
        late_arrival_cost=late_arrival_cost,
        uav_fly_time_to_customer=uav_fly_time_to_customer,
        node=node,
    )

    # 2) 把原始成本和时间窗惩罚叠加
    total_cost_dict = combine_plan_and_tw_cost(
        best_plan_cost=best_plan_cost,
        uav_tw_violation_cost=uav_tw_violation_cost
    )

    # 3) 总成本（可以理解为“所有客户的无人机方案成本 + 时间窗惩罚”）
    window_total_cost = float(sum(total_cost_dict.values()))

    return window_total_cost, uav_tw_violation_cost, total_cost_dict

def calculate_customer_window_cost(best_customer_plan,
                          vehicle,
                          veh_arrival_times,
                          customer_time_windows_h,
                          early_arrival_cost,
                          late_arrival_cost,
                          uav_fly_time_to_customer,
                          node):

    # 1) 先算每个客户的时间窗惩罚
    uav_tw_violation_cost = compute_uav_tw_violation_cost(
        best_customer_plan=best_customer_plan,
        vehicle=vehicle,
        veh_arrival_times=veh_arrival_times,
        customer_time_windows_h=customer_time_windows_h,
        early_arrival_cost=early_arrival_cost,
        late_arrival_cost=late_arrival_cost,
        uav_fly_time_to_customer=uav_fly_time_to_customer,
        node=node,
    )

    uav_tw_violation_cost = float(sum(uav_tw_violation_cost.values()))
    return uav_tw_violation_cost

def combine_plan_and_tw_cost(best_plan_cost, uav_tw_violation_cost):
    """
    将原有的 best_plan_cost 与时间窗违背成本逐项相加。

    Parameters
    ----------
    best_plan_cost : dict
        key -> 原始成本（比如你 cal_low_cost 得到的 cost）。
    uav_tw_violation_cost : dict
        key -> 时间窗惩罚成本。

    Returns
    -------
    total_cost_dict : dict
        key -> 原始成本 + 时间窗惩罚。
    """

    total_cost_dict = defaultdict(float)

    all_keys = set(best_plan_cost.keys()) | set(uav_tw_violation_cost.keys())
    for k in all_keys:
        base_cost = float(best_plan_cost.get(k, 0.0))
        tw_cost = float(uav_tw_violation_cost.get(k, 0.0))
        total_cost_dict[k] = base_cost + tw_cost

    return total_cost_dict


from collections import defaultdict

def compute_uav_tw_violation_cost(best_customer_plan,
                                  veh_arrival_times,
                                  vehicle,
                                  customer_time_windows_h,
                                  early_arrival_cost,
                                  late_arrival_cost,
                                  uav_fly_time_to_customer,node):
    """
    计算无人机在每个客户点的时间窗违背成本。
    Parameters
    ----------
    best_customer_plan : dict
        key -> (uav_id, i_vtp, customer_node, j_vtp, v_id, recv_v_id)
        这里 key 通常与 customer_time_windows_h 的 key 对应。
    veh_arrival_times : dict
        veh_arrival_times[v_id][node_id] = 卡车到达该节点的时间 (小时)。
    vehicle : dict
        vehicle[v_id] / vehicle[uav_id] -> make_vehicle 对象。
    customer_time_windows_h : dict
        tw_idx -> {'cust', 'ready_h', 'due_h', 'service_h'}。
    early_arrival_cost : list or tuple
        [penalty_per_hour, penalty_per_minute]，此处只用 penalty_per_hour。
    late_arrival_cost : list or tuple
        [penalty_per_hour, penalty_per_minute]，此处只用 penalty_per_hour。
    uav_fly_time_to_customer : dict
        uav_fly_time_to_customer[key] = 无人机从发射 VTP 到客户点的飞行时间 (小时)。
        若你愿意，也可以在函数内部改成从 uav_travel 计算。
    Returns
    -------
    uav_tw_violation_cost : dict
        key -> 时间窗违背成本（仅无人机部分），结构与 best_plan_cost 一致。
    """
    # 每小时的提前 / 迟到惩罚系数
    early_penalty_per_hour = float(early_arrival_cost[0])
    late_penalty_per_hour = float(late_arrival_cost[0])

    uav_tw_violation_cost = defaultdict(float)

    for key_idx, plan in best_customer_plan.items():
        if plan is None:
            continue

        uav_id, i_vtp, customer_node, j_vtp, v_id, recv_v_id = plan
        map_i_vtp = node[i_vtp].map_key
        map_j_vtp = node[j_vtp].map_key
        # # 只对无人机任务计时间窗成本，如果你要卡车也算，可以去掉这一段判断
        # if hasattr(vehicle[uav_id], "vehicleType"):
        #     try:
        #         vtype = vehicle[uav_id].vehicleType
        #     except Exception:
        #         vtype = None
        # else:
        #     vtype = None

        # # 如果不是 UAV，就不在这里算时间窗成本
        # # （如果你车也要罚，自己放开这句）
        # # 假设 TYPE_UAV 是已定义常量
        # try:
        #     TYPE_UAV
        # except NameError:
        #     TYPE_UAV = 1  # 保险起见给个默认值，实际工程里用你自己的

        # if vtype is not None and vtype != TYPE_UAV:
        #     uav_tw_violation_cost[key_idx] = 0.0
        #     continue

        # 对应的时间窗信息
        tw = customer_time_windows_h.get(key_idx, None)
        if tw is None:
            # 没配时间窗就认为不罚
            uav_tw_violation_cost[key_idx] = 0.0
            print(f"没有时间窗信息，不罚")
            continue

        ready_h = float(tw['ready_h'])
        due_h = float(tw['due_h'])
        # service_h = float(tw['service_h'])  # 若需要可参与后续计算

        # 卡车在发射 VTP 的到达时间
        try:
            truck_arrival_time = float(veh_arrival_times[v_id][i_vtp])
        except KeyError:
            # 没有这个到达时间，说明方案不完整，时间窗惩罚设 0 或者你可以 raise
            uav_tw_violation_cost[key_idx] = 0.0
            print(f"没有卡车到达时间，请检查")
            continue
        
        # 无人机抵达节点时间
        arrival_time_at_customer = uav_fly_time_to_customer[uav_id][map_i_vtp][customer_node].totalTime + truck_arrival_time

        # 提前 & 迟到（单位：小时）
        early = max(0.0, ready_h - arrival_time_at_customer)
        late = max(0.0, arrival_time_at_customer - due_h)

        early_cost = early * early_penalty_per_hour
        late_cost = late * late_penalty_per_hour

        tw_cost = early_cost + late_cost
        uav_tw_violation_cost[key_idx] = tw_cost

    return uav_tw_violation_cost


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

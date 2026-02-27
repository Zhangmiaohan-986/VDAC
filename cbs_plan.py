import numpy as np
import copy
from task_data import vehicle_task
from cost_y import calculate_plan_cost
from task_data import *
from cost_y import *
from call_function import *
from initialize import *
import heapq
import itertools
from collections import deque
import networkx as nx
import time
import itertools
# from cbs_plan import *

# def low_update_time(uav_task_dict, best_customer_plan, best_uav_plan, best_vehicle_route, vehicle_task_data, vehicle_arrival_time, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee):
def low_update_time(uav_task_dict, best_uav_plan, best_vehicle_route, vehicle_task_data, vehicle_arrival_time, node, V, T, vehicle, uav_travel):
    """
    对粗略的初始方案规划正确的无人机及车辆的时间，并进行简单的任务分配情况描述
    """

    detailed_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)
    trucks_id = T
    uav_id = V
    time_uav_task_dict = {}
    time_customer_plan = {}
    vehicle_plan_time = {}
    for id in uav_id:
        time_uav_task_dict[id] = []
    vehicle_events = defaultdict(lambda: defaultdict(list))
    for drone_id, missions in uav_task_dict.items():
        for flight_tuple in missions:
            _, launch_node, _, recovery_node, launch_veh_id, recovery_veh_id = flight_tuple
            vehicle_events[launch_veh_id][launch_node].append({
                "type": "launch", "task_type": TASK_DRONE_LAUNCH, "drone_id": drone_id, "flight_tuple": flight_tuple
            })
            vehicle_events[recovery_veh_id][recovery_node].append({
                "type": "recovery", "task_type": TASK_DRONE_RECOVERY, "drone_id": drone_id, "flight_tuple": flight_tuple
            })
    sorted_items = sorted(best_uav_plan.items(), key=lambda item: item[1]['launch_time'])
    sorted_mission_keys = [item[0] for item in sorted_items]

    # 更新初始化车辆任务，涵盖到达时间，离开时间，任务类型, 以及详细的任务列表
    for id in trucks_id:
        idx = id -1 
        vehicle_route = best_vehicle_route[idx]
        # vehicle = vehicle[id]
        # 获得车辆在各个节点基础初始信息
        for index, node_id in enumerate(vehicle_route[:-1]):
            if index == 0:
                # arrive_time = 0
                arrive_time = vehicle_arrival_time[id][node_id]
                departure_time = vehicle_arrival_time[id][node_id]
                detailed_vehicle_task_data[id][node_id].add_node(node_id, arrive_time, departure_time)
                detailed_vehicle_task_data[id][node_id].prcise_arrive_time = arrive_time
                detailed_vehicle_task_data[id][node_id].prcise_departure_time = departure_time  # 只有在初始点，其离开时间更新为最终任务完成时间
                
                carry_drone_list = detailed_vehicle_task_data[id][node_id].drone_list
                launch_drone_list = detailed_vehicle_task_data[id][node_id].launch_drone_list
                recovery_drone_list = detailed_vehicle_task_data[id][node_id].recovery_drone_list
                launch_drone_list = normalize_input_data(launch_drone_list)
                carry_drone_list = normalize_input_data(carry_drone_list)
                recovery_drone_list = normalize_input_data(recovery_drone_list)
                # 判断列表是否为空
                if carry_drone_list is not None:
                    for drone_id in carry_drone_list:
                        detailed_vehicle_task_data[drone_id][node_id].drone_belong = id
                        detailed_vehicle_task_data[drone_id][node_id].add_node(node_id, arrive_time, departure_time)
                        detailed_vehicle_task_data[drone_id][node_id].prcise_arrive_time = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_arrive_time'] = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].prcise_departure_time = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_departure_time'] = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].arrive_times.append(arrive_time)
                        detailed_vehicle_task_data[drone_id][node_id].departure_times.append(arrive_time)
                if launch_drone_list is not None:
                    for drone_id in launch_drone_list:
                        detailed_vehicle_task_data[drone_id][node_id].drone_belong = id
                        detailed_vehicle_task_data[drone_id][node_id].add_node(node_id, arrive_time, departure_time)
                        detailed_vehicle_task_data[drone_id][node_id].prcise_arrive_time = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_arrive_time'] = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].prcise_departure_time = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_departure_time'] = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].arrive_times.append(arrive_time)
                        detailed_vehicle_task_data[drone_id][node_id].departure_times.append(arrive_time)
            else:
                arrive_time = vehicle_arrival_time[id][node_id]
                departure_time = arrive_time
                detailed_vehicle_task_data[id][node_id].add_node(node_id, arrive_time, departure_time)
                detailed_vehicle_task_data[id][node_id].prcise_arrive_time = arrive_time
                # detailed_vehicle_task_data[id][node_id].dict_vehicle[id].prcise_arrive_time = arrive_time
                detailed_vehicle_task_data[id][node_id].prcise_departure_time = departure_time
                # detailed_vehicle_task_data[id][node_id].dict_vehicle[id].prcise_departure_time = departure_time
                detailed_vehicle_task_data[id][node_id].arrive_times.append(arrive_time)
                # detailed_vehicle_task_data[id][node_id].dict_vehicle[id].arrive_times.append(arrive_time)
                detailed_vehicle_task_data[id][node_id].departure_times.append(departure_time)
                # detailed_vehicle_task_data[id][node_id].dict_vehicle[id].departure_times.append(departure_time)
                carry_drone_list = detailed_vehicle_task_data[id][node_id].drone_list
                launch_drone_list = detailed_vehicle_task_data[id][node_id].launch_drone_list
                recovery_drone_list = detailed_vehicle_task_data[id][node_id].recovery_drone_list
                carry_drone_list = normalize_input_data(carry_drone_list)
                launch_drone_list = normalize_input_data(launch_drone_list)
                recovery_drone_list = normalize_input_data(recovery_drone_list)
                if carry_drone_list is not None:
                    for drone_id in carry_drone_list:
                        detailed_vehicle_task_data[drone_id][node_id].drone_belong = id
                        detailed_vehicle_task_data[drone_id][node_id].add_node(node_id, arrive_time, departure_time)
                        detailed_vehicle_task_data[drone_id][node_id].prcise_arrive_time = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_arrive_time'] = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].prcise_departure_time = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_departure_time'] = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].arrive_times.append(arrive_time)
                        detailed_vehicle_task_data[drone_id][node_id].departure_times.append(arrive_time)
                if launch_drone_list is not None:
                    for drone_id in launch_drone_list:
                        detailed_vehicle_task_data[drone_id][node_id].drone_belong = id
                        detailed_vehicle_task_data[drone_id][node_id].add_node(node_id, arrive_time, departure_time)
                        detailed_vehicle_task_data[drone_id][node_id].prcise_arrive_time = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_arrive_time'] = arrive_time
                        detailed_vehicle_task_data[drone_id][node_id].prcise_departure_time = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].dict_vehicle[id]['prcise_departure_time'] = departure_time
                        detailed_vehicle_task_data[drone_id][node_id].arrive_times.append(arrive_time)
                        detailed_vehicle_task_data[drone_id][node_id].departure_times.append(departure_time)

    # 根据分配的任务，更新车辆任务
    for index, y_ijkd in enumerate(sorted_mission_keys):
        drone_id, launch_node, customer_node, recovery_node, launch_veh_id, recovery_veh_id = y_ijkd
        time_customer_plan[customer_node] = y_ijkd  # 各个客户点的服务顺序
        time_uav_task_dict[drone_id].append(y_ijkd)  # 各个无人机任务顺序
        launch_map_key = node[launch_node].map_key
        recovery_map_key = node[recovery_node].map_key
        truck_departure_time = detailed_vehicle_task_data[launch_veh_id][launch_node].prcise_departure_time
        drone_start_launch_time = truck_departure_time  # 无人机准备发射阶段
        drone_end_launch_time = drone_start_launch_time + vehicle[drone_id].launchTime
        # 添加该阶段任务描述,更新无人机发射阶段实况任务
        # detailed_vehicle_task_data[launch_veh_id][launch_node].add_task(launch_node, 13, drone_start_launch_time, drone_end_launch_time)
        detailed_vehicle_task_data[drone_id][launch_node].add_task(launch_node, 13, drone_start_launch_time, drone_end_launch_time)
        # detailed_vehicle_task_data[drone_id][launch_node].dict_vehicle[launch_veh_id]['task'].add_task(launch_node, 13, drone_start_launch_time, drone_end_launch_time)
        detailed_vehicle_task_data[launch_veh_id][launch_node].prcise_departure_time = drone_end_launch_time  # 添加车辆离开时间候选
        detailed_vehicle_task_data[drone_id][launch_node].prcise_departure_time = drone_end_launch_time  # 添加无人机离开时间候选
        detailed_vehicle_task_data[drone_id][launch_node].dict_vehicle[launch_veh_id]['prcise_departure_time'] = drone_end_launch_time  # 添加无人机离开时间候选
        detailed_vehicle_task_data[drone_id][launch_node].departure_times.append(drone_end_launch_time)  # 添加无人机离开时间候选
        detailed_vehicle_task_data[drone_id][launch_node].dict_vehicle[launch_veh_id]['departure_times'].append(drone_end_launch_time)  # 添加无人机离开时间候选
        detailed_vehicle_task_data[launch_veh_id][launch_node].departure_times.append(drone_end_launch_time)  # 添加车辆离开时间候选
        # 无人机飞行阶段-客户点情况,完成客户任务情况
        uav_to_c_time = uav_travel[drone_id][launch_map_key][customer_node].totalTime
        uav_arr_c_time = drone_end_launch_time + uav_to_c_time    
        detailed_vehicle_task_data[drone_id][launch_node].add_task(launch_node, 18, drone_end_launch_time, uav_arr_c_time)
        # detailed_vehicle_task_data[drone_id][launch_node].dict_vehicle[launch_veh_id]['task'].add_task(launch_node, 18, drone_end_launch_time, uav_arr_c_time)

        customer_service_time = node[customer_node].serviceTimeUAV
        uav_depart_customer_time = uav_arr_c_time + customer_service_time
        detailed_vehicle_task_data[drone_id][customer_node].add_task(customer_node, 15, uav_arr_c_time, uav_depart_customer_time)
        # detailed_vehicle_task_data[drone_id][customer_node].dict_vehicle[launch_veh_id]['task'].add_task(customer_node, 15, uav_arr_c_time, uav_depart_customer_time)
        detailed_vehicle_task_data[drone_id][customer_node].prcise_arrive_time = uav_arr_c_time
        detailed_vehicle_task_data[drone_id][customer_node].dict_vehicle[launch_veh_id]['prcise_arrive_time'] = uav_arr_c_time
        detailed_vehicle_task_data[drone_id][customer_node].prcise_departure_time = uav_depart_customer_time
        detailed_vehicle_task_data[drone_id][customer_node].dict_vehicle[launch_veh_id]['prcise_departure_time'] = uav_depart_customer_time
        detailed_vehicle_task_data[drone_id][customer_node].arrive_times.append(uav_arr_c_time)  # 到达客户时间列表
        detailed_vehicle_task_data[drone_id][customer_node].departure_times.append(uav_depart_customer_time)  # 离开客户时间列表
        detailed_vehicle_task_data[drone_id][customer_node].dict_vehicle[launch_veh_id]['arrive_times'].append(uav_arr_c_time)  # 到达客户时间列表
        detailed_vehicle_task_data[drone_id][customer_node].dict_vehicle[launch_veh_id]['departure_times'].append(uav_depart_customer_time)  # 离开客户时间列表
        # 无人机从客户点-返回情况
        uav_to_j_time = uav_travel[drone_id][customer_node][recovery_map_key].totalTime
        uav_arr_j_time = uav_depart_customer_time + uav_to_j_time
        detailed_vehicle_task_data[drone_id][recovery_node].add_task(recovery_node, 19, uav_depart_customer_time, uav_arr_j_time)
        # detailed_vehicle_task_data[drone_id][recovery_node].dict_vehicle[recovery_veh_id]['task'].add_task(recovery_node, 19, uav_depart_customer_time, uav_arr_j_time)
        detailed_vehicle_task_data[drone_id][recovery_node].arrive_times.append(uav_arr_j_time)
        detailed_vehicle_task_data[drone_id][recovery_node].dict_vehicle[recovery_veh_id]['arrive_times'].append(uav_arr_j_time)
        detailed_vehicle_task_data[drone_id][recovery_node].prcise_arrive_time = uav_arr_j_time
        detailed_vehicle_task_data[drone_id][recovery_node].dict_vehicle[recovery_veh_id]['prcise_arrive_time'] = uav_arr_j_time

        # 根据车辆路径，到达节点情况，更新后续detailed_vehicle_task_data的回收数据
        # delta_time = detailed_vehicle_task_data[launch_veh_id][launch_node].prcise_departure_time - detailed_vehicle_task_data[launch_veh_id][launch_node].prcise_arrive_time
        delta_time = detailed_vehicle_task_data[launch_veh_id][launch_node].prcise_departure_time - drone_start_launch_time
        detailed_vehicle_task_data = update_delta_time(delta_time, detailed_vehicle_task_data, best_vehicle_route, y_ijkd, vehicle)
    # 根据更新后的detailed_vehicle_task_data，更新best_customer_plan, best_uav_plan
    uav_plan_time = update_uav_plan(detailed_vehicle_task_data, best_uav_plan)  # 更新精确时间的
    # uav_plan_time = update_uav_plan(detailed_vehicle_task_data, sorted_mission_keys)  # 更新精确时间的
    vehicle_plan_time = update_vehicle_arrive_time(detailed_vehicle_task_data, vehicle_arrival_time)
    # 对uav_plan_time进行排序，按照launch_time排序
    sorted_uav_plan_time = sorted(uav_plan_time.items(), key=lambda item: item[1]['launch_time'])
    uav_plan_time = {k: v for k, v in sorted_uav_plan_time}

    return time_uav_task_dict, time_customer_plan, uav_plan_time, vehicle_plan_time, detailed_vehicle_task_data


# 设计基于滚动时间迭代的CBS算法，来获得精确的成本，以及精确的时间任务安排
def rolling_time_cbs(
    vehicle_arrive_time, best_vehicle_route, time_uav_task_dict, time_customer_plan, time_uav_plan, vehicle_plan_time, 
    vehicle_task_data, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, 
    veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, 
    air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee
):
    """
    修改后的函数：实现基于滚动时间迭代的CBS算法。
    """
    # 1. 初始化
    # 按发射时间对所有无人机任务进行排序
    sorted_missions = sorted(time_uav_plan.items(), key=lambda item: item[1]['launch_time'])
    unplanned_missions_q = deque(sorted_missions)
    # unplanned_missions_q = deque([item[0] for item in sorted_missions]) # 使用双端队列，方便高效地移除已规划任务
    current_batch = []
    # 全局状态变量
    global_reservation_table = {}  # 关键！用于存储已规划路径的时空占用信息 {(node, time): uav_id}  # {(node, time): mission_tuple}
    final_uav_plan = {}            # 用于存储所有无人机最终的、无冲突的详细路径 {mission_tuple: {'path': [...], 'cost': ...}}
    final_solution = {}            # {mission_tuple: {'path': [...], 'cost': ...}}
    best_uav_cost = {}
    cbs_solver = Time_cbs_Batch_Solver(global_reservation_table, time_uav_task_dict, time_customer_plan, time_uav_plan, vehicle_plan_time, vehicle_task_data, 
    node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, 
    A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types, 
    A_c, xeee)
    decimals = 4

    # 2. 滚动迭代主循环
    while unplanned_missions_q:
        # --- 步骤 A: 选择当前规划批次 ---
        anchor_mission_tuple, anchor_mission_dict = unplanned_missions_q[0]
        # 定义时间窗口来选择批次
        t_start = anchor_mission_dict['launch_time']
        # 估算锚点任务的结束时间，以确定窗口大小
        # t_end_window = anchor_mission_dict['recovery_time'] * 1.0
        # >>> FIX-A: 用“持续时长”定义窗口，而不是绝对recovery_time
        anchor_dur = max(0.5, anchor_mission_dict['recovery_time'] - anchor_mission_dict['launch_time'])
        ROLLING_FACTOR = 1.2  # 可调：1.0~1.5
        t_end_window = t_start + anchor_dur * ROLLING_FACTOR

        current_batch_missions = []
        MAX_BATCH_SIZE = 15  # 超参数：限制单次CBS处理的无人机数量，防止计算爆炸,批次规划中容纳的无人机属两个数
        selected_drone_ids = set()
        temp_q = unplanned_missions_q.copy()
        while temp_q and len(current_batch_missions) < MAX_BATCH_SIZE:  # 从队列的左侧移除并返回一个元素
            mission_tuple, mission_dict = temp_q.popleft()
            drone_id = mission_tuple[0]
            if mission_dict['launch_time'] <= t_end_window and drone_id not in selected_drone_ids:
                # 判断该架次无人机是否已经加入了当前的批次任务处理中
                current_batch_missions.append((mission_tuple, mission_dict))
                selected_drone_ids.add(drone_id)
            else:
                break
        
        batch_ids_for_log = [m[0] for m in current_batch_missions]
        # print(f"[Rolling Horizon] Planning batch for missions: {batch_ids_for_log}")

        # --- 步骤 B: 实例化并运行CBS求解器 ---
        # 创建一个新的CBS实例来解决当前批次的子问题
        # 注意：我们将当前批次的任务和全局预留表传入
        # 传入动态参数：批次任务 和 当前的全局预留表
        # 针对该批次，初始化其全局预留表
        # batch_result = cbs_solver.solve_for_batch(current_batch_missions, global_reservation_table)


        # # --- 步骤 C: 处理结果并更新全局状态 ---
        # if batch_result is None:
        #     # 异常处理：如果批次无解，缩小批次后重试
        #     print(f"Warning: CBS failed for batch. Retrying with anchor mission only.")
        #     current_batch_missions = [unplanned_missions_q[0]]
        #     batch_result = cbs_solver.solve_for_batch(current_batch_missions, global_reservation_table)

        #     if batch_result is None:
        #         raise RuntimeError(f"Fatal: Could not find a path even for a single mission: {anchor_mission_tuple}")

        # --- 步骤 B: 实例化并运行CBS求解器 ---
        # >>> FIX-B: 自适应缩小批次，避免反复timeout
        batch_result = None
        batch_try = list(current_batch_missions)
        timeout_s = 60

        while True:
            batch_result = cbs_solver.solve_for_batch(batch_try, global_reservation_table, timeout=timeout_s)
            if batch_result is not None:
                # 成功
                break

            # 失败：缩小批次
            if len(batch_try) <= 1:
                print("Warning: CBS failed even for single mission in this window.")
                # 这里不立刻fatal，交给下一个FIX（延后发射）处理
                break

            # print(f"Warning: CBS failed for batch size={len(batch_try)}. Shrinking batch and retrying...")
            batch_try = batch_try[: max(1, len(batch_try)//2)]
            timeout_s = max(15, int(timeout_s * 0.7))  # 可选：越小批次给更短超时
        if batch_result is None:
            # >>> FIX-B: 直接只处理窗口里的第一个任务“延后发射”，而不是死磕CBS
            anchor_mission_tuple, anchor_mission_dict = unplanned_missions_q[0]
            # 将anchor任务发射时间推迟一个TIME_STEP，再放回（简单但有效）
            anchor_mission_dict['launch_time'] += getattr(cbs_solver, "TIME_STEP", 0.05)
            # 重新排序队列（因为launch_time变了）
            tmp = list(unplanned_missions_q)
            tmp[0] = (anchor_mission_tuple, anchor_mission_dict)
            tmp.sort(key=lambda x: x[1]['launch_time'])
            unplanned_missions_q = deque(tmp)
            continue  # 回到while unplanned_missions_q

        # 候选备选，深拷贝
        best_uav_plan = {k: v.copy() for k, v in time_uav_plan.items()}
        # 如果成功，更新全局状态
        update_launch_task = {}
        update_recovery_task = {}
        for mission_tuple, path_data in batch_result.items():
            # 1. 存储最终解
            final_solution[mission_tuple] = path_data
            drone_id,launch_node, customer, recovery_node, launch_vehicle,recovery_vehicle = mission_tuple
            route_list = [item[0] for item in path_data]
            launch_vtp_node = node[route_list[0]].map_key
            recovery_vtp_node = node[route_list[-1]].map_key
            route_list.insert(0, launch_vtp_node)
            route_list.append(recovery_vtp_node)
            global_reservation_table = cbs_solver._update_reservation_table_with_path(global_reservation_table, path_data, drone_id, mission_tuple)
            best_uav_plan[mission_tuple]['uav_route'] = route_list
            launch_time, recovery_time = update_launch_recovery_time(node, path_data, vehicle, drone_id)
            if round(launch_time, decimals) != round(best_uav_plan[mission_tuple]['launch_time'], decimals):
                best_uav_plan[mission_tuple]['launch_time'] = launch_time
                update_launch_task[mission_tuple] = launch_time
                # print('发射时间产生改变。')
            if round(recovery_time, decimals) != round(best_uav_plan[mission_tuple]['recovery_time'], decimals):
                best_uav_plan[mission_tuple]['recovery_time'] = recovery_time
                update_recovery_task[mission_tuple] = recovery_time
                # print('回收时间产生改变。')
            work_time = recovery_time - launch_time
            best_uav_plan[mission_tuple]['time'] = work_time  # 从发射到服务到降落的整体无人机任务时间
            uav_route_cost, uav_time_cost = update_uav_cost(node, route_list, work_time, vehicle, drone_id, G_air, G_ground, mission_tuple) 
            best_uav_plan[mission_tuple]['uav_route_cost'] = uav_route_cost
            best_uav_plan[mission_tuple]['uav_time_cost'] = uav_time_cost
            best_uav_cost[customer] = uav_route_cost
        # 根据到达的时间排序，更新后续的detailed_vehicle_task_data,此时间更新后，需对应更新车辆的离开出发时间
        if update_recovery_task:
            sorted_recovery_task = sorted(update_recovery_task.items(), key=lambda item: item[1])
            for mission_tuple, recovery_time in sorted_recovery_task:
                drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle = mission_tuple
                delta_time = time_uav_plan[mission_tuple]['recovery_time'] - best_uav_plan[mission_tuple]['recovery_time']
                if delta_time > 0:  # 如果回收时间比原计划大，则需要更新detailed_vehicle_task_data
                    vehicle_task_data = update_delta_time(delta_time, vehicle_task_data, best_vehicle_route, mission_tuple, vehicle)
        
        # 3. 从待规划队列中移除已完成的任务 (使用deque的高效popleft)
        for _ in range(len(batch_result)):
            unplanned_missions_q.popleft()
    vehicle_plan_time = update_vehicle_arrive_time(vehicle_task_data, vehicle_arrive_time)
    print("[Rolling Horizon] All missions planned successfully.")
        
    # 根据您的函数签名，返回对应的数据结构
    return best_uav_plan, best_uav_cost, vehicle_plan_time, vehicle_task_data, global_reservation_table

def update_launch_recovery_time(node, path_data, vehicle, drone_id):
    launch_node = path_data[0][0]
    recovery_node = path_data[-1][0]
    air_launch_time = path_data[0][1]
    air_recovery_time = path_data[-1][1]
    init_node_alt = abs(node[launch_node].map_position[2] - node[launch_node].position[2])
    start_air_time = init_node_alt / vehicle[drone_id].takeoffSpeed
    end_node_alt = abs(node[recovery_node].map_position[2] - node[recovery_node].position[2])
    end_air_time = end_node_alt / vehicle[drone_id].landingSpeed
    launch_time = air_launch_time - start_air_time
    recovery_time = end_air_time + air_recovery_time
    return launch_time, recovery_time

def update_uav_cost(node, route_list, work_time, vehicle, drone_id, G_air, G_ground, mission_tuple):
    uav_route_cost = 0
    route_length = 0
    uav_time_cost = vehicle[drone_id].time_cost * work_time
    launch_node = route_list[0]
    recovery_node = route_list[-1]
    air_route_list = route_list[1:-1]
    for i in range(len(air_route_list) - 1):
        current_node = air_route_list[i]
        next_node = air_route_list[i+1]
        if current_node == next_node:  # 处理无人机在原节点等待从而避障的情况
            continue
        else:
            route_length += G_air[current_node][next_node]['weight']
    init_node_alt = abs(node[launch_node].map_position[2] - node[launch_node].position[2])
    end_node_alt = abs(node[recovery_node].map_position[2] - node[recovery_node].position[2])
    drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = mission_tuple
    # 还需考虑客户端的上下路线距离
    cust_node_alt = 2 * abs(node[customer_node].position[2] - node[customer_node].map_position[2])
    route_length += init_node_alt + end_node_alt + cust_node_alt  # 初始高度，降落高度，以及客户点的起飞和降落距离
    uav_route_cost = route_length * vehicle[drone_id].per_cost

    return uav_route_cost, uav_time_cost  # 分别输出以飞行距离和飞行成本为衡量的无人机任务成本

# 设计基于滚动时间迭代的CBS算法，来获得精确的成本，以及精确的时间任务安排
# --- 这里放入您提供的 detect_collision, standard_splitting 等函数 ---
# 我将根据您的参考代码重写/调整它们，使其适应我们的数据结构
def get_path_location(path, time):
    """ 获取无人机在特定时间的节点位置 """
    # path is a list of (node, time_at_node)
    if not path:
        return None
    
    # 如果时间小于路径开始时间，无人机还未起飞
    if time < path[0][1]:
        return None 
    
    # 找到第一个时间点大于或等于查询时间的路径点
    for i in range(len(path)):
        if path[i][1] >= time:
            # 如果是路径上的精确点
            if path[i][1] == time:
                return path[i][0]
            # 如果在两个路径点之间，说明在飞行中
            elif i > 0:
                # 简化处理：我们只关心节点上的冲突，所以认为它在前一个节点
                # 更精细的可以返回一个(edge, progress)元组
                return path[i-1][0] 
            else: # time is before the first move
                return path[0][0]

    # 如果时间超过了路径终点，则认为停留在终点
    return path[-1][0]

def detect_all_collisions(paths_dict):
    """
    检测一批路径中的所有首次冲突
    :param paths_dict: {uav_id: [(node, time), ...]}
    :return: list of collision dicts
    """
    collisions = []
    uav_ids = list(paths_dict.keys())
    
    max_time = 0
    for path in paths_dict.values():
        if path:
            max_time = max(max_time, path[-1][1])

    for uav1_id, uav2_id in itertools.combinations(uav_ids, 2):
        path1 = paths_dict[uav1_id]
        path2 = paths_dict[uav2_id]
        
        # 遍历所有时间步
        for t in range(max_time + 1):
            loc1 = get_path_location(path1, t)
            loc2 = get_path_location(path2, t)

            # 顶点冲突
            if loc1 is not None and loc1 == loc2:
                collisions.append({'uavs': [uav1_id, uav2_id], 'loc': [loc1], 'timestep': t, 'type': 'vertex'})
                break # 只找这对无人机间的第一个冲突

            # 边冲突
            next_loc1 = get_path_location(path1, t + 1)
            next_loc2 = get_path_location(path2, t + 1)
            if loc1 is not None and loc2 is not None and loc1 == next_loc2 and loc2 == next_loc1:
                collisions.append({'uavs': [uav1_id, uav2_id], 'loc': [loc1, loc2], 'timestep': t + 1, 'type': 'edge'})
                break # 只找这对无人机间的第一个冲突
        
    return collisions

def standard_splitting(collision):
    """ 为冲突生成约束 """
    constraints = []
    uav1, uav2 = collision['uavs']
    
    constraints.append({
        'agent': uav1,
        'loc': collision['loc'],
        'timestep': collision['timestep']
    })
    constraints.append({
        'agent': uav2,
        'loc': list(reversed(collision['loc'])) if len(collision['loc']) > 1 else collision['loc'],
        'timestep': collision['timestep']
    })
    return constraints

def get_sum_of_cost(paths_dict):
    """ 计算路径总成本 (例如，总飞行时间) """
    total_cost = 0
    for path in paths_dict.values():
        if path:
            total_cost += (path[-1][1] - path[0][1]) # 结束时间 - 开始时间
    return total_cost
    
# 主类
class Time_cbs_Batch_Solver:
    def __init__(self, global_reservation_table, time_uav_task_dict, time_customer_plan, time_uav_plan, vehicle_plan_time, vehicle_task_data, 
    node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, 
    A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types, 
    A_c, xeee):
        self._launch_delay_seen = set()  # >>> FIX-6
        self.global_reservation_table = global_reservation_table
        self.time_uav_task_dict = time_uav_task_dict
        self.time_customer_plan = time_customer_plan
        self.time_uav_plan = time_uav_plan
        self.vehicle_plan_time = vehicle_plan_time
        self.vehicle_task_data = vehicle_task_data
        self.node = node
        self.DEPOT_nodeID = DEPOT_nodeID
        self.V = V
        self.T = T
        self.vehicle = vehicle
        self.uav_travel = uav_travel
        self.veh_distance = veh_distance
        self.veh_travel = veh_travel
        self.N = N
        self.N_zero = N_zero
        self.N_plus = N_plus
        self.A_total = A_total
        self.A_cvtp = A_cvtp
        self.A_vtp = A_vtp
        self.A_aerial_relay_node = A_aerial_relay_node
        self.G_air = G_air
        self.G_ground = G_ground
        self.air_matrix = air_matrix
        self.ground_matrix = ground_matrix
        self.air_node_types = air_node_types
        self.ground_node_types = ground_node_types
        self.A_c = A_c
        self.xeee = xeee
        self.uav_ids = V
        self.truck_ids = T
        self.G_air = G_air
        # self.heuristics = {n: nx.single_source_dijkstra_path_length(G_air, n, weight='weight') for n in G_air.nodes()}
        # CBS内部数据结构
        self.open_list = []
        self.num_generated = 0
        self.num_expanded = 0
        self.EPSILON = 1e-4
        self.TIME_STEP = 0.05
        self.MAX_WAIT_SHIFT = 5
        
        # 预计算所有节点到所有节点的启发式距离
        self.heuristics = {}
        for node in self.G_air.nodes():
            self.heuristics[node] = nx.single_source_dijkstra_path_length(self.G_air, node, weight='weight')
    # >>> FIX-1: 预留表深拷贝（dict -> list -> dict）
    def _rt_deepcopy(self, rt):
        return {k: [d.copy() for d in v] for k, v in rt.items()}

    def solve_for_batch(self, batch_missions, reservation_table, timeout=60):
        """ 
        执行CBS算法，为当前批次找到无冲突路径。
        :param batch_missions: 当前批次的任务列表。
        :param reservation_table: 批次开始前已存在的预留表（例如来自更早批次的路径）。
        :param timeout: 算法的超时时间（秒）。
        :return: 包含所有无冲突路径的字典，或在失败/超时时返回 None。
        """
        start_time = time.time()
        MAX_EXPANSIONS = 20000 # >>> FIX-2: 防止无限扩展，先用这个量级
        expansions = 0
        # 使用局部变量，这是独立求解函数的标准做法
        open_list = []
        node_id_counter = 0

        # # 初始化，更新预留表，在该阶段中，更新任务中无人机在节点的起飞占用时间
        # reservation_table = self._init_global_reservation_table(batch_missions, reservation_table)

        # --- 第一阶段: 创建根节点 ---
        # 最佳实践：使用深拷贝处理输入的预留表，避免修改原始传入对象
        # 这在多次调用此函数时尤为重要
        # reservation_table_for_root = copy.copy(reservation_table)
        reservation_table_for_root = self._rt_deepcopy(reservation_table)
        
        root_paths = {}
        root_cost = 0
        
        # 对任务排序可以使初始解更稳定
        # sorted_missions = sorted(batch_missions, key=lambda x: x[0][0])
        sorted_missions = sorted(batch_missions, key=lambda x: x[1]['launch_time'])
        for mission_tuple, mission_dict in sorted_missions:
            drone_id = mission_tuple[0]

            # 在一个被逐步填充的预留表上进行贪心规划
            path, reservation_table_for_root = self._plan_full_mission_path(mission_tuple, mission_dict, [], 
            reservation_table_for_root)
            
            if path is None:
                print(f"错误: CBS初始化规划失败, 无人机 {drone_id} 的任务 {mission_tuple} 无解")
                return None
            
            root_paths[mission_tuple] = path
            # reservation_table_for_root = reservation_table
            # 成本是所有任务完成时间之和,此处应该至到达节点上方对应的空中中继点处
            completion_time = path[-1][1]
            root_cost += completion_time 

            # # 更新预留表，为本轮初始规划中的下一个无人机做准备,该更新任务更新任务过程中的预留表
            # self._update_reservation_table_with_path(reservation_table_for_root, path, drone_id)

        root_collisions = self._detect_collisions(root_paths, [])
        
        # root_node = {
        #     'cost': root_cost,
        #     'constraints': [],
        #     'paths': root_paths,
        #     'collisions': root_collisions
        # }
        root_node = {
            'cost': root_cost,
            'constraints': [],
            'constraint_keys': set(),   # >>> FIX-2A
            'paths': root_paths,
            'collisions': root_collisions
        }

        heapq.heappush(open_list, (root_node['cost'], node_id_counter, root_node))
        node_id_counter += 1

        # --- 第二阶段: CBS 主循环 ---
        while open_list:
            expansions += 1
            # >>> FIX-2: 超时/扩展上限
            if expansions > MAX_EXPANSIONS:
                print(f"[CBS] Abort: expansions>{MAX_EXPANSIONS}. Return None.")
                return None
            if time.time() - start_time > timeout:
                print(f"[CBS] Timeout {timeout}s. expansions={expansions} open={len(open_list)} "
                f"root_collisions={len(root_collisions) if 'root_collisions' in locals() else 'NA'} "
                f"constraints={len(current_node['constraints']) if 'current_node' in locals() else 'NA'}")
                print(f"[CBS] Timeout {timeout}s. Return None.")
                return None

            _, _, current_node = heapq.heappop(open_list)

            if not current_node['collisions']:
                # print(f"找到解决方案！成本: {current_node['cost']:.2f}，用时: {time.time() - start_time:.2f} 秒。")
                print(f"找到解决方案！成本: {current_node['cost']:.2f}。")
                return current_node['paths']

            collision = current_node['collisions'][0]
            # agents_in_conflict = collision['agents']
            participants = collision['participants']

            # 【修正】为每个参与者创建分支
            for participant in participants:
                agent_id = participant['agent_id']
                mission_tuple_to_replan = participant['mission_tuple']
                
                # 【修正】不再需要 next()! 直接从 participant 获取 mission_tuple
                # 我们还需要 mission_dict，这仍然需要从原始列表中查找，但这次使用完整的元组来确保唯一性
                mission_tuple, mission_dict = next((m for m in batch_missions if m[0] == mission_tuple_to_replan), (None, None))

                if mission_dict is None: continue # 安全检查

                new_constraint = self._create_constraint_from_collision(collision, agent_id, mission_tuple)
                # >>> FIX-S1: new_constraint 可能为 None，必须保护
                if new_constraint is None:
                    print("[CBS] Skip: cannot create constraint from collision type=",
                        collision.get('type'), "agent=", agent_id, "mission=", mission_tuple)
                    print("      collision keys:", list(collision.keys()))
                    if 'constraint' in collision:
                        print("      collision['constraint']=", collision['constraint'])
                    continue
                # >>> FIX-2A: 约束key（按时间步对齐）
                key = (new_constraint['agent_id'],
                    tuple(new_constraint['loc']),
                    round(new_constraint['t_start']/self.TIME_STEP),
                    round(new_constraint['t_end']/self.TIME_STEP),
                    new_constraint['kind'])

                if key in current_node.get('constraint_keys', set()):
                    continue

                child_constraints = current_node['constraints'] + [new_constraint]  # 父代约束 + 新约束
                child_keys = set(current_node.get('constraint_keys', set()))
                child_keys.add(key)

                # mission_tuple, mission_dict = next((m for m in batch_missions if m[0][0] == agent_id), None)
                
                # 为重新规划路径，重建一个临时的、干净的预留表。这是核心步骤。
                # 它基于最原始的预留表，加上当前CBS节点中其他无人机的路径。是否提前删除当前王人机任务?
                temp_reservation_table = self._rebuild_reservation_table(
                    reservation_table, # 使用最原始、未被修改的输入表
                    current_node['paths'], 
                    agent_id,
                    mission_tuple
                )
                
                new_path_tuple = self._plan_full_mission_path(
                    mission_tuple, 
                    mission_dict, 
                    child_constraints, # <--- 应用所有累积的约束
                    temp_reservation_table,
                )
                if new_path_tuple is None:
                    continue
                new_path, _ = new_path_tuple
                # ⭐关键：无路就别扩展这个子节点
                if new_path is None:
                    continue
                if new_path == current_node['paths'].get(mission_tuple):
                    continue
                # new_path, _ = new_path_tuple
                 # 4. 构建子节点的完整解
                child_paths = current_node['paths'].copy()
                child_paths[mission_tuple] = new_path
                 # 5. 计算新解的成本和冲突
                child_cost = self._calculate_solution_cost(child_paths)
                child_collisions = self._detect_collisions(child_paths, child_constraints)
                # 6. 创建子节点并放入OPEN list
                # child_node = {
                #     'cost': child_cost,
                #     'constraints': child_constraints,
                #     'paths': child_paths,
                #     'collisions': child_collisions
                # }
                child_node = {
                    'cost': child_cost,
                    'constraints': child_constraints,
                    'constraint_keys': child_keys,   # >>> FIX-2A
                    'paths': child_paths,
                    'collisions': child_collisions
                }
                heapq.heappush(open_list, (child_node['cost'], node_id_counter, child_node))
                node_id_counter += 1

        # print("CBS求解失败，未找到解决方案。")
        return None
        
    # =================================================
    # 3. CBS 所需的辅助函数
    # =================================================
    def _init_global_reservation_table(self, batch_missions, reservation_table):  # 如果降落速度不同的话，需要在添加一个参数，来阐述降落时间占用率
        """ 初始化全局预留表 """
        updated_table = copy.copy(reservation_table)
        # 遍历批次中的所有任务
        for mission_tuple, mission_dict in batch_missions:
            # 从任务信息中提取所需数据
            drone_id = mission_tuple[0]
            launch_node = mission_tuple[1]
            launch_time = mission_dict.get('launch_time', 0.0) # 使用 .get 提供默认值
            init_node_alt = abs(self.node[launch_node].map_position[2] - self.node[launch_node].position[2])
            start_air_time = init_node_alt / self.vehicle[drone_id].landSpeed
            end_air_time = start_air_time + launch_time


            # 如果起飞时间大于0，说明存在一段等待时间需要被预留
            if launch_time > 0:
                # 定义此无人机在起飞点的占用时间区间
                # 我们假设它从时间0就开始在起飞点等待
                occupancy_interval = {
                    'start': start_air_time,
                    'end': end_air_time,
                    'drone_id': drone_id
                }

                # 将此占用信息添加到预留表中
                # 如果节点首次出现，则为其初始化一个空列表
                if launch_node not in updated_table:
                    updated_table[launch_node] = []
                
                updated_table[launch_node].append(occupancy_interval)
        
        return updated_table
    

    def _rebuild_reservation_table(self, base_table, all_paths, agent_to_exclude, mission_to_exclude):
        """ 
        为重新规划路径创建一个临时的预留表。
        包含基础表和除被排除agent外的所有路径。
        """
        # 直接删除对应的元组任务
        temp_table = copy.deepcopy(base_table)
        for mission_tuple, path in all_paths.items():
            # drone_id = mission_tuple[0]
            # if drone_id != agent_to_exclude:
            #     self._update_reservation_table_with_path(temp_table, path, drone_id, mission_tuple)
            if mission_tuple != mission_to_exclude:
                drone_id = mission_tuple[0]
                self._update_reservation_table_with_path(temp_table, path, drone_id, mission_tuple)
        return temp_table

    def _calculate_solution_cost(self, paths_dict):
        """ 根据给定的路径集计算总成本 """
        total_cost = 0
        for mission_tuple, path in paths_dict.items():
            if not path: continue
            launch_time = path[0][1]
            completion_time = path[-1][1]
            total_cost += (completion_time - launch_time)
        return total_cost

    def _update_reservation_table_with_path(self, reservation_table, path, drone_id, mission_tuple):
        """
        >>> FIX-RT: 1) 早退返回rt 2) land也合并 3) merge按type区分 4) touched_nodes末尾一次性merge（性能提升）
        """
        if not path or len(path) < 2:
            return reservation_table

        EPS = getattr(self, "EPSILON", 1e-4)
        touched_nodes = set()

        def _append_only(node, itv):
            reservation_table.setdefault(node, []).append(itv)
            touched_nodes.add(node)

        def _merge_node(node):
            lst = reservation_table.get(node, [])
            if len(lst) <= 1:
                return
            lst.sort(key=lambda x: x['start'])
            merged = []
            for cur in lst:
                if not merged:
                    merged.append(cur)
                    continue
                last = merged[-1]
                same_owner = (
                    cur.get('drone_id') == last.get('drone_id')
                    and cur.get('mission_tuple') == last.get('mission_tuple')
                    and cur.get('type') == last.get('type')   # >>> FIX: 防止flight/land误合并
                )
                if same_owner and cur['start'] <= last['end'] + EPS:
                    last['end'] = max(last['end'], cur['end'])
                    if 'arrive_time' in cur:
                        last['arrive_time'] = max(last.get('arrive_time', last['end']), cur['arrive_time'])
                else:
                    merged.append(cur)
            reservation_table[node] = merged

        # --- 阶段1: climb ---
        launch_node, launch_ready_time = path[0]
        climb_altitude = abs(self.node[launch_node].map_position[2] - self.node[launch_node].position[2])
        climb_time = climb_altitude / self.vehicle[drone_id].takeoffSpeed
        climb_start_time = launch_ready_time - climb_time

        _append_only(launch_node, {
            'start': climb_start_time,
            'end': launch_ready_time,
            'arrive_time': launch_ready_time,
            'drone_id': drone_id,
            'mission_tuple': mission_tuple,
            'type': 'climb'
        })

        # --- 阶段2: flight ---
        for i in range(len(path) - 1):
            start_node, start_time = path[i]
            end_node, end_time = path[i + 1]
            if start_time >= end_time:
                continue

            # 你当前策略：航段内占用起点&终点（保守但更难解）
            nodes_to_reserve = [start_node, end_node]

            for idx, node in enumerate(nodes_to_reserve):
                _append_only(node, {
                    'start': start_time,
                    'end': end_time,
                    'arrive_time': (start_time if idx == 0 else end_time),
                    'drone_id': drone_id,
                    'mission_tuple': mission_tuple,
                    'type': 'flight'
                })

        # --- 阶段3: land ---
        land_node, land_arrival_time = path[-1]
        land_altitude = abs(self.node[land_node].map_position[2] - self.node[land_node].position[2])
        land_time = land_altitude / self.vehicle[drone_id].landingSpeed
        land_complete_time = land_arrival_time + land_time

        _append_only(land_node, {
            'start': land_arrival_time,
            'end': land_complete_time,
            'arrive_time': land_complete_time,
            'drone_id': drone_id,
            'mission_tuple': mission_tuple,
            'type': 'land'
        })

        # >>> FIX: 函数末尾统一merge，避免每次append都sort
        for n in touched_nodes:
            _merge_node(n)

        return reservation_table

    # def _update_reservation_table_with_path(self, reservation_table, path, drone_id, mission_tuple):
    #     """
    #     用一条规划好的、包含物理意义的路径来更新预留表。

    #     该函数精确处理三个阶段的时空占用：
    #     1. 初始爬升：在第一个节点，从地面爬升到飞行高度的时间。
    #     2. 水平飞行：在航线各段之间，飞行占用的时间。同时考虑了客户点占用时间窗情况
    #     3. 最终降落：在最后一个节点，从飞行高度下降到地面的时间。

    #     :param reservation_table: 要更新的预留表。
    #     :param path: [(node, time), ...] 格式的路径。其中time是到达节点飞行高度的时刻。
    #     :param drone_id: 占用这些时空资源的无人机ID。
    #     """
    #     # 路径至少需要一个起点和一个终点才能形成航段
    #     # if not path or len(path) < 2:
    #     #     return
    #     if not path or len(path) < 2:
    #         return reservation_table  # >>> FIX-3: 必须返回 rt，避免外层赋成 None
    #     EPS = 1e-4  # >>> FIX-3
    #     def _append_and_merge(node, itv):
    #         reservation_table.setdefault(node, []).append(itv)
    #         lst = reservation_table[node]
    #         lst.sort(key=lambda x: x['start'])

    #         merged = []
    #         for cur in lst:
    #             if not merged:
    #                 merged.append(cur)
    #                 continue
    #             last = merged[-1]
    #             same_owner = (cur.get('drone_id') == last.get('drone_id')
    #                         and cur.get('mission_tuple') == last.get('mission_tuple'))
    #             if same_owner and cur['start'] <= last['end'] + EPS:
    #                 last['end'] = max(last['end'], cur['end'])
    #                 # arrive_time 保留更大的那个（可选）
    #                 if 'arrive_time' in cur:
    #                     last['arrive_time'] = max(last.get('arrive_time', last['end']), cur['arrive_time'])
    #             else:
    #                 merged.append(cur)
    #         reservation_table[node] = merged
    #     # --- 阶段1: 处理初始起飞阶段的垂直爬升占用 ---
    #     launch_node, launch_ready_time = path[0]
        
    #     # 假设 self.vehicle[drone_id] 拥有 climbSpeed 和 landSpeed 属性
    #     # 计算从地面爬升到飞行高度所需的时间
    #     climb_altitude = abs(self.node[launch_node].map_position[2] - self.node[launch_node].position[2])
    #     climb_time = climb_altitude / self.vehicle[drone_id].takeoffSpeed
        
    #     # 实际从地面开始爬升的时间 = 到达飞行高度的时刻 - 爬升耗时
    #     climb_start_time = launch_ready_time - climb_time
        
    #     # 在起飞点预留垂直爬升占用的时间段
    #     if launch_node not in reservation_table:
    #         reservation_table[launch_node] = []
    #     # reservation_table[launch_node].append({
    #     #     'start': climb_start_time,
    #     #     'end': launch_ready_time,
    #     #     'arrive_time': launch_ready_time,
    #     #     'drone_id': drone_id,
    #     #     'mission_tuple': mission_tuple,
    #     #     'type': 'climb' # 添加类型方便调试
    #     # })
    #     _append_and_merge(launch_node, {
    #         'start': climb_start_time,
    #         'end': launch_ready_time,
    #         'arrive_time': launch_ready_time,
    #         'drone_id': drone_id,
    #         'mission_tuple': mission_tuple,
    #         'type': 'climb'
    #     })

    #     # --- 阶段2: 处理所有水平飞行航段的占用 ---
    #     for i in range(len(path) - 1):
    #         start_node, start_time = path[i]
    #         end_node, end_time = path[i+1]

    #         if start_time >= end_time:
    #             continue
            
    #         # 为了安全，保守地认为在整个飞行航段 [start_time, end_time] 内，
    #         # 起点和终点两个节点都被占用。这可以防止另一架无人机在终点等待时发生冲突。该出可以表示服务节点的vtp节点也是被占用的情况
    #         nodes_to_reserve = [start_node, end_node]
            
    #         for index, node in enumerate(nodes_to_reserve):
    #             if node not in reservation_table:
    #                 reservation_table[node] = []
    #             if index == 0:
    #                 # reservation_table[node].append({
    #                 #     'start': start_time, 
    #                 #     'end': end_time, 
    #                 #     'arrive_time': start_time,
    #                 #     'drone_id': drone_id,
    #                 #     'mission_tuple': mission_tuple,
    #                 #     'type': 'flight' # 添加类型方便调试
    #                 # })
    #                 _append_and_merge(node, {
    #                     'start': start_time, 
    #                     'end': end_time, 
    #                     'arrive_time': start_time,
    #                     'drone_id': drone_id,
    #                     'mission_tuple': mission_tuple,
    #                     'type': 'flight' # 添加类型方便调试
    #                 })
    #             else:
    #                 _append_and_merge(node, {    
    #                     'start': start_time, 
    #                     'end': end_time, 
    #                     'arrive_time': end_time,
    #                     'drone_id': drone_id,
    #                     'mission_tuple': mission_tuple,
    #                     'type': 'flight' # 添加类型方便调试
    #                 })

    #     # --- 阶段3: 处理最终降落阶段的垂直下降占用 ---
    #     land_node, land_arrival_time = path[-1]
        
    #     # 计算从飞行高度下降到地面所需的时间
    #     land_altitude = abs(self.node[land_node].map_position[2] - self.node[land_node].position[2])
    #     land_time = land_altitude / self.vehicle[drone_id].landingSpeed
        
    #     # 实际在地面完成降落的时刻 = 到达飞行高度的时刻 + 降落耗时
    #     land_complete_time = land_arrival_time + land_time
        
    #     # 在降落点预留垂直下降占用的时间段
    #     if land_node not in reservation_table:
    #         reservation_table[land_node] = []
    #     reservation_table[land_node].append({
    #         'start': land_arrival_time,
    #         'end': land_complete_time,
    #         'arrive_time': land_complete_time,
    #         'drone_id': drone_id,
    #         'mission_tuple': mission_tuple,
    #         'type': 'land' # 添加类型方便调试
    #     })
    #     return reservation_table
    # 注意：函数直接修改传入的 reservation_table 对象，所以不需要返回

    # 将节点按代价、冲突数和生成顺序加入优先队列。
    def push_node(self, node):
        # 优先队列：(cost, num_collisions, generation_id, node_data)
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_generated, node))
        self.num_generated += 1

    # 从优先队列中弹出代价最小的节点。
    def pop_node(self):
        _, _, _, node = heapq.heappop(self.open_list)
        self.num_expanded += 1
        return node

    def _plan_full_mission_path(self, mission_tuple, mission_dict, constraints, reservation_table):
        """
        为单个无人机规划完整的两段式任务路径 (Launch -> Customer -> Recovery)。
        """
        drone_id, launch_node, customer_node, recovery_node, _, _ = mission_tuple
        service_time = self.node[customer_node].serviceTimeUAV # 假设客户点的服务时间为1个时间单位

        # =========================================================
        # 【新增修正 1】：动态调整发射时间 (解决发射点堵塞问题)
        # =========================================================
        adjusted_launch_time = mission_dict['launch_time']
        
        # 获取 launch node 的 map key，确保查找的是空中节点
        launch_air_node = self.node[launch_node].map_key
        
        # 查找 launch_air_node 上的预留信息
        launch_intervals = reservation_table.get(launch_air_node, [])
        
        # 按时间排序，便于检查
        launch_intervals.sort(key=lambda x: x['start']) 
        
        # for interval in launch_intervals:
        #     # 跳过本无人机自己的预留
        #     if interval['drone_id'] == drone_id: continue
            
        #     # 检查预期的发射时间是否落在其他无人机的占用区间 [start, end)
        #     # 使用 self.EPSILON 进行浮点数比较，避免边界问题
        #     if interval['start'] <= adjusted_launch_time < interval['end'] - self.EPSILON:
        #         # 冲突！将发射时间推迟到冲突结束之后
        #         print(f"[CBS Launch Delay] Drone {drone_id} launch delayed from {adjusted_launch_time:.4f} to {interval['end']:.4f} due to Drone {interval['drone_id']} at node {launch_air_node}.")
        #         adjusted_launch_time = interval['end'] + self.EPSILON # 推迟到结束时间 + 极小缓冲
            
        #     # 由于间隔已排序，如果当前发射时间晚于当前间隔的结束，可以跳过后续检查（假设间隔不重叠）
        #     if adjusted_launch_time < interval['start']:
        #         break

        adjusted_launch_time = self._next_free_time(
            launch_air_node,
            mission_dict['launch_time'],
            reservation_table,
            eps=self.EPSILON,
            ignore_drone_id=None   # 我建议这里不要忽略自己，能避免同一无人机跨批次自撞时间线
        )
        # if adjusted_launch_time > mission_dict['launch_time'] + self.EPSILON:
        #     print(f"[CBS Launch Delay] Drone {drone_id} launch delayed from {mission_dict['launch_time']:.4f} to {adjusted_launch_time:.4f} at node {launch_air_node}.")
        if adjusted_launch_time > mission_dict['launch_time'] + self.EPSILON:
            key = (drone_id, launch_air_node,
                round(mission_dict['launch_time'], 4),
                round(adjusted_launch_time, 4))
            if key not in self._launch_delay_seen:
                self._launch_delay_seen.add(key)
                # print(f"[CBS Launch Delay] Drone {drone_id} launch delayed from {mission_dict['launch_time']:.4f} "
                    # f"to {adjusted_launch_time:.4f} at node {launch_air_node}.")
        # >>> FIX-C
        mission_dict['launch_time'] = adjusted_launch_time

        current_launch_time = adjusted_launch_time
        
        # ------------------
        # 路径第一段: 发射点 -> 客户点
        # ------------------
        climb_altitude = abs(self.node[launch_node].map_position[2] - self.node[launch_node].position[2])
        climb_time = climb_altitude / self.vehicle[drone_id].takeoffSpeed
        
        # A* 搜索的开始时间：修正后的发射时间 + 爬升到空中节点的时间
        path1_start_time = current_launch_time + climb_time
        
        path1_start_node = self.node[launch_node].map_key # 空中节点
        
        # =========================================================
        # 【修正 2】：A* 调用使用动态调整后的时间
        # =========================================================
        path1 = self.spatio_temporal_a_star(
            self.G_air,
            path1_start_node, 
            customer_node,
            path1_start_time,  # 使用修正后的启动时间
            self.heuristics[customer_node], 
            constraints,
            reservation_table,
            drone_id
        )

        if not path1:
            print(f"找不到可行路径！Debug: Path1 failed for {drone_id}")
            return None, None # 第一段就找不到路
        
        # 1. 计算各项时间
        arrival_time_in_air = path1[-1][1] 

        # 计算降落耗时
        land_altitude = abs(self.node[customer_node].map_position[2] - self.node[customer_node].position[2])
        land_duration = land_altitude / self.vehicle[drone_id].landingSpeed
        
        # 计算爬升耗时
        climb_duration = land_altitude / self.vehicle[drone_id].takeoffSpeed

        # 计算第二段路径的真正起飞时刻
        # 起飞时刻 = 空中到达时刻 + 降落耗时 + 地面服务耗时 + 爬升耗时
        departure_time_for_path2 = arrival_time_in_air + land_duration + service_time + climb_duration

        # 更新预留表
        # self._update_reservation_table_with_path(reservation_table, path1, drone_id, mission_tuple)
        # 更新预留表
        reservation_table = self._update_reservation_table_with_path(  # >>> FIX-4: 统一接返回值
            reservation_table, path1, drone_id, mission_tuple
        )

        # 2. 为第二段路径准备一个高度精确的临时预留表
        temp_reservation_table = self._rt_deepcopy(reservation_table)  # >>> FIX-4: 深拷贝，禁止浅拷贝
        
        # 2. 为第二段路径准备一个高度精确的临时预留表
        # temp_reservation_table = copy.copy(reservation_table)

        
        # 为第二段路径创建一个临时的预留表，包含服务时间
        if customer_node not in temp_reservation_table:
            temp_reservation_table[customer_node] = []
            
        temp_reservation_table[customer_node].append({
            'start': arrival_time_in_air + land_duration,
            'end': arrival_time_in_air + land_duration + service_time,
            'arrive_time': arrival_time_in_air + land_duration,
            'drone_id': drone_id,
            'type': 'service_stop' 
        })

        path2 = self.spatio_temporal_a_star(
            self.G_air,
            customer_node,
            self.node[recovery_node].map_key,
            departure_time_for_path2, # 第二段的开始时间
            self.heuristics[self.node[recovery_node].map_key], 
            constraints,
            temp_reservation_table,
            drone_id
        )
        
        if not path2:
            return None, None # 第二段找不到路
        
        # self._update_reservation_table_with_path(temp_reservation_table, path2, drone_id, mission_tuple)
        temp_reservation_table = self._update_reservation_table_with_path(  # >>> FIX-4
            temp_reservation_table, path2, drone_id, mission_tuple
        )
        
        # ------------------
        # 合并路径并返回
        # ------------------
        full_path = path1 + path2[1:] 
        
        return full_path, temp_reservation_table

    # def _plan_full_mission_path(self, mission_tuple, mission_dict, constraints, reservation_table):
    #         """
    #         为单个无人机规划完整的两段式任务路径 (Launch -> Customer -> Recovery)。
    #         """
    #         drone_id, launch_node, customer_node, recovery_node, _, _ = mission_tuple
    #         service_time = self.node[customer_node].serviceTimeUAV # 假设客户点的服务时间为1个时间单位，可以设为参数
    #         # --- 修正：动态调整发射时间 ---
    #         # 检查 launch_node 在 launch_time 是否被占用
    #         adjusted_launch_time = mission_dict['launch_time']

    #         # ------------------
    #         # 路径第一段: 发射点 -> 客户点
    #         # ------------------
    #         climb_altitude = abs(self.node[launch_node].map_position[2] - self.node[launch_node].position[2])
    #         climb_time = climb_altitude / self.vehicle[drone_id].takeoffSpeed
    #         path1 = self.spatio_temporal_a_star(
    #             self.G_air,
    #             self.node[launch_node].map_key,  # 使用原始节点ID
    #             customer_node,
    #             mission_dict['launch_time'] + climb_time,  # 更新精确的无人机到达空中节点的时间状况
    #             self.heuristics[customer_node], # 启发函数需要目标是客户点
    #             constraints,
    #             reservation_table,
    #             drone_id
    #         )

    #         if not path1:
    #             print(f"找不到可行路径！Debug: Path1 failed for {drone_id}")
    #             return None, None # 第一段就找不到路
            
    #         # 1. 计算各项时间
    #         arrival_time_in_air = path1[-1][1]  # 到达客户点上空，准备降落的时刻

    #         # 计算降落耗时
    #         land_altitude = abs(self.node[customer_node].map_position[2] - self.node[customer_node].position[2])
    #         land_duration = land_altitude / self.vehicle[drone_id].landingSpeed
            
    #         # 计算爬升耗时
    #         climb_duration = land_altitude / self.vehicle[drone_id].takeoffSpeed

    #         # 计算第二段路径的真正起飞时刻
    #         # 起飞时刻 = 空中到达时刻 + 降落耗时 + 地面服务耗时 + 爬升耗时
    #         departure_time_for_path2 = arrival_time_in_air + land_duration + service_time + climb_duration

    #         # 注意：这里我们应该调用统一的、精确的更新函数来完成
    #         # reservation_table = self._update_reservation_table_with_path(reservation_table, path1, drone_id, mission_tuple)
    #         self._update_reservation_table_with_path(reservation_table, path1, drone_id, mission_tuple)
    #         # 2. 为第二段路径准备一个高度精确的临时预留表
    #         temp_reservation_table = copy.copy(reservation_table)
            
    #         # 为第二段路径创建一个临时的预留表，包含第一段的路径和服务时间
    #         if customer_node not in temp_reservation_table:
    #             temp_reservation_table[customer_node] = []
                
    #         temp_reservation_table[customer_node].append({
    #             'start': arrival_time_in_air + land_duration,
    #             'end': arrival_time_in_air + land_duration + service_time,
    #             'arrive_time': arrival_time_in_air + land_duration,
    #             'drone_id': drone_id,
    #             'type': 'service_stop' # 标记类型，方便调试
    #         })

    #         path2 = self.spatio_temporal_a_star(
    #             self.G_air,
    #             customer_node,
    #             self.node[recovery_node].map_key,
    #             departure_time_for_path2 , # 第二段的开始时间
    #             self.heuristics[self.node[recovery_node].map_key], # 启发函数需要目标是回收点
    #             constraints,
    #             temp_reservation_table,
    #             drone_id
    #         )
            
    #         if not path2:
    #             # print(f"Debug: Path2 failed for {drone_id}")
    #             return None, None # 第二段找不到路
            
    #         # temp_reservation_table = self._update_reservation_table_with_path(temp_reservation_table, path2, drone_id, mission_tuple)
    #         self._update_reservation_table_with_path(temp_reservation_table, path2, drone_id, mission_tuple)
    #         # ------------------
    #         # 合并路径并返回
    #         # ------------------
    #         # 完整路径 = path1 + 在客户点等待的时间 + path2 (去掉重复的客户点)
    #         full_path = path1 + path2[1:]  # 第一个path1代表在第一次到达cus中的空中air的到达时间，第二个path2代表发射完成准备执行任务的path2
    #         # 更新预留表，以及无人机到达各个节点的到达时间
    #         return full_path, temp_reservation_table

    def _detect_collisions(self, paths, constraints):
        """
        检测给定路径集中的所有冲突，包括顶点冲突和边冲突。
        """
        all_collisions = []
        
        # --- 1. 检测路径间的物理冲突 (顶点冲突和边冲突) ---
        temp_reservation_table = {}
        for mission_tuple, path in paths.items():
            drone_id = mission_tuple[0]
            self._update_reservation_table_with_path(temp_reservation_table, path, drone_id, mission_tuple)

        # 1a. 检测顶点冲突 (通过扫描预留表)
        for node, intervals in temp_reservation_table.items():
            for interval1, interval2 in itertools.combinations(intervals, 2):
                if interval1['drone_id'] == interval2['drone_id']:
                    continue
                time_diff = abs(interval1['arrive_time'] - interval2['arrive_time'])
                if time_diff < 1e-6:
                    collision_time = interval1['arrive_time']
                    colliding_agents = sorted([interval1['drone_id'], interval2['drone_id']])
                    collision = {
                        'type': 'vertex',
                        'loc': [node],
                        'time': collision_time,
                        'participants': [
                            {'agent_id': interval1['drone_id'], 'mission_tuple': interval1['mission_tuple']},
                            {'agent_id': interval2['drone_id'], 'mission_tuple': interval2['mission_tuple']}
                        ]
                    }
                    # 去重逻辑：确保同一对无人机在同一点的冲突只报告一次
                    is_duplicate = any(
                        c['type'] == 'vertex' and c['loc'][0] == node and 
                        set(p['agent_id'] for p in c['participants']) == {interval1['drone_id'], interval2['drone_id']}
                        for c in all_collisions
                    )
                    if not is_duplicate:
                        all_collisions.append(collision)

        # 1b. 检测边冲突 (交换冲突)
        path_items = list(paths.items())
        for (mission1, path1), (mission2, path2) in itertools.combinations(path_items, 2):
            drone1_id = mission1[0]
            drone2_id = mission2[0]

            for i in range(len(path1) - 1):
                p1_start_node, p1_start_time = path1[i]
                p1_end_node, p1_end_time = path1[i+1]

                for j in range(len(path2) - 1):
                    p2_start_node, p2_start_time = path2[j]
                    p2_end_node, p2_end_time = path2[j+1]

                    # 空间条件：检查是否为相向移动
                    is_swapping = (p1_start_node == p2_end_node and p1_end_node == p2_start_node)

                    if not is_swapping:
                        continue

                    # 时间条件：检查两个航段的时间区间是否重叠
                    time_overlap = max(p1_start_time, p2_start_time) < min(p1_end_time, p2_end_time)

                    if time_overlap:
                        collision = {
                        'type': 'edge',
                        'participants': [
                            {'agent_id': drone1_id, 'mission_tuple': mission1, 'loc': [p1_start_node, p1_end_node], 'time': p1_end_time},
                            {'agent_id': drone2_id, 'mission_tuple': mission2, 'loc': [p2_start_node, p2_end_node], 'time': p2_end_time}
                        ]}
                        # 去重逻辑：检查同一对无人机是否已在该条无向边上报告过冲突
                        nodes_involved = frozenset([p1_start_node, p1_end_node])
                        is_duplicate = any(
                            c['type'] == 'edge' and 
                            set(p['agent_id'] for p in c['participants']) == {drone1_id, drone2_id} and
                            frozenset(c['participants'][0]['loc']) == nodes_involved
                            for c in all_collisions
                        )
                        if not is_duplicate:
                            all_collisions.append(collision)
        # --- 2. 检测是否违反CBS约束 ---
        for constraint in constraints:
            # 从约束中直接获取所有必要信息
            agent_id = constraint['agent']
            constrained_mission_tuple = constraint['mission_tuple']
            constrained_loc = constraint['loc']
            constrained_time = constraint['time']
            
            # 【关键修正】直接通过 mission_tuple 获取路径，避免了 next() 的BUG
            path = paths.get(constrained_mission_tuple)
            if not path: continue

            # 检查顶点约束
            if len(constrained_loc) == 1:
                node_to_check = constrained_loc[0]
                for node, time in path:
                    if node == node_to_check and abs(time - constrained_time) < 1e-9:
                        violation = {
                            'type': 'constraint_violation',
                            'constraint': constraint, # 记录违反了哪条具体约束
                            'participants': [{'agent_id': agent_id, 'mission_tuple': constrained_mission_tuple}]
                        }
                        if violation not in all_collisions:
                            all_collisions.append(violation)
                        break # 已找到违规，检查下一条约束
            
            # 检查边约束
            elif len(constrained_loc) == 2:
                [c_start, c_end] = constrained_loc
                for i in range(len(path) - 1):
                    p_start, _ = path[i]
                    p_end, p_end_time = path[i+1]
                    if p_start == c_start and p_end == c_end and abs(p_end_time - constrained_time) < 1e-9:
                        violation = {
                            'type': 'constraint_violation',
                            'constraint': constraint,
                            'participants': [{'agent_id': agent_id, 'mission_tuple': constrained_mission_tuple}]
                        }
                        if violation not in all_collisions:
                            all_collisions.append(violation)
                        break # 已找到违规，检查下一条约束
        return all_collisions

    def _create_constraint_from_collision(self, collision, agent_id_to_constrain, mission_tuple):
        """
        >>> FIX-C1: 将“点约束”升级为“区间约束”，并保持向后兼容
        - 旧字段：agent / mission_tuple / loc / time / type   （你现有A*还能用）
        - 新字段：agent_id / kind / t_start / t_end           （用于区间命中，解决constraints爆炸）
        """
        import math

        TS  = getattr(self, "TIME_STEP", 0.05)     # 你的统一时间步长
        EPS = getattr(self, "EPSILON", 1e-4)

        def q_down(t):  # 向下对齐到 TIME_STEP
            return math.floor((t + EPS) / TS) * TS

        def q_up(t):    # 向上对齐到 TIME_STEP
            return math.ceil((t - EPS) / TS) * TS

        collision_type = collision.get('type', None)
        if collision_type == 'constraint_violation':
            c = collision.get('constraint', None)
            if not c:
                return None

            cid = c.get('agent_id', c.get('agent', None))
            if cid != agent_id_to_constrain:
                return None

            loc  = c.get('loc', None)
            kind = c.get('kind', c.get('type', None))
            if not loc or kind not in ('vertex', 'edge'):
                return None

            # 补齐区间字段（你collision里其实已经有了）
            t0 = c.get('t_start', c.get('time', None))
            t1 = c.get('t_end', None)
            if t0 is None:
                return None
            if t1 is None:
                t1 = t0 + TS

            t0, t1 = q_down(t0), q_up(t1)
            if t1 <= t0 + EPS:
                t1 = t0 + TS

            # 返回“统一格式”（兼容旧字段 + 新字段）
            return {
                'agent': cid,
                'mission_tuple': c.get('mission_tuple', mission_tuple),
                'loc': loc,
                'time': c.get('time', t0),
                'type': kind,

                'agent_id': cid,
                'kind': kind,
                't_start': t0,
                't_end': t1,
            }
        if collision_type not in ('vertex', 'edge'):
            return None

        # -------- 顶点冲突：collision['loc'] = [node], collision['time'] = t --------
        if collision_type == 'vertex':
            loc = collision.get('loc', None)
            if not loc:
                return None

            t = collision.get('time', None)
            if t is None:
                return None

            # >>> FIX-C1: 统一成区间约束 [t_start, t_end)
            # 如果你未来在 detect_collisions 里能给出真实重叠区间，可以优先用它
            t0 = collision.get('t_start', t)
            t1 = collision.get('t_end', t + TS)
            t0, t1 = q_down(t0), q_up(t1)
            if t1 <= t0 + EPS:
                t1 = t0 + TS

            return {
                # --- 旧字段（兼容你现有代码） ---
                'agent': agent_id_to_constrain,
                'mission_tuple': mission_tuple,
                'loc': loc,
                'time': t,
                'type': 'vertex',

                # --- 新字段（区间版，给A*用） ---
                'agent_id': agent_id_to_constrain,
                'kind': 'vertex',
                't_start': t0,
                't_end': t1,
            }

        # -------- 边冲突：你已经在 participants 里存了每个agent的loc/time --------
        else:  # collision_type == 'edge'
            participants = collision.get('participants', [])
            constraint_details = next(
                (p for p in participants if p.get('agent_id') == agent_id_to_constrain),
                None
            )
            if not constraint_details:
                return None

            loc = constraint_details.get('loc', None)   # [u, v]
            t   = constraint_details.get('time', None)  # 到达v的时刻
            if not loc or t is None:
                return None

            # >>> FIX-C1: 区间约束
            # 先用简单版：禁止在 [t, t+TS) 到达这条边的终点（足够止血）
            # 如果 collision 未来给出 t_start/t_end，则优先使用
            t0 = constraint_details.get('t_start', collision.get('t_start', t))
            t1 = constraint_details.get('t_end',   collision.get('t_end',   t + TS))
            t0, t1 = q_down(t0), q_up(t1)
            if t1 <= t0 + EPS:
                t1 = t0 + TS

            return {
                # --- 旧字段（兼容） ---
                'agent': agent_id_to_constrain,
                'mission_tuple': mission_tuple,
                'loc': loc,
                'time': t,
                'type': 'edge',

                # --- 新字段（区间版） ---
                'agent_id': agent_id_to_constrain,
                'kind': 'edge',
                't_start': t0,
                't_end': t1,
            }

    # def _create_constraint_from_collision(self, collision, agent_id_to_constrain, mission_tuple):

    #     collision_type = collision['type']

    #     if collision_type == 'vertex':
    #         # 顶点冲突，位置是共享的，时间也是共享的
    #         return {
    #             'agent': agent_id_to_constrain,
    #             'mission_tuple': mission_tuple, # 传入的 mission_tuple
    #             'loc': collision['loc'],         # 冲突发生的单个节点 [node]
    #             'time': collision['time'],       # 冲突发生的时刻
    #             'type': 'vertex'
    #         }
            
    #     elif collision_type == 'edge':
    #         # 【【【核心修正】】】
    #         # 我们不再查找 'details' 键。
    #         # 而是遍历 'participants' 列表，找到与我们当前要约束的 agent_id 匹配的那一项。
            
    #         # 使用 next() 和生成器表达式安全地查找参与者信息
    #         constraint_details = next(
    #             (p for p in collision['participants'] if p['agent_id'] == agent_id_to_constrain), 
    #             None # 如果没找到（理论上不可能），返回None
    #         )

    #         # 安全检查：如果由于某种原因没有找到对应的参与者，则不创建约束
    #         if not constraint_details:
    #             return None

    #         # 从找到的参与者信息中提取 loc 和 time
    #         # 'loc' 是该智能体移动的有向边，例如 [68, 19]
    #         # 'time' 是该智能体完成该移动（即到达终点）的时刻
    #         return {
    #             'agent': agent_id_to_constrain,
    #             'mission_tuple': mission_tuple,
    #             'loc': constraint_details['loc'],  # 正确的有向 loc
    #             'time': constraint_details['time'], # 该智能体到达目标点的时间
    #             'type': 'edge'
    #         }
            
    #     else:
    #         # 其他或未知的冲突类型，例如 'constraint_violation'
    #         # 这部分逻辑可以根据需要进行扩展
    #         return None

    def spatio_temporal_a_star(self, graph, start_node, goal_node, launch_time,
                            heuristic, constraints, reservation_table, drone_id):
        import math
        hit = 0
        open_list = []
        came_from = {}

        TIME_STEP = self.TIME_STEP           # >>> FIX-5
        EPSILON   = self.EPSILON
        MAX_SHIFT = self.MAX_WAIT_SHIFT      # >>> FIX-5

        def _quantize_up(t):
            return math.ceil(t / TIME_STEP) * TIME_STEP

        launch_time = _quantize_up(launch_time)

        heapq.heappush(
            open_list,
            (launch_time + heuristic[start_node] / self.vehicle[drone_id].cruiseSpeed,
            launch_time,
            (launch_time, start_node))
        )

        g_score = {(launch_time, start_node): launch_time}

        while open_list:
            _, current_g, current_state = heapq.heappop(open_list)
            current_time, current_node = current_state

            if current_g > g_score.get(current_state, float('inf')):
                continue

            if current_node == goal_node:
                path = []
                temp = current_state
                while temp in came_from:
                    path.append((temp[1], temp[0]))
                    temp = came_from[temp]
                path.append((start_node, launch_time))
                return path[::-1]

            all_possible_neighbors = list(graph.neighbors(current_node)) + [current_node]

            for neighbor in all_possible_neighbors:
                # 1) travel time
                if neighbor == current_node:
                    travel_time = TIME_STEP  # >>> FIX-5: 统一等待步长
                else:
                    edge_data = graph.get_edge_data(current_node, neighbor)
                    distance = edge_data.get('weight', 1)
                    travel_time = distance / self.vehicle[drone_id].cruiseSpeed

                earliest_arrival_time = _quantize_up(current_time + travel_time)
                safe_arrival_time = earliest_arrival_time

                # 2) 反复处理：CBS约束 -> 预留表跳转
                ok = False
                for _ in range(10):
                    # A) CBS 约束（点/边）
                    conflict = False
                    push_to = safe_arrival_time
                    # >>> FIX-C2: 约束命中：优先区间，其次点
                    for c in constraints:
                        # 只约束当前无人机（如果你不想过滤也可以删掉这一段）
                        if c.get('agent_id', c.get('agent')) not in (None, drone_id):
                            continue
                        # hit += 1
                        # print("[A*] drone", drone_id, "node", neighbor, "t", safe_arrival_time, "constraint_hits", hit)

                        loc = c.get('loc', [])
                        # 1) 区间约束
                        if 't_start' in c and 't_end' in c:
                            t0, t1 = c['t_start'], c['t_end']

                            # vertex
                            if len(loc) == 1 and loc[0] == neighbor:
                                if t0 - EPSILON <= safe_arrival_time < t1 - EPSILON:
                                    conflict = True
                                    push_to = max(push_to, t1)

                            # edge
                            if neighbor != current_node and len(loc) == 2 and loc[0] == current_node and loc[1] == neighbor:
                                if t0 - EPSILON <= safe_arrival_time < t1 - EPSILON:
                                    conflict = True
                                    push_to = max(push_to, t1)

                        # 2) 旧点约束（回退）
                        else:
                            if len(loc) == 1 and loc[0] == neighbor:
                                if abs(c['time'] - safe_arrival_time) < EPSILON:
                                    conflict = True
                                    push_to = max(push_to, c['time'] + TIME_STEP)
                            if neighbor != current_node and len(loc) == 2 and loc == [current_node, neighbor]:
                                if abs(c['time'] - safe_arrival_time) < EPSILON:
                                    conflict = True
                                    push_to = max(push_to, c['time'] + TIME_STEP)

                    if conflict:
                        safe_arrival_time = _quantize_up(push_to + EPSILON)
                        if safe_arrival_time - earliest_arrival_time > MAX_SHIFT:
                            break
                        continue
                    # for c in constraints:
                    #     if len(c['loc']) == 1 and c['loc'][0] == neighbor:
                    #         if abs(c['time'] - safe_arrival_time) < EPSILON:
                    #             conflict = True
                    #             push_to = max(push_to, c['time'] + TIME_STEP)

                    #     if neighbor != current_node and len(c['loc']) == 2:
                    #         if c['loc'] == [current_node, neighbor] and abs(c['time'] - safe_arrival_time) < EPSILON:
                    #             conflict = True
                    #             push_to = max(push_to, c['time'] + TIME_STEP)

                    # if conflict:
                    #     safe_arrival_time = _quantize_up(push_to + EPSILON)
                    #     if safe_arrival_time - earliest_arrival_time > MAX_SHIFT:
                    #         break
                    #     continue

                    # B) 预留表：直接跳到空闲时间
                    safe_arrival_time = _quantize_up(
                        self._next_free_time(
                            neighbor, safe_arrival_time, reservation_table,
                            eps=EPSILON, ignore_drone_id=drone_id
                        )
                    )

                    if safe_arrival_time - earliest_arrival_time > MAX_SHIFT:
                        break

                    # 再检查一次是否落在 interval 内（保险）
                    intervals = reservation_table.get(neighbor, [])
                    in_conflict = False
                    for itv in intervals:
                        if itv.get('drone_id') == drone_id:
                            continue
                        if itv['start'] - EPSILON <= safe_arrival_time < itv['end'] - EPSILON:
                            in_conflict = True
                            safe_arrival_time = _quantize_up(itv['end'] + EPSILON)
                            break

                    if not in_conflict:
                        ok = True
                        break

                if not ok:
                    continue

                next_state = (safe_arrival_time, neighbor)
                new_g = safe_arrival_time

                if new_g < g_score.get(next_state, float('inf')):
                    g_score[next_state] = new_g
                    h = heuristic.get(neighbor, 0) / self.vehicle[drone_id].cruiseSpeed
                    f = new_g + h
                    heapq.heappush(open_list, (f, new_g, next_state))
                    came_from[next_state] = current_state

        return None

    # def spatio_temporal_a_star(self, graph, start_node, goal_node, launch_time,
    #                        heuristic, constraints, reservation_table, drone_id):
    #     """
    #     时空A*搜索算法，为单个无人机规划路径。
    #     """
    #     open_list = []
    #     came_from = {}

    #     # =====【新增/建议】统一时间粒度，避免 0.02 / 0.05 / EPS 混战导致碎片化 =====
    #     TIME_STEP = 0.05          # <<<【改动1】建议统一成 0.05（你CBS里也用0.05）
    #     EPSILON   = 1e-4          # 你原来是 1e-4，先不动

    #     # 起点入队
    #     heapq.heappush(
    #         open_list,
    #         (launch_time + heuristic[start_node] / self.vehicle[drone_id].cruiseSpeed,
    #         launch_time,
    #         (launch_time, start_node))
    #     )

    #     # 注意：状态用 (time, node)
    #     g_score = {(launch_time, start_node): launch_time}

    #     while open_list:
    #         _, current_g, current_state = heapq.heappop(open_list)
    #         current_time, current_node = current_state

    #         if current_g > g_score.get(current_state, float('inf')):
    #             continue

    #         # 终点
    #         if current_node == goal_node:
    #             path = []
    #             temp = current_state
    #             while temp in came_from:
    #                 path.append((temp[1], temp[0]))  # (node, time)
    #                 temp = came_from[temp]
    #             path.append((start_node, launch_time))
    #             return path[::-1]

    #         # 探索邻居（含原地等待）
    #         all_possible_neighbors = list(graph.neighbors(current_node)) + [current_node]

    #         for neighbor in all_possible_neighbors:
    #             # --- 1) 计算基础 travel_time ---
    #             if neighbor == current_node:
    #                 travel_time = TIME_STEP  # <<<【改动2】原来是 0.02，改成统一 TIME_STEP
    #             else:
    #                 edge_data = graph.get_edge_data(current_node, neighbor)
    #                 distance = edge_data.get('weight', 1)
    #                 travel_time = distance / self.vehicle[drone_id].cruiseSpeed

    #             earliest_arrival_time = current_time + travel_time

    #             # --- 2) 计算 safe_arrival_time：先避开CBS约束，再跳过预留表占用 ---
    #             safe_arrival_time = earliest_arrival_time

    #             # <<<【改动3】把你原来的 while(MAX_LOOPS) 精简成“有限次调整”，避免碎片化卡死
    #             MAX_ADJUST_ITERS = 10
    #             adjust_ok = False

    #             for _ in range(MAX_ADJUST_ITERS):
    #                 conflict_found = False
    #                 max_end = safe_arrival_time  # 用于推进

    #                 # A. CBS 约束检查：顶点/边（仍按你原来的“点约束”逻辑）
    #                 for c in constraints:
    #                     # 顶点约束
    #                     if len(c['loc']) == 1 and c['loc'][0] == neighbor:
    #                         if abs(c['time'] - safe_arrival_time) < EPSILON:
    #                             conflict_found = True
    #                             max_end = max(max_end, c['time'] + TIME_STEP)

    #                     # 边约束（移动才检查）
    #                     if neighbor != current_node and len(c['loc']) == 2:
    #                         if c['loc'] == [current_node, neighbor] and abs(c['time'] - safe_arrival_time) < EPSILON:
    #                             conflict_found = True
    #                             max_end = max(max_end, c['time'] + TIME_STEP)

    #                 if conflict_found:
    #                     safe_arrival_time = max_end + EPSILON
    #                     continue

    #                 # B. 预留表：一行跳到下一个空闲时间点（你的核心修复点）
    #                 new_t = self._next_free_time(
    #                     node_id=neighbor,
    #                     t=safe_arrival_time,
    #                     reservation_table=reservation_table,
    #                     eps=EPSILON,
    #                     ignore_drone_id=drone_id
    #                 )

    #                 # 如果没变化，说明已经安全
    #                 if abs(new_t - safe_arrival_time) < EPSILON:
    #                     adjust_ok = True
    #                     break

    #                 # 否则更新继续检查（因为跳完后有可能正好撞上CBS点约束）
    #                 safe_arrival_time = new_t

    #             if not adjust_ok:
    #                 # <<<【改动4】放弃该 neighbor（当前时刻不可达）
    #                 if neighbor == 65:  # 你日志里最炸的点
    #                     intervals = reservation_table.get(neighbor, [])
    #                     print(f"[A* ADJUST FAIL] drone={drone_id} node={current_node}->{neighbor} "
    #                             f"t0={earliest_arrival_time:.4f} stuck={safe_arrival_time:.4f} "
    #                             f"intervals={len(intervals)} constraints={len(constraints)}")
    #                 continue

    #             # --- 3) 更新路径信息 ---
    #             next_state = (safe_arrival_time, neighbor)
    #             new_g = safe_arrival_time

    #             if new_g < g_score.get(next_state, float('inf')):
    #                 g_score[next_state] = new_g
    #                 h_value = heuristic.get(neighbor, 0) / self.vehicle[drone_id].cruiseSpeed
    #                 f_value = new_g + h_value

    #                 heapq.heappush(open_list, (f_value, new_g, next_state))
    #                 came_from[next_state] = current_state

    #     return None  # 未找到路径


    # # (这个函数需要放在Time_cbs类之外，或者作为其静态方法)
    # def spatio_temporal_a_star(self, graph, start_node, goal_node, launch_time, heuristic, constraints, reservation_table, drone_id):
    #     """
    #     时空A*搜索算法，为单个无人机规划路径。
    #     :param graph: NetworkX图 (G_air)
    #     :param start_node: 起始节点
    #     :param goal_node: 目标节点
    #     :param launch_time: 无人机可用的最早发射时间
    #     :param heuristic: 启发函数（从各节点到目标节点的最短路径字典）
    #     :param constraints: CBS高层传递的该无人机的约束列表
    #     :param reservation_table: 全局时空预留表
    #     :return: [(node, time), ...], or None
    #     """
    #     open_list = []
    #     # 状态: (f_value, g_value, (time, node))
    #     # start_node = self.node[start_node].map_key
    #     # goal_node = self.node[goal_node].map_key,开始节点均输入为空中网络节点，cbs只负责空中层次网络规划，通过预留表给每个节点的起降位置占用节点时间执行更新
    #     heapq.heappush(open_list, (launch_time + heuristic[start_node]/self.vehicle[drone_id].cruiseSpeed, launch_time, (launch_time, start_node)))
    #     came_from = {}
    #     g_score = { (launch_time, start_node): launch_time }
    #     # 定义微小时间量，用于浮点数比较和避让
    #     EPSILON = 1e-4
    #     while open_list:
    #         _, current_g, current_state = heapq.heappop(open_list)
    #         current_time, current_node = current_state

    #         if current_g > g_score.get(current_state, float('inf')):
    #             continue
    #         if current_node == goal_node:
    #             # 重建路径
    #             path = []
    #             temp = current_state
    #             while temp in came_from:
    #                 path.append((temp[1], temp[0])) # (node, time)
    #                 temp = came_from[temp]
    #             path.append((start_node, launch_time))
    #             return path[::-1]
    #         # 探索邻居节点
    #         # 1. 移动到邻居
    #         all_possible_neighbors = list(graph.neighbors(current_node)) + [current_node]
    #         for neighbor in all_possible_neighbors:
    #             # --- 1. 计算基础旅行时间和到达时间 ---
    #             if neighbor == current_node:
    #                 # 这是“原地等待”动作
    #                 travel_time = 0.02  # 假设等待的最小时间步长为1，此处表明原地等待0.02s
    #             else:
    #                 # 这是“移动到邻居”动作
    #                 edge_data = graph.get_edge_data(current_node, neighbor)
    #                 distance = edge_data.get('weight', 1)
    #                 travel_time = distance / self.vehicle[drone_id].cruiseSpeed

    #             earliest_arrival_time = current_time + travel_time
    #             # --- 2. 解决冲突，计算实际可行的到达时间 ---
    #             # 从最早可能到达的时间开始，向未来搜索一个没有冲突的时间点
    #             safe_arrival_time = earliest_arrival_time
    #             loop_count = 0
    #             MAX_LOOPS = 50 # 限制单次节点扩展的最大尝试次数，防止死循环
    #             while loop_count < MAX_LOOPS:
    #                 loop_count += 1
    #                 found_conflict = False
    #                 max_conflict_end_time = -1

    #                 # A. 检查 CBS 约束
    #                 # 建议：如果 CBS 约束很多，这里应该优化为二分查找或区间树，而不是线性遍历
    #                 for c in constraints:
    #                     # 检查顶点约束 (Loc: [u], Time: t)
    #                     # 使用 abs < EPSILON 处理浮点相等
    #                     if len(c['loc']) == 1 and c['loc'][0] == neighbor:
    #                         if abs(c['time'] - safe_arrival_time) < EPSILON:
    #                             found_conflict = True
    #                             # 如果是点约束，只推迟一点点；如果是区间约束，推迟到区间结束
    #                             max_conflict_end_time = max(max_conflict_end_time, c['time'] + 0.05) # 假设离散时间步长至少 0.05

    #                     # 检查边约束 (Loc: [u, v], Time: t)
    #                     if neighbor != current_node and len(c['loc']) == 2:
    #                         if c['loc'] == [current_node, neighbor] and abs(c['time'] - safe_arrival_time) < EPSILON:
    #                             found_conflict = True
    #                             max_conflict_end_time = max(max_conflict_end_time, c['time'] + 0.05)

    #                 # B. 检查预留表 (Reservation Table)
    #                 reserved_intervals = reservation_table.get(neighbor, [])
    #                 for interval in reserved_intervals:
    #                     if interval['drone_id'] == drone_id:
    #                         continue
                        
    #                     start_t, end_t = interval['start'], interval['end']
                        
    #                     # 检查时间重叠：[start, end)
    #                     # 使用宽松的判定：start - eps <= t < end - eps
    #                     if start_t - EPSILON <= safe_arrival_time < end_t - EPSILON:
    #                         found_conflict = True
    #                         # 关键修复：直接跳到冲突结束时间，而不是 += 0.02
    #                         max_conflict_end_time = max(max_conflict_end_time, end_t)

    #                 if found_conflict:
    #                     # 智能跳转：直接跳到所有已知冲突的最晚结束时间
    #                     # 并加上一个极小值确保不落在边界上
    #                     if max_conflict_end_time > safe_arrival_time:
    #                         safe_arrival_time = max_conflict_end_time + EPSILON
    #                     else:
    #                         # 如果 max_conflict_end_time 没有更新（比如仅有点约束），则手动步进
    #                         safe_arrival_time += 0.05 
    #                     continue # 重新检查新时间点是否安全
    #                 else:
    #                     # 没有发现任何冲突，该时间点安全
    #                     break
                
    #             if loop_count >= MAX_LOOPS:
    #                 # 如果尝试了太多次仍无解，视为该邻居不可达（当前时刻）
    #                 continue
                    
    #                 # 更新 safe_arrival_time 为躲过CBS约束后的时间
    #                 safe_arrival_time = temp_arrival_time

    #                 # 2. 检查 reservation_table (智能跳转)
    #                 # 获取该邻居节点的所有预留时间段
    #                 reserved_intervals = reservation_table.get(neighbor, [])
    #                 reservation_conflict_found = False

    #                 for interval_dict in reserved_intervals:
    #                     # 从字典中获取 start 和 end 时间
    #                     start_t = interval_dict['start']
    #                     end_t = interval_dict['end']
    #                     if interval_dict['drone_id'] == drone_id:
    #                         continue
    #                     # 检查我们的到达时间是否落入任何一个预留区间。
    #                     # 占用区间通常定义为 [start, end)，即开始时刻占用，结束时刻不占用。
    #                     # 这允许一个任务在t=10结束，另一个任务在t=10开始。
    #                     if start_t <= safe_arrival_time < end_t:
    #                         # 冲突！记录下这次冲突的结束时间
    #                         last_conflict_end_time = max(last_conflict_end_time, end_t)
    #                         reservation_conflict_found = True

    #                 if reservation_conflict_found or last_conflict_end_time != -1:
    #                     if cbs_conflict_found:
    #                             # safe_arrival_time += 0.02 
    #                             safe_arrival_time += 1
    #                     # 如果是预留表冲突（时间段），可以智能跳转到冲突结束的时刻
    #                     if reservation_conflict_found:
    #                         safe_arrival_time = max(safe_arrival_time, last_conflict_end_time)
                        
    #                     continue # 返回 while True 的开头，用新的 safe_arrival_time 重新检查所有约束
    #                 # 如果代码能运行到这里，说明当前 safe_arrival_time 是安全的
    #                 break

    #             # --- 3. 更新路径信息 ---
    #             next_state = (safe_arrival_time, neighbor)
    #             new_g = safe_arrival_time # g值就是到达时间

    #             if new_g < g_score.get(next_state, float('inf')):
    #                 g_score[next_state] = new_g
    #                 h_value = heuristic.get(neighbor, 0) / self.vehicle[drone_id].cruiseSpeed
    #                 f_value = new_g + h_value
    #                 heapq.heappush(open_list, (f_value, new_g, next_state))
                    
    #                 # 记录路径来源。注意：来源状态是 current_state，而不是等待前的某个状态
    #                 came_from[next_state] = current_state
    #     return None # 未找到路径
    #     print('无人机的低空城市走廊规划任务中找到了无解的路径，请检查任务后重新规划')

    def _next_free_time(self, node_id, t, reservation_table, eps=1e-4, ignore_drone_id=None):
        intervals = reservation_table.get(node_id, [])
        if not intervals:
            return t

        intervals = sorted(intervals, key=lambda x: x['start'])
        tt = t
        for it in intervals:
            if ignore_drone_id is not None and it.get('drone_id') == ignore_drone_id:
                continue
            s, e = it['start'], it['end']

            # 完全在当前时间之前
            if e <= tt + eps:
                continue
            # 下一段占用还没开始，tt 已经安全
            if tt < s - eps:
                break
            # 落在占用内：直接跳到占用结束
            if s - eps <= tt < e - eps:
                tt = e + eps

        return tt


import math
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# 检查方案插入同一个车辆中是否满足约束条件
def check_same_vehicle_conflict(v_id, drone_id, i_vtp, j_vtp, 
                               solution_route, solution, vehicle_task_data, vehicle):
    """同一车辆冲突检测"""
    v_idx = v_id - 1
    i_idx = solution_route[v_idx].index(i_vtp)
    j_idx = solution_route[v_idx].index(j_vtp)
    
    # 基本顺序检查
    if i_idx >= j_idx:
        return False
    # 判断开始阶段无人机是否在车辆上
    if drone_id not in vehicle_task_data[v_id][i_vtp].drone_list:
        return False
    # 中间节点冲突检测
    route_segment = solution_route[v_idx][i_idx:j_idx+1]
    # for node in route_segment[1:-1]:  # 排除首尾节点
    for node in route_segment:
        task_data = vehicle_task_data[v_id][node]
        if drone_id in task_data.launch_drone_list or drone_id in task_data.recovery_drone_list:
            return False
    
    # 容量检查
    if len(vehicle_task_data[v_id][j_vtp].drone_list) >= vehicle[v_id].maxDrones:
        return False
    
    # 时间顺序验证
    if solution[v_id][i_vtp] > solution[v_id][j_vtp]:
        return False
    
    return True

# 检查将方案插入到不同的车辆中是否满足约束条件
def check_cross_vehicle_conflict_fixed(launch_v_id, recover_v_id, drone_id,
                                       i_vtp, j_vtp, solution_route, solution,
                                       vehicle_task_data, launch_time, recovery_time, vehicle):
    """跨车辆冲突检测系统（修正版）"""
    # =================================================================
    # 第一部分：发射车辆验证
    # =================================================================
    launch_veh_route = solution_route[launch_v_id - 1]
    i_idx = launch_veh_route.index(i_vtp)
    # 判断开始阶段无人机是否在车辆上
    if drone_id not in vehicle_task_data[launch_v_id][i_vtp].drone_list:
        return False

    # 1.1 检查发射车后续任务，确保无人机生命周期闭环
    for post_idx in range(i_idx + 1, len(launch_veh_route)-1): # -1是确保最后一个节点是车辆返回depot，不执行发射回收任务
        post_node = launch_veh_route[post_idx]
        post_task = vehicle_task_data[launch_v_id][post_node]

        # 如果后续又发射了此无人机
        if drone_id in post_task.launch_drone_list:
            has_intermediate_recovery = False
            # 必须确保在这两次发射之间，无人机被回收过
            route_slice = launch_veh_route[i_idx + 1:post_idx+1]
            for idx, mid_node in enumerate(route_slice):  # 一直到后续最后的节点
                if idx == len(route_slice) - 1:# 最后一个节点情况，如果先回收车辆，在发射则满足约束
                    if drone_id in vehicle_task_data[launch_v_id][mid_node].recovery_drone_list:
                        has_intermediate_recovery = True
                        break
                if drone_id in vehicle_task_data[launch_v_id][mid_node].recovery_drone_list:
                    # 本车上的回收时间，不能早于计划中在另一台车上的回收时间
                    if recovery_time > solution[launch_v_id][mid_node]:
                        return False
                    has_intermediate_recovery = True
                    break
            if not has_intermediate_recovery:
                return False  # 错误：无人机未回收就再次发射

        # 如果后续在本车又重复回收了此无人机
        if drone_id in post_task.recovery_drone_list:
            # 本车上的回收时间，不能早于计划中在另一台车上的回收时间
            if solution[launch_v_id][post_node] < recovery_time:
                return False  # 错误：时间逻辑冲突

    # =================================================================
    # 第二部分：回收车辆验证 (核心逻辑修正)
    # =================================================================
    recover_veh_route = solution_route[recover_v_id - 1]
    j_idx = recover_veh_route.index(j_vtp)
    # 如果无人机已经在回收车辆上，则不符合约束
    if drone_id in vehicle_task_data[recover_v_id][j_vtp].drone_list or drone_id in vehicle_task_data[recover_v_id][j_vtp].recovery_drone_list or \
        len(vehicle_task_data[recover_v_id][j_vtp].drone_list) >= vehicle[recover_v_id].maxDrones:
        return False

    # 2.1 逆向检测：检查回收点之前的任务
    for pre_idx in range(j_idx - 1, -1, -1):
        pre_node = recover_veh_route[pre_idx]
        pre_task = vehicle_task_data[recover_v_id][pre_node]

        # 如果之前的任务是发射此无人机
        if drone_id in pre_task.launch_drone_list:
            # 检查从那次发射到当前回收之间，是否已经有过回收
            has_intermediate_recovery = False
            for mid_idx in range(pre_idx + 1, j_idx - 1):
                mid_node = recover_veh_route[mid_idx]
                if drone_id in vehicle_task_data[recover_v_id][mid_node].recovery_drone_list:
                    has_intermediate_recovery = True
                    break
            if has_intermediate_recovery:
                return False  # 错误：发射->回收->再回收，无人机已被回收
            break  # 找到了最近的发射点，无需再往前找

        # 如果之前的任务是回收此无人机
        if drone_id in pre_task.launch_drone_list:
            # 那么前序回收必须发生在当前发射之前
            if solution[recover_v_id][pre_node] > launch_time:
                 return False # 错误: 前序回收比当前发射还晚
            break # 找到了最近的回收点，无需再往前找

    # 2.2 正向检测：检查回收点之后的任务
    for post_idx in range(j_idx + 1, len(recover_veh_route)-1):
        post_node = recover_veh_route[post_idx]
        post_task = vehicle_task_data[recover_v_id][post_node]

        # 如果之后的任务是发射此无人机
        if drone_id in post_task.launch_drone_list:
            # # 必须确保当前回收早于未来发射
            # if recovery_time > solution[recover_v_id][post_node]:
            #     return False
            # 且中间不能再有回收任务
            for mid_node in recover_veh_route[j_idx+1:post_idx+1]:
                if drone_id in vehicle_task_data[recover_v_id][mid_node].recovery_drone_list:
                    return False # 错误：回收 -> 回收 -> 发射
            break # 找到了最近的发射点，无需再往后找
        
        # 如果之后的任务是回收此无人机
        if drone_id in post_task.launch_drone_list:
            # 必须确保当前回收早于未来发射
            if recovery_time > solution[recover_v_id][post_node]:
                return False
            break # 找到了最近的回收点，无需再往后找

    # =================================================================
    # 第三部分：容量验证
    # =================================================================
    recover_task_info = vehicle_task_data[recover_v_id][j_vtp]
    # 注意：这里的 drone_list 应该指在到达该点时，车上已有的无人机列表
    if len(recover_task_info.drone_list) >= vehicle[recover_v_id].maxDrones:
        return False  # 错误：回收车辆容量不足

    return True


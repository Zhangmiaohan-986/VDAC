import numpy as np
import random
import copy
import math
import gurobipy as gp

def greedy_insert_feasible_plan(un_visit_customer, vehicle_route, vehicle_arrival_time, vehicle_task_data, best_customer_plan):
    """
    贪婪插入可行方案
    :param un_visit_customer: 未访问客户点
    :param vehicle_route: 车辆路线
    :param vehicle_arrival_time: 车辆到达时间
    :param vehicle_task_data: 车辆任务数据
    :return: 最优方案
    """
    # 1. 计算每个无人机任务的时间差值
    time_diff = {}
    for customer, plan in best_customer_plan.items():
        drone_id, launch_node, _, recovery_node, launch_vehicle, recovery_vehicle = plan
        launch_time = vehicle_arrival_time[launch_vehicle][launch_node]
        recovery_time = vehicle_arrival_time[recovery_vehicle][recovery_node]
        time_diff[customer] = recovery_time - launch_time
    
    # 2. 按时间差值从大到小排序客户点
    sorted_customers = sorted(time_diff.items(), key=lambda x: x[1], reverse=True)

    best_cost = float('inf')
    best_plan = None
    best_original_customer = None
    # 5. 遍历排序后的客户点（从时间差值最大的开始）
    for customer, time_difference in sorted_customers:
        # 获取原始任务信息
        drone_id, orig_launch_node, _, orig_recovery_node, launch_vehicle, recovery_vehicle = best_customer_plan[customer]
        launch_idx = launch_vehicle - 1
        recovery_idx = recovery_vehicle -1 
        if launch_vehicle == recovery_vehicle:
            segment = vehicle_route[launch_idx:recovery_idx+1]  # 获得车辆任务路径
        else:
            # 不同车辆：获取整个车辆路线
            segment = vehicle_route




        



    return None, None, None, None, None



import numpy as np
import random
import copy
import math
import gurobipy as gp

def greedy_insert_feasible_plan(self, un_visit_customer, vehicle_route, vehicle_arrival_time, vehicle_task_data, best_customer_plan):
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
    plan_cost = {}
    plan_y = {}
    plan_time = {}
    plan_uav_route = {}
    # 5. 遍历排序后的客户点（从时间差值最大的开始）
    for customer, time_difference in sorted_customers:
        # 获取原始任务信息
        drone_id, orig_launch_node, _, orig_recovery_node, launch_vehicle, recovery_vehicle = best_customer_plan[customer]
        # test_vehicle_task_data = copy.deepcopy(vehicle_task_data)
        test_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)
        # 更新vehicle_task_data
        remove_vehicle_task_data = remove_vehicle_task(test_vehicle_task_data, best_customer_plan[customer], vehicle_route)
        launch_vehicle_index = launch_vehicle - 1
        recovery_vehicle_index = recovery_vehicle -1
        launch_node_index = vehicle_route[launch_vehicle_index].index(orig_launch_node)
        recovery_node_index = vehicle_route[recovery_vehicle_index].index(orig_recovery_node)
        total_customer = [customer]
        total_customer.append(un_visit_customer)
        if launch_vehicle == recovery_vehicle:
            route_segment = vehicle_route[launch_vehicle_index][launch_node_index:recovery_node_index+1]  # 获得车辆任务路径
            # 计算无人机任务的插入位置
            for c in total_customer:
                plan, cost, time, uav_route = self.find_total_customer_plan(c, route_segment, drone_id, launch_vehicle)
                plan_cost[c] = cost
                plan_y[c] = plan
                plan_time[c] = time
                plan_uav_route[c] = uav_route
            # 按成本从小到大排序客户点
            sort_cost, sort_plan, sort_time, sort_uav_route = sort_customer_plans(plan_cost, plan_y, plan_time, plan_uav_route)
            for new_index, new_y in enumerate(sort_plan[un_visit_customer]):
                # new_vehicle_task_data = copy.deepcopy(remove_vehicle_task_data)
                new_vehicle_task_data = deep_copy_vehicle_task_data(remove_vehicle_task_data)
                drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_y
                new_cost = sort_cost[un_visit_customer][new_index]
                new_time = sort_time[un_visit_customer][new_index]
                new_uav_route = sort_uav_route[un_visit_customer][new_index]
                new_plan = {
                            'drone_id': drone_id,
                            'launch_vehicle': launch_vehicle,
                            'recovery_vehicle': recovery_vehicle,
                            'launch_node': new_launch_node,
                            'recovery_node': new_recovery_node,
                            'customer': new_customer,
                            'launch_time': vehicle_arrival_time[new_launch_vehicle][new_launch_node],
                            'recovery_time': vehicle_arrival_time[new_recovery_vehicle][new_recovery_node],
                            'energy': self.xeee[drone_id][self.node[new_launch_node].map_key][new_customer][self.node[new_recovery_node].map_key],
                            'cost': new_cost,
                            'time': new_time,
                            'uav_route': new_uav_route
                        }
                new_vehicle_task_data = update_vehicle_task(
                    new_vehicle_task_data, new_y, vehicle_route
                )
                # 随后遍历另一个客户点的任务
                for orig_index, y in enumerate(sort_plan[customer]):
                    drone_id, orig_launch_node, orig_customer, orig_recovery_node, launch_vehicle, recovery_vehicle = y
                    orig_cost = sort_cost[customer][orig_index]
                    orig_time = sort_time[customer][orig_index]
                    orig_uav_route = sort_uav_route[customer][orig_index]
                    orig_plan = {
                        'drone_id': drone_id,
                        'launch_vehicle': launch_vehicle,
                        'recovery_vehicle': recovery_vehicle,
                        'launch_node': orig_launch_node,
                        'recovery_node': orig_recovery_node,
                        'customer': orig_customer,
                        'launch_time': vehicle_arrival_time[launch_vehicle][orig_launch_node],
                        'recovery_time': vehicle_arrival_time[recovery_vehicle][orig_recovery_node],
                        'energy': self.xeee[drone_id][self.node[orig_launch_node].map_key][orig_customer][self.node[orig_recovery_node].map_key],
                        'cost': orig_cost,
                        'time': orig_time,
                        'uav_route': orig_uav_route
                    }
                    is_valid_plan = check_same_vehicle_conflict(
                                v_id=launch_vehicle,
                                drone_id=drone_id,
                                i_vtp=orig_launch_node,
                                j_vtp=orig_recovery_node,
                                solution_route=vehicle_route,
                                solution=vehicle_arrival_time,
                                vehicle_task_data=new_vehicle_task_data,    
                                vehicle = self.vehicle
                            )
                    if not is_valid_plan:
                        continue
                    else:
                        if orig_cost + new_cost < best_cost:
                            best_orig_y = y
                            best_new_y = new_y
                            best_orig_cost = orig_cost
                            best_new_cost = new_cost
                            best_cost = orig_cost + new_cost
                            best_orig_y_cijkdu_plan = orig_plan
                            best_new_y_cijkdu_plan = new_plan
        else:  # 跨车辆发射和回收情况
            launch_vehicle_segment = []
            recovery_vehicle_segment = []
            launch_vehicle_node_time = vehicle_arrival_time[launch_vehicle][orig_launch_node]
            for index, node in enumerate(vehicle_route[launch_vehicle_index][launch_node_index:],start = launch_node_index):
                if drone_id not in vehicle_task_data[launch_vehicle][node].recovery_drone_list:
                    if node != self.depot_id:
                        launch_vehicle_segment.append(node)
            # 回收车辆从回收节点从后向前遍历
            for i in range(len(vehicle_route[recovery_vehicle_index][:recovery_node_index]),-1,-1):
                # 回收时间大于发射时间
                if vehicle_arrival_time[recovery_vehicle][vehicle_route[recovery_vehicle_index][i]] > launch_vehicle_node_time:
                    if drone_id not in vehicle_task_data[recovery_vehicle][vehicle_route[recovery_vehicle_index][i]].launch_drone_list:
                        if vehicle_route[recovery_vehicle_index][i] != self.depot_id:
                            recovery_vehicle_segment.append(vehicle_route[recovery_vehicle_index][i])
            recovery_vehicle_segment.reverse()
            # 回收车辆从回收点向后遍历
            sub_route = vehicle_route[recovery_vehicle_index][recovery_node_index+1:]
            for index, node in enumerate(sub_route):
                if vehicle_arrival_time[recovery_vehicle][node] > launch_vehicle_node_time:
                    if drone_id not in vehicle_task_data[recovery_vehicle][node].launch_drone_list:  # 记录直到下次任务发射之前
                        if node != self.depot_id:
                                recovery_vehicle_segment.append(node)
            # 根据两条车辆路径，完成客户点的任务插入
            for c in total_customer:
                plan, cost, time, uav_route = self.find_cross_total_customer_plan(c, launch_vehicle_segment, recovery_vehicle_segment, drone_id, launch_vehicle, recovery_vehicle, vehicle_arrival_time)
                plan_cost[c] = cost
                plan_y[c] = plan
                plan_time[c] = time
                plan_uav_route[c] = uav_route
            # 按成本从小到大排序客户点
            sort_cost, sort_plan, sort_time, sort_uav_route = sort_customer_plans(plan_cost, plan_y, plan_time, plan_uav_route)
            for new_index, new_y in enumerate(sort_plan[un_visit_customer]):
                # new_vehicle_task_data = self._create_initial_vehicle_task_data()
                # new_vehicle_task_data = copy.deepcopy(remove_vehicle_task_data)
                new_vehicle_task_data = deep_copy_vehicle_task_data(remove_vehicle_task_data)
                drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_y
                new_cost = sort_cost[un_visit_customer][new_index]
                new_time = sort_time[un_visit_customer][new_index]
                new_uav_route = sort_uav_route[un_visit_customer][new_index]
                new_plan = {
                            'drone_id': drone_id,
                            'launch_vehicle': launch_vehicle,
                            'recovery_vehicle': recovery_vehicle,
                            'launch_node': new_launch_node,
                            'recovery_node': new_recovery_node,
                            'customer': new_customer,
                            'launch_time': vehicle_arrival_time[new_launch_vehicle][new_launch_node],
                            'recovery_time': vehicle_arrival_time[new_recovery_vehicle][new_recovery_node],
                            'energy': self.xeee[drone_id][self.node[new_launch_node].map_key][new_customer][self.node[new_recovery_node].map_key],
                            'cost': new_cost,
                            'time': new_time,
                            'uav_route': new_uav_route
                        }
                # 判断是否存在冲突情况
                if new_launch_vehicle == new_recovery_vehicle:
                    is_valid_plan = check_same_vehicle_conflict(
                                v_id=new_launch_vehicle,
                                drone_id=drone_id,
                                i_vtp=new_launch_node,
                                j_vtp=new_recovery_node,
                                solution_route=vehicle_route,
                                solution=vehicle_arrival_time,
                                vehicle_task_data=new_vehicle_task_data,    
                                vehicle = self.vehicle
                            )
                else:
                    is_valid_plan = check_cross_vehicle_conflict_fixed(
                        new_launch_vehicle,
                        new_recovery_vehicle,
                        drone_id,
                        new_launch_node,
                        new_recovery_node,
                        vehicle_route,
                        vehicle_arrival_time,
                        new_vehicle_task_data,
                        vehicle_arrival_time[new_launch_vehicle][new_launch_node],
                        vehicle_arrival_time[new_recovery_vehicle][new_recovery_node],
                        self.vehicle
                    )
                if not is_valid_plan:
                    continue
                else:
                    new_vehicle_task_data = update_vehicle_task(
                        new_vehicle_task_data, new_y, vehicle_route
                    )
                # 随后遍历另一个客户点的任务
                for orig_index, y in enumerate(sort_plan[customer]):
                    drone_id, orig_launch_node, orig_customer, orig_recovery_node, launch_vehicle, recovery_vehicle = y
                    orig_cost = sort_cost[customer][orig_index]
                    orig_time = sort_time[customer][orig_index]
                    orig_uav_route = sort_uav_route[customer][orig_index]
                    orig_plan = {
                        'drone_id': drone_id,
                        'launch_vehicle': launch_vehicle,
                        'recovery_vehicle': recovery_vehicle,
                        'launch_node': orig_launch_node,
                        'recovery_node': orig_recovery_node,
                        'customer': orig_customer,
                        'launch_time': vehicle_arrival_time[launch_vehicle][orig_launch_node],
                        'recovery_time': vehicle_arrival_time[recovery_vehicle][orig_recovery_node],
                        'energy': self.xeee[drone_id][self.node[orig_launch_node].map_key][orig_customer][self.node[orig_recovery_node].map_key],
                        'cost': orig_cost,
                        'time': orig_time,
                        'uav_route': orig_uav_route
                    }
                    # 判断是否存在冲突情况
                    if launch_vehicle == recovery_vehicle:
                        is_valid_plan = check_same_vehicle_conflict(
                                    v_id=launch_vehicle,
                                    drone_id=drone_id,
                                    i_vtp=orig_launch_node,
                                    j_vtp=orig_recovery_node,
                                    solution_route=vehicle_route,
                                    solution=vehicle_arrival_time,
                                    vehicle_task_data=new_vehicle_task_data,    
                                    vehicle = self.vehicle
                                )
                    else:
                        is_valid_plan = check_cross_vehicle_conflict_fixed(
                            launch_vehicle,
                            recovery_vehicle,
                            drone_id,
                            orig_launch_node,
                            orig_recovery_node,
                            vehicle_route,
                            vehicle_arrival_time,
                            new_vehicle_task_data,
                            vehicle_arrival_time[launch_vehicle][orig_launch_node],
                            vehicle_arrival_time[recovery_vehicle][orig_recovery_node],
                            self.vehicle
                        )
                    if not is_valid_plan:
                        continue
                    else:
                        if orig_cost + new_cost < best_cost:
                            best_orig_y = y
                            best_new_y = new_y
                            best_orig_cost = orig_cost
                            best_new_cost = new_cost
                            best_cost = orig_cost + new_cost
                            best_orig_y_cijkdu_plan = orig_plan
                            best_new_y_cijkdu_plan = new_plan
    return best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan



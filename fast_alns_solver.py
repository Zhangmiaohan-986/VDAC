#!/usr/bin/env python
"""
mFSTSP问题的高效ALNS求解框架 - 使用浅拷贝和增量更新
"""

import copy
import numpy as np
import numpy.random as rnd
from collections import defaultdict
import time
from parseCSV import *

from collections import defaultdict
import copy
from initialize import init_agent, initialize_drone_vehicle_assignments
from create_vehicle_route import *

import os
from main import find_keys_and_indices
from mfstsp_heuristic_1_partition import *
from mfstsp_heuristic_2_asgn_uavs import *
from mfstsp_heuristic_3_timing import *

from local_search import *
from rm_node_sort_node import rm_empty_node

import main
import endurance_calculator
import distance_functions

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from alns import ALNS
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.select import RouletteWheel, AlphaUCB
from alns.stop import MaxRuntime, MaxIterations
from destroy_repair_operator import *
from initialize import deep_copy_vehicle_task_data
from cost_y import calculate_plan_cost
class FastMfstspState:
    """
    高效的mFSTSP解状态类 - 使用浅拷贝和增量更新
    """
    
    def __init__(self, vehicle_routes, uav_assignments, customer_plan,
                 vehicle_task_data, global_reservation_table, total_cost=None, init_uav_plan=None, uav_cost=None,
                 init_vehicle_plan_time=None, 
                 node=None, DEPOT_nodeID=None, V=None, T=None, vehicle=None, uav_travel=None, veh_distance=None, 
                 veh_travel=None, N=None, N_zero=None, N_plus=None, A_total=None, A_cvtp=None, A_vtp=None, 
                 A_aerial_relay_node=None, G_air=None, G_ground=None, air_matrix=None, ground_matrix=None, 
                 air_node_types=None, ground_node_types=None, A_c=None, xeee=None):

        self.vehicle_routes = vehicle_routes
        self.uav_assignments = uav_assignments
        self.customer_plan = customer_plan
        self.vehicle_task_data = vehicle_task_data
        self.global_reservation_table = global_reservation_table
        self._total_cost = total_cost
        self.uav_plan = init_uav_plan
        self.uav_cost = uav_cost
        self.vehicle_plan_time = init_vehicle_plan_time
        self.node = node
        self.DEPOT_nodeID = DEPOT_nodeID
        self.vehicle = vehicle
        self.uav_travel = uav_travel
        self.veh_distance = veh_distance
        self.veh_travel = veh_travel
        self.N = N
        self.V = V
        self.T = T
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
        self.rm_empty_vehicle_task_data = deep_copy_vehicle_task_data(self.vehicle_task_data)
        # self.update_rm_empty_task()  # 更新空跑节点及其任务状态，后续需要删除空跑节点对应的key
        # self.empty_node_cost = self.update_calculate_plan_cost(self.uav_cost, self.rm_empty_vehicle_route) # 更新初始任务完成后的空跑节点
        # self.rm_empty_vehicle_arrive_time = self.calculate_rm_empty_vehicle_arrive_time()
        # 记录修改历史，用于快速回滚
        self._modification_history = []
        # self.base_vehicle_task_data = deep_copy_vehicle_task_data(self.vehicle_task_data)
        # self.re_update_time(self.rm_empty_vehicle_route, self.rm_empty_vehicle_arrive_time, self.base_vehicle_task_data)
        # self.check_all_vehicle_finish_task_time(self.re_vehicle_plan_time)
    
    def calculate_rm_empty_vehicle_arrive_time(self):  # 实际是计算去除空跑节点后每辆车到达各节点的时间
        """
        计算去除空跑节点后每辆车到达各节点的时间
        返回: dict，key为vehicle_id，value为{node_id: 到达时间}
        """
        rm_empty_vehicle_arrive_time = {}
        # for vehicle_id, route in enumerate(self.rm_empty_vehicle_route):
        for vehicle_id, route in enumerate(self.rm_empty_vehicle_route):
            vehicle_id = vehicle_id + 1
            arrive_time_dict = {}
            for idx, node_j in enumerate(route):
                if idx == 0:
                    arrive_time_dict[node_j] = 0
                else:
                    node_i = route[idx - 1]
                    # 这里假设 self.veh_travel[vehicle_id][node_i][node_j] 是车辆行驶时间
                    arrive_time_dict[node_j] = arrive_time_dict[node_i] + self.veh_travel[vehicle_id][node_i][node_j]
            rm_empty_vehicle_arrive_time[vehicle_id] = arrive_time_dict
        return rm_empty_vehicle_arrive_time
    
    # 设计一个函数，其主要功能为基于处理掉空跑节点后，根据无人机的任务分配，重新规划整体时间
    def re_update_time(self, vehicle_route, vehicle_arrive_time, vehicle_task_data):
        """
        基于处理掉空跑节点后，根据无人机的任务分配，重新规划整体时间
        """
        new_vehicle_task_data = vehicle_task_data.copy()  # 待处理
        self.re_time_uav_task_dict, self.re_time_customer_plan, self.re_time_uav_plan, self.re_vehicle_plan_time, self.re_vehicle_task_data = low_update_time(self.uav_assignments, 
        self.uav_plan, vehicle_route, new_vehicle_task_data, vehicle_arrive_time, 
        self.node, self.V, self.T, self.vehicle, self.uav_travel)
        # 输出更修车辆后的详细方案及时间分配等情况
        final_uav_plan, final_uav_cost, final_vehicle_plan_time, final_vehicle_task_data, final_global_reservation_table = rolling_time_cbs(vehicle_arrive_time, 
        vehicle_route, self.re_time_uav_task_dict, self.re_time_customer_plan, self.re_time_uav_plan, 
        self.re_vehicle_plan_time, self.re_vehicle_task_data, self.node, self.DEPOT_nodeID, self.V, self.T, self.vehicle, 
        self.uav_travel, self.veh_distance, self.veh_travel, self.N, self.N_zero, self.N_plus, self.A_total, self.A_cvtp, 
        self.A_vtp, self.A_aerial_relay_node, self.G_air, self.G_ground, self.air_matrix, self.ground_matrix, 
        self.air_node_types, self.ground_node_types, self.A_c, self.xeee)
        self.final_total_cost = calculate_plan_cost(final_uav_cost, vehicle_route, self.vehicle, self.T, self.V, self.veh_distance)
        return final_uav_plan, final_uav_cost, final_vehicle_plan_time, final_vehicle_task_data, final_global_reservation_table
    
    # 设计功能函数，主要判断所有车辆全部完成任务后的总任务时间
    def check_all_vehicle_finish_task_time(self, vehicle_plan_time):
        """
        判断所有车辆全部完成任务后的总任务时间
        vehicle_plan_time格式: {vehicle_id: {node_id: [start_time, end_time]}}
        """
        # 计算所有车辆完成任务后的总任务时间
        total_task_time = 0
        for vehicle_id, node_times in vehicle_plan_time.items():
            if node_times:
                # 找到该车辆所有任务中的最大结束时间
                latest_finish_time_for_vehicle = max(times[1] for times in node_times.values())
                total_task_time += latest_finish_time_for_vehicle
        # print("所有车辆全部完成任务后的总任务时间：", total_task_time)
        return total_task_time

    def objective(self):
        """目标函数：计算总成本"""
        # if self._total_cost is not None:
        #     return self._total_cost
        
        # 简化成本计算
        # vehicle_cost = len(self.vehicle_routes) * 10
        # uav_cost = sum(len(assignments) for assignments in self.uav_assignments.values()) * 5
        # self._total_cost = vehicle_cost + uav_cost
        self._total_cost = calculate_plan_cost(self.uav_cost, self.vehicle_routes, self.vehicle, self.T, self.V, self.veh_distance)
        return self._total_cost

    # 根据状态更新空跑节点
    def update_rm_empty_task(self):
        rm_empty_vehicle_route, empty_nodes_by_vehicle = rm_empty_node(self.customer_plan, self.vehicle_routes)
        self.rm_empty_vehicle_route = rm_empty_vehicle_route
        self.empty_nodes_by_vehicle = empty_nodes_by_vehicle
        # self.rm_empty_node_cost = calculate_plan_cost(self.uav_cost, self.rm_empty_vehicle_route, self.vehicle, self.T, self.V, self.veh_distance)
        # for i, route in enumerate(self.rm_empty_vehicle_route):
        #     vehicle_id = i + 1
        #     self.rm_empty_vehicle_route[vehicle_id] = route
        return self.rm_empty_vehicle_route, self.empty_nodes_by_vehicle

    def update_calculate_plan_cost(self, uav_cost, empty_vehicle_route):
        empty_node_cost = calculate_plan_cost(uav_cost, empty_vehicle_route, self.vehicle, self.T, self.V, self.veh_distance)
        return empty_node_cost

    def sorted_import_node(self, destroyed_plan, current_plan):
        """
        根据破坏计划和当前计划，将每个车辆的路线节点分为三个等级
        
        Args:
            destroyed_plan: 被破坏的客户计划 {customer_id: (drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle)}
            current_plan: 当前客户计划 {customer_id: (drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle)}
            
        Returns:
            dict: {vehicle_id: {'level1': [nodes], 'level2': [nodes], 'level3': [nodes]}}
                 level1: 当前使用的发射/回收节点
                 level2: 被破坏的发射/回收节点  
                 level3: 从未使用的节点
        """
        # 初始化结果字典
        vehicle_node_levels = {}
        
        # 遍历每个车辆的路线
        for vehicle_id, route in self.vehicle_routes.items():
            if vehicle_id not in vehicle_node_levels:
                vehicle_node_levels[vehicle_id] = {
                    'level1': [],  # 当前使用的节点
                    'level2': [],  # 被破坏的节点
                    'level3': []   # 从未使用的节点
                }
            
            # 收集当前计划中该车辆的所有发射和回收节点
            current_launch_nodes = set()
            current_recovery_nodes = set()
            for customer, assignment in current_plan.items():
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                if launch_vehicle == vehicle_id:
                    current_launch_nodes.add(launch_node)
                if recovery_vehicle == vehicle_id:
                    current_recovery_nodes.add(recovery_node)
            
            # 收集被破坏计划中该车辆的所有发射和回收节点
            destroyed_launch_nodes = set()
            destroyed_recovery_nodes = set()
            for customer, assignment in destroyed_plan.items():
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                if launch_vehicle == vehicle_id:
                    destroyed_launch_nodes.add(launch_node)
                if recovery_vehicle == vehicle_id:
                    destroyed_recovery_nodes.add(recovery_node)
            
            # 遍历车辆路线中的每个节点，进行分类
            for node in route:
                if node in current_launch_nodes or node in current_recovery_nodes:
                    # 等级1：当前使用的发射/回收节点
                    vehicle_node_levels[vehicle_id]['level1'].append(node)
                elif node in destroyed_launch_nodes or node in destroyed_recovery_nodes:
                    # 等级2：被破坏的发射/回收节点
                    vehicle_node_levels[vehicle_id]['level2'].append(node)
                else:
                    # 等级3：从未使用的节点
                    vehicle_node_levels[vehicle_id]['level3'].append(node)
        
        return vehicle_node_levels

    def fast_copy(self):
        """快速浅拷贝 - 只复制引用，不复制数据"""
        # vehicle_task_data 用 fast_copy
        vehicle_task_data_copy = self.vehicle_task_data.__class__()  # 保持原类型
        for k, v in self.vehicle_task_data.items():
            vehicle_task_data_copy[k] = v.fast_copy() if hasattr(v, 'fast_copy') else copy.deepcopy(v)
        # global_reservation_table 建议也做深拷贝或类似处理
        # global_reservation_table_copy = copy.deepcopy(self.global_reservation_table)
        global_reservation_table_copy = copy.copy(self.global_reservation_table)
        
        # 创建新的状态对象
        new_state = FastMfstspState(
            vehicle_routes=self.vehicle_routes.copy(),
            uav_assignments={k: v.copy() for k, v in self.uav_assignments.items()},
            customer_plan=self.customer_plan.copy(),
            vehicle_task_data=vehicle_task_data_copy,
            global_reservation_table=global_reservation_table_copy,
            total_cost=self._total_cost,
            init_uav_plan=self.uav_plan,
            uav_cost=self.uav_cost,
            init_vehicle_plan_time=self.vehicle_plan_time,
            node=self.node,
            DEPOT_nodeID=self.DEPOT_nodeID,
            V=self.V,
            T=self.T,
            vehicle=self.vehicle,
            uav_travel=self.uav_travel,
            veh_distance=self.veh_distance,
            veh_travel=self.veh_travel,
            N=self.N,
            N_zero=self.N_zero,
            N_plus=self.N_plus,
            A_total=self.A_total,
            A_cvtp=self.A_cvtp,
            A_vtp=self.A_vtp,
            A_aerial_relay_node=self.A_aerial_relay_node,
            G_air=self.G_air,
            G_ground=self.G_ground,
            air_matrix=self.air_matrix,
            ground_matrix=self.ground_matrix,
            air_node_types=self.air_node_types,
            ground_node_types=self.ground_node_types,
            A_c=self.A_c,
            xeee=self.xeee
        )
        
        # 复制可能存在的额外属性
        if hasattr(self, 'destroyed_node_cost'):
            new_state.destroyed_node_cost = self.destroyed_node_cost
        else:
            new_state.destroyed_node_cost = None
            
        if hasattr(self, 'final_uav_plan'):
            new_state.final_uav_plan = self.final_uav_plan
        else:
            new_state.final_uav_plan = None
            
        if hasattr(self, 'final_uav_cost'):
            new_state.final_uav_cost = self.final_uav_cost
        else:
            new_state.final_uav_cost = None
            
        if hasattr(self, 'final_vehicle_plan_time'):
            new_state.final_vehicle_plan_time = self.final_vehicle_plan_time
        else:
            new_state.final_vehicle_plan_time = None
            
        if hasattr(self, 'final_vehicle_task_data'):
            new_state.final_vehicle_task_data = self.final_vehicle_task_data
        else:
            new_state.final_vehicle_task_data = None
            
        if hasattr(self, 'final_global_reservation_table'):
            new_state.final_global_reservation_table = self.final_global_reservation_table
        else:
            new_state.final_global_reservation_table = None
            
        if hasattr(self, 'destroyed_customers_info'):
            new_state.destroyed_customers_info = self.destroyed_customers_info.copy() if self.destroyed_customers_info else {}
        else:
            new_state.destroyed_customers_info = {}
            
        if hasattr(self, 'rm_empty_vehicle_route'):
            new_state.rm_empty_vehicle_route = self.rm_empty_vehicle_route.copy() if self.rm_empty_vehicle_route else []
        else:
            new_state.rm_empty_vehicle_route = []
            
        if hasattr(self, 'empty_nodes_by_vehicle'):
            new_state.empty_nodes_by_vehicle = self.empty_nodes_by_vehicle.copy() if self.empty_nodes_by_vehicle else {}
        else:
            new_state.empty_nodes_by_vehicle = {}
            
        if hasattr(self, 'rm_empty_vehicle_arrive_time'):
            new_state.rm_empty_vehicle_arrive_time = self.rm_empty_vehicle_arrive_time.copy() if self.rm_empty_vehicle_arrive_time else {}
        else:
            new_state.rm_empty_vehicle_arrive_time = {}
            
        if hasattr(self, 'rm_empty_node_cost'):
            new_state.rm_empty_node_cost = self.rm_empty_node_cost
        else:
            new_state.rm_empty_node_cost = None
            
        if hasattr(self, 'final_total_cost'):
            new_state.final_total_cost = self.final_total_cost
        else:
            new_state.final_total_cost = None
        
        return new_state

    def record_modification(self, operation, data):
        """记录修改操作，用于回滚"""
        self._modification_history.append((operation, data))
    
    def rollback_last_modification(self):
        """回滚最后一次修改"""
        if self._modification_history:
            operation, data = self._modification_history.pop()
            # 根据操作类型进行回滚
            if operation == "remove_customer":
                customer, assignment = data
                self.customer_plan[customer] = assignment
                uav_id, _, _, _, _ = assignment
                if uav_id in self.uav_assignments:
                    self.uav_assignments[uav_id].append(assignment)
            elif operation == "modify_route":
                vehicle_id, old_route = data
                self.vehicle_routes[vehicle_id] = old_route
            # 重置成本缓存
            self._total_cost = None

class IncrementalALNS:
    """增量式ALNS求解器 - 使用修改记录和回滚机制"""
    
    def __init__(self, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, 
        N_zero, N_plus, A_total, A_cvtp, A_vtp, 
		A_aerial_relay_node, G_air, G_ground,air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee,
        max_iterations=1000, max_runtime=60):
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
        self.max_iterations = max_iterations
        self.max_runtime = max_runtime
        self.rng = rnd.default_rng(42)

    def repair_greedy_insertion(self, state):
        """
        贪婪插入修复算子：将被移除的客户点按成本最小原则重新插入，记录所有插入方案。
        返回修复后的状态和所有破坏节点的最优插入方案列表。
        """
        repaired_state = state.fast_copy()
        # destroy_node = list(set(self.A_c) - set(state.customer_plan.keys()))
        destroy_node = list(state.destroyed_customers_info.keys())  # 总结出了所有的待插入的破坏节点
        insert_plan = []  # 记录所有破坏节点的最优插入方案

        print(f"贪婪修复：需要插入 {len(destroy_node)} 个客户点: {destroy_node}")

        while len(destroy_node) > 0:
            best_global_cost = float('inf')
            best_global_scheme = None
            best_global_customer = None
            
            # 遍历所有待插入客户点，找出全局最优插入方案
            for customer in destroy_node:
                min_cost = float('inf')
                best_scheme = None
                
                # 获取当前状态的数据
                vehicle_route = repaired_state.rm_empty_vehicle_route
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time()
                
                # 获取所有可行的插入位置
                all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
                
                # 遍历所有可行插入位置，找到成本最小的方案
                for drone_id, inert_positions in all_insert_position.items():
                    for inert_position in inert_positions:
                        launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                        insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        if insert_cost < min_cost:
                            min_cost = insert_cost
                            best_scheme = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                
                if best_scheme is None:
                    continue  # 没有可行插入
                
                # 进行局部优化
                k_opt_step = 5
                near_node_list = self.get_near_node_list(best_scheme, k_opt_step, vehicle_route)
                best_scheme_opt, best_cost_opt = self.multiopt_update_best_scheme(best_scheme, near_node_list, vehicle_route, vehicle_task_data)
                
                if best_cost_opt < best_global_cost:
                    best_global_cost = best_cost_opt
                    best_global_scheme = best_scheme_opt
                    best_global_customer = customer

            # 应用全局最优插入方案
            if best_global_scheme is not None:
                # 实际应用插入方案
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_global_scheme
                
                # 更新customer_plan
                repaired_state.customer_plan[customer_node] = best_global_scheme
                
                # 更新uav_assignments
                if drone_id not in repaired_state.uav_assignments:
                    repaired_state.uav_assignments[drone_id] = []
                repaired_state.uav_assignments[drone_id].append(best_global_scheme)
                
                # 更新uav_cost
                if repaired_state.uav_cost is None:
                    repaired_state.uav_cost = {}
                repaired_state.uav_cost[customer_node] = best_global_cost
                
                # 更新vehicle_task_data
                repaired_state.vehicle_task_data = self.update_vehicle_task(
                    repaired_state.vehicle_task_data, best_global_scheme, vehicle_route
                )
                
                # 记录插入方案
                insert_plan.append((best_global_customer, best_global_scheme, best_global_cost))
                destroy_node.remove(best_global_customer)
                
                # 更新空跑节点等状态
                repaired_state.update_rm_empty_task()
                repaired_state._total_cost = None  # 重置成本缓存
                
                print(f"成功插入客户点 {best_global_customer}，成本: {best_global_cost}")
            else:
                print("未找到可行的插入方案，剩余节点：", destroy_node)
                break
        
        print(f"贪婪修复完成：成功插入 {len(insert_plan)} 个客户点")
        return repaired_state, insert_plan

    def get_near_node_list(self, best_scheme, k, vehicle_route):
        """
        根据best_scheme的车辆id，找到该车辆的路线vehicle_route[v_id-1]，
        然后找到距离客户点c最近的聚类的k个地面节点，且这些节点不能出现在该车辆路线中。
        如果发射车辆和回收车辆不同，则返回dict，key为车辆id，value为各自可插入节点list；否则返回单一车辆的list。
        """
        # best_scheme: (drone_id, launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
        _, _, customer, _, launch_vehicle_id, recovery_vehicle_id = best_scheme
        customer_vtp_dict = self.cluster_vtp_for_customers(k=20)  # 取较大k，后面筛选
        near_vtp_candidates = customer_vtp_dict.get(customer, [])

        # 发射车辆
        route_launch = vehicle_route[launch_vehicle_id - 1]
        route_launch_set = set(route_launch)
        filtered_launch = [vtp for vtp in near_vtp_candidates if vtp not in self.node[route_launch_set].map_key]  # 映射对应的空中节点
        launch_list = filtered_launch[:k]

        if launch_vehicle_id == recovery_vehicle_id:
            return launch_list
        else:
            # 回收车辆
            route_recovery = vehicle_route[recovery_vehicle_id - 1]
            route_recovery_set = set(route_recovery)
            filtered_recovery = [vtp for vtp in near_vtp_candidates if vtp not in route_recovery_set]
            recovery_list = filtered_recovery[:k]
            return {launch_vehicle_id: launch_list, recovery_vehicle_id: recovery_list}

    def drone_insert_cost(self, drone_id, customer, launch_node, recovery_node):
        insert_cost = 0
        launch_node_map_index = self.node[launch_node].map_key
        recovery_node_map_index = self.node[recovery_node].map_key
        customer_map_index = self.node[customer].map_key
        insert_cost = self.uav_travel[drone_id][launch_node_map_index][customer].totalDistance+ self.uav_travel[drone_id][customer][recovery_node_map_index].totalDistance
        per_cost = self.vehicle[drone_id].per_cost
        insert_cost = insert_cost * per_cost
        return insert_cost

    def repair_regret_insertion(self, state):
        """
        后悔插入修复算子：考虑插入位置的后悔值
        """
        # 这里简化实现，实际应该实现后悔插入策略
        repaired_state = state.fast_copy()
        # 获得被破坏的客户节点
        destroy_node = list(set(self.A_c)-set(state.customer_plan.keys()))
        insert_customer_cost = np.full(len(destroy_node), float('inf'))
        # 直到所有的客户点被插入，任务结束
        while len(destroy_node) > 0:
            # 遍历所有客户点，计算插入成本
            for i, customer in enumerate(destroy_node):
                # 得到可行所有的可行插入位置，即算法的解空间
                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time()
                all_inert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
                # 遍历所有可行插入位置，计算插入成本
                for drone_id, inert_positions in all_inert_position.items():
                    for inert_position in inert_positions:
                        launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                        # 计算插入成本，直接是更换无人机的服务客户带来的插入成本
                        insert_cost = self.drone_insert_cost(drone_id, vehicle_route, vehicle_task_data, customer, launch_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
        


        # 重置成本缓存
        repaired_state._total_cost = None
        return repaired_state

    def repair_k_insertion(self, state):
        """
        K插入修复算子：考虑多个插入位置的组合
        """
        # 这里简化实现，实际应该实现K插入策略
        repaired_state = state.fast_copy()
        # 重置成本缓存
        repaired_state._total_cost = None
        return repaired_state

    def get_all_insert_position(self, vehicle_route, vehicle_task_data, customer, vehicle_arrive_time):
        all_insert_position = {drone_id: [] for drone_id in self.V}

        for drone_id in self.V:
            for launch_vehicle_idx, route in enumerate(vehicle_route):
                launch_vehicle_id = launch_vehicle_idx + 1
                n = len(route)
                i = 1
                while i < n - 1:
                    launch_node = route[i]
                    # 只在drone_list中才可发射
                    if drone_id not in vehicle_task_data[launch_vehicle_id][launch_node].drone_list:
                        i += 1
                        continue
                    # 找连续片段
                    j = i + 1
                    while j < n - 1:
                        node = route[j]
                        in_drone_list = drone_id in vehicle_task_data[launch_vehicle_id][node].drone_list
                        in_launch_list = drone_id in vehicle_task_data[launch_vehicle_id][node].launch_drone_list
                        if not in_drone_list:
                            if in_launch_list:
                                # 片段终点包含该节点
                                j += 1
                            break
                        j += 1
                    # 现在[i, j)是连续片段，j可能因为break提前终止
                    # 片段终点为j-1，如果j-1节点是发射点（即不在drone_list但在launch_drone_list），包含它
                    end = j
                    if j < n - 1:
                        node = route[j]
                        if drone_id not in vehicle_task_data[launch_vehicle_id][node].drone_list and \
                        drone_id in vehicle_task_data[launch_vehicle_id][node].launch_drone_list:
                            end = j + 1  # 包含发射点
                    for k in range(i + 1, end):
                        recovery_node = route[k]
                        all_insert_position[drone_id].append(
                            (launch_node, customer, recovery_node, launch_vehicle_id, launch_vehicle_id)
                        )
                    i = j
                # 跨车查找（同前）
                for i in range(1, n - 1):
                    launch_node = route[i]
                    if drone_id not in vehicle_task_data[launch_vehicle_id][launch_node].drone_list:
                        continue
                    launch_time = vehicle_arrive_time[launch_vehicle_id][launch_node]
                    for recovery_vehicle_idx, other_route in enumerate(vehicle_route):
                        recovery_vehicle_id = recovery_vehicle_idx + 1
                        if recovery_vehicle_id == launch_vehicle_id:
                            continue
                        for recovery_node in other_route[1:-1]:
                            if drone_id not in vehicle_task_data[recovery_vehicle_id][recovery_node].drone_list:
                                continue
                            # 新增：排除发射点和回收点完全相同的情况
                            if launch_vehicle_id == recovery_vehicle_id and launch_node == recovery_node:
                                continue  # 跨车时也不允许同节点
                            if launch_node == recovery_node:
                                continue  # 跨车时也不允许同节点
                            recovery_time = vehicle_arrive_time[recovery_vehicle_id][recovery_node]
                            if recovery_time <= launch_time:
                                continue
                            idx = other_route.index(recovery_node)
                            conflict = False
                            for m in range(1, idx):
                                if drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].drone_list or \
                                drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].launch_drone_list or \
                                drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].recovery_drone_list:
                                    conflict = True
                                    break
                            for m in range(idx + 1, len(other_route) - 1):
                                if drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].launch_drone_list:
                                    conflict = True
                                    break
                            if not conflict:
                                all_insert_position[drone_id].append(
                                    (launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                                )
        return all_insert_position

    def multiopt_update_best_scheme(self, best_scheme, near_node_list, vehicle_route, vehicle_task_data, sample_size=30):
            """
            加速多opt邻域搜索：对near_node_list随机采样sample_size个发射-回收节点组合，
            只计算本无人机和同节点相关无人机的成本，贪婪选择最优。
            返回(最优方案, 最优总成本)。
            """
            import random
            drone_id, launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
            best = best_scheme
            best_cost = float('inf')

            # 辅助：获取同节点相关无人机
            def get_related_drones(vehicle_id, node, task_data):
                related = set()
                if hasattr(task_data[vehicle_id][node], 'drone_list'):
                    related.update(task_data[vehicle_id][node].drone_list)
                if hasattr(task_data[vehicle_id][node], 'launch_drone_list'):
                    related.update(task_data[vehicle_id][node].launch_drone_list)
                if hasattr(task_data[vehicle_id][node], 'recovery_drone_list'):
                    related.update(task_data[vehicle_id][node].recovery_drone_list)
                return related

            # 计算本无人机和同节点相关无人机的总成本
            def get_greedy_cost(vehicle_id, l_n, r_n):
                total = 0
                # 本无人机
                total += self.drone_insert_cost(drone_id, customer, l_n, r_n)
                # 相关无人机（发射/回收节点）
                related = get_related_drones(vehicle_id, l_n, vehicle_task_data) | get_related_drones(vehicle_id, r_n, vehicle_task_data)
                related.discard(drone_id)
                for d_id in related:
                    # 查找d_id的发射/回收节点
                    launch_n, recovery_n = None, None
                    route = vehicle_route[vehicle_id - 1]
                    for n2 in route:
                        if hasattr(vehicle_task_data[vehicle_id][n2], 'launch_drone_list') and d_id in vehicle_task_data[vehicle_id][n2].launch_drone_list:
                            launch_n = n2
                        if hasattr(vehicle_task_data[vehicle_id][n2], 'recovery_drone_list') and d_id in vehicle_task_data[vehicle_id][n2].recovery_drone_list:
                            recovery_n = n2
                    if launch_n and recovery_n:
                        total += self.drone_insert_cost(d_id, customer, launch_n, recovery_n)
                return total

            # 单车情况
            if launch_vehicle_id == recovery_vehicle_id:
                node_list = near_node_list
                # 采样sample_size个不同组合
                candidates = set()
                while len(candidates) < sample_size:
                    l_n = random.choice(node_list)
                    r_n = random.choice(node_list)
                    if l_n != r_n:
                        candidates.add((l_n, r_n))
                for l_n, r_n in candidates:
                    cost = get_greedy_cost(launch_vehicle_id, l_n, r_n)
                    if cost < best_cost:
                        best = (drone_id, l_n, customer, r_n, launch_vehicle_id, recovery_vehicle_id)
                        best_cost = cost
                return best, best_cost
            else:
                # 异车情况
                launch_list = near_node_list[launch_vehicle_id]
                recovery_list = near_node_list[recovery_vehicle_id]
                candidates = set()
                while len(candidates) < sample_size:
                    l_n = random.choice(launch_list)
                    r_n = random.choice(recovery_list)
                    if l_n != r_n:
                        candidates.add((l_n, r_n))
                for l_n, r_n in candidates:
                    cost = get_greedy_cost(launch_vehicle_id, l_n, r_n) + get_greedy_cost(recovery_vehicle_id, l_n, r_n)
                    if cost < best_cost:
                        best = (drone_id, l_n, customer, r_n, launch_vehicle_id, recovery_vehicle_id)
                        best_cost = cost
                return best, best_cost

    def solve(self, initial_state):
        """
        增量式ALNS主循环：轮盘赌选择算子，模拟退火接受准则，记录解状态
        """
        # 1. 算子池
        destroy_operators = [
            # self.destroy_random_removal,
            self.destroy_worst_removal,
            # self.destroy_important_removal,
            # self.destroy_shaw_removal
        ]
        repair_operators = [
            self.repair_greedy_insertion,
            # self.repair_regret_insertion,
            # self.repair_k_insertion,
        ]
        destroy_weights = [1.0] * len(destroy_operators)  # 使用列表而不是numpy数组
        repair_weights = [1.0] * len(repair_operators)

        # 2. 初始化
        current_state = initial_state.fast_copy()
        best_state = current_state.fast_copy()
        best_objective = best_state.objective()
        current_objective = best_objective
        start_time = time.time()
        iteration = 0
        cluster_vtp_dict = self.cluster_vtp_for_customers()
        # 3. 轮盘赌对象 - 使用简化的轮盘赌实现
        # 4. 模拟退火 - 使用简化的模拟退火实现
        temperature = 1000.0
        cooling_rate = 0.995

        print(f"开始ALNS求解，初始成本: {best_objective}")

        # 5. 主循环
        # while iteration < self.max_iterations and (time.time() - start_time) < self.max_runtime:
        while iteration < self.max_iterations:
            try:
                # 选择算子 - 使用简化的轮盘赌
                destroy_idx = self._roulette_wheel_select(destroy_weights)
                repair_idx = self._roulette_wheel_select(repair_weights)
                destroy_op = destroy_operators[destroy_idx]
                repair_op = repair_operators[repair_idx]

                # 记录当前解
                prev_state = current_state.fast_copy()
                prev_objective = current_objective
                # 更新空跑节点及其任务状态，后续需要删除空跑节点对应的key
                prev_state.rm_empty_vehicle_route, prev_state.empty_nodes_by_vehicle = prev_state.update_rm_empty_task()
                # 更新初始任务完成后的空跑节点
                prev_state.destroyed_node_cost = prev_state.update_calculate_plan_cost(prev_state.uav_cost, prev_state.rm_empty_vehicle_route)
                prev_state.rm_empty_vehicle_arrive_time = prev_state.calculate_rm_empty_vehicle_arrive_time()
                prev_state.final_uav_plan, prev_state.final_uav_cost, prev_state.final_vehicle_plan_time, prev_state.final_vehicle_task_data, prev_state.final_global_reservation_table = prev_state.re_update_time(prev_state.rm_empty_vehicle_route, prev_state.rm_empty_vehicle_arrive_time, prev_state.vehicle_task_data)
                # 上述prev得到了去除了所有空跑节点后得到的最终经过cbs处理后的结果，在每代的破坏和修复任务中不重新计算空跑节点，每经过一次迭代后在处理cbs获得最终的结果
                
                # 破坏阶段
                print(f"迭代 {iteration}: 使用破坏算子 {destroy_op.__name__}")
                destroyed_state = destroy_op(current_state)
                
                # 检查破坏后的状态是否有效
                if not destroyed_state.customer_plan:
                    print("破坏后没有客户点，跳过此次迭代")
                    iteration += 1
                    continue
                
                # 修复阶段
                print(f"迭代 {iteration}: 使用修复算子 {repair_op.__name__}")
                repaired_state, insert_plan = repair_op(destroyed_state)
                
                # 检查修复后的状态是否有效
                if not repaired_state.customer_plan:
                    print("修复后没有客户点，跳过此次迭代")
                    iteration += 1
                    continue

                # 计算新目标值
                new_objective = repaired_state.objective()
                print(f"迭代 {iteration}: 当前成本 {current_objective:.2f} -> 新成本 {new_objective:.2f}")

                # 模拟退火判断是否接受
                accept = self._simulated_annealing_accept(current_objective, new_objective, temperature)
                
                if accept:
                    current_state = repaired_state.fast_copy()
                    current_objective = new_objective
                    # 算子奖励
                    destroy_weights[destroy_idx] += 1
                    repair_weights[repair_idx] += 1
                    print(f"迭代 {iteration}: 接受新解")
                    
                    # 更新最优解
                    if new_objective < best_objective:
                        best_state = repaired_state.fast_copy()
                        best_objective = new_objective
                        print(f"迭代 {iteration}: 发现更优解，成本: {best_objective:.2f}")
                else:
                    # 不接受，回滚
                    current_state = prev_state
                    current_objective = prev_objective
                    # 算子惩罚
                    destroy_weights[destroy_idx] *= 0.95
                    repair_weights[repair_idx] *= 0.95
                    print(f"迭代 {iteration}: 拒绝新解，回滚")

                # 温度衰减
                temperature *= cooling_rate

                # 记录日志
                if iteration % 10 == 0:  # 更频繁的日志输出用于调试
                    elapsed_time = time.time() - start_time
                    print(f"迭代 {iteration}, 当前成本: {current_objective:.2f}, 最优成本: {best_objective:.2f}, 温度: {temperature:.2f}, 运行时间: {elapsed_time:.2f}秒")

            except Exception as e:
                print(f"迭代 {iteration} 发生错误: {e}")
                # 发生错误时回滚到之前的状态
                current_state = prev_state
                current_objective = prev_objective

            iteration += 1

        elapsed_time = time.time() - start_time
        statistics = {
            'iterations': iteration,
            'runtime': elapsed_time,
            'best_objective': best_objective
        }
        print(f"ALNS求解完成，最终成本: {best_objective}, 迭代次数: {iteration}, 运行时间: {elapsed_time:.2f}秒")
        return best_state, best_objective, statistics

    def _roulette_wheel_select(self, weights):
        """
        简化的轮盘赌选择
        """
        total_weight = sum(weights)
        if total_weight == 0:
            return self.rng.integers(0, len(weights))
        
        r = self.rng.random() * total_weight
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return i
        return len(weights) - 1

    def _simulated_annealing_accept(self, current_cost, new_cost, temperature):
        """
        简化的模拟退火接受准则
        """
        if new_cost < current_cost:
            return True
        else:
            delta = new_cost - current_cost
            probability = np.exp(-delta / temperature)
            return self.rng.random() < probability
    
    def destroy_random_removal(self, state):
        """随机销毁客户点"""
        # 拷贝当前解情况
        new_state = state.fast_copy()
        rm_empty_vehicle_route = state.rm_empty_vehicle_route
        new_state.vehicle_route = rm_empty_vehicle_route
        all_customers = self.A_c
        if not all_customers:
            print("没有客户点")
            return new_state
        num_to_remove = self.rng.integers(
            max(1, len(all_customers) // 5), 
            max(2, len(all_customers) // 3)
        )
        customers_to_remove = self.rng.choice(all_customers, num_to_remove, replace=False)
        # 4. 移除这些客户点及相关无人机任务
        for customer in customers_to_remove:
            if customer in new_state.customer_plan:
                assignment = new_state.customer_plan.pop(customer)
                uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                # 从无人机分配中移除相关任务
                if uav_id in new_state.uav_assignments:
                    new_state.uav_assignments[uav_id] = [
                        task for task in new_state.uav_assignments[uav_id]
                        if task[1] != customer_node
                    ]
                # 更新破坏的无人机空中成本
                new_state.uav_cost.pop(customer_node)
                # 更新对应的vehicle_task_data
                vehicle_task_data = new_state.vehicle_task_data
                vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                new_state.vehicle_task_data = vehicle_task_data

        # 5. 更新空跑节点等状态
        new_state.update_rm_empty_task()
        new_state.total_cost = new_state.objective()
        # 6. 返回新状态
        return new_state

    # 最坏节点破坏
    def destroy_worst_removal(self, state):
        """最差客户点移除：删除20%-30%成本最高的客户点任务"""
        # 拷贝当前解
        new_state = state.fast_copy()
        # new_state.rm_empty_vehicle_route, new_state.empty_nodes_by_vehicle = new_state.update_rm_empty_task()
        # new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.rm_empty_vehicle_route)
        # # 获取当前解中的客户点（而不是所有可能的客户点）
        current_customers = list(new_state.customer_plan.keys())
        if not current_customers:
            print("没有客户点需要移除")
            return new_state

        new_state.vehicle_routes = new_state.rm_empty_vehicle_route  # 更新路径
        # 1. 计算每个客户点的成本
        customer_costs = []
        for customer in current_customers:
            # 从uav_cost中获取该客户点的成本
            cost = new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0
            customer_costs.append((customer, cost))

        # 2. 按成本降序排序
        customer_costs.sort(key=lambda x: x[1], reverse=True)

        # 3. 选取20%-30%最贵的客户点
        n = len(customer_costs)
        num_to_remove = self.rng.integers(
            max(1, int(n * 0.2)),
            max(2, int(n * 0.3)) + 1
        )
        customers_to_remove = [customer for customer, _ in customer_costs[:num_to_remove]]

        print(f"最差破坏：移除 {len(customers_to_remove)} 个客户点: {customers_to_remove}")
        destroyed_customers_info = {}
        # 4. 移除这些客户点及相关无人机任务
        for customer in customers_to_remove:
            if customer in new_state.customer_plan:
                assignment = new_state.customer_plan.pop(customer)
                uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                
                # 记录被破坏客户节点的详细信息
                customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
                # customer_info = tuple(customer_info)
                destroyed_customers_info[customer] = customer_info
                
                # 从无人机分配中移除相关任务
                if uav_id in new_state.uav_assignments:
                    new_state.uav_assignments[uav_id] = [
                        task for task in new_state.uav_assignments[uav_id]
                        if task[2] != customer_node  # 修正索引：task[2]是customer_node
                    ]
                
                # 更新破坏的无人机空中成本
                if new_state.uav_cost and customer_node in new_state.uav_cost:
                    new_state.uav_cost.pop(customer_node, None)
                
                # 更新对应的vehicle_task_data
                vehicle_task_data = new_state.vehicle_task_data
                vehicle_task_data = self.remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                new_state.vehicle_task_data = vehicle_task_data
        # 5. 更新空跑节点等状态
        # new_state.update_rm_empty_task()
        new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.rm_empty_vehicle_route)
        # new_state._total_cost = None  # 重置成本缓存
        
        # 将破坏的客户节点信息存储到状态中，供修复阶段使用
        new_state.destroyed_customers_info = destroyed_customers_info
        
        print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
        print("=== 破坏阶段完成 ===\n")
        return new_state

    def destroy_shaw_removal(self, state):
        """
        Shaw相似性破坏算子：随机选一个客户点，移除与其空间位置最相近的若干客户点
        """
        new_state = state.fast_copy()
        rm_empty_vehicle_route = state.rm_empty_vehicle_route
        new_state.vehicle_routes = rm_empty_vehicle_route
        all_customers = list(self.A_c)
        if not all_customers:
            print("没有客户点")
            return new_state

        # 1. 随机选一个客户点作为种子
        seed_customer = self.rng.choice(all_customers)
        seed_pos = np.array([
            self.node[seed_customer].latDeg,
            self.node[seed_customer].lonDeg,
            self.node[seed_customer].altMeters
        ])

        # 2. 计算所有其他客户与种子的空间距离
        customer_distances = []
        for customer in all_customers:
            if customer == seed_customer:
                continue
            pos = np.array([
                self.node[customer].latDeg,
                self.node[customer].lonDeg,
                self.node[customer].altMeters
            ])
            dist = np.linalg.norm(pos - seed_pos)
            customer_distances.append((customer, dist))

        # 3. 按距离升序排序，选出最相似的若干客户
        customer_distances.sort(key=lambda x: x[1])
        n = len(customer_distances)
        num_to_remove = self.rng.integers(
            max(1, int(n * 0.2)),
            max(2, int(n * 0.3)) + 1
        )
        # 选出距离最近的num_to_remove个客户，加上种子
        customers_to_remove = [seed_customer] + [customer for customer, _ in customer_distances[:num_to_remove]]

        # 4. 从解中移除这些客户及相关无人机任务
        for customer in customers_to_remove:
            if customer in new_state.customer_plan:
                assignment = new_state.customer_plan.pop(customer)
                uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                # 从无人机分配中移除相关任务
                if uav_id in new_state.uav_assignments:
                    new_state.uav_assignments[uav_id] = [
                        task for task in new_state.uav_assignments[uav_id]
                        if task[2] != customer_node
                    ]
                # 更新无人机空中成本
                if hasattr(new_state, 'uav_cost') and new_state.uav_cost is not None:
                    new_state.uav_cost.pop(customer_node, None)
                # 更新对应的vehicle_task_data
                vehicle_task_data = new_state.vehicle_task_data
                vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                new_state.vehicle_task_data = vehicle_task_data

        # 5. 更新空跑节点等状态
        new_state.update_rm_empty_task()
        new_state.total_cost = new_state.objective()
        return new_state
    
    def destroy_important_removal(self, state):
        """
        重要性节点破坏：优先破坏无人机任务数量多的客户节点（发射+回收），
        但通过噪声实现一定的随机性，避免每次都只破坏最重要的节点。
        """
        new_state = state.fast_copy()
        # 更新路径
        rm_empty_vehicle_route = state.rm_empty_vehicle_route
        new_state.vehicle_routes = rm_empty_vehicle_route
        all_customers = list(self.A_c)
        new_state.vehicle_routes = rm_empty_vehicle_route
        new_state.destroyed_customers_info = state.destroyed_customers_info
        if not all_customers:
            print("没有客户点")
            return new_state

        # 1. 统计每个客户点的无人机任务数量（发射+回收）
        task_count = {c: 0 for c in all_customers}
        # 统计发射和回收任务
        for uav_id, tasks in new_state.uav_assignments.items():
            for task in tasks:
                # task结构：(drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle)
                _, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle = task
                if customer in task_count:
                    task_count[customer] += 1
                # 也可以统计launch_node和recovery_node是否为客户节点（如有需要）

        # 2. 按任务数量降序排序，加噪声
        # 生成噪声（正态分布，均值0，标准差1）
        noise = {c: self.rng.normal(0, 1) for c in all_customers}
        # 排序：任务数大+噪声高的优先
        customer_scores = [(c, task_count[c] + noise[c]) for c in all_customers]
        customer_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. 随机决定要破坏多少个节点（20%-30%）
        n = len(customer_scores)
        num_to_remove = self.rng.integers(
            max(1, int(n * 0.2)),
            max(2, int(n * 0.3)) + 1
        )
        customers_to_remove = [c for c, _ in customer_scores[:num_to_remove]]

        # 4. 依次移除这些节点的无人机任务
        for customer in customers_to_remove:
            if customer in new_state.customer_plan:
                assignment = new_state.customer_plan.pop(customer)
                uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                # 从无人机分配中移除相关任务
                if uav_id in new_state.uav_assignments:
                    new_state.uav_assignments[uav_id] = [
                        task for task in new_state.uav_assignments[uav_id]
                        if task[1] != customer_node
                    ]
                # 更新无人机空中成本
                if hasattr(new_state, 'uav_cost') and new_state.uav_cost is not None:
                    new_state.uav_cost.pop(customer_node, None)
                # 更新对应的vehicle_task_data
                vehicle_task_data = new_state.vehicle_task_data
                vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                new_state.vehicle_task_data = vehicle_task_data

        # 5. 更新空跑节点等状态
        new_state.update_rm_empty_task()
        new_state.total_cost = new_state.objective()
        return new_state
    
    def cluster_vtp_for_customers(self, k=8):
        """
        为每个客户点分配k个最近的VTP节点，按距离升序排列。
        返回: dict，key为客户点id，value为VTP节点list（按距离升序）
        """
        # import numpy as np
        vtp_ids = list(self.A_vtp)
        customer_ids = list(self.A_c)
        if not customer_ids or not vtp_ids:
            return {}

        # 获取所有VTP节点的坐标
        vtp_coords = np.array([
            [self.node[vtp_id].latDeg, self.node[vtp_id].lonDeg, self.node[vtp_id].altMeters]
            for vtp_id in vtp_ids
        ])
        # 获取所有客户节点的坐标
        customer_coords = np.array([
            [self.node[cid].latDeg, self.node[cid].lonDeg, self.node[cid].altMeters]
            for cid in customer_ids
        ])

        # 计算每个客户点到所有VTP的距离
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(customer_coords, vtp_coords)  # shape: (n_customers, n_vtp)

        customer_vtp_dict = {}
        for i, cid in enumerate(customer_ids):
            # 得到距离最近的k个VTP索引
            sorted_indices = np.argsort(dist_matrix[i])[:k]
            # 按距离升序排列的VTP节点
            sorted_vtps = [vtp_ids[j] for j in sorted_indices]
            customer_vtp_dict[cid] = sorted_vtps

        return customer_vtp_dict

    def remove_vehicle_task(self, vehicle_task_data, assignment, vehicle_routes):
        """
        从vehicle_task_data中移除指定的任务分配
        """
        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
        
        # 从发射车辆的任务数据中移除
        if launch_vehicle in vehicle_task_data and launch_node in vehicle_task_data[launch_vehicle]:
            task_data = vehicle_task_data[launch_vehicle][launch_node]
            if hasattr(task_data, 'launch_drone_list') and uav_id in task_data.launch_drone_list:
                task_data.launch_drone_list.remove(uav_id)
        
        # 从回收车辆的任务数据中移除
        if recovery_vehicle in vehicle_task_data and recovery_node in vehicle_task_data[recovery_vehicle]:
            task_data = vehicle_task_data[recovery_vehicle][recovery_node]
            if hasattr(task_data, 'recovery_drone_list') and uav_id in task_data.recovery_drone_list:
                task_data.recovery_drone_list.remove(uav_id)
        
        return vehicle_task_data

    def update_vehicle_task(self, vehicle_task_data, assignment, vehicle_routes):
        """
        向vehicle_task_data中添加指定的任务分配
        """
        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
        
        # 向发射车辆的任务数据中添加
        if launch_vehicle in vehicle_task_data and launch_node in vehicle_task_data[launch_vehicle]:
            task_data = vehicle_task_data[launch_vehicle][launch_node]
            if hasattr(task_data, 'launch_drone_list'):
                if uav_id not in task_data.launch_drone_list:
                    task_data.launch_drone_list.append(uav_id)
        
        # 向回收车辆的任务数据中添加
        if recovery_vehicle in vehicle_task_data and recovery_node in vehicle_task_data[recovery_vehicle]:
            task_data = vehicle_task_data[recovery_vehicle][recovery_node]
            if hasattr(task_data, 'recovery_drone_list'):
                if uav_id not in task_data.recovery_drone_list:
                    task_data.recovery_drone_list.append(uav_id)
        
        return vehicle_task_data

    def _create_snapshot(self, state):
        """创建状态快照 - 只在必要时进行深拷贝"""
        return FastMfstspState(
            copy.deepcopy(state.vehicle_routes),
            copy.deepcopy(state.uav_assignments),
            copy.deepcopy(state.customer_plan),
            copy.deepcopy(state.vehicle_task_data),  # 不复制，直接引用
            copy.deepcopy(state.global_reservation_table),  # 不复制，直接引用
            copy.deepcopy(state._total_cost)
        )
    
    def _incremental_destroy(self, state, modification_stack):
        """增量破坏：记录修改而不立即应用"""
        all_customers = list(state.customer_plan.keys())
        if not all_customers:
            return
        
        num_to_remove = self.rng.integers(
            max(1, len(all_customers) // 5), 
            max(2, len(all_customers) // 3)
        )
        
        customers_to_remove = self.rng.choice(all_customers, num_to_remove, replace=False)
        
        for customer in customers_to_remove:
            if customer in state.customer_plan:
                assignment = state.customer_plan.pop(customer)
                modification_stack.append(("add_customer", customer, assignment))
                
                # 从无人机分配中移除相关任务
                uav_id, _, _, _, _ = assignment
                if uav_id in state.uav_assignments:
                    old_assignments = state.uav_assignments[uav_id].copy()
                    state.uav_assignments[uav_id] = [
                        task for task in state.uav_assignments[uav_id]
                        if task[1] != customer
                    ]
                    modification_stack.append(("restore_uav_assignments", uav_id, old_assignments))
        
        state._total_cost = None
    
    def _incremental_repair(self, state, modification_stack):
        """增量修复：记录修改而不立即应用"""
        # 这里简化处理，实际应该实现更复杂的修复策略
        pass
    
    def _rollback_modifications(self, state, modification_stack):
        """回滚所有修改"""
        for modification in reversed(modification_stack):
            if modification[0] == "add_customer":
                _, customer, assignment = modification
                state.customer_plan[customer] = assignment
            elif modification[0] == "restore_uav_assignments":
                _, uav_id, assignments = modification
                state.uav_assignments[uav_id] = assignments
        
        state._total_cost = None


def create_fast_initial_state(init_total_cost, init_uav_plan, init_customer_plan, init_uav_cost,
                             init_time_uav_task_dict, init_vehicle_route, 
                             init_vehicle_plan_time, init_vehicle_task_data, 
                             init_global_reservation_table,node, DEPOT_nodeID, 
                             V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, 
                             A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, 
                             air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee
                             ):
    """
    从初始解创建FastMfstspState对象
    """
    # 转换车辆路线格式
    # vehicle_routes = {}
    # for i, route in enumerate(init_vehicle_route):
    #     vehicle_id = i + 1
    #     vehicle_routes[vehicle_id] = route
    
    return FastMfstspState(
        vehicle_routes=init_vehicle_route,
        uav_assignments=init_time_uav_task_dict,
        customer_plan=init_customer_plan,
        vehicle_task_data=init_vehicle_task_data,
        global_reservation_table=init_global_reservation_table,
        total_cost=init_total_cost,
        uav_cost = init_uav_cost,
        init_uav_plan=init_uav_plan,
        init_vehicle_plan_time = init_vehicle_plan_time,
        vehicle = vehicle,
        T = T,
        V = V,
        veh_distance = veh_distance,
        veh_travel = veh_travel,
        node = node,
        DEPOT_nodeID = DEPOT_nodeID,
        uav_travel = uav_travel,
        N = N,
        N_zero = N_zero,
        N_plus = N_plus,
        A_total = A_total,
        A_cvtp = A_cvtp,
        A_vtp = A_vtp,
        A_aerial_relay_node = A_aerial_relay_node,
        G_air = G_air,
        G_ground = G_ground,
        air_matrix = air_matrix,
        ground_matrix = ground_matrix,
        air_node_types = air_node_types,
        ground_node_types = ground_node_types,
        A_c = A_c,
        xeee = xeee
    )


def solve_with_fast_alns(initial_solution, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, 
		A_aerial_relay_node, G_air, G_ground,air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee,
        max_iterations=1000, max_runtime=60, use_incremental=True):
    """
    使用高效ALNS求解mFSTSP问题
    
    Args:
        initial_solution: 初始解
        max_iterations: 最大迭代次数
        max_runtime: 最大运行时间（秒）
        use_incremental: 是否使用增量式算法
        
    Returns:
        tuple: (best_solution, best_objective, statistics)
    """
    if use_incremental:
        # 使用增量式ALNS
        alns_solver = IncrementalALNS(node, DEPOT_nodeID, V, T, vehicle, uav_travel, 
        veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, 
		A_aerial_relay_node, G_air, G_ground,air_matrix, ground_matrix, air_node_types, 
        ground_node_types, A_c, xeee,
        max_iterations=max_iterations, max_runtime=max_runtime)
    else:
        # 使用快速ALNS
        alns_solver = FastALNS(max_iterations=max_iterations, max_runtime=max_runtime)
    
    # 使用ALNS求解
    best_solution, best_objective, statistics = alns_solver.solve(initial_solution)
    
    return best_solution, best_objective, statistics 

class FastALNS:
    """高效的ALNS求解器 - 使用浅拷贝和增量更新"""
    
    def __init__(self, max_iterations=1000, max_runtime=60):
        self.max_iterations = max_iterations
        self.max_runtime = max_runtime
        self.rng = rnd.default_rng(42)
        
    def solve(self, initial_state):
        """
        使用高效的ALNS算法求解
        
        Args:
            initial_state: 初始解状态
            
        Returns:
            tuple: (best_solution, best_objective, statistics)
        """
        current_state = initial_state.fast_copy()
        best_state = current_state.fast_copy()
        best_objective = best_state.objective()
        
        start_time = time.time()
        iteration = 0
        
        print(f"开始快速ALNS求解，初始成本: {best_objective}")
        
        while iteration < self.max_iterations and (time.time() - start_time) < self.max_runtime:
            # 破坏阶段 - 使用增量修改
            destroyed_state = self._fast_destroy(current_state)
            
            # 修复阶段 - 使用增量修改
            repaired_state = self._fast_repair(destroyed_state)
            
            # 接受准则（爬山法）
            if repaired_state.objective() < current_state.objective():
                current_state = repaired_state
                
                # 更新最优解
                if current_state.objective() < best_objective:
                    best_state = current_state.fast_copy()
                    best_objective = best_state.objective()
                    print(f"迭代 {iteration}: 发现更优解，成本: {best_objective}")
            
            iteration += 1
            
            # 每100次迭代输出一次进度
            if iteration % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"迭代 {iteration}, 当前成本: {current_state.objective()}, 最优成本: {best_objective}, 运行时间: {elapsed_time:.2f}秒")
        
        elapsed_time = time.time() - start_time
        statistics = {
            'iterations': iteration,
            'runtime': elapsed_time,
            'best_objective': best_objective
        }
        
        print(f"快速ALNS求解完成，最终成本: {best_objective}, 迭代次数: {iteration}, 运行时间: {elapsed_time:.2f}秒")
        
        return best_state, best_objective, statistics
    
    def _fast_destroy(self, state):
        """快速破坏算子：使用增量修改"""
        destroyed = state.fast_copy()
        
        # 获取所有客户点
        all_customers = list(destroyed.customer_plan.keys())
        if not all_customers:
            return destroyed
        
        # 随机移除20%-40%的客户点
        num_to_remove = self.rng.integers(
            max(1, len(all_customers) // 5), 
            max(2, len(all_customers) // 3)
        )
        
        customers_to_remove = self.rng.choice(all_customers, num_to_remove, replace=False)
        
        for customer in customers_to_remove:
            if customer in destroyed.customer_plan:
                assignment = destroyed.customer_plan.pop(customer)
                
                # 记录修改，用于可能的回滚
                destroyed.record_modification("remove_customer", (customer, assignment))
                
                # 从无人机分配中移除相关任务
                uav_id, _, _, _, _ = assignment
                if uav_id in destroyed.uav_assignments:
                    destroyed.uav_assignments[uav_id] = [
                        task for task in destroyed.uav_assignments[uav_id]
                        if task[1] != customer
                    ]
        
        destroyed._total_cost = None
        return destroyed
    
    def _fast_repair(self, destroyed_state):
        """快速修复算子：使用增量修改"""
        repaired = destroyed_state.fast_copy()
        
        # 这里简化处理，实际应该实现更复杂的修复策略
        # 对于被移除的客户点，可以重新分配到最佳的无人机和车辆组合
        
        repaired._total_cost = None
        return repaired

    def multiopt_update_best_scheme(self, best_scheme, near_node_list, vehicle_route, vehicle_task_data, sample_size=30):
        """
        加速多opt邻域搜索：对near_node_list随机采样sample_size个发射-回收节点组合，
        只计算本无人机和同节点相关无人机的成本，贪婪选择最优。
        返回(最优方案, 最优总成本)。
        """
        import random
        drone_id, launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
        best = best_scheme
        best_cost = float('inf')

        # 辅助：获取同节点相关无人机
        def get_related_drones(vehicle_id, node, task_data):
            related = set()
            if hasattr(task_data[vehicle_id][node], 'drone_list'):
                related.update(task_data[vehicle_id][node].drone_list)
            if hasattr(task_data[vehicle_id][node], 'launch_drone_list'):
                related.update(task_data[vehicle_id][node].launch_drone_list)
            if hasattr(task_data[vehicle_id][node], 'recovery_drone_list'):
                related.update(task_data[vehicle_id][node].recovery_drone_list)
            return related

        # 计算本无人机和同节点相关无人机的总成本
        def get_greedy_cost(vehicle_id, l_n, r_n):
            total = 0
            # 本无人机
            total += self.drone_insert_cost(drone_id, customer, l_n, r_n)
            # 相关无人机（发射/回收节点）
            related = get_related_drones(vehicle_id, l_n, vehicle_task_data) | get_related_drones(vehicle_id, r_n, vehicle_task_data)
            related.discard(drone_id)
            for d_id in related:
                # 查找d_id的发射/回收节点
                launch_n, recovery_n = None, None
                route = vehicle_route[vehicle_id - 1]
                for n2 in route:
                    if hasattr(vehicle_task_data[vehicle_id][n2], 'launch_drone_list') and d_id in vehicle_task_data[vehicle_id][n2].launch_drone_list:
                        launch_n = n2
                    if hasattr(vehicle_task_data[vehicle_id][n2], 'recovery_drone_list') and d_id in vehicle_task_data[vehicle_id][n2].recovery_drone_list:
                        recovery_n = n2
                if launch_n and recovery_n:
                    total += self.drone_insert_cost(d_id, customer, launch_n, recovery_n)
            return total

        # 单车情况
        if launch_vehicle_id == recovery_vehicle_id:
            node_list = near_node_list
            # 采样sample_size个不同组合
            candidates = set()
            while len(candidates) < sample_size:
                l_n = random.choice(node_list)
                r_n = random.choice(node_list)
                if l_n != r_n:
                    candidates.add((l_n, r_n))
            for l_n, r_n in candidates:
                cost = get_greedy_cost(launch_vehicle_id, l_n, r_n)
                if cost < best_cost:
                    best = (drone_id, l_n, customer, r_n, launch_vehicle_id, recovery_vehicle_id)
                    best_cost = cost
            return best, best_cost
        else:
            # 异车情况
            launch_list = near_node_list[launch_vehicle_id]
            recovery_list = near_node_list[recovery_vehicle_id]
            candidates = set()
            while len(candidates) < sample_size:
                l_n = random.choice(launch_list)
                r_n = random.choice(recovery_list)
                if l_n != r_n:
                    candidates.add((l_n, r_n))
            for l_n, r_n in candidates:
                cost = get_greedy_cost(launch_vehicle_id, l_n, r_n) + get_greedy_cost(recovery_vehicle_id, l_n, r_n)
                if cost < best_cost:
                    best = (drone_id, l_n, customer, r_n, launch_vehicle_id, recovery_vehicle_id)
                    best_cost = cost
            return best, best_cost
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
from task_data import *
import main
import endurance_calculator
import distance_functions
from visualization_best import visualize_plan
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

    def calculate_rm_empty_vehicle_arrive_time(self, vehicle_route):  # 实际是计算去除空跑节点后每辆车到达各节点的时间
        """
        计算去除空跑节点后每辆车到达各节点的时间
        返回: dict，key为vehicle_id，value为{node_id: 到达时间}
        """
        rm_empty_vehicle_arrive_time = {}
        # for vehicle_id, route in enumerate(self.rm_empty_vehicle_route):
        for vehicle_id, route in enumerate(vehicle_route):
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
        # 关键修复：确保customer_plan始终是普通字典，不是defaultdict
        if isinstance(self.customer_plan, defaultdict):
            customer_plan_copy = dict(self.customer_plan)
        else:
            customer_plan_copy = self.customer_plan.copy()
            
        new_state = FastMfstspState(
            vehicle_routes=self.vehicle_routes.copy(),
            uav_assignments={k: v.copy() for k, v in self.uav_assignments.items()},
            customer_plan=customer_plan_copy,
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
            new_state.rm_empty_vehicle_route = [route[:] for route in self.rm_empty_vehicle_route] if self.rm_empty_vehicle_route else []
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
            
        # 修复：添加缺失的destroyed_vts_info属性复制
        if hasattr(self, 'destroyed_vts_info'):
            new_state.destroyed_vts_info = self.destroyed_vts_info.copy() if self.destroyed_vts_info else {}
        else:
            new_state.destroyed_vts_info = {}
        
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
        max_iterations=50, max_runtime=60):
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
        self.dis_k = 25  # 修改距离客户点最近的vtp节点集合，增加解空间
        self.base_drone_assignment = self.base_drone_assigment()
        # self.base_vehicle_task_data = DiverseRouteGenerator.create_vehicle_task_data(self.node, self.DEPOT_nodeID, self.V, self.T, self.vehicle, self.uav_travel, self.veh_distance, self.veh_travel, self.N, self.N_zero, self.N_plus, self.A_total, self.A_cvtp, self.A_vtp, self.A_aerial_relay_node, self.G_air, self.G_ground, self.air_matrix, self.ground_matrix, self.air_node_types, self.ground_node_types, self.A_c, self.xeee)
        # 破坏算子参数
        self.customer_destroy_ratio = (0.2, 0.4)
        self.vtp_destroy_quantity = {'random': (1, 2), 'worst': 1, 'shaw': (2, 4)}
        self.cluster_vtp_dict, self.map_cluster_vtp_dict = self.cluster_vtp_for_customers(k=self.dis_k)
        # 定义算子池，方便后续引用
        # self.destroy_operators = [self.destroy_random_removal, self.destroy_worst_removal, self.destroy_shaw_removal]
        self.destroy_operators = [self.destroy_random_removal, self.destroy_worst_removal]
        # self.destroy_operators = [self.destroy_random_removal]

        self.repair_operators = [self.repair_greedy_insertion]
        # self.repair_operators = [self.repair_greedy_insertion, self.repair_regret_insertion,self.repair_k_insertion]
        # --- 1. 定义两层自适应权重体系 ---
        # 第一层：战略权重
        # --- 2. 定义评分和学习参数 ---
        self.reward_scores = {
            'new_best': 10,  # 找到全局最优解的得分
            'better_than_current': 5,  # 找到比当前解更好的解的得分
            'accepted': 2  # 接受一个较差解（探索成功）的得分
        }
        self.reaction_factor = 0.5 # 学习率  0.5-0.9
        self.strategy_weights = {
            'structural': 1.0,
            'internal': 1.0
        }

        # 第二层：与策略绑定的算子权重
        self.operator_weights = {
            'structural': {
                'destroy': {op.__name__: 1.0 for op in self.destroy_operators},
                'repair':  {op.__name__: 1.0 for op in self.repair_operators}
            },
            'internal': {
                'destroy': {op.__name__: 1.0 for op in self.destroy_operators},
                'repair':  {op.__name__: 1.0 for op in self.repair_operators}
            }
        }
    
    def base_drone_assigment(self):
        """
        基础无人机分配函数
        将无人机均匀分配给车辆，每个车辆分配连续的无人机ID
        
        Returns:
            dict: 车辆ID为key，无人机ID列表为value的字典
            例如: 6个无人机，3个车辆 -> {1: [1, 2], 2: [3, 4], 3: [5, 6]}
        """
        # 获取车辆数量和无人机数量
        num_vehicles = len(self.T)
        num_drones = len(self.V)
        
        # 创建基础分配字典
        base_assignment = {}
        
        # 计算每个车辆应该分配的无人机数量
        drones_per_vehicle = num_drones // num_vehicles
        remaining_drones = num_drones % num_vehicles
        
        drone_start = 1+num_drones  # 无人机ID从1开始
        
        for vehicle_id in range(1, num_vehicles + 1):
            # 计算当前车辆应该分配的无人机数量
            current_drone_count = drones_per_vehicle
            if vehicle_id <= remaining_drones:  # 前几个车辆多分配一个无人机
                current_drone_count += 1
            
            # 分配连续的无人机ID
            vehicle_drones = list(range(drone_start, drone_start + current_drone_count))
            base_assignment[vehicle_id] = vehicle_drones
            
            # 更新下一个车辆的起始无人机ID
            drone_start += current_drone_count
        
        # print(f"基础无人机分配完成:")
        # for vehicle_id, drones in base_assignment.items():
        #     print(f"  车辆 {vehicle_id}: 无人机 {drones}")
        
        return base_assignment


    def repair_greedy_insertion(self, state, strategic_bonus, num_destroyed, force_vtp_mode):
        """
        贪婪插入修复算子：将被移除的客户点按成本最小原则重新插入，记录所有插入方案。
        返回修复后的状态和所有破坏节点的最优插入方案列表。
        """
        # 添加调试信息
        print(f"DEBUG: repair_greedy_insertion开始，state.customer_plan类型: {type(state.customer_plan)}")
        print(f"DEBUG: repair_greedy_insertion开始，state.customer_plan的ID: {id(state.customer_plan)}")
        print(f"DEBUG: repair_greedy_insertion开始，节点72是否在customer_plan中: {72 in state.customer_plan}")
        if 72 in state.customer_plan:
            print(f"DEBUG: repair_greedy_insertion开始，节点72的值: {state.customer_plan[72]}")
        
        # 关键修复：必须创建状态副本，避免修改原始状态
        repaired_state = state.fast_copy()  # 修复：创建真正的副本
        # repaired_state = state  # 这里不复制会导致destroyed_state被意外修改！
        
        print(f"DEBUG: repaired_state.customer_plan的ID: {id(repaired_state.customer_plan)}")
        print(f"DEBUG: state和repaired_state是否指向同一个对象: {repaired_state is state}")
        print(f"DEBUG: customer_plan是否指向同一个对象: {repaired_state.customer_plan is state.customer_plan}")
        print(f"DEBUG: repaired_state.customer_plan类型: {type(repaired_state.customer_plan)}")
        
        # 额外安全检查：确保repaired_state.customer_plan不是defaultdict
        if isinstance(repaired_state.customer_plan, defaultdict):
            print("DEBUG: 警告！repaired_state.customer_plan仍然是defaultdict，强制转换为普通字典")
            repaired_state.customer_plan = dict(repaired_state.customer_plan)
        # destroy_node = list(set(self.A_c) - set(state.customer_plan.keys()))
        destroy_node = list(state.destroyed_customers_info.keys())  # 总结出了所有的待插入的破坏节点
        insert_plan = []  # 记录所有破坏节点的最优插入方案

        # print(f"贪婪修复：需要插入 {len(destroy_node)} 个客户点: {destroy_node}")
        if force_vtp_mode:
            num_repaired = 0
            while len(destroy_node) > 0:
                best_option_overall = None
                best_customer_to_insert = None
                min_overall_eval_cost = float('inf')
                # a. 计算本轮决策的"最终奖励"(final_bonus)
                tactical_multiplier = (num_destroyed - num_repaired) / num_destroyed
                final_bonus = strategic_bonus * tactical_multiplier * 0.3
                # final_bonus = 0
                
                # 获取当前状态的数据
                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                # vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                
                # 存储所有候选方案
                all_candidates = []
                
                # 遍历所有待插入客户点，计算每个节点的最优插入成本
                for customer in destroy_node:
                    customer_candidates = []
                    
                    # 1. 首先尝试传统插入方案（使用现有节点）
                    traditional_result = self._evaluate_traditional_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
                    if traditional_result is not None:
                        traditional_cost, traditional_scheme = traditional_result
                        customer_candidates.append({
                            'customer': customer,
                            'scheme': traditional_scheme,
                            'cost': traditional_cost,
                            'type': 'traditional',
                            'vtp_node': None
                        })
                    else:
                        # 传统插入方案失败，设置成本为无穷大
                        customer_candidates.append({
                            'customer': customer,
                            'scheme': None,
                            'cost': float('inf'),
                            'type': 'traditional',
                            'vtp_node': None
                        })
                    
                    # 2. 考虑VTP扩展插入方案（为每个客户点考虑新增VTP节点）
                    vtp_result,vtp_infor = self._evaluate_vtp_expansion_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
                    # 通过vtp_infor获得插入信息
                    if vtp_infor is not None:
                        vtp_node = vtp_infor[0]
                        vtp_insert_vehicle_id = vtp_infor[1]
                        vtp_insert_index = vtp_infor[2]
                        if vtp_result is not None:
                            vtp_cost, vtp_scheme = vtp_result
                            # 应用最终奖励来增加VTP插入在前期被选中的概率
                            adjusted_cost = vtp_cost - final_bonus
                            
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': vtp_scheme,
                                'cost': adjusted_cost,
                                'type': 'vtp_expansion',
                                'vtp_node': vtp_node,  # launch_node就是VTP节点
                                'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                                'vtp_insert_index': vtp_insert_index,
                                'original_cost': vtp_cost
                            })
                    
                # 3. 为当前客户点选择最优方案
                if customer_candidates:
                    best_customer_scheme = min(customer_candidates, key=lambda x: x['cost'])
                    all_candidates.append(best_customer_scheme)
                
                # 选择全局最优的插入方案
                if not all_candidates:
                    print("所有剩余节点都没有可行插入方案，修复终止")
                    break
                
                # 按成本排序所有候选方案
                all_candidates.sort(key=lambda x: x['cost'])
                
                # 尝试每个候选方案，直到找到满足约束的方案
                success = False
                for candidate in all_candidates:
                    customer = candidate['customer']
                    best_scheme = candidate['scheme']
                    best_cost = candidate['cost']
                    
                    # 根据方案类型执行不同的插入逻辑
                    if candidate['type'] == 'traditional':
                        print(f"尝试使用传统方案插入客户点 {customer}，成本: {best_cost:.2f}")
                        # 使用传统插入方案 - 采用统一的后续处理方式
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        
                        # 创建临时状态进行约束检查
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        
                        # 检查时间约束
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                            print(f"传统方案时间约束不满足，尝试下一个候选方案")
                            continue
                        
                        # 约束满足，执行插入
                        # 更新customer_plan
                        repaired_state.customer_plan[customer_node] = best_scheme
                        
                        # 更新uav_assignments
                        if drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[drone_id] = []
                        repaired_state.uav_assignments[drone_id].append(best_scheme)
                        
                        # 更新uav_cost
                        if repaired_state.uav_cost is None:
                            repaired_state.uav_cost = {}
                        repaired_state.uav_cost[customer_node] = best_cost
                        
                        # 更新vehicle_task_data
                        vehicle_task_data = update_vehicle_task(
                            vehicle_task_data, best_scheme, vehicle_route
                        )
                        
                        # 记录插入方案
                        insert_plan.append((customer, best_scheme, best_cost, 'traditional'))
                        # print(f"成功使用传统方案插入客户点 {customer}，成本: {best_cost:.2f}")
                        success = True
                        break
                        
                    elif candidate['type'] == 'vtp_expansion':
                        # VTP扩展插入方案 - 采用统一的后续处理方式，并额外更新车辆路线
                        # print(f"尝试使用VTP扩展方案插入客户点 {customer}，成本: {best_cost:.2f}")
                        vtp_node = candidate['vtp_node']
                        vtp_insert_index = candidate['vtp_insert_index']
                        vtp_insert_vehicle_id = candidate['vtp_insert_vehicle_id']
                        original_cost = candidate['original_cost']
                    
                        # 1. 首先将VTP节点插入到车辆路径中
                        # 从方案中提取车辆ID和插入位置
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme

                        # 2. 创建临时状态进行约束检查
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        # 生成临时的车辆路线，避免指向同一对象
                        temp_vehicle_route = [route[:] for route in vehicle_route]
                        temp_route = temp_vehicle_route[vtp_insert_vehicle_id - 1]
                        temp_route.insert(vtp_insert_index, vtp_node)
                        temp_vehicle_route[vtp_insert_vehicle_id - 1] = temp_route
                        repaired_state.temp_vehicle_routes = temp_vehicle_route
                        # 计算临时车辆到达时间
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                        
                        # 3. 检查时间约束
                        time_feasible = is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time)
                        
                        if not time_feasible:
                            # 时间约束不满足，尝试下一个候选方案
                            print(f"VTP扩展方案时间约束不满足，尝试下一个候选方案")
                            continue
                        else:
                            # 找到VTP节点在路径中的插入位置
                            route = vehicle_route[vtp_insert_vehicle_id - 1]

                            # 更新车辆路线 - VTP方案特有的操作
                            route.insert(vtp_insert_index, vtp_node)
                            # 找到上一个客户点更新vehicle_task_data的新插入数据，随后更新
                            last_customer_node = route[vtp_insert_index - 1]
                            # 如果索引是1或者前一个节点是起始节点，那么按照初始的无人机分配来
                            if vtp_insert_index == 1 or last_customer_node == self.DEPOT_nodeID:
                                last_drone_list = self.base_drone_assignment[vtp_insert_vehicle_id][:]
                            else:
                                last_drone_list = vehicle_task_data[vtp_insert_vehicle_id][last_customer_node].drone_list[:]

                            # last_drone_list = vehicle_task_data[vtp_insert_vehicle_id][last_customer_node].drone_list[:]
                            vehicle_task_data[vtp_insert_vehicle_id][vtp_node].drone_list = last_drone_list
                            vehicle_task_data[vtp_insert_vehicle_id][vtp_node].launch_drone_list = []
                            vehicle_task_data[vtp_insert_vehicle_id][vtp_node].recovery_drone_list = []

                            # 更新vehicle_task_data以反映新的VTP节点
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                            # 2. 采用统一的后续处理方式
                            # 更新customer_plan
                            repaired_state.customer_plan[customer_node] = best_scheme
                            
                            # 更新uav_assignments
                            if drone_id not in repaired_state.uav_assignments:
                                repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(best_scheme)
                            
                            # 更新uav_cost
                            if repaired_state.uav_cost is None:
                                repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = original_cost
                            
                            # 更新vehicle_task_data（VTP方案已经通过_update_vehicle_task_data_for_vtp更新过）
                            repaired_state.rm_empty_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
                            repaired_state.rm_empty_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                            # 记录插入方案
                            insert_plan.append((customer, best_scheme, original_cost, 'vtp_expansion'))
                            print(f"成功新增VTP节点 {vtp_node} 并插入客户点 {customer}，总成本: {original_cost:.2f}")
                            success = True
                            break
                
                # 如果所有候选方案都不满足约束，跳过当前客户点
                if not success:
                    print(f"客户点 {customer} 的所有候选方案都不满足约束，跳过")
                    continue
                
                # 从待插入列表中移除已处理的客户点
                if customer in destroy_node:
                    destroy_node.remove(customer)
                
                num_repaired += 1
        else:

            while len(destroy_node) > 0:
                # print(f"当前剩余待插入节点: {destroy_node}")
                
                # 存储所有节点的插入成本信息
                all_insertion_costs = []
            
                # 获取当前状态的数据
                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                
                # 存储直接插入和启发式交换的候选方案
                direct_insertion_candidates = []  # 直接插入候选方案
                heuristic_swap_candidates = []    # 启发式交换候选方案
                
                # 遍历所有待插入客户点，计算每个节点的最优插入成本
                for customer in destroy_node:
                    min_cost = float('inf')
                    best_scheme = None
                    
                    # 获取所有可行的插入位置
                    all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
                    if all_insert_position is not None:
                        # 统计该客户点的插入位置总数
                        total_positions = sum(len(positions) for positions in all_insert_position.values())
                        # print(f"客户点 {customer} 找到 {total_positions} 个可行插入位置")
                        # 遍历所有可行插入位置，找到成本最小的方案
                        for drone_id, inert_positions in all_insert_position.items():
                            for inert_position in inert_positions:
                                launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                                insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                                if insert_cost < min_cost:
                                    min_cost = insert_cost
                                    best_scheme = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                    if best_scheme is not None:
                        # 直接插入模式：记录直接插入方案
                        direct_insertion_candidates.append({
                            'customer': customer,
                            'scheme': best_scheme,
                            'cost': min_cost,
                            'type': 'direct'
                        })
                        # print(f"客户点 {customer} 直接插入成本: {min_cost:.2f}")
                    else:
                        # 启发式交换模式：尝试交换策略（增加成本阈值检查）
                        print(f"客户点 {customer} 没有直接插入方案，尝试启发式交换策略")
                        
                        # 计算当前解的总成本作为基准
                        current_total_cost = sum(repaired_state.uav_cost.values()) if repaired_state.uav_cost else 0
                        cost_improvement_threshold = 0.1  # 只有当交换能带来10%以上的成本改善时才进行
                        
                        # try:
                            # 通过启发式的贪婪算法插入方案（交换策略）
                        best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = DiverseRouteGenerator.greedy_insert_feasible_plan(
                            customer, vehicle_route, vehicle_arrive_time, vehicle_task_data, repaired_state.customer_plan
                        )
                            
                        if best_orig_y is not None and best_new_y is not None:
                            # 计算总成本（移除成本 + 插入成本）
                            total_swap_cost = best_orig_cost + best_new_cost
                            
                            # 检查成本改善是否达到阈值
                            cost_improvement = (current_total_cost - total_swap_cost) / max(current_total_cost, 1)
                            if cost_improvement >= cost_improvement_threshold:
                                heuristic_swap_candidates.append({
                                    'customer': customer,
                                    'orig_scheme': best_orig_y,
                                    'new_scheme': best_new_y,
                                    'orig_cost': best_orig_cost,
                                    'new_cost': best_new_cost,
                                    'total_cost': total_swap_cost,
                                    'orig_plan': best_orig_y_cijkdu_plan,
                                    'new_plan': best_new_y_cijkdu_plan,
                                    'type': 'heuristic_swap'
                                })
                                print(f"客户点 {customer} 启发式交换总成本: {total_swap_cost:.2f} (移除: {best_orig_cost:.2f} + 插入: {best_new_cost:.2f}), 成本改善: {cost_improvement:.2%}")
                            else:
                                print(f"客户点 {customer} 启发式交换成本改善不足 ({cost_improvement:.2%} < {cost_improvement_threshold:.2%})，跳过")
                        else:
                            print(f"客户点 {customer} 启发式交换也失败")
                        # except Exception as e:
                        #     print(f"客户点 {customer} 启发式插入失败: {e}")
                
                # 选择最优插入方案
                all_candidates = direct_insertion_candidates + heuristic_swap_candidates
                
                if not all_candidates:
                    print("所有剩余节点都没有可行插入方案，修复终止")
                    return repaired_state, insert_plan
                
                # 选择成本最低的方案
                if direct_insertion_candidates:
                    # 优先选择直接插入方案中成本最低的
                    best_candidate = min(direct_insertion_candidates, key=lambda x: x['cost'])
                    # print(f"选择直接插入模式：客户点 {best_candidate['customer']}，成本: {best_candidate['cost']:.2f}")
                else:
                    # 如果没有直接插入方案，选择启发式交换中总成本最低的
                    best_candidate = min(heuristic_swap_candidates, key=lambda x: x['total_cost'])
                    # print(f"选择启发式交换模式：客户点 {best_candidate['customer']}，总成本: {best_candidate['total_cost']:.2f}")
                
                # 根据方案类型执行不同的插入逻辑
                if best_candidate['type'] == 'direct':
                    # 直接插入模式
                    customer = best_candidate['customer']
                    best_scheme = best_candidate['scheme']
                    best_cost = best_candidate['cost']
                    
                    # 应用直接插入方案
                    drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                    
                    # 更新customer_plan
                    repaired_state.customer_plan[customer_node] = best_scheme
                    
                    # 更新uav_assignments
                    if drone_id not in repaired_state.uav_assignments:
                        repaired_state.uav_assignments[drone_id] = []
                    repaired_state.uav_assignments[drone_id].append(best_scheme)
                    
                    # 更新uav_cost
                    if repaired_state.uav_cost is None:
                        repaired_state.uav_cost = {}
                    repaired_state.uav_cost[customer_node] = best_cost
                    
                    # 更新vehicle_task_data
                    repaired_state.vehicle_task_data = update_vehicle_task(
                        repaired_state.vehicle_task_data, best_scheme, vehicle_route
                    )
                    
                    # 记录插入方案
                    insert_plan.append((customer, best_scheme, best_cost))
                    if customer in destroy_node:  # 安全检查，避免重复删除
                        destroy_node.remove(customer)
                    
                    # print(f"成功直接插入客户点 {customer}，成本: {best_cost:.2f}")    
                else:
                    # 启发式交换模式
                    customer = best_candidate['customer']
                    best_orig_y = best_candidate['orig_scheme']
                    best_new_y = best_candidate['new_scheme']
                    best_orig_cost = best_candidate['orig_cost']
                    best_new_cost = best_candidate['new_cost']
                    best_orig_y_cijkdu_plan = best_candidate['orig_plan']
                    best_new_y_cijkdu_plan = best_candidate['new_plan']
                    
                    # 解析交换方案
                    orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = best_orig_y
                    new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = best_new_y
                    
                    # 确定要移除的客户点
                    if orig_customer == customer:
                        remove_customer = new_customer
                    else:
                        remove_customer = orig_customer
                    
                    # print(f"启发式交换：移除客户点 {remove_customer}，插入客户点 {customer}")
                    
                    # 删除被移除的方案
                    if remove_customer in repaired_state.customer_plan:
                        y = repaired_state.customer_plan[remove_customer]
                        del repaired_state.customer_plan[remove_customer]
                        if repaired_state.uav_cost and remove_customer in repaired_state.uav_cost:
                            del repaired_state.uav_cost[remove_customer]
                        
                        # 从无人机分配中移除
                        drone_id_remove, _, _, _, _, _ = y
                        if drone_id_remove in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[drone_id_remove] = [
                                task for task in repaired_state.uav_assignments[drone_id_remove]
                                if task[2] != remove_customer
                            ]
                        
                        vehicle_task_data = remove_vehicle_task(vehicle_task_data, y, vehicle_route)
                    
                    # 根据时间优先级选择插入方案
                    if best_new_y_cijkdu_plan['launch_time'] < best_orig_y_cijkdu_plan['launch_time']:
                        # 插入新方案
                        repaired_state.customer_plan[new_customer] = best_new_y
                        if repaired_state.uav_cost is None:
                            repaired_state.uav_cost = {}
                        repaired_state.uav_cost[new_customer] = best_new_cost
                        
                        # 更新无人机分配
                        if new_drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[new_drone_id] = []
                        repaired_state.uav_assignments[new_drone_id].append(best_new_y)
                        
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                        final_scheme = best_new_y
                        final_cost = best_new_cost
                    else:
                        # 插入原方案
                        repaired_state.customer_plan[orig_customer] = best_orig_y
                        if repaired_state.uav_cost is None:
                            repaired_state.uav_cost = {}
                        repaired_state.uav_cost[orig_customer] = best_orig_cost
                        
                        # 更新无人机分配
                        if orig_drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[orig_drone_id] = []
                        repaired_state.uav_assignments[orig_drone_id].append(best_orig_y)
                        
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)
                        final_scheme = best_orig_y
                        final_cost = best_orig_cost
                    
                    # 记录插入方案
                    insert_plan.append((customer, final_scheme, final_cost))
                    if customer in destroy_node:  # 安全检查，避免重复删除
                        destroy_node.remove(customer)
                    
                    print(f"成功启发式交换插入客户点 {customer}，最终成本: {final_cost:.2f}")
            
            # 更新空跑节点等状态
            # repaired_state.update_rm_empty_task()
            # repaired_state._total_cost = None  # 重置成本缓存
            
            # print(f"当前已插入 {len(insert_plan)} 个节点")
            # print("-" * 50)
        # 更新修复完成后的成本
        repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        # print(f"贪婪修复完成：成功插入 {len(insert_plan)} 个客户点")
        
        # 添加调试信息
        print(f"DEBUG: repair_greedy_insertion结束，repaired_state.customer_plan类型: {type(repaired_state.customer_plan)}")
        print(f"DEBUG: repair_greedy_insertion结束，节点72是否在customer_plan中: {72 in repaired_state.customer_plan}")
        if 72 in repaired_state.customer_plan:
            print(f"DEBUG: repair_greedy_insertion结束，节点72的值: {repaired_state.customer_plan[72]}")
            if repaired_state.customer_plan[72] == []:
                print("DEBUG: 警告！repair_greedy_insertion结束时节点72的值为空列表！")
        
        return repaired_state, insert_plan

    def get_near_node_list(self, best_scheme, k, vehicle_route):
        """
        根据best_scheme的车辆id，找到该车辆的路线vehicle_route[v_id-1]，
        然后找到距离客户点c最近的聚类的k个地面节点，且这些节点不能出现在该车辆路线中。
        如果发射车辆和回收车辆不同，则返回dict，key为车辆id，value为各自可插入节点list；否则返回单一车辆的list。
        """
        # best_scheme: (drone_id, launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
        _, _, customer, _, launch_vehicle_id, recovery_vehicle_id = best_scheme
        customer_vtp_dict = self.cluster_vtp_for_customers()  # 取较大k，后面筛选
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
        # insert_cost = 0
        launch_node_map_index = self.node[launch_node].map_key
        recovery_node_map_index = self.node[recovery_node].map_key
        customer_map_index = self.node[customer].map_key
        insert_cost = self.uav_travel[drone_id][launch_node_map_index][customer].totalDistance+ self.uav_travel[drone_id][customer][recovery_node_map_index].totalDistance
        per_cost = self.vehicle[drone_id].per_cost
        insert_cost = insert_cost * per_cost
        return insert_cost

    def repair_regret_insertion(self, state):
        """
        后悔插入修复算子：将被移除的客户点按后悔值最大原则重新插入，记录所有插入方案。
        后悔值 = 次优插入成本 - 最优插入成本，后悔值越大说明越应该优先插入。
        返回修复后的状态和所有破坏节点的最优插入方案列表。
        """
        # repaired_state = state.fast_copy()
        repaired_state = state
        # destroy_node = list(set(self.A_c) - set(state.customer_plan.keys()))
        destroy_node = list(state.destroyed_customers_info.keys())  # 总结出了所有的待插入的破坏节点
        insert_plan = []  # 记录所有破坏节点的最优插入方案

        # print(f"后悔修复：需要插入 {len(destroy_node)} 个客户点: {destroy_node}")

        while len(destroy_node) > 0:
            # print(f"当前剩余待插入节点: {destroy_node}")
            
            # 存储所有节点的后悔值信息
            regret_candidates = []  # 后悔值候选方案
            
            # 获取当前状态的数据
            vehicle_route = repaired_state.rm_empty_vehicle_route
            vehicle_task_data = repaired_state.vehicle_task_data
            vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
        
            # 遍历所有待插入客户点，计算每个节点的后悔值
            for customer in destroy_node:
                # 存储该客户点的所有插入成本
                all_costs = []
                all_schemes = []
                
                # 获取所有可行的插入位置
                all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
                if all_insert_position is not None:
                    # 统计该客户点的插入位置总数
                    total_positions = sum(len(positions) for positions in all_insert_position.values())
                    # print(f"后悔插入 - 客户点 {customer} 找到 {total_positions} 个可行插入位置")
                    # 遍历所有可行插入位置，收集所有成本
                    for drone_id, inert_positions in all_insert_position.items():
                        for inert_position in inert_positions:
                                launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                                insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                                scheme = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                                all_costs.append(insert_cost)
                                all_schemes.append(scheme)
                
                # 计算后悔值
                if len(all_costs) >= 2:
                    # 有多个选择，计算后悔值
                    sorted_costs = sorted(all_costs)
                    best_cost = sorted_costs[0]
                    second_best_cost = sorted_costs[1]
                    regret_value = second_best_cost - best_cost
                    
                    # 找到最优方案
                    best_scheme = all_schemes[all_costs.index(best_cost)]
                    
                    regret_candidates.append({
                        'customer': customer,
                        'scheme': best_scheme,
                        'cost': best_cost,
                        'regret_value': regret_value,
                        'type': 'direct'
                    })
                    # print(f"客户点 {customer} 直接插入成本: {best_cost:.2f}, 后悔值: {regret_value:.2f}")
                    
                elif len(all_costs) == 1:
                    # 只有一个选择，后悔值为0
                    best_cost = all_costs[0]
                    best_scheme = all_schemes[0]
                    regret_value = 0
                    
                    regret_candidates.append({
                        'customer': customer,
                        'scheme': best_scheme,
                        'cost': best_cost,
                        'regret_value': regret_value,
                        'type': 'direct'
                    })
                    # print(f"客户点 {customer} 直接插入成本: {best_cost:.2f}, 后悔值: {regret_value:.2f}")
                    
                else:
                    # 没有直接插入方案，尝试启发式交换策略
                    print(f"客户点 {customer} 没有直接插入方案，尝试启发式交换策略")
                    
                    try:
                        # 通过启发式的贪婪算法插入方案（交换策略）
                        best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = DiverseRouteGenerator.greedy_insert_feasible_plan(
                            customer, vehicle_route, vehicle_arrive_time, vehicle_task_data, repaired_state.customer_plan
                        )
                        
                        if best_orig_y is not None and best_new_y is not None:
                            # 计算总成本（移除成本 + 插入成本）
                            total_swap_cost = best_orig_cost + best_new_cost
                            
                            regret_candidates.append({
                                'customer': customer,
                                'orig_scheme': best_orig_y,
                                'new_scheme': best_new_y,
                                'orig_cost': best_orig_cost,
                                'new_cost': best_new_cost,
                                'total_cost': total_swap_cost,
                                'orig_plan': best_orig_y_cijkdu_plan,
                                'new_plan': best_new_y_cijkdu_plan,
                                'regret_value': float('inf'),  # 启发式交换的后悔值设为无穷大，优先选择
                                'type': 'heuristic_swap'
                            })
                            print(f"客户点 {customer} 启发式交换总成本: {total_swap_cost:.2f} (移除: {best_orig_cost:.2f} + 插入: {best_new_cost:.2f}), 后悔值: ∞")
                        else:
                            print(f"客户点 {customer} 启发式交换也失败")
                    except Exception as e:
                        print(f"客户点 {customer} 启发式插入失败: {e}")
            
            # 选择后悔值最大的方案
            if not regret_candidates:
                print("所有剩余节点都没有可行插入方案，修复终止")
                break
            
            # 按后悔值降序排序，选择后悔值最大的
            regret_candidates.sort(key=lambda x: x['regret_value'], reverse=True)
            best_candidate = regret_candidates[0]
            
            # if best_candidate['type'] == 'direct':
            #     print(f"选择直接插入模式：客户点 {best_candidate['customer']}，成本: {best_candidate['cost']:.2f}, 后悔值: {best_candidate['regret_value']:.2f}")
            # else:
            #     print(f"选择启发式交换模式：客户点 {best_candidate['customer']}，总成本: {best_candidate['total_cost']:.2f}, 后悔值: ∞")
            
            # 根据方案类型执行不同的插入逻辑
            if best_candidate['type'] == 'direct':
                # 直接插入模式
                customer = best_candidate['customer']
                best_scheme = best_candidate['scheme']
                best_cost = best_candidate['cost']
                
                # 应用直接插入方案
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                
                # 更新customer_plan
                repaired_state.customer_plan[customer_node] = best_scheme
                
                # 更新uav_assignments
                if drone_id not in repaired_state.uav_assignments:
                    repaired_state.uav_assignments[drone_id] = []
                repaired_state.uav_assignments[drone_id].append(best_scheme)
                
                # 更新uav_cost
                if repaired_state.uav_cost is None:
                    repaired_state.uav_cost = {}
                repaired_state.uav_cost[customer_node] = best_cost
                
                # 更新vehicle_task_data
                repaired_state.vehicle_task_data = update_vehicle_task(
                    repaired_state.vehicle_task_data, best_scheme, vehicle_route
                )
                
                # 记录插入方案
                insert_plan.append((customer, best_scheme, best_cost))
                if customer in destroy_node:  # 安全检查，避免重复删除
                    destroy_node.remove(customer)
                
                # print(f"成功直接插入客户点 {customer}，成本: {best_cost:.2f}")    
            else:
                # 启发式交换模式
                customer = best_candidate['customer']
                best_orig_y = best_candidate['orig_scheme']
                best_new_y = best_candidate['new_scheme']
                best_orig_cost = best_candidate['orig_cost']
                best_new_cost = best_candidate['new_cost']
                best_orig_y_cijkdu_plan = best_candidate['orig_plan']
                best_new_y_cijkdu_plan = best_candidate['new_plan']
                
                # 解析交换方案
                orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = best_orig_y
                new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = best_new_y
                
                # 确定要移除的客户点
                if orig_customer == customer:
                    remove_customer = new_customer
                else:
                    remove_customer = orig_customer
                
                print(f"启发式交换：移除客户点 {remove_customer}，插入客户点 {customer}")
                
                # 删除被移除的方案
                if remove_customer in repaired_state.customer_plan:
                    y = repaired_state.customer_plan[remove_customer]
                    del repaired_state.customer_plan[remove_customer]
                    if repaired_state.uav_cost and remove_customer in repaired_state.uav_cost:
                        del repaired_state.uav_cost[remove_customer]
                    
                    # 从无人机分配中移除
                    drone_id_remove, _, _, _, _, _ = y
                    if drone_id_remove in repaired_state.uav_assignments:
                        repaired_state.uav_assignments[drone_id_remove] = [
                            task for task in repaired_state.uav_assignments[drone_id_remove]
                            if task[2] != remove_customer
                        ]
                    
                    vehicle_task_data = remove_vehicle_task(vehicle_task_data, y, vehicle_route)
                
                # 根据时间优先级选择插入方案
                if best_new_y_cijkdu_plan['launch_time'] < best_orig_y_cijkdu_plan['launch_time']:
                    # 插入新方案
                    repaired_state.customer_plan[new_customer] = best_new_y
                    if repaired_state.uav_cost is None:
                        repaired_state.uav_cost = {}
                    repaired_state.uav_cost[new_customer] = best_new_cost
                    
                    # 更新无人机分配
                    if new_drone_id not in repaired_state.uav_assignments:
                        repaired_state.uav_assignments[new_drone_id] = []
                    repaired_state.uav_assignments[new_drone_id].append(best_new_y)
                    
                    vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                    final_scheme = best_new_y
                    final_cost = best_new_cost
                else:
                    # 插入原方案
                    repaired_state.customer_plan[orig_customer] = best_orig_y
                    if repaired_state.uav_cost is None:
                        repaired_state.uav_cost = {}
                    repaired_state.uav_cost[orig_customer] = best_orig_cost
                    
                    # 更新无人机分配
                    if orig_drone_id not in repaired_state.uav_assignments:
                        repaired_state.uav_assignments[orig_drone_id] = []
                    repaired_state.uav_assignments[orig_drone_id].append(best_orig_y)
                    
                    vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)
                    final_scheme = best_orig_y
                    final_cost = best_orig_cost
                
                # 记录插入方案
                insert_plan.append((customer, final_scheme, final_cost))
                if customer in destroy_node:  # 安全检查，避免重复删除
                    destroy_node.remove(customer)
                
                print(f"成功启发式交换插入客户点 {customer}，最终成本: {final_cost:.2f}")
            
            # 更新空跑节点等状态
            # repaired_state.update_rm_empty_task()
            # repaired_state._total_cost = None  # 重置成本缓存
            
            # print(f"当前已插入 {len(insert_plan)} 个节点")
            # print("-" * 50)
        # 更新修复完成后的成本
        repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        # print(f"后悔修复完成：成功插入 {len(insert_plan)} 个客户点")
        return repaired_state, insert_plan

    def repair_k_insertion(self, state):
        """
        快速K步插入修复算子：使用采样和启发式方法提高性能
        策略：采样少量K步序列，选择最优的插入方案
        """
        repaired_state = state
        destroy_node = list(state.destroyed_customers_info.keys())  # 获取所有待插入的破坏节点
        insert_plan = []  # 记录所有破坏节点的最优插入方案
        
        print(f"快速K步修复：需要插入 {len(destroy_node)} 个客户点: {destroy_node}")
        
        # 平衡精度和速度的K步参数
        k_steps = 3  # 恢复到3步，保持精度
        max_samples = 15  # 增加采样数，提高精度
        candidate_limit = 6  # 限制候选节点数，控制复杂度
        
        while len(destroy_node) > 0:
            print(f"当前剩余待插入节点: {destroy_node}")
            
            # 获取当前状态的数据
            vehicle_route = repaired_state.vehicle_routes
            vehicle_task_data = repaired_state.vehicle_task_data
            vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
            
            # 如果剩余节点数少于等于3，直接使用贪婪策略
            if len(destroy_node) <= 3:
                print(f"剩余节点数({len(destroy_node)}) <= 3，使用贪婪策略")
                best_customer, best_scheme, best_cost = self._greedy_select_best_insertion(
                    destroy_node, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                )
                if best_customer is not None:
                    # 应用最佳插入方案
                    self._apply_insertion(repaired_state, best_customer, best_scheme, best_cost)
                    insert_plan.append((best_customer, best_scheme, best_cost))
                    destroy_node.remove(best_customer)
                    print(f"快速K步修复：成功插入客户点 {best_customer}，成本: {best_cost:.2f}")
                else:
                    print("快速K步修复：没有找到可行的插入方案")
                    break
            else:
                # 使用平衡精度和速度的K步策略
                best_customer, best_scheme, best_cost = self._balanced_k_step_selection(
                    destroy_node, k_steps, max_samples, candidate_limit, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                )
                
                if best_customer is not None:
                    # 应用最佳插入方案
                    self._apply_insertion(repaired_state, best_customer, best_scheme, best_cost)
                    insert_plan.append((best_customer, best_scheme, best_cost))
                    destroy_node.remove(best_customer)
                    print(f"快速K步修复：成功插入客户点 {best_customer}，成本: {best_cost:.2f}")
                else:
                    # 如果快速K步策略失败，回退到贪婪策略
                    print("快速K步策略失败，回退到贪婪策略")
                    best_customer, best_scheme, best_cost = self._greedy_select_best_insertion(
                        destroy_node, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                    )
                    if best_customer is not None:
                        self._apply_insertion(repaired_state, best_customer, best_scheme, best_cost)
                        insert_plan.append((best_customer, best_scheme, best_cost))
                        destroy_node.remove(best_customer)
                        print(f"快速K步修复：贪婪回退插入客户点 {best_customer}，成本: {best_cost:.2f}")
                    else:
                        print("快速K步修复：没有找到可行的插入方案")
                        break
        
        # 更新修复完成后的成本
        repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        print(f"快速K步修复完成：成功插入 {len(insert_plan)} 个客户点")
        return repaired_state, insert_plan
    
    def repair_vtp_insertion(self, state):
        """
        VTP节点插入修复算子：不仅考虑现有VTP节点，还会考虑插入全新的VTP节点到车辆路径中
        逻辑：
        1. 对于待修复的客户，从全局VTP集合中找出距离最近的K个VTP节点
        2. 考虑将这些VTP节点插入到车辆路径的各个位置
        3. 计算总成本（车辆行驶成本 + 无人机飞行成本），选择最优方案
        """
        repaired_state = state
        destroy_node = list(state.destroyed_customers_info.keys())
        insert_plan = []
        
        print(f"VTP插入修复：需要插入 {len(destroy_node)} 个客户点: {destroy_node}")
        
        # VTP插入参数
        k_vtp_candidates = 10  # 考虑距离最近的10个VTP节点
        
        while len(destroy_node) > 0:
            print(f"当前剩余待插入节点: {destroy_node}")
            
            # 获取当前状态的数据
            vehicle_route = repaired_state.vehicle_routes
            vehicle_task_data = repaired_state.vehicle_task_data
            vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
            
            best_customer = None
            best_scheme = None
            best_cost = float('inf')
            best_vtp_insertion = None  # 记录最优的VTP插入方案
            
            # 遍历所有待插入客户点
            for customer in destroy_node:
                # 获取距离该客户点最近的K个VTP节点
                candidate_vtps = self._get_nearest_vtp_candidates(customer, k_vtp_candidates, vehicle_route)
                
                # 评估每个候选VTP节点的插入方案
                for vtp_candidate in candidate_vtps:
                    # 计算将该VTP节点插入到各个车辆路径位置的成本
                    insertion_costs = self._evaluate_vtp_insertion_costs(
                        vtp_candidate, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                    )
                    
                    # 选择最优的插入方案
                    for (vehicle_id, insert_position, total_cost, scheme) in insertion_costs:
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_scheme = scheme
                            best_customer = customer
                            best_vtp_insertion = (vtp_candidate, vehicle_id, insert_position)
            
            if best_customer is not None and best_vtp_insertion is not None:
                # 应用最优的VTP插入方案
                vtp_node, vehicle_id, insert_position = best_vtp_insertion
                
                # 1. 将VTP节点插入到车辆路径中
                route = repaired_state.vehicle_routes[vehicle_id]
                route.insert(insert_position, vtp_node)
                repaired_state.vehicle_routes[vehicle_id] = route
                
                # 2. 应用客户点的插入方案
                self._apply_insertion(repaired_state, best_customer, best_scheme, best_cost)
                insert_plan.append((best_customer, best_scheme, best_cost))
                destroy_node.remove(best_customer)
                
                print(f"VTP插入修复：成功插入VTP节点 {vtp_node} 到车辆 {vehicle_id} 位置 {insert_position}")
                print(f"VTP插入修复：成功插入客户点 {best_customer}，总成本: {best_cost:.2f}")
            else:
                # 如果VTP插入策略失败，回退到传统贪婪策略
                print("VTP插入策略失败，回退到传统贪婪策略")
                best_customer, best_scheme, best_cost = self._greedy_select_best_insertion(
                    destroy_node, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state, base_cost
                )
                if best_customer is not None:
                    self._apply_insertion(repaired_state, best_customer, best_scheme, best_cost)
                    insert_plan.append((best_customer, best_scheme, best_cost))
                    destroy_node.remove(best_customer)
                    print(f"VTP插入修复：贪婪回退插入客户点 {best_customer}，成本: {best_cost:.2f}")
                else:
                    print("VTP插入修复：没有找到可行的插入方案")
                    break
        
        # 更新修复完成后的成本
        repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        print(f"VTP插入修复完成：成功插入 {len(insert_plan)} 个客户点")
        return repaired_state, insert_plan
    
    def _get_nearest_vtp_candidates(self, customer, k, vehicle_route):
        """
        获取距离客户点最近的K个VTP节点候选
        """
        # 获取客户点坐标
        customer_pos = np.array([
            self.node[customer].latDeg,
            self.node[customer].lonDeg,
            self.node[customer].altMeters
        ])
        
        # 获取所有VTP节点坐标
        vtp_candidates = []
        for vtp_id in self.A_vtp:
            # 检查该VTP节点是否已经在任何车辆路径中
            vtp_in_route = False
            for route in vehicle_route:
                if vtp_id in route:
                    vtp_in_route = True
                    break
            
            if not vtp_in_route:  # 只考虑未使用的VTP节点
                vtp_pos = np.array([
                    self.node[vtp_id].latDeg,
                    self.node[vtp_id].lonDeg,
                    self.node[vtp_id].altMeters
                ])
                distance = np.linalg.norm(vtp_pos - customer_pos)
                vtp_candidates.append((vtp_id, distance))
        
        # 按距离排序，选择最近的K个
        vtp_candidates.sort(key=lambda x: x[1])
        return [vtp_id for vtp_id, _ in vtp_candidates[:k]]
    
    def _evaluate_vtp_insertion_costs(self, vtp_node, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state):
        """
        评估将VTP节点插入到各个车辆路径位置的成本
        """
        insertion_costs = []
        
        # 遍历所有车辆
        for vehicle_id, route in vehicle_route.items():
            # 遍历该车辆路径的所有可能插入位置（除了起点和终点）
            for insert_position in range(1, len(route)):
                # 计算车辆行驶成本增量
                vehicle_cost_increase = self._calculate_vehicle_cost_increase(
                    vehicle_id, route, insert_position, vtp_node
                )
                
                # 计算无人机从该VTP节点服务客户点的成本
                uav_cost = self._calculate_uav_cost_from_vtp(vtp_node, customer)
                
                # 总成本
                total_cost = vehicle_cost_increase + uav_cost
                
                # 生成插入方案
                scheme = self._generate_vtp_insertion_scheme(
                    vtp_node, customer, vehicle_id, vehicle_route, vehicle_task_data
                )
                
                if scheme is not None:
                    insertion_costs.append((vehicle_id, insert_position, total_cost, scheme))
        
        return insertion_costs
    
    def _calculate_vehicle_cost_increase(self, vehicle_id, route, insert_position, vtp_node):
        """
        计算将VTP节点插入到指定位置后车辆行驶成本的增量
        """
        try:
            if insert_position <= 0 or insert_position >= len(route):
                return float('inf')
            
            # 原路径：route[insert_position-1] -> route[insert_position]
            # 新路径：route[insert_position-1] -> vtp_node -> route[insert_position]
            vehicle_id = vehicle_id + 1
            prev_node = route[insert_position - 1]
            next_node = route[insert_position]        
            # 检查距离矩阵是否存在
            # if (vehicle_id not in self.veh_distance or 
            #     prev_node not in self.veh_distance[vehicle_id] or
            #     next_node not in self.veh_distance[vehicle_id][prev_node] or
            #     vtp_node not in self.veh_distance[vehicle_id][prev_node] or
            #     next_node not in self.veh_distance[vehicle_id][vtp_node]):
            #     return float('inf')
            # 原距离
            original_distance = self.veh_distance[vehicle_id][prev_node][next_node]
            
            # 新距离
            new_distance = (self.veh_distance[vehicle_id][prev_node][vtp_node] + 
                           self.veh_distance[vehicle_id][vtp_node][next_node])
            
            # # 检查车辆对象是否存在
            # if vehicle_id not in self.vehicle:
            #     return float('inf')
            
            # 成本增量
            cost_increase = (new_distance - original_distance) * self.vehicle[vehicle_id].per_cost
            
            return cost_increase
            
        except Exception as e:
            return float('inf')
    
    def _calculate_uav_cost_from_vtp(self, vtp_node, customer):
        """
        计算无人机从VTP节点服务客户点的成本
        """
        # 这里简化计算，实际应该考虑所有无人机的成本
        min_cost = float('inf')
        
        for drone_id in self.V:
            # 计算从VTP节点到客户点的飞行成本
            vtp_map_index = self.node[vtp_node].map_key
            customer_map_index = self.node[customer].map_key
            
            # 这里需要找到合适的回收节点，简化处理
            # 实际应该考虑所有可能的回收节点
            cost = self.uav_travel[drone_id][vtp_map_index][customer].totalDistance * self.vehicle[drone_id].per_cost
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def _update_vehicle_task_data_for_vtp(self, repaired_state, vtp_node, vehicle_id, insert_position):
        """
        更新vehicle_task_data以反映新插入的VTP节点
        """
        # 获取车辆路径
        route = repaired_state.vehicle_routes[vehicle_id - 1]
        
        # 为新插入的VTP节点创建任务数据
        from task_data import VehicleTaskData
        
        # 初始化VTP节点的任务数据
        vtp_task_data = VehicleTaskData()
        vtp_task_data.drone_list = list(self.V)  # 所有无人机都可以在该VTP节点回收
        vtp_task_data.launch_drone_list = list(self.V)  # 所有无人机都可以在该VTP节点发射
        
        # 更新vehicle_task_data
        if vehicle_id not in repaired_state.vehicle_task_data:
            repaired_state.vehicle_task_data[vehicle_id] = {}
        
        repaired_state.vehicle_task_data[vehicle_id][vtp_node] = vtp_task_data
    
    def _generate_vtp_insertion_scheme(self, vtp_node, customer, vehicle_id, vehicle_route, vehicle_task_data):
        """
        生成VTP插入方案
        """
        # 简化实现：使用第一个可用的无人机，同车插入
        if not self.V:
            return None
        
        drone_id = self.V[0]  # 使用第一个无人机
        launch_node = vtp_node
        customer_node = customer
        recovery_node = vtp_node  # 同车插入，回收节点也是VTP节点
        launch_vehicle_id = vehicle_id
        recovery_vehicle_id = vehicle_id
        
        return (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
    
    def _greedy_select_best_insertion(self, destroy_node, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state, base_cost):
        """
        贪婪选择最佳插入方案（辅助函数）
        为每个客户点考虑新增VTP节点的方案，扩大解空间
        """
        best_customer = None
        best_scheme = None
        best_cost = float('inf')
        
        for customer in destroy_node:
            # 1. 首先尝试传统插入方案（使用现有节点）
            traditional_result = self._evaluate_traditional_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
            if traditional_result is not None:
                traditional_cost, traditional_scheme = traditional_result
                if traditional_cost < best_cost:
                    best_cost = traditional_cost
                    best_scheme = traditional_scheme
                    best_customer = customer
            
            # 2. 考虑新增VTP节点的方案
            vtp_cost, vtp_scheme = self._evaluate_vtp_expansion_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
            if vtp_scheme is not None:
                if vtp_cost-base_cost < best_cost:
                    best_cost = vtp_cost
                    best_scheme = vtp_scheme
                    best_customer = customer

        
        return best_customer, best_scheme, best_cost
    
    def _evaluate_traditional_insertion(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state=None):
        """
        评估传统插入方案的成本和方案（使用现有节点）
        包括直接插入和启发式插入两种模式
        返回: (cost, scheme) 或 None
        """
        # try:
        # 1. 首先尝试直接插入方案
        all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
        if all_insert_position is not None:
            best_scheme = None
            min_cost = float('inf')
            for drone_id, inert_positions in all_insert_position.items():
                for inert_position in inert_positions:
                    launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                    insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                    if insert_cost < min_cost:
                        min_cost = insert_cost
                        best_scheme = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
            
            if min_cost != float('inf'):
                return (min_cost, best_scheme)
        
        # 2. 如果直接插入失败，尝试启发式插入模式
        if repaired_state is not None:
            try:
                best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = DiverseRouteGenerator.greedy_insert_feasible_plan(
                    customer, vehicle_route, vehicle_arrive_time, vehicle_task_data, repaired_state.customer_plan
                )
                
                if best_orig_y is not None and best_new_y is not None:
                    # 计算总成本（移除成本 + 插入成本）
                    total_swap_cost = best_orig_cost + best_new_cost
                    heuristic_scheme = {
                                'customer': customer,
                                'orig_scheme': best_orig_y,
                                'new_scheme': best_new_y,
                                'orig_cost': best_orig_cost,
                                'new_cost': best_new_cost,
                                'total_cost': total_swap_cost,
                                'orig_plan': best_orig_y_cijkdu_plan,
                                'new_plan': best_new_y_cijkdu_plan,
                                'type': 'heuristic_swap'
                    }
                    return (total_swap_cost, heuristic_scheme)
            except Exception as e:
                print(f"客户点 {customer} 启发式插入失败: {e}")
        
        # 3. 如果两种方案都失败，返回None
        # print(f"客户点 {customer} 传统插入评估失败: {e}")
        return None,None
            
        # except Exception as e:
        #     print(f"客户点 {customer} 传统插入评估失败: {e}")
        #     return None
    
    def _evaluate_vtp_expansion_insertion(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state):
        """
        评估VTP扩展插入方案的成本和方案（为每个客户点考虑新增VTP节点）
        返回: (cost, scheme) 或 None
        """
        best_scheme = None
        best_vtp_infor = [None, None, None]
        min_cost = float('inf')
        vtp_task_data = deep_copy_vehicle_task_data(repaired_state.vehicle_task_data)
        
        # 遍历所有车辆，为每个客户点考虑在该车辆路径上新增VTP节点,该处的vehicle_id为idx，是索引
        for vehicle_id in range(len(vehicle_route)):

            route = vehicle_route[vehicle_id]
            if len(route) < 2:  # 路径至少需要起点和终点
                continue
            
            # 获取该车辆路径中不存在的节点（排除已有节点）
            available_nodes = self._get_available_nodes_for_vehicle(vehicle_id, route, repaired_state)
            
            # 为每个可用节点计算插入成本，并选择成本最低的3-5个位置
            candidate_positions = self._get_best_insertion_positions(
                customer, vehicle_id, route, available_nodes, vehicle_route, vehicle_task_data, repaired_state
            )
            # 测试每个候选插入位置,测试的全局无人机的成本情况
            for node, insert_pos in candidate_positions:
                result, scheme = self._calculate_vtp_expansion_cost(customer, vehicle_id, insert_pos, vehicle_route, vtp_task_data, repaired_state, node)
                if result is not None:
                    if result < min_cost:
                        min_cost = result
                        best_scheme = scheme
                        best_vtp_infor[0] = node
                        best_vtp_infor[1] = vehicle_id+1
                        best_vtp_infor[2] = insert_pos
        return (min_cost, best_scheme), best_vtp_infor if min_cost != float('inf') else None
    
    def _get_available_nodes_for_vehicle(self, vehicle_id, route, repaired_state):
        """
        获取该车辆路径中不存在的节点（排除已有节点）
        """
        available_nodes = []
        
        # 获取所有可能的节点（排除车辆路径中已有的节点）
        existing_nodes = set(route)
        
        # 当前的所有节点
        all_nodes = repaired_state.N
        # 遍历所有节点，排除已有节点和客户点
        for node_id in all_nodes:
            if node_id not in existing_nodes:
                available_nodes.append(node_id)
        
        return available_nodes
    
    def _get_best_insertion_positions(self, customer, vehicle_id, route, available_nodes, vehicle_route, vehicle_task_data, repaired_state):
        """
        为每个可用节点计算插入成本，并选择成本最低的3-5个位置
        """
        position_costs = []
        
        # 为每个可用节点计算所有可能的插入位置的成本
        for node in available_nodes:
            for insert_pos in range(1, len(route)):  # 不在起点和终点插入
                cost = self._calculate_insertion_cost_estimate(customer, vehicle_id, insert_pos, route, node, vehicle_route, vehicle_task_data, repaired_state)
                if cost is not None:
                    position_costs.append((node, insert_pos, cost))
        
        # 按成本排序，选择最低的3-5个位置
        position_costs.sort(key=lambda x: x[2])
        max_candidates = min(25, len(position_costs))
        
        return [(node, pos) for node, pos, _ in position_costs[:max_candidates]]
    
    def _calculate_insertion_cost_estimate(self, customer, vehicle_id, insert_pos, route, node, vehicle_route, vehicle_task_data, repaired_state):
        """
        快速估算插入成本（用于预筛选）
        """
        try:
            # 1. 计算车辆路径成本增量（简化版）
            vehicle_cost_increase = self._calculate_vehicle_cost_increase(vehicle_id, route, insert_pos, node)
            
            # 2. 计算无人机执行任务成本（遍历所有潜在无人机）
            uav_cost = float('inf')
            
            drone_id = self.V[0]
            # for drone_id in self.V:
                # 计算从VTP节点到客户点的无人机成本
            uav_cost = self._calculate_uav_mission_cost_estimate(drone_id, customer, node)
            
            # 3. 总成本估算
            total_cost = vehicle_cost_increase + uav_cost
            
            return total_cost
            
        except Exception as e:
            return None
    
    def _calculate_uav_mission_cost_estimate(self, drone_id, customer, vtp_node):
        """
        快速估算无人机执行任务的成本
        """
        try:
            # 使用欧几里得距离快速估算
            # vtp_x, vtp_y = self._get_node_coordinates(vtp_node)
            # customer_x, customer_y = self._get_node_coordinates(customer)
            
            # distance = ((vtp_x - customer_x) ** 2 + (vtp_y - customer_y) ** 2) ** 0.5
            map_vtp_node = self.node[vtp_node].map_key
            # map_customer = self.node[customer].map_key
            distance = self.uav_travel[drone_id][map_vtp_node][customer].totalDistance * 0.5
            cost = distance * self.vehicle[drone_id].per_cost
            
            return cost
        except:
            return None
    
    def _is_customer_node(self, node_id):
        """检查是否为客户节点"""
        return hasattr(self.node[node_id], 'customer') and self.node[node_id].customer
    
    def _is_vehicle_node(self, node_id):
        """检查是否为车辆节点"""
        return node_id in self.vehicle
    
    def _calculate_vtp_expansion_cost(self, customer, vehicle_id, insert_pos, vehicle_route, vtp_vehicle_task_data, repaired_state, vtp_node=None):
        """
        计算VTP扩展插入的成本并返回最优方案
        包括：车辆路径成本增量 + 无人机执行任务成本 + 融合降落成本
        返回：(total_cost, best_scheme) 或 None
        """
        try:
            vehicle_idx = vehicle_id
            vehicle_id = vehicle_id + 1
            all_route = [sub_route.copy() for sub_route in vehicle_route]  # 避免指向同一对象
            route = all_route[vehicle_idx]
            in_route = vehicle_route[vehicle_idx]
            # 将vtp节点插入车辆路径中，同时避免指向同一对象
            route.insert(insert_pos, vtp_node)
            all_route[vehicle_idx] = route
            # 查找插如的前一个节点,可继承对应的状态
            prev_node = route[insert_pos - 1]
            if prev_node == self.DEPOT_nodeID or insert_pos == 1:
                drone_list = self.base_drone_assignment[vehicle_id][:]
            else:
                drone_list = vtp_vehicle_task_data[vehicle_id][prev_node].drone_list[:]
            launch_drone_list = vtp_vehicle_task_data[vehicle_id][prev_node].launch_drone_list[:]
            recovery_drone_list = vtp_vehicle_task_data[vehicle_id][prev_node].recovery_drone_list[:]
            vtp_vehicle_task_data[vehicle_id][vtp_node].drone_list = drone_list
            vtp_vehicle_task_data[vehicle_id][vtp_node].launch_drone_list = []
            vtp_vehicle_task_data[vehicle_id][vtp_node].recovery_drone_list = []

            # 1. 计算车辆路径成本增量
            vehicle_cost_increase = self._calculate_vehicle_cost_increase(vehicle_id, in_route, insert_pos, vtp_node)
            
            # 2. 计算无人机执行任务成本（遍历所有潜在无人机作为发射点或回收点）
            min_uav_cost = float('inf')
            best_drone_scheme = None
            best_scheme = None

            # 处理从新增vtp节点作为发射和回收的逻辑关系
            for drone_id in self.V:
                # 测试无人机作为发射点的成本
                if drone_id not in drone_list: # 不在drone_list中，则不测试
                    continue
                launch_cost,scheme = self._calculate_uav_mission_cost(drone_id, customer, vehicle_id, insert_pos, all_route, vtp_node, vtp_vehicle_task_data, repaired_state,'launch')
                if launch_cost is not None and launch_cost < min_uav_cost:
                    min_uav_cost = launch_cost
                    best_drone_scheme = (drone_id, customer, vehicle_id, insert_pos, 'launch')
                    best_scheme = scheme
            # 测试无人机作为回收点的成本
            for drone_id in self.V:
                recovery_cost,scheme = self._calculate_uav_mission_cost(drone_id, customer, vehicle_id, insert_pos, all_route, vtp_node, vtp_vehicle_task_data, repaired_state,'recovery')
                if recovery_cost is not None and recovery_cost < min_uav_cost:
                    min_uav_cost = recovery_cost
                    best_drone_scheme = (drone_id, customer, vehicle_id, insert_pos, 'recovery')
                    best_scheme = scheme
            if min_uav_cost == float('inf'):
                return float('inf'), None
        # # 3. 计算融合降落成本
        # landing_cost = self._calculate_landing_cost(customer, vehicle_id, insert_pos, route, best_drone_scheme)
        
            # 4. 总成本
            total_cost = vehicle_cost_increase + min_uav_cost
        
        # # 5. 生成最优方案
        # if vtp_node is None:
        #     vtp_node = f"vtp_{vehicle_id}_{insert_pos}_{customer}"
        
        # # 根据最优无人机方案生成完整的插入方案
        # drone_id, _, _, _, mission_type = best_drone_scheme
        # if mission_type == 'launch':
        #     # 无人机作为发射点：从VTP节点到客户点
        #     best_scheme = (drone_id, vtp_node, customer, vtp_node, vehicle_id, vehicle_id)
        # else:  # recovery
        #     # 无人机作为回收点：从客户点到VTP节点
        #     best_scheme = (drone_id, vtp_node, customer, vtp_node, vehicle_id, vehicle_id)
        
            return total_cost, best_scheme
            
        except Exception as e:
            return float('inf'), None
    
    def _calculate_uav_mission_cost(self, drone_id, customer, vehicle_id, insert_pos, route, vtp_node, vtp_vehicle_task_data, repaired_state, mission_type='launch'):
        """
        计算无人机执行任务的成本
        支持无人机作为发射点或回收点的不同成本计算
        """
        # try:
        # 使用传入的VTP节点或生成新的节点ID
        if vtp_node is None:
            vtp_node = f"vtp_{vehicle_id}_{insert_pos}_{customer}"  # 生成唯一的VTP节点ID
        repaired_state.add_vehicle_route = route
        vtp_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.add_vehicle_route)
        # 根据任务类型计算不同的成本
        if mission_type == 'launch':
            # 无人机作为发射点：从VTP节点到客户点
            cost,scheme = self._calculate_launch_mission_cost(drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id)
        elif mission_type == 'recovery':
            # 无人机作为回收点：从客户点到VTP节点
            cost,scheme = self._calculate_recovery_mission_cost(drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id)
        else:
            return None
        
        return cost,scheme
            
        # except:
        #     return None
    
    def _calculate_launch_mission_cost(self, drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id):
        """
        计算无人机作为发射点的成本（从VTP节点到客户点）
        参考get_all_insert_position函数的规则，考虑同车和跨车两种情况
        """
        # try:
        # 获取该客户点的最近VTP节点集合
        customer_vtp_candidates = self.map_cluster_vtp_dict[customer]
        vehicle_idx = vehicle_id -1
        # 找到vtp_node在route中的索引
        route = route[vehicle_idx]
        vtp_node_idx = route.index(vtp_node)
        launch_vehicle_id = vehicle_id
        
        min_cost = float('inf')
        best_scheme = None
        
        # 同车情况：找到下一次无人机发射任务，确定解空间范围
        n = len(route)
        next_launch_idx = n - 1  # 默认到最后一个节点之前

        if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][vtp_node].drone_list:
            return None, None
        
        # 查找下一次无人机发射任务
        for k in range(vtp_node_idx + 1, n - 1):
            if drone_id in vtp_vehicle_task_data[launch_vehicle_id][route[k]].launch_drone_list:
                next_launch_idx = k
                break
        
        # 遍历从vtp_node到下一次发射节点之间的所有节点作为回收点
        for k in range(vtp_node_idx + 1, next_launch_idx+1):
            recovery_node = route[k]
            if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][recovery_node].drone_list or \
                drone_id not in vtp_vehicle_task_data[launch_vehicle_id][recovery_node].launch_drone_list:
                continue
            # 检查从发射点到回收点之间，中间每一个节点都要有drone_list
            has_conflict = False
            for m in range(vtp_node_idx + 1, k):
                if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][route[m]].drone_list:
                    has_conflict = True
                    break
            if has_conflict:
                continue
            # # 检查回收节点是否支持该无人机
            # if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][recovery_node].drone_list:
            #     continue
            # 不允许同点发射及降落
            if vtp_node == recovery_node:
                continue
            # 计算从VTP节点到客户点，再从客户点到回收节点的成本
            cost = self.drone_insert_cost(drone_id, customer, vtp_node, recovery_node)
            if cost is not None and cost < min_cost:
                min_cost = cost
                best_scheme = (drone_id, vtp_node, customer, recovery_node, launch_vehicle_id, launch_vehicle_id)

        # 跨车情况：检查其他车辆的所有可能回收点
        for recovery_vehicle_idx, other_route in enumerate(repaired_state.vehicle_routes):
            recovery_vehicle_id = recovery_vehicle_idx + 1
            if recovery_vehicle_id == launch_vehicle_id:
                continue
            
            launch_time = vtp_vehicle_arrive_time[launch_vehicle_id][vtp_node]
            
            for recovery_node in other_route[1:-1]:
                
                # 排除发射点和回收点完全相同的情况
                if vtp_node == recovery_node:
                    continue
                
                recovery_time = vtp_vehicle_arrive_time[recovery_vehicle_id][recovery_node]
                if recovery_time <= launch_time:
                    continue
                
                if drone_id in vtp_vehicle_task_data[recovery_vehicle_id][recovery_node].launch_drone_list or \
                    drone_id in vtp_vehicle_task_data[recovery_vehicle_id][recovery_node].recovery_drone_list:
                    continue

                # 检查发射车辆路线中的冲突
                conflict = False
                for m in range(vtp_node_idx + 1, len(route)):
                    if drone_id in vtp_vehicle_task_data[launch_vehicle_id][route[m]].launch_drone_list or \
                        drone_id not in vtp_vehicle_task_data[launch_vehicle_id][route[m]].drone_list:
                        conflict = True
                        break
                
                if not conflict:
                    # 计算跨车成本
                    cost = self.drone_insert_cost(drone_id, customer, vtp_node, recovery_node)
                    if cost is not None and cost < min_cost:
                        min_cost = cost
                        best_scheme = (drone_id, vtp_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
        
        return min_cost if min_cost != float('inf') else None, best_scheme
        
        # except Exception as e:
        #     print(f"Error in _calculate_launch_mission_cost: {e}")
        #     return None
    
    def _calculate_recovery_mission_cost(self, drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id):
        """
        计算无人机作为回收点的成本（从客户点到VTP节点）
        遍历所有车辆路线作为发射点，判断是否能将VTP作为回收点
        """
        # try:
        # 获取该客户点的最近VTP节点集合
        # customer_vtp_candidates = self.map_cluster_vtp_dict[customer]
        vehicle_idx = vehicle_id - 1
        # 找到vtp_node在route中的索引
        route = route[vehicle_idx]
        vtp_node_idx = route.index(vtp_node)
        recovery_vehicle_id = vehicle_id
        
        min_cost = float('inf')
        best_scheme = None
        
        # 遍历所有车辆路线作为发射点
        for launch_vehicle_idx, launch_route in enumerate(repaired_state.vehicle_routes):
            launch_vehicle_id = launch_vehicle_idx + 1
            
            # 同车情况：查找VTP节点向前索引最近回收该无人机的回收点
            if launch_vehicle_id == recovery_vehicle_id:
                # if drone_id in vtp_vehicle_task_data[recovery_vehicle_id][vtp_node].drone_list: # 新增关键约束
                # 查找VTP节点之前的最近回收点
                has_conflict = False
                for index,node in enumerate(launch_route[1:-1]):
                    if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][node].drone_list:
                        has_conflict = True
                        break
                if has_conflict:
                    continue

                nearest_recovery_idx = -1
                for k in range(vtp_node_idx - 1, 0, -1):  # 从vtp_node向前查找
                    if drone_id in vtp_vehicle_task_data[launch_vehicle_id][route[k]].recovery_drone_list:
                        nearest_recovery_idx = k
                        break
                
                if nearest_recovery_idx == -1:
                    nearest_recovery_idx = 1 # 代表前方没任务
                    # 找到从开始到索引点，无人机id是否在list同时没有被发射，找到关联的所有节点.
                    for k in range(nearest_recovery_idx, vtp_node_idx):
                        has_conflict = False
                        for m in range(k, vtp_node_idx+1):
                            if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][route[m]].drone_list:
                                has_conflict = True
                                break
                        if has_conflict:
                            continue
                        launch_node = route[k]
                        if launch_node == vtp_node:
                            continue
                        if drone_id in vtp_vehicle_task_data[launch_vehicle_id][launch_node].drone_list:
                             # 计算从发射点到客户点，再从客户点到VTP节点的成本
                            cost = self.drone_insert_cost(drone_id, customer, launch_node, vtp_node)
                            if cost is not None and cost < min_cost:
                                min_cost = cost
                                best_scheme = (drone_id, launch_node, customer, vtp_node, launch_vehicle_id, recovery_vehicle_id)  
                else:
                    # 遍历从最近回收点到VTP节点的所有节点作为发射点
                    for k in range(nearest_recovery_idx, vtp_node_idx + 1):
                        has_conflict = False
                        for m in range(k, vtp_node_idx):
                            if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][route[m]].drone_list:
                                has_conflict = True
                                break
                        if has_conflict:
                            continue
                        launch_node = route[k]
                        # 不允许同点发射及降落
                        if launch_node == vtp_node:
                            continue
                        
                        # 计算从发射点到客户点，再从客户点到VTP节点的成本
                        cost = self.drone_insert_cost(drone_id, customer, launch_node, vtp_node)
                        if cost is not None and cost < min_cost:
                            min_cost = cost
                            best_scheme = (drone_id, launch_node, customer, vtp_node, launch_vehicle_id, recovery_vehicle_id)
                
            # 跨车情况：检查时间约束和冲突
            else:
                # 遍历发射车辆的所有节点作为发射点
                for launch_node in launch_route[1:-1]:
                    # 排除发射点和回收点完全相同的情况
                    launch_node_idx = launch_route.index(launch_node)
                    if launch_node == vtp_node:
                        continue
                    if vtp_vehicle_arrive_time[launch_vehicle_id][launch_node] >= vtp_vehicle_arrive_time[recovery_vehicle_id][vtp_node]:
                        continue
                    # if drone_id in vtp_vehicle_task_data[launch_vehicle_id][launch_node].launch_drone_list:
                    #     continue
                    if drone_id not in vtp_vehicle_task_data[launch_vehicle_id][launch_node].drone_list:  # 新增关键约束
                        continue
                    # 检查该节点后的路线是否有该无人机的发射任务
                    has_conflict = False
                    for m in range(launch_node_idx + 1, len(launch_route) - 1):
                        if drone_id in vtp_vehicle_task_data[launch_vehicle_id][launch_route[m]].launch_drone_list or \
                            drone_id not in vtp_vehicle_task_data[launch_vehicle_id][launch_route[m]].drone_list:
                            has_conflict = True
                            break
                    if has_conflict:
                        continue
                    # 计算跨车成本
                    cost = self.drone_insert_cost(drone_id, customer, launch_node, vtp_node)
                    if cost is not None and cost < min_cost:
                        min_cost = cost
                        best_scheme = (drone_id, launch_node, customer, vtp_node, launch_vehicle_id, recovery_vehicle_id)
        
        return min_cost if min_cost != float('inf') else None, best_scheme
           
        # except Exception as e:
        #     print(f"Error in _calculate_recovery_mission_cost: {e}")
        #     return None, None
    
    def _estimate_uav_cost(self, drone_id, vtp_node, customer):
        """
        估算无人机成本（当无法获取精确数据时）
        """
        try:
            # 使用欧几里得距离估算
            vtp_x, vtp_y = self._get_node_coordinates(vtp_node)
            customer_x, customer_y = self._get_node_coordinates(customer)
            
            distance = ((vtp_x - customer_x) ** 2 + (vtp_y - customer_y) ** 2) ** 0.5
            cost = distance * self.vehicle[drone_id].per_cost
            
            return cost
        except:
            return float('inf')
    
    def _get_node_coordinates(self, node_id):
        """
        获取节点坐标
        """
        try:
            if hasattr(self.node[node_id], 'x') and hasattr(self.node[node_id], 'y'):
                return self.node[node_id].x, self.node[node_id].y
            else:
                # 如果节点没有坐标信息，返回默认值
                return 0, 0
        except:
            return 0, 0
    
    def _calculate_landing_cost(self, customer, vehicle_id, insert_pos, route, drone_scheme):
        """
        计算融合降落成本
        """
        try:
            # 简化实现：降落成本通常包括时间成本和操作成本
            # 这里可以根据具体需求调整
            base_landing_cost = 10.0  # 基础降落成本
            
            # 根据客户点位置和车辆路径调整成本
            route_length_factor = len(route) / 10.0  # 路径长度因子
            customer_priority = 1.0  # 客户优先级因子
            
            total_landing_cost = base_landing_cost * route_length_factor * customer_priority
            
            return total_landing_cost
            
        except:
            return 0.0
    
    
    def _find_vtp_insert_position(self, route, vtp_node, customer):
        """
        找到VTP节点在路径中的插入位置
        """
        try:
            # 从VTP节点名称中提取信息
            # vtp_node格式: "vtp_{vehicle_id}_{insert_pos}_{customer}"
            parts = vtp_node.split('_')
            if len(parts) >= 3:
                insert_pos = int(parts[2])  # 获取插入位置
                return insert_pos
            
            # 如果无法从名称中提取，使用启发式方法
            # 在路径中间位置插入
            return len(route) // 2 if len(route) > 1 else 1
            
        except:
            # 默认在路径中间插入
            return len(route) // 2 if len(route) > 1 else 1
    
    def _balanced_k_step_selection(self, destroy_node, k_steps, max_samples, candidate_limit, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state):
        """
        平衡精度和速度的K步选择：结合精确计算和启发式估计
        """
        import random
        
        # 智能候选节点筛选
        if len(destroy_node) > candidate_limit:
            # 评估每个节点的单步插入成本
            node_costs = []
            for customer in destroy_node:
                all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
                if all_insert_position is not None:
                    min_cost = float('inf')
                    for drone_id, inert_positions in all_insert_position.items():
                        for inert_position in inert_positions:
                            launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                            insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                            min_cost = min(min_cost, insert_cost)
                    if min_cost != float('inf'):
                        node_costs.append((customer, min_cost))
            
            # 选择成本最低的前candidate_limit个节点
            node_costs.sort(key=lambda x: x[1])
            candidate_nodes = [customer for customer, _ in node_costs[:candidate_limit]]
        else:
            candidate_nodes = destroy_node
        
        # 生成候选序列：结合贪心和随机策略
        candidate_sequences = []
        
        # 1. 贪心序列：按单步成本排序
        greedy_sequence = candidate_nodes[:k_steps] if len(candidate_nodes) >= k_steps else candidate_nodes
        candidate_sequences.append(greedy_sequence)
        
        # 2. 随机采样序列
        sample_size = min(max_samples - 1, len(candidate_nodes))
        for _ in range(sample_size):
            if len(candidate_nodes) >= k_steps:
                sequence = random.sample(candidate_nodes, k_steps)
            else:
                sequence = candidate_nodes
            candidate_sequences.append(sequence)
        
        # 评估每个候选序列
        best_customer = None
        best_scheme = None
        best_cost = float('inf')
        
        for sequence in candidate_sequences:
            # 精确计算K步序列的总成本
            sequence_cost = self._evaluate_k_step_sequence_cost(
                sequence, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
            )
            
            if sequence_cost < best_cost:
                best_cost = sequence_cost
                # 获取第一个客户点的最佳插入方案
                first_customer = sequence[0]
                best_scheme = self._get_best_insertion_scheme(
                    first_customer, vehicle_route, vehicle_task_data, vehicle_arrive_time
                )
                best_customer = first_customer
        
        return best_customer, best_scheme, best_cost
    
    def _evaluate_k_step_sequence_cost(self, sequence, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state):
        """
        精确评估K步序列的总成本（简化版，只计算前2步的精确成本）
        """
        if len(sequence) == 1:
            # 单步情况，直接计算
            customer = sequence[0]
            all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
            if all_insert_position is not None:
                min_cost = float('inf')
                for drone_id, inert_positions in all_insert_position.items():
                    for inert_position in inert_positions:
                        launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                        insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        min_cost = min(min_cost, insert_cost)
                return min_cost if min_cost != float('inf') else float('inf')
        
        # 多步情况：精确计算前2步，启发式估计后续步骤
        total_cost = 0
        temp_state = repaired_state.fast_copy()
        
        # 精确计算前2步
        for i, customer in enumerate(sequence[:2]):
            all_insert_position = self.get_all_insert_position(
                temp_state.vehicle_routes, temp_state.vehicle_task_data, customer, vehicle_arrive_time
            )
            
            if all_insert_position is not None:
                min_cost = float('inf')
                best_scheme = None
                
                for drone_id, inert_positions in all_insert_position.items():
                    for inert_position in inert_positions:
                        launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                        insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        if insert_cost < min_cost:
                            min_cost = insert_cost
                            best_scheme = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                
                if best_scheme is not None:
                    total_cost += min_cost
                    # 模拟插入，更新临时状态
                    self._simulate_insertion_simple(temp_state, customer, best_scheme)
                else:
                    return float('inf')
            else:
                return float('inf')
        
        # 启发式估计后续步骤
        if len(sequence) > 2:
            remaining_customers = sequence[2:]
            avg_cost_estimate = self._estimate_average_cost(remaining_customers, vehicle_route, vehicle_task_data, vehicle_arrive_time)
            total_cost += avg_cost_estimate * len(remaining_customers) * 0.9  # 0.9是折扣因子
        
        return total_cost
    
    def _get_best_insertion_scheme(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time):
        """
        获取客户点的最佳插入方案
        """
        all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
        if all_insert_position is not None:
            min_cost = float('inf')
            best_scheme = None
            
            for drone_id, inert_positions in all_insert_position.items():
                for inert_position in inert_positions:
                    launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                    insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                    if insert_cost < min_cost:
                        min_cost = insert_cost
                        best_scheme = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
            
            return best_scheme
        return None
    
    def _simulate_insertion_simple(self, temp_state, customer, scheme):
        """
        简化的模拟插入操作（只更新关键数据结构）
        """
        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = scheme
        
        # 更新customer_plan
        temp_state.customer_plan[customer_node] = scheme
        
        # 更新uav_assignments
        if drone_id not in temp_state.uav_assignments:
            temp_state.uav_assignments[drone_id] = []
        temp_state.uav_assignments[drone_id].append(scheme)
        
        # 更新uav_cost
        if temp_state.uav_cost is None:
            temp_state.uav_cost = {}
        temp_state.uav_cost[customer_node] = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
        
        # 简化更新vehicle_task_data（只更新关键信息）
        # 这里可以进一步优化，只更新必要的字段
        temp_state.vehicle_task_data = update_vehicle_task(
            temp_state.vehicle_task_data, scheme, temp_state.vehicle_routes
        )
    
    def _estimate_average_cost(self, customers, vehicle_route, vehicle_task_data, vehicle_arrive_time):
        """
        快速估计剩余客户点的平均插入成本
        """
        if not customers:
            return 0
        
        total_cost = 0
        valid_customers = 0
        
        # 只评估前3个客户点来估计平均成本
        sample_customers = customers[:3]
        
        for customer in sample_customers:
            all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
            if all_insert_position is not None:
                min_cost = float('inf')
                for drone_id, inert_positions in all_insert_position.items():
                    for inert_position in inert_positions:
                        launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                        insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        min_cost = min(min_cost, insert_cost)
                if min_cost != float('inf'):
                    total_cost += min_cost
                    valid_customers += 1
        
        return total_cost / max(valid_customers, 1)
    
    
    def _apply_insertion(self, repaired_state, customer, scheme, cost):
        """
        实际应用插入操作
        """
        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = scheme
        
        # 更新customer_plan
        repaired_state.customer_plan[customer_node] = scheme
        
        # 更新uav_assignments
        if drone_id not in repaired_state.uav_assignments:
            repaired_state.uav_assignments[drone_id] = []
        repaired_state.uav_assignments[drone_id].append(scheme)
        
        # 更新uav_cost
        if repaired_state.uav_cost is None:
            repaired_state.uav_cost = {}
        repaired_state.uav_cost[customer_node] = cost
        
        # 更新vehicle_task_data
        repaired_state.vehicle_task_data = update_vehicle_task(
            repaired_state.vehicle_task_data, scheme, repaired_state.vehicle_routes
        )

    def get_all_insert_position(self, vehicle_route, vehicle_task_data, customer, vehicle_arrive_time):
        """
        获取所有可行的插入位置，通过cluster_vtp_dict限制解空间以提高效率
        
        Args:
            vehicle_route: 车辆路线
            vehicle_task_data: 车辆任务数据
            customer: 客户点ID
            vehicle_arrive_time: 车辆到达时间
            
        Returns:
            dict: {drone_id: [(launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id), ...]}
        """
        all_insert_position = {drone_id: [] for drone_id in self.V}
        
        # 获取该客户点的最近VTP节点集合
        customer_vtp_candidates = self.map_cluster_vtp_dict[customer]
        # print(f"客户点 {customer} 的VTP候选节点: {customer_vtp_candidates[:5]}...")  # 只显示前5个

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
                    
                    # 检查发射节点是否在客户点的VTP候选集合中（放宽限制）
                    if launch_node not in customer_vtp_candidates:
                        # 如果不在候选集合中，仍然允许，但降低优先级
                        pass
                    
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
                    
                    # 同车插入：寻找所有可能的回收节点
                    for k in range(i + 1, n - 1):
                        recovery_node = route[k]
                        
                        # 检查回收节点是否支持该无人机
                        if drone_id not in vehicle_task_data[launch_vehicle_id][recovery_node].drone_list:
                            continue
                        
                        # 检查发射节点和回收节点之间是否存在冲突
                        # 规则：回收节点前(不含回收节点)，发射节点后不能存在该无人机的发射任务
                        launch_idx = i
                        recovery_idx = k
                        
                        # 检查发射节点之后到回收节点之前是否有该无人机的发射任务
                        has_conflict = False
                        for m in range(launch_idx + 1, recovery_idx):
                            if drone_id in vehicle_task_data[launch_vehicle_id][route[m]].launch_drone_list:
                                has_conflict = True
                                break
                        
                        if has_conflict:
                            # print(f"[DEBUG] 同车插入跳过：无人机 {drone_id} 从节点 {launch_node} 到节点 {recovery_node} 之间存在发射任务冲突")
                            continue
                        
                        # 检查回收节点是否在客户点的VTP候选集合中（放宽限制）
                        # 无论是否在候选集合中，都允许插入，但可以标记优先级
                        all_insert_position[drone_id].append(
                            (launch_node, customer, recovery_node, launch_vehicle_id, launch_vehicle_id)
                        )
                    i = j
                
                # 跨车查找：检查发射节点是否在VTP候选集合中
                for i in range(1, n - 1):
                    launch_node = route[i]
                    if drone_id not in vehicle_task_data[launch_vehicle_id][launch_node].drone_list:
                        continue
                    
                    # 检查发射节点是否在客户点的VTP候选集合中（放宽限制）
                    if launch_node not in customer_vtp_candidates:
                        # 如果不在候选集合中，仍然允许，但降低优先级
                        pass
                    
                    launch_time = vehicle_arrive_time[launch_vehicle_id][launch_node]
                    for recovery_vehicle_idx, other_route in enumerate(vehicle_route):
                        recovery_vehicle_id = recovery_vehicle_idx + 1
                        if recovery_vehicle_id == launch_vehicle_id:
                            continue
                        for recovery_node in other_route[1:-1]:
                            if drone_id not in vehicle_task_data[recovery_vehicle_id][recovery_node].drone_list:
                                continue
                            
                            # 检查回收节点是否在客户点的VTP候选集合中（放宽限制）
                            if recovery_node not in customer_vtp_candidates:
                                # 如果不在候选集合中，仍然允许，但降低优先级
                                pass
                            
                            # 新增：排除发射点和回收点完全相同的情况
                            # if launch_vehicle_id == recovery_vehicle_id and launch_node == recovery_node:
                            if launch_vehicle_id == recovery_vehicle_id:
                                continue  # 跨车时也不允许同节点
                            if launch_node == recovery_node:
                                continue  # 跨车时也不允许同节点
                            recovery_time = vehicle_arrive_time[recovery_vehicle_id][recovery_node]
                            if recovery_time <= launch_time:
                                continue
                            idx = other_route.index(recovery_node)
                            conflict = False
                            
                            # 检查回收车辆路线中的冲突（放宽限制）
                            # for m in range(1, idx):
                            #     if drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].launch_drone_list:
                            #         # 只检查发射冲突，允许回收冲突
                            #         conflict = True
                            #         break
                            # for m in range(idx + 1, len(other_route) - 1):
                            #     if drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].launch_drone_list:
                            #         conflict = True
                            #         break
                            
                            # 检查发射车辆路线中的冲突（放宽限制）
                            launch_idx = route.index(launch_node)
                            for m in range(launch_idx + 1, len(route) - 1):
                                if drone_id in vehicle_task_data[launch_vehicle_id][route[m]].launch_drone_list:
                                    # 只检查发射冲突，允许回收冲突
                                    conflict = True
                                    # print(f"[DEBUG] 跨车插入冲突：无人机 {drone_id} 从车辆 {launch_vehicle_id} 节点 {launch_node} 发射到车辆 {recovery_vehicle_id} 节点 {recovery_node}，但车辆 {launch_vehicle_id} 的节点 {route[m]} 还有该无人机的发射任务")
                                    break
                            
                            if not conflict:
                                all_insert_position[drone_id].append(
                                    (launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                                )
        
        # 统计每个无人机的可行插入位置数量
        total_positions = 0
        for drone_id in self.V:
            positions_count = len(all_insert_position[drone_id])
            total_positions += positions_count
            # if positions_count > 0:
                # print(f"无人机 {drone_id} 有 {positions_count} 个可行插入位置")
        
        # print(f"客户点 {customer} 总共有 {total_positions} 个可行插入位置")
        
        # 如果插入位置太少，输出警告
        if total_positions < 5:
            print(f"警告：客户点 {customer} 的可行插入位置过少 ({total_positions} 个)，可能影响优化效果")
        return all_insert_position
    # 计算不同发射回收点的成本状况
    def calculate_multiopt_cost(self, repair_state, best_scheme):
        """
        计算当前版本的总无人机成本消耗,计算设计发射和回收点的所有无人机的成本价格
        """
        drone_id, launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
        total_cost = 0
        for drone_id in self.V:
            total_cost += self.drone_insert_cost(drone_id, repair_state.vehicle_routes, repair_state.vehicle_task_data, customer, launch_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
        return total_cost

    def multiopt_update_best_scheme(self, best_scheme, near_node_list, vehicle_route, vehicle_task_data, repair_state, sample_size=30):
            """
            加速多opt邻域搜索：对near_node_list随机采样sample_size个发射-回收节点组合，
            只计算本无人机和同节点相关无人机的成本，贪婪选择最优。同时需要进一步考虑更换后的起始节点对其他无人机任务的影响状况及成本影响
            返回(最优方案, 最优总成本)。
            """
            # 计算当前版本的总无人机成本消耗,计算设计发射和回收点的所有无人机的成本价格
            init_multiopt_cost = self.calculate_multiopt_cost(repair_state, best_scheme)

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
        # 1. 算子池 (现在由__init__中的self.destroy_operators和self.repair_operators管理)
        #    我们不再需要在这里定义临时的算子列表和权重列表。
        
        # 2. 初始化解和日志
        y_best = []
        y_cost = []
        current_state = initial_state.fast_copy()
        
        # (你对初始解的预处理，这部分完全保留)
        # current_state.rm_empty_vehicle_route, current_state.empty_nodes_by_vehicle = current_state.update_rm_empty_task()
        current_state.rm_empty_vehicle_route = [route[:] for route in current_state.vehicle_routes]
        current_state.destroyed_node_cost = current_state.update_calculate_plan_cost(current_state.uav_cost, current_state.rm_empty_vehicle_route)
        current_state.rm_empty_vehicle_arrive_time = current_state.calculate_rm_empty_vehicle_arrive_time(current_state.rm_empty_vehicle_route)
        current_state.final_uav_plan, current_state.final_uav_cost, current_state.final_vehicle_plan_time, current_state.final_vehicle_task_data, current_state.final_global_reservation_table = current_state.re_update_time(current_state.rm_empty_vehicle_route, current_state.rm_empty_vehicle_arrive_time, current_state.vehicle_task_data)
        
        best_state = current_state.fast_copy()
        best_objective = current_state.destroyed_node_cost
        # current_state.vehicle_routes = [route.copy() for route in current_state.rm_empty_vehicle_route]
        current_objective = best_objective
        y_best.append(best_objective)
        y_cost.append(best_objective)
        
        start_time = time.time()
        # 基础奖励值设置为无人机平均成本
        print(f"DEBUG: 计算base_flexibility_bonus前，current_state.uav_cost包含 {len(current_state.uav_cost)} 个客户")
        print(f"DEBUG: 计算base_flexibility_bonus前，节点72是否在uav_cost中: {72 in current_state.uav_cost}")
        if 72 in current_state.uav_cost:
            print(f"DEBUG: 计算base_flexibility_bonus前，节点72的uav_cost值: {current_state.uav_cost[72]}")
        
        init_uav_cost = list(current_state.uav_cost.values())
        base_flexibility_bonus = sum(init_uav_cost) / len(init_uav_cost)
        
        print(f"DEBUG: 计算base_flexibility_bonus后，base_flexibility_bonus = {base_flexibility_bonus}")
        # 3. 初始化模拟退火和双重衰减奖励模型
        #    【重要建议】: 对于更复杂的搜索，建议增加迭代次数并减缓降温速率
        temperature = 500.0
        initial_temperature = temperature  # 记录初始温度，用于战略奖励计算
        cooling_rate = 0.995  # 缓慢降温以进行更充分的探索

        print(f"开始ALNS求解，初始成本: {best_objective:.2f}")

        # --------------------------------------------------------------------------
        # 阶段二：智能ALNS主循环
        # --------------------------------------------------------------------------
        for iteration in range(self.max_iterations):
            # if time.time() - start_time > self.max_runtime:
            #     print("达到最大运行时间，终止。")
            #     break

            # =================================================================
            # 步骤 2.1: 两层自适应选择 (宏观战略 + 具体战术)
            # =================================================================
            # 2.1.1 [第一层决策]: 根据策略权重，选择宏观战略
            strategy_names = list(self.strategy_weights.keys())
            # strategy_names = ['structural']
            strategy_w = np.array(list(self.strategy_weights.values()))
            # 使用轮盘赌选择一个策略 ('structural' 或 'internal')
            chosen_strategy = self.rng.choice(strategy_names, p=strategy_w / np.sum(strategy_w))
            # chosen_strategy = 'structural'

            # 2.1.2 [第二层决策]: 根据选定的策略，选择具体的破坏和修复算子
            # 获取当前策略专属的算子权重档案
            destroy_op_weights = self.operator_weights[chosen_strategy]['destroy']
            repair_op_weights = self.operator_weights[chosen_strategy]['repair']

            # 为破坏算子进行轮盘赌
            destroy_op_names = list(destroy_op_weights.keys())
            destroy_w = np.array(list(destroy_op_weights.values()))
            chosen_destroy_op_name = self.rng.choice(destroy_op_names, p=destroy_w / np.sum(destroy_w))
            destroy_op = getattr(self, chosen_destroy_op_name)

            # 为修复算子进行轮盘赌
            repair_op_names = list(repair_op_weights.keys())
            repair_w = np.array(list(repair_op_weights.values()))
            chosen_repair_op_name = self.rng.choice(repair_op_names, p=repair_w / np.sum(repair_w))
            repair_op = getattr(self, chosen_repair_op_name)

            print(f"\n--- 迭代 {iteration} | 温度: {temperature:.2f} | 选择策略: {chosen_strategy.upper()} ---")
            print(f"  > 战术组合: {chosen_destroy_op_name} + {chosen_repair_op_name}")
            
            # =================================================================
            # 步骤 2.2: 执行策略绑定的破坏与修复
            # =================================================================
            prev_state = current_state.fast_copy()
            prev_objective = current_objective
            if chosen_strategy == 'structural':
                # **策略一：结构性重组** (强制VTP破坏 + 带双重衰减奖励的修复)
                print(f"DEBUG: 准备调用destroy_op，prev_state.customer_plan包含 {len(prev_state.customer_plan)} 个客户")
                print(f"DEBUG: prev_state.customer_plan keys: {list(prev_state.customer_plan.keys())}")
                try:
                    destroyed_state = destroy_op(prev_state, force_vtp_mode=True)
                    print(f"DEBUG: destroy_op调用成功，destroyed_state.customer_plan包含 {len(destroyed_state.customer_plan)} 个客户")
                    print(f"DEBUG: destroyed_state.destroyed_customers_info包含 {len(destroyed_state.destroyed_customers_info)} 个被破坏的客户")
                except Exception as e:
                    print(f"DEBUG: destroy_op调用失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果destroy_op失败，创建一个空的destroyed_state
                    destroyed_state = prev_state.fast_copy()
                    destroyed_state.destroyed_customers_info = {}
                    destroyed_state.destroyed_vts_info = {}
                
                # 计算本轮迭代的战略奖励基准值
                strategic_bonus = base_flexibility_bonus * (temperature / initial_temperature)
                num_destroyed = len(destroyed_state.destroyed_customers_info)
                
                # 在调用repair_op前添加调试信息
                print(f"DEBUG: 调用repair_op前，destroyed_state.customer_plan包含 {len(destroyed_state.customer_plan)} 个客户")
                print(f"DEBUG: 调用repair_op前，节点72是否在customer_plan中: {72 in destroyed_state.customer_plan}")
                if 72 in destroyed_state.customer_plan:
                    print(f"DEBUG: 调用repair_op前，节点72的值: {destroyed_state.customer_plan[72]}")
                
                repaired_state, _ = repair_op(destroyed_state, strategic_bonus, num_destroyed, force_vtp_mode=True)
                
                # 在调用repair_op后添加调试信息
                print(f"DEBUG: 调用repair_op后，repaired_state.customer_plan包含 {len(repaired_state.customer_plan)} 个客户")
                print(f"DEBUG: 调用repair_op后，节点72是否在customer_plan中: {72 in repaired_state.customer_plan}")
                if 72 in repaired_state.customer_plan:
                    print(f"DEBUG: 调用repair_op后，节点72的值: {repaired_state.customer_plan[72]}")
                    if repaired_state.customer_plan[72] == []:
                        print("DEBUG: 警告！节点72的值为空列表！")
                
            else: # chosen_strategy == 'internal'
                # **策略二：内部精细优化** (强制客户破坏 + 无奖励的修复)
                destroyed_state = destroy_op(prev_state, force_vtp_mode=False)

                num_destroyed = len(destroyed_state.destroyed_customers_info)
                # 传入零奖励，关闭“战略投资”模式
                repaired_state, _ = repair_op(destroyed_state, strategic_bonus=0, num_destroyed=num_destroyed, force_vtp_mode=False)

            if not destroyed_state.customer_plan or not repaired_state.customer_plan:
                print("  > 破坏或修复后方案为空，跳过此次迭代。")
                iteration += 1
                continue

            # =================================================================
            # 步骤 2.3: 评估结果并为本次行动评分
            # =================================================================
            new_objective = repaired_state.objective()
            score = 0
            accepted = False
            
            print(f"  > 成本变化: {current_objective:.2f} -> {new_objective:.2f}")

            # 2.3.1 根据KPI标准为本次行动打分
            if new_objective < best_objective:
                score = self.reward_scores['new_best']
                print(f"  > 结果: 发现新的全局最优解! 奖励 {score} 分。")
            elif new_objective < current_objective:
                score = self.reward_scores['better_than_current']
                print(f"  > 结果: 找到更优解。奖励 {score} 分。")
            elif self._simulated_annealing_accept(current_objective, new_objective, temperature):
                score = self.reward_scores['accepted']
                print(f"  > 结果: 接受一个较差解（探索成功）。奖励 {score} 分。")

            # 2.3.2 根据模拟退火决定是否接受新解
            if new_objective < current_objective or (score == self.reward_scores['accepted']):
                accepted = True
            
            # =================================================================
            # 步骤 2.4: 学习与进化 - 更新两层权重
            # =================================================================
            if score > 0: # 只有有价值的行动才参与学习
                # 2.4.1 更新顶层的策略权重
                self.strategy_weights[chosen_strategy] = \
                    (1 - self.reaction_factor) * self.strategy_weights[chosen_strategy] + \
                    self.reaction_factor * score

                # 2.4.2 更新第二层的、具体的算子权重
                # 更新破坏算子
                self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] = \
                    (1 - self.reaction_factor) * self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] + \
                    self.reaction_factor * score
                
                # 更新修复算子
                self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] = \
                    (1 - self.reaction_factor) * self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] + \
                    self.reaction_factor * score

            # =================================================================
            # 步骤 2.5: 更新状态并进入下一次迭代
            # =================================================================
            if accepted:
                current_state = repaired_state
                current_objective = new_objective
                if new_objective < best_objective: # 再次检查以更新最优状态
                    best_state = repaired_state.fast_copy()
                    best_objective = new_objective
                    y_best.append(best_objective)
            else:
                # 不接受，状态自动保持为 prev_state (因为我们是在副本上操作)
                # 无需像原来那样显式回滚
                pass

            # 温度衰减
            temperature *= cooling_rate
            y_cost.append(current_objective)
            
            # 日志记录 (保留)
            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"  > 进度: 迭代 {iteration}, 当前成本: {current_objective:.2f}, 最优成本: {best_objective:.2f}, 运行时间: {elapsed_time:.2f}秒")
                # 打印权重以供调试
                # print(f"  > 策略权重: {self.strategy_weights}")
                
            iteration += 1

        # elapsed_time = time.time() - start_time
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
    
    def destroy_random_removal(self, state, force_vtp_mode = None):
        """随机客户点移除：随机删除20%-30%的客户点任务"""
        """
        随机破坏算子，实现了双重模式以适应自适应策略选择框架。
        它既可以随机移除少量VTP节点以重构路径，也可以随机移除大量客户以重组任务。
        Args:
            state (FastMfstspState): 当前解的状态。
            force_vtp_mode (bool, optional): 
                - True:  强制执行VTP破坏模式 (用于“结构性重组”策略)。
                - False: 强制执行客户破坏模式 (用于“内部精细优化”策略)。
                - None: (默认) 按预设概率随机选择一种模式 (此模式在当前框架下不会被触发，但保留以备后用)。
        
        Returns:
            FastMfstspState: 被部分破坏后的解的状态。
        """

        # 拷贝当前解
        new_state = state.fast_copy()
        # 获取当前解中的客户点（而不是所有可能的客户点）
        current_customers = list(new_state.customer_plan.keys())
        if not current_customers:
            print("没有客户点需要移除")
            return new_state

        # new_state.vehicle_routes = new_state.rm_empty_vehicle_route  # 更新路径
        mode = 'vtp' if force_vtp_mode else 'customer'
        print(f"  > [破坏模式]: 随机破坏 ({'VTP模式' if mode == 'vtp' else '客户模式'})")
        vehicle_task_data = new_state.vehicle_task_data
        if mode == 'vtp':
            # 收集所有活跃的VTP节点
            active_vtps = []
            destroyed_vts_info = {}
            for vehicle_id, route in enumerate(new_state.vehicle_routes):
                v_id = vehicle_id + 1
                for vtp_node in route[1:-1]:
                    active_vtps.append((v_id, vtp_node))
            
            low, high = self.vtp_destroy_quantity['random']
            num_to_remove = self.rng.integers(low, min(len(active_vtps), high) + 1)
            print(f"VTP破坏策略：目标破坏 {num_to_remove} 个VTP节点，候选池共有 {len(active_vtps)} 个节点")
            
            # 开始执行vtp节点任务的破坏策略
            destroyed_customers_info = {}  # 用于存储被破坏的客户节点信息
            destroyed_vtp_count = 0  # 实际破坏的VTP节点数量
            max_attempts = len(active_vtps) * 2  # 最大尝试次数，避免无限循环
            attempt_count = 0
            
            # 创建候选节点池的副本，用于随机选择,避免指向同一对象
            candidate_vtps = active_vtps.copy()
            
            while destroyed_vtp_count < num_to_remove and candidate_vtps and attempt_count < max_attempts:
                attempt_count += 1
                
                # 从候选池中随机选择一个VTP节点
                if not candidate_vtps:
                    print(f"候选池已空，无法继续破坏VTP节点")
                    break
                    
                selected_index = self.rng.integers(0, len(candidate_vtps))
                vehicle_id, vtp_node = candidate_vtps.pop(selected_index)
                if vtp_node not in new_state.rm_empty_vehicle_route[vehicle_id-1]:
                    continue
                    
                # 1. 首先收集所有需要删除的相关客户点任务
                customers_to_remove = []
                for customer, assignment in list(new_state.customer_plan.items()):
                    uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                    
                    # 检查发射节点或回收节点是否与破坏的vtp_node一致，且车辆ID相同
                    if (launch_node == vtp_node and launch_vehicle == vehicle_id) or \
                       (recovery_node == vtp_node and recovery_vehicle == vehicle_id):
                        customers_to_remove.append(customer)
                
                # 2. 收集所有需要链式删除的任务
                all_tasks_to_remove = []
                temp_vehicle_task_data = vehicle_task_data  # 用于链式删除分析
                # 构建临时客户点集合
                temp_customer_plan = {k: v for k, v in new_state.customer_plan.items()}
                temp_rm_empty_vehicle_route = [route[:] for route in new_state.rm_empty_vehicle_route]
                
                for customer in customers_to_remove:
                    if customer in temp_customer_plan:
                        assignment = temp_customer_plan[customer]
                        all_tasks_to_remove.append((customer, assignment))
                        
                        # 通过链式找到这个无人机后续的所有服务任务
                        need_to_remove_tasks = find_chain_tasks(assignment, temp_customer_plan, new_state.vehicle_routes, temp_vehicle_task_data)
                        all_tasks_to_remove.extend(need_to_remove_tasks)
                        # # 更新临时vehicle_task_data用于后续链式分析
                        # temp_vehicle_task_data = remove_vehicle_task(temp_vehicle_task_data, assignment, new_state.vehicle_routes)
                        # for chain_customer, chain_assignment in need_to_remove_tasks:
                        #     temp_vehicle_task_data = deep_remove_vehicle_task(temp_vehicle_task_data, chain_assignment, new_state.vehicle_routes)
                
                # 从临时状态中移除所有相关任务
                for customer, assignment in all_tasks_to_remove:
                    if customer in temp_customer_plan:
                        temp_customer_plan.pop(customer, None)
                
                # 从临时车辆路线中移除VTP节点
                if vtp_node in temp_rm_empty_vehicle_route[vehicle_id-1]:
                    temp_rm_empty_vehicle_route[vehicle_id-1].remove(vtp_node)
                
                # 4. 计算临时车辆到达时间并检查约束
                temp_rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_rm_empty_vehicle_route)
                
                # 5. 检查时间约束
                if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                    print(f"VTP节点 {vtp_node} 删除后不满足时间约束，跳过删除 (尝试 {attempt_count}/{max_attempts})")
                    continue
                
                # 6. 约束满足，执行实际删除操作
                print(f"成功破坏VTP节点: 车辆{vehicle_id}的节点{vtp_node} (进度: {destroyed_vtp_count + 1}/{num_to_remove})")
                
                # 从车辆路线中移除VTP节点,测试通过，开始正常处理任务
                new_state.rm_empty_vehicle_route[vehicle_id-1].remove(vtp_node)
                destroyed_vts_info[(vehicle_id, vtp_node)] = True
                destroyed_vtp_count += 1  # 增加破坏计数
                
                # 处理所有需要删除的客户点任务
                for customer, assignment in all_tasks_to_remove:
                    if customer in new_state.customer_plan:
                        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                        
                        # 记录被破坏客户节点的详细信息
                        customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
                        destroyed_customers_info[customer] = customer_info
                        
                        # 从customer_plan中移除
                        new_state.customer_plan.pop(customer, None)
                        
                        # 从无人机分配中移除相关任务
                        if uav_id in new_state.uav_assignments:
                            new_state.uav_assignments[uav_id] = [
                                task for task in new_state.uav_assignments[uav_id]
                                if task[2] != customer_node  # task[2]是customer_node
                            ]
                        
                        # 更新破坏的无人机空中成本
                        if new_state.uav_cost and customer_node in new_state.uav_cost:
                            new_state.uav_cost.pop(customer_node, None)
                        
                        # 更新vehicle_task_data
                        vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                        
                        # 处理链式删除的任务
                        from task_data import deep_remove_vehicle_task
                        need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, vehicle_task_data)
                        for chain_customer, chain_assignment in need_to_remove_tasks:
                            if chain_customer in new_state.customer_plan:
                                chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                                
                                # 记录被破坏客户节点的详细信息
                                chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                                destroyed_customers_info[chain_customer] = chain_customer_info
                                
                                # 从customer_plan中移除
                                new_state.customer_plan.pop(chain_customer, None)
                                
                                # 从无人机分配中移除相关任务
                                if chain_uav_id in new_state.uav_assignments:
                                    new_state.uav_assignments[chain_uav_id] = [
                                        task for task in new_state.uav_assignments[chain_uav_id]
                                        if task[2] != chain_customer_node
                                    ]
                                
                                # 更新破坏的无人机空中成本
                                if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                                    new_state.uav_cost.pop(chain_customer_node, None)
                                
                                print(f"VTP链式删除客户点 {chain_customer}")
                                vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)
            
            # 输出破坏策略执行结果
            if destroyed_vtp_count == num_to_remove:
                print(f"VTP破坏策略成功完成：目标 {num_to_remove} 个，实际破坏 {destroyed_vtp_count} 个VTP节点，共删除 {len(destroyed_customers_info)} 个客户点")
            elif destroyed_vtp_count > 0:
                print(f"VTP破坏策略部分完成：目标 {num_to_remove} 个，实际破坏 {destroyed_vtp_count} 个VTP节点，共删除 {len(destroyed_customers_info)} 个客户点")
            else:
                print(f"VTP破坏策略失败：目标 {num_to_remove} 个，实际破坏 {destroyed_vtp_count} 个VTP节点，共删除 {len(destroyed_customers_info)} 个客户点")
                print(f"警告：VTP破坏失败，destroyed_customers_info为空: {destroyed_customers_info}")
                # 如果VTP破坏完全失败，回退到客户破坏模式
                print("VTP破坏失败，回退到客户破坏模式...")
                
                # 回退到客户破坏模式：随机选择一个客户进行破坏
                if current_customers:
                    fallback_customer = self.rng.choice(current_customers)
                    if fallback_customer in new_state.customer_plan:
                        assignment = new_state.customer_plan.pop(fallback_customer)
                        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                        
                        # 记录被破坏客户节点的详细信息
                        customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(fallback_customer, 0) if new_state.uav_cost else 0]
                        destroyed_customers_info[fallback_customer] = customer_info
                        
                        # 从无人机分配中移除相关任务
                        if uav_id in new_state.uav_assignments:
                            new_state.uav_assignments[uav_id] = [
                                task for task in new_state.uav_assignments[uav_id]
                                if task[2] != customer_node
                            ]
                        
                        # 更新破坏的无人机空中成本
                        if new_state.uav_cost and customer_node in new_state.uav_cost:
                            new_state.uav_cost.pop(customer_node, None)
                        
                        # 更新vehicle_task_data
                        vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                        
                        print(f"回退破坏：成功破坏客户点 {fallback_customer}")
                    else:
                        print("回退破坏：无法找到可破坏的客户点")
                else:
                    print("回退破坏：没有可用的客户点")
            
            # 更新对应的vehicle_task_data
            # vehicle_task_data = new_state.vehicle_task_data
            # vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
            # new_state.vehicle_task_data = vehicle_task_data

            # 更新状态
            new_state.destroyed_vts_info = destroyed_vts_info
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            # 更新空跑节点等状态
            # new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            new_state.vehicle_routes = [route[:] for route in new_state.rm_empty_vehicle_route]  # vtp节点被破坏后重更新
            # 更新基础达到时间
            new_state.rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(new_state.vehicle_routes)
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            # print(f"破坏后剩余VTP节点: {sum(len(route) - 2 for route in new_state.vehicle_routes)}")  # 减去起点和终点
            # print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== VTP破坏阶段完成 ===\n")
        else:
            # 开始执行客户点层面的破坏策略
            # 1. 随机选择要移除的客户点
            n = len(current_customers)
            num_to_remove = self.rng.integers(
                max(1, int(n * 0.2)),
                max(2, int(n * 0.3)) + 1
            )
            customers_to_remove = self.rng.choice(current_customers, num_to_remove, replace=False)

            print(f"随机破坏：移除 {len(customers_to_remove)} 个客户点: {customers_to_remove}")
            destroyed_customers_info = {}
            
            # 2. 移除这些客户点及相关无人机任务
            for customer in customers_to_remove:
                if customer in new_state.customer_plan:
                    assignment = new_state.customer_plan.pop(customer)
                    uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                    
                    # 记录被破坏客户节点的详细信息
                    customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
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
                    # 删除对应的状态任务
                    vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                    # 进一步通过链式找到这个无人机后续的所有服务任务，同车则不变，异车则全部删除对应的后续所有任务，并整理出所有需要删除的任务
                    need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, new_state.vehicle_task_data)
                    
                    # 处理链式删除的任务
                    for chain_customer, chain_assignment in need_to_remove_tasks:
                        if chain_customer in new_state.customer_plan:
                            chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                            
                            # 记录被破坏客户节点的详细信息
                            chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                            destroyed_customers_info[chain_customer] = chain_customer_info
                            
                            # 从customer_plan中移除
                            new_state.customer_plan.pop(chain_customer, None)
                            
                            # 从无人机分配中移除相关任务
                            if chain_uav_id in new_state.uav_assignments:
                                new_state.uav_assignments[chain_uav_id] = [
                                    task for task in new_state.uav_assignments[chain_uav_id]
                                    if task[2] != chain_customer_node
                                ]
                            
                            # 更新破坏的无人机空中成本
                            if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                                new_state.uav_cost.pop(chain_customer_node, None)
                            
                            # print(f"链式删除客户点 {chain_customer}")
                            vehicle_task_data = remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)

                    # 更新对应的vehicle_task_data
                    # vehicle_task_data = new_state.vehicle_task_data
                    # vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                    # new_state.vehicle_task_data = vehicle_task_data

            # 3. 更新空跑节点等状态
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            
            # 将破坏的客户节点信息存储到状态中，供修复阶段使用
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== 破坏阶段完成 ===\n")
        return new_state

    # 最坏节点破坏
    def destroy_worst_removal(self, state, force_vtp_mode = None):
        """最差节点移除：基于成本效益比删除最差的VTP节点或客户点任务"""
        """
        最差破坏算子，实现了双重模式以适应自适应策略选择框架。
        它既可以基于成本效益比移除最差的VTP节点以重构路径，也可以移除成本最高的客户以重组任务。
        Args:
            state (FastMfstspState): 当前解的状态。
            force_vtp_mode (bool, optional): 
                - True:  强制执行VTP破坏模式 (用于"结构性重组"策略)。
                - False: 强制执行客户破坏模式 (用于"内部精细优化"策略)。
                - None: (默认) 按预设概率随机选择一种模式 (此模式在当前框架下不会被触发，但保留以备后用)。
        
        Returns:
            FastMfstspState: 被部分破坏后的解的状态。
        """

        # 拷贝当前解
        new_state = state.fast_copy()
        # 获取当前解中的客户点（而不是所有可能的客户点）
        current_customers = list(new_state.customer_plan.keys())
        if not current_customers:
            print("没有客户点需要移除")
            return new_state

        # new_state.vehicle_routes = new_state.rm_empty_vehicle_route  # 更新路径
        mode = 'vtp' if force_vtp_mode else 'customer'
        print(f"  > [破坏模式]: 最差破坏 ({'VTP模式' if mode == 'vtp' else '客户模式'})")
        vehicle_task_data = new_state.vehicle_task_data
        
        if mode == 'vtp':
            # 收集所有活跃的VTP节点并计算成本效益比
            active_vtps_with_cost_ratio = []
            destroyed_vts_info = {}
            
            for vehicle_id, route in enumerate(new_state.vehicle_routes):
                v_id = vehicle_id + 1
                for vtp_node in route[1:-1]:
                    # 计算该VTP节点的任务数和总成本
                    launch_tasks = 0  # 发射任务数
                    recovery_tasks = 0  # 回收任务数
                    total_cost = 0.0  # 总成本
                    
                    # 统计发射任务：从该VTP节点出发的无人机任务
                    for customer, assignment in new_state.customer_plan.items():
                        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                        if launch_node == vtp_node and launch_vehicle == v_id:
                            launch_tasks += 1
                            # 计算从VTP到客户的成本，计算从vtp到客户的成本
                            launch_node_map = self.node[launch_node].map_key
                            total_cost += self.uav_travel[uav_id][launch_node_map][customer_node].totalDistance * self.vehicle[uav_id].per_cost
                    
                    # 统计回收任务：返回该VTP节点的无人机任务
                    for customer, assignment in new_state.customer_plan.items():
                        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                        if recovery_node == vtp_node and recovery_vehicle == v_id:
                            recovery_tasks += 1
                            # 计算从客户到VTP的成本（这里使用客户点的成本作为近似）
                            recovery_node_map = self.node[recovery_node].map_key
                            total_cost += self.uav_travel[uav_id][customer_node][recovery_node_map].totalDistance * self.vehicle[uav_id].per_cost
                    
                    # 计算成本效益比：总成本 / 任务数
                    total_tasks = launch_tasks + recovery_tasks
                    if total_tasks > 0:
                        cost_ratio = total_cost / total_tasks
                    else:
                        # 没有发射和回收任务的VTP节点设置为最大成本，优先被破坏
                        cost_ratio = float('inf')
                    
                    active_vtps_with_cost_ratio.append(((v_id, vtp_node), cost_ratio, total_cost, total_tasks))
            
            # 按成本效益比降序排序（成本效益比越高，越应该被删除）
            active_vtps_with_cost_ratio.sort(key=lambda x: x[1], reverse=True)
            
            # 选择要删除的VTP节点数量
            num_to_remove = self.vtp_destroy_quantity['worst']
            num_to_remove = min(num_to_remove, len(active_vtps_with_cost_ratio))
            
            # 选择最差的VTP节点
            vtps_to_destroy = [item[0] for item in active_vtps_with_cost_ratio[:num_to_remove]]
            
            # 显示将要破坏的VTP节点信息
            # print(f"VTP最差破坏策略：目标破坏 {num_to_remove} 个VTP节点")
            for i, (vehicle_id, vtp_node) in enumerate(vtps_to_destroy):
                vtp_info = active_vtps_with_cost_ratio[i]
                cost_ratio, total_cost, total_tasks = vtp_info[1], vtp_info[2], vtp_info[3]

            # 开始执行vtp节点任务的破坏策略
            destroyed_customers_info = {}  # 用于存储被破坏的客户节点信息
            destroyed_vtp_count = 0  # 实际破坏的VTP节点数量
            max_attempts = len(vtps_to_destroy) * 2  # 最大尝试次数，避免无限循环
            attempt_count = 0
            
            # 创建候选节点池的副本，用于按优先级选择
            candidate_vtps = vtps_to_destroy.copy()
            
            while destroyed_vtp_count < len(vtps_to_destroy) and candidate_vtps and attempt_count < max_attempts:
                attempt_count += 1
                
                # 从候选池中选择下一个VTP节点（按成本效益比排序）
                if not candidate_vtps:
                    print(f"候选池已空，无法继续破坏VTP节点")
                    break
                    
                vehicle_id, vtp_node = candidate_vtps.pop(0)  # 按优先级顺序选择
                if vtp_node not in new_state.rm_empty_vehicle_route[vehicle_id-1]:
                    continue
                
                # 1. 首先收集所有需要删除的相关客户点任务
                customers_to_remove = []
                for customer, assignment in list(new_state.customer_plan.items()):
                    uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                    
                    # 检查发射节点或回收节点是否与破坏的vtp_node一致，且车辆ID相同
                    if (launch_node == vtp_node and launch_vehicle == vehicle_id) or \
                       (recovery_node == vtp_node and recovery_vehicle == vehicle_id):
                        customers_to_remove.append(customer)
                
                # 2. 收集所有需要链式删除的任务
                all_tasks_to_remove = []
                temp_vehicle_task_data = vehicle_task_data  # 用于链式删除分析
                # 构建临时客户点集合
                temp_customer_plan = {k: v for k, v in new_state.customer_plan.items()}
                temp_rm_empty_vehicle_route = [route[:] for route in new_state.rm_empty_vehicle_route]
                
                for customer in customers_to_remove:
                    if customer in temp_customer_plan:
                        assignment = temp_customer_plan[customer]
                        all_tasks_to_remove.append((customer, assignment))
                        
                        # 通过链式找到这个无人机后续的所有服务任务
                        need_to_remove_tasks = find_chain_tasks(assignment, temp_customer_plan, new_state.vehicle_routes, temp_vehicle_task_data)
                        all_tasks_to_remove.extend(need_to_remove_tasks)
                
                # 从临时状态中移除所有相关任务
                for customer, assignment in all_tasks_to_remove:
                    if customer in temp_customer_plan:
                        temp_customer_plan.pop(customer, None)
                
                # 从临时车辆路线中移除VTP节点
                if vtp_node in temp_rm_empty_vehicle_route[vehicle_id-1]:
                    temp_rm_empty_vehicle_route[vehicle_id-1].remove(vtp_node)
                
                # 4. 计算临时车辆到达时间并检查约束
                temp_rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_rm_empty_vehicle_route)
                
                # 5. 检查时间约束
                if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                    print(f"VTP节点 {vtp_node} 删除后不满足时间约束，跳过删除 (尝试 {attempt_count}/{max_attempts})")
                    continue
                
                # 6. 约束满足，执行实际删除操作
                print(f"成功破坏VTP节点: 车辆{vehicle_id}的节点{vtp_node} (进度: {destroyed_vtp_count + 1}/{len(vtps_to_destroy)})")
                
                # 从车辆路线中移除VTP节点
                new_state.rm_empty_vehicle_route[vehicle_id-1].remove(vtp_node)
                destroyed_vts_info[(vehicle_id-1, vtp_node)] = True  # 均统一为索引形式
                destroyed_vtp_count += 1  # 增加破坏计数
                
                # 处理所有需要删除的客户点任务
                for customer, assignment in all_tasks_to_remove:
                    if customer in new_state.customer_plan:
                        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                        
                        # 记录被破坏客户节点的详细信息
                        customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
                        destroyed_customers_info[customer] = customer_info
                        
                        # 从customer_plan中移除
                        new_state.customer_plan.pop(customer, None)
                        
                        # 从无人机分配中移除相关任务
                        if uav_id in new_state.uav_assignments:
                            new_state.uav_assignments[uav_id] = [
                                task for task in new_state.uav_assignments[uav_id]
                                if task[2] != customer_node  # task[2]是customer_node
                            ]
                        
                        # 更新破坏的无人机空中成本
                        if new_state.uav_cost and customer_node in new_state.uav_cost:
                            new_state.uav_cost.pop(customer_node, None)
                        
                        # 更新vehicle_task_data
                        vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                        
                        # 处理链式删除的任务
                        from task_data import deep_remove_vehicle_task
                        need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, vehicle_task_data)
                        for chain_customer, chain_assignment in need_to_remove_tasks:
                            if chain_customer in new_state.customer_plan:
                                chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                                
                                # 记录被破坏客户节点的详细信息
                                chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                                destroyed_customers_info[chain_customer] = chain_customer_info
                                
                                # 从customer_plan中移除
                                new_state.customer_plan.pop(chain_customer, None)
                                
                                # 从无人机分配中移除相关任务
                                if chain_uav_id in new_state.uav_assignments:
                                    new_state.uav_assignments[chain_uav_id] = [
                                        task for task in new_state.uav_assignments[chain_uav_id]
                                        if task[2] != chain_customer_node
                                    ]
                                
                                # 更新破坏的无人机空中成本
                                if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                                    new_state.uav_cost.pop(chain_customer_node, None)
                                
                                print(f"VTP链式删除客户点 {chain_customer}")
                                vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)
            
            print(f"VTP最差破坏策略完成：成功破坏 {destroyed_vtp_count}/{len(vtps_to_destroy)} 个VTP节点")
            
            # 更新状态
            new_state.destroyed_vts_info = destroyed_vts_info
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            # 更新空跑节点等状态
            new_state.vehicle_routes = [route.copy() for route in new_state.rm_empty_vehicle_route]  # vtp节点被破坏后重更新
            new_state.rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(new_state.vehicle_routes)
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            print(f"破坏后剩余VTP节点: {sum(len(route) - 2 for route in new_state.vehicle_routes)}")  # 减去起点和终点
            print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== VTP破坏阶段完成 ===\n")
        else:
            # 开始执行客户点层面的破坏策略
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

            print(f"最差客户破坏：移除 {len(customers_to_remove)} 个客户点: {customers_to_remove}")
            destroyed_customers_info = {}
            
            # 4. 移除这些客户点及相关无人机任务
            for customer in customers_to_remove:
                if customer in new_state.customer_plan:
                    assignment = new_state.customer_plan.pop(customer)
                    uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                    
                    # 记录被破坏客户节点的详细信息
                    customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
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
                    # 删除对应的状态任务
                    vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                    # 进一步通过链式找到这个无人机后续的所有服务任务，同车则不变，异车则全部删除对应的后续所有任务，并整理出所有需要删除的任务
                    need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, new_state.vehicle_task_data)
                    
                    # 处理链式删除的任务
                    for chain_customer, chain_assignment in need_to_remove_tasks:
                        if chain_customer in new_state.customer_plan:
                            chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                            
                            # 记录被破坏客户节点的详细信息
                            chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                            destroyed_customers_info[chain_customer] = chain_customer_info
                            
                            # 从customer_plan中移除
                            new_state.customer_plan.pop(chain_customer, None)
                            
                            # 从无人机分配中移除相关任务
                            if chain_uav_id in new_state.uav_assignments:
                                new_state.uav_assignments[chain_uav_id] = [
                                    task for task in new_state.uav_assignments[chain_uav_id]
                                    if task[2] != chain_customer_node
                                ]
                            
                            # 更新破坏的无人机空中成本
                            if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                                new_state.uav_cost.pop(chain_customer_node, None)
                            
                            print(f"链式删除客户点 {chain_customer}")
                            vehicle_task_data = remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)

            # 5. 更新空跑节点等状态
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            
            # 将破坏的客户节点信息存储到状态中，供修复阶段使用
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== 破坏阶段完成 ===\n")
        return new_state

    def destroy_shaw_removal(self, state, force_vtp_mode = None):
        """
        Shaw相似性破坏算子：基于空间位置相似性移除客户点
        随机选择一个种子客户点，然后移除与其在空间位置上最相似的若干客户点
        """
        # 拷贝当前解
        new_state = state.fast_copy()
        # 获取当前解中的客户点（而不是所有可能的客户点）
        current_customers = list(new_state.customer_plan.keys())
        if not current_customers:
            print("没有客户点需要移除")
            return new_state

        # 1. 随机选择一个种子客户点
        seed_customer = self.rng.choice(current_customers)
        seed_pos = np.array([
            self.node[seed_customer].latDeg,
            self.node[seed_customer].lonDeg,
            self.node[seed_customer].altMeters
        ])

        print(f"Shaw破坏：选择种子客户点 {seed_customer}")

        # 2. 计算所有其他客户点与种子的空间距离
        customer_distances = []
        for customer in current_customers:
            if customer == seed_customer:
                continue
            pos = np.array([
                self.node[customer].latDeg,
                self.node[customer].lonDeg,
                self.node[customer].altMeters
            ])
            # 计算欧几里得距离
            dist = np.linalg.norm(pos - seed_pos)
            customer_distances.append((customer, dist))

        # 3. 按距离升序排序，选出最相似的若干客户
        customer_distances.sort(key=lambda x: x[1])
        n = len(current_customers)
        num_to_remove = self.rng.integers(
            max(1, int(n * 0.2)),
            max(2, int(n * 0.3)) + 1
        )
        
        # 选出距离最近的客户点，包括种子
        customers_to_remove = [seed_customer] + [customer for customer, _ in customer_distances[:num_to_remove-1]]

        print(f"Shaw破坏：移除 {len(customers_to_remove)} 个相似客户点: {customers_to_remove}")
        destroyed_customers_info = {}
        
        # 4. 移除这些客户点及相关无人机任务
        for customer in customers_to_remove:
            if customer in new_state.customer_plan:
                assignment = new_state.customer_plan.pop(customer)
                uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                
                # 记录被破坏客户节点的详细信息
                customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
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
                
                # 进一步通过链式找到这个无人机后续的所有服务任务，同车则不变，异车则全部删除对应的后续所有任务，并整理出所有需要删除的任务
                need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, new_state.vehicle_task_data)
                
                # 处理链式删除的任务
                for chain_customer, chain_assignment in need_to_remove_tasks:
                    if chain_customer in new_state.customer_plan:
                        chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                        
                        # 记录被破坏客户节点的详细信息
                        chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                        destroyed_customers_info[chain_customer] = chain_customer_info
                        
                        # 从customer_plan中移除
                        new_state.customer_plan.pop(chain_customer, None)
                        
                        # 从无人机分配中移除相关任务
                        if chain_uav_id in new_state.uav_assignments:
                            new_state.uav_assignments[chain_uav_id] = [
                                task for task in new_state.uav_assignments[chain_uav_id]
                                if task[2] != chain_customer_node
                            ]
                        
                        # 更新破坏的无人机空中成本
                        if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                            new_state.uav_cost.pop(chain_customer_node, None)
                        
                        print(f"链式删除客户点 {chain_customer}")

                # 更新对应的vehicle_task_data
                vehicle_task_data = new_state.vehicle_task_data
                vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                new_state.vehicle_task_data = vehicle_task_data

        # 5. 更新空跑节点等状态
        new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
        
        # 将破坏的客户节点信息存储到状态中，供修复阶段使用
        new_state.destroyed_customers_info = destroyed_customers_info
        
        print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
        print("=== Shaw破坏阶段完成 ===\n")
        return new_state
    
    def destroy_vtp_removal(self, state):
        """
        VTP节点移除破坏算子：直接移除车辆路径中的VTP节点，颠覆性地改变车辆路径结构
        逻辑：
        1. 随机选择车辆路径中的VTP节点进行移除
        2. 移除VTP节点后，所有以该节点为起降点的无人机任务失效
        3. 将这些失效任务服务的客户点加入待修复列表
        """
        new_state = state.fast_copy()
        
        # 获取所有车辆路径中的VTP节点
        all_vtp_in_routes = []
        for vehicle_id, route in new_state.vehicle_routes.items():
            for node in route[1:-1]:  # 排除起点和终点
                if node in self.A_vtp:  # 如果是VTP节点
                    all_vtp_in_routes.append((vehicle_id, node))
        
        if not all_vtp_in_routes:
            print("VTP破坏：没有找到可移除的VTP节点")
            return new_state
        
        # 随机选择1-2个VTP节点进行移除
        num_to_remove = self.rng.integers(1, min(3, len(all_vtp_in_routes)) + 1)
        vtp_to_remove = self.rng.choice(all_vtp_in_routes, num_to_remove, replace=False)
        
        print(f"VTP破坏：选择移除 {len(vtp_to_remove)} 个VTP节点: {vtp_to_remove}")
        
        destroyed_customers_info = {}
        removed_vtp_info = {}  # 记录被移除的VTP节点信息
        
        # 处理每个要移除的VTP节点
        for vehicle_id, vtp_node in vtp_to_remove:
            print(f"VTP破坏：移除车辆 {vehicle_id} 的VTP节点 {vtp_node}")
            
            # 1. 从车辆路径中移除VTP节点
            route = new_state.vehicle_routes[vehicle_id]
            if vtp_node in route:
                route.remove(vtp_node)
                new_state.vehicle_routes[vehicle_id] = route
                removed_vtp_info[(vehicle_id, vtp_node)] = True
            
            # 2. 找到所有以该VTP节点为起降点的无人机任务
            affected_customers = self._find_customers_using_vtp(vtp_node, new_state.customer_plan)
            
            # 3. 移除这些失效的客户任务
            for customer in affected_customers:
                if customer in new_state.customer_plan:
                    assignment = new_state.customer_plan.pop(customer)
                    uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                    
                    # 记录被破坏客户节点的详细信息
                    customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, 
                                   new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
                    destroyed_customers_info[customer] = customer_info
                    
                    # 从无人机分配中移除相关任务
                    if uav_id in new_state.uav_assignments:
                        new_state.uav_assignments[uav_id] = [
                            task for task in new_state.uav_assignments[uav_id]
                            if task[2] != customer_node
                        ]
                    
                    # 更新破坏的无人机空中成本
                    if new_state.uav_cost and customer_node in new_state.uav_cost:
                        new_state.uav_cost.pop(customer_node, None)
                    
                    # 链式删除相关任务
                    need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, new_state.vehicle_task_data)
                    
                    for chain_customer, chain_assignment in need_to_remove_tasks:
                        if chain_customer in new_state.customer_plan:
                            chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                            
                            chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, 
                                                 chain_launch_vehicle, chain_recovery_vehicle, 
                                                 new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                            destroyed_customers_info[chain_customer] = chain_customer_info
                            
                            new_state.customer_plan.pop(chain_customer, None)
                            
                            if chain_uav_id in new_state.uav_assignments:
                                new_state.uav_assignments[chain_uav_id] = [
                                    task for task in new_state.uav_assignments[chain_uav_id]
                                    if task[2] != chain_customer_node
                                ]
                            
                            if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                                new_state.uav_cost.pop(chain_customer_node, None)
                            
                            print(f"VTP破坏：链式删除客户点 {chain_customer}")
                    
                    # 更新vehicle_task_data
                    vehicle_task_data = new_state.vehicle_task_data
                    vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                    new_state.vehicle_task_data = vehicle_task_data
                    
                    print(f"VTP破坏：移除客户点 {customer}（使用VTP节点 {vtp_node}）")
        
        # 4. 更新状态
        new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
        new_state.destroyed_customers_info = destroyed_customers_info
        new_state.removed_vtp_info = removed_vtp_info  # 记录被移除的VTP信息，供修复算子使用
        
        print(f"VTP破坏：移除 {len(vtp_to_remove)} 个VTP节点，影响 {len(destroyed_customers_info)} 个客户点")
        print(f"VTP破坏后剩余客户点: {len(new_state.customer_plan)}")
        print("=== VTP破坏阶段完成 ===\n")
        return new_state
    
    def _find_customers_using_vtp(self, vtp_node, customer_plan):
        """
        找到所有使用指定VTP节点作为起降点的客户点
        """
        affected_customers = []
        for customer, assignment in customer_plan.items():
            uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
            if launch_node == vtp_node or recovery_node == vtp_node:
                affected_customers.append(customer)
        return affected_customers
    
    def destroy_important_removal(self, state):
        """
        重要性节点破坏：优先破坏无人机任务数量多的客户节点（发射+回收），
        但通过噪声实现一定的随机性，避免每次都只破坏最重要的节点。
        """
        new_state = state.fast_copy()
        # 更新路径
        rm_empty_vehicle_route = state.rm_empty_vehicle_route
        new_state.vehicle_routes = [route.copy() for route in rm_empty_vehicle_route]
        all_customers = list(self.A_c)
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
    
    def cluster_vtp_for_customers(self, k):
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
        # 记录其映射关系
        map_customer_vtp_dict = {}
        for cid, sorted_vtps in customer_vtp_dict.items():
            map_customer_vtp_dict[cid] = [self.node[vtp_id].map_key for vtp_id in sorted_vtps]

        return customer_vtp_dict, map_customer_vtp_dict

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
        max_iterations, max_runtime=60, use_incremental=True):
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
        max_iterations, max_runtime=max_runtime)
    # else:
    #     # 使用快速ALNS
    #     alns_solver = FastALNS(max_iterations=max_iterations, max_runtime=max_runtime)
    
    # 使用ALNS求解
    best_solution, best_objective, statistics = alns_solver.solve(initial_solution)
    
    return best_solution, best_objective, statistics 

# class FastALNS:
#     """高效的ALNS求解器 - 使用浅拷贝和增量更新"""
    
#     def __init__(self, max_iterations=1000, max_runtime=60):
#         self.max_iterations = max_iterations
#         self.max_runtime = max_runtime
#         self.rng = rnd.default_rng(42)
        
#     def solve(self, initial_state):
#         """
#         使用高效的ALNS算法求解
        
#         Args:
#             initial_state: 初始解状态
            
#         Returns:
#             tuple: (best_solution, best_objective, statistics)
#         """
#         current_state = initial_state.fast_copy()
#         best_state = current_state.fast_copy()
#         best_objective = best_state.objective()
        
#         start_time = time.time()
#         iteration = 0
        
#         print(f"开始快速ALNS求解，初始成本: {best_objective}")
        
#         while iteration < self.max_iterations and (time.time() - start_time) < self.max_runtime:
#             # 破坏阶段 - 使用增量修改
#             destroyed_state = self._fast_destroy(current_state)
            
#             # 修复阶段 - 使用增量修改
#             repaired_state = self._fast_repair(destroyed_state)
            
#             # 接受准则（爬山法）
#             if repaired_state.objective() < current_state.objective():
#                 current_state = repaired_state
                
#                 # 更新最优解
#                 if current_state.objective() < best_objective:
#                     best_state = current_state.fast_copy()
#                     best_objective = best_state.objective()
#                     print(f"迭代 {iteration}: 发现更优解，成本: {best_objective}")
            
#             iteration += 1
            
#             # 每100次迭代输出一次进度
#             if iteration % 100 == 0:
#                 elapsed_time = time.time() - start_time
#                 print(f"迭代 {iteration}, 当前成本: {current_state.objective()}, 最优成本: {best_objective}, 运行时间: {elapsed_time:.2f}秒")
        
#         elapsed_time = time.time() - start_time
#         statistics = {
#             'iterations': iteration,
#             'runtime': elapsed_time,
#             'best_objective': best_objective
#         }
        
#         print(f"快速ALNS求解完成，最终成本: {best_objective}, 迭代次数: {iteration}, 运行时间: {elapsed_time:.2f}秒")
        
#         return best_state, best_objective, statistics
    
#     def _fast_destroy(self, state):
#         """快速破坏算子：使用增量修改"""
#         destroyed = state.fast_copy()
        
#         # 获取所有客户点
#         all_customers = list(destroyed.customer_plan.keys())
#         if not all_customers:
#             return destroyed
        
#         # 随机移除20%-40%的客户点
#         num_to_remove = self.rng.integers(
#             max(1, len(all_customers) // 5), 
#             max(2, len(all_customers) // 3)
#         )
        
#         customers_to_remove = self.rng.choice(all_customers, num_to_remove, replace=False)
        
#         for customer in customers_to_remove:
#             if customer in destroyed.customer_plan:
#                 assignment = destroyed.customer_plan.pop(customer)
                
#                 # 记录修改，用于可能的回滚
#                 destroyed.record_modification("remove_customer", (customer, assignment))
                
#                 # 从无人机分配中移除相关任务
#                 uav_id, _, _, _, _ = assignment
#                 if uav_id in destroyed.uav_assignments:
#                     destroyed.uav_assignments[uav_id] = [
#                         task for task in destroyed.uav_assignments[uav_id]
#                         if task[1] != customer
#                     ]
        
#         destroyed._total_cost = None
#         return destroyed
    
    # def _fast_repair(self, destroyed_state):
    #     """快速修复算子：使用增量修改"""
    #     repaired = destroyed_state.fast_copy()
        
    #     # 这里简化处理，实际应该实现更复杂的修复策略
    #     # 对于被移除的客户点，可以重新分配到最佳的无人机和车辆组合
        
    #     repaired._total_cost = None
    #     return repaired

    # def multiopt_update_best_scheme(self, best_scheme, near_node_list, vehicle_route, vehicle_task_data, sample_size=30):
    #     """
    #     加速多opt邻域搜索：对near_node_list随机采样sample_size个发射-回收节点组合，
    #     只计算本无人机和同节点相关无人机的成本，贪婪选择最优。
    #     返回(最优方案, 最优总成本)。
    #     """
    #     import random
    #     drone_id, launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
    #     best = best_scheme
    #     best_cost = float('inf')

    #     # 辅助：获取同节点相关无人机
    #     def get_related_drones(vehicle_id, node, task_data):
    #         related = set()
    #         if hasattr(task_data[vehicle_id][node], 'drone_list'):
    #             related.update(task_data[vehicle_id][node].drone_list)
    #         if hasattr(task_data[vehicle_id][node], 'launch_drone_list'):
    #             related.update(task_data[vehicle_id][node].launch_drone_list)
    #         if hasattr(task_data[vehicle_id][node], 'recovery_drone_list'):
    #             related.update(task_data[vehicle_id][node].recovery_drone_list)
    #         return related

    #     # 计算本无人机和同节点相关无人机的总成本
    #     def get_greedy_cost(vehicle_id, l_n, r_n):
    #         total = 0
    #         # 本无人机
    #         total += self.drone_insert_cost(drone_id, customer, l_n, r_n)
    #         # 相关无人机（发射/回收节点）
    #         related = get_related_drones(vehicle_id, l_n, vehicle_task_data) | get_related_drones(vehicle_id, r_n, vehicle_task_data)
    #         related.discard(drone_id)
    #         for d_id in related:
    #             # 查找d_id的发射/回收节点
    #             launch_n, recovery_n = None, None
    #             route = vehicle_route[vehicle_id - 1]
    #             for n2 in route:
    #                 if hasattr(vehicle_task_data[vehicle_id][n2], 'launch_drone_list') and d_id in vehicle_task_data[vehicle_id][n2].launch_drone_list:
    #                     launch_n = n2
    #                 if hasattr(vehicle_task_data[vehicle_id][n2], 'recovery_drone_list') and d_id in vehicle_task_data[vehicle_id][n2].recovery_drone_list:
    #                     recovery_n = n2
    #             if launch_n and recovery_n:
    #                 total += self.drone_insert_cost(d_id, customer, launch_n, recovery_n)
    #         return total

    #     # 单车情况
    #     if launch_vehicle_id == recovery_vehicle_id:
    #         node_list = near_node_list
    #         # 采样sample_size个不同组合
    #         candidates = set()
    #         while len(candidates) < sample_size:
    #             l_n = random.choice(node_list)
    #             r_n = random.choice(node_list)
    #             if l_n != r_n:
    #                 candidates.add((l_n, r_n))
    #         for l_n, r_n in candidates:
    #             cost = get_greedy_cost(launch_vehicle_id, l_n, r_n)
    #             if cost < best_cost:
    #                 best = (drone_id, l_n, customer, r_n, launch_vehicle_id, recovery_vehicle_id)
    #                 best_cost = cost
    #         return best, best_cost
    #     else:
    #         # 异车情况
    #         launch_list = near_node_list[launch_vehicle_id]
    #         recovery_list = near_node_list[recovery_vehicle_id]
    #         candidates = set()
    #         while len(candidates) < sample_size:
    #             l_n = random.choice(launch_list)
    #             r_n = random.choice(recovery_list)
    #             if l_n != r_n:
    #                 candidates.add((l_n, r_n))
    #         for l_n, r_n in candidates:
    #             cost = get_greedy_cost(launch_vehicle_id, l_n, r_n) + get_greedy_cost(recovery_vehicle_id, l_n, r_n)
    #             if cost < best_cost:
    #                 best = (drone_id, l_n, customer, r_n, launch_vehicle_id, recovery_vehicle_id)
    #                 best_cost = cost
    #         return best, best_cost


def find_chain_tasks(assignment, customer_plan, vehicle_routes, vehicle_task_data):
    """
    通过链式找到这个无人机后续的所有服务任务，跟踪无人机任务链直到返回原始发射车辆
    
    Args:
        assignment: 被删除的任务 (drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle)
        customer_plan: 当前客户计划
        vehicle_routes: 车辆路线
        vehicle_task_data: 车辆任务数据
    
    Returns:
        list: 需要删除的任务列表 [(customer, assignment), ...]
    """
    drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle = assignment
    need_to_remove_tasks = []
    
    # 如果发射车辆和回收车辆相同，则无需删除后续任务
    if launch_vehicle == recovery_vehicle:
        # print(f"无人机 {drone_id} 任务为同车任务，无需删除后续任务")
        return need_to_remove_tasks
    
    # print(f"无人机 {drone_id} 任务为异车任务，开始查找后续任务链")
    # print(f"原始发射车辆: {launch_vehicle}, 当前回收车辆: {recovery_vehicle}")
    
    # 使用递归函数跟踪无人机任务链
    def track_drone_chain(current_vehicle, current_node_index, original_launch_vehicle, visited_vehicles=None):
        """
        递归跟踪无人机任务链
        
        Args:
            current_vehicle: 当前车辆ID
            current_node_index: 当前节点在路线中的索引
            original_launch_vehicle: 原始发射车辆ID
            visited_vehicles: 已访问的车辆集合（防止循环）
        """
        if visited_vehicles is None:
            visited_vehicles = set()
        
        # 防止无限循环
        if current_vehicle in visited_vehicles:
            print(f"检测到循环，停止跟踪车辆 {current_vehicle}")
            return
        
        visited_vehicles.add(current_vehicle)
        
        # 获取当前车辆路线
        if current_vehicle - 1 >= len(vehicle_routes):
            print(f"车辆 {current_vehicle} 索引超出范围")
            return
        
        current_route = vehicle_routes[current_vehicle - 1]
        
        # 从当前节点开始遍历后续节点
        for i in range(current_node_index, len(current_route)):
            node = current_route[i]
            
            # 检查该节点是否有该无人机的发射任务
            if (node in vehicle_task_data[current_vehicle] and 
                hasattr(vehicle_task_data[current_vehicle][node], 'launch_drone_list') and 
                drone_id in vehicle_task_data[current_vehicle][node].launch_drone_list):
                
                # print(f"在车辆 {current_vehicle} 的节点 {node} 发现无人机 {drone_id} 的发射任务")
                
                # 查找该发射任务对应的客户点
                for customer_id, customer_assignment in customer_plan.items():
                    cust_drone_id, cust_launch_node, cust_customer, cust_recovery_node, cust_launch_vehicle, cust_recovery_vehicle = customer_assignment
                    
                    # 如果找到匹配的无人机和发射节点
                    if (cust_drone_id == drone_id and cust_launch_node == node and 
                        cust_launch_vehicle == current_vehicle):
                        
                        # print(f"找到需要删除的客户任务: 客户点 {customer_id}, 从车辆 {current_vehicle} 发射到车辆 {cust_recovery_vehicle}")
                        need_to_remove_tasks.append((customer_id, customer_assignment))
                        
                        # 如果回收车辆是原始发射车辆，则停止删除后续任务
                        if cust_recovery_vehicle == original_launch_vehicle:
                            # print(f"客户点 {customer_id} 的回收车辆 {cust_recovery_vehicle} 是原始发射车辆，停止删除后续任务")
                            continue
                        
                        # 如果回收车辆不是原始发射车辆，继续跟踪
                        if cust_launch_vehicle != cust_recovery_vehicle:
                            # print(f"客户点 {customer_id} 的回收车辆 {cust_recovery_vehicle} 不是原始发射车辆，继续跟踪")
                            
                            # 找到回收节点在回收车辆路线中的位置
                            if cust_recovery_vehicle - 1 < len(vehicle_routes):
                                recovery_route = vehicle_routes[cust_recovery_vehicle - 1]
                                recovery_node_index = recovery_route.index(cust_recovery_node) if cust_recovery_node in recovery_route else -1
                                
                                if recovery_node_index != -1:
                                    # 递归跟踪回收车辆的任务链
                                    track_drone_chain(cust_recovery_vehicle, recovery_node_index, original_launch_vehicle, visited_vehicles.copy())
                                else:
                                    print(f"回收节点 {cust_recovery_node} 不在回收车辆 {cust_recovery_vehicle} 的路线中")
                        break
    
    # 开始跟踪任务链
    # 找到回收节点在回收车辆路线中的位置
    recovery_vehicle_index = recovery_vehicle - 1
    if recovery_vehicle_index >= len(vehicle_routes):
        print(f"回收车辆 {recovery_vehicle} 索引超出范围")
        return need_to_remove_tasks
    
    recovery_route = vehicle_routes[recovery_vehicle_index]
    recovery_node_index = recovery_route.index(recovery_node) if recovery_node in recovery_route else -1
    
    if recovery_node_index == -1:
        print(f"回收节点 {recovery_node} 不在回收车辆 {recovery_vehicle} 的路线中")
        return need_to_remove_tasks
    
    # 从回收节点开始跟踪任务链
    track_drone_chain(recovery_vehicle, recovery_node_index, launch_vehicle)
    
    # 去重（避免重复删除）
    unique_tasks = []
    seen_customers = set()
    for customer_id, assignment in need_to_remove_tasks:
        if customer_id not in seen_customers:
            unique_tasks.append((customer_id, assignment))
            seen_customers.add(customer_id)
    
    print(f"无人机 {drone_id} 的链式删除任务总数: {len(unique_tasks)}")
    for customer_id, _ in unique_tasks:
        print(f"  - 客户点 {customer_id}")
    
    return unique_tasks

def is_time_feasible(customer_plan, rm_vehicle_arrive_time):
    """
    简洁的时间约束检查函数：验证无人机任务的发射时间是否小于回收时间
    
    Args:
        customer_plan: 客户计划字典
        rm_vehicle_arrive_time: 车辆到达时间字典
        
    Returns:
        bool: True表示约束满足，False表示约束违反
    """
    for customer_node, plan in customer_plan.items():
        _, launch_node, _, recovery_node, launch_vehicle_id, recovery_vehicle_id = plan
        
        try:
            launch_time = rm_vehicle_arrive_time[launch_vehicle_id][launch_node]
            recovery_time = rm_vehicle_arrive_time[recovery_vehicle_id][recovery_node]
            
            if launch_time >= recovery_time:
                return False
                
        except KeyError:
            return False
            
    return True
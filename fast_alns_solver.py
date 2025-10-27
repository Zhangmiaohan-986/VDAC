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
# from insert_plan import greedy_insert_feasible_plan
import os
from main import find_keys_and_indices
from mfstsp_heuristic_1_partition import *
from mfstsp_heuristic_2_asgn_uavs import *
from mfstsp_heuristic_3_timing import *
from task_data import deep_remove_vehicle_task
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
from create_vehicle_route import DiverseRouteGenerator
from constraint_validator import validate_state_constraints, quick_validate

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
    
    def validate_constraints(self, verbose=True):
        """
        验证当前状态的约束条件
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            dict: 验证结果
        """
        return validate_state_constraints(self, verbose)
    
    def is_constraints_satisfied(self):
        """
        快速检查约束是否满足
        
        Returns:
            bool: True表示约束满足，False表示违反
        """
        return quick_validate(self)

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
            vehicle_routes=[route.copy() for route in self.vehicle_routes],
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
        # self.max_iterations = max_iterations
        self.max_iterations = 500

        self.max_runtime = max_runtime
        self.rng = rnd.default_rng(42)
        self.vtp_coords = np.array([self.node[i].position for i in self.A_vtp])
        self.num_clusters = min(len(self.T), len(self.A_vtp))
        self.dis_k = 25  # 修改距离客户点最近的vtp节点集合，增加解空间
        self.base_drone_assignment = self.base_drone_assigment()
        # self.base_vehicle_task_data = DiverseRouteGenerator.create_vehicle_task_data(self.node, self.DEPOT_nodeID, self.V, self.T, self.vehicle, self.uav_travel, self.veh_distance, self.veh_travel, self.N, self.N_zero, self.N_plus, self.A_total, self.A_cvtp, self.A_vtp, self.A_aerial_relay_node, self.G_air, self.G_ground, self.air_matrix, self.ground_matrix, self.air_node_types, self.ground_node_types, self.A_c, self.xeee)
        # 破坏算子参数
        self.customer_destroy_ratio = (0.2, 0.4)
        self.vtp_destroy_quantity = {'random': (1, 2), 'worst': 1, 'shaw': 2}
        self.cluster_vtp_dict, self.map_cluster_vtp_dict = self.cluster_vtp_for_customers(k=self.dis_k)
        # 定义算子池，方便后续引用
        self.destroy_operators = [self.destroy_random_removal, self.destroy_worst_removal, self.destroy_comprehensive_removal,self.destroy_shaw_rebalance_removal]
        # self.destroy_operators = [self.destroy_random_removal, self.destroy_worst_removal]
        # self.destroy_operators = [self.destroy_shaw_rebalance_removal]

        # self.destroy_operators = [self.destroy_random_removal]

        self.repair_operators = [self.repair_greedy_insertion, self.repair_regret_insertion]
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
        
        # 关键修复：必须创建状态副本，避免修改原始状态
        repaired_state = state.fast_copy()  # 修复：创建真正的副本
        repaired_state.repair_objective = 0
        destroy_node = list(state.destroyed_customers_info.keys())  # 总结出了所有的待插入的破坏节点
        insert_plan = []  # 记录所有破坏节点的最优插入方案

        force_vtp_mode = True
        if force_vtp_mode:
            num_repaired = 0
            while len(destroy_node) > 0:
                best_option_overall = None
                best_customer_to_insert = None
                min_overall_eval_cost = float('inf')
                # a. 计算本轮决策的"最终奖励"(final_bonus)
                tactical_multiplier = (num_destroyed - num_repaired) / num_destroyed
                final_bonus = strategic_bonus * tactical_multiplier * 0.3
                final_bonus = 0
                
                # 获取当前状态的数据
                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                # vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                
                # 存储所有候选方案
                all_candidates = []
                customer_candidates = []
                # 遍历所有待插入客户点，计算每个节点的最优插入成本
                for customer in destroy_node:
                    
                    # 1. 首先尝试传统插入方案（使用现有节点）
                    traditional_result,is_heuristic_swap = self._evaluate_traditional_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
                    if traditional_result is not None:
                        traditional_cost, traditional_scheme = traditional_result
                        if is_heuristic_swap:
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': traditional_scheme,
                                'cost': traditional_cost,
                                'type': 'heuristic_swap',
                                'vtp_node': None
                            })
                        else:
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
                customer_candidates = [item for item in customer_candidates if item['scheme'] is not None]
                # 对customer_candidates的cost由小到大排序
                candidates_plan = sorted(customer_candidates, key=lambda x: x['cost'])
                
                # 尝试每个候选方案，直到找到满足约束的方案
                success = False

                for candidate in candidates_plan:
                    customer = candidate['customer']
                    # best_scheme = candidate['scheme']
                    # best_cost = candidate['cost']
                    
                    # 根据方案类型执行不同的插入逻辑
                    if candidate['type'] == 'traditional':
                        # print(f"尝试使用传统方案插入客户点 {customer}，成本: {best_cost:.2f}")
                        
                        customer = candidate['customer']
                        best_scheme = candidate['scheme']
                        best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
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
                    # 考虑到启发式的交换策略，因此需要重新设计一种模式来处理其插入方案
                    elif candidate['type'] == 'heuristic_swap':
                        print(f"尝试使用启发式交换方案插入客户点 {customer}，成本: {best_cost:.2f}")
                        # 使用启发式交换方案 - 采用统一的后续处理方式
                        orig_scheme = candidate['scheme']['orig_scheme']
                        new_scheme = candidate['scheme']['new_scheme']
                        orig_cost = candidate['scheme']['orig_cost']
                        new_cost = candidate['scheme']['new_cost']
                        orig_plan = candidate['scheme']['orig_plan']
                        new_plan = candidate['scheme']['new_plan']
                        # delete_customer = candidate['customer']
                        orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = orig_scheme
                        new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_scheme
                        customer = new_customer
                        delete_customer = orig_customer
                        # delete_task_plan = state.customer_plan[orig_customer]
                        # 创建临时状态进行约束检查
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        delete_task_plan = temp_customer_plan[orig_customer]
                        del temp_customer_plan[orig_customer]
                        temp_customer_plan[orig_customer] = orig_scheme
                        temp_customer_plan[new_customer] = new_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                            print(f"启发式交换方案时间约束不满足，尝试下一个候选方案")
                            continue  
                        else:
                            # 更新customer_plan
                            del repaired_state.customer_plan[delete_customer]
                            repaired_state.customer_plan[orig_customer] = orig_scheme
                            repaired_state.customer_plan[new_customer] = new_scheme
                            # 更新uav_assignments
                            if orig_drone_id not in repaired_state.uav_assignments:
                                repaired_state.uav_assignments[orig_drone_id] = []
                            repaired_state.uav_assignments[orig_drone_id].append(orig_scheme)
                            if new_drone_id not in repaired_state.uav_assignments:
                                repaired_state.uav_assignments[new_drone_id] = []
                            repaired_state.uav_assignments[new_drone_id].append(new_scheme)
                            # 更新uav_cost
                            del repaired_state.uav_cost[delete_customer]
                            repaired_state.uav_cost[orig_customer] = orig_cost
                            repaired_state.uav_cost[new_customer] = new_cost
                            # 更新vehicle_task_data
                            vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                            orig_launch_time = temp_rm_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                            new_launch_time = temp_rm_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                            if orig_launch_time <= new_launch_time:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                            else:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                            # vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                            # vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                            # vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                            # 记录插入方案
                            insert_plan.append((delete_customer, orig_scheme, orig_cost, 'heuristic_swap'))
                            insert_plan.append((customer, new_scheme, new_cost, 'heuristic_swap'))
                            success = True
                            break
                    # 开始执行VTP扩展插入方案
                    elif candidate['type'] == 'vtp_expansion':
                        # VTP扩展插入方案 - 采用统一的后续处理方式，并额外更新车辆路线
                        # print(f"尝试使用VTP扩展方案插入客户点 {customer}，成本: {best_cost:.2f}")
                        customer = candidate['customer']
                        vtp_node = candidate['vtp_node']
                        vtp_insert_index = candidate['vtp_insert_index']
                        vtp_insert_vehicle_id = candidate['vtp_insert_vehicle_id']
                        best_scheme = candidate['scheme']
                        # best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
                        # original_cost = candidate['original_cost']
                    
                        # 1. 首先将VTP节点插入到车辆路径中
                        # 从方案中提取车辆ID和插入位置
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        original_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)

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
                            # print(f"成功新增VTP节点 {vtp_node} 并插入客户点 {customer}，总成本: {original_cost:.2f}")
                            success = True
                            break
                
                # 如果所有候选方案都不满足约束，跳过当前客户点
                if not success:
                    print(f"客户点 {customer} 的所有候选方案都不满足约束，跳过")
                    repaired_state.repair_objective = float('inf')
                    return repaired_state, insert_plan
                    # continue
                
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
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                
                # 存储直接插入和启发式交换的候选方案
                direct_insertion_candidates = []  # 直接插入候选方案
                heuristic_swap_candidates = []    # 启发式交换候选方案
                
                # 遍历所有待插入客户点，计算每个节点的最优插入成本
                for customer in destroy_node:
                    min_cost = float('inf')
                    best_scheme = None
                    greedy_total_cost = float('inf')
                    greedy_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
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
                                    if customer_node in greedy_customer_plan:
                                        del greedy_customer_plan[customer_node]
                                    greedy_customer_plan[customer_node] = best_scheme
                                    greedy_total_cost = sum(repaired_state.uav_cost.values())
                    if best_scheme is not None:
                    # 直接插入模式：记录直接插入方案
                        direct_insertion_candidates.append({
                            'customer': customer,
                            'scheme': best_scheme,
                            'cost': min_cost,
                            'total_cost': greedy_total_cost,
                            'type': 'direct'
                        })
                        print(f"客户点 {customer} 直接插入成本: {min_cost:.2f}")
                    else:
                        # 启发式交换模式：尝试交换策略（增加成本阈值检查）
                        print(f"客户点 {customer} 没有直接插入方案，尝试启发式交换策略")
                        
                        # 计算当前解的总成本作为基准
                        current_total_cost = sum(repaired_state.uav_cost.values()) if repaired_state.uav_cost else 0
                        cost_improvement_threshold = 0.1  # 只有当交换能带来10%以上的成本改善时才进行
                        # 初始化类
                        generator = DiverseRouteGenerator(self.node, self.DEPOT_nodeID, self.A_vtp, self.V, self.T, self.vehicle, self.uav_travel, self.veh_distance, self.veh_travel, self.vtp_coords, self.num_clusters, self.G_air, self.G_ground, self.air_matrix, self.ground_matrix, self.air_node_types, self.ground_node_types, self.A_c, self.xeee)
                        
                        try:
                            # 通过启发式的贪婪算法插入方案（交换策略）
                            best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = generator.greedy_insert_feasible_plan(
                                customer, vehicle_route, vehicle_arrive_time, vehicle_task_data, repaired_state.customer_plan
                            )
                            
                            orig_scheme = (best_orig_y[0], best_orig_y[1], best_orig_y[2], best_orig_y[3], best_orig_y[4], best_orig_y[5])
                            new_scheme = (best_new_y[0], best_new_y[1], best_new_y[2], best_new_y[3], best_new_y[4], best_new_y[5])
                            if best_orig_y[2] == customer:
                                delete_customer = best_new_y[2]
                            else:
                                delete_customer = best_orig_y[2]
                            # 创建临时状态进行约束检查
                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            delete_task_plan = temp_customer_plan[delete_customer]
                            del temp_customer_plan[delete_customer]
                            temp_customer_plan[best_orig_y[2]] = orig_scheme
                            temp_customer_plan[best_new_y[2]] = new_scheme
                            temp_customer_cost = {k: v for k, v in repaired_state.uav_cost.items()}
                            del temp_customer_cost[delete_customer]
                            temp_customer_cost[best_orig_y[2]] = best_orig_cost
                            temp_customer_cost[best_new_y[2]] = best_new_cost
                            temp_cost = sum(temp_customer_cost.values())
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                            current_cost = sum(repaired_state.uav_cost.values())
                            heuristic_swap_candidates.append({
                                'customer': customer,
                                'orig_scheme': best_orig_y,
                                'new_scheme': best_new_y,
                                'orig_cost': best_orig_cost,
                                'new_cost': best_new_cost,
                                'total_cost': temp_cost,
                                'cost': best_orig_cost + best_new_cost,
                                'orig_plan': best_orig_y_cijkdu_plan,
                                'new_plan': best_new_y_cijkdu_plan,
                                'type': 'heuristic_swap'
                            })
                            
                        except Exception as e:
                            print(f"客户点 {customer} 启发式交换失败: {e}")
                            print(f"回退到未被破坏的状态，跳过客户点 {customer}")
                            # 当启发式交换失败时，跳过当前客户点，继续处理其他客户点
                            print(f'启发式无法找到最优解方案，目标函数值设置为无穷大返回')
                            repaired_state.repair_objective = float('inf')
                            return repaired_state, insert_plan
                
                # 选择最优插入方案
                all_candidates = direct_insertion_candidates + heuristic_swap_candidates
                # 删除all_candidates中cost为inf或None的候选解
                all_candidates = [c for c in all_candidates if c.get('cost') is not None and not math.isinf(c.get('cost', 0))]
                
                if not all_candidates:
                    print("所有剩余节点都没有可行插入方案，修复终止")
                    return repaired_state, insert_plan
                best_candidate = min(all_candidates, key=lambda x: x['total_cost'])
                
                # 根据方案类型执行不同的插入逻辑
                if best_candidate['type'] == 'direct':
                    # 直接插入模式
                    customer = best_candidate['customer']
                    best_scheme = best_candidate['scheme']
                    best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
                    
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
                    # if best_new_y_cijkdu_plan['launch_time'] < best_orig_y_cijkdu_plan['launch_time']:
                    # 插入新方案
                    repaired_state.customer_plan[new_customer] = best_new_y
                    if repaired_state.uav_cost is None:
                        repaired_state.uav_cost = {}
                    repaired_state.uav_cost[new_customer] = best_new_cost
                    
                    # 更新无人机分配
                    if new_drone_id not in repaired_state.uav_assignments:
                        repaired_state.uav_assignments[new_drone_id] = []
                    repaired_state.uav_assignments[new_drone_id].append(best_new_y)
                    
                    # vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                    # final_scheme = best_new_y
                    # final_cost = best_new_cost

                    repaired_state.customer_plan[remove_customer] = best_orig_y
                    if repaired_state.uav_cost is None:
                        repaired_state.uav_cost = {}
                    repaired_state.uav_cost[remove_customer] = best_orig_cost
                    if orig_drone_id not in repaired_state.uav_assignments:
                        repaired_state.uav_assignments[orig_drone_id] = []
                    repaired_state.uav_assignments[orig_drone_id].append(best_orig_y)
                    # vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                    # vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)
                    orig_launch_time = repaired_state.rm_empty_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                    new_launch_time = repaired_state.rm_empty_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                    if orig_launch_time <= new_launch_time:
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                    else:
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)

                    final_scheme = (best_orig_y,best_new_y)
                    final_cost = best_orig_cost + best_new_cost
                    
                    # 记录插入方案
                    insert_plan.append((customer, final_scheme, final_cost))
                    if customer in destroy_node:  # 安全检查，避免重复删除
                        destroy_node.remove(customer)
                    
                    print(f"成功启发式交换插入客户点 {customer}，最终成本: {final_cost:.2f}")
            
        # 更新修复完成后的成本
        repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        
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

    def repair_regret_insertion(self, state, strategic_bonus=0, num_destroyed=1, force_vtp_mode=False):
        """
        与贪婪修复保持相同框架（含VTP扩展与统一约束检查），但选择策略改为后悔值：
        对每个待插入客户，计算其候选方案中(次优成本 - 最优成本)作为后悔值，优先插入后悔值最大的客户，
        并在其候选方案中按成本从低到高依次尝试，直到满足约束。
        """
        repaired_state = state.fast_copy()
        repaired_state.repair_objective = 0
        destroy_node = list(state.destroyed_customers_info.keys())
        insert_plan = []
        force_vtp_mode = True
        if force_vtp_mode:
            num_repaired = 0
            while len(destroy_node) > 0:
                # 计算当轮bonus（与贪婪框架一致，但不改变策略，仅保留变量结构）
                tactical_multiplier = (num_destroyed - num_repaired) / max(num_destroyed, 1)
                final_bonus = strategic_bonus * tactical_multiplier * 0.3
                final_bonus = 0

                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                # 为每个客户构建候选集合并计算后悔值
                per_customer_candidates = {}
                regret_list = []

                for customer in destroy_node:
                    candidates = []

                    # 1) 传统插入
                    traditional_result, is_heuristic_swap = self._regret_evaluate_traditional_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
                    candidates.extend(traditional_result)
                    # if traditional_result is not None:
                    #     traditional_cost, traditional_scheme = traditional_result
                    #     if is_heuristic_swap:
                    #         candidates.append({
                    #             'customer': customer,
                    #             'scheme': traditional_scheme,
                    #             'cost': traditional_cost,
                    #             'type': 'heuristic_swap',
                    #             'vtp_node': None
                    #         })
                    #     else:
                    #         candidates.append({
                    #             'customer': customer,
                    #             'scheme': traditional_scheme,
                    #             'cost': traditional_cost,
                    #             'type': 'traditional',
                    #             'vtp_node': None
                    #         })
                    # 2) VTP扩展插入
                    total_options = self._regret_evaluate_vtp_expansion_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
                    candidates.extend(total_options)
                    # if vtp_infor is not None:
                    #     vtp_node = vtp_infor[0]
                    #     vtp_insert_vehicle_id = vtp_infor[1]
                    #     vtp_insert_index = vtp_infor[2]
                    #     if vtp_result is not None:
                    #         vtp_cost, vtp_scheme = vtp_result
                    #         adjusted_cost = vtp_cost - final_bonus
                    #         candidates.append({
                    #             'customer': customer,
                    #             'scheme': vtp_scheme,
                    #             'cost': adjusted_cost,
                    #             'type': 'vtp_expansion',
                    #             'vtp_node': vtp_node,
                    #             'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                    #             'vtp_insert_index': vtp_insert_index,
                    #             'original_cost': vtp_cost
                    #         })

                    # 计算后悔值
                    if len(candidates) == 0:
                        print(f'在regret的修复策略中，客户点{customer}没有可行的插入方案，包括传统插入和VTP扩展插入,跳过')
                        continue
                    # # 删除候选解中eval_cost数值为inf的内容
                    # import math
                    # # 过滤掉eval_cost为inf或None的候选解
                    # candidates = [c for c in candidates if c.get('eval_cost') is not None and not math.isinf(c.get('eval_cost', 0))]
                    
                    candidates_sorted = sorted(candidates, key=lambda x: x['eval_cost'])
                    best_cost = candidates_sorted[0]['eval_cost']
                    best_type = candidates_sorted[0]['type']
                    second_best_cost = candidates_sorted[1]['eval_cost'] if len(candidates_sorted) >= 2 else best_cost
                    second_best_type = candidates_sorted[1]['type'] if len(candidates_sorted) >= 2 else None
                    if second_best_type == None:
                        regret_value = 0
                    elif best_type and second_best_type == 'vtp_expansion':
                        regret_value = second_best_cost - best_cost
                    elif best_type and second_best_type == 'traditional':
                        regret_value = best_cost - second_best_cost
                    elif best_type == 'vtp_expansion' and second_best_type == 'traditional' or best_type == 'traditional' and second_best_type == 'vtp_expansion':
                        regret_value = second_best_cost - best_cost
                    elif best_type == 'heuristic_swap' or second_best_type == 'heuristic_swap':
                        regret_value = candidates_sorted[1]['total_cost'] - candidates_sorted[0]['total_cost']
                    else:
                        regret_value = second_best_cost - best_cost
                    # regret_value = second_best_cost - best_cost
                    per_customer_candidates[customer] = candidates_sorted
                    regret_list.append({'customer': customer, 'regret': regret_value, 'best_cost': best_cost, 'best_type': best_type, 'second_best_cost': second_best_cost, 'second_best_type': second_best_type})

                if not regret_list:
                    # 无任何客户可行
                    break

                # 选择后悔值最大的客户（若相同则选择最小best_cost）
                regret_list.sort(key=lambda x: (-x['regret'], x['best_cost']))

                success_any = False
                for entry in regret_list:
                    customer = entry['customer']
                    candidates_sorted = per_customer_candidates[customer]
                    candidates_sorted = [item for item in candidates_sorted if item.get('scheme') is not None]
                    # 依次尝试候选方案，直到满足约束
                    for candidate in candidates_sorted:
                        # best_scheme = candidate['scheme']
                        # best_cost = candidate['eval_cost']

                        if candidate['type'] == 'traditional':
                            # 约束检查
                            best_scheme = candidate['scheme']
                            customer = best_scheme[2]
                            best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            temp_customer_plan[best_scheme[2]] = best_scheme
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                            if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                                continue

                            drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme

                            repaired_state.customer_plan[customer_node] = best_scheme
                            if drone_id not in repaired_state.uav_assignments:
                                repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(best_scheme)
                            if repaired_state.uav_cost is None:
                                repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = best_cost
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                            insert_plan.append((customer, best_scheme, best_cost, 'traditional'))
                            success_any = True
                            break
                        elif candidate['type'] == 'heuristic_swap':
                            print(f"尝试使用启发式交换方案插入客户点 {customer}，成本: {best_cost:.2f}")
                            # 使用启发式交换方案 - 采用统一的后续处理方式
                            orig_scheme = candidate['orig_scheme']
                            new_scheme = candidate['new_scheme']
                            orig_cost = candidate['orig_cost']
                            new_cost = candidate['new_cost']
                            # orig_plan = candidate['scheme']['orig_plan']
                            # new_plan = candidate['scheme']['new_plan']
                            # delete_customer = candidate['customer']
                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = orig_scheme
                            new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_scheme
                            customer = orig_customer
                            delete_task_plan = temp_customer_plan[orig_customer]
                            delete_customer = orig_customer
                            # 创建临时状态进行约束检查
                            del temp_customer_plan[delete_customer]
                            temp_customer_plan[orig_customer] = orig_scheme
                            temp_customer_plan[new_customer] = new_scheme
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                            if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                                print(f"启发式交换方案时间约束不满足，尝试下一个候选方案")
                                continue  
                            else:
                                # 更新customer_plan
                                del repaired_state.customer_plan[delete_customer]
                                repaired_state.customer_plan[orig_customer] = orig_scheme
                                repaired_state.customer_plan[new_customer] = new_scheme
                                # 更新uav_assignments
                                if orig_drone_id not in repaired_state.uav_assignments:
                                    repaired_state.uav_assignments[orig_drone_id] = []
                                repaired_state.uav_assignments[orig_drone_id].append(orig_scheme)
                                if new_drone_id not in repaired_state.uav_assignments:
                                    repaired_state.uav_assignments[new_drone_id] = []
                                repaired_state.uav_assignments[new_drone_id].append(new_scheme)
                                # 更新uav_cost
                                del repaired_state.uav_cost[delete_customer]
                                repaired_state.uav_cost[orig_customer] = orig_cost
                                repaired_state.uav_cost[new_customer] = new_cost
                                # 更新vehicle_task_data
                                vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                                orig_launch_time = repaired_state.rm_empty_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                                new_launch_time = repaired_state.rm_empty_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                                if orig_launch_time <= new_launch_time:
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                else:
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                # vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                # vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                # 记录插入方案
                                insert_plan.append((delete_customer, orig_scheme, orig_cost, 'heuristic_swap'))
                                insert_plan.append((customer, new_scheme, new_cost, 'heuristic_swap'))
                                success_any = True
                                break
                        elif candidate['type'] == 'vtp_expansion':
                            customer = candidate['customer']
                            vtp_node = candidate['vtp_node']
                            vtp_insert_index = candidate['vtp_insert_index']
                            vtp_insert_vehicle_id = candidate['vtp_insert_vehicle_id']
                            drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = candidate['scheme']
                            original_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)

                            # 临时状态检查
                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            temp_customer_plan[customer_node] = candidate['scheme']
                            temp_vehicle_route = [route[:] for route in vehicle_route]
                            temp_route = temp_vehicle_route[vtp_insert_vehicle_id - 1]
                            temp_route.insert(vtp_insert_index, vtp_node)
                            temp_vehicle_route[vtp_insert_vehicle_id - 1] = temp_route
                            repaired_state.temp_vehicle_routes = temp_vehicle_route
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                            if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                                continue

                            # 执行插入
                            route = vehicle_route[vtp_insert_vehicle_id - 1]
                            route.insert(vtp_insert_index, vtp_node)

                            last_customer_node = route[vtp_insert_index - 1]
                            if vtp_insert_index == 1 or last_customer_node == self.DEPOT_nodeID:
                                last_drone_list = self.base_drone_assignment[vtp_insert_vehicle_id][:]
                            else:
                                last_drone_list = vehicle_task_data[vtp_insert_vehicle_id][last_customer_node].drone_list[:]
                            vehicle_task_data[vtp_insert_vehicle_id][vtp_node].drone_list = last_drone_list
                            vehicle_task_data[vtp_insert_vehicle_id][vtp_node].launch_drone_list = []
                            vehicle_task_data[vtp_insert_vehicle_id][vtp_node].recovery_drone_list = []

                            vehicle_task_data = update_vehicle_task(vehicle_task_data, candidate['scheme'], vehicle_route)

                            repaired_state.customer_plan[customer_node] = candidate['scheme']
                            if drone_id not in repaired_state.uav_assignments:
                                repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(candidate['scheme'])
                            if repaired_state.uav_cost is None:
                                repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = original_cost
                            repaired_state.rm_empty_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
                            repaired_state.rm_empty_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                            insert_plan.append((customer, candidate['scheme'], original_cost, 'vtp_expansion'))
                            success_any = True
                            break

                    if success_any:
                        if customer in destroy_node:
                            destroy_node.remove(customer)
                        num_repaired += 1
                        break
                    else:
                        print(f'在regret的修复策略中，客户点{customer}没有可行的插入方案，跳过，插入方案失败')
                        repaired_state.repair_objective = float('inf')
                        return repaired_state, insert_plan

                if not success_any:
                    # 本轮没有任何可行插入，直接终止
                    break
        else:
            # 非VTP强制模式：保持与贪婪分支同构，但采用后悔值选择客户
            while len(destroy_node) > 0:
                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                per_customer_info = []
                direct_buckets = {}
                swap_buckets = {}

                for customer in destroy_node:
                    # 直接插入候选（多位置）
                    all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
                    direct_candidates = []
                    if all_insert_position is not None:
                        for drone_id, inert_positions in all_insert_position.items():
                            for inert_position in inert_positions:
                                launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                                insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                                direct_candidates.append({
                                    'customer': customer,
                                    'scheme': (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id),
                                    'cost': insert_cost,
                                    'type': 'direct'
                                })
                    direct_candidates.sort(key=lambda x: x['cost'])
                    direct_buckets[customer] = direct_candidates

                    if len(direct_candidates) == 0:
                        # 尝试启发式交换
                        try:
                            generator = DiverseRouteGenerator(self.node, self.DEPOT_nodeID, self.A_vtp, self.V, self.T, self.vehicle, self.uav_travel, self.veh_distance, self.veh_travel, self.vtp_coords, self.num_clusters, self.G_air, self.G_ground, self.air_matrix, self.ground_matrix, self.air_node_types, self.ground_node_types, self.A_c, self.xeee)
                            best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = generator.greedy_insert_feasible_plan(
                                customer, vehicle_route, vehicle_arrive_time, vehicle_task_data, repaired_state.customer_plan
                            )
                            orig_scheme = (best_orig_y[0], best_orig_y[1], best_orig_y[2], best_orig_y[3], best_orig_y[4], best_orig_y[5])
                            new_scheme = (best_new_y[0], best_new_y[1], best_new_y[2], best_new_y[3], best_new_y[4], best_new_y[5])
                            if best_orig_y[2] == customer:
                                delete_customer = best_new_y[2]
                            else:
                                delete_customer = best_orig_y[2]
                            # 创建临时状态进行约束检查
                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            delete_task_plan = temp_customer_plan[delete_customer]
                            del temp_customer_plan[delete_customer]
                            temp_customer_plan[best_orig_y[2]] = orig_scheme
                            temp_customer_plan[best_new_y[2]] = new_scheme
                            temp_customer_cost = {k: v for k, v in repaired_state.uav_cost.items()}
                            del temp_customer_cost[delete_customer]
                            temp_customer_cost[best_orig_y[2]] = best_orig_cost
                            temp_customer_cost[best_new_y[2]] = best_new_cost
                            temp_cost = sum(temp_customer_cost.values())
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                            current_cost = sum(repaired_state.uav_cost.values())
                            if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                                print(f"启发式交换方案时间约束不满足，尝试下一个候选方案")
                                continue  
                            swap_buckets[customer] = {
                                    'customer': customer,
                                    'orig_scheme': best_orig_y,
                                    'new_scheme': best_new_y,
                                    'orig_cost': best_orig_cost,
                                    'new_cost': best_new_cost,
                                    'total_cost': best_orig_cost + best_new_cost,
                                    'orig_plan': best_orig_y_cijkdu_plan,
                                    'new_plan': best_new_y_cijkdu_plan,
                                    'type': 'heuristic_swap'
                                }
                            # if best_orig_y is not None and best_new_y is not None:
                            #     swap_buckets[customer] = {
                            #         'customer': customer,
                            #         'orig_scheme': best_orig_y,
                            #         'new_scheme': best_new_y,
                            #         'orig_cost': best_orig_cost,
                            #         'new_cost': best_new_cost,
                            #         'total_cost': best_orig_cost + best_new_cost,
                            #         'orig_plan': best_orig_y_cijkdu_plan,
                            #         'new_plan': best_new_y_cijkdu_plan,
                            #         'type': 'heuristic_swap'
                            #     }
                        except Exception as e:
                            print(f"客户点 {customer} 启发式交换失败: {e}")
                            print(f"回退到未被破坏的状态，跳过客户点 {customer}")
                            # 当启发式交换失败时，跳过当前客户点，继续处理其他客户点
                            print(f'启发式无法找到最优解方案，目标函数值设置为无穷大返回')
                            repaired_state.repair_objective = float('inf')
                            return repaired_state, insert_plan

                    # 计算后悔值
                    if len(direct_candidates) >= 1:
                        best_cost = direct_candidates[0]['cost']
                        # second_best_cost = direct_candidates[1]['cost'] if len(direct_candidates) >= 2 else float('inf')
                        second_best_cost = direct_candidates[1]['cost'] if len(direct_candidates) >= 2 else best_cost  # 如果只有一种方案，则后悔值为0
                        regret_value = second_best_cost - best_cost
                        per_customer_info.append({'customer': customer, 'mode': 'direct', 'regret': regret_value, 'best_cost': best_cost})
                    elif customer in swap_buckets:
                        # 没有直接候选但有交换候选，视为高后悔值以优先考虑,设置后悔值为0，之前为float('inf')
                        per_customer_info.append({'customer': customer, 'mode': 'heuristic_swap', 'regret': 0, 'best_cost': swap_buckets[customer]['total_cost']})

                if not per_customer_info:
                    print(f'没有可行的插入方案，目标函数值设置为无穷大返回')
                    repaired_state.repair_objective = float('inf')
                    return repaired_state, insert_plan
                    # break

                # 选择后悔值最大（若相同选择best_cost较小者）
                per_customer_info.sort(key=lambda x: (-x['regret'], x['best_cost']))

                inserted = False
                for info in per_customer_info:
                    customer = info['customer']
                    if info['mode'] == 'direct':
                        for cand in direct_buckets[customer]:
                            best_scheme = cand['scheme']
                            # best_cost = cand['cost']
                            best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])

                            # 直接应用（该分支与贪婪一致，无需额外时间可行性检查函数，这里直接更新）
                            drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                            repaired_state.customer_plan[customer_node] = best_scheme
                            if drone_id not in repaired_state.uav_assignments:
                                repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(best_scheme)
                            if repaired_state.uav_cost is None:
                                repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = best_cost
                            repaired_state.vehicle_task_data = update_vehicle_task(repaired_state.vehicle_task_data, best_scheme, vehicle_route)
                            insert_plan.append((customer, best_scheme, best_cost))
                            if customer in destroy_node:
                                destroy_node.remove(customer)
                            inserted = True
                            break
                        if inserted:
                            break
                    else:
                        # 启发式交换
                        swap  = swap_buckets.get(customer)
                        if swap is None:
                            continue
                        # 启发式交换模式
                        customer = swap['customer']
                        best_orig_y = swap['orig_scheme']
                        best_new_y = swap['new_scheme']
                        best_orig_cost = swap['orig_cost']
                        best_new_cost = swap['new_cost']
                        best_orig_y_cijkdu_plan = swap['orig_plan']
                        best_new_y_cijkdu_plan = swap['new_plan']
                        orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = best_orig_y
                        new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = best_new_y

                        if orig_customer == customer:
                            remove_customer = new_customer
                        else:
                            remove_customer = orig_customer
                        # 创建临时状态进行约束检查
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        delete_task_plan = temp_customer_plan[remove_customer]
                        del temp_customer_plan[remove_customer]
                        temp_customer_plan[orig_customer] = best_orig_y
                        temp_customer_plan[new_customer] = best_new_y
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        # if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                        #     print(f"启发式交换方案时间约束不满足，尝试下一个候选方案")
                        #     continue  
                        # else:
                        # 更新customer_plan
                        # delete_task_plan = repaired_state.customer_plan[remove_customer]
                        del repaired_state.customer_plan[remove_customer]
                        repaired_state.customer_plan[orig_customer] = best_orig_y

                        repaired_state.customer_plan[new_customer] = best_new_y
                        # 更新uav_assignments
                        if orig_drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[orig_drone_id] = []
                        repaired_state.uav_assignments[orig_drone_id].append(best_orig_y)
                        if new_drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[new_drone_id] = []
                        repaired_state.uav_assignments[new_drone_id].append(best_new_y)
                        # 更新uav_cost
                        del repaired_state.uav_cost[remove_customer]
                        repaired_state.uav_cost[orig_customer] = best_orig_cost
                        repaired_state.uav_cost[new_customer] = best_new_cost
                        # 更新vehicle_task_data
                        vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                        # 按照顺序来对应的插入方案，不然会违背约束条件
                        orig_launch_time = temp_rm_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                        new_launch_time = temp_rm_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                        if orig_launch_time <= new_launch_time:
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                        else:
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)
                        # vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                        
                        final_scheme = (best_orig_y,best_new_y)
                        final_cost = best_orig_cost + best_new_cost
                        
                        # 记录插入方案
                        insert_plan.append((customer, final_scheme, final_cost))
                        if customer in destroy_node:  # 安全检查，避免重复删除
                            destroy_node.remove(customer)
                        inserted = True
                        break

        repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        return repaired_state, insert_plan

    def repair_k_insertion(self, state):
        """
        快速K步插入修复算子：使用采样和启发式方法提高性能
        策略：采样少量K步序列，选择最优的插入方案
        """
        # repaired_state = state
        repaired_state = state.fast_copy()
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
        # repaired_state = state
        repaired_state = state.fast_copy()
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
                    destroy_node, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state, 0
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
        base_cost = 0
        
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

    def _regret_evaluate_traditional_insertion(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state=None):
        """
        评估将 customer 插入到现有VTP网络的所有【直接插入】方案，
        并【尝试】进行启发式交换，将所有评估结果以 options 列表形式返回。

        Args:
            customer (int): 待评估的客户ID。
            vehicle_route (list): 当前车辆路线列表。
            vehicle_task_data (dict): 当前车辆任务数据。
            vehicle_arrive_time (dict): 当前车辆到达时间。
            repaired_state (FastMfstspState, optional): 当前修复中的状态，用于启发式交换。

        Returns:
            list: 一个包含方案字典的列表。每个字典包含:
                {'eval_cost': float, 'real_cost': float, 'plan': tuple or dict, 
                'type': str ('traditional' or 'heuristic_swap'), 'extra_info': None}
                如果没有任何可行方案，则返回空列表。
        """
        options = []
        is_heuristic_swap = False

        # ----------------------------------------------------------------------
        # 1. 评估所有【直接插入】方案 (利用 get_all_insert_position)
        # ----------------------------------------------------------------------
        try:
            all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
            
            if all_insert_position:
                for drone_id, inert_positions in all_insert_position.items():
                    for inert_position in inert_positions:
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_cost = {k: v for k, v in repaired_state.uav_cost.items()}
                        launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                        if customer_node in temp_customer_plan:
                            del temp_customer_plan[customer_node]
                        temp_customer_plan[customer_node] = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                        if customer_node in temp_customer_cost:
                            del temp_customer_cost[customer_node]
                        temp_customer_cost[customer_node] = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        temp_total_cost = sum(temp_customer_cost.values())
                        # a. 计算成本
                        real_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        
                        # b. 【重要】在此处加入时间可行性等约束检查
                        #    您需要一个 is_time_feasible 函数来验证这个方案是否可行
                        #    plan_to_check = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                        #    if is_time_feasible(plan_to_check, vehicle_arrive_time): # 假设需要 arrive_time
                        
                        if real_cost is not None: # 假设 drone_insert_cost 在不可行时返回 None
                            plan = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                            
                            # eval_cost 等于 real_cost，因为传统方案没有奖励
                            options.append({
                                'customer': customer,
                                'eval_cost': real_cost, 
                                'real_cost': real_cost,
                                'total_cost': temp_total_cost,
                                'scheme': plan, 
                                'type': 'traditional', 
                                'extra_info': None
                            })

        except Exception as e:
            print(f"  > 警告: 在评估客户 {customer} 的直接插入方案时发生错误: {e}")

        # ----------------------------------------------------------------------
        # 2. 【尝试】评估【启发式交换】方案 (如果直接插入方案较少或不存在)
        # ----------------------------------------------------------------------
        # 您可以设定一个阈值，例如，只有当直接插入方案少于 N 个时才尝试代价更高的启发式交换
        initiate_heuristic_swap = (len(options) < 2) # 示例：少于3个直接方案时尝试

        if initiate_heuristic_swap:
            print(f"  > 客户点 {customer} 直接插入方案不足，尝试启发式交换策略...")
            try:
                # 创建 DiverseRouteGenerator 实例 (如果它不依赖特定状态，可以在__init__中创建)
                generator = DiverseRouteGenerator(self.node, self.DEPOT_nodeID, self.A_vtp, self.V, self.T, self.vehicle, self.uav_travel, self.veh_distance, self.veh_travel, self.vtp_coords, self.num_clusters, self.G_air, self.G_ground, self.air_matrix, self.ground_matrix, self.air_node_types, self.ground_node_types, self.A_c, self.xeee)
                is_heuristic_swap = True
                best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = generator.greedy_insert_feasible_plan(
                    customer, vehicle_route, vehicle_arrive_time, vehicle_task_data, repaired_state.customer_plan
                )
                
                if best_orig_y is not None and best_new_y is not None:
                    # a. 计算总成本 (移除成本 + 插入成本)
                    real_cost = best_orig_cost + best_new_cost
                    orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = best_orig_y
                    new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = best_new_y
                    # temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                    temp_customer_cost = {k: v for k, v in repaired_state.uav_cost.items()}
                    # if orig_customer in temp_customer_plan:
                    #     del temp_customer_plan[orig_customer]
                    # temp_customer_plan[orig_customer] = best_orig_y
                    if orig_customer in temp_customer_cost:
                        del temp_customer_cost[orig_customer]
                    temp_customer_cost[orig_customer] = best_orig_cost
                    temp_customer_cost[new_customer] = best_new_cost
                    temp_total_cost = sum(temp_customer_cost.values())
                    
                    # b. 将启发式交换的结果打包成一个特殊的 'plan' 字典
                    #    这样 _execute_insertion 函数才能识别并正确处理它
                    # plan_dict = {
                    #     'customer': customer,
                    #     'orig_scheme': best_orig_y,
                    #     'new_scheme': best_new_y,
                    #     'orig_cost': best_orig_cost,
                    #     'new_cost': best_new_cost,
                    #     'total_cost': temp_total_cost,
                    #     'orig_plan_details': best_orig_y_cijkdu_plan, # 保留详细信息
                    #     'new_plan_details': best_new_y_cijkdu_plan
                    # }
                    options.append(
                        {
                        'customer': customer,
                        'orig_scheme': best_orig_y,
                        'new_scheme': best_new_y,
                        'orig_cost': best_orig_cost,
                        'new_cost': best_new_cost,
                        'eval_cost': real_cost,
                        'real_cost': real_cost,
                        'total_cost': temp_total_cost,
                        'type': 'heuristic_swap', 
                        'extra_info': None,
                        'orig_plan_details': best_orig_y_cijkdu_plan, # 保留详细信息
                        'new_plan_details': best_new_y_cijkdu_plan
                    }
                    )
                    
                    # c. 【重要】可选：加入成本改善检查，避免太差的交换方案污染结果
                    #    current_total_cost = sum(repaired_state.uav_cost.values()) if repaired_state.uav_cost else 0
                    #    cost_improvement_threshold = 0.1 
                    #    cost_improvement = (current_total_cost - real_cost) / max(current_total_cost, 1)
                    #    if cost_improvement >= -cost_improvement_threshold: # 允许少量恶化

                    # eval_cost 等于 real_cost
                    # options.append({
                    #     'customer': customer,
                    #     'eval_cost': real_cost, 
                    #     'real_cost': real_cost, 
                    #     'total_cost': temp_total_cost,
                    #     'scheme': plan_dict, # 存储打包后的字典
                    #     'type': 'heuristic_swap', 
                    #     'extra_info': None
                    # })
                    print(f"    - 找到启发式交换方案，总成本: {real_cost:.2f}")

            except Exception as e:
                print(f"  > 警告: 客户点 {customer} 启发式交换失败: {e}")

        # ----------------------------------------------------------------------
        # 3. 返回收集到的所有可行方案列表
        # ----------------------------------------------------------------------
        return options, is_heuristic_swap

    def _evaluate_traditional_insertion(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state=None):
        """
        评估传统插入方案的成本和方案（使用现有节点）
        包括直接插入和启发式插入两种模式
        返回: (cost, scheme) 或 None
        """
        # try:
        # 1. 首先尝试直接插入方案
        is_heuristic_swap = False
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
                return (min_cost, best_scheme), is_heuristic_swap
        
        # 2. 如果直接插入失败，尝试启发式插入模式
        if repaired_state is not None:
            try:
                # 创建 DiverseRouteGenerator 实例
                generator = DiverseRouteGenerator(self.node, self.DEPOT_nodeID, self.A_vtp, self.V, self.T, self.vehicle, self.uav_travel, self.veh_distance, self.veh_travel, self.vtp_coords, self.num_clusters, self.G_air, self.G_ground, self.air_matrix, self.ground_matrix, self.air_node_types, self.ground_node_types, self.A_c, self.xeee)
                best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = generator.greedy_insert_feasible_plan(
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
                    # heuristic_scheme['type'] = 'heuristic_scheme'
                    return (total_swap_cost, heuristic_scheme), True
            except Exception as e:
                print(f"客户点 {customer} 启发式插入失败: {e}")
        
        # 3. 如果两种方案都失败，返回None
        # print(f"客户点 {customer} 传统插入评估失败: {e}")
        return (None,None), False
            
        # except Exception as e:
        #     print(f"客户点 {customer} 传统插入评估失败: {e}")
        #     return None
    def _regret_evaluate_vtp_expansion_insertion(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state):
            """
            评估VTP扩展插入方案的成本和方案（为每个客户点考虑新增VTP节点）
            返回: (cost, scheme) 或 None
            """
            best_scheme = None
            best_vtp_infor = [None, None, None]
            vtp_infor = [None, None, None]
            min_cost = float('inf')
            vtp_task_data = deep_copy_vehicle_task_data(repaired_state.vehicle_task_data)
            options_result = []
            options_vtp_infor = []
            options_scheme = []
            total_options = []
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
                # 测试每个候选插入位置,测试的全局无人机的成本情况,该处只挑选了距离车辆节点近的位置进行测试，可添加无人机位置综合评估
                for node, insert_pos in candidate_positions:
                    result, scheme = self._calculate_vtp_expansion_cost(customer, vehicle_id, insert_pos, vehicle_route, vtp_task_data, repaired_state, node)
                    options_result.append(result)
                    vtp_infor[0] = node
                    vtp_infor[1] = vehicle_id+1
                    vtp_infor[2] = insert_pos
                    options_vtp_infor.append(vtp_infor)
                    options_scheme.append(scheme)
                    temp_customer_cost = {k: v for k, v in repaired_state.uav_cost.items()}
                    if customer in temp_customer_cost:
                        del temp_customer_cost[customer]
                    temp_customer_cost[customer] = result
                    temp_total_cost = sum(temp_customer_cost.values())
                    total_options.append({
                        'customer': customer,
                        'scheme': scheme,
                        'eval_cost': result,
                        'real_cost': result,
                        'total_cost': temp_total_cost,
                        'type': 'vtp_expansion',
                        'vtp_node': node,
                        'vtp_insert_vehicle_id': vehicle_id+1,
                        'vtp_insert_index': insert_pos,
                        'infor': vtp_infor
                    })
                    # if result is not None:
                    #     if result < min_cost:
                    #         min_cost = result
                    #         best_scheme = scheme
                            # best_vtp_infor[0] = node
                            # best_vtp_infor[1] = vehicle_id+1
                            # best_vtp_infor[2] = insert_pos
            return total_options


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
            # 测试每个候选插入位置,测试的全局无人机的成本情况,该处只挑选了距离车辆节点近的位置进行测试，可添加无人机位置综合评估
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
            distance = self.uav_travel[drone_id][map_vtp_node][customer].totalDistance * 1
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
            all_route = [sub_route[:] for sub_route in vehicle_route]  # 避免指向同一对象
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
    

    # def _calculate_launch_mission_cost(self, drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id):
    #     """
    #     计算无人机作为发射点的成本（从VTP节点到客户点）
    #     参考get_all_insert_position函数的规则，考虑同车和跨车两种情况
    #     """
    #     # try:
    #     # 获取该客户点的最近VTP节点集合
    #     customer_vtp_candidates = self.map_cluster_vtp_dict[customer]
    #     vehicle_idx = vehicle_id -1
    #     # 找到vtp_node在route中的索引
    #     launch_route = route[vehicle_idx]
    #     launch_idx = launch_route.index(vtp_node)
    #     # vtp_node_idx = route.index(vtp_node)
    #     launch_vehicle_id = vehicle_id
        
    #     min_cost = float('inf')
    #     best_scheme = None
    #     # 无人机必须在发射节点上才能发射
    #     if drone_id not in vtp_vehicle_task_data[vehicle_id][vtp_node].drone_list:
    #         # 使用 .get() 避免 KeyError
    #         # print(f"  > 诊断: 无人机 {drone_id} 不在发射节点 {vtp_node} (车辆 {vehicle_id}) 的 drone_list 中。")
    #         return None, None
            
    #     launch_time = vtp_vehicle_arrive_time[vehicle_id][vtp_node]
    #     n_launch = len(launch_route)

    #     # ----------------------------------------------------------------------
    #     # 1. 同车回收 (Intra-vehicle Recovery) - 【核心逻辑重写】
    #     # ----------------------------------------------------------------------
    #     # 遍历发射点之后的所有节点作为潜在回收点
    #     for k in range(launch_idx + 1, n_launch - 1): # 回收点不能是最后一个Depot
    #         recovery_node = launch_route[k]

    #         has_conflict = False
    #         for m in range(launch_idx + 1, k): # 关键：范围不包含 k
    #             intermediate_node = launch_route[m]
    #             # 检查是否有 *已规划的* 发射任务
    #             if drone_id in vtp_vehicle_task_data[vehicle_id][recovery_node].launch_drone_list:
    #                 has_conflict = True
    #                 # print(f"    - 同车冲突: 在节点 {intermediate_node} 发现无人机 {drone_id} 的发射任务，无法在 {recovery_node} 回收。")
    #                 break
    #         if has_conflict:
    #             continue
                
    #         # c. 计算成本并更新最优方案
    #         cost = self.drone_insert_cost(drone_id, customer, vtp_node, recovery_node)
    #         if cost is not None and cost < min_cost:
    #             min_cost = cost
    #             best_scheme = (drone_id, vtp_node, customer, recovery_node, vehicle_id, vehicle_id)
    #             # print(f"    - 找到同车方案: ... 回收于 {recovery_node}, 成本 {cost:.2f}")
    #     # ----------------------------------------------------------------------
    #     # 2. 跨车回收 (Inter-vehicle Recovery) - 【逻辑保持不变，但确认正确】
    #     # ----------------------------------------------------------------------
    #     for rec_veh_idx, rec_route in enumerate(route): # 使用传入的 vehicle_routes
    #         rec_veh_id = rec_veh_idx + 1
    #         if rec_veh_id == vehicle_id:
    #             continue
                
    #         n_rec = len(rec_route)
    #         for k in range(1, n_rec - 1):
    #             recovery_node = rec_route[k]
    #             # a. 回收节点的可行性检查 (基于之前的讨论，使用保守策略)
    #             # if drone_id in vtp_vehicle_task_data[rec_veh_id].get(recovery_node, {}).get('launch_drone_list', []):
    #             #     continue # 保守：如果该节点已有发射计划，不允许回收
                
    #             # b. 检查时序约束
    #             recovery_time = vtp_vehicle_arrive_time[rec_veh_id][recovery_node]
    #             if recovery_time <= launch_time:
    #                 continue

    #             # c. 检查两条路径上的后续/先前任务冲突 (这部分逻辑是正确的)
    #             conflict = False
    #             # 检查发射车辆：发射后不能再有发射任务
    #             for m in range(launch_idx + 1, n_launch - 1):
    #                 if drone_id in vtp_vehicle_task_data[vehicle_id][launch_route[m]].launch_drone_list:
    #                     conflict = True
    #                     break
    #             if conflict: continue

    #             # # 检查回收车辆：回收前不能有发射任务
    #             # for m in range(1, k):
    #             #     if drone_id in vtp_vehicle_task_data[rec_veh_id].get(rec_route[m], {}).get('launch_drone_list', []):
    #             #         conflict = True
    #             #         break
    #             # if conflict: continue

    #             # d. 计算成本并更新最优方案
    #             cost = self.drone_insert_cost(drone_id, customer, vtp_node, recovery_node)
    #             if cost is not None and cost < min_cost:
    #                 min_cost = cost
    #                 best_scheme = (drone_id, vtp_node, customer, recovery_node, vehicle_id, rec_veh_id)
    #                 # print(f"    - 找到跨车方案: ... 回收于 {recovery_node} (车辆 {rec_veh_id}), 成本 {cost:.2f}")
    #     # ----------------------------------------------------------------------
    #     # 3. 返回结果
    #     # ----------------------------------------------------------------------
    #     if best_scheme:
    #         return min_cost, best_scheme
    #     else:
    #         return None, None

    # def _calculate_recovery_mission_cost(self, drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id):
    #     """
    #     计算无人机作为回收点的成本（从客户点到VTP节点）
    #     遍历所有车辆路线作为发射点，判断是否能将VTP作为回收点
    #     """
    #     # try:
    #     # 获取该客户点的最近VTP节点集合
    #     # customer_vtp_candidates = self.map_cluster_vtp_dict[customer]
    #     vehicle_idx = vehicle_id - 1
    #     # 找到vtp_node在route中的索引
    #     recovery_route = route[vehicle_idx]
    #     # vtp_node_idx = route.index(vtp_node)
    #     recovery_vehicle_id = vehicle_id
        
    #     min_cost = float('inf')
    #     best_scheme = None

    #     # ----------------------------------------------------------------------
    #     # 0. 基础检查：回收点可行性
    #     # ----------------------------------------------------------------------
    #     # a. 获取回收车辆路线及回收点索引
    #     recovery_idx = recovery_route.index(vtp_node)
            
    #     recovery_time = vtp_vehicle_arrive_time[vehicle_id][vtp_node]
    #     n_recovery = len(recovery_route)

    #     # b. 检查回收节点是否存在直接冲突（保守策略，与launch函数保持一致）
    #     if drone_id in vtp_vehicle_task_data[vehicle_id].get(vtp_node, {}).get('launch_drone_list', []):
    #         # print(f"  > 诊断: 回收节点 {vtp_node} (车辆 {vehicle_id}) 已有无人机 {drone_id} 的发射计划，无法回收。")
    #         return None, None # 如果节点已有发射计划，不允许回收（保守）

    #     # ----------------------------------------------------------------------
    #     # 遍历所有车辆的所有节点，作为潜在的【发射点】
    #     # ----------------------------------------------------------------------
    #     for launch_veh_idx, launch_route in enumerate(route):
    #         launch_veh_id = launch_veh_idx + 1
    #         n_launch = len(launch_route)
            
    #         for i in range(1, n_launch - 1): # 发射点不能是Depot
    #             launch_node = launch_route[i]
                
    #             # a. 检查发射点可行性：无人机必须在车上
    #             if drone_id not in vtp_vehicle_task_data[launch_veh_id][launch_node].drone_list:
    #                 continue
                    
    #             launch_time = vtp_vehicle_arrive_time[launch_veh_id][launch_node]

    #             # b. 检查时序约束：发射必须早于回收
    #             if launch_time >= recovery_time:
    #                 continue

    #             # c. 区分同车与跨车，进行路径冲突检查
    #             conflict = False
    #             if launch_veh_id == vehicle_id:
    #                 # --- 同车情况 ---
    #                 # 检查在 [i + 1, recovery_idx - 1] 区间内是否有发射任务冲突
    #                 launch_idx = i # 发射点索引就是 i
    #                 for m in range(launch_idx + 1, recovery_idx):
    #                     intermediate_node = launch_route[m]
    #                     if drone_id in vtp_vehicle_task_data[launch_veh_id][intermediate_node].launch_drone_list:
    #                         conflict = True
    #                         break
    #             else:
    #                 # --- 跨车情况 ---
    #                 launch_idx = i # 发射点索引
    #                 # 检查发射车辆：发射后不能再有发射任务
    #                 for m in range(launch_idx + 1, n_launch - 1):
    #                     if drone_id in vtp_vehicle_task_data[launch_veh_id][launch_route[m]].launch_drone_list:
    #                         conflict = True
    #                         break
    #                 if conflict: continue
                    
    #                 # # 检查回收车辆：回收前不能有发射任务
    #                 # for m in range(1, recovery_idx):
    #                 #     if drone_id in vtp_vehicle_task_data[vehicle_id].get(recovery_route[m], {}).get('launch_drone_list', []):
    #                 #         conflict = True
    #                 #         break

    #             if conflict:
    #                 continue

    #             # d. 计算成本并更新最优方案
    #             cost = self.drone_insert_cost(drone_id, customer, launch_node, vtp_node)
    #             if cost is not None and cost < min_cost:
    #                 min_cost = cost
    #                 best_scheme = (drone_id, launch_node, customer, vtp_node, launch_veh_id, vehicle_id)
    #                 # print(f"    - 找到可行发射点: {launch_node} (车辆 {launch_veh_id}), 成本 {cost:.2f}")

    #     # ----------------------------------------------------------------------
    #     # 3. 返回结果
    #     # ----------------------------------------------------------------------
    #     if best_scheme:
    #         return min_cost, best_scheme
    #     else:
    #         # print(f"  > 诊断: 无法为无人机 {drone_id} 在节点 {vtp_node} (车辆 {vehicle_id}) 回收找到可行的发射方案。")
    #         return None, None
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
                for vehicle_idx, route in enumerate(vehicle_route):
                    v_id = vehicle_idx + 1
                    n = len(route)
                    
                    # 【简化】: 使用两层for循环，遍历所有 (发射点, 回收点) 组合
                    for i in range(1, n - 1):
                        launch_node = route[i]
                        
                        # 检查发射点是否有效
                        if drone_id not in vehicle_task_data[v_id][launch_node].drone_list:
                            continue
                        
                        for k in range(i + 1, n - 1):
                            recovery_node = route[k]

                            # 检查回收点是否有效
                            if drone_id not in vehicle_task_data[v_id][recovery_node].drone_list:
                                continue
                                
                            # 【正确逻辑】: 检查在 [i+1, k-1] 区间内是否有该无人机的发射任务冲突
                            has_conflict = False
                            for m in range(i + 1, k):
                                intermediate_node = route[m]
                                if drone_id in vehicle_task_data[v_id][intermediate_node].launch_drone_list:
                                    has_conflict = True
                                    break
                            
                            if not has_conflict:
                                # 所有检查通过，这是一个可行的方案
                                all_insert_position[drone_id].append(
                                    (launch_node, customer, recovery_node, v_id, v_id)
                                )
                    
                # ======================================================================
                # 2. 跨车插入 (Inter-vehicle Insertion) - 【逻辑修正】
                # ======================================================================
                for launch_veh_idx, launch_route in enumerate(vehicle_route):
                    launch_v_id = launch_veh_idx + 1
                    
                    # for i in range(1, len(launch_route) - 1):
                    # 遍历所有节点，不包括起始点和终点
                    for i in range(1, len(launch_route)-1):
                        launch_node = launch_route[i]

                        # 检查发射点是否有效
                        if drone_id not in vehicle_task_data[launch_v_id][launch_node].drone_list:
                            continue
                        
                        launch_time = vehicle_arrive_time[launch_v_id][launch_node]

                        for rec_veh_idx, rec_route in enumerate(vehicle_route):
                            rec_v_id = rec_veh_idx + 1
                            if launch_v_id == rec_v_id:
                                continue

                            for k in range(1, len(rec_route) - 1):
                                recovery_node = rec_route[k]
                                
                                # 检查回收点是否有效
                                if drone_id not in vehicle_task_data[rec_v_id][recovery_node].drone_list:
                                    continue
                                
                                recovery_time = vehicle_arrive_time[rec_v_id][recovery_node]

                                # a. 检查时序约束
                                if recovery_time <= launch_time:
                                    continue
                                
                                # b. 【正确逻辑】: 检查两条路径上的冲突
                                conflict = False
                                # 检查发射车辆：在发射后，该无人机不能再有发射任务
                                for m in range(i + 1, len(launch_route) - 1):
                                    if drone_id in vehicle_task_data[launch_v_id][launch_route[m]].launch_drone_list:
                                        conflict = True
                                        break
                                if conflict: continue

                                # 检查回收车辆：在回收前，该无人机不能有发射任务
                                # for m in range(1, k):
                                #     if drone_id in vehicle_task_data[rec_v_id][rec_route[m]].launch_drone_list:
                                #         conflict = True
                                #         break
                                # if conflict: continue

                                # 所有检查通过，这是一个可行的方案
                                all_insert_position[drone_id].append(
                                    (launch_node, customer, recovery_node, launch_v_id, rec_v_id)
                                )
                                
            total_positions = sum(len(positions) for positions in all_insert_position.values())
            if total_positions < 5:
                print(f"警告：客户点 {customer} 的可行插入位置过少 ({total_positions} 个)，可能影响优化效果")

            return all_insert_position

    # def get_all_insert_position(self, vehicle_route, vehicle_task_data, customer, vehicle_arrive_time):
    #     """
    #     获取所有可行的插入位置，通过cluster_vtp_dict限制解空间以提高效率
        
    #     Args:
    #         vehicle_route: 车辆路线
    #         vehicle_task_data: 车辆任务数据
    #         customer: 客户点ID
    #         vehicle_arrive_time: 车辆到达时间
            
    #     Returns:
    #         dict: {drone_id: [(launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id), ...]}
    #     """
    #     all_insert_position = {drone_id: [] for drone_id in self.V}
        
    #     # 获取该客户点的最近VTP节点集合
    #     customer_vtp_candidates = self.map_cluster_vtp_dict[customer]
    #     # print(f"客户点 {customer} 的VTP候选节点: {customer_vtp_candidates[:5]}...")  # 只显示前5个

    #     for drone_id in self.V:
    #         for launch_vehicle_idx, route in enumerate(vehicle_route):
    #             launch_vehicle_id = launch_vehicle_idx + 1
    #             n = len(route)
    #             i = 1
    #             while i < n - 1:
    #                 launch_node = route[i]
    #                 # 只在drone_list中才可发射
    #                 if drone_id not in vehicle_task_data[launch_vehicle_id][launch_node].drone_list:
    #                     i += 1
    #                     continue
                    
    #                 # 检查发射节点是否在客户点的VTP候选集合中（放宽限制）
    #                 if launch_node not in customer_vtp_candidates:
    #                     # 如果不在候选集合中，仍然允许，但降低优先级
    #                     pass
                    
    #                 # 找连续片段
    #                 j = i + 1
    #                 while j < n - 1:
    #                     node = route[j]
    #                     in_drone_list = drone_id in vehicle_task_data[launch_vehicle_id][node].drone_list
    #                     in_launch_list = drone_id in vehicle_task_data[launch_vehicle_id][node].launch_drone_list
    #                     if not in_drone_list:
    #                         if in_launch_list:
    #                             # 片段终点包含该节点
    #                             j += 1
    #                         break
    #                     j += 1
    #                 # 现在[i, j)是连续片段，j可能因为break提前终止
    #                 # 片段终点为j-1，如果j-1节点是发射点（即不在drone_list但在launch_drone_list），包含它
    #                 end = j
    #                 if j < n - 1:
    #                     node = route[j]
    #                     if drone_id not in vehicle_task_data[launch_vehicle_id][node].drone_list and \
    #                     drone_id in vehicle_task_data[launch_vehicle_id][node].launch_drone_list:
    #                         end = j + 1  # 包含发射点
                    
    #                 # 同车插入：寻找所有可能的回收节点
    #                 for k in range(i + 1, n - 1):
    #                     recovery_node = route[k]
                        
    #                     # 检查回收节点是否支持该无人机
    #                     if drone_id not in vehicle_task_data[launch_vehicle_id][recovery_node].drone_list:
    #                         continue
                        
    #                     # 检查发射节点和回收节点之间是否存在冲突
    #                     # 规则：回收节点前(不含回收节点)，发射节点后不能存在该无人机的发射任务
    #                     launch_idx = i
    #                     recovery_idx = k
                        
    #                     # 检查发射节点之后到回收节点之前是否有该无人机的发射任务
    #                     has_conflict = False
    #                     for m in range(launch_idx + 1, recovery_idx):
    #                         if drone_id in vehicle_task_data[launch_vehicle_id][route[m]].launch_drone_list:
    #                             has_conflict = True
    #                             break
                        
    #                     if has_conflict:
    #                         # print(f"[DEBUG] 同车插入跳过：无人机 {drone_id} 从节点 {launch_node} 到节点 {recovery_node} 之间存在发射任务冲突")
    #                         continue
                        
    #                     # 检查回收节点是否在客户点的VTP候选集合中（放宽限制）
    #                     # 无论是否在候选集合中，都允许插入，但可以标记优先级
    #                     all_insert_position[drone_id].append(
    #                         (launch_node, customer, recovery_node, launch_vehicle_id, launch_vehicle_id)
    #                     )
    #                 i = j
                
    #             # 跨车查找：检查发射节点是否在VTP候选集合中
    #             for i in range(1, n - 1):
    #                 launch_node = route[i]
    #                 if drone_id not in vehicle_task_data[launch_vehicle_id][launch_node].drone_list:
    #                     continue
                    
    #                 # 检查发射节点是否在客户点的VTP候选集合中（放宽限制）
    #                 if launch_node not in customer_vtp_candidates:
    #                     # 如果不在候选集合中，仍然允许，但降低优先级
    #                     pass
                    
    #                 launch_time = vehicle_arrive_time[launch_vehicle_id][launch_node]
    #                 for recovery_vehicle_idx, other_route in enumerate(vehicle_route):
    #                     recovery_vehicle_id = recovery_vehicle_idx + 1
    #                     if recovery_vehicle_id == launch_vehicle_id:
    #                         continue
    #                     for recovery_node in other_route[1:-1]:
    #                         if drone_id not in vehicle_task_data[recovery_vehicle_id][recovery_node].drone_list:
    #                             continue
                            
    #                         # 检查回收节点是否在客户点的VTP候选集合中（放宽限制）
    #                         if recovery_node not in customer_vtp_candidates:
    #                             # 如果不在候选集合中，仍然允许，但降低优先级
    #                             pass
                            
    #                         # 新增：排除发射点和回收点完全相同的情况
    #                         # if launch_vehicle_id == recovery_vehicle_id and launch_node == recovery_node:
    #                         if launch_vehicle_id == recovery_vehicle_id:
    #                             continue  # 跨车时也不允许同节点
    #                         if launch_node == recovery_node:
    #                             continue  # 跨车时也不允许同节点
    #                         recovery_time = vehicle_arrive_time[recovery_vehicle_id][recovery_node]
    #                         if recovery_time <= launch_time:
    #                             continue
    #                         idx = other_route.index(recovery_node)
    #                         conflict = False
                            
    #                         # 检查回收车辆路线中的冲突（放宽限制）
    #                         # for m in range(1, idx):
    #                         #     if drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].launch_drone_list:
    #                         #         # 只检查发射冲突，允许回收冲突
    #                         #         conflict = True
    #                         #         break
    #                         # for m in range(idx + 1, len(other_route) - 1):
    #                         #     if drone_id in vehicle_task_data[recovery_vehicle_id][other_route[m]].launch_drone_list:
    #                         #         conflict = True
    #                         #         break
                            
    #                         # 检查发射车辆路线中的冲突（放宽限制）
    #                         launch_idx = route.index(launch_node)
    #                         for m in range(launch_idx + 1, len(route) - 1):
    #                             if drone_id in vehicle_task_data[launch_vehicle_id][route[m]].launch_drone_list:
    #                                 # 只检查发射冲突，允许回收冲突
    #                                 conflict = True
    #                                 # print(f"[DEBUG] 跨车插入冲突：无人机 {drone_id} 从车辆 {launch_vehicle_id} 节点 {launch_node} 发射到车辆 {recovery_vehicle_id} 节点 {recovery_node}，但车辆 {launch_vehicle_id} 的节点 {route[m]} 还有该无人机的发射任务")
    #                                 break
                            
    #                         if not conflict:
    #                             all_insert_position[drone_id].append(
    #                                 (launch_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
    #                             )
        
    #     # 统计每个无人机的可行插入位置数量
    #     total_positions = 0
    #     for drone_id in self.V:
    #         positions_count = len(all_insert_position[drone_id])
    #         total_positions += positions_count
    #         # if positions_count > 0:
    #             # print(f"无人机 {drone_id} 有 {positions_count} 个可行插入位置")
        
    #     # print(f"客户点 {customer} 总共有 {total_positions} 个可行插入位置")
        
    #     # 如果插入位置太少，输出警告
    #     if total_positions < 5:
    #         print(f"警告：客户点 {customer} 的可行插入位置过少 ({total_positions} 个)，可能影响优化效果")
    #     return all_insert_position
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
        # 设置对不可行破坏或修复方案的惩罚机制
        decay_factor = 0.95

        # (你对初始解的预处理，这部分完全保留)
        current_state.rm_empty_vehicle_route, current_state.empty_nodes_by_vehicle = current_state.update_rm_empty_task()
        # current_state.rm_empty_vehicle_route = [route[:] for route in current_state.vehicle_routes]
        current_state.vehicle_routes = [route[:] for route in current_state.rm_empty_vehicle_route]
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
        
        init_uav_cost = list(current_state.uav_cost.values())
        base_flexibility_bonus = sum(init_uav_cost) / len(init_uav_cost)
        
        # 3. 初始化模拟退火和双重衰减奖励模型
        #    【重要建议】: 对于更复杂的搜索，建议增加迭代次数并减缓降温速率
        # temperature = 100.0
        temperature = 500.0
        initial_temperature = temperature  # 记录初始温度，用于战略奖励计算
        # cooling_rate = 0.95  # 缓慢降温以进行更充分的探索
        cooling_rate = 0.985  # 缓慢降温以进行更充分的探索
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
            print(f'当前的任务客户点数量为:{len(current_state.customer_plan.keys())}')
            
            # 调试条件：检查vehicle_task_data[2][129].drone_list
            if hasattr(current_state, 'vehicle_task_data') and 2 in current_state.vehicle_task_data and 129 in current_state.vehicle_task_data[2]:
                if hasattr(current_state.vehicle_task_data[2][129], 'drone_list'):
                    if current_state.vehicle_task_data[2][129].drone_list == [10, 9]:
                        print("调试：找到vehicle_task_data[2][129].drone_list == [10,9]")
                        # import pdb; pdb.set_trace()
            # prev_objective = current_objective
            if chosen_strategy == 'structural':
                # **策略一：结构性重组** (强制VTP破坏 + 带双重衰减奖励的修复)
                destroyed_state = destroy_op(prev_state, force_vtp_mode=True)
                
                # 计算本轮迭代的战略奖励基准值
                strategic_bonus = base_flexibility_bonus * (temperature / initial_temperature)
                num_destroyed = len(destroyed_state.destroyed_customers_info)
        
                repaired_state, _ = repair_op(destroyed_state, strategic_bonus, num_destroyed, force_vtp_mode=True)
                if repaired_state.repair_objective == float('inf'):
                    print("  > 修复后方案为空，跳过此次迭代。")
                    iteration += 1
                    # 将所使用的算子进行降分处理，暂缓选入的方案
                    # 惩罚破坏算子
                    self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] *= decay_factor
                    # 防止权重过低
                    if self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] < 0.1: 
                        self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] = 0.1
                    # 惩罚修复算子
                    self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] *= decay_factor
                    if self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] < 0.1:
                        self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] = 0.1
                    # 温度仍然需要衰减
                    temperature *= cooling_rate
                    # 记录失败的成本（可以记录前一个状态的成本）
                    y_cost.append(current_objective) 
                    # 将修复后的状态重置为初始状态
                    repaired_state.repair_objective = 0
                    continue
                
            else: # chosen_strategy == 'internal'
                # **策略二：内部精细优化** (强制客户破坏 + 无奖励的修复)
                destroyed_state = destroy_op(prev_state, force_vtp_mode=False)

                num_destroyed = len(destroyed_state.destroyed_customers_info)
                # 传入零奖励，关闭“战略投资”模式
                repaired_state, _ = repair_op(destroyed_state, strategic_bonus=0, num_destroyed=num_destroyed, force_vtp_mode=False)
                if repaired_state.repair_objective == float('inf'):
                    print("  > 修复后方案为空，跳过此次迭代。")
                    iteration += 1
                    # 将所使用的算子进行降分处理，暂缓选入的方案
                    # 惩罚破坏算子
                    self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] *= decay_factor
                    # 防止权重过低
                    if self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] < 0.1: 
                        self.operator_weights[chosen_strategy]['destroy'][chosen_destroy_op_name] = 0.1

                    # 惩罚修复算子
                    self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] *= decay_factor
                    if self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] < 0.1:
                        self.operator_weights[chosen_strategy]['repair'][chosen_repair_op_name] = 0.1

                    # 温度仍然需要衰减
                    temperature *= cooling_rate
                    # 记录失败的成本（可以记录前一个状态的成本）
                    y_cost.append(current_objective) 
                    # 将修复后的状态重置为初始状态
                    repaired_state.repair_objective = 0
                    continue

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
                print(f"  > 策略权重: {self.strategy_weights}")
                print(f"  > 算子权重: {self.operator_weights}")
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
                temp_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)  # 用于链式删除分析
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
                            from task_data import deep_remove_vehicle_task
                            # print(f"链式删除客户点 {chain_customer}")
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)

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

    # 考虑帕累托的多目标最差节点破坏
    def destroy_comprehensive_removal(self, state, force_vtp_mode = None):
        new_state = state.fast_copy() # 确保在副本上操作
        current_customers = list(new_state.customer_plan.keys())
        vehicle_task_data = new_state.vehicle_task_data
        mode = 'vtp' if force_vtp_mode else 'customer'
        # print(f"  > [破坏模式]: 综合最差破坏 ({'VTP模式' if mode == 'vtp' else '客户模式'})")
        # mode = 'customer'
        # ----------------------------------------------------------------------
        # 2. VTP破坏模式：移除综合效率最低的VTP (Pareto + TopK随机)
        # ----------------------------------------------------------------------
        if mode == 'vtp':
            # --- 步骤 1: 计算基础效率分数 ---
            # 字典存储每个VTP的指标: {(veh_id, vtp_node): {'score_drone': float, 'score_vehicle': float, 'task_count': int}}
            vtp_metrics = {} 

            # 1a. 计算无人机相关指标 (按比例归因)
            # 使用 defaultdict 简化初始化
            vtp_drone_performance = defaultdict(lambda: {'total_cost': 0.0, 'task_count': 0}) 
            
            # 【完整代码】遍历 customer_plan, 计算 cost_leg1, cost_leg2 并正确归因
            for customer, assignment in new_state.customer_plan.items():
                try:
                    # 从 assignment 中解包所需信息
                    uav_id, launch_node, _, recovery_node, launch_veh, recovery_veh = assignment
                    
                    # 获取用于查询成本/距离矩阵的映射索引或键
                    launch_node_map_key = self.node[launch_node].map_key
                    recovery_node_map_key = self.node[recovery_node].map_key
                    customer_map_key = customer # 假设客户ID可以直接用于 uav_travel 查询
                    
                    # 获取无人机的单位成本
                    per_cost = self.vehicle[uav_id].per_cost
                    
                    # 计算发射段成本 (VTP -> Customer)
                    # 假设 self.uav_travel 结构是 [uav_id][from_map_key][to_map_key].totalDistance
                    distance_leg1 = self.uav_travel[uav_id][launch_node_map_key][customer_map_key].totalDistance
                    cost_leg1 = distance_leg1 * per_cost
                    
                    # 计算回收段成本 (Customer -> VTP)
                    distance_leg2 = self.uav_travel[uav_id][customer_map_key][recovery_node_map_key].totalDistance
                    cost_leg2 = distance_leg2 * per_cost
                    
                except (KeyError, AttributeError, TypeError, IndexError) as e:
                    print(f"  > 警告: 无法为客户 {customer} 任务计算分段成本 ({assignment})，跳过归因。错误: {e}")
                    continue # 跳过这个任务的成本归因
                # 将【发射段成本】归因给发射VTP
                launch_key = (launch_veh, launch_node)
                vtp_drone_performance[launch_key]['total_cost'] += cost_leg1
                vtp_drone_performance[launch_key]['task_count'] += 1

                # 将【回收段成本】归因给回收VTP
                recovery_key = (recovery_veh, recovery_node)
                vtp_drone_performance[recovery_key]['total_cost'] += cost_leg2
                vtp_drone_performance[recovery_key]['task_count'] += 1

            # 1b. 计算车辆相关指标 (绕路成本) 并合并指标
            epsilon = 1e-6 # 防止除零
            active_vtps_keys = set() # 记录所有活动的VTP key
            for vehicle_id_minus_1, route in enumerate(new_state.vehicle_routes):
                vehicle_id = vehicle_id_minus_1 + 1
                if len(route) <= 2: continue
                
                for node_idx in range(1, len(route) - 1):
                    vtp_node = route[node_idx]
                    vtp_key = (vehicle_id, vtp_node)
                    active_vtps_keys.add(vtp_key)

                    # 计算 Score_Drone (从已计算好的 vtp_drone_performance 获取)
                    drone_data = vtp_drone_performance.get(vtp_key, {'total_cost': 0.0, 'task_count': 0})
                    task_count = drone_data['task_count']
                    score_drone = drone_data['total_cost'] / (task_count + epsilon)

                    # 计算 Score_Vehicle
                    prev_node = route[node_idx - 1]
                    next_node = route[node_idx + 1]
                    # 确保计算绕路成本时节点有效
                    try:
                        if prev_node not in self.veh_distance[vehicle_id] or \
                        vtp_node not in self.veh_distance[vehicle_id][prev_node] or \
                        next_node not in self.veh_distance[vehicle_id][vtp_node] or \
                        next_node not in self.veh_distance[vehicle_id][prev_node]:
                            raise KeyError("Missing distance data") # 抛出异常以便统一处理

                        detour_cost = self.veh_distance[vehicle_id][prev_node][vtp_node] + \
                                    self.veh_distance[vehicle_id][vtp_node][next_node] - \
                                    self.veh_distance[vehicle_id][prev_node][next_node]
                    except (KeyError, IndexError) as e:
                        print(f"  > 警告: 无法计算VTP {vtp_key} 的绕路成本 ({prev_node}->{vtp_node}->{next_node})。设为0。错误: {e}")
                        detour_cost = 0.0 
                        
                    score_vehicle = detour_cost / (task_count + epsilon)

                    # 存储所有指标
                    vtp_metrics[vtp_key] = {
                        'score_drone': score_drone, 
                        'score_vehicle': score_vehicle, 
                        'task_count': task_count
                    }
            # --- 步骤 2: Pareto筛选 ---
            # 获取所有活动VTP的 key 列表
            active_vtp_list = list(vtp_metrics.keys())
            N = len(active_vtp_list)

            # 检查是否有可评估的VTP
            if N == 0:
                print("  > 警告: 没有可评估的活动VTP节点，本次破坏无操作。")
                return new_state # 返回副本

            # Pareto筛选阈值 (例如，选择效率排在后30%的)
            P_thresh = 0.3 
            # 计算排名阈值 T (至少为1，即使只有一个VTP也要参与排名)
            T = max(1, math.ceil(N * P_thresh)) 

            # 按 Score_Drone 降序排名 (越高越差)
            sorted_by_drone = sorted(active_vtp_list, key=lambda k: vtp_metrics[k]['score_drone'], reverse=True)
            # 按 Score_Vehicle 降序排名 (越高越差)
            sorted_by_vehicle = sorted(active_vtp_list, key=lambda k: vtp_metrics[k]['score_vehicle'], reverse=True)

            # 找出两个排名都靠前的VTP (索引小于T)
            P_worst_drone = set(sorted_by_drone[:T])
            P_worst_vehicle = set(sorted_by_vehicle[:T])
            
            # 找出“双差生”集合 (Pareto前沿)
            P_pareto = P_worst_drone.intersection(P_worst_vehicle)

            # --- 步骤 3: 确定最终候选池 ---
            candidate_keys_sorted = [] # 存储排序后的候选VTP key

            if P_pareto:
                print(f"  > Pareto筛选: 找到 {len(P_pareto)} 个双差生VTP。")
                # 如果存在双差生，优先考虑它们，并按无人机效率排序
                candidate_keys_sorted = sorted(list(P_pareto), key=lambda k: vtp_metrics[k]['score_drone'], reverse=True)
            elif P_worst_drone: # 没有双差生，退而求其次
                print("  > Pareto筛选: 未找到双差生，仅基于无人机效率选择。")
                # 直接取无人机效率最差的T个作为候选
                candidate_keys_sorted = sorted_by_drone[:T] 
            else:
                # 理论上 P_worst_drone 不会为空，除非N=0已处理
                print("  > 警告: 无法确定候选VTP池，本次破坏无操作。")
                return new_state
                
            # 再次检查候选池是否为空
            if not candidate_keys_sorted:
                print("  > 警告: 最终候选VTP池为空，本次破坏无操作。")
                return new_state

            # --- 步骤 4: Top-K 带权随机选择 ---
            # 确定要移除的数量 (从 __init__ 获取，通常为 1)
            # num_to_remove = self.vtp_destroy_quantity['worst'] 
            num_to_remove = 5
            # 确保移除数量不超过候选数量
            num_to_remove = min(num_to_remove, len(candidate_keys_sorted)) 

            # 设定Top-K候选池的大小
            K = 5 
            # 从排序后的候选者中选出Top-K
            top_k_candidates_keys = candidate_keys_sorted[:K]

            vtps_to_destroy = [] # 存储最终要破坏的VTP列表
            
            # 处理特殊情况
            if not top_k_candidates_keys:
                print("  > 警告: Top-K 候选池为空，本次破坏无操作。")
                return new_state
            elif len(top_k_candidates_keys) == 1 or num_to_remove == 0: 
                # 如果只有一个候选或无需移除，直接选择
                vtps_to_destroy = top_k_candidates_keys[:num_to_remove] 
            else:
                # 计算权重 (线性排名: Top1权重最高)
                weights = np.arange(len(top_k_candidates_keys), 0, -1)
                # 归一化权重，处理总和为0的情况
                weight_sum = np.sum(weights)
                probabilities = weights / weight_sum if weight_sum > 0 else None

                if probabilities is None:
                    print("  > 警告: 无法计算选择概率，将选择Top-N。")
                    # 如果无法计算概率，直接选择排名最靠前的 num_to_remove 个
                    vtps_to_destroy = top_k_candidates_keys[:num_to_remove]
                else:
                    # 带权重随机选择 num_to_remove 个 VTP 的索引
                    chosen_indices = self.rng.choice(len(top_k_candidates_keys), 
                                                    size=num_to_remove, 
                                                    p=probabilities, 
                                                    replace=False) # 无放回选择
                    # 获取被选中的VTP key
                    vtps_to_destroy = [top_k_candidates_keys[i] for i in chosen_indices]

            # 打印选择信息
            # print(f"  > Top-{min(K, len(candidate_keys_sorted))} 候选 (DroneScore|VehScore): "f"{[f'{k}:{vtp_metrics[k].get("score_drone", float("inf")):.1f}|{vtp_metrics[k].get("score_vehicle", float("inf")):.1f}' for k in top_k_candidates_keys]}") # 使用.get增加健壮性
            # print(f"  > 最终选择移除 VTP: {vtps_to_destroy}")

            # --- 步骤 5: 执行破坏 (包含时间约束检查) ---
            destroyed_customers_info = new_state.destroyed_customers_info 
            # 使用 getattr 安全获取属性，如果不存在则初始化为空字典
            destroyed_vts_info = getattr(new_state, 'destroyed_vts_info', {}) 
            # vehicle_task_data = new_state.vehicle_task_data # 直接在 new_state 上修改

            destroyed_vtp_count = 0
            actual_destroyed_vtps = [] 

            # 开始执行vtp节点任务的破坏策略
            destroyed_customers_info = {}  # 用于存储被破坏的客户节点信息
            destroyed_vtp_count = 0  # 实际破坏的VTP节点数量
            max_attempts = len(vtps_to_destroy) * 2  # 最大尝试次数，避免无限循环
            attempt_count = 0
            
            # 创建候选节点池的副本，用于按优先级选择
            candidate_vtps = vtps_to_destroy.copy()
            
            while destroyed_vtp_count < self.vtp_destroy_quantity['worst'] and candidate_vtps and attempt_count < max_attempts:
                attempt_count += 1
                
                # 从候选池中选择下一个VTP节点（按成本效益比排序）
                if not candidate_vtps:
                    print(f"候选池已空，无法继续破坏VTP节点")
                    break
                    
                vehicle_id, vtp_node = candidate_vtps.pop(0)  # 按优先级顺序选择
                if vtp_node not in new_state.rm_empty_vehicle_route[vehicle_id-1]:
                    # candidate_vtps = [top_k_candidates_keys[attempt_count]]
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
                temp_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)  # 用于链式删除分析
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
                    # candidate_vtps = candidate_keys_sorted[attempt_count]
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
            new_state.vehicle_routes = [route[:] for route in new_state.rm_empty_vehicle_route]  # vtp节点被破坏后重更新
            new_state.rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(new_state.vehicle_routes)
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            # print(f"破坏后剩余VTP节点: {sum(len(route) - 2 for route in new_state.vehicle_routes)}")  # 减去起点和终点
            # print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== VTP破坏阶段完成 ===\n")
        else:
            # 开始执行客户点层面的破坏策略
            print("  > [破坏模式]: 综合最差破坏 (客户模式 - Pareto)")
        
            # 3.1 收集所有已服务客户
            current_customers = list(new_state.customer_plan.keys())
            if not current_customers:
                print("  > 警告: 没有已服务的客户可供破坏。")
                return new_state
                
            # --- 步骤 1: 计算基础效率分数 ---
            # 字典存储每个客户的指标: {customer_id: {'score_cost': float, 'score_slack': float}}
            customer_metrics = {} 
            
            # 为了计算slack，我们需要车辆的到达/离开时间
            # 【注意】: 这可能需要您调用更详细的时间计算函数
            # 作为简化，我们先使用 uav_cost 作为成本指标
            # 并计算一个简化的“任务时长”作为时间紧张度的代理指标

            for customer in current_customers:
                cost = new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0
                
                # a. 指标1: 任务成本 (越高越差)
                score_cost = cost
                
                # b. 指标2: 任务时长 (越高越差)
                #    (这是一个示例指标，您可以替换为更精确的“时间窗口紧张度”或“Slack Time”)
                score_duration = 0.0
                try:
                    assignment = new_state.customer_plan[customer]
                    uav_id, launch_node, _, recovery_node, launch_veh, recovery_veh = assignment
                    
                    # 计算无人机总飞行时间
                    launch_node_map_key = self.node[launch_node].map_key
                    recovery_node_map_key = self.node[recovery_node].map_key
                    # 假设 uav_travel 存储的是 TravelInfo 对象
                    time_leg1 = self.uav_travel[uav_id][launch_node_map_key][customer].totalTime
                    time_leg2 = self.uav_travel[uav_id][customer][recovery_node_map_key].totalTime
                    score_duration = time_leg1 + time_leg2
                    
                except Exception as e:
                    # print(f"  > 警告: 客户模式 - 无法计算客户 {customer} 的任务时长: {e}")
                    score_duration = 0.0 # 计算失败则设为0
                
                customer_metrics[customer] = {
                    'score_cost': score_cost,
                    'score_duration': score_duration
                }

            if not customer_metrics:
                return new_state # 不应发生

            # --- 步骤 2: Pareto筛选 ---
            active_customer_list = list(customer_metrics.keys())
            N = len(active_customer_list)
            P_thresh = 0.5 # 筛选阈值 (后30%)
            T = max(1, math.ceil(N * P_thresh)) 

            # 按 Score_Cost 降序排名 (越高越差)
            sorted_by_cost = sorted(active_customer_list, key=lambda k: customer_metrics[k]['score_cost'], reverse=True)
            # 按 Score_Duration 降序排名 (越高越差)
            sorted_by_duration = sorted(active_customer_list, key=lambda k: customer_metrics[k]['score_duration'], reverse=True)

            K_worst_cost = set(sorted_by_cost[:T])
            K_worst_duration = set(sorted_by_duration[:T])
            
            # 找出“双差生”集合 (成本又高，耗时又长)
            K_pareto = K_worst_cost.intersection(K_worst_duration)

            # --- 步骤 3: 确定最终候选池 ---
            candidate_keys_sorted = [] 
            if K_pareto:
                print(f"  > Pareto筛选: 找到 {len(K_pareto)} 个双差生客户。")
                candidate_keys_sorted = sorted(list(K_pareto), key=lambda k: customer_metrics[k]['score_cost'], reverse=True)
            elif K_worst_cost: # 没有双差生，退而求其次
                print("  > Pareto筛选: 未找到双差生，仅基于任务成本选择。")
                candidate_keys_sorted = sorted_by_cost[:T] 
            else:
                print("  > 警告: 无法确定客户候选池，本次破坏无操作。")
                return new_state
                
            if not candidate_keys_sorted:
                print("  > 警告: 最终客户候选池为空，本次破坏无操作。")
                return new_state

            # --- 步骤 4: Top-K 带权随机选择 ---
            # 确定破坏数量：动态百分比 (与您原代码一致)
            n = len(current_customers)
            num_to_remove = self.rng.integers(
                max(1, int(n * self.customer_destroy_ratio[0])),
                max(2, int(n * self.customer_destroy_ratio[1])) + 1
            )
            num_to_remove = min(num_to_remove, n)

            # 设定Top-K候选池的大小
            K = max(10, 2 * num_to_remove) 
            top_k_candidates_keys = candidate_keys_sorted[:K]

            customers_to_destroy = []
            
            if not top_k_candidates_keys:
                print("  > 警告: Top-K 客户候选池为空，本次破坏无操作。")
                return new_state
            elif len(top_k_candidates_keys) == 1 or num_to_remove == 0: 
                customers_to_destroy = top_k_candidates_keys[:num_to_remove] 
            else:
                weights = np.arange(len(top_k_candidates_keys), 0, -1)
                weight_sum = np.sum(weights)
                probabilities = weights / weight_sum if weight_sum > 0 else None

                if probabilities is None:
                    print("  > 警告: 无法计算客户选择概率，将选择成本最高的。")
                    customers_to_destroy = top_k_candidates_keys[:num_to_remove]
                else:
                    num_to_select = min(num_to_remove, len(top_k_candidates_keys))
                    chosen_indices = self.rng.choice(len(top_k_candidates_keys), size=num_to_select, p=probabilities, replace=False)
                    customers_to_destroy = [top_k_candidates_keys[i] for i in chosen_indices]
            
            print(f"  > 计划移除 {len(customers_to_destroy)} 个综合最差客户 (Top-{min(K, n)}随机): {customers_to_destroy}")

            # --- 步骤 5: 执行破坏 (与您原有的框架一致) ---
            destroyed_customers_info = new_state.destroyed_customers_info
            vehicle_task_data = new_state.vehicle_task_data
            
            # 4. 移除这些客户点及相关无人机任务
            for customer in customers_to_destroy:
                if customer in new_state.customer_plan:
                    # 删除每个客户点需要检测时间约束
                    temp_customer_plan = {k: v for k, v in new_state.customer_plan.items()}
                    temp_vehicle_routes = [route[:] for route in new_state.vehicle_routes]
                    temp_vehicle_task_data = deep_copy_vehicle_task_data(new_state.vehicle_task_data)
                    temp_assignment = new_state.customer_plan[customer]
                    temp_chain_tasks = find_chain_tasks(temp_assignment, temp_customer_plan, temp_vehicle_routes, temp_vehicle_task_data)
                    temp_customer_plan.pop(customer, None)
                    for chain_customer, chain_assignment in temp_chain_tasks:
                        temp_customer_plan.pop(chain_customer, None)
                    temp_rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_routes)
                    if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                        continue

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
                            from task_data import deep_remove_vehicle_task
                            print(f"链式删除客户点 {chain_customer}")
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)

            # 5. 更新空跑节点等状态
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            
            # 将破坏的客户节点信息存储到状态中，供修复阶段使用
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== 破坏阶段完成 ===\n")
        return new_state

    def _calculate_vehicle_load(self, state):
        """
        计算每辆车的综合负载分数。
        Returns:
            dict: {vehicle_id: score} (分数越高越忙)
        """
        vehicle_load = {}
        costs = {}
        tasks = {}
        epsilon = 1e-6

        # 1. 收集每辆车的成本和任务数
        for vehicle_id_minus_1, route in enumerate(state.vehicle_routes):
            vehicle_id = vehicle_id_minus_1 + 1
            route_cost = 0
            task_count = 0
            
            # a. 计算路径成本
            if len(route) > 1:
                for i in range(len(route) - 1):
                    try:
                        route_cost += self.veh_distance[vehicle_id][route[i]][route[i+1]]
                    except KeyError:
                        pass # 忽略 Depot 间距离等
            
            # b. 计算任务数
            for node in route[1:-1]:
                key = (vehicle_id, node)
                task_count += len(state.customer_plan.get(key, {}).get('launch_drone_list', [])) # 您的数据结构可能不同
                task_count += len(state.customer_plan.get(key, {}).get('recovery_drone_list', []))
                
            costs[vehicle_id] = route_cost
            tasks[vehicle_id] = task_count

        # 2. 归一化并计算综合负载分数
        max_cost = max(costs.values()) if costs else 1
        max_tasks = max(tasks.values()) if tasks else 1
        
        # 权重 (可调超参数)
        w_route = 0.5 # 路径成本占 50%
        w_task = 0.5  # 任务数量占 50%

        for v_id in costs.keys():
            norm_cost = costs[v_id] / (max_cost + epsilon)
            norm_task = tasks[v_id] / (max_tasks + epsilon)
            score_load = w_route * norm_cost + w_task * norm_task
            vehicle_load[v_id] = score_load

        return vehicle_load

    # 考虑负载不均衡的shaw破坏策略
    def destroy_shaw_rebalance_removal(self, state, force_vtp_mode = None):
        new_state = state.fast_copy() # 确保在副本上操作
        current_customers = list(new_state.customer_plan.keys())
        vehicle_task_data = new_state.vehicle_task_data
        mode = 'vtp' if force_vtp_mode else 'customer'
        # print(f"  > [破坏模式]: 综合最差破坏 ({'VTP模式' if mode == 'vtp' else '客户模式'})")
        # mode = 'customer'

        # --- 步骤 1: 识别“最忙”和“最闲”的车辆 ---
        vehicle_load_scores = self._calculate_vehicle_load(new_state)
        if not vehicle_load_scores or len(vehicle_load_scores) < 2:
            print("  > 警告: 无法计算车辆负载或车辆数不足，退化为随机破坏。")
            return self.destroy_random_removal(state, force_vtp_mode) # 调用另一个算子作为后备

        sorted_vehicles = sorted(vehicle_load_scores.items(), key=lambda item: item[1])
        v_min_id = sorted_vehicles[0][0]  # 最闲车辆ID
        v_max_id = sorted_vehicles[-1][0] # 最忙车辆ID

        if v_min_id == v_max_id:
            print("  > 警告: 车辆负载相同，退化为随机破坏。")
            return self.destroy_random_removal(state, force_vtp_mode)
            
        print(f"  > 负载分析: 最忙车辆 V{v_max_id} (Score: {sorted_vehicles[-1][1]:.2f}), 最闲车辆 V{v_min_id} (Score: {sorted_vehicles[0][1]:.2f})")
        
        # 获取最闲车辆的VTP节点坐标列表
        v_min_route_nodes = new_state.vehicle_routes[v_min_id - 1][1:-1]
        v_min_positions = [(self.node[node].latDeg, self.node[node].lonDeg) for node in v_min_route_nodes if node in self.node]
    
        epsilon = 1e-6
        # ----------------------------------------------------------------------
        # 2. VTP破坏模式：shaw破坏策略
        # ----------------------------------------------------------------------
        if mode == 'vtp':
            # --- 步骤 2: 随机选择“种子VTP” (必须在最忙的车上) ---
            v_max_route_nodes = new_state.vehicle_routes[v_max_id - 1][1:-1]
            if not v_max_route_nodes:
                print("  > 警告: 最忙车辆没有可破坏的VTP节点。")
                return new_state
            
            seed_vtp_node = self.rng.choice(v_max_route_nodes)
            seed_key = (v_max_id, seed_vtp_node)
            seed_pos = (self.node[seed_vtp_node].latDeg, self.node[seed_vtp_node].lonDeg)

            # --- 步骤 3: 计算所有其他VTP的“重平衡相关性”分数 ---
            relatedness_scores = []
            # 收集所有VTP (除了种子)
            all_other_vtps = [(v_id, node) for v_id, route in enumerate(new_state.vehicle_routes) for node in route[1:-1] if (v_id + 1, node) != seed_key]
            
            if not all_other_vtps:
                return new_state # 只有一个VTP，无法破坏

            # 归一化因子
            max_dist_seed = 0
            max_dist_idle = 0
            temp_scores = []
            
            for v_id, vtp_node in all_other_vtps:
                pos = (self.node[vtp_node].latDeg, self.node[vtp_node].lonDeg)
                
                # a. 与种子的地理距离
                dist_seed = math.sqrt((pos[0] - seed_pos[0])**2 + (pos[1] - seed_pos[1])**2)
                max_dist_seed = max(max_dist_seed, dist_seed)
                
                # b. 与最闲车辆路线的最短距离
                dist_idle = float('inf')
                if not v_min_positions: # 如果最闲车辆没有VTP
                    dist_idle = 0 # 设为0或一个中性值
                else:
                    for idle_pos in v_min_positions:
                        dist_idle = min(dist_idle, math.sqrt((pos[0] - idle_pos[0])**2 + (pos[1] - idle_pos[1])**2))
                max_dist_idle = max(max_dist_idle, dist_idle)
                
                temp_scores.append({'key': (v_id, vtp_node), 'dist_seed': dist_seed, 'dist_idle': dist_idle})

            # --- 步骤 4: 归一化并计算最终相关性分数 (越低越相关) ---
            w_seed = 0.6 # 60% 权重给“地理邻近性”
            w_idle = 0.4 # 40% 权重给“靠近最闲车”
            
            for item in temp_scores:
                norm_dist_seed = item['dist_seed'] / (max_dist_seed + epsilon)
                norm_dist_idle = item['dist_idle'] / (max_dist_idle + epsilon)
                
                # 最终分数：越近越好
                score_shaw_rebalance = w_seed * norm_dist_seed + w_idle * norm_dist_idle
                relatedness_scores.append({'key': item['key'], 'score': score_shaw_rebalance})

            # 按“重平衡相关性”分数升序排序
            relatedness_scores.sort(key=lambda x: x['score'])
            
            # --- 步骤 5: 选择目标 ---
            # num_to_remove = self.vtp_destroy_quantity['shaw'] # e.g., 3
            num_to_remove = 10 # e.g., 3
            # num_to_remove = min(num_to_remove, len(active_vtps_keys)+1) # +1 包括种子
            
            num_neighbors_to_remove = num_to_remove - 1
            vtps_to_destroy = [seed_key]
            if num_neighbors_to_remove > 0:
                vtps_to_destroy.extend([item['key'] for item in relatedness_scores[:num_neighbors_to_remove]])
            
            print(f"  > Shaw重平衡: 种子VTP {seed_key}，计划移除集群: {vtps_to_destroy}")


            # --- 步骤 5: 执行破坏 (包含时间约束检查) ---
            destroyed_customers_info = new_state.destroyed_customers_info 
            # 使用 getattr 安全获取属性，如果不存在则初始化为空字典
            destroyed_vts_info = getattr(new_state, 'destroyed_vts_info', {}) 
            # vehicle_task_data = new_state.vehicle_task_data # 直接在 new_state 上修改

            destroyed_vtp_count = 0
            actual_destroyed_vtps = [] 

            # 开始执行vtp节点任务的破坏策略
            destroyed_customers_info = {}  # 用于存储被破坏的客户节点信息
            destroyed_vtp_count = 0  # 实际破坏的VTP节点数量
            max_attempts = len(vtps_to_destroy) * 2  # 最大尝试次数，避免无限循环
            attempt_count = 0
            
            # 创建候选节点池的副本，用于按优先级选择
            candidate_vtps = vtps_to_destroy.copy()
            # len_vtp_destroy = len(candidate_vtps)
            while destroyed_vtp_count < self.vtp_destroy_quantity['shaw'] and candidate_vtps and attempt_count < max_attempts:
                attempt_count += 1
                
                # 从候选池中选择下一个VTP节点（按成本效益比排序）
                if not candidate_vtps:
                    print(f"候选池已空，无法继续破坏VTP节点")
                    break
                    
                vehicle_index, vtp_node = candidate_vtps.pop(0)  # 按优先级顺序选择
                vehicle_id = vehicle_index + 1
                if vtp_node not in new_state.rm_empty_vehicle_route[vehicle_id-1]:
                    # candidate_vtps = [top_k_candidates_keys[attempt_count]]
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
                temp_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)  # 用于链式删除分析
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
                    # candidate_vtps = candidate_keys_sorted[attempt_count]
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
            new_state.vehicle_routes = [route[:] for route in new_state.rm_empty_vehicle_route]  # vtp节点被破坏后重更新
            new_state.rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(new_state.vehicle_routes)
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            # print(f"破坏后剩余VTP节点: {sum(len(route) - 2 for route in new_state.vehicle_routes)}")  # 减去起点和终点
            # print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== VTP破坏阶段完成 ===\n")
        else:
            # 开始执行客户点层面的破坏策略
            print("  > [破坏模式]: 综合最差破坏 (客户模式 - Pareto)")
        
            # 3.1 收集所有已服务客户
            if not current_customers:
                print("  > 警告: 没有已服务的客户可供破坏。")
                return new_state
                
            # --- 步骤 1: (可选) 计算车辆负载分数 ---
            #     为了计算“负载相关性”，我们需要知道哪些车辆是繁忙的
            #     我们复用 _calculate_vehicle_load 辅助函数
            try:
                vehicle_load_scores = self._calculate_vehicle_load(new_state)
                max_load_score = max(vehicle_load_scores.values()) if vehicle_load_scores else 1.0
            except Exception as e:
                print(f"  > 警告: Shaw客户破坏 - 无法计算车辆负载: {e}。负载指标将设为0。")
                vehicle_load_scores = {}
                max_load_score = 1.0

            # --- 步骤 2: 随机选择一个“种子”客户 ---
            seed_customer = self.rng.choice(current_customers)
            seed_pos = (self.node[seed_customer].latDeg, self.node[seed_customer].lonDeg)
            print(f"  > Shaw种子客户: {seed_customer}")

            # --- 步骤 3: 计算所有其他客户的“智能相关性”分数 ---
            relatedness_scores_cust = []
            all_other_customers = [c for c in current_customers if c != seed_customer]
            
            if not all_other_customers:
                print("  > 警告: 只有一个客户，无法执行Shaw破坏。")
                return new_state # 只有一个客户，无法形成集群

            # 归一化因子 (先计算最大值)
            max_dist = 0
            max_cost = 0
            temp_scores_c = []
            epsilon = 1e-6

            for k in all_other_customers:
                # a. 地理距离
                pos_k = (self.node[k].latDeg, self.node[k].lonDeg)
                dist_k = math.sqrt((pos_k[0] - seed_pos[0])**2 + (pos_k[1] - seed_pos[1])**2)
                max_dist = max(max_dist, dist_k)
                
                # b. 成本指标
                cost_k = new_state.uav_cost.get(k, 0)
                max_cost = max(max_cost, cost_k)
                
                # c. 车辆负载指标
                load_k = 0.0
                try:
                    assignment_k = new_state.customer_plan[k]
                    lv_k, rv_k = assignment_k[4], assignment_k[5]
                    load_k = (vehicle_load_scores.get(lv_k, 0) + vehicle_load_scores.get(rv_k, 0)) / 2.0
                except (KeyError, IndexError):
                    pass # 忽略计算错误
                
                temp_scores_c.append({'key': k, 'dist': dist_k, 'cost': cost_k, 'load': load_k})

            # --- 步骤 4: 归一化并计算最终相关性分数 (越低越相关) ---
            # 权重 (超参数，可调)
            w_dist = 0.5  # 地理邻近性
            w_cost = 0.3  # 成本相关性
            w_load = 0.2  # 负载相关性
            
            for item in temp_scores_c:
                # 归一化 (值越低越相关)
                norm_dist = item['dist'] / (max_dist + epsilon)
                # 归一化 (成本越高 -> 值越低 -> 越相关)
                norm_cost = 1.0 - (item['cost'] / (max_cost + epsilon))
                # 归一化 (负载越高 -> 值越低 -> 越相关)
                norm_load = 1.0 - (item['load'] / (max_load_score + epsilon))
                
                score_shaw = w_dist * norm_dist + w_cost * norm_cost + w_load * norm_load
                relatedness_scores_cust.append({'key': item['key'], 'score': score_shaw})

            # --- 步骤 5: Top-K 选择 (选择 k-1 个邻居) ---
            # 按“智能相关性”分数升序排序
            relatedness_scores_cust.sort(key=lambda x: x['score'])
            
            # 确定破坏数量 (与您原代码一致)
            n = len(current_customers)
            num_to_remove = self.rng.integers(
                max(1, int(n * self.customer_destroy_ratio[0])),
                max(2, int(n * self.customer_destroy_ratio[1])) + 1
            )
            num_to_remove = min(num_to_remove, n)
            
            num_neighbors_to_remove = num_to_remove - 1
            
            # 最终移除列表 = 种子 + (k-1)个最相关的邻居
            customers_to_destroy = [seed_customer]
            if num_neighbors_to_remove > 0:
                customers_to_destroy.extend([item['key'] for item in relatedness_scores_cust[:num_neighbors_to_remove]])

            print(f"  > 计划移除 {len(customers_to_destroy)} 个智能相关客户: {customers_to_destroy}")

            # --- 步骤 5: 执行破坏 (与您原有的框架一致) ---
            destroyed_customers_info = new_state.destroyed_customers_info
            vehicle_task_data = new_state.vehicle_task_data
            
            # 4. 移除这些客户点及相关无人机任务
            for customer in customers_to_destroy:
                if customer in new_state.customer_plan:
                    # 删除每个客户点需要检测时间约束
                    temp_customer_plan = {k: v for k, v in new_state.customer_plan.items()}
                    temp_vehicle_routes = [route[:] for route in new_state.vehicle_routes]
                    temp_vehicle_task_data = deep_copy_vehicle_task_data(new_state.vehicle_task_data)
                    temp_assignment = new_state.customer_plan[customer]
                    temp_chain_tasks = find_chain_tasks(temp_assignment, temp_customer_plan, temp_vehicle_routes, temp_vehicle_task_data)
                    temp_customer_plan.pop(customer, None)
                    for chain_customer, chain_assignment in temp_chain_tasks:
                        temp_customer_plan.pop(chain_customer, None)
                    temp_rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_routes)
                    if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                        continue

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
                            from task_data import deep_remove_vehicle_task
                            print(f"链式删除客户点 {chain_customer}")
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)

            # 5. 更新空跑节点等状态
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
                temp_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)  # 用于链式删除分析
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
                            from task_data import deep_remove_vehicle_task
                            print(f"链式删除客户点 {chain_customer}")
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes)

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

    def validate_customer_plan(self, vehicle_routes, customer_plan, base_drone_assignment):
        """
        验证给定的 customer_plan 是否符合无人机的顺序和状态约束 (增强版)。
        模拟无人机在车辆路线上的状态变化，检测多种逻辑冲突并清晰报告。

        Args:
            vehicle_routes (list or dict): 车辆路线列表/字典。
            customer_plan (dict): {customer: (drone_id, ln, cn, rn, lv, rv)}
            base_drone_assignment (dict): {vehicle_id: [drone_id1, drone_id2, ...]}

        Returns:
            bool: 如果计划有效则返回 True，否则返回 False 并打印详细错误信息。
        """
        is_valid = True # 初始假设计划有效

        # ----------------------------------------------------------------------
        # 1. 初始化无人机状态 (使用深拷贝以隔离验证过程)
        # ----------------------------------------------------------------------
        # drone_state: 记录每个无人机的详细状态
        # 'location': vehicle_id (在车上) 或 'flying' (飞行中) 或 'depot' (初始在仓库)
        # 'last_event_node': (vehicle_id, node_id) 上次发生事件的节点
        # 'current_task': customer_id 正在执行的任务 (None 如果不在执行任务)
        drone_state = {}
        all_drones = set(d for drones in base_drone_assignment.values() for d in drones)
        
        # 尝试从 customer_plan 中也获取无人机，以防 base_assignment 不全
        try:
            drones_in_plan = set(assignment[0] for assignment in customer_plan.values())
            all_drones.update(drones_in_plan)
        except (TypeError, IndexError):
            print("  > 警告: customer_plan 格式可能不完全正确，无法提取所有无人机ID。")
            
        for drone_id in all_drones:
            drone_state[drone_id] = {'location': 'depot', 'last_event_node': None, 'current_task': None}

        for vehicle_id, drones_on_vehicle in base_drone_assignment.items():
            for drone_id in drones_on_vehicle:
                if drone_id in drone_state:
                    drone_state[drone_id]['location'] = vehicle_id # 初始在对应的车上
                else:
                    print(f"  > 警告: 基础分配中的无人机 {drone_id} 未在状态字典中初始化。")

        # ----------------------------------------------------------------------
        # 2. 构建任务查找表 (按节点组织)
        # ----------------------------------------------------------------------
        launch_tasks_at_node = {}    # {(vehicle_id, vtp_node): [(drone_id, customer, assignment_tuple), ...]}
        recovery_tasks_at_node = {} # {(vehicle_id, vtp_node): [(drone_id, customer, assignment_tuple), ...]}

        for customer, assignment in customer_plan.items():
            try:
                # 检查 assignment 结构是否有效
                if len(assignment) != 6:
                    raise ValueError("Assignment tuple length mismatch")
                drone_id, ln, _, rn, lv, rv = assignment
                
                # 检查无人机ID是否存在
                if drone_id not in drone_state:
                    print(f"  > 错误: 客户 {customer} 的任务引用了未知的无人机 ID: {drone_id}。")
                    is_valid = False
                    continue # 跳过这个无效任务

                launch_key = (lv, ln)
                if launch_key not in launch_tasks_at_node: launch_tasks_at_node[launch_key] = []
                launch_tasks_at_node[launch_key].append((drone_id, customer, assignment))
                
                recovery_key = (rv, rn)
                if recovery_key not in recovery_tasks_at_node: recovery_tasks_at_node[recovery_key] = []
                recovery_tasks_at_node[recovery_key].append((drone_id, customer, assignment))
            except (TypeError, ValueError, IndexError) as e:
                print(f"  > 错误: customer_plan 中客户 {customer} 的任务数据格式无效: {assignment}。错误: {e}")
                is_valid = False
                # return False # 可以选择提前退出

        if not is_valid: return False

        # ----------------------------------------------------------------------
        # 3. 模拟车辆行驶并验证无人机状态变化
        #    【重要】: 此模拟基于节点顺序，不考虑精确时间，检查的是逻辑顺序冲突。
        # ----------------------------------------------------------------------
        # 处理 vehicle_routes 是列表还是字典
        processed_routes = []
        # ... (与上一版本相同的代码，将 routes 转换为 [(vid, route_list), ...]) ...

        for vehicle_id, route in processed_routes:
            if len(route) < 2: continue
            print(f"\n--- 正在验证车辆 {vehicle_id} 的路线: {route} ---")
            
            # 遍历路线中的每个 VTP 节点 (跳过起点和终点 Depot)
            for node_idx in range(1, len(route) - 1):
                vtp_node = route[node_idx]
                node_key = (vehicle_id, vtp_node)
                
                print(f"  节点 {vtp_node} (索引 {node_idx}):")

                # --- 3.1 处理在该节点的【回收】任务 (必须先于发射处理) ---
                if node_key in recovery_tasks_at_node:
                    for drone_id, customer, assignment in recovery_tasks_at_node[node_key]:
                        print(f"    - 检查回收: 无人机 {drone_id} (来自客户 {customer})")
                        state = drone_state[drone_id]
                        
                        # 【验证规则 1】: 无人机必须处于飞行状态 ('flying')
                        if state['location'] != 'flying':
                            error_msg = (f"    -> !! 回收冲突 !! 无人机 {drone_id} 试图在节点 {vtp_node} (车辆 {vehicle_id}) 回收，"
                                        f"但其当前状态是 '{state['location']}' (应为 'flying')。")
                            if state['last_event_node']:
                                error_msg += f" 上次事件发生在 {state['last_event_node']}."
                            print(error_msg)
                            is_valid = False
                        
                        # 【验证规则 2】: 回收的任务必须是当前正在执行的任务
                        elif state['current_task'] != customer:
                            print(f"    -> !! 任务不匹配 !! 无人机 {drone_id} 试图回收服务客户 {customer} 的任务，"
                                f"但记录显示它正在执行的任务是 {state['current_task']}。")
                            is_valid = False
                        
                        else:
                            # 更新状态：无人机现在在这辆车上
                            state['location'] = vehicle_id
                            state['last_event_node'] = node_key
                            state['current_task'] = None # 任务完成
                            print(f"      状态更新: 无人机 {drone_id} 已回收至车辆 {vehicle_id}。")

                # --- 3.2 处理在该节点的【发射】任务 ---
                if node_key in launch_tasks_at_node:
                    for drone_id, customer, assignment in launch_tasks_at_node[node_key]:
                        print(f"    - 检查发射: 无人机 {drone_id} (飞往客户 {customer})")
                        state = drone_state[drone_id]

                        # 【验证规则 3】: 无人机必须在当前车辆上才能被发射
                        if state['location'] != vehicle_id:
                            error_msg = (f"    -> !! 发射冲突 !! 无人机 {drone_id} 试图从节点 {vtp_node} (车辆 {vehicle_id}) 发射，"
                                        f"但其当前状态是 '{state['location']}' (应在车辆 {vehicle_id} 上)。")
                            if state['last_event_node']:
                                error_msg += f" 上次事件发生在 {state['last_event_node']}."
                            print(error_msg)
                            is_valid = False
                        
                        # 【验证规则 4】: 无人机不能已经在执行任务（即上次发射后未回收）
                        elif state['current_task'] is not None:
                            print(f"    -> !! 状态冲突 !! 无人机 {drone_id} 试图发射新任务 (客户 {customer})，"
                                f"但它仍在执行上一个任务 (客户 {state['current_task']})。")
                            is_valid = False

                        else:
                            # 更新状态：无人机现在处于飞行状态，并记录当前任务
                            state['location'] = 'flying'
                            state['last_event_node'] = node_key
                            state['current_task'] = customer
                            print(f"      状态更新: 无人机 {drone_id} 已发射，状态为 'flying'，目标客户 {customer}。")

        # ----------------------------------------------------------------------
        # 4. 最终全局检查：所有任务是否都已完成？
        # ----------------------------------------------------------------------
        unfinished_drones = []
        for drone_id, state in drone_state.items():
            if state['location'] == 'flying' or state['current_task'] is not None:
                unfinished_drones.append((drone_id, state['current_task']))

        if unfinished_drones:
            print(f"\n  > 警告: 验证结束时，以下无人机仍处于飞行状态或有未完成的任务:")
            for d_id, c_id in unfinished_drones:
                print(f"    - 无人机 {d_id} (目标客户: {c_id})")
            # is_valid = False # 取决于您的业务规则是否允许任务不闭环

        # ----------------------------------------------------------------------
        # 5. 返回最终验证结果
        # ----------------------------------------------------------------------
        if is_valid:
            print("\n=== customer_plan 约束验证通过 ===")
        else:
            print("\n=== customer_plan 存在约束冲突 ===")
            
        return is_valid


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
        finish_flag = False
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
                            finish_flag = True
                            break
                        
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
                if finish_flag: # 如果已经找到原始发射车辆，则停止跟踪
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
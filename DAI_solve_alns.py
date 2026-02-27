import copy
import numpy as np
import numpy.random as rnd
from collections import defaultdict
import time
from parseCSV import *
from utils_shared import *
from collections import defaultdict
import copy
from initialize import init_agent, initialize_drone_vehicle_assignments
from create_vehicle_route import *
# from insert_plan import greedy_insert_feasible_plan
import os
# from main import find_keys_and_indices
from mfstsp_heuristic_1_partition import *
from mfstsp_heuristic_2_asgn_uavs import *
from mfstsp_heuristic_3_timing import *
from task_data import deep_remove_vehicle_task
from local_search import *
from rm_node_sort_node import rm_empty_node
from task_data import *
from visualization_best import visualize_plan
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from destroy_repair_operator import *
from initialize import deep_copy_vehicle_task_data
from cost_y import calculate_plan_cost
from create_vehicle_route import DiverseRouteGenerator
from constraints_satisfied import is_constraints_satisfied
from random import seed
from fast_alns_solver import FastMfstspState
from drone_plan_constraint_checker import validate_customer_plan_constraints
# 该策略算法模拟传统ALNS算法，不考虑插入模式。
class DAI_IncrementalALNS:
    """传统增量式ALNS求解器 - 使用修改记录和回滚机制"""
    
    def __init__(self, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, 
    N_zero, N_plus, A_total, A_cvtp, A_vtp, 
		A_aerial_relay_node, G_air, G_ground,air_matrix, ground_matrix, air_node_types, 
        ground_node_types, A_c, xeee, customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName,
        iter, max_iterations, max_runtime=60, seed=None, algo_seed=None):
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
        self.customer_time_windows_h = customer_time_windows_h
        self.early_arrival_cost = early_arrival_cost
        self.late_arrival_cost = late_arrival_cost
        self.iter = iter # 获得仿真实验次数
        self.problemName = problemName # 获得问题名称
        # self.max_iterations = max_iterations
        self.max_iterations = max_iterations
        self.temperature = max_iterations
        self.initial_temperature = max_iterations
        # self.temperature = 500.0
        # self.initial_temperature = 500.0
        self.max_runtime = max_runtime
        self.rng = rnd.default_rng(seed)
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
        self.destroy_operators = [self.destroy_random_removal, self.destroy_worst_removal, self.proximit_removal, self.window_removal, self.shaw_removal]
        self.repair_operators = [self.repair_random_greedy_insertion, self.repair_closest_insertion, self.repair_regret_insertion, self.repair_nearby_random_insertion]
        self.M_PENALTY = 1000
        self.segment_length = 50        # 每50次迭代更新一次权重（经典做法）
        self.min_weight = 0.1           # 防止权重衰减到0
        self.vtp_mode_prob = 0.5        # 可选：传统ALNS里用概率决定是否force_vtp_mode（你也可设0只做客户破坏）

        # 单层权重：destroy / repair
        self.destroy_weights = {op.__name__: 1.0 for op in self.destroy_operators}
        self.repair_weights  = {op.__name__: 1.0 for op in self.repair_operators}

        # 与solve中的分段权重更新保持一致的学习率参数
        self.reaction_factor = 0.5
        # 与保存接口保持一致
        self.strategy_weights = {'single_layer': 1.0}
        self.operator_weights = {
            'single_layer': {
                'destroy': dict(self.destroy_weights),
                'repair': dict(self.repair_weights),
            }
        }

        # 接受策略评分：四档奖励（SA接受劣解参与学习，拒绝施加乘法惩罚）
        self.reward_scores = {
            'new_best': 12.0,           # σ1：发现全局新最优
            'better_than_current': 8.0, # σ2：优于当前解
            'accepted_worse': 3.0,      # σ3：SA接受劣解（探索成功）
            'rejected': 0.0,            # σ4：拒绝（触发乘法衰减惩罚）
        }
        self.sigma1 = self.reward_scores['new_best']
        self.sigma2 = self.reward_scores['better_than_current']
        self.sigma3 = self.reward_scores['accepted_worse']
        self.sigma4 = self.reward_scores['rejected']
        if algo_seed is None:
            algo_seed = 42
        self.rng = rnd.default_rng(algo_seed)
        random.seed(algo_seed)
        np.random.seed(algo_seed)
        self.base_drone_assigment = self.base_drone_assigment()
        
    def base_drone_assigment(self):
        """
        基础无人机分配函数 (修正版)
        按照self.V中的实际ID，均匀切片分配给self.T中的车辆

        Returns:
        dict: 车辆ID为key，无人机ID列表为value的字典
        """
        # 1. 获取车辆和无人机的实际ID列表
        vehicle_ids = self.T  # 例如 [1, 2, 3, 4, 5]
        drone_ids = self.V    # 例如 [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        num_vehicles = len(vehicle_ids)
        num_drones = len(drone_ids)

        base_assignment = {}

        # 2. 计算分配逻辑
        # 每个车辆最少分到的数量
        drones_per_vehicle = num_drones // num_vehicles
        # 余数：前 n 个车辆需要多承载 1 架
        remainder = num_drones % num_vehicles

        # 3. 开始分配
        # 维护一个指针，指向 self.V 中当前还未分配的无人机起始位置
        current_drone_idx = 0

        # 遍历每一个车辆ID（同时获取它的索引 i 用于判断余数分配）
        for i, v_id in enumerate(vehicle_ids):
            # 计算当前车辆 v_id 应该分几架
            # 如果当前索引 i 小于余数，说明它是前几个需要多拿1架的车辆
            count = drones_per_vehicle + 1 if i < remainder else drones_per_vehicle

            # 从 self.V 中切片取出对应数量的实际无人机ID
            # 例如：第一次循环取 drone_ids[0 : 2] -> [7, 8]
            assigned_drones = drone_ids[current_drone_idx : current_drone_idx + count]

            # 存入字典
            base_assignment[v_id] = assigned_drones

            # 移动指针，为下一辆车做准备
            current_drone_idx += count

        # 打印调试信息（可选）
        # print(f"修正分配结果: {base_assignment}")

        return base_assignment

    def repair_closest_insertion(self, state, strategic_bonus=0, num_destroyed=1, force_vtp_mode=True):
        """
        最近邻插入算子 (Closest Insertion) - 对应 Algorithm 10
        逻辑流：
        1. 随机选客户 c。
        2. 找解中离它最近的客户 c'，定位 c' 的车辆 r。
        3. 仅尝试将 c 插入车辆 r，成功则执行，失败则放入 cN 池。
        4. 最后对 cN 池的所有客户执行随机贪婪插入兜底 (Algorithm 8)。
        """
        repaired_state = state.fast_copy()
        repaired_state.repair_objective = 0
        destroy_node = list(state.destroyed_customers_info.keys())  # L 集合
        insert_plan = []
        
        # 兜底池 cN (Algorithm 10, Line 1)
        cN = []

        force_vtp_mode = True
        if force_vtp_mode:
            
            # ==========================================
            # Phase 1: Closest Insertion (Algorithm 10, Line 2-9)
            # ==========================================
            while len(destroy_node) > 0:
                # Line 3: 随机选一个客户
                target_idx = self.rng.integers(0, len(destroy_node))
                customer = destroy_node.pop(target_idx)
                
                # 获取该破坏节点的物理坐标
                customer_info = repaired_state.destroyed_customers_info.get(customer)
                # 防止异常，兜底获取物理节点 (索引2为 customer_node)
                c_node = customer_info[2] if customer_info else customer

                # Line 4 & 5: 寻找当前解中离 c 最近的客户 c'，及其所在的车辆 r
                current_customers = list(repaired_state.customer_plan.keys())
                target_vehicle_id = None
                
                if current_customers:
                    min_dist = float('inf')
                    closest_c_prime = None
                    for c_prime in current_customers:
                        # 🔴 核心修复：提取目标客户点 c_prime 分配的无人机编号
                        c_prime_uav_id = repaired_state.customer_plan[c_prime][0]
                        c_prime_node = repaired_state.customer_plan[c_prime][2]
                        
                        # 🔴 核心修复：加入 uav_id 层级，并调用 .totalDistance
                        dist = self.uav_travel[c_prime_uav_id][c_node][c_prime_node].totalDistance
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_c_prime = c_prime
                            
                    if closest_c_prime is not None:
                        # 找到 c' 所在的车辆 (取其 launch_vehicle 作为目标路线 r)
                        target_vehicle_id = repaired_state.customer_plan[closest_c_prime][4]

                # --- 准备评估 ---
                try:
                    current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                    base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                        repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                        self.vehicle, self.customer_time_windows_h, 
                        self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                    )
                except Exception:
                    base_total_cost = float('inf')

                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                customer_candidates = []
                
                # 1. 评估传统插入
                traditional_result, is_heuristic_swap = self._evaluate_traditional_insertion(
                    customer, vehicle_route, vehicle_task_data, vehicle_arrive_time,
                    base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state
                )

                if traditional_result[0] or traditional_result[1] is not None:
                    traditional_cost, traditional_scheme = traditional_result
                    if is_heuristic_swap:
                        customer_candidates.append({
                            'customer': customer, 'scheme': traditional_scheme, 'cost': traditional_cost, 
                            'win_cost': traditional_scheme['win_cost'], 'total_cost': traditional_scheme['total_cost'], 
                            'type': 'heuristic_swap', 'vtp_node': None
                        })
                    else:
                        customer_candidates.append({
                            'customer': customer, 'scheme': traditional_scheme, 'cost': traditional_cost, 
                            'win_cost': traditional_cost, 'total_cost': traditional_cost + base_total_cost, 
                            'type': 'traditional', 'vtp_node': None
                        })
                
                # 2. 评估VTP扩展
                vtp_result, vtp_infor = self._evaluate_vtp_expansion_insertion(
                    customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                )
                if vtp_infor is not None:
                    vtp_node, vtp_insert_vehicle_id, vtp_insert_index = vtp_infor
                    if vtp_result is not None:
                        vtp_cost, vtp_scheme = vtp_result
                        customer_candidates.append({
                            'customer': customer, 'scheme': vtp_scheme, 'cost': vtp_cost, 'total_cost': vtp_cost,
                            'type': 'vtp_expansion', 'vtp_node': vtp_node, 'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                            'vtp_insert_index': vtp_insert_index, 'original_cost': vtp_cost
                        })

                customer_candidates = [item for item in customer_candidates if item['scheme'] is not None]
                
                # >>> 核心修改：过滤出只属于车辆 r (target_vehicle_id) 的方案 <<<
                if target_vehicle_id is not None:
                    filtered_candidates = []
                    for cand in customer_candidates:
                        if cand['type'] == 'heuristic_swap':
                            # 交换策略看新生成的 scheme 是否在目标车上
                            if cand['scheme']['new_scheme'][4] == target_vehicle_id or cand['scheme']['new_scheme'][5] == target_vehicle_id:
                                filtered_candidates.append(cand)
                        else:
                            # 传统/扩展看分配的 launch/recovery 车辆是否在目标车上
                            if cand['scheme'][4] == target_vehicle_id or cand['scheme'][5] == target_vehicle_id:
                                filtered_candidates.append(cand)
                    customer_candidates = filtered_candidates
                # >>> 过滤结束 <<<

                candidates_plan = sorted(customer_candidates, key=lambda x: x['total_cost'])
                success = False

                # 尝试执行局部插入 (Line 6: Attempt Best Insertion(c, r))
                for candidate in candidates_plan:
                    # ==== 执行逻辑完全复用你的底层代码 ====
                    if candidate['type'] == 'traditional':
                        best_scheme = candidate['scheme']
                        best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue
                            
                        repaired_state.customer_plan[customer_node] = best_scheme
                        if drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[drone_id] = []
                        repaired_state.uav_assignments[drone_id].append(best_scheme)
                        if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                        repaired_state.uav_cost[customer_node] = best_cost
                        
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                        insert_plan.append((customer, best_scheme, best_cost, 'traditional'))
                        success = True
                        break

                    elif candidate['type'] == 'heuristic_swap':
                        orig_scheme = candidate['scheme']['orig_scheme']
                        new_scheme = candidate['scheme']['new_scheme']
                        orig_cost = candidate['scheme']['orig_cost']
                        new_cost = candidate['scheme']['new_cost']
                        orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = orig_scheme
                        new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_scheme
                        delete_customer = orig_customer
                        
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        delete_task_plan = temp_customer_plan[orig_customer]
                        del temp_customer_plan[orig_customer]
                        temp_customer_plan[orig_customer] = orig_scheme
                        temp_customer_plan[new_customer] = new_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue  
                        else:
                            del repaired_state.customer_plan[delete_customer]
                            repaired_state.customer_plan[orig_customer] = orig_scheme
                            repaired_state.customer_plan[new_customer] = new_scheme
                            
                            if orig_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[orig_drone_id] = []
                            repaired_state.uav_assignments[orig_drone_id].append(orig_scheme)
                            if new_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[new_drone_id] = []
                            repaired_state.uav_assignments[new_drone_id].append(new_scheme)
                            del repaired_state.uav_cost[delete_customer]
                            repaired_state.uav_cost[orig_customer] = orig_cost
                            repaired_state.uav_cost[new_customer] = new_cost
                            
                            vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                            orig_launch_time = temp_rm_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                            new_launch_time = temp_rm_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                            if orig_launch_time <= new_launch_time:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                            else:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                
                            insert_plan.append((delete_customer, orig_scheme, orig_cost, 'heuristic_swap'))
                            insert_plan.append((customer, new_scheme, new_cost, 'heuristic_swap'))
                            success = True
                            break

                    elif candidate['type'] == 'vtp_expansion':
                        vtp_node = candidate['vtp_node']
                        vtp_insert_index = candidate['vtp_insert_index']
                        vtp_insert_vehicle_id = candidate['vtp_insert_vehicle_id']
                        best_scheme = candidate['scheme']
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        original_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)

                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        temp_vehicle_route = [route[:] for route in vehicle_route]
                        temp_route = temp_vehicle_route[vtp_insert_vehicle_id - 1]
                        temp_route.insert(vtp_insert_index, vtp_node)
                        temp_vehicle_route[vtp_insert_vehicle_id - 1] = temp_route
                        repaired_state.temp_vehicle_routes = temp_vehicle_route
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue
                        else:
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
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                            
                            repaired_state.customer_plan[customer_node] = best_scheme
                            if drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(best_scheme)
                            if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = original_cost
                            
                            repaired_state.rm_empty_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
                            repaired_state.rm_empty_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                            insert_plan.append((customer, best_scheme, original_cost, 'vtp_expansion'))
                            success = True
                            break
                    # ==== 执行逻辑复用结束 ====

                # Line 7: 如果插入该路线失败，将其丢入 cN 池
                if not success:
                    cN.append(customer)
            # ==========================================
            # Phase 2: Fallback (Algorithm 10, Line 10-12)
            # 逻辑：对 cN 池执行无过滤的贪婪随机插入 (Algo 8)
            # ==========================================
            if len(cN) > 0:
                print(f"  > [最近邻插入] 存在 {len(cN)} 个客户需要兜底修复...")
                while len(cN) > 0:
                    target_idx = self.rng.integers(0, len(cN))
                    customer = cN.pop(target_idx)
                    
                    try:
                        current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                        base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                            repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                            self.vehicle, self.customer_time_windows_h, 
                            self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                        )
                    except Exception:
                        base_total_cost = float('inf')

                    vehicle_route = repaired_state.vehicle_routes
                    vehicle_task_data = repaired_state.vehicle_task_data
                    vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                    customer_candidates = []
                    
                    traditional_result, is_heuristic_swap = self._evaluate_traditional_insertion(
                        customer, vehicle_route, vehicle_task_data, vehicle_arrive_time,
                        base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state
                    )

                    if traditional_result[0] or traditional_result[1] is not None:
                        traditional_cost, traditional_scheme = traditional_result
                        if is_heuristic_swap:
                            customer_candidates.append({
                                'customer': customer, 'scheme': traditional_scheme, 'cost': traditional_cost, 
                                'win_cost': traditional_scheme['win_cost'], 'total_cost': traditional_scheme['total_cost'], 
                                'type': 'heuristic_swap', 'vtp_node': None
                            })
                        else:
                            customer_candidates.append({
                                'customer': customer, 'scheme': traditional_scheme, 'cost': traditional_cost, 
                                'win_cost': traditional_cost, 'total_cost': traditional_cost + base_total_cost, 
                                'type': 'traditional', 'vtp_node': None
                            })
                    
                    vtp_result, vtp_infor = self._evaluate_vtp_expansion_insertion(
                        customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                    )
                    if vtp_infor is not None:
                        vtp_node, vtp_insert_vehicle_id, vtp_insert_index = vtp_infor
                        if vtp_result is not None:
                            vtp_cost, vtp_scheme = vtp_result
                            customer_candidates.append({
                                'customer': customer, 'scheme': vtp_scheme, 'cost': vtp_cost, 'total_cost': vtp_cost,
                                'type': 'vtp_expansion', 'vtp_node': vtp_node, 'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                                'vtp_insert_index': vtp_insert_index, 'original_cost': vtp_cost
                            })

                    # >>> 注意：这里不再进行过滤限制，相当于开新路线或其他路线都允许 <<<
                    customer_candidates = [item for item in customer_candidates if item['scheme'] is not None]
                    candidates_plan = sorted(customer_candidates, key=lambda x: x['total_cost'])
                    
                    success = False
                    for candidate in candidates_plan:
                        # ==== Phase 2 的代码执行部分 ====
                        if candidate['type'] == 'traditional':
                            best_scheme = candidate['scheme']
                            best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
                            drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                            
                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            temp_customer_plan[customer_node] = best_scheme
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                            
                            if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue
                                
                            repaired_state.customer_plan[customer_node] = best_scheme
                            if drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(best_scheme)
                            if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = best_cost
                            
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                            insert_plan.append((customer, best_scheme, best_cost, 'traditional'))
                            success = True
                            break

                        elif candidate['type'] == 'heuristic_swap':
                            orig_scheme = candidate['scheme']['orig_scheme']
                            new_scheme = candidate['scheme']['new_scheme']
                            orig_cost = candidate['scheme']['orig_cost']
                            new_cost = candidate['scheme']['new_cost']
                            orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = orig_scheme
                            new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_scheme
                            delete_customer = orig_customer
                            
                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            delete_task_plan = temp_customer_plan[orig_customer]
                            del temp_customer_plan[orig_customer]
                            temp_customer_plan[orig_customer] = orig_scheme
                            temp_customer_plan[new_customer] = new_scheme
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                            
                            if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue  
                            else:
                                del repaired_state.customer_plan[delete_customer]
                                repaired_state.customer_plan[orig_customer] = orig_scheme
                                repaired_state.customer_plan[new_customer] = new_scheme
                                
                                if orig_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[orig_drone_id] = []
                                repaired_state.uav_assignments[orig_drone_id].append(orig_scheme)
                                if new_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[new_drone_id] = []
                                repaired_state.uav_assignments[new_drone_id].append(new_scheme)
                                del repaired_state.uav_cost[delete_customer]
                                repaired_state.uav_cost[orig_customer] = orig_cost
                                repaired_state.uav_cost[new_customer] = new_cost
                                
                                vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                                orig_launch_time = temp_rm_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                                new_launch_time = temp_rm_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                                if orig_launch_time <= new_launch_time:
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                else:
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                    vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                    
                                insert_plan.append((delete_customer, orig_scheme, orig_cost, 'heuristic_swap'))
                                insert_plan.append((customer, new_scheme, new_cost, 'heuristic_swap'))
                                success = True
                                break

                        elif candidate['type'] == 'vtp_expansion':
                            vtp_node = candidate['vtp_node']
                            vtp_insert_index = candidate['vtp_insert_index']
                            vtp_insert_vehicle_id = candidate['vtp_insert_vehicle_id']
                            best_scheme = candidate['scheme']
                            drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                            original_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)

                            temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                            temp_customer_plan[customer_node] = best_scheme
                            temp_vehicle_route = [route[:] for route in vehicle_route]
                            temp_route = temp_vehicle_route[vtp_insert_vehicle_id - 1]
                            temp_route.insert(vtp_insert_index, vtp_node)
                            temp_vehicle_route[vtp_insert_vehicle_id - 1] = temp_route
                            repaired_state.temp_vehicle_routes = temp_vehicle_route
                            temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                            
                            if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue
                            else:
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
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                                
                                repaired_state.customer_plan[customer_node] = best_scheme
                                if drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[drone_id] = []
                                repaired_state.uav_assignments[drone_id].append(best_scheme)
                                if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                                repaired_state.uav_cost[customer_node] = original_cost
                                
                                repaired_state.rm_empty_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
                                repaired_state.rm_empty_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                                insert_plan.append((customer, best_scheme, original_cost, 'vtp_expansion'))
                                success = True
                                break
                    
                    if not success:
                        print(f"客户点 {customer} 的兜底修复方案也失败了")
                        repaired_state.repair_objective = float('inf')
                        repaired_state.destroyed_customers_info = {}
                        return repaired_state, insert_plan

        repaired_state._total_cost = repaired_state.win_total_objective()
        repaired_state.destroyed_customers_info = {}
        
        return repaired_state, insert_plan


    def repair_nearby_random_insertion(self, state, strategic_bonus=0, num_destroyed=1, force_vtp_mode=None):
        """
        基于邻近度的随机插入算子 (Nearby Insertion)
        对应论文描述：找到 5km 内的客户所在的路线，随机选择一个可行位置插入，不考虑成本最小化。
        如果无法插入，则利用所有其他方案进行兜底（相当于开辟新路线）。
        
        Args:
            distance_threshold (float): 论文中提到的 5km 阈值。请根据你 self.uav_travel 的实际单位（公里/米）进行调整。
        """
        repaired_state = state.fast_copy()
        distance_threshold=25.0
        repaired_state.repair_objective = 0
        destroy_node = list(state.destroyed_customers_info.keys())  # L 集合
        insert_plan = []
        force_vtp_mode = True  # 强制开启 VTP 模式，确保在插入评估中包含 VTP 扩展方案
        if force_vtp_mode:
            num_repaired = 0
            
            while len(destroy_node) > 0:
                # 1. 每次随机从移除列表中选择一个客户 (Stage 1, Step 1)
                target_idx = self.rng.integers(0, len(destroy_node))
                customer = destroy_node.pop(target_idx)
                
                # 获取该破坏节点的物理坐标
                customer_info = repaired_state.destroyed_customers_info.get(customer)
                c_node = customer_info[2] if customer_info else customer

                # 2. 识别距离小于 5km 的客户，并锁定它们所在的路线 (Stage 1, Step 2)
                nearby_vehicles = set()
                current_customers = list(repaired_state.customer_plan.keys())
                
                if current_customers:
                    for c_prime in current_customers:
                        c_prime_node = repaired_state.customer_plan[c_prime][2]
                        c_prime_uav_id = repaired_state.customer_plan[c_prime][0]
                        dist = self.uav_travel[c_prime_uav_id][c_node][c_prime_node].totalDistance
                        
                        if dist < distance_threshold:
                            # 将该邻近客户所在的车辆加入目标路线集合
                            nearby_vehicles.add(repaired_state.customer_plan[c_prime][4]) # launch_vehicle
                            nearby_vehicles.add(repaired_state.customer_plan[c_prime][5]) # recovery_vehicle

                # --- 准备评估 ---
                try:
                    current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                    base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                        repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                        self.vehicle, self.customer_time_windows_h, 
                        self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                    )
                except Exception:
                    base_total_cost = float('inf')

                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                all_candidates = []
                
                # [完全复用] 评估传统插入
                traditional_result, is_heuristic_swap = self._evaluate_traditional_insertion(
                    customer, vehicle_route, vehicle_task_data, vehicle_arrive_time,
                    base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state
                )
                if traditional_result[0] or traditional_result[1] is not None:
                    traditional_cost, traditional_scheme = traditional_result
                    if is_heuristic_swap:
                        all_candidates.append({
                            'customer': customer, 'scheme': traditional_scheme, 'cost': traditional_cost, 
                            'win_cost': traditional_scheme['win_cost'], 'total_cost': traditional_scheme['total_cost'], 
                            'type': 'heuristic_swap', 'vtp_node': None
                        })
                    else:
                        all_candidates.append({
                            'customer': customer, 'scheme': traditional_scheme, 'cost': traditional_cost, 
                            'win_cost': traditional_cost, 'total_cost': traditional_cost + base_total_cost, 
                            'type': 'traditional', 'vtp_node': None
                        })
                
                # [完全复用] 评估VTP扩展
                vtp_result, vtp_infor = self._evaluate_vtp_expansion_insertion(
                    customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                )
                if vtp_infor is not None:
                    vtp_node, vtp_insert_vehicle_id, vtp_insert_index = vtp_infor
                    if vtp_result is not None:
                        vtp_cost, vtp_scheme = vtp_result
                        all_candidates.append({
                            'customer': customer, 'scheme': vtp_scheme, 'cost': vtp_cost, 'total_cost': vtp_cost,
                            'type': 'vtp_expansion', 'vtp_node': vtp_node, 'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                            'vtp_insert_index': vtp_insert_index, 'original_cost': vtp_cost
                        })

                all_candidates = [item for item in all_candidates if item['scheme'] is not None]

                # ==========================================
                # 核心机制修改：过滤并随机打乱候选方案
                # ==========================================
                nearby_candidates = []
                fallback_candidates = []
                
                for cand in all_candidates:
                    # 获取该方案的关联车辆
                    if cand['type'] == 'heuristic_swap':
                        v1 = cand['scheme']['new_scheme'][4]
                        v2 = cand['scheme']['new_scheme'][5]
                    else:
                        v1 = cand['scheme'][4]
                        v2 = cand['scheme'][5]
                        
                    # 判断是否属于 5km 内的邻近路线
                    if v1 in nearby_vehicles or v2 in nearby_vehicles:
                        nearby_candidates.append(cand)
                    else:
                        fallback_candidates.append(cand)

                # 论文逻辑："a feasible location is randomly selected" -> 彻底抛弃成本排序，随机打乱！
                if nearby_candidates:
                    self.rng.shuffle(nearby_candidates)
                
                # 兜底逻辑："If insertion is not possible, a new route is opened" -> 按成本排序进行兜底
                if fallback_candidates:
                    fallback_candidates = sorted(fallback_candidates, key=lambda x: x['total_cost'])
                
                # 组合候选方案：优先尝试随机打乱的周边方案，都不行再尝试兜底方案
                candidates_plan = nearby_candidates + fallback_candidates
                # ==========================================

                success = False

                # 尝试执行局部插入 (按打乱或排序后的顺序寻找第一个 feasible 的位置)
                for candidate in candidates_plan:
                    # ==== 执行逻辑完全复用你的底层代码 ====
                    if candidate['type'] == 'traditional':
                        best_scheme = candidate['scheme']
                        best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue
                            
                        repaired_state.customer_plan[customer_node] = best_scheme
                        if drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[drone_id] = []
                        repaired_state.uav_assignments[drone_id].append(best_scheme)
                        if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                        repaired_state.uav_cost[customer_node] = best_cost
                        
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                        insert_plan.append((customer, best_scheme, best_cost, 'traditional'))
                        success = True
                        break

                    elif candidate['type'] == 'heuristic_swap':
                        orig_scheme = candidate['scheme']['orig_scheme']
                        new_scheme = candidate['scheme']['new_scheme']
                        orig_cost = candidate['scheme']['orig_cost']
                        new_cost = candidate['scheme']['new_cost']
                        orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = orig_scheme
                        new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_scheme
                        delete_customer = orig_customer
                        
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        delete_task_plan = temp_customer_plan[orig_customer]
                        del temp_customer_plan[orig_customer]
                        temp_customer_plan[orig_customer] = orig_scheme
                        temp_customer_plan[new_customer] = new_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue  
                        else:
                            del repaired_state.customer_plan[delete_customer]
                            repaired_state.customer_plan[orig_customer] = orig_scheme
                            repaired_state.customer_plan[new_customer] = new_scheme
                            
                            if orig_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[orig_drone_id] = []
                            repaired_state.uav_assignments[orig_drone_id].append(orig_scheme)
                            if new_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[new_drone_id] = []
                            repaired_state.uav_assignments[new_drone_id].append(new_scheme)
                            del repaired_state.uav_cost[delete_customer]
                            repaired_state.uav_cost[orig_customer] = orig_cost
                            repaired_state.uav_cost[new_customer] = new_cost
                            
                            vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                            orig_launch_time = temp_rm_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                            new_launch_time = temp_rm_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                            if orig_launch_time <= new_launch_time:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                            else:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                
                            insert_plan.append((delete_customer, orig_scheme, orig_cost, 'heuristic_swap'))
                            insert_plan.append((customer, new_scheme, new_cost, 'heuristic_swap'))
                            success = True
                            break

                    elif candidate['type'] == 'vtp_expansion':
                        vtp_node = candidate['vtp_node']
                        vtp_insert_index = candidate['vtp_insert_index']
                        vtp_insert_vehicle_id = candidate['vtp_insert_vehicle_id']
                        best_scheme = candidate['scheme']
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        original_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)

                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        temp_vehicle_route = [route[:] for route in vehicle_route]
                        temp_route = temp_vehicle_route[vtp_insert_vehicle_id - 1]
                        temp_route.insert(vtp_insert_index, vtp_node)
                        temp_vehicle_route[vtp_insert_vehicle_id - 1] = temp_route
                        repaired_state.temp_vehicle_routes = temp_vehicle_route
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time): continue
                        else:
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
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                            
                            repaired_state.customer_plan[customer_node] = best_scheme
                            if drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(best_scheme)
                            if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = original_cost
                            
                            repaired_state.rm_empty_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
                            repaired_state.rm_empty_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                            insert_plan.append((customer, best_scheme, original_cost, 'vtp_expansion'))
                            success = True
                            break
                    # ==== 执行逻辑复用结束 ====
                
                # 如果这个被随机选中的客户所有方案都不行 (非常罕见，连兜底新路线都建不了)
                if not success:
                    print(f"客户点 {customer} 的随机/兜底修复方案均失败")
                    repaired_state.repair_objective = float('inf')
                    repaired_state.destroyed_customers_info = {}
                    return repaired_state, insert_plan
                
                num_repaired += 1

        repaired_state._total_cost = repaired_state.win_total_objective()
        repaired_state.destroyed_customers_info = {}
        
        return repaired_state, insert_plan

    def repair_random_greedy_insertion(self, state, strategic_bonus=0, num_destroyed=1, force_vtp_mode=True):
        """
        基于论文思想的随机-贪婪插入修复算子 (对应 Greedy truck-first, sortie-second insertion 的逻辑流)
        算法特征：每次随机选择一个客户点，然后为其寻找局部最优的插入位置（包含传统VTP组合与VTP扩展兜底）。
        """
        
        # 关键修复：必须创建状态副本，避免修改原始状态
        repaired_state = state.fast_copy()
        repaired_state.repair_objective = 0
        destroy_node = list(state.destroyed_customers_info.keys())  # L 集合
        insert_plan = []

        force_vtp_mode = True
        if force_vtp_mode:
            num_repaired = 0
            
            # Algorithm 8, Line 1: while L != ∅ do
            while len(destroy_node) > 0:
                
                # Algorithm 8, Line 2 & 4: 随机选择一个客户并从集合中移除
                # 使用 rng 保证 ALNS 的随机性一致
                target_idx = self.rng.integers(0, len(destroy_node))
                customer = destroy_node.pop(target_idx)
                
                # 计算当前状态的基准成本
                try:
                    current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                    base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                        repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                        self.vehicle, self.customer_time_windows_h, 
                        self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                    )
                except Exception:
                    base_total_cost = float('inf')

                # 获取当前状态的数据
                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)

                # Algorithm 9: Best Insertion 仅针对当前选中的 customer
                customer_candidates = []
                
                # 1. 尝试传统插入方案 (利用现有 VTP)
                traditional_result, is_heuristic_swap = self._evaluate_traditional_insertion(
                    customer, vehicle_route, vehicle_task_data, vehicle_arrive_time,
                    base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state
                )

                if traditional_result[0] or traditional_result[1] is not None:
                    traditional_cost, traditional_scheme = traditional_result
                    if is_heuristic_swap:
                        current_total_cost = traditional_scheme['total_cost']
                        deta_total_cost = traditional_scheme['win_cost']
                        customer_candidates.append({
                            'customer': customer, 'scheme': traditional_scheme,
                            'cost': traditional_cost, 'win_cost': deta_total_cost,
                            'total_cost': current_total_cost, 'type': 'heuristic_swap', 'vtp_node': None
                        })
                    else:
                        current_total_cost = traditional_cost + base_total_cost
                        customer_candidates.append({
                            'customer': customer, 'scheme': traditional_scheme,
                            'cost': traditional_cost, 'win_cost': traditional_cost,
                            'total_cost': current_total_cost, 'type': 'traditional', 'vtp_node': None
                        })
                
                # 2. 尝试 VTP 扩展方案 (等价于 Algorithm 9 Line 16: 开辟新路径/新增停靠点)
                vtp_result, vtp_infor = self._evaluate_vtp_expansion_insertion(
                    customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state
                )
                if vtp_infor is not None:
                    vtp_node, vtp_insert_vehicle_id, vtp_insert_index = vtp_infor
                    if vtp_result is not None:
                        vtp_cost, vtp_scheme = vtp_result
                        customer_candidates.append({
                            'customer': customer, 'scheme': vtp_scheme,
                            'cost': vtp_cost, 'total_cost': vtp_cost,
                            'type': 'vtp_expansion', 'vtp_node': vtp_node,
                            'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                            'vtp_insert_index': vtp_insert_index,
                            'original_cost': vtp_cost
                        })

                # 过滤出有效方案并按总成本从小到大排序
                customer_candidates = [item for item in customer_candidates if item['scheme'] is not None]
                candidates_plan = sorted(customer_candidates, key=lambda x: x['total_cost'])
                
                success = False

                # 尝试执行局部最优插入方案
                for candidate in candidates_plan:
                    # ==== 下方逻辑与你原始代码完全一致，无需任何改动 ====
                    if candidate['type'] == 'traditional':
                        best_scheme = candidate['scheme']
                        best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                            continue
                            
                        repaired_state.customer_plan[customer_node] = best_scheme
                        if drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[drone_id] = []
                        repaired_state.uav_assignments[drone_id].append(best_scheme)
                        
                        if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                        repaired_state.uav_cost[customer_node] = best_cost
                        
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                        insert_plan.append((customer, best_scheme, best_cost, 'traditional'))
                        success = True
                        break

                    elif candidate['type'] == 'heuristic_swap':
                        orig_scheme = candidate['scheme']['orig_scheme']
                        new_scheme = candidate['scheme']['new_scheme']
                        orig_cost = candidate['scheme']['orig_cost']
                        new_cost = candidate['scheme']['new_cost']
                        
                        orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = orig_scheme
                        new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_scheme
                        delete_customer = orig_customer
                        
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        delete_task_plan = temp_customer_plan[orig_customer]
                        del temp_customer_plan[orig_customer]
                        temp_customer_plan[orig_customer] = orig_scheme
                        temp_customer_plan[new_customer] = new_scheme
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                            continue  
                        else:
                            del repaired_state.customer_plan[delete_customer]
                            repaired_state.customer_plan[orig_customer] = orig_scheme
                            repaired_state.customer_plan[new_customer] = new_scheme
                            
                            if orig_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[orig_drone_id] = []
                            repaired_state.uav_assignments[orig_drone_id].append(orig_scheme)
                            if new_drone_id not in repaired_state.uav_assignments: repaired_state.uav_assignments[new_drone_id] = []
                            repaired_state.uav_assignments[new_drone_id].append(new_scheme)
                            
                            del repaired_state.uav_cost[delete_customer]
                            repaired_state.uav_cost[orig_customer] = orig_cost
                            repaired_state.uav_cost[new_customer] = new_cost
                            
                            vehicle_task_data = remove_vehicle_task(vehicle_task_data, delete_task_plan, vehicle_route)
                            orig_launch_time = temp_rm_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                            new_launch_time = temp_rm_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                            if orig_launch_time <= new_launch_time:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                            else:
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, new_scheme, vehicle_route)
                                vehicle_task_data = update_vehicle_task(vehicle_task_data, orig_scheme, vehicle_route)
                                
                            insert_plan.append((delete_customer, orig_scheme, orig_cost, 'heuristic_swap'))
                            insert_plan.append((customer, new_scheme, new_cost, 'heuristic_swap'))
                            success = True
                            break

                    elif candidate['type'] == 'vtp_expansion':
                        vtp_node = candidate['vtp_node']
                        vtp_insert_index = candidate['vtp_insert_index']
                        vtp_insert_vehicle_id = candidate['vtp_insert_vehicle_id']
                        best_scheme = candidate['scheme']
                        
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        original_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)

                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        temp_vehicle_route = [route[:] for route in vehicle_route]
                        temp_route = temp_vehicle_route[vtp_insert_vehicle_id - 1]
                        temp_route.insert(vtp_insert_index, vtp_node)
                        temp_vehicle_route[vtp_insert_vehicle_id - 1] = temp_route
                        repaired_state.temp_vehicle_routes = temp_vehicle_route
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                        
                        if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                            continue
                        else:
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

                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, vehicle_route)
                            
                            repaired_state.customer_plan[customer_node] = best_scheme
                            if drone_id not in repaired_state.uav_assignments:
                                repaired_state.uav_assignments[drone_id] = []
                            repaired_state.uav_assignments[drone_id].append(best_scheme)
                            
                            if repaired_state.uav_cost is None: repaired_state.uav_cost = {}
                            repaired_state.uav_cost[customer_node] = original_cost
                            
                            repaired_state.rm_empty_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
                            repaired_state.rm_empty_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.rm_empty_vehicle_route)
                            insert_plan.append((customer, best_scheme, original_cost, 'vtp_expansion'))
                            success = True
                            break
                    # ==== 上方逻辑与你原始代码完全一致 ====
                
                # 如果这个被随机选中的客户所有方案都不行 (等价于论文中连 a new truck route 都建不了的情况)
                if not success:
                    print(f"客户点 {customer} 的所有候选方案都不满足约束，修复失败")
                    repaired_state.repair_objective = float('inf')
                    repaired_state.destroyed_customers_info = {}
                    return repaired_state, insert_plan
                
                num_repaired += 1

        repaired_state._total_cost = repaired_state.win_total_objective()
        repaired_state.destroyed_customers_info = {}
        # print(f'修复策略完成，修复后总成本: {repaired_state._total_cost}')
        
        return repaired_state, insert_plan


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
                # 2. 计算初始状态的全局总成本 (作为基准)
                try:
                    current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                    base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                        repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                        self.vehicle, self.customer_time_windows_h, 
                        self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                    )
                except Exception:
                    base_total_cost = float('inf') # 初始状态

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
                    traditional_result,is_heuristic_swap = self._evaluate_traditional_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time,
                    base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state)

                    if traditional_result[0] or traditional_result[1] is not None:
                        traditional_cost, traditional_scheme = traditional_result
                        if is_heuristic_swap:
                            
                            current_total_cost = traditional_scheme['total_cost']
                            deta_total_cost = traditional_scheme['win_cost']
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': traditional_scheme,
                                'cost': traditional_cost,
                                'win_cost': deta_total_cost,
                                'total_cost': current_total_cost,
                                'type': 'heuristic_swap',
                                'vtp_node': None
                            })
                        else:
                            # 计算新插入的方案带时间窗及路线的成本
                            # win_traditional_cost = calculate_customer_window_cost(traditional_scheme, self.vehicle, current_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                            current_total_cost = traditional_cost + base_total_cost
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': traditional_scheme,
                                'cost': traditional_cost,
                                'win_cost': traditional_cost,
                                'total_cost': current_total_cost,
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
                            vtp_cost, vtp_scheme = vtp_result  # 这里的vtp_cost指的是插入后改变时间的惩罚成本+原本飞行路线+车辆绕行+新插入路线和惩罚成本的总和
                            # 应用最终奖励来增加VTP插入在前期被选中的概率
                            adjusted_cost = vtp_cost - final_bonus
                            
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': vtp_scheme,
                                'cost': adjusted_cost,
                                'total_cost': vtp_cost,
                                'type': 'vtp_expansion',
                                'vtp_node': vtp_node,  # launch_node就是VTP节点
                                'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                                'vtp_insert_index': vtp_insert_index,
                                'original_cost': vtp_cost
                            })
                customer_candidates = [item for item in customer_candidates if item['scheme'] is not None]
                # 对customer_candidates的cost由小到大排序
                candidates_plan = sorted(customer_candidates, key=lambda x: x['total_cost'])
                
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
                        best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])  # 这里获得的是路径成本
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
                        best_cost = candidate['cost']
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
                    # 清空破坏信息，即使修复失败也要清空，避免影响下一轮迭代
                    repaired_state.destroyed_customers_info = {}
                    return repaired_state, insert_plan
                    # continue
                
                # 从待插入列表中移除已处理的客户点
                if customer in destroy_node:
                    destroy_node.remove(customer)
                
                num_repaired += 1

        # 更新修复完成后的成本
        # repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        repaired_state._total_cost = repaired_state.win_total_objective()
        # 清空破坏信息，确保修复后的状态不包含已修复的破坏节点信息
        repaired_state.destroyed_customers_info = {}
        print(f'修复策略完成，修复后总成本计算完成')
        print(f"修复后总成本: {repaired_state._total_cost}")
        
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
                try:
                    current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                    base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                        repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                        self.vehicle, self.customer_time_windows_h, 
                        self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                    )
                except Exception:
                    base_total_cost = float('inf') # 初始状态

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
                    traditional_result, is_heuristic_swap = self._regret_evaluate_traditional_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state)
                    candidates.extend(traditional_result)
                    # 2) VTP扩展插入
                    total_options = self._regret_evaluate_vtp_expansion_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, repaired_state)
                    candidates.extend(total_options)
                    # 计算后悔值
                    if len(candidates) == 0:
                        print(f'在regret的修复策略中，客户点{customer}没有可行的插入方案，包括传统插入和VTP扩展插入,跳过')
                        continue
                    # 删除候选解中eval_cost数值为inf的内容
                    import math
                    # 过滤掉eval_cost为inf或None的候选解
                    candidates = [c for c in candidates if c.get('eval_cost') is not None and not math.isinf(c.get('eval_cost', 0))]
                    
                    candidates_sorted = sorted(candidates, key=lambda x: x['eval_cost'])
                    best_cost = candidates_sorted[0]['total_cost']
                    best_type = candidates_sorted[0]['type']
                    second_best_cost = candidates_sorted[1]['total_cost'] if len(candidates_sorted) >= 2 else best_cost
                    second_best_type = candidates_sorted[1]['type'] if len(candidates_sorted) >= 2 else None
                    
                    regret_value = second_best_cost - best_cost
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
                        # 清空破坏信息，即使修复失败也要清空，避免影响下一轮迭代
                        repaired_state.destroyed_customers_info = {}
                        return repaired_state, insert_plan

                if not success_any:
                    # 本轮没有任何可行插入，直接终止
                    break
        repaired_state._total_cost = repaired_state.win_total_objective()
        print(f'修复策略完成，修复后总成本计算完成')
        print(f"修复后总成本: {repaired_state._total_cost}")
        return repaired_state, insert_plan

    def repair_kNN_regret(self, state, strategic_bonus=0, num_destroyed=1, force_vtp_mode=False):
        # # ======= [PROF] 启动 profiler（新增）=======
        # profiler = cProfile.Profile()
        # profiler.enable()
        # # ======= [PROF] 启动 profiler 结束 =======
        # try:
        try:
            repaired_state = state.fast_copy()
            repaired_state.repair_objective = 0
            vehicle_routes = repaired_state.vehicle_routes
            vehicle_task_data = repaired_state.vehicle_task_data
            vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_routes)
            destroy_node = list(state.destroyed_customers_info.keys())
            insert_plan = []
            force_vtp_mode = True
            num_repaired = 0
            k_neighbors = self.params['k_neighbors']
            K_revest_position = 5
            k_neighbors = self.params.get('k_neighbors', 3)
            K_BEST_POSITIONS = self.params.get('K_BEST_POSITIONS', 5) 
            w_impact = self.params.get('w_impact', 0.5)
            # temp_vtp_task_data = deep_copy_vehicle_task_data(repaired_state.vehicle_task_data)
            if force_vtp_mode:
                while len(destroy_node) > 0:
                    try:
                        current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                        base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                            repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                            self.vehicle, self.customer_time_windows_h, 
                            self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                        )
                    except Exception:
                        base_total_cost = float('inf') # 初始状态

                    final_bonus = 0.0
                    if force_vtp_mode and len(destroy_node) > 0:
                        tactical_multiplier = (len(destroy_node) - num_repaired) / max(len(destroy_node), 1)
                        final_bonus = strategic_bonus * tactical_multiplier
                    else:
                        final_bonus = 0.0
                    cached_baseline_costs = {}
                    regret_list = []
                    candidates = []
                    # 获取全新的VTP节点
                    candidate_new_vtps = self._get_all_candidate_new_vtps(destroy_node, repaired_state)
                    # 使用公共共享VTP节点
                    used_vtps_set = {node for route in repaired_state.vehicle_routes for node in route[1:-1]}
                    # 获得当前状态下的临时task_data方案
                    temp_vtp_task_data = deep_copy_vehicle_task_data(repaired_state.vehicle_task_data)
                    # temp_state_after = repaired_state.fast_copy()
                    # K_BEST_POSITIONS = 10
                    for customer in destroy_node:
                        # 找到客户的k个最近的，待修复的邻居
                        candidates = []
                        k_nearest_neighbors = self._find_k_nearest_unassigned(customer, k_neighbors, destroy_node)
                        # temp_vtp_task_data = deep_copy_vehicle_task_data(repaired_state.vehicle_task_data)
                        # temp_vtp_task_data = restore_vehicle_task_data_for_vehicles(temp_vtp_task_data, repaired_state.vehicle_task_data, self.T)

                        for vtp_new in candidate_new_vtps:
                            # temp_vtp_task_data = deep_copy_vehicle_task_data(repaired_state.vehicle_task_data)
                            temp_vtp_task_data = restore_vehicle_task_data_for_vehicles(temp_vtp_task_data, repaired_state.vehicle_task_data, self.T)
                            # 找到插入 vtp_new 的最佳车辆和成本 # best_positions 返回: [(veh_id, insert_idx, veh_delta_cost), ...],尝试插入前几个最优的方案，以防止数据维度爆炸
                            best_positions = self._find_k_best_vehicle_for_new_vtp(vtp_new, repaired_state,K_revest_position)  # 输出的车辆id并非索引而是代号 输出的全部的新vtp节点和插入索引
                            if not best_positions: continue
                            # 【核心修改】: 遍历这K个最佳插入位置，评估每一个的潜力
                            for (veh_id, insert_idx, veh_delta_cost) in best_positions:
                                # 估算总收益,潜在危险，temp_vtp_task_data会被假设插入的内容修改里面的drone_list列表，但是目前没有报错
                                total_benefit, affected_customers = self._calculate_vtp_benefits(
                                    vtp_new, (veh_id, insert_idx), repaired_state, customer, temp_vtp_task_data
                                )
                                for customer, scheme in affected_customers:
                                    cost = scheme[0]
                                    scheme_plan = scheme[1]
                                    candidates.append({'customer': customer, 'type': 'vtp_expansion', 'vtp_node': vtp_new, 'vtp_insert_index': insert_idx, 
                                    'vtp_insert_vehicle_id': veh_id, 'scheme': scheme_plan, 'eval_cost': cost, 'total_cost': cost})
                        for vtp_shared in used_vtps_set:
                            # 【核心修改】: 为这个共享VTP，在所有【尚未】使用它的车辆中，找到K个最佳插入位置
                            best_shared_positions = self._find_k_best_vehicles_for_shared_vtp(vtp_shared, repaired_state, K_BEST_POSITIONS)
                            # temp_vtp_task_data = deep_copy_vehicle_task_data(repaired_state.vehicle_task_data)
                            temp_vtp_task_data = restore_vehicle_task_data_for_vehicles(temp_vtp_task_data, repaired_state.vehicle_task_data, self.T)
                            if not best_shared_positions: continue
                            # 【核心修改】: 遍历这K个最佳共享位置
                            for (veh_id, insert_idx, veh_delta_cost) in best_shared_positions:
                                
                                # 估算这个“共享方案”带来的总收益
                                total_benefit, affected_customers = self._calculate_vtp_benefits(
                                    vtp_shared, (veh_id, insert_idx), repaired_state, customer, temp_vtp_task_data
                                )
                                for customer, scheme in affected_customers:
                                    cost = scheme[0]  # cost指的是插入成本,包含了时间窗口的惩罚成本
                                    scheme_plan = scheme[1]
                                    candidates.append({'customer': customer, 'type': 'vtp_expansion', 'vtp_node': vtp_shared, 'vtp_insert_index': insert_idx, 
                                    'vtp_insert_vehicle_id': veh_id, 'scheme': scheme_plan, 'eval_cost': cost, 'total_cost': cost})
                        # 使用传统的candidates方案，改为传统插入测试
                        traditional_options_list, is_heuristic_swap = self._regret_evaluate_traditional_insertion(
                            customer, vehicle_routes, vehicle_task_data, vehicle_arrive_time, base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state)
                        if is_heuristic_swap:
                            for trad_opt in traditional_options_list:
                                if trad_opt['type'] == 'heuristic_swap':
                                    candidates.append({
                                    'customer': customer,
                                    'type': 'heuristic_swap',
                                    'vtp_node': None, # 传统方案不涉及VTP插入
                                    'vtp_insert_index': None,
                                    'vtp_insert_vehicle_id': None,
                                    'orig_scheme': trad_opt['orig_scheme'],
                                    'new_scheme': trad_opt['new_scheme'],
                                    'orig_cost': trad_opt['orig_cost'],
                                    'new_cost': trad_opt['new_cost'],
                                    'eval_cost': trad_opt['eval_cost'],
                                    'total_cost': trad_opt['total_cost']
                                    })
                                else:
                                    candidates.append({
                                        'customer': customer,
                                        'type': 'traditional',
                                        'vtp_node': None, # 传统方案不涉及VTP插入
                                        'vtp_insert_index': None,
                                        'vtp_insert_vehicle_id': None,
                                        'scheme': trad_opt['scheme'],
                                        'eval_cost': trad_opt['eval_cost'],
                                        'total_cost': trad_opt['total_cost']
                                    })
                        else:
                            for trad_opt in traditional_options_list:
                                candidates.append({
                                    'customer': customer,
                                    'type': 'traditional',
                                    'vtp_node': None, # 传统方案不涉及VTP插入
                                    'vtp_insert_index': None,
                                    'vtp_insert_vehicle_id': None,
                                    'scheme': trad_opt['scheme'],
                                    'eval_cost': trad_opt['eval_cost'],
                                    'total_cost': trad_opt['total_cost']
                                })
                        # 对candidates中eval_cost为inf的去除
                        # candidates = [c for c in candidates if not np.isinf(c['eval_cost'])]
                        candidates = [c for c in candidates if not np.isinf(c['total_cost'])]
                        # 进一步，对candidates进行排序，按照eval_cost从低到高，选取前K个方案计算其k步后悔值
                        candidates_sorted = sorted(candidates, key=lambda x: x['total_cost'])[:K_BEST_POSITIONS]
                        plan_scores = []
                        # for option_tuple in candidates_sorted[:K_BEST_POSITIONS]:
                        # temp_state_after = repaired_state.fast_copy()
                        for option_tuple in candidates_sorted[:]:
                            temp_state_after = repaired_state.temp_fast_copy(vehicles_to_copy=self.T)
                            # 【修正】: 按key取值
                            current_eval_cost = option_tuple['total_cost']
                            plan_type = option_tuple.get('type', 'traditional') # 安全获取
                            plan_scheme = option_tuple.get('scheme', None)
                            if plan_scheme is None:
                                print('plan_scheme is None')
                                continue
                            neigh = self._find_k_nearest_unassigned(option_tuple['customer'], k_neighbors, destroy_node)
                            # 【k-Step Lookahead】: 估算此方案对邻居的“未来影响”
                            future_impact = 0.0
                            # 只对有重大结构性影响的方案(新增/共享VTP)计算未来影响
                            if plan_type == 'vtp_expansion':
                                future_impact = self._calculate_future_impact(
                                    option_tuple, neigh, repaired_state, temp_state_after, base_total_cost, uav_tw_violation_cost, total_cost_dict
                                )
                            else:
                                # 该阶段评估用传统算法插入过程中，对后续任务产生的未来影响
                                future_impact = self._calculate_tradition_future_impact(option_tuple, neigh, repaired_state, temp_state_after, base_total_cost, uav_tw_violation_cost, total_cost_dict)
                            total_kNN_score = current_eval_cost + w_impact * future_impact
                            # 【修正】: 存储 kNN分数 和 完整的【方案字典】
                            plan_scores.append({'kNN_score': total_kNN_score, 'option_dict': option_tuple})

                        # d. 按“k-步综合评估分数”排序
                        if not plan_scores: continue
                        plan_scores.sort(key=lambda x: x['kNN_score'])
                        
                        # e. 正确计算后悔值
                        best_kNN_option = plan_scores[0]['option_dict'] 
                        best_kNN_score = plan_scores[0]['kNN_score']
                        second_best = plan_scores[1]['kNN_score'] if len(plan_scores) >= 2 else best_kNN_score
                        regret_value = second_best - best_kNN_score
                        # if plan_scores[0]['option_dict']['scheme'] == (12,129,97,113,1,1):
                        #     print(f"调试：找到plan_scores[0]['option_dict']['scheme'] == (12,129,97,113,1,1)")

                        # f. 存储结果
                        regret_list.append({
                            'customer': customer,
                            'regret': regret_value,
                            'best_kNN_score': best_kNN_score,
                            'best_option': best_kNN_option,
                            'type': best_kNN_option['type'],
                        })
                    # 选择regret从大到小排序，如果regret值一样，择对应的best_kNN_score从小到大排序
                    regret_list_sorted = sorted(regret_list, key=lambda x: (-x['regret'], x['best_kNN_score']))
                    if not regret_list_sorted:
                        repaired_state.repair_objective = float('inf')
                        return repaired_state, insert_plan
                    # 执行最佳后悔值方案
                    # 取“最大后悔值”的客户
                    best_entry = regret_list_sorted[0]    # ← 修复方向
                    best_opt = best_entry['best_option']
                    best_type = best_entry['type']
                    best_cust = best_entry['customer']
                    if best_type == 'traditional':
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_opt['scheme']
                        best_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)

                        repaired_state.customer_plan[customer_node] = best_opt['scheme']
                        if drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[drone_id] = []
                        repaired_state.uav_assignments[drone_id].append(best_opt['scheme'])

                        if repaired_state.uav_cost is None:
                            repaired_state.uav_cost = {}
                        repaired_state.uav_cost[customer_node] = best_cost

                        # ← 修复变量名
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, best_opt['scheme'], vehicle_routes)

                        insert_plan.append((best_cust, best_opt['scheme'], best_cost, 'traditional'))
                    elif best_type == 'vtp_expansion':
                        customer = best_opt['customer']
                        vtp_node = best_opt['vtp_node']
                        vtp_insert_index = best_opt['vtp_insert_index']
                        vtp_insert_vehicle_id = best_opt['vtp_insert_vehicle_id']
                        best_scheme = best_opt['scheme']
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = best_scheme
                        original_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        # 2. 创建临时状态进行约束检查
                        temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        temp_customer_plan[customer_node] = best_scheme
                        # 生成临时的车辆路线，避免指向同一对象
                        temp_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
                        temp_route = temp_vehicle_route[vtp_insert_vehicle_id - 1]
                        temp_route.insert(vtp_insert_index, vtp_node)
                        temp_vehicle_route[vtp_insert_vehicle_id - 1] = temp_route
                        repaired_state.temp_vehicle_routes = temp_vehicle_route
                        # 计算临时车辆到达时间
                        temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                        time_feasible = is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time)
                        if not time_feasible:
                            # 时间约束不满足，尝试下一个候选方案
                            print(f"VTP扩展方案时间约束不满足，尝试下一个候选方案")
                            continue
                        else:
                            # 找到VTP节点在路径中的插入位置
                            route = repaired_state.vehicle_routes[vtp_insert_vehicle_id - 1]

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
                            vehicle_task_data = update_vehicle_task(vehicle_task_data, best_scheme, repaired_state.vehicle_routes)
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
                            # insert_plan.append((best_cust, best_opt['scheme'], best_opt['eval_cost'], 'vtp_expansion'))
                    # 正确移除已修复客户
                    destroy_node.remove(best_cust)
                    # num_repaired += 1

                    # 若后续循环还要用到到达时间，建议每轮重算
                    vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_routes)
                    num_repaired += 1
            return repaired_state, insert_plan
        except Exception as e:
            print(f"repair_kNN_regret 修复失败: {e}")
            repaired_state.repair_objective = float('inf')
            insert_plan = []
            return repaired_state, insert_plan
        # finally:
        #     # ======= [PROF] 停止 profiler 并打印结果（新增）=======
        #     profiler.disable()
        #     s = io.StringIO()
        #     # 按累积时间排序，优先看“真正慢”的函数
        #     ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        #     # 你可以加上过滤，只看相关函数；不想过滤就用 ps.print_stats(30)
        #     ps.print_stats(
        #         'repair_kNN_regret|_get_all_candidate_new_vtps|_find_k_nearest_unassigned|_find_k_best_vehicle_for_new_vtp|_find_k_best_vehicles_for_shared_vtp|_calculate_vtp_benefits|_regret_evaluate_traditional_insertion|_calculate_future_impact|_calculate_tradition_future_impact|calculate_rm_empty_vehicle_arrive_time'
        #     )
        #     print(s.getvalue())
        #     print('检查完成')

    # 在 IncrementalALNS 类中
    def _calculate_tradition_future_impact(self, option_dict, k_neighbors, original_state, temp_state_after, base_total_cost, uav_tw_violation_cost, total_cost_dict):
        """
        【k-step 评估器】(字典修正版)
        估算执行 'option_dict' 对 k_neighbors 修复成本的影响。
        """
        M_PENALTY = self.M_PENALTY
        total_impact = 0.0
        
        # 1. 计算邻居们在【插入前】的基线成本
        costs_before = {}
        orig_routes = original_state.vehicle_routes
        orig_task_data = original_state.vehicle_task_data
        orig_arrive_time = original_state.calculate_rm_empty_vehicle_arrive_time(orig_routes)
        orig_total_cost = sum(original_state.uav_cost.values())

        for customer in k_neighbors:
            # 假设 _evaluate_traditional_insertion 返回一个 options 字典列表
            trad_options, is_heuristic_swap = self._evaluate_traditional_insertion(
                customer, orig_routes, orig_task_data,
                orig_arrive_time, base_total_cost, uav_tw_violation_cost, total_cost_dict, original_state)
            if trad_options[0] or trad_options[1] is not None:
                trad_cost, traditional_scheme = trad_options
                if is_heuristic_swap:
                    # min_real_cost = trad_options[1]['delta_cost']
                    # costs_before[customer] = min_real_cost
                    current_total_cost = traditional_scheme['total_cost']
                    deta_total_cost = traditional_scheme['win_cost']
                    costs_before[customer] = current_total_cost
                else:
                    # min_real_cost = trad_options[0] 
                    # costs_before[customer] = min_real_cost
                    current_total_cost = trad_cost + base_total_cost
                    costs_before[customer] = current_total_cost
            else:
                costs_before[customer] = float('inf')

        # 2. 创建一个【模拟】的未来状态
        # temp_state_after = original_state.fast_copy()
        try:
            if option_dict['type'] == 'heuristic_swap':
                orig_scheme = option_dict['orig_scheme']
                new_scheme = option_dict['new_scheme']
                orig_cost = option_dict['orig_cost']
                new_cost = option_dict['new_cost']
                delta_cost = orig_cost + new_cost - option_dict['delta_cost']
                orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = orig_scheme
                new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = new_scheme
                customer = new_customer
                delete_customer = orig_customer
                temp_customer_plan = {k: v for k, v in temp_state_after.customer_plan.items()}
                delete_task_plan = temp_customer_plan[orig_customer]
                del temp_customer_plan[orig_customer]
                temp_customer_plan[orig_customer] = orig_scheme
                temp_customer_plan[new_customer] = new_scheme
                temp_rm_vehicle_arrive_time = temp_state_after.calculate_rm_empty_vehicle_arrive_time(temp_state_after.vehicle_routes)
                if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                    print(f"启发式交换方案时间约束不满足，尝试下一个候选方案")
                    return M_PENALTY
                # 更新customer_plan
                del temp_customer_plan[delete_customer]
                temp_customer_plan.customer_plan[orig_customer] = orig_scheme
                temp_customer_plan.customer_plan[new_customer] = new_scheme
                # 更新uav_assignments
                if orig_drone_id not in temp_customer_plan.uav_assignments:
                    temp_customer_plan.uav_assignments[orig_drone_id] = []
                temp_customer_plan.uav_assignments[orig_drone_id].append(orig_scheme)
                if new_drone_id not in temp_customer_plan.uav_assignments:
                    temp_customer_plan.uav_assignments[new_drone_id] = []
                temp_customer_plan.uav_assignments[new_drone_id].append(new_scheme)
                # 更新uav_cost
                del temp_customer_plan.uav_cost[delete_customer]
                temp_customer_plan.uav_cost[orig_customer] = orig_cost
                temp_customer_plan.uav_cost[new_customer] = new_cost
                # 更新vehicle_task_data
                temp_state_after.vehicle_task_data = remove_vehicle_task(temp_state_after.vehicle_task_data, delete_task_plan, temp_customer_plan.vehicle_routes)
                orig_launch_time = temp_rm_vehicle_arrive_time[orig_launch_vehicle][orig_launch_node]
                new_launch_time = temp_rm_vehicle_arrive_time[new_launch_vehicle][new_launch_node]
                if orig_launch_time <= new_launch_time:
                    temp_state_after.vehicle_task_data = update_vehicle_task(temp_state_after.vehicle_task_data, orig_scheme, temp_state_after.vehicle_routes)
                    temp_state_after.vehicle_task_data = update_vehicle_task(temp_state_after.vehicle_task_data, new_scheme, temp_state_after.vehicle_routes)
                else:
                    temp_state_after.vehicle_task_data = update_vehicle_task(temp_state_after.vehicle_task_data, new_scheme, temp_state_after.vehicle_routes)
                    temp_state_after.vehicle_task_data = update_vehicle_task(temp_state_after.vehicle_task_data, orig_scheme, temp_state_after.vehicle_routes)
            else:
                # 在此阶段模拟用传统插入策略造成的影响
                scheme = option_dict['scheme']  # (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = scheme

                # 创建临时状态进行约束检查
                temp_customer_plan = {k: v for k, v in temp_state_after.customer_plan.items()}
                temp_customer_plan[customer_node] = scheme
                temp_rm_vehicle_arrive_time = temp_state_after.calculate_rm_empty_vehicle_arrive_time(
                    temp_state_after.vehicle_routes
                )

                # 检查时间约束
                if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                    return M_PENALTY
                
                # 约束满足，执行插入
                # 更新 customer_plan
                temp_state_after.customer_plan[customer_node] = scheme
                
                # 更新 uav_assignments
                if drone_id not in temp_state_after.uav_assignments:
                    temp_state_after.uav_assignments[drone_id] = []
                temp_state_after.uav_assignments[drone_id].append(scheme)
                
                # 更新 uav_cost
                if temp_state_after.uav_cost is None:
                    temp_state_after.uav_cost = {}
                # real_cost = option_dict.get('real_cost', option_dict.get('eval_cost'))
                real_cost = self.drone_insert_cost(scheme[0], scheme[2], scheme[1], scheme[3])
                if real_cost is not None:
                    temp_state_after.uav_cost[customer_node] = real_cost

                # 更新 vehicle_task数据
                temp_state_after.vehicle_task_data = update_vehicle_task(
                    temp_state_after.vehicle_task_data, scheme, temp_state_after.vehicle_routes
                )
        except Exception as e:
            # print(f"  > 警告: k-NN 模拟插入失败: {e}")
            return 0 

        # 3. 计算邻居们在【插入后】的新基线成本
        costs_after = {}
        temp_arrive_time = temp_rm_vehicle_arrive_time
        temp_base_total_cost, temp_uav_tw_violation_cost, temp_total_cost_dict = calculate_window_cost(
            temp_state_after.customer_plan, temp_state_after.uav_cost, temp_arrive_time, 
            self.vehicle, self.customer_time_windows_h, 
            self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
        )

        for customer in k_neighbors:
            trad_options_after, is_heuristic_swap = self._evaluate_traditional_insertion(
                customer, temp_state_after.vehicle_routes, temp_state_after.vehicle_task_data,
                temp_arrive_time, temp_base_total_cost, temp_uav_tw_violation_cost, temp_total_cost_dict, temp_state_after)
            if is_heuristic_swap:
                # min_real_cost = trad_options_after[1]['delta_cost']
                # costs_after[customer] = min_real_cost
                current_total_cost = trad_options_after[1]['total_cost']
                deta_total_cost = trad_options_after[1]['win_cost']
                costs_after[customer] = current_total_cost
            else:
                if trad_options_after == (None,None):
                    costs_after[customer] = float('inf')
                    continue
                else:
                    cost = trad_options_after[0] + temp_base_total_cost
                    costs_after[customer] = cost
                
        # 4. 计算总影响 (Cost_After - Cost_Before)
        for customer in k_neighbors:
            cost_b = M_PENALTY if costs_before.get(customer, float('inf')) == float('inf') else costs_before.get(customer)
            cost_a = M_PENALTY if costs_after.get(customer, float('inf')) == float('inf') else costs_after.get(customer)
            total_impact += (cost_a - cost_b)
            
        return total_impact


    # 在 IncrementalALNS 类中
    def _calculate_future_impact(self, option_dict, k_neighbors, original_state, temp_state_after, base_total_cost, uav_tw_violation_cost, total_cost_dict):
        """
        【k-step 评估器】(字典修正版)
        估算执行 'option_dict' 对 k_neighbors 修复成本的影响。
        """
        
        M_PENALTY = self.M_PENALTY
        total_impact = 0.0
        
        # 1. 计算邻居们在【插入前】的基线成本
        costs_before = {}
        orig_routes = original_state.vehicle_routes
        orig_task_data = original_state.vehicle_task_data
        orig_arrive_time = original_state.calculate_rm_empty_vehicle_arrive_time(orig_routes)

        for customer in k_neighbors:
            # 假设 _evaluate_traditional_insertion 返回一个 options 字典列表
            trad_options, is_heuristic_swap = self._evaluate_traditional_insertion(
                customer, orig_routes, orig_task_data,
                orig_arrive_time, base_total_cost, uav_tw_violation_cost, total_cost_dict, original_state)
            if trad_options:
                min_real_cost = trad_options[0] 
                costs_before[customer] = min_real_cost
            else:
                costs_before[customer] = float('inf')

        # 2. 创建一个【模拟】的未来状态
        # temp_state_after = original_state.fast_copy()
        try:
            # 【修正】: 从字典中提取执行所需的信息
            plan_type = option_dict['type']
            
            if plan_type == 'vtp_expansion' or plan_type == 'investment' or plan_type == 'sharing':
                real_cost = option_dict['eval_cost']
                total_cost = option_dict['total_cost']
                plan = option_dict['scheme']
                vtp_node = option_dict['vtp_node']
                vtp_insert_index = option_dict['vtp_insert_index']
                vtp_insert_vehicle_id = option_dict['vtp_insert_vehicle_id']
            else:
                print(f"  > 错误: 未知插入方案类型: {plan_type}")
                return 0      
            option_to_execute = (real_cost, plan, plan_type, vtp_node, vtp_insert_index, vtp_insert_vehicle_id, total_cost)
            self._execute_insertion(temp_state_after, option_to_execute)
            
        except Exception as e:
            # print(f"  > 警告: k-NN 模拟插入失败: {e}")
            return 0 

        # 3. 计算邻居们在【插入后】的新基线成本
        costs_after = {}
        temp_arrive_time = temp_state_after.calculate_rm_empty_vehicle_arrive_time(temp_state_after.vehicle_routes)
        base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                        temp_state_after.customer_plan, temp_state_after.uav_cost, temp_arrive_time, 
                        self.vehicle, self.customer_time_windows_h, 
                        self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                    )
        for customer in k_neighbors:
            trad_options_after, is_heuristic_swap = self._evaluate_traditional_insertion(
                customer, temp_state_after.vehicle_routes, temp_state_after.vehicle_task_data,
                temp_arrive_time, base_total_cost, uav_tw_violation_cost, total_cost_dict, temp_state_after)
            if is_heuristic_swap:
                costs_after[customer] = float('inf')
                continue
            if trad_options_after == (None,None):
                costs_after[customer] = float('inf')
                continue
            cost = trad_options_after[0]
            costs_after[customer] = cost
                
        # 4. 计算总影响 (Cost_After - Cost_Before)
        for customer in k_neighbors:
            cost_b = M_PENALTY if costs_before.get(customer, float('inf')) == float('inf') else costs_before.get(customer)
            cost_a = M_PENALTY if costs_after.get(customer, float('inf')) == float('inf') else costs_after.get(customer)
            total_impact += (cost_a - cost_b)
            
        return total_impact

    def _execute_insertion(self, state, option):
        """(辅助函数) 专门用于执行插入方案的函数。"""
        real_cost, plan, plan_type, vtp_node, vtp_insert_index, vtp_insert_vehicle_id, total_cost = option
        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = plan
        routes = state.vehicle_routes
        task_data = state.vehicle_task_data
        arrive_time = state.calculate_rm_empty_vehicle_arrive_time(routes)
        routes[vtp_insert_vehicle_id - 1].insert(vtp_insert_index, vtp_node)
        # 检查是否符合约束条件
        if not is_time_feasible(state.customer_plan, arrive_time):
            return False
        state.vehicle_routes = routes
        last_node = routes[vtp_insert_vehicle_id - 1][vtp_insert_index - 1]
        if last_node == self.DEPOT_nodeID or vtp_insert_index == 1:
            drone_list = self.base_drone_assignment[vtp_insert_vehicle_id][:]
        else:
            drone_list = task_data[vtp_insert_vehicle_id][last_node].drone_list[:]
        task_data[vtp_insert_vehicle_id][vtp_node].drone_list = drone_list
        task_data[vtp_insert_vehicle_id][vtp_node].launch_drone_list = []
        state.vehicle_task_data = task_data
        arrive_time = state.calculate_rm_empty_vehicle_arrive_time(routes)
        state.vehicle_arrive_time = arrive_time  # 获得完成节点插入后的更新车辆时间
        state.customer_plan[customer_node] = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
        if drone_id not in state.uav_assignments:
            state.uav_assignments[drone_id] = []
        state.uav_assignments[drone_id].append((customer_node, launch_node, recovery_node, launch_vehicle_id, recovery_vehicle_id))
        state.uav_assignments[drone_id].append(plan)
        if state.uav_cost is None:
            state.uav_cost = {}
            state.uav_cost[customer_node] = real_cost
        task_data = update_vehicle_task(task_data, plan, routes)
        state.vehicle_task_data = task_data
        return True

    def noise_regret_insertion(self, state, strategic_bonus=0, num_destroyed=1, force_vtp_mode=False):
        # 关键修复：必须创建状态副本，避免修改原始状态
        repaired_state = state.fast_copy()  # 修复：创建真正的副本
        repaired_state.repair_objective = 0
        destroy_node = list(state.destroyed_customers_info.keys())  # 总结出了所有的待插入的破坏节点
        insert_plan = []  # 记录所有破坏节点的最优插入方案
        # 噪声策略参数
        rcs_k = 3 # 受限候选集的大小 (建议 3-5)。值越大，随机性越高，搜索范围越广。
        noise_temperature = 1.0 # 控制概率选择的温度 (可选)，如果使用简单的随机选择则不需要
        force_vtp_mode = True
        if force_vtp_mode:
            num_repaired = 0
            while len(destroy_node) > 0:
                # 2. 计算初始状态的全局总成本 (作为基准)
                try:
                    current_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                    base_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(
                        repaired_state.customer_plan, repaired_state.uav_cost, current_arrive_time, 
                        self.vehicle, self.customer_time_windows_h, 
                        self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
                    )
                except Exception:
                    base_total_cost = float('inf') # 初始状态
                # 计算动态奖励 (随着修复进程，奖励逐渐降低，后期偏向纯贪婪)
                num_repaired = num_destroyed - len(destroy_node)
                tactical_multiplier = (len(destroy_node)) / num_destroyed
                final_bonus = strategic_bonus * tactical_multiplier * 0.3
                
                # 获取当前状态引用
                vehicle_route = repaired_state.vehicle_routes
                vehicle_task_data = repaired_state.vehicle_task_data
                vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
                # final_bonus = 0
                # 存储所有候选方案
                all_candidates = []
                customer_candidates = []
                global_best_moves = []
                # 遍历所有待插入客户点，计算每个节点的最优插入成本
                for customer in destroy_node:
                    # 1. 首先尝试传统插入方案（使用现有节点）
                    customer_specific_candidates = []
                    traditional_result,is_heuristic_swap = self._evaluate_traditional_insertion(customer, vehicle_route, vehicle_task_data, vehicle_arrive_time,base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state)

                    if traditional_result[0] or traditional_result[1] is not None:
                        traditional_cost, traditional_scheme = traditional_result
                        if is_heuristic_swap:
                            # 计算新插入的方案带时间窗及路线的成本
                            current_total_cost = traditional_scheme['total_cost']
                            deta_total_cost = traditional_scheme['win_cost']
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': traditional_scheme,
                                'cost': traditional_cost,
                                'win_cost': deta_total_cost,
                                'total_cost': current_total_cost,
                                'type': 'heuristic_swap',
                                'vtp_node': None
                            })
                        else:
                            # 计算新插入的方案带时间窗及路线的成本
                            # win_traditional_cost = calculate_customer_window_cost(traditional_scheme, self.vehicle, current_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                            current_total_cost = traditional_cost + base_total_cost
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': traditional_scheme,
                                'cost': traditional_cost,
                                'win_cost': traditional_cost,
                                'total_cost': current_total_cost,
                                'type': 'traditional',
                                'vtp_node': None
                            })
                    else:
                        # 传统插入方案失败，设置成本为无穷大
                        customer_candidates.append({
                            'customer': customer,
                            'scheme': None,
                            'cost': float('inf'),
                            'total_cost': float('inf'),
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
                            vtp_cost, vtp_scheme = vtp_result  # 这里的vtp_cost指的是插入后改变时间的惩罚成本+原本飞行路线+车辆绕行+新插入路线和惩罚成本的总和
                            # 应用最终奖励来增加VTP插入在前期被选中的概率
                            adjusted_cost = vtp_cost - final_bonus
                            
                            customer_candidates.append({
                                'customer': customer,
                                'scheme': vtp_scheme,
                                'cost': adjusted_cost,
                                'total_cost': vtp_cost,
                                'type': 'vtp_expansion',
                                'vtp_node': vtp_node,  # launch_node就是VTP节点
                                'vtp_insert_vehicle_id': vtp_insert_vehicle_id,
                                'vtp_insert_index': vtp_insert_index,
                                'original_cost': vtp_cost
                            })
                    if customer_candidates:
                        customer_candidates = [item for item in customer_candidates if item['scheme'] is not None]
                        if customer_candidates:  # 有可能过滤后的任务为空
                            customer_candidates.sort(key=lambda x: x['total_cost'])
                            best_move_for_this_customer = customer_candidates[0]
                            global_best_moves.append(best_move_for_this_customer)
                    customer_candidates = []
                if not global_best_moves:
                    print('在噪声的修复策略中，无法为剩余客户找到任何可行位置')
                    repaired_state.repair_objective = float('inf')
                    repaired_state.destroyed_customers_info = {}
                    return repaired_state, insert_plan
                global_best_moves.sort(key=lambda x: x['total_cost'])
                # 2. 使用加权选择构建执行队列
                # 这比 random.shuffle 靠谱，因为它尊重了成本的物理意义
                execution_queue = weighted_choice_sub(global_best_moves, rcs_k)
                
                # 下面是你原有的执行逻辑，直接复用
                success = False

                for candidate in execution_queue:
                    customer = candidate['customer']
                    # best_scheme = candidate['scheme']
                    # best_cost = candidate['cost']
                    
                    # 根据方案类型执行不同的插入逻辑
                    if candidate['type'] == 'traditional':
                        # print(f"尝试使用传统方案插入客户点 {customer}，成本: {best_cost:.2f}")
                        
                        customer = candidate['customer']
                        best_scheme = candidate['scheme']
                        best_cost = self.drone_insert_cost(best_scheme[0], best_scheme[2], best_scheme[1], best_scheme[3])  # 这里获得的是路径成本
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
                        best_cost = candidate['cost']
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
                    # 清空破坏信息，即使修复失败也要清空，避免影响下一轮迭代
                    repaired_state.destroyed_customers_info = {}
                    return repaired_state, insert_plan
                    # continue
                
                # 从待插入列表中移除已处理的客户点
                if customer in destroy_node:
                    destroy_node.remove(customer)
                
                num_repaired += 1
                
        # 更新修复完成后的成本
        # repaired_state._total_cost = repaired_state.update_calculate_plan_cost(repaired_state.uav_cost, repaired_state.vehicle_routes)
        repaired_state._total_cost = repaired_state.win_total_objective()
        # 清空破坏信息，确保修复后的状态不包含已修复的破坏节点信息
        repaired_state.destroyed_customers_info = {}
        print(f'修复策略完成，修复后总成本计算完成')
        print(f"修复后总成本: {repaired_state._total_cost}")
        
        return repaired_state, insert_plan

    def _find_k_best_vehicle_for_new_vtp(self, vtp_new, state, k):
        """
        (VTP-Centric 辅助函数)
        为单个【新】VTP候选节点，在【所有】车辆中找到 K 个成本最低的插入方案。
        
        Returns:
            list: [(veh_id, insert_idx, veh_delta_cost), ...] (按成本升序排列, 最多K个)
        """
        all_insertion_options = [] # 存储所有可能的插入方案
        
        for vehicle_idx, route in enumerate(state.vehicle_routes):
            vehicle_id = vehicle_idx + 1
            if len(route) < 2: continue
                
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                next_node = route[i]
                try:
                    delta_cost = self.veh_distance[vehicle_id][prev_node][vtp_new] + \
                                self.veh_distance[vehicle_id][vtp_new][next_node] - \
                                self.veh_distance[vehicle_id][prev_node][next_node]
                    
                    all_insertion_options.append((vehicle_id, i, delta_cost))
                    
                except KeyError:
                    continue 

        # 按成本升序排序
        all_insertion_options.sort(key=lambda x: x[2])
        
        # 返回前 K 个
        return all_insertion_options[:k]
        # return all_insertion_options[:]

    def _calculate_synergy_score(self, opt, neighbors, all_vtps, state, vehicle_task_data, k_route_nodes=5):
        """
        (VTP-Knn-regret 辅助函数)
        计算协同分 (Synergy Score)
        对于VTP插入方案，综合考虑被破坏的相邻neighbors的支持作用、距离和VTP节点剩余无人机数量。
        
        Args:
            opt (dict): 操作选项，包含 'customer', 'type', 'vtp_insert_vehicle_id', 'vtp_node', 'scheme' 等
            neighbors (list): 被破坏的相邻客户ID列表（在destroy_node中，距离当前客户最近的k个）
            all_vtps (set): 所有VTP节点的集合
            state (FastMfstspState): 修复后的状态
            vehicle_task_data (dict): 车辆任务数据，格式为 vehicle_task_data[vehicle_id][node_id]
            k_route_nodes (int): 考虑路径上周围k个节点（默认5个），因为允许无人机跨车运输
            
        Returns:
            float: 协同分数，值越大表示协同程度越高
        """
        synergy_score = 0.0
        vtp_insert_vehicle_id = opt.get('vtp_insert_vehicle_id')
        vtp_insert_index = opt.get('vtp_insert_index')
        vehicle_routes = [route[:] for route in state.vehicle_routes] 
        try:
            # 只对VTP扩展方案计算协同分
            if opt.get('type') != 'vtp_expansion':
                return 0.0
            
            # 1. 获取被操作客户的ID和坐标
            customer_id = opt.get('customer')
            if customer_id is None:
                scheme = opt.get('scheme')
                if scheme and len(scheme) >= 3:
                    customer_id = scheme[2]  # scheme[2] 是 customer_node
            
            if customer_id is None or customer_id not in self.node:
                return 0.0
            
            customer_node = self.node[customer_id]
            customer_lat = customer_node.latDeg
            customer_lon = customer_node.lonDeg
            customer_alt = customer_node.altMeters
            
            # 2. 获取VTP节点信息和坐标
            vtp_node = opt.get('vtp_node')
            vtp_insert_vehicle_id = opt.get('vtp_insert_vehicle_id')
            
            if vtp_node is None or vtp_insert_vehicle_id is None:
                return 0.0
            
            # 获取VTP节点的坐标
            if vtp_node not in self.node:
                return 0.0
            
            vtp_node_obj = self.node[vtp_node]
            vtp_lat = vtp_node_obj.latDeg
            vtp_lon = vtp_node_obj.lonDeg
            vtp_alt = vtp_node_obj.altMeters
            
            # 3. 获取VTP节点在该车辆上的剩余无人机数量
            # 注意：VTP节点可能尚未插入到vehicle_task_data中，需要模拟其状态
            # 如果VTP节点已经存在于vehicle_task_data中，直接获取
            # 否则，需要根据插入位置的前一个节点来推断无人机数量
            
            drone_count = 0
            temp_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)
            # 上一个节点的drone_list信息
            last_customer_node = vehicle_routes[vtp_insert_vehicle_id - 1][vtp_insert_index - 1]
            if vtp_insert_index == 1 or last_customer_node == self.DEPOT_nodeID:
                drone_count = len(self.base_drone_assignment[vtp_insert_vehicle_id])
            else:
                drone_count = len(temp_vehicle_task_data[vtp_insert_vehicle_id][last_customer_node].drone_list)

            # 如果无法获取无人机数量，使用默认值（假设有足够的无人机）
            if drone_count == 0:
                print('无人机数量获取失败，使用默认值')
                # # 尝试从车辆的基础无人机分配获取
                # if vtp_insert_vehicle_id in self.base_drone_assignment:
                #     drone_count = len(self.base_drone_assignment[vtp_insert_vehicle_id])
                # else:
                #     # 如果还是0，给一个默认值（比如所有无人机的数量）
                #     drone_count = len(self.V) if self.V else 1
            
            # 4. 计算新建VTP与路径上周围k个节点的距离（考虑路径成本影响，允许无人机跨车运输）
            route_proximity_score = 0.0
            route_node_count = 0
            if isinstance(state.vehicle_routes, list):
                route_idx = vtp_insert_vehicle_id - 1
                if 0 <= route_idx < len(state.vehicle_routes):
                    target_route = state.vehicle_routes[route_idx]
                    vtp_insert_idx = opt.get('vtp_insert_index', 0)
                    
                    # 获取插入位置周围的k个节点（前后各k/2个，或尽可能多）
                    # 因为允许无人机跨车运输，需要考虑周围更多节点
                    start_idx = max(0, vtp_insert_idx - k_route_nodes // 2)
                    end_idx = min(len(target_route), vtp_insert_idx + k_route_nodes // 2 + 1)
                    
                    # 计算VTP节点与路径上周围k个节点的距离
                    for i in range(start_idx, end_idx):
                        if i == vtp_insert_idx:
                            continue  # 跳过插入位置本身（VTP节点会插入在这里）
                        
                        route_node_id = target_route[i]
                        if route_node_id in self.node:
                            route_node = self.node[route_node_id]
                            route_node_lat = route_node.latDeg
                            route_node_lon = route_node.lonDeg
                            route_node_alt = route_node.altMeters
                            
                            lat_diff = vtp_lat - route_node_lat
                            lon_diff = vtp_lon - route_node_lon
                            alt_diff = vtp_alt - route_node_alt
                            route_distance = np.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
                            
                            # 距离越近，路径协同效应越强
                            # 使用距离的倒数，并考虑节点在路径上的位置权重（越近权重越大）
                            position_weight = 1.0 / (1.0 + abs(i - vtp_insert_idx))  # 距离插入位置越近，权重越大
                            route_proximity_score += position_weight / (1.0 + route_distance)
                            route_node_count += 1
                    
                    # 归一化：取平均值
                    if route_node_count > 0:
                        route_proximity_score = route_proximity_score / route_node_count
            
            # 5. 计算新建VTP与客户节点的距离（VTP到当前客户的协同）
            customer_proximity_score = 0.0
            if customer_id in self.node:
                lat_diff = vtp_lat - customer_lat
                lon_diff = vtp_lon - customer_lon
                alt_diff = vtp_alt - customer_alt
                customer_distance = np.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
                # 距离越近，VTP到客户的协同效应越强
                customer_proximity_score = 1.0 / (1.0 + customer_distance)
            
            # 6. 计算新建VTP与所有destroyed_node的距离（考虑VTP对所有被破坏节点的覆盖能力）
            destroyed_node_score = 0.0
            destroyed_node_count = 0
            if hasattr(state, 'destroyed_customers_info') and state.destroyed_customers_info:
                destroyed_node_list = list(state.destroyed_customers_info.keys())
                for destroyed_node_id in destroyed_node_list:
                    if destroyed_node_id == customer_id:
                        continue  # 跳过当前客户（已经在customer_proximity_score中考虑）
                    if destroyed_node_id not in self.node:
                        continue
                    
                    destroyed_node = self.node[destroyed_node_id]
                    destroyed_lat = destroyed_node.latDeg
                    destroyed_lon = destroyed_node.lonDeg
                    destroyed_alt = destroyed_node.altMeters
                    
                    lat_diff = vtp_lat - destroyed_lat
                    lon_diff = vtp_lon - destroyed_lon
                    alt_diff = vtp_alt - destroyed_alt
                    destroyed_distance = np.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
                    
                    # 距离越近，VTP对被破坏节点的覆盖能力越强
                    destroyed_node_score += 1.0 / (1.0 + destroyed_distance)
                    destroyed_node_count += 1
                
                # 归一化：取平均值
                if destroyed_node_count > 0:
                    destroyed_node_score = destroyed_node_score / destroyed_node_count
            
            # 7. 计算新建VTP与其他VTP节点的距离（考虑VTP网络协同，基于时间和距离）
            # 根据scheme判断VTP是发射点还是回收点，然后找k个时间往后/往前的距离最近的节点打分
            vtp_network_score = 0.0
            k_vtp_nodes = 5  # 考虑k个最近的VTP节点
            
            # 获取scheme，判断VTP是发射点还是回收点
            scheme = opt.get('scheme')
            is_launch_point = False
            is_recovery_point = False
            
            if scheme and len(scheme) >= 4:
                # scheme格式: (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                launch_node = scheme[1] if len(scheme) > 1 else None
                recovery_node = scheme[3] if len(scheme) > 3 else None
                
                if vtp_node == launch_node:
                    is_launch_point = True
                    is_vehicle_id = scheme[4]
                elif vtp_node == recovery_node:
                    is_recovery_point = True
                    is_vehicle_id = scheme[5]
            
            # 计算当前VTP节点的到达时间
            # 需要模拟插入VTP后的路线来计算到达时间
            temp_vehicle_routes = [route[:] for route in state.vehicle_routes]
            if vtp_insert_index is not None and vtp_insert_vehicle_id is not None:
                route_idx = vtp_insert_vehicle_id - 1
                if 0 <= route_idx < len(temp_vehicle_routes):
                    # 在临时路线中插入VTP节点（如果还没有插入）
                    if vtp_insert_index < len(temp_vehicle_routes[route_idx]):
                        if temp_vehicle_routes[route_idx][vtp_insert_index] != vtp_node:
                            temp_vehicle_routes[route_idx].insert(vtp_insert_index, vtp_node)
                    else:
                        temp_vehicle_routes[route_idx].append(vtp_node)
            
            # 计算所有车辆的到达时间
            temp_vehicle_arrive_time = state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_routes)
            
            # 获取当前VTP节点的到达时间
            vtp_arrive_time = None
            if vtp_insert_vehicle_id in temp_vehicle_arrive_time:
                if vtp_node in temp_vehicle_arrive_time[vtp_insert_vehicle_id]:
                    vtp_arrive_time = temp_vehicle_arrive_time[vtp_insert_vehicle_id][vtp_node]
            
            # 根据是发射点还是回收点，找k个时间往后/往前的距离最近的节点
            candidate_nodes = []  # 存储候选节点及其距离和时间
            
            # 遍历所有车辆的路线，找到符合条件的节点
            for vehicle_id, route in enumerate(state.vehicle_routes):
                vehicle_id = vehicle_id + 1
                if vehicle_id == is_vehicle_id:
                    continue
                if vehicle_id not in temp_vehicle_arrive_time:
                    continue
                
                vehicle_arrive_time_dict = temp_vehicle_arrive_time[vehicle_id]
                
                for node_id in route:
                    # 跳过初始节点
                    if node_id == self.DEPOT_nodeID:
                        continue
                    if node_id == vtp_node:
                        continue  # 跳过自身
                    
                    # 跳过非VTP节点（只考虑VTP节点和客户节点）
                    # 对于客户节点，通过map_cluster_vtp_dict找到最近的VTP节点
                    if node_id not in vehicle_arrive_time_dict:
                        continue
                    
                    node_arrive_time = vehicle_arrive_time_dict[node_id]
                    
                    # 判断时间条件
                    time_valid = False
                    if is_launch_point:
                        # 发射点：找时间往后的节点（node_arrive_time > vtp_arrive_time）
                        time_valid = node_arrive_time > vtp_arrive_time
                    elif is_recovery_point:
                        # 回收点：找时间往前的节点（node_arrive_time < vtp_arrive_time）
                        time_valid = node_arrive_time < vtp_arrive_time
                    else:
                        # 如果无法判断，则考虑所有节点
                        time_valid = True

                    if not time_valid:
                        continue
                    
                    # 计算距离
                    # 优先检查是否是VTP节点
                    if node_id in self.node:
                        # 直接是VTP节点
                        node_obj = self.node[node_id]
                        node_lat = node_obj.latDeg
                        node_lon = node_obj.lonDeg
                        node_alt = node_obj.altMeters
                        
                        lat_diff = vtp_lat - node_lat
                        lon_diff = vtp_lon - node_lon
                        alt_diff = vtp_alt - node_alt
                        distance = np.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
                        
                        candidate_nodes.append({
                            'node_id': node_id,
                            'distance': distance,
                            'arrive_time': node_arrive_time,
                            'vehicle_id': vehicle_id
                        })
                
            # 去重：同一个VTP节点可能被多个客户节点映射到，只保留距离最近的
            unique_nodes = {}
            for node_info in candidate_nodes:
                if node_info['node_id'] == self.DEPOT_nodeID:
                    continue
                node_id = node_info['node_id']
                if node_id not in unique_nodes:
                    unique_nodes[node_id] = node_info
            
            # 根据距离排序，选择k个最近的节点
            unique_candidate_nodes = list(unique_nodes.values())
            unique_candidate_nodes.sort(key=lambda x: x['distance'])
            selected_nodes = unique_candidate_nodes[:k_vtp_nodes]
            
            # 计算协同分数
            if selected_nodes:
                for node_info in selected_nodes:
                    distance = node_info['distance']
                    # 距离越近，协同效应越强
                    # 同时考虑时间因素：时间越接近，协同效应越强
                    time_diff = abs(node_info['arrive_time'] - vtp_arrive_time)
                    time_factor = 1.0 / (1.0 + time_diff)  # 时间越接近，因子越大
                    distance_factor = 1.0 / (1.0 + distance)
                    vtp_network_score += distance_factor * time_factor
                
                # 归一化：取平均值
                vtp_network_score = vtp_network_score / len(selected_nodes)
            
            # 8. 计算与neighbors的协同分数
            # neighbors是距离当前客户最近的k个被破坏的客户
            # 这些neighbors可以通过同一个VTP节点一起服务，产生协同效应
            # 注意：这里应该计算VTP节点到neighbor的距离，而不是customer到neighbor的距离
            
            # if not neighbors:
                # 如果没有neighbors，返回基础分（综合考虑所有因素）
                # 基础分 = 无人机数量因子 + 路径距离协同 + 客户距离协同 + 被破坏节点覆盖 + VTP网络协同
            base_score = 0.5 * (1.0 + np.log(1.0 + drone_count))
            synergy_score = (base_score + 
                            0.20 * route_proximity_score + 
                            0.25 * customer_proximity_score + 
                            0.20 * destroyed_node_score + 
                            0.15 * vtp_network_score)
            return synergy_score
            
            # # 计算每个neighbor的协同贡献（使用VTP节点到neighbor的距离）
            # neighbor_synergy_sum = 0.0
            # valid_neighbors = 0
            
            # for neighbor_id in neighbors:
            #     if neighbor_id == customer_id:
            #         continue  # 跳过自身
                
            #     if neighbor_id not in self.node:
            #         continue
                
            #     neighbor_node = self.node[neighbor_id]
            #     neighbor_lat = neighbor_node.latDeg
            #     neighbor_lon = neighbor_node.lonDeg
            #     neighbor_alt = neighbor_node.altMeters
                
            #     # 计算VTP节点与neighbor的距离（而不是customer与neighbor的距离）
            #     lat_diff = vtp_lat - neighbor_lat
            #     lon_diff = vtp_lon - neighbor_lon
            #     alt_diff = vtp_alt - neighbor_alt
            #     distance = np.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
                
            #     # 距离越近，协同贡献越大
            #     # 使用 1/(1+distance) 作为距离因子，避免除以0
            #     distance_factor = 1.0 / (1.0 + distance)
                
            #     # 计算neighbor的协同贡献
            #     neighbor_synergy = distance_factor
            #     neighbor_synergy_sum += neighbor_synergy
            #     valid_neighbors += 1
            
            # # 9. 综合计算协同分数
            # # 协同分数 = (neighbors的协同贡献) * (无人机数量因子) + (路径距离因子) + (客户距离因子) + (被破坏节点覆盖因子) + (VTP网络因子)
            # # 无人机数量因子：无人机越多，可以服务的neighbors越多，协同效应越强
            
            # if valid_neighbors > 0:
            #     # 平均每个neighbor的协同贡献
            #     avg_neighbor_synergy = neighbor_synergy_sum / valid_neighbors
                
            #     # 无人机数量因子：考虑无人机数量对协同的放大作用
            #     # 无人机数量越多，可以同时服务的neighbors越多
            #     # 使用对数函数避免无人机数量过多时分数过大
            #     drone_factor = 1.0 + np.log(1.0 + drone_count) / np.log(1.0 + len(self.V) if self.V else 1)
                
            #     # 考虑可以服务的neighbors数量（受无人机数量限制）
            #     # 假设每个无人机可以服务一个neighbor（除了当前客户）
            #     serviceable_neighbors = min(valid_neighbors, max(0, drone_count - 1))
                
            #     # 邻居协同分数（主要部分）
            #     neighbor_synergy_score = avg_neighbor_synergy * valid_neighbors * drone_factor * (1.0 + serviceable_neighbors / max(1, valid_neighbors))
                
            #     # 最终协同分数 = 邻居协同 + 路径距离协同 + 客户距离协同 + 被破坏节点覆盖 + VTP网络协同
            #     # 权重分配：邻居协同（主要）> 路径距离 > 客户距离 > 被破坏节点覆盖 > VTP网络
            #     synergy_score = (neighbor_synergy_score + 
            #                     0.20 * route_proximity_score + 
            #                     0.15 * customer_proximity_score + 
            #                     0.15 * destroyed_node_score + 
            #                     0.10 * vtp_network_score)
            # else:
            #     # 没有有效的neighbors，综合考虑所有因素
            #     base_score = 0.5 * (1.0 + np.log(1.0 + drone_count))
            #     synergy_score = (base_score + 
            #                     0.25 * route_proximity_score + 
            #                     0.20 * customer_proximity_score + 
            #                     0.20 * destroyed_node_score + 
            #                     0.15 * vtp_network_score)
            
        except Exception as e:
            print(f"Error calculating synergy score: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        
        return synergy_score

    def _calculate_partner_score(self, opt, all_vtps, state):
        """
        (VTP-Knn-regret 辅助函数)
        计算伙伴分 (Partner Score)
        """
        return 0.0

    def _find_k_nearest_unassigned(self, customer_id, k, destroy_node):
        """
        (VTP-Knn-regret 辅助函数)
        为单个客户customer_id，在所有待修复客户destroy_node中，找到 K 个距离最近的客户。
        
        Args:
            customer_id (int): 目标客户节点ID
            k (int): 需要找到的最近邻居数量
            destroy_node (list): 待修复客户节点ID列表
            
        Returns:
            list: 距离customer_id最近的k个客户节点ID列表（按距离从近到远排序）
        """
        if customer_id not in self.node:
            return []
        
        # 获取目标客户的坐标
        target_node = self.node[customer_id]
        target_lat = target_node.latDeg
        target_lon = target_node.lonDeg
        target_alt = target_node.altMeters
        
        # 计算距离函数
        def calculate_distance(other_id):
            if other_id not in self.node:
                return float('inf')
            other_node = self.node[other_id]
            # 计算欧几里得距离（考虑经纬度和高度）
            lat_diff = target_lat - other_node.latDeg
            lon_diff = target_lon - other_node.lonDeg
            alt_diff = target_alt - other_node.altMeters
            distance = np.sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)
            return distance
        
        # 排除自身（如果customer_id在destroy_node中）
        candidates = [x for x in destroy_node if x != customer_id]
        
        if not candidates:
            return []
        
        # 按距离排序并返回前k个
        sorted_candidates = sorted(candidates, key=calculate_distance)
        return sorted_candidates[:k]

    def _find_k_best_vehicles_for_shared_vtp(self, vtp_shared, state, k):
        """
        (VTP-Centric 辅助函数)
        为单个【已使用】的VTP，在所有【尚未】使用它的车辆中，找到 K 个成本最低的插入方案。
        
        Returns:
            list: [(veh_id, insert_idx, veh_delta_cost), ...] (按成本升序排列, 最多K个)
        """
        all_insertion_options = []
        
        for vehicle_idx, route in enumerate(state.vehicle_routes):
            vehicle_id = vehicle_idx + 1
            
            # 【关键】: 检查该车辆是否【尚未】使用此VTP (满足您的约束)
            if vtp_shared not in route:
                if len(route) < 2: continue
                
                # 找到插入到该车辆的最佳位置
                best_idx_for_this_vehicle = -1
                min_delta_for_this_vehicle = float('inf')
                
                for i in range(1, len(route)):
                    prev_node = route[i - 1]
                    next_node = route[i]
                    try:
                        delta_cost = self.veh_distance[vehicle_id][prev_node][vtp_shared] + \
                                    self.veh_distance[vehicle_id][vtp_shared][next_node] - \
                                    self.veh_distance[vehicle_id][prev_node][next_node]
                        delta_cost = delta_cost * self.vehicle[vehicle_id].per_cost
                        all_insertion_options.append((vehicle_id, i, delta_cost))
                        # if delta_cost < min_delta_for_this_vehicle:
                        #     min_delta_for_this_vehicle = delta_cost
                        #     best_idx_for_this_vehicle = i
                    except KeyError:
                        continue
                
                    # # 如果找到了一个可插入的位置，就加入候选,修改前索引在前面
                    # if best_idx_for_this_vehicle != -1:
                    #     all_insertion_options.append((vehicle_id, best_idx_for_this_vehicle, min_delta_for_this_vehicle))

        # 按成本升序排序
        all_insertion_options.sort(key=lambda x: x[2])
        
        # 返回前 K 个
        return all_insertion_options[:k]

    def _create_temp_state_with_new_vtp(self, state, vtp_new, veh_id, insert_idx):
        """(新) 创建一个插入了vtp_new的临时状态用于评估。"""
        temp_state = state.fast_copy()
        temp_route = temp_state.vehicle_routes[veh_id - 1]
        temp_route.insert(insert_idx, vtp_new)
        
        try:
            last_customer_node = temp_route[insert_idx - 1]
            if insert_idx == 1 or last_customer_node == self.DEPOT_nodeID:
                last_drone_list = self.base_drone_assignment[veh_id][:]
            else:
                last_drone_list = temp_state.vehicle_task_data[veh_id][last_customer_node].drone_list[:]
            
            from task_data import TaskData # 假设
            temp_state.vehicle_task_data[veh_id][vtp_new] = TaskData(drone_list=last_drone_list, launch_drone_list=[], recovery_drone_list=[])
            
            return temp_state, True
        except Exception as e:
            print(f"  > 警告: _create_temp_state_with_new_vtp 初始化 {vtp_new} 失败: {e}")
            return temp_state, False

    def _evaluate_insertion_with_specific_vtp(self, customer, vtp_new, vtp_info, temp_state, temp_arrive_time):
        """
        (新) 评估客户 customer 使用【特定】新VTP vtp_new 的最低成本方案。
        (这是一个简化的评估器，只检查以 vtp_new 为一端的方案)
        """
        min_cost = float('inf')
        best_scheme = None
        (launch_veh_id, launch_idx) = vtp_info # vtp_new 被插入的车辆和索引

        # 1. 模拟新发 (vtp_new) -> 旧收 (P_used)
        new_launch_time = temp_arrive_time[launch_veh_id][vtp_new]
        for rec_veh_id_idx, rec_route in enumerate(temp_state.vehicle_routes):
            rec_veh_id = rec_veh_id_idx + 1
            for rec_vtp in rec_route[1:-1]:
                if rec_vtp == vtp_new and launch_veh_id == rec_veh_id: continue # 避免同点
                
                # (省略复杂的冲突检查，仅做时序和成本)
                if temp_arrive_time[rec_veh_id][rec_vtp] > new_launch_time:
                    for drone_id in self.V: # 遍历所有无人机
                        # (应检查无人机此时是否在 vtp_new 上)
                        cost = self.drone_insert_cost(drone_id, customer, vtp_new, rec_vtp)
                        if cost < min_cost:
                            min_cost = cost
                            best_scheme = (drone_id, vtp_new, customer, rec_vtp, launch_veh_id, rec_veh_id)

        # 2. 模拟旧发 (P_used) -> 新收 (vtp_new)
        new_rec_time = temp_arrive_time[launch_veh_id][vtp_new]
        for launch_veh_id_idx, launch_route in enumerate(temp_state.vehicle_routes):
            launch_veh_id_old = launch_veh_id_idx + 1
            for launch_vtp in launch_route[1:-1]:
                if launch_vtp == vtp_new and launch_veh_id_old == launch_veh_id: continue
                
                if new_rec_time > temp_arrive_time[launch_veh_id_old][launch_vtp]:
                    for drone_id in self.V:
                        # (应检查无人机此时是否在 launch_vtp 上)
                        cost = self.drone_insert_cost(drone_id, customer, launch_vtp, vtp_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_scheme = (drone_id, launch_vtp, customer, vtp_new, launch_veh_id_old, launch_veh_id)
                            
        if best_scheme:
            return min_cost, best_scheme
        else:
            return float('inf'), None

    def _get_all_candidate_new_vtps(self, customers, state):
        """(新) 从所有待修复客户的邻近VTP中，筛选出未被使用的候选VTP。"""
        used_vtps = {node for route in state.vehicle_routes for node in route[1:-1]}
        candidate_vtps = set()
        for customer in customers:
            K_NEIGHBORS = 20 
            neighbors = self.map_cluster_vtp_dict.get(customer, [])[:K_NEIGHBORS]
            # neighbors = self.map_cluster_vtp_dict.get(customer, [])[:]
            for vtp in neighbors:
                if vtp not in used_vtps:
                    candidate_vtps.add(vtp)
        return list(candidate_vtps)



    # 在 IncrementalALNS 类中
    def _find_best_vehicle_for_new_vtp(self, vtp_new, repaired_state):
        """
        (VTP-Centric 辅助函数)
        为单个【新】VTP候选节点，在【所有】车辆中找到能以最低代价接纳它的“家”。
        
        此函数会遍历所有车辆的所有路径段，计算插入 vtp_new 的车辆绕路成本，
        并返回全局最优（成本最低）的插入方案。

        Args:
            vtp_new (int): 待评估的【新】VTP节点ID。
            repaired_state (FastMfstspState): 当前正在修复中的状态对象，
                                            包含 vehicle_routes。

        Returns:
            tuple: (best_vehicle_id, best_insert_idx, min_overall_delta)
                - best_vehicle_id (int): 最佳插入车辆的ID (从1开始)。
                - best_insert_idx (int): 在该车辆路线中的最佳插入索引 (从1开始)。
                - min_overall_delta (float): 对应的最低车辆绕路成本。
                
                如果找不到任何可插入的位置（例如距离数据缺失），
                则返回 (-1, -1, float('inf'))。
        """
        best_vehicle_id = -1
        best_insert_idx = -1
        min_overall_delta = float('inf')
        epsilon = 1e-6 # 用于比较
        # vehice_per_cost = self.vehicle[1].per_cost

        # 遍历所有车辆的路线
        # 假设 repaired_state.vehicle_routes 是一个列表，索引 0 对应车辆 1
        for vehicle_idx, route in enumerate(repaired_state.vehicle_routes):
            vehicle_id = vehicle_idx + 1 # 车辆ID从1开始
            
            # 路径必须至少有两个节点（如 [Depot, Depot]）才能插入
            if len(route) < 2:
                continue
                
            # 遍历路线中的【每一个】路段 (i-1) -> (i)，尝试插入 vtp_new
            # 插入索引 i 的范围是从 1 (在第一个节点后) 到 len(route)-1 (在倒数第二个节点后)
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                next_node = route[i]
                
                try:
                    # 计算车辆绕路成本（“投资成本”）
                    # delta = (I -> J) + (J -> K) - (I -> K)
                    delta_cost = self.veh_distance[vehicle_id][prev_node][vtp_new] + \
                                self.veh_distance[vehicle_id][vtp_new][next_node] - \
                                self.veh_distance[vehicle_id][prev_node][next_node]
                    delta_cost = delta_cost * self.vehicle[vehicle_id].per_cost
                    # 检查这是否是迄今为止全局最好的插入方案
                    if delta_cost < min_overall_delta:
                        min_overall_delta = delta_cost
                        best_vehicle_id = vehicle_id
                        best_insert_idx = i # 插入到索引 i 处

                except (KeyError, IndexError) as e:
                    # 如果缺少距离数据（例如 vtp_new 不在距离矩阵中），则跳过此位置
                    # print(f"  > 警告: 无法计算VTP {vtp_new} 插入 车辆{vehicle_id} 路线 {prev_node}->{next_node} 的绕路成本。错误: {e}")
                    continue # 跳到下一个插入位置

        # 如果 min_overall_delta 仍然是无穷大，保持返回 (-1, -1, float('inf'))
        if min_overall_delta == float('inf'):
            print(f"  > 警告: 无法为新VTP {vtp_new} 找到任何可插入的车辆路径。")
            return -1, -1, float('inf')
            
        return best_vehicle_id, best_insert_idx, min_overall_delta

    def _create_temp_state_with_new_vtp(self, state, vtp_new, veh_id, insert_idx):
        """
        (VTP-Centric 辅助函数)
        创建一个插入了 vtp_new 的临时状态副本，并正确初始化新节点的 vehicle_task_data。

        Args:
            state (FastMfstspState): 【原始】的被破坏状态。
            vtp_new (int): 待插入的新VTP节点ID。
            veh_id (int): 要插入的车辆ID (从1开始)。
            insert_idx (int): 在该车辆路线中的插入索引 (从1开始)。

        Returns:
            tuple: (temp_state, success_flag)
                - temp_state (FastMfstspState): 一个【新的】状态副本，
                                            包含了修改后的 vehicle_routes 和 vehicle_task_data。
                - success_flag (bool): 初始化是否成功。
        """
        
        # 1. 创建一个安全的状态副本
        temp_state = state.fast_copy()
        
        try:
            # 2. 在副本上【物理插入】VTP
            route_index = veh_id - 1 # 转换为0-based索引
            
            # 确保 vehicle_routes 索引有效
            if route_index < 0 or route_index >= len(temp_state.vehicle_routes):
                print(f"  > 警告: _create_temp_state - 车辆索引 {route_index} (ID: {veh_id}) 无效。")
                return state, False # 返回原始状态和失败

            temp_route = temp_state.vehicle_routes[route_index]
            
            # 确保插入索引有效 (范围应为 [1, len(route)])
            if insert_idx < 1 or insert_idx > len(temp_route):
                print(f"  > 警告: _create_temp_state - 插入索引 {insert_idx} 在路线 {route} 中无效。")
                return state, False
                
            temp_route.insert(insert_idx, vtp_new)
            
            # 3. 【关键】为新插入的VTP初始化 vehicle_task_data
            
            # a. 找到前一个节点以继承无人机列表
            prev_node = temp_route[insert_idx - 1]
            
            # b. 确定新节点的初始无人机列表
            initial_drone_list = []
            if insert_idx == 1 or prev_node == self.DEPOT_nodeID:
                # 如果插在最前面(索引1)，或前一个节点是仓库，则使用车辆的基础分配
                initial_drone_list = self.base_drone_assignment.get(veh_id, [])[:]
            else:
                # 否则，继承前一个节点的 drone_list
                # 使用 .get() 链确保安全访问
                prev_task_data = temp_state.vehicle_task_data.get(veh_id, {}).get(prev_node)
                if prev_task_data and hasattr(prev_task_data, 'drone_list'):
                    initial_drone_list = prev_task_data.drone_list[:]
                else:
                    print(f"  > 警告: _create_temp_state - 无法找到前序节点 {prev_node} 的 drone_list。")
                    # 此处可以根据您的业务逻辑决定是失败还是使用空列表
                    # return state, False # 严格模式：失败
                    initial_drone_list = [] # 宽松模式：使用空列表

            # c. 创建并设置新节点的 TaskData
            # from task_data import TaskData # 确保 TaskData 类已导入
            if veh_id not in temp_state.vehicle_task_data:
                temp_state.vehicle_task_data[veh_id] = {}
                
            temp_state.vehicle_task_data[veh_id][vtp_new].drone_list = initial_drone_list
            temp_state.vehicle_task_data[veh_id][vtp_new].launch_drone_list = []
            temp_state.vehicle_task_data[veh_id][vtp_new].recovery_drone_list = []
            temp_vehicle_time = temp_state.calculate_rm_empty_vehicle_arrive_time(temp_state.vehicle_routes)
            # 判断方案是否可行
            if is_time_feasible(temp_state.customer_plan, temp_vehicle_time):
                return temp_state, True
            else:
                return state, False # 返回原始状态和失败

        except Exception as e:
            import traceback
            print(f"  > 严重错误: _create_temp_state_with_new_vtp 失败: {e}")
            traceback.print_exc()
            return state, False

    def _evaluate_insertion_with_specific_vtp(self, customer, 
                                            vtp_new, vtp_info, 
                                            temp_state, temp_arrive_time):
        """
        (VTP-Centric 辅助函数)
        在【临时状态】下，评估客户 customer 使用【特定】新VTP vtp_new 的最低成本方案。
        
        Args:
            customer (int): 目标客户ID。
            vtp_new (int): 【已插入】的VTP节点ID。
            vtp_info (tuple): (vehicle_id, insert_idx) vtp_new被插入的信息。
            temp_state (FastMfstspState): 已经【包含】vtp_new的临时状态对象。
            temp_arrive_time (dict): 基于 temp_state 的【新】到达时间。

        Returns:
            tuple: (min_cost, best_scheme) 或 (float('inf'), None)
        """
        min_cost = float('inf')
        best_scheme = None
        
        (new_vtp_veh_id, new_vtp_idx) = vtp_info
        
        # ------------------------------------------------------------------
        # 场景 A: “新发旧收” (vtp_new 作为发射点)
        # ------------------------------------------------------------------
        
        # 1a. 检查 vtp_new 作为发射点的可行性
        try:
            launch_node = vtp_new
            launch_veh_id = new_vtp_veh_id
            launch_route = temp_state.vehicle_routes[launch_veh_id - 1]
            launch_idx = new_vtp_idx # 我们已经知道它的索引
            launch_time = temp_arrive_time[launch_veh_id][launch_node]
            n_launch = len(launch_route)
        except Exception as e:
            # print(f"  > 警告: VTP评估 - 无法获取新发射点 {vtp_new} 的信息: {e}")
            pass # 如果出错，则跳过场景A
        else:
            # 1b. 遍历所有【现有VTP】作为回收点 (包括 vtp_new 所在的车辆)
            for rec_veh_idx, rec_route in enumerate(temp_state.vehicle_routes):
                rec_veh_id = rec_veh_idx + 1
                n_rec = len(rec_route)

                for k in range(1, n_rec - 1): # 遍历所有节点 (包括 vtp_new 自身)
                    recovery_node = rec_route[k]
                    
                    # 不允许同点起降
                    if launch_veh_id == rec_veh_id and launch_node == recovery_node:
                        continue
                    
                    try:
                        recovery_time = temp_arrive_time[rec_veh_id][recovery_node]
                    except KeyError:
                        continue # 节点时间不可达

                    # c. 检查时序
                    if recovery_time <= launch_time:
                        continue
                    
                    # d. 遍历所有无人机
                    for drone_id in self.V:
                        # i. 检查无人机是否在发射点
                        if drone_id not in temp_state.vehicle_task_data[launch_veh_id][launch_node].drone_list:
                            continue
                            
                        # ii. 检查路径冲突 (与 _calculate_launch... 逻辑相同)
                        conflict = False
                        if launch_veh_id == rec_veh_id: # 同车
                            rec_idx = k
                            for m in range(launch_idx + 1, rec_idx):
                                if drone_id in temp_state.vehicle_task_data[launch_veh_id][launch_route[m]].launch_drone_list:
                                    conflict = True; break
                        else: # 跨车
                            # 检查发射车
                            for m in range(launch_idx + 1, n_launch - 1):
                                if drone_id in temp_state.vehicle_task_data[launch_veh_id][launch_route[m]].launch_drone_list:
                                    conflict = True; break
                            if conflict: continue
                            # 检查回收车
                            for m in range(1, k):
                                if drone_id in temp_state.vehicle_task_data[rec_veh_id][rec_route[m]].launch_drone_list:
                                    conflict = True; break
                        
                        if conflict: continue

                        # e. 计算成本
                        cost = self.drone_insert_cost(drone_id, customer, launch_node, recovery_node)
                        if cost is not None and cost < min_cost:
                            min_cost = cost
                            best_scheme = (drone_id, launch_node, customer, recovery_node, launch_veh_id, rec_veh_id)

        # ------------------------------------------------------------------
        # 场景 B: “旧发新收” (vtp_new 作为回收点)
        # ------------------------------------------------------------------
        
        # 2a. 获取 vtp_new 作为回收点的信息
        try:
            recovery_node = vtp_new
            rec_veh_id = new_vtp_veh_id
            rec_route = temp_state.vehicle_routes[rec_veh_id - 1]
            rec_idx = new_vtp_idx
            recovery_time = temp_arrive_time[rec_veh_id][recovery_node]
            n_recovery = len(rec_route)
        except Exception as e:
            # print(f"  > 警告: VTP评估 - 无法获取新回收点 {vtp_new} 的信息: {e}")
            pass # 如果出错，则跳过场景B
        else:
            # 2b. 遍历所有【现有VTP】作为发射点
            for launch_veh_idx, launch_route in enumerate(temp_state.vehicle_routes):
                launch_veh_id = launch_veh_idx + 1
                n_launch = len(launch_route)

                for i in range(1, n_launch - 1):
                    launch_node = launch_route[i]
                    
                    # 不允许同点起降
                    if launch_veh_id == rec_veh_id and launch_node == recovery_node:
                        continue

                    try:
                        launch_time = temp_arrive_time[launch_veh_id][launch_node]
                    except KeyError:
                        continue

                    # c. 检查时序
                    if recovery_time <= launch_time:
                        continue
                        
                    # d. 遍历所有无人机
                    for drone_id in self.V:
                        # i. 检查无人机是否在发射点
                        if drone_id not in temp_state.vehicle_task_data[launch_veh_id][launch_node].drone_list:
                            continue

                        # ii. 检查路径冲突 (与 _calculate_recovery... 逻辑相同)
                        conflict = False
                        if launch_veh_id == rec_veh_id: # 同车
                            for m in range(i + 1, rec_idx):
                                if drone_id in temp_state.vehicle_task_data[launch_veh_id][launch_route[m]].launch_drone_list:
                                    conflict = True; break
                        else: # 跨车
                            # 检查发射车
                            for m in range(i + 1, n_launch - 1):
                                if drone_id in temp_state.vehicle_task_data[launch_veh_id][launch_route[m]].launch_drone_list:
                                    conflict = True; break
                            if conflict: continue
                            # 检查回收车
                            for m in range(1, rec_idx):
                                if drone_id in temp_state.vehicle_task_data[rec_veh_id][rec_route[m]].launch_drone_list:
                                    conflict = True; break
                        
                        if conflict: continue
                        
                        # e. 计算成本
                        cost = self.drone_insert_cost(drone_id, customer, launch_node, recovery_node)
                        if cost is not None and cost < min_cost:
                            min_cost = cost
                            best_scheme = (drone_id, launch_node, customer, recovery_node, launch_veh_id, rec_veh_id)

        # ------------------------------------------------------------------
        # 3. 返回结果
        # ------------------------------------------------------------------
        if best_scheme:
            return min_cost, best_scheme
        else:
            return float('inf'), None

    def _calculate_vtp_benefits(self, vtp_new, vtp_info, state, customers_to_repair,temp_vtp_task_data):
        """
        (VTP-Centric 辅助函数)
        计算一个【特定】的VTP投资方案 (插入 vtp_new 到 vtp_info 指定的位置)
        能为所有待修复客户带来的【总净收益】。

        Args:
            vtp_new (int): 待评估的【新】VTP节点ID。
            vtp_info (tuple): (vehicle_id, insert_idx) VTP的插入位置信息。
            state (FastMfstspState): 【原始】的被破坏状态 (fast_copy将在内部创建)。
            customers_to_repair (list): 待修复的客户ID列表。
            baseline_costs (dict): {customer: (cost, scheme)}，不新增VTP时的最低成本。

        Returns:
            tuple: (total_benefit, affected_customers_dict)
                - total_benefit (float): 所有客户净收益的总和。
                - affected_customers_dict (dict): {customer: (new_cost, new_scheme)} 
                                                仅包含那些实际获益的客户。
        """
        total_benefit = {}
        affected_customers = {} # 存储 {customer: (new_cost, new_scheme)}
        (veh_id, insert_idx) = vtp_info
        epsilon = 1e-6 # 用于浮点数比较，避免 0.00001 的误差
        # ------------------------------------------------------------------
        # 2. 遍历所有待修复客户，计算使用新VTP的收益
        # ------------------------------------------------------------------
        vehicle_route = [route[:] for route in state.vehicle_routes]
        # vtp_task_data = deep_copy_vehicle_task_data(state.vehicle_task_data)
        vtp_task_data = temp_vtp_task_data
        # 模拟插入最优vtp节点后的
        temp_vehicle_route = [route[:] for route in vehicle_route]
        temp_route = temp_vehicle_route[veh_id - 1]
        temp_route.insert(insert_idx, vtp_new)
        temp_vehicle_route[veh_id - 1] = temp_route
        temp_rm_vehicle_arrive_time = state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
        if not is_time_feasible(state.customer_plan, temp_rm_vehicle_arrive_time):
            return float('inf'), {}

        # for customer in customers_to_repair:
        customer = customers_to_repair
        # b. 计算客户 k 使用这个【特定】新VTP的最低成本 (新发旧收/旧发新收)
        #    这个函数需要在【临时】状态上操作
        vehicle_id = vtp_info[0]
        insert_pos = vtp_info[1]
        vehicle_idx = vehicle_id - 1
        new_cost, new_scheme = self._calculate_vtp_expansion_cost(customer, vehicle_idx, insert_pos, vehicle_route, vtp_task_data, state, vtp_new)  # 此处返回总成本
        # if new_cost == float('inf'):
        #     continue
        
        if new_scheme: # 如果找到了一个使用新VTP的可行方案
            total_benefit[customer] = new_cost
            affected_customers[customer] = (new_cost, new_scheme)
        # 根据affected_customers。按照cost排序,从低到高
        affected_customers = sorted(affected_customers.items(), key=lambda x: x[1])
        return total_benefit, affected_customers

    # def _calculate_vtp_benefits(self, vtp_new, vtp_info, state, customers_to_repair, baseline_costs):
    #     """
    #     (VTP-Centric 辅助函数)
    #     计算一个【特定】的VTP投资方案 (插入 vtp_new 到 vtp_info 指定的位置)
    #     能为所有待修复客户带来的【总净收益】。

    #     Args:
    #         vtp_new (int): 待评估的【新】VTP节点ID。
    #         vtp_info (tuple): (vehicle_id, insert_idx) VTP的插入位置信息。
    #         state (FastMfstspState): 【原始】的被破坏状态 (fast_copy将在内部创建)。
    #         customers_to_repair (list): 待修复的客户ID列表。
    #         baseline_costs (dict): {customer: (cost, scheme)}，不新增VTP时的最低成本。

    #     Returns:
    #         tuple: (total_benefit, affected_customers_dict)
    #             - total_benefit (float): 所有客户净收益的总和。
    #             - affected_customers_dict (dict): {customer: (new_cost, new_scheme)} 
    #                                             仅包含那些实际获益的客户。
    #     """
    #     total_benefit = 0.0
    #     affected_customers = {} # 存储 {customer: (new_cost, new_scheme)}
    #     (veh_id, insert_idx) = vtp_info
    #     epsilon = 1e-6 # 用于浮点数比较，避免 0.00001 的误差
    #     # ------------------------------------------------------------------
    #     # 2. 遍历所有待修复客户，计算使用新VTP的收益
    #     # ------------------------------------------------------------------
    #     vehicle_route = [route[:] for route in state.vehicle_routes]
    #     vtp_task_data = deep_copy_vehicle_task_data(state.vehicle_task_data)
    #     # 模拟插入最优vtp节点后的
    #     temp_vehicle_route = [route[:] for route in vehicle_route]
    #     temp_route = temp_vehicle_route[veh_id - 1]
    #     temp_route.insert(insert_idx, vtp_new)
    #     temp_vehicle_route[veh_id - 1] = temp_route
    #     temp_rm_vehicle_arrive_time = state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
    #     if not is_time_feasible(state.customer_plan, temp_rm_vehicle_arrive_time):
    #         return 0.0, {}

    #     for customer in customers_to_repair:
            
    #         # a. 获取该客户的基线成本 (不使用新VTP的最低成本)
    #         baseline_cost = baseline_costs.get(customer, (float('inf'), None))[0]
    #         # 【核心修正】: 用“大M惩罚值”替换 'inf'，以便进行数学比较
    #         if baseline_cost == float('inf'):
    #             # (假设 M_PENALTY 在 __init__ 中定义，例如 self.M_PENALTY = 100000.0)
    #             baseline_cost = 1000
    #         # b. 计算客户 k 使用这个【特定】新VTP的最低成本 (新发旧收/旧发新收)
    #         #    这个函数需要在【临时】状态上操作
    #         vehicle_id = vtp_info[0]
    #         insert_pos = vtp_info[1]
    #         vehicle_idx = vehicle_id - 1
    #         new_cost, new_scheme = self._calculate_vtp_expansion_cost(customer, vehicle_idx, insert_pos, vehicle_route, vtp_task_data, state, vtp_new)
    #         if new_cost == float('inf'):
    #             continue
            
    #         if new_scheme: # 如果找到了一个使用新VTP的可行方案
    #             # c. 计算此客户的净收益
    #             #    net_benefit > 0 意味着使用新VTP比基线方案更好
    #             net_benefit = baseline_cost - new_cost
                
    #             if net_benefit > epsilon: # 必须是严格的正收益
    #                 total_benefit += net_benefit
    #                 affected_customers[customer] = (new_cost, new_scheme)
    #             # else:
    #             #    print(f"  > 诊断[VTP评估]: 客户 {customer} 使用 VTP {vtp_new} 成本({new_cost}) 不优于基线 ({baseline_cost})。")
            
    #     return total_benefit, affected_customers

    def repair_vtp_centric(self, state, strategic_bonus=0, num_destroyed=1, force_vtp_mode=False):
        """
        设计对应的vtp中心批量修复策略
        主动寻找并插入“最具潜力”的新VTP，然后批量修复所有能从中获益的客户。
        """
        repaired_state = state.fast_copy()
        repaired_state.repair_objective = 0
        destroy_node = list(state.destroyed_customers_info.keys())
        insert_plan = []

        force_vtp_mode = True
        if force_vtp_mode:
            num_repaired = 0
            # --- 步骤 1: 评估所有待修复客户的“基线成本” ---
            #     (不新增VTP的情况下，修复每个客户的最低成本)
            baseline_costs = {} # {customer: (cost, scheme)}
            vehicle_route = repaired_state.vehicle_routes
            vehicle_task_data = repaired_state.vehicle_task_data
            vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
            for customer in destroy_node:
                # 仅评估传统插入方案 (调用您现有的评估函数)
                traditional_result, is_swap = self._evaluate_traditional_insertion(
                    customer, vehicle_route, vehicle_task_data,
                    vehicle_arrive_time, repaired_state, 
                )
                if traditional_result and not is_swap:
                    baseline_costs[customer] = (traditional_result[0], traditional_result[1])
                else:
                    baseline_costs[customer] = (float('inf'), None)
            # --- 步骤 2: 寻找“最具潜力”的【结构性改变】 ---
            best_modification = None # 存储最佳方案的完整信息
            best_vtp_score = 0.0 # 【重要】: 净收益必须大于0才值得投资
            best_vtp_affected_customers = {} 

            # a. 计算本轮的“最终奖励”(final_bonus)
            final_bonus = 0.0
            if num_destroyed > 0 and strategic_bonus > 0:
                tactical_multiplier = (num_destroyed - num_repaired) / max(num_destroyed, 1)
                final_bonus = strategic_bonus * tactical_multiplier 
                
            print(f"  > [VTP中心决策]: (Bonus: {final_bonus:.2f})")

            # --- 2.1 策略A：评估“投资新VTP” (Investment) ---
            candidate_new_vtps = self._get_all_candidate_new_vtps(destroy_node, repaired_state)
            print(f"  > [VTP中心-投资]: 评估 {len(candidate_new_vtps)} 个全新VTP...")
            # K_position = len(candidate_new_vtps)
            K_position = 10
            for vtp_new in candidate_new_vtps:
                # 找到插入 vtp_new 的最佳车辆和成本 # best_positions 返回: [(veh_id, insert_idx, veh_delta_cost), ...]
                best_positions = self._find_k_best_vehicle_for_new_vtp(vtp_new, repaired_state,K_position)  # 输出的车辆id并非索引而是代号
                if not best_positions: continue
                # 【核心修改】: 遍历这K个最佳插入位置，评估每一个的潜力
                for (veh_id, insert_idx, veh_delta_cost) in best_positions:
                    
                    # 估算总收益
                    total_benefit, affected_customers = self._calculate_vtp_benefits(
                        vtp_new, (veh_id, insert_idx), repaired_state, destroy_node, baseline_costs
                    )
                    
                    # 计算潜力分数
                    score = (total_benefit * (1 + final_bonus)) - veh_delta_cost

                    if score > best_vtp_score:
                        best_vtp_score = score
                        best_modification = {'vtp_node': vtp_new, 'veh_id': veh_id, 'idx': insert_idx, 'type': 'investment'}
                        best_vtp_affected_customers = affected_customers
            # --- 2.2 策略B：评估“共享现有VTP” (Sharing) ---
            used_vtps_set = {node for route in repaired_state.vehicle_routes for node in route[1:-1]}
            print(f"  > [VTP中心-共享]: 评估 {len(used_vtps_set)} 个现有VTP的共享潜力...")
            K_BEST_POSITIONS = len(used_vtps_set)
            # K_BEST_POSITIONS = 10
            for vtp_shared in used_vtps_set:
                # 【核心修改】: 为这个共享VTP，在所有【尚未】使用它的车辆中，找到K个最佳插入位置
                best_shared_positions = self._find_k_best_vehicles_for_shared_vtp(vtp_shared, repaired_state, K_BEST_POSITIONS)

                if not best_shared_positions: continue

                # 【核心修改】: 遍历这K个最佳共享位置
                for (veh_id, insert_idx, veh_delta_cost) in best_shared_positions:
                    
                    # 估算这个“共享方案”带来的总收益
                    total_benefit, affected_customers = self._calculate_vtp_benefits(
                        vtp_shared, (veh_id, insert_idx), repaired_state, destroy_node, baseline_costs
                    )
                    
                    # 计算潜力分数
                    score = (total_benefit * (1 + final_bonus)) - veh_delta_cost

                    if score > best_vtp_score:
                        best_vtp_score = score
                        best_modification = {'vtp_node': vtp_shared, 'veh_id': veh_id, 'idx': insert_idx, 'type': 'sharing'}
                        best_vtp_affected_customers = affected_customers
            # --- 步骤 3: 执行最佳决策 (无论是投资还是共享) ---
            if best_modification: # best_vtp_score 必须大于 0
                vtp_node = best_modification['vtp_node']
                vtp_insert_vehicle_id = best_modification['veh_id']
                vtp_insert_index = best_modification['idx']
                
                print(f"  > [VTP中心决策]: {best_modification['type']} VTP {vtp_node} (车辆 {vtp_insert_vehicle_id}), 潜力分数: {best_vtp_score:.2f}, 批量修复 {len(best_vtp_affected_customers)} 个客户。")
                
                # a. 真实地插入VTP
                route = repaired_state.vehicle_routes[vtp_insert_vehicle_id - 1]
                route.insert(vtp_insert_index, vtp_node)
                
                # b. 为新VTP初始化 vehicle_task_data
                # (与您 regret 中的 vtp_expansion 插入逻辑完全一致)
                last_customer_node = route[vtp_insert_index - 1]
                if vtp_insert_index == 1 or last_customer_node == self.DEPOT_nodeID:
                    last_drone_list = self.base_drone_assignment[vtp_insert_vehicle_id][:]
                else:
                    last_drone_list = vehicle_task_data[vtp_insert_vehicle_id][last_customer_node].drone_list[:]
                
                vehicle_task_data[vtp_insert_vehicle_id][vtp_node].drone_list = last_drone_list
                vehicle_task_data[vtp_insert_vehicle_id][vtp_node].launch_drone_list = []
                vehicle_task_data[vtp_insert_vehicle_id][vtp_node].recovery_drone_list = []
                
                # c. 【批量修复】所有受益的客户
                #    按收益降序排列，优先修复获益最大的
                sorted_affected_customers = sorted(
                    best_vtp_affected_customers.items(), 
                    key=lambda item: baseline_costs.get(item[0], (float('inf'), None))[0] - item[1][0], # 按净收益排序
                    reverse=True
                )

                for customer, (real_cost, scheme) in sorted_affected_customers:
                    if customer in destroy_node:
                        # 检查是否满足当前的约束条件 
                        # temp_customer_plan = {k: v for k, v in repaired_state.customer_plan.items()}
                        # temp_rm_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(vehicle_route)
                        # if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                        #     continue
                        if not is_constraints_satisfied(repaired_state, vehicle_task_data, scheme):
                            continue
                        vehicle_task_data = update_vehicle_task(vehicle_task_data, scheme, vehicle_route)
                        repaired_state.customer_plan[customer] = scheme
                        drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = scheme
                        state.destroyed_customers_info.pop(customer)
                        if drone_id not in repaired_state.uav_assignments:
                            repaired_state.uav_assignments[drone_id] = []
                        repaired_state.uav_assignments[drone_id].append(scheme)
                        if repaired_state.uav_cost is None:
                            repaired_state.uav_cost = {}
                        repaired_state.uav_cost[customer] = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        destroy_node.remove(customer)
                        num_repaired += 1
                        insert_plan.append((customer, scheme, real_cost, 'vtp_centric_batch'))

                # --- 步骤 4: 收尾 - 修复剩余客户 ---
                if destroy_node:
                    print(f"  > [VTP中心收尾]: 仍有 {len(destroy_node)} 个客户待修复，转交标准贪婪或者后悔之策略插入修复...")
                    # 通过后悔值或贪婪策略插入剩余解方案,随机选择1或者2,1为贪婪，2为后悔值方案
                    random_choice = random.randint(1, 2)
                    if random_choice == 1:
                        repaired_state, insert_plan = self.repair_regret_insertion(repaired_state,strategic_bonus=0, num_destroyed=len(destroy_node), force_vtp_mode=True)
                    else:
                        repaired_state, insert_plan = self.repair_greedy_insertion(repaired_state,strategic_bonus=0, num_destroyed=len(destroy_node), force_vtp_mode=True)
                    return repaired_state, insert_plan
            else:
                # 如果没有可评估的vtp节点，直接调用后悔值策略插入修复
                repaired_state, insert_plan = self.repair_regret_insertion(repaired_state,strategic_bonus=0, num_destroyed=len(destroy_node), force_vtp_mode=True)
                # repaired_state.repair_objective = float('inf')
                return repaired_state, insert_plan
            
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

    def _regret_evaluate_traditional_insertion(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state=None):
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
            insert_plan = {}
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
                        insert_plan[customer_node] = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                        real_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                        real_cost += calculate_customer_window_cost(insert_plan, self.vehicle, vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                        total_cost = real_cost + base_total_cost
                        # b. 【重要】在此处加入时间可行性等约束检查
                        #    您需要一个 is_time_feasible 函数来验证这个方案是否可行
                        #    plan_to_check = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                        #    if is_time_feasible(plan_to_check, vehicle_arrive_time): # 假设需要 arrive_time
                        insert_plan.pop(customer_node)
                        if real_cost is not None: # 假设 drone_insert_cost 在不可行时返回 None
                            plan = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                            
                            # eval_cost 等于 real_cost，因为传统方案没有奖励
                            options.append({
                                'customer': customer,
                                'eval_cost': real_cost, 
                                'real_cost': real_cost,
                                'total_cost': total_cost,
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
                orig_customer = best_orig_y[2]
                new_customer = best_new_y[2]
                temp_delta_cost = repaired_state.uav_cost[orig_customer]
                delta_cost = best_orig_cost + best_new_cost - temp_delta_cost  # 单纯的路线差值
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
                        temp_delta_cost = temp_customer_cost[orig_customer]
                        del temp_customer_cost[orig_customer]
                    temp_customer_cost[orig_customer] = best_orig_cost
                    temp_customer_cost[new_customer] = best_new_cost
                    temp_total_cost = sum(temp_customer_cost.values())
                    # 计算总成本（移除成本 + 插入成本）
                    temp_orig_scheme = {}
                    temp_new_scheme = {}
                    total_swap_cost = best_orig_cost + best_new_cost
                    delete_customer = orig_customer
                    delete_traditional_cost = total_cost_dict.get(delete_customer, 0.0)
                    traditional_orig_scheme = best_orig_y
                    temp_orig_scheme[orig_customer] = traditional_orig_scheme
                    traditional_new_scheme = best_new_y
                    temp_new_scheme[new_customer] = traditional_new_scheme
                    traditional_orig_win_cost = total_cost_dict.get(traditional_orig_scheme, 0.0)
                    orig_cost = calculate_customer_window_cost(temp_orig_scheme, self.vehicle, vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                    new_cost = calculate_customer_window_cost(temp_new_scheme, self.vehicle, vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                    trad_total_cost = orig_cost + new_cost
                    current_total_cost = base_total_cost - traditional_orig_win_cost + trad_total_cost + best_orig_cost + best_new_cost
                    deta_total_cost = current_total_cost - base_total_cost
                    options.append(
                        {
                        'customer': customer,
                        'orig_scheme': best_orig_y,
                        'new_scheme': best_new_y,
                        'orig_cost': best_orig_cost,
                        'new_cost': best_new_cost,
                        'eval_cost': deta_total_cost,
                        'real_cost': deta_total_cost,
                        'delta_cost': deta_total_cost,
                        'total_cost': current_total_cost,
                        'type': 'heuristic_swap', 
                        'extra_info': None,
                        'orig_plan_details': best_orig_y_cijkdu_plan, # 保留详细信息
                        'new_plan_details': best_new_y_cijkdu_plan
                    }
                    )
                    print(f"    - 找到启发式交换方案，总成本: {real_cost:.2f}")

            except Exception as e:
                print(f"  > 警告: 客户点 {customer} 启发式交换失败: {e}")

        # ----------------------------------------------------------------------
        # 3. 返回收集到的所有可行方案列表
        # ----------------------------------------------------------------------
        return options, is_heuristic_swap

    def _evaluate_traditional_insertion(self, customer, vehicle_route, vehicle_task_data, vehicle_arrive_time, base_total_cost, uav_tw_violation_cost, total_cost_dict, repaired_state=None):
        """
        评估传统插入方案的成本和方案（使用现有节点）
        包括直接插入和启发式插入两种模式
        返回: (cost, scheme) 或 None
        """
        # try:
        # 1. 首先尝试直接插入方案
        is_heuristic_swap = False
        all_insert_position = self.get_all_insert_position(vehicle_route, vehicle_task_data, customer, vehicle_arrive_time)
        insert_plan = {}
        if all_insert_position is not None:
            best_scheme = None
            min_cost = float('inf')
            for drone_id, inert_positions in all_insert_position.items():
                for inert_position in inert_positions:
                    launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id = inert_position
                    insert_plan[customer_node] = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                    # 如果发射点和回收点相同，则跳过
                    if launch_node == recovery_node:
                        continue
                    insert_cost = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                    insert_cost += calculate_customer_window_cost(insert_plan, self.vehicle, vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                    if insert_cost < min_cost:
                        min_cost = insert_cost
                        best_scheme = (drone_id, launch_node, customer_node, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                    # 处理测试插入的数据方案
                    insert_plan.pop(customer_node)
            
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
                orig_customer = best_orig_y[2]
                new_customer = best_new_y[2]
                temp_delta_cost = repaired_state.uav_cost[orig_customer]
                delta_cost = best_orig_cost + best_new_cost - temp_delta_cost
                if best_orig_y is not None and best_new_y is not None:
                    # 计算总成本（移除成本 + 插入成本）
                    temp_orig_scheme = {}
                    temp_new_scheme = {}
                    total_swap_cost = best_orig_cost + best_new_cost
                    delete_customer = orig_customer
                    delete_traditional_cost = total_cost_dict.get(delete_customer, 0.0)
                    traditional_orig_scheme = best_orig_y
                    temp_orig_scheme[orig_customer] = traditional_orig_scheme
                    traditional_new_scheme = best_new_y
                    temp_new_scheme[new_customer] = traditional_new_scheme
                    traditional_orig_win_cost = total_cost_dict.get(traditional_orig_scheme, 0.0)
                    orig_cost = calculate_customer_window_cost(temp_orig_scheme, self.vehicle, vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                    new_cost = calculate_customer_window_cost(temp_new_scheme, self.vehicle, vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                    trad_total_cost = orig_cost + new_cost
                    current_total_cost = base_total_cost - traditional_orig_win_cost + trad_total_cost + best_orig_cost + best_new_cost
                    deta_total_cost = current_total_cost - base_total_cost

                    heuristic_scheme = {
                                'customer': customer,
                                'orig_scheme': best_orig_y,
                                'new_scheme': best_new_y,
                                'orig_cost': best_orig_cost,
                                'new_cost': best_new_cost,
                                'orig_win_cost': orig_cost,# 没有考虑路线，单纯获得的惩罚成本
                                'new_win_cost': new_cost,
                                'total_cost': current_total_cost,
                                'orig_plan': best_orig_y_cijkdu_plan,
                                'new_plan': best_new_y_cijkdu_plan,
                                'win_cost': deta_total_cost,
                                'delta_cost': delta_cost,
                                'type': 'heuristic_swap'
                    }
                    # heuristic_scheme['type'] = 'heuristic_scheme'
                    return (deta_total_cost, heuristic_scheme), True
            except Exception as e:
                print(f"客户点 {customer} 启发式插入失败: {e}")
        
        # 3. 如果两种方案都失败，返回None
        # print(f"客户点 {customer} 传统插入评估失败: {e}")
        # return (None,None), False
        return (None,None),False

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
                    # temp_customer_cost = {k: v for k, v in repaired_state.uav_cost.items()}
                    # if customer in temp_customer_cost:
                    #     del temp_customer_cost[customer]
                    # temp_customer_cost[customer] = result
                    # temp_total_cost = sum(temp_customer_cost.values())
                    total_options.append({
                        'customer': customer,
                        'scheme': scheme,
                        'eval_cost': result,
                        'real_cost': result,
                        'total_cost': result,
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
            
            # 为每个可用节点计算插入成本，并选择成本最低的50个位置
            candidate_positions = self._get_best_insertion_positions(
                customer, vehicle_id, route, available_nodes, vehicle_route, vehicle_task_data, repaired_state
            )
            # 测试每个候选插入位置,测试的全局无人机的成本情况,该处只挑选了距离车辆节点近的位置进行测试，可添加无人机位置综合评估
            for node, insert_pos in candidate_positions:
                # 在此阶段中添加插入vtp节点后的时间变化产生的成本变化，插入成本变化，给出优秀得vtp插入策略方案，待修改。
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
        max_candidates = min(50, len(position_costs))
        # max_candidates = len(position_costs)
        
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
    
    def _calculate_vtp_expansion_cost(self, customer, vehicle_id, insert_pos, vehicle_route, vtp_vehicle_task_data, repaired_state, vtp_node):
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
            # 计算临时all_route的到达时间，以及其对应的总体成本
            temp_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(all_route)
            # 计算更新完时间的带惩罚成本的所有价值
            temp_total_cost, temp_uav_tw_violation_cost, temp_total_cost_dict = calculate_window_cost(
                repaired_state.customer_plan, repaired_state.uav_cost, temp_vehicle_arrive_time, 
                self.vehicle, self.customer_time_windows_h, 
                self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node
            )
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
                launch_cost,scheme = self._calculate_uav_mission_cost(drone_id, customer, vehicle_id, insert_pos, all_route, vtp_node, vtp_vehicle_task_data, repaired_state,'launch', temp_vehicle_arrive_time)
                if launch_cost is not None and launch_cost < min_uav_cost:
                    min_uav_cost = launch_cost
                    best_drone_scheme = (drone_id, customer, vehicle_id, insert_pos, 'launch')
                    best_scheme = scheme
            # 测试无人机作为回收点的成本
            for drone_id in self.V:
                recovery_cost,scheme = self._calculate_uav_mission_cost(drone_id, customer, vehicle_id, insert_pos, all_route, vtp_node, vtp_vehicle_task_data, repaired_state,'recovery', temp_vehicle_arrive_time)
                if recovery_cost is not None and recovery_cost < min_uav_cost:
                    min_uav_cost = recovery_cost
                    best_drone_scheme = (drone_id, customer, vehicle_id, insert_pos, 'recovery')
                    best_scheme = scheme
            if min_uav_cost == float('inf'):
                return float('inf'), None
        # # 3. 计算融合降落成本
        # landing_cost = self._calculate_landing_cost(customer, vehicle_id, insert_pos, route, best_drone_scheme)
        
            # 4. 总成本, 涵盖了总体的成本，包含了绕路+未被时间窗口+更新过后的传统成本
            total_cost = vehicle_cost_increase + min_uav_cost + temp_total_cost
        
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
    
    def _calculate_uav_mission_cost(self, drone_id, customer, vehicle_id, insert_pos, route, vtp_node, vtp_vehicle_task_data, repaired_state, mission_type, temp_vehicle_arrive_time):
        """
        计算无人机执行任务的成本
        支持无人机作为发射点或回收点的不同成本计算
        """
        # try:
        # 使用传入的VTP节点或生成新的节点ID
        if vtp_node is None:
            vtp_node = f"vtp_{vehicle_id}_{insert_pos}_{customer}"  # 生成唯一的VTP节点ID
        repaired_state.add_vehicle_route = route
        vtp_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.add_vehicle_route)  # 其应该与传入的temp_vehicle_arrive_time保持一致
        # 根据任务类型计算不同的成本
        if mission_type == 'launch':
            # 无人机作为发射点：从VTP节点到客户点
            cost,scheme = self._calculate_launch_mission_cost(drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id, temp_vehicle_arrive_time)
        elif mission_type == 'recovery':
            # 无人机作为回收点：从客户点到VTP节点
            cost,scheme = self._calculate_recovery_mission_cost(drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id, temp_vehicle_arrive_time)
        else:
            return None
        
        return cost,scheme
            
        # except:
        #     return None
    

    def _calculate_launch_mission_cost(self, drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id, temp_vehicle_arrive_time):
        """
        计算无人机作为发射点的成本（从VTP节点到客户点）
        参考get_all_insert_position函数的规则，考虑同车和跨车两种情况
        """
        # try:
        # 获取该客户点的最近VTP节点集合
        customer_vtp_candidates = self.map_cluster_vtp_dict[customer]
        # 得到临时路线
        temp_route = [in_route[:] for in_route in route]
        vehicle_idx = vehicle_id -1
        # 找到vtp_node在route中的索引
        route = route[vehicle_idx]
        vtp_node_idx = route.index(vtp_node)
        launch_vehicle_id = vehicle_id
        
        min_cost = float('inf')
        best_scheme = None
        temp_scheme = {}
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
            route_cost = self.drone_insert_cost(drone_id, customer, vtp_node, recovery_node)
            # 在此处计算包含时间窗口惩罚在内的成本计算
            temp_scheme[customer] = (drone_id, vtp_node, customer, recovery_node, launch_vehicle_id, launch_vehicle_id)
            win_cost = calculate_customer_window_cost(temp_scheme, self.vehicle, temp_vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
            total_cost = route_cost + win_cost
            if total_cost is not None and total_cost < min_cost:
                # min_cost = cost
                min_cost = total_cost
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
                    temp_scheme[customer] = (drone_id, vtp_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
                    win_cost = calculate_customer_window_cost(temp_scheme, self.vehicle, temp_vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                    route_cost = self.drone_insert_cost(drone_id, customer, vtp_node, recovery_node)
                    total_cost = route_cost + win_cost
                    if total_cost is not None and total_cost < min_cost:
                        # min_cost = cost
                        min_cost = total_cost
                        best_scheme = (drone_id, vtp_node, customer, recovery_node, launch_vehicle_id, recovery_vehicle_id)
        
        return min_cost if min_cost != float('inf') else None, best_scheme
        
    def _calculate_recovery_mission_cost(self, drone_id, vtp_node, customer, route, vtp_vehicle_arrive_time, vtp_vehicle_task_data, repaired_state, vehicle_id, temp_vehicle_arrive_time):
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
        temp_scheme = {}
        
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
                            temp_scheme[customer] = (drone_id, launch_node, customer, vtp_node, launch_vehicle_id, recovery_vehicle_id)
                            route_cost = self.drone_insert_cost(drone_id, customer, launch_node, vtp_node)
                            win_cost = calculate_customer_window_cost(temp_scheme, self.vehicle, temp_vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                            total_cost = route_cost + win_cost
                            if total_cost is not None and total_cost < min_cost:
                                # min_cost = cost
                                min_cost = total_cost
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
                        temp_scheme[customer] = (drone_id, launch_node, customer, vtp_node, launch_vehicle_id, recovery_vehicle_id)
                        route_cost = self.drone_insert_cost(drone_id, customer, launch_node, vtp_node)
                        win_cost = calculate_customer_window_cost(temp_scheme, self.vehicle, temp_vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                        total_cost = route_cost + win_cost
                        if total_cost is not None and total_cost < min_cost:
                            # min_cost = cost
                            min_cost = total_cost
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
                    temp_scheme[customer] = (drone_id, launch_node, customer, vtp_node, launch_vehicle_id, recovery_vehicle_id)
                    route_cost = self.drone_insert_cost(drone_id, customer, launch_node, vtp_node)
                    win_cost = calculate_customer_window_cost(temp_scheme, self.vehicle, temp_vehicle_arrive_time, self.customer_time_windows_h, self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
                    total_cost = route_cost + win_cost
                    if total_cost is not None and total_cost < min_cost:
                        # min_cost = cost
                        min_cost = total_cost
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
            # if total_positions < 5:
            #     print(f"警告：客户点 {customer} 的可行插入位置过少 ({total_positions} 个)，可能影响优化效果")

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
        win_cost = [] # 记录每代时间窗口惩罚的适应度函数变化矩阵
        uav_route_cost = [] # 记录每代无人机路线成本的适应度函数变化矩阵
        vehicle_route_cost = [] # 记录每代地面路线成本的适应度函数变化矩阵
        final_uav_cost = []  # 记录完成空中避障的适应度变化曲线
        final_total_list = []  # 记录完成空中避障的总成本变化曲线
        final_win_cost = []
        final_vehicle_route_cost = []  # 记录完成空中避障的地面路线成本变化曲线
        final_total_objective = []
        work_time = []  # 不考虑空中交通拥堵的迭代次数排序
        final_work_time = []  # 考虑空中交通拥堵的迭代次数任务排序
        current_state = initial_state.fast_copy()
        # 设置对不可行破坏或修复方案的惩罚机制
        decay_factor = 0.95

        # (你对初始解的预处理，这部分完全保留)
        # best_state.rm_empty_vehicle_route, best_state.empty_nodes_by_vehicle = best_state.update_rm_empty_task()
        current_state.rm_empty_vehicle_route, current_state.empty_nodes_by_vehicle = current_state.update_rm_empty_task()
        # current_state.rm_empty_vehicle_route = [route[:] for route in current_state.vehicle_routes]
        current_state.vehicle_routes = [route[:] for route in current_state.rm_empty_vehicle_route]
        # current_state.destroyed_node_cost = current_state.update_calculate_plan_cost(current_state.uav_cost, current_state.rm_empty_vehicle_route)
        current_state.destroyed_node_cost = current_state.win_total_objective()
        print(f"初始解总成本: {current_state.destroyed_node_cost}")
        current_state.rm_empty_vehicle_arrive_time = current_state.calculate_rm_empty_vehicle_arrive_time(current_state.rm_empty_vehicle_route)
        current_state.vehicle_arrive_time = current_state.calculate_rm_empty_vehicle_arrive_time(current_state.vehicle_routes)
        # current_state.final_uav_plan, current_state.final_uav_cost, current_state.final_vehicle_plan_time, current_state.final_vehicle_task_data, current_state.final_global_reservation_table = current_state.re_update_time(current_state.rm_empty_vehicle_route, current_state.rm_empty_vehicle_arrive_time, current_state.vehicle_task_data, current_state)
        current_state.final_uav_plan, current_state.final_uav_cost, current_state.final_vehicle_plan_time, current_state.final_vehicle_task_data, current_state.final_global_reservation_table = current_state.re_update_time(current_state.vehicle_routes, current_state.vehicle_arrive_time, current_state.vehicle_task_data, current_state)
        final_vehicle_arrive_time = extract_arrive_time_from_plan(current_state.final_vehicle_plan_time)
        final_vehicle_max_times, final_global_max_time = get_max_completion_time(final_vehicle_arrive_time)
        final_work_time.append(final_global_max_time)
        final_window_total_cost, final_uav_tw_violation_cost, final_total_cost_dict  = calculate_window_cost(current_state.customer_plan,
                    current_state.final_uav_cost,
                    final_vehicle_arrive_time,
                    self.vehicle,
                    self.customer_time_windows_h,
                    self.early_arrival_cost,
                    self.late_arrival_cost,
                    self.uav_travel,
                    self.node)
        final_total_list.append(final_window_total_cost)
        final_total_objective_value = current_state.update_calculate_plan_cost(final_total_cost_dict, current_state.vehicle_routes)
        final_total_objective.append(final_total_objective_value)
        final_vehicle_route_cost.append(final_total_objective_value - final_window_total_cost)  # 记录考虑空中避障场景下的车辆路径规划成本
        best_final_objective = final_total_objective_value
        final_current_objective = final_total_objective_value

        best_state = current_state.fast_copy()
        best_objective = current_state.destroyed_node_cost
        # current_state.vehicle_routes = [route.copy() for route in current_state.rm_empty_vehicle_route]
        current_objective = best_objective
        # 保存初始当前状态
        y_best.append(best_objective)
        y_cost.append(best_objective)
        current_window_total_cost, current_uav_tw_violation_cost, current_total_cost_dict  = calculate_window_cost(current_state.customer_plan,
                    current_state.uav_cost,
                    current_state.rm_empty_vehicle_arrive_time,
                    self.vehicle,
                    self.customer_time_windows_h,
                    self.early_arrival_cost,
                    self.late_arrival_cost,
                    self.uav_travel,
                    self.node)
        current_vehicle_max_times, current_global_max_time = get_max_completion_time(current_state.vehicle_arrive_time)
        work_time.append(current_global_max_time)
        current_total_violation_cost = sum(current_uav_tw_violation_cost.values())
        win_cost.append(current_total_violation_cost)
        current_state._total_cost = current_state.update_calculate_plan_cost(current_total_cost_dict, current_state.vehicle_routes)
        uav_route_cost.append(current_window_total_cost - current_total_violation_cost)
        vehicle_route_cost.append(current_objective - current_window_total_cost)

        final_best_objective = best_final_objective

        start_time = time.time()
        
        init_uav_cost = list(current_state.uav_cost.values())
        base_flexibility_bonus = sum(init_uav_cost) / len(init_uav_cost)
        
        # 3. 初始化模拟退火（自适应初始温度 + 自动计算降温系数）
        #    目标：初始时以 20% 概率接受典型劣解（delta ≈ 5% 成本）；
        #          到最后一轮时温度衰减至接近 0（T_final ≈ 0.01），基本不接受劣解
        initial_delta_estimate = best_objective * 0.05        # 典型成本差估计
        initial_accept_prob    = 0.20                          # 期望初始接受率
        self.temperature = max(
            -initial_delta_estimate / np.log(initial_accept_prob),
            1e-3
        )
        self.initial_temperature = self.temperature
        T_final      = 0.01
        cooling_rate = (T_final / self.temperature) ** (1.0 / max(self.max_iterations, 1))
        cooling_rate = max(0.950, min(0.9999, cooling_rate))   # 限制在合理范围
        print(f"开始ALNS求解，初始成本: {best_objective:.2f}")
        print(f"自适应初始温度: {self.temperature:.4f}, 降温系数: {cooling_rate:.6f}")
        print(f'初始服务客户节点数量为{len(current_state.customer_plan)}')

        # ========== 逐迭代权重更新（替代原分段更新）==========
        # 乘法衰减系数（修复失败 / 被拒绝时使用）
        reject_decay = 0.95

        def roulette_select(weight_dict):
            names = list(weight_dict.keys())
            w = np.array([max(0.0, float(weight_dict[n])) for n in names], dtype=float)
            s = w.sum()
            if s <= 1e-12:
                p = np.ones_like(w) / len(w)
            else:
                p = w / s
            return self.rng.choice(names, p=p)

        def update_weights_reward(d_name, r_name, score):
            """score > 0 时指数加权平均奖励；score == 0 时乘法衰减惩罚。"""
            if score > 0:
                self.destroy_weights[d_name] = max(
                    self.min_weight,
                    (1 - self.reaction_factor) * self.destroy_weights[d_name]
                    + self.reaction_factor * score
                )
                self.repair_weights[r_name] = max(
                    self.min_weight,
                    (1 - self.reaction_factor) * self.repair_weights[r_name]
                    + self.reaction_factor * score
                )
            else:
                # 拒绝：乘法惩罚，让坏组合权重缓慢下降
                self.destroy_weights[d_name] = max(
                    self.min_weight,
                    self.destroy_weights[d_name] * reject_decay
                )
                self.repair_weights[r_name] = max(
                    self.min_weight,
                    self.repair_weights[r_name] * reject_decay
                )
            # 同步镜像字典
            self.operator_weights['single_layer']['destroy'] = dict(self.destroy_weights)
            self.operator_weights['single_layer']['repair']  = dict(self.repair_weights)

        # 停滞重启参数
        no_improve_count     = 0
        no_improve_threshold = max(50, self.max_iterations // 10)

        start_time   = time.time()
        elapsed_time = 0.0
        # --------------------------------------------------------------------------
        # 阶段二：智能ALNS主循环
        # --------------------------------------------------------------------------
        for iteration in range(self.max_iterations):
            # if iteration == 3:
            #     print("中途输出当前解的详细信息：")
            elapsed_time = time.time() - start_time
            prev_state = current_state.fast_copy()

            # 3.1 选择destroy/repair（传统：单层轮盘赌）
            d_name = roulette_select(self.destroy_weights)
            r_name = roulette_select(self.repair_weights)

            destroy_op = getattr(self, d_name)
            repair_op  = getattr(self, r_name)

            # 你原算子有force_vtp_mode：baseline里用概率随机开关（可设0/1固定）
            # 高温阶段做结构性VTP探索，低温阶段转精细化客户优化
            force_vtp_mode = (self.temperature > self.initial_temperature * 0.3)

            print(f"\n--- iter {iteration} ---")
            print(f"  > ops: {d_name} + {r_name} | force_vtp_mode={force_vtp_mode}")
            print(f"  > 当前客户数: {len(current_state.customer_plan.keys())}, uav_cost数量: {len(current_state.uav_cost.keys())}")

            # 3.2 destroy + repair
            destroyed_state = destroy_op(prev_state, force_vtp_mode=force_vtp_mode)
            valid, report = self._check_and_print_state_constraints(destroyed_state, "destroyed_state")
            # if not valid:
            #     print(f"  > 破坏方案不可行，跳过此次迭代。约束报告: {report}")
            #     y_cost.append(current_objective)
            #     current_state = prev_state.fast_copy()
            #     continue

            num_destroyed = len(getattr(destroyed_state, "destroyed_customers_info", {}) or {})
            repaired_state, _ = repair_op(
            destroyed_state,
            strategic_bonus=0,              # baseline不使用战略奖励
            num_destroyed=num_destroyed,
            force_vtp_mode=force_vtp_mode
            )
            valid, report = self._check_and_print_state_constraints(repaired_state, "repaired_state")
            # if not valid:
            #     print(f"  > 修复方案不可行，跳过此次迭代。约束报告: {report}")
            #     y_cost.append(current_objective)
            #     current_state = prev_state.fast_copy()
            #     continue
            if repaired_state.repair_objective == float('inf'):
                print("  > 修复后方案为空，跳过此次迭代。")
                repaired_state.destroyed_customers_info = {}
                # 修复失败：对所使用的算子施加乘法衰减惩罚
                self.destroy_weights[d_name] = max(self.min_weight, self.destroy_weights[d_name] * reject_decay)
                self.repair_weights[r_name]  = max(self.min_weight, self.repair_weights[r_name]  * reject_decay)
                self.operator_weights['single_layer']['destroy'] = dict(self.destroy_weights)
                self.operator_weights['single_layer']['repair']  = dict(self.repair_weights)
                y_cost.append(current_objective)
                repaired_state.repair_objective = 0
                current_state = best_state.fast_copy()
                self.temperature = max(self.temperature * cooling_rate, 1e-8)
                continue
            if len(repaired_state.customer_plan) < 15:
                print(f"  > 修复后客户节点过少（{len(repaired_state.customer_plan)}），可能是过度破坏或修复失败，跳过此次迭代。")    

            # =================================================================
            # 步骤 2.3: 评估结果并为本次行动评分
            # =================================================================
            new_objective = repaired_state.win_total_objective()
            score = 0
            accepted = False
            # new_objective = repaired_state.objective()
            current_vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
            current_window_total_cost, current_uav_tw_violation_cost, current_total_cost_dict  = calculate_window_cost(repaired_state.customer_plan,
                    repaired_state.uav_cost,
                    current_vehicle_arrive_time,
                    self.vehicle,
                    self.customer_time_windows_h,
                    self.early_arrival_cost,
                    self.late_arrival_cost,
                    self.uav_travel,
                    self.node)
            current_vehicle_max_times, current_global_max_time = get_max_completion_time(current_vehicle_arrive_time)
            current_total_violation_cost = sum(current_uav_tw_violation_cost.values())
            win_cost.append(current_total_violation_cost)
            uav_route_cost.append(current_window_total_cost - current_total_violation_cost)
            vehicle_route_cost.append(new_objective - current_window_total_cost)
            work_time.append(current_global_max_time)
            # 添加更新空中无人机避障后的信息内容
            repaired_state.vehicle_arrive_time = repaired_state.calculate_rm_empty_vehicle_arrive_time(repaired_state.vehicle_routes)
            repaired_state.final_uav_plan, repaired_state.final_uav_cost, repaired_state.final_vehicle_plan_time, repaired_state.final_vehicle_task_data, repaired_state.final_global_reservation_table = repaired_state.re_update_time(repaired_state.vehicle_routes, repaired_state.vehicle_arrive_time, repaired_state.vehicle_task_data, repaired_state)
            final_vehicle_arrive_time = extract_arrive_time_from_plan(repaired_state.final_vehicle_plan_time)
            finial_window_total_cost, finial_uav_tw_violation_cost, finial_total_cost_dict  = calculate_window_cost(repaired_state.customer_plan,
            repaired_state.final_uav_cost,
            final_vehicle_arrive_time,
            self.vehicle,
            self.customer_time_windows_h,
            self.early_arrival_cost,
            self.late_arrival_cost,
            self.uav_travel,
            self.node)
            final_uav_cost.append(sum(repaired_state.final_uav_cost.values()))
            final_total_list.append(finial_window_total_cost)
            final_win_cost.append(sum(finial_uav_tw_violation_cost.values()))
            final_total_objective_value = repaired_state.update_calculate_plan_cost(finial_total_cost_dict, repaired_state.vehicle_routes)
            final_total_objective.append(final_total_objective_value)
            final_vehicle_max_times, final_global_max_time = get_max_completion_time(final_vehicle_arrive_time)
            final_work_time.append(final_global_max_time)
            final_vehicle_route_cost.append(final_total_objective_value - finial_window_total_cost)  # 记录考虑空中避障场景下的车辆路径规划成本
            final_new_objective = final_total_objective_value

            # =================================================================
            # 3.5 接受准则：SA模拟退火（四档评分 + 逐迭代权重更新）
            # =================================================================
            accepted = False
            score    = 0.0
            print(f'修复策略任务完成，当前修复后的客户点数量为: {len(repaired_state.customer_plan.keys())}')

            if new_objective < best_objective:
                accepted = True
                score    = self.sigma1
                print(f"  > 接受：新全局最优 score={score}")
            elif new_objective < current_objective:
                accepted = True
                score    = self.sigma2
                print(f"  > 接受：优于当前解 score={score}")
            elif self._simulated_annealing_accept(current_objective, new_objective, self.temperature):
                accepted = True
                score    = self.sigma3
                print(f"  > 接受：SA接受劣解 score={score}, T={self.temperature:.4f}")
            else:
                accepted = False
                score    = self.sigma4          # 0.0，触发乘法惩罚
                print(f"  > 拒绝：未通过SA score={score}, T={self.temperature:.4f}")

            # 3.6 逐迭代权重更新（奖励/惩罚）
            update_weights_reward(d_name, r_name, score)

            # 3.7 状态更新
            if accepted:
                repaired_state.destroyed_customers_info = {}
                current_state    = repaired_state.fast_copy()
                current_objective = new_objective
                if new_objective < best_objective:
                    best_state    = repaired_state.fast_copy()
                    best_objective = new_objective
                    y_best.append(best_objective)

            # 3.8 停滞重启机制
            if new_objective < best_objective:
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= no_improve_threshold:
                print(f"  > [Diversification] 连续 {no_improve_count} 轮无改善，重置到全局最优并局部升温")
                current_state    = best_state.fast_copy()
                current_objective = best_objective
                self.temperature  = min(self.temperature * 3.0, self.initial_temperature * 0.5)
                no_improve_count  = 0

            # final-best更新（保留原逻辑）
            if final_new_objective < final_best_objective:
                best_final_state = repaired_state.fast_copy()
                final_best_objective = final_new_objective
                best_final_state.final_best_objective = final_best_objective
                best_final_uav_cost = sum(repaired_state.final_uav_cost.values())
                best_final_objective = final_best_objective
                best_final_win_cost  = sum(finial_uav_tw_violation_cost.values())
                best_final_vehicle_max_times = final_vehicle_max_times
                best_final_global_max_time   = final_global_max_time
                best_total_win_cost          = finial_window_total_cost
                best_final_vehicle_route_cost = final_total_objective_value - finial_window_total_cost

            y_cost.append(current_objective)

            # 3.9 温度衰减
            self.temperature = max(self.temperature * cooling_rate, 1e-8)

            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"  > 进度: iter={iteration}, current={current_objective:.2f}, best={best_objective:.2f}, T={self.temperature:.4f}, time={elapsed_time:.2f}s")
                print(f"  > 权重 destroy: {self.destroy_weights}")
                print(f"  > 权重 repair : {self.repair_weights}")

        # ========== 4) 收尾统计与保存（保留你的接口） ==========
        elapsed_time = time.time() - start_time
        statistics = {'iterations': self.max_iterations, 'runtime': elapsed_time, 'best_objective': best_objective}

        best_arrive_time = best_state.calculate_rm_empty_vehicle_arrive_time(best_state.vehicle_routes)
        best_window_total_cost, best_uav_tw_violation_cost, best_total_cost_dict  = calculate_window_cost(best_state.customer_plan,
                          best_state.uav_cost,
                          best_state.rm_empty_vehicle_arrive_time,
                          self.vehicle,
                          self.customer_time_windows_h,
                          self.early_arrival_cost,
                          self.late_arrival_cost,
                          self.uav_travel,
                          self.node)
        # 记录完成时间
        best_vehicle_max_times, best_global_max_time = get_max_completion_time(best_arrive_time)
        best_total_uav_tw_violation_cost = sum(best_uav_tw_violation_cost.values())
        best_total_vehicle_cost = best_objective - best_window_total_cost

        # 保存运行数据
        save_alns_results(
            instance_name=self.problemName + str('DAI') + "_" + str(self.iter),  # 换成你实际的算例名
            y_best=y_best,
            y_cost=y_cost,
            win_cost=win_cost,
            uav_route_cost=uav_route_cost,
            vehicle_route_cost=vehicle_route_cost,
            strategy_weights=self.strategy_weights,
            operator_weights=self.operator_weights,
            elapsed_time=elapsed_time,
            best_objective=best_objective,
            best_vehicle_max_times=best_vehicle_max_times,
            best_global_max_time=best_global_max_time,
            best_arrive_time=best_arrive_time,
            best_window_total_cost=best_window_total_cost,
            best_uav_tw_violation_cost=best_uav_tw_violation_cost,
            best_total_cost_dict=best_total_cost_dict,
            best_state=best_state,
            # === 新增传参 ===
            best_final_uav_cost=best_final_uav_cost,
            best_final_objective=best_final_objective,
            best_final_win_cost=best_final_win_cost,
            best_total_win_cost=best_total_win_cost,
            best_final_vehicle_route_cost=best_final_vehicle_route_cost,
            final_uav_cost=final_uav_cost,
            final_total_list=final_total_list,
            final_win_cost=final_win_cost,
            final_total_objective=final_total_objective,
            final_vehicle_route_cost=final_vehicle_route_cost,
            # 新增完成时间维度参数
            best_final_vehicle_max_times=best_final_vehicle_max_times,      # 最终方案下车辆完成时间（标量）
            best_final_global_max_time=best_final_global_max_time,        # 最终方案下全局最大完成时间（标量）
            work_time=work_time,                         # 每一代当前解完成时间 list
            final_work_time=final_work_time,                   # 每一代最终方案完成时间 list
            best_final_state=best_final_state,
        )
        print(f"T_I_ALNS求解完成，最终成本: {best_objective}, 迭代次数: {iteration}, 运行时间: {elapsed_time:.2f}秒")
        return best_state, best_final_state, best_objective, best_final_objective, best_final_uav_cost, best_final_win_cost, best_total_win_cost, best_final_global_max_time, best_global_max_time, best_window_total_cost, best_total_uav_tw_violation_cost, best_total_vehicle_cost, elapsed_time, win_cost, uav_route_cost, vehicle_route_cost, final_uav_cost, final_total_list, final_win_cost, final_total_objective, y_cost, y_best, work_time, final_work_time

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
    
    def _sync_cost_with_plan(self, state, context_label=""):
        """对齐state.uav_cost与state.customer_plan，清理多余并补全缺失，输出诊断。"""
        if state is None:
            return
        if state.uav_cost is None:
            state.uav_cost = {}
        try:
            plan_keys = set(state.customer_plan.keys())
            cost_keys = set(state.uav_cost.keys())
        except Exception:
            return
        # 清理不在计划中的成本
        extra_cost = cost_keys - plan_keys
        for c in extra_cost:
            state.uav_cost.pop(c, None)
        # 补全计划中缺失的成本
        missing_cost = plan_keys - cost_keys
        for c in missing_cost:
            try:
                assignment = state.customer_plan[c]
                drone_id, launch_node, customer_node, recovery_node, _, _ = assignment
                est = self.drone_insert_cost(drone_id, customer_node, launch_node, recovery_node)
                state.uav_cost[c] = est
            except Exception:
                state.uav_cost[c] = 0

    def shaw_removal(self, state, force_vtp_mode=None):
        new_state = state.fast_copy() # 确保在副本上操作
        # 清空上一轮迭代的破坏信息，确保每次破坏都是全新的
        new_state.destroyed_customers_info = {}
        current_customers = list(new_state.customer_plan.keys())
        vehicle_task_data = new_state.vehicle_task_data
        mode = 'vtp' if force_vtp_mode else 'customer'
        
        # 论文参数设置
        phi_1 = 9.0  # 距离权重
        phi_2 = 3.0  # 时间权重
        phi_3 = 2.0  # 载重权重

        # 计算当前的车辆到达时间（用于提取时间特征 t_i, t_max）
        temp_vehicle_route = [route[:] for route in new_state.vehicle_routes]
        temp_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)

        # 获取一个默认的无人机ID，用于 VTP 模式下没有明确指派无人机时的距离计算
        # 假设 self.uav_travel 是字典格式: {uav_id: {n1: {n2: 对象}}}
        default_uav_id = list(self.uav_travel.keys())[0] if isinstance(self.uav_travel, dict) and self.uav_travel else 1

        # ----------------------------------------------------------------------
        # 1. VTP破坏模式：基于论文公式的 Shaw 破坏策略
        # ----------------------------------------------------------------------
        if mode == 'vtp':
            # 1. 收集所有活跃的VTP节点
            active_vtps = []
            for vehicle_id, route in enumerate(new_state.vehicle_routes):
                v_id = vehicle_id + 1
                for vtp_node in route[1:-1]:
                    active_vtps.append((v_id, vtp_node))
                    
            if not active_vtps:
                print("  > 警告: 没有可破坏的VTP节点。")
                return new_state

            # 2. 严格按论文逻辑：随机选择第一个 VTP 作为种子
            seed_idx = self.rng.integers(0, len(active_vtps))
            seed_key = active_vtps[seed_idx]
            seed_v_id, seed_vtp_node = seed_key
            print(f"  > 经典Shaw(VTP): 随机选中种子 VTP {seed_vtp_node} (车辆 {seed_v_id})")
            air_seed_vtp_node = self.node[seed_vtp_node].map_key
            # 3. 提取特征并计算全局最大值 (d_max, t_max, l_max)
            v_nodes = [vtp for vid, vtp in active_vtps]
            # 得到对应空中节点映射v_nodes
            air_v_nodes = [self.node[vtp_node].map_key for vtp_node in v_nodes]
            # 修复: VTP模式下获取 d_max 的正确方式
            if air_v_nodes:
                d_max = max([self.uav_travel[default_uav_id][n1][n2].totalDistance for n1 in air_v_nodes for n2 in air_v_nodes])
            else:
                d_max = 1.0
            d_max = d_max if d_max > 0 else 1.0
            
            # 提取时间 t_i 和负载 l_i
            t_dict = {}
            l_dict = {}
            for vid, vtp in active_vtps:
                t_dict[(vid, vtp)] = temp_vehicle_arrive_time[vid][vtp]
                # VTP 的负载可以视为依附于它的所有无人机任务的总重量，为简化框架兼容，这里均置为1
                l_dict[(vid, vtp)] = 1.0 
                
            t_max = max(t_dict.values()) if t_dict else 1.0
            t_max = t_max if t_max > 0 else 1.0
            l_max = max(l_dict.values()) if l_dict else 1.0
            l_max = l_max if l_max > 0 else 1.0

            # 4. 计算剩余 VTP 与种子的相似度 R(i, j)
            relatedness_scores = []
            seed_t = t_dict[seed_key]
            seed_l = l_dict[seed_key]
            
            for v_id, vtp_node in active_vtps:
                if (v_id, vtp_node) == seed_key:
                    continue
                
                # 修复: a. 距离特征调用方式
                d_ij = self.uav_travel[default_uav_id][air_seed_vtp_node][self.node[vtp_node].map_key].totalDistance
                # b. 时间特征
                t_j = t_dict[(v_id, vtp_node)]
                # c. 负载特征
                l_j = l_dict[(v_id, vtp_node)]
                
                # 论文公式 (46)
                R_ij = phi_1 * (d_ij / d_max) + phi_2 * (abs(seed_t - t_j) / t_max) + phi_3 * (abs(seed_l - l_j) / l_max)
                
                # R_ij 越小，说明越相似（距离近、时间近、载重近）
                relatedness_scores.append({'key': (v_id, vtp_node), 'score': R_ij})

            # 按相似度从小到大排序（最相似的排在前面）
            relatedness_scores.sort(key=lambda x: x['score'])
            
            # 确定移除数量
            total_available_candidates = len(relatedness_scores) + 1
            num_to_remove = min(self.vtp_destroy_quantity['shaw'], total_available_candidates) 
            
            # 构建移除列表
            vtps_to_destroy = [seed_key]
            num_neighbors_to_remove = num_to_remove - 1
            
            if num_neighbors_to_remove > 0 and relatedness_scores:
                # 论文要求：select the customer with the highest similarity
                vtps_to_destroy.extend([item['key'] for item in relatedness_scores[:num_neighbors_to_remove]])
                
            print(f"  > 经典Shaw(VTP): 计划移除集群: {vtps_to_destroy}")

            # --- 步骤 5: 执行破坏 (保留原有逻辑) ---
            destroyed_vts_info = getattr(new_state, 'destroyed_vts_info', {}) 
            destroyed_vtp_count = 0
            destroyed_customers_info = {}
            max_attempts = len(vtps_to_destroy) * 2
            attempt_count = 0
            candidate_vtps = vtps_to_destroy.copy()
            
            while destroyed_vtp_count < self.vtp_destroy_quantity['shaw'] and candidate_vtps and attempt_count < max_attempts:
                attempt_count += 1
                if not candidate_vtps:
                    break
                    
                vehicle_id, vtp_node = candidate_vtps.pop(0)
                if vtp_node not in new_state.rm_empty_vehicle_route[vehicle_id-1]:
                    continue
                
                customers_to_remove = []
                for customer, assignment in list(new_state.customer_plan.items()):
                    uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                    if (launch_node == vtp_node and launch_vehicle == vehicle_id) or \
                    (recovery_node == vtp_node and recovery_vehicle == vehicle_id):
                        customers_to_remove.append(customer)
                
                all_tasks_to_remove = []
                temp_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)
                temp_customer_plan = {k: v for k, v in new_state.customer_plan.items()}
                temp_rm_empty_vehicle_route = [route[:] for route in new_state.rm_empty_vehicle_route]
                
                for customer in customers_to_remove:
                    if customer in temp_customer_plan:
                        assignment = temp_customer_plan[customer]
                        all_tasks_to_remove.append((customer, assignment))
                        need_to_remove_tasks = find_chain_tasks(assignment, temp_customer_plan, new_state.vehicle_routes, temp_vehicle_task_data)
                        all_tasks_to_remove.extend(need_to_remove_tasks)
                
                for customer, assignment in all_tasks_to_remove:
                    if customer in temp_customer_plan:
                        temp_customer_plan.pop(customer, None)
                
                if vtp_node in temp_rm_empty_vehicle_route[vehicle_id-1]:
                    temp_rm_empty_vehicle_route[vehicle_id-1].remove(vtp_node)
                
                temp_rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_rm_empty_vehicle_route)
                
                if not is_time_feasible(temp_customer_plan, temp_rm_vehicle_arrive_time):
                    continue
                
                new_state.rm_empty_vehicle_route[vehicle_id-1].remove(vtp_node)
                destroyed_vts_info[(vehicle_id-1, vtp_node)] = True
                destroyed_vtp_count += 1
                
                for customer, assignment in all_tasks_to_remove:
                    if customer in new_state.customer_plan:
                        uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                        customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
                        destroyed_customers_info[customer] = customer_info
                        new_state.customer_plan.pop(customer, None)
                        
                        if uav_id in new_state.uav_assignments:
                            new_state.uav_assignments[uav_id] = [task for task in new_state.uav_assignments[uav_id] if task[2] != customer_node]
                        
                        if new_state.uav_cost and customer_node in new_state.uav_cost:
                            new_state.uav_cost.pop(customer_node, None)
                        
                        vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                        
                        from task_data import deep_remove_vehicle_task
                        need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, vehicle_task_data)
                        orig_vehicle_id = assignment[4]
                        for chain_customer, chain_assignment in need_to_remove_tasks:
                            if chain_customer in new_state.customer_plan:
                                chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                                chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                                destroyed_customers_info[chain_customer] = chain_customer_info
                                new_state.customer_plan.pop(chain_customer, None)
                                
                                if chain_uav_id in new_state.uav_assignments:
                                    new_state.uav_assignments[chain_uav_id] = [task for task in new_state.uav_assignments[chain_uav_id] if task[2] != chain_customer_node]
                                
                                if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                                    new_state.uav_cost.pop(chain_customer_node, None)
                                
                                vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)
            
            new_state.destroyed_vts_info = destroyed_vts_info
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            new_state.vehicle_routes = [route[:] for route in new_state.rm_empty_vehicle_route]
            new_state.rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(new_state.vehicle_routes)
            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            print("=== VTP破坏阶段完成 ===\n")

        # ----------------------------------------------------------------------
        # 2. 客户破坏模式：基于论文公式的 Shaw 破坏策略
        # ----------------------------------------------------------------------
        else:
            print("  > [破坏模式]: 经典Shaw破坏策略 (Customer)")
            if not current_customers:
                print("  > 警告: 没有已服务的客户可供破坏。")
                return new_state

            # 1. 严格按论文逻辑：随机选择第一个客户作为种子
            seed_customer = self.rng.choice(current_customers)
            print(f"  > 经典Shaw(Customer): 随机选中种子 {seed_customer}")

            # 2. 提取特征并计算全局最大值 (d_max, t_max, l_max)
            # 修复: 提取所有客户点的无人机和节点映射
            c_nodes_map = {c: new_state.customer_plan[c][2] for c in current_customers}
            
            # 修复: 计算 d_max，考虑到不同客户可能使用不同的 uav_id
            if current_customers:
                distances = []
                for c1 in current_customers:
                    u1_id = new_state.customer_plan[c1][0]
                    n1 = c_nodes_map[c1]
                    for c2 in current_customers:
                        n2 = c_nodes_map[c2]
                        distances.append(self.uav_travel[u1_id][n1][n2].totalDistance)
                d_max = max(distances) if distances else 1.0
            else:
                d_max = 1.0
                
            d_max = d_max if d_max > 0 else 1.0
            
            # 提取时间 t_i 和负载 l_i
            t_dict = {}
            l_dict = {}
            for c in current_customers:
                assignment = new_state.customer_plan[c]
                launch_node = assignment[1]
                launch_vehicle = assignment[4]
                # 近似用起飞VTP的到达时间作为特征比对标准
                t_dict[c] = temp_vehicle_arrive_time[launch_vehicle][launch_node]
                # 如果你有实际的包裹重量矩阵，可替换此处。此处向下兼容默认为1
                l_dict[c] = getattr(self, 'customer_demand', {}).get(c, 1.0) 
                
            t_max = max(t_dict.values()) if t_dict else 1.0
            t_max = t_max if t_max > 0 else 1.0
            l_max = max(l_dict.values()) if l_dict else 1.0
            l_max = l_max if l_max > 0 else 1.0

            # 3. 计算所有其他客户与种子客户的相似度 R(i, j)
            relatedness_list = []
            seed_node_id = c_nodes_map[seed_customer]
            seed_uav_id = new_state.customer_plan[seed_customer][0]  # 修复: 提取种子客户分配的无人机编号
            seed_t = t_dict[seed_customer]
            seed_l = l_dict[seed_customer]
            
            for k in current_customers:
                if k == seed_customer:
                    continue
                    
                k_node_id = c_nodes_map[k]
                
                # 修复: a. 距离特征调用方式
                d_ij = self.uav_travel[seed_uav_id][seed_node_id][k_node_id].totalDistance
                
                # b. 时间特征
                t_j = t_dict[k]
                # c. 负载特征
                l_j = l_dict[k]
                
                # 论文公式 (46)
                R_ij = phi_1 * (d_ij / d_max) + phi_2 * (abs(seed_t - t_j) / t_max) + phi_3 * (abs(seed_l - l_j) / l_max)
                
                relatedness_list.append({'key': k, 'score': R_ij})

            # 4. 按相似度从小到大排序（R越小说明特征差异越小、越相似）
            relatedness_list.sort(key=lambda x: x['score'])
            
            # 确定移除数量
            n = len(current_customers)
            num_to_remove = self.rng.integers(
                max(1, int(n * self.customer_destroy_ratio[0])),
                max(2, int(n * self.customer_destroy_ratio[1])) + 1
            )
            num_to_remove = min(num_to_remove, n)
            
            # 构建移除列表
            customers_to_destroy = [seed_customer]
            if num_to_remove > 1:
                neighbors_needed = num_to_remove - 1
                # 直接选取相似度最高（Score最小）的前 neighbors_needed 个客户
                customers_to_destroy.extend([x['key'] for x in relatedness_list[:neighbors_needed]])

            print(f"  > 计划移除 {len(customers_to_destroy)} 个高相似度客户: {customers_to_destroy}")

            # --- 步骤 5: 执行破坏 (保留原有逻辑) ---
            destroyed_customers_info = new_state.destroyed_customers_info
            vehicle_task_data = new_state.vehicle_task_data
            
            for customer in customers_to_destroy:
                if customer in new_state.customer_plan:
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
                    
                    customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0]
                    destroyed_customers_info[customer] = customer_info
                    
                    if uav_id in new_state.uav_assignments:
                        new_state.uav_assignments[uav_id] = [task for task in new_state.uav_assignments[uav_id] if task[2] != customer_node]
                    
                    if new_state.uav_cost and customer_node in new_state.uav_cost:
                        new_state.uav_cost.pop(customer_node, None)
                        
                    vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                    need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, new_state.vehicle_task_data)
                    orig_vehicle_id = assignment[4]
                    
                    for chain_customer, chain_assignment in need_to_remove_tasks:
                        if chain_customer in new_state.customer_plan:
                            chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                            chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                            destroyed_customers_info[chain_customer] = chain_customer_info
                            new_state.customer_plan.pop(chain_customer, None)
                            
                            if chain_uav_id in new_state.uav_assignments:
                                new_state.uav_assignments[chain_uav_id] = [task for task in new_state.uav_assignments[chain_uav_id] if task[2] != chain_customer_node]
                            
                            if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                                new_state.uav_cost.pop(chain_customer_node, None)
                                
                            from task_data import deep_remove_vehicle_task
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)

            new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== 破坏阶段完成 ===\n")
            
        return new_state

    
    def _check_and_print_state_constraints(self, state, stage_name):
        """
        对指定状态做无人机任务链约束检查；若违背约束，打印详细错误。
        """
        if state is None:
            print(f"[约束检查][{stage_name}] 状态为空，无法检查。")
            return False, {"errors": [{"reason": "state is None"}], "error_count": 1}

        try:
            is_valid, report = self.validate_customer_plan(
                vehicle_routes=getattr(state, "vehicle_routes", None),
                customer_plan=getattr(state, "customer_plan", None),
                base_drone_assignment=self.base_drone_assignment
            )
        except Exception as e:
            print(f"[约束检查][{stage_name}] 执行异常: {e}")
            return False, {"errors": [{"reason": str(e)}], "error_count": 1}

        if not is_valid:
            errors = report.get("errors", []) if isinstance(report, dict) else []
            error_count = report.get("error_count", len(errors)) if isinstance(report, dict) else len(errors)
            print(f"\n[约束检查][{stage_name}] 发现违背约束，数量: {error_count}")
            max_show = 50
            for idx, err in enumerate(errors[:max_show], 1):
                vehicle_id = err.get("vehicle_id")
                node_id = err.get("node_id")
                drone_id = err.get("drone_id")
                customer_id = err.get("customer_id")
                rule = err.get("rule")
                reason = err.get("reason")
                print(
                    f"  {idx}. 车辆={vehicle_id}, 节点={node_id}, 无人机={drone_id}, "
                    f"客户={customer_id}, 规则={rule}, 原因={reason}"
                )
            if error_count > max_show:
                print(f"  ... 其余 {error_count - max_show} 条未展开")
            print("")

        return is_valid, report

    def window_removal(self, state, force_vtp_mode = None):
        # 拷贝当前解
        new_state = state.fast_copy()
        # 清空上一轮迭代的破坏信息，确保每次破坏都是全新的
        new_state.destroyed_customers_info = {}
        vehicle_task_data = new_state.vehicle_task_data

        # 获取当前解中的客户点（而不是所有可能的客户点）
        current_customers = list(new_state.customer_plan.keys())
        if not current_customers:
            print("没有客户点需要移除")
            return new_state
        # 开始执行客户点层面的破坏策略 (Time Window-based removal)
        n = len(current_customers)
        num_to_remove = self.rng.integers(
            max(1, int(n * 0.2)),
            max(2, int(n * 0.3)) + 1
        )
        print(f"基于时间窗相似度破坏：目标移除 {num_to_remove} 个客户点")
        
        destroyed_customers_info = {}
        removed_count = 0
        
        if not current_customers:
            return new_state
            
        # 1. 随机选择第一个客户点作为初始“种子”
        seed_customer = self.rng.choice(current_customers)
        
        while removed_count < num_to_remove and current_customers:
            if seed_customer in new_state.customer_plan:
                assignment = new_state.customer_plan.pop(seed_customer)
                uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                
                # --- 核心修改 1：获取种子节点的时间窗左边界 ready_h ---
                # 考虑到字典的键可能是 customer_id 也可能是 physical node_id，做双重适配检查
                seed_ready_h = 0
                if seed_customer in self.customer_time_windows_h:
                    seed_ready_h = self.customer_time_windows_h[seed_customer]['ready_h']
                elif customer_node in self.customer_time_windows_h:
                    seed_ready_h = self.customer_time_windows_h[customer_node]['ready_h']
                
                # --- 开始复用你的状态清理及链式删除逻辑 ---
                customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(seed_customer, 0) if new_state.uav_cost else 0]
                destroyed_customers_info[seed_customer] = customer_info
                
                if uav_id in new_state.uav_assignments:
                    new_state.uav_assignments[uav_id] = [
                        task for task in new_state.uav_assignments[uav_id]
                        if task[2] != customer_node
                    ]
                
                if new_state.uav_cost and customer_node in new_state.uav_cost:
                    new_state.uav_cost.pop(customer_node, None)
                    
                orig_vehicle_id = assignment[4]
                vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                
                removed_count += 1
                current_customers.remove(seed_customer)
                
                # 处理链式删除的任务
                need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, new_state.vehicle_task_data)
                for chain_customer, chain_assignment in need_to_remove_tasks:
                    if chain_customer in new_state.customer_plan:
                        chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                        
                        chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                        destroyed_customers_info[chain_customer] = chain_customer_info
                        
                        new_state.customer_plan.pop(chain_customer, None)
                        
                        if chain_uav_id in new_state.uav_assignments:
                            new_state.uav_assignments[chain_uav_id] = [
                                task for task in new_state.uav_assignments[chain_uav_id]
                                if task[2] != chain_customer_node
                            ]
                        
                        if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                            new_state.uav_cost.pop(chain_customer_node, None)
                            
                        from task_data import deep_remove_vehicle_task
                        vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)
                        
                        removed_count += 1
                        if chain_customer in current_customers:
                            current_customers.remove(chain_customer)
                # --- 状态清理逻辑结束 ---

                # --- 核心修改 2：基于左时间窗的绝对差值寻找下一个种子 ---
                if removed_count < num_to_remove and current_customers:
                    best_similarity_customer = None
                    min_time_diff = float('inf')
                    
                    for candidate in current_customers:
                        # 提取候选节点的 customer_node (索引2)
                        candidate_node = new_state.customer_plan[candidate][2]
                        candidate_ready_h = 0
                        
                        # 获取候选节点的 ready_h
                        if candidate in self.customer_time_windows_h:
                            candidate_ready_h = self.customer_time_windows_h[candidate]['ready_h']
                        elif candidate_node in self.customer_time_windows_h:
                            candidate_ready_h = self.customer_time_windows_h[candidate_node]['ready_h']
                        
                        # 论文逻辑：左时间窗的绝对差值即为相似度（越小说明越相似）
                        time_diff = abs(candidate_ready_h - seed_ready_h)
                        
                        # 更新具有最高相似度（差值最小）的客户
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_similarity_customer = candidate
                    
                    # 将找到的时间窗最相似的客户设为下一轮要删除的种子
                    seed_customer = best_similarity_customer
                    
            else:
                # 容错：如果当前的 seed_customer 异常（例如在前面的链式联动中被删掉了），则重新随机选一个
                if current_customers:
                    seed_customer = self.rng.choice(current_customers)

        # 3. 更新空跑节点等全局状态
        new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
        new_state.destroyed_customers_info = destroyed_customers_info
        new_state.vehicle_task_data = vehicle_task_data
        
        print(f"破坏后剩余客户点: {len(new_state.customer_plan)} (实际移除: {removed_count})")
        print("=== 破坏阶段完成 ===\n")
        
        return new_state

    def proximit_removal(self, state, force_vtp_mode = None):
        # 拷贝当前解
        new_state = state.fast_copy()
        # 清空上一轮迭代的破坏信息，确保每次破坏都是全新的
        new_state.destroyed_customers_info = {}
        # 获取当前状态的数据
        vehicle_route = new_state.vehicle_routes
        vehicle_task_data = new_state.vehicle_task_data
        # 获取当前解中的客户点（而不是所有可能的客户点）
        current_customers = list(new_state.customer_plan.keys())
        if not current_customers:
            print("没有客户点需要移除")
            return new_state

        # 开始执行客户点层面的破坏策略 (Proximity-based removal)
        n = len(current_customers)
        num_to_remove = self.rng.integers(
            max(1, int(n * 0.2)),
            max(2, int(n * 0.3)) + 1
        )
        print(f"基于邻近度破坏：目标移除 {num_to_remove} 个客户点")
        
        destroyed_customers_info = {}
        removed_count = 0
        
        # 1. 随机选择第一个客户点作为初始“种子”
        if not current_customers:
            return new_state
            
        seed_customer = self.rng.choice(current_customers)
        
        while removed_count < num_to_remove and current_customers:
            if seed_customer in new_state.customer_plan:
                assignment = new_state.customer_plan.pop(seed_customer)
                uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = assignment
                
                # 记录种子节点的物理位置，用于下一次计算邻近度
                seed_customer_node = customer_node
                
                # --- 开始复用你的状态清理逻辑 ---
                customer_info = [uav_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle, new_state.uav_cost.get(seed_customer, 0) if new_state.uav_cost else 0]
                destroyed_customers_info[seed_customer] = customer_info
                
                if uav_id in new_state.uav_assignments:
                    new_state.uav_assignments[uav_id] = [
                        task for task in new_state.uav_assignments[uav_id]
                        if task[2] != customer_node
                    ]
                
                if new_state.uav_cost and customer_node in new_state.uav_cost:
                    new_state.uav_cost.pop(customer_node, None)
                    
                orig_vehicle_id = assignment[4]
                vehicle_task_data = remove_vehicle_task(vehicle_task_data, assignment, new_state.vehicle_routes)
                
                removed_count += 1
                current_customers.remove(seed_customer)
                
                # 处理链式删除的任务
                need_to_remove_tasks = find_chain_tasks(assignment, new_state.customer_plan, new_state.vehicle_routes, new_state.vehicle_task_data)
                for chain_customer, chain_assignment in need_to_remove_tasks:
                    if chain_customer in new_state.customer_plan:
                        chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle = chain_assignment
                        
                        chain_customer_info = [chain_uav_id, chain_launch_node, chain_customer_node, chain_recovery_node, chain_launch_vehicle, chain_recovery_vehicle, new_state.uav_cost.get(chain_customer, 0) if new_state.uav_cost else 0]
                        destroyed_customers_info[chain_customer] = chain_customer_info
                        
                        new_state.customer_plan.pop(chain_customer, None)
                        
                        if chain_uav_id in new_state.uav_assignments:
                            new_state.uav_assignments[chain_uav_id] = [
                                task for task in new_state.uav_assignments[chain_uav_id]
                                if task[2] != chain_customer_node
                            ]
                        
                        if new_state.uav_cost and chain_customer_node in new_state.uav_cost:
                            new_state.uav_cost.pop(chain_customer_node, None)
                            
                        from task_data import deep_remove_vehicle_task
                        vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)
                        
                        # 更新计数和候选池
                        removed_count += 1
                        if chain_customer in current_customers:
                            current_customers.remove(chain_customer)
                # --- 状态清理逻辑结束 ---

                # 2. 如果还没达到目标数量，寻找与上一个被删节点最接近的客户作为下一个“种子”
                if removed_count < num_to_remove and current_customers:
                    best_proximity_customer = None
                    min_distance = float('inf')
                    
                    for candidate in current_customers:
                        # 核心修改点：同时获取候选节点的物理节点ID和其被分配的无人机代号
                        candidate_uav_id = new_state.customer_plan[candidate][0]
                        candidate_node = new_state.customer_plan[candidate][2]
                        
                        # 修复：调用时加入无人机代号层级，并访问 .totalDistance
                        dist = self.uav_travel[candidate_uav_id][seed_customer_node][candidate_node].totalDistance
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_proximity_customer = candidate
                    
                    # 将找到的距离最近的客户设为下一轮要删除的种子
                    seed_customer = best_proximity_customer
                    
            else:
                # 容错：如果 seed_customer 异常（例如在前面的链式删除中被意外移除），则重新随机选一个
                if current_customers:
                    seed_customer = self.rng.choice(current_customers)

        # 3. 更新空跑节点等状态
        new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
        new_state.destroyed_customers_info = destroyed_customers_info
        new_state.vehicle_task_data = vehicle_task_data
        
        print(f"破坏后剩余客户点: {len(new_state.customer_plan)} (实际移除: {removed_count})")
        print("=== 破坏阶段完成 ===\n")
        
        return new_state

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
        # 清空上一轮迭代的破坏信息，确保每次破坏都是全新的
        new_state.destroyed_customers_info = {}
        # 获取当前解中的客户点（而不是所有可能的客户点）
        current_customers = list(new_state.customer_plan.keys())
        if not current_customers:
            print("没有客户点需要移除")
            return new_state

        # new_state.vehicle_routes = new_state.rm_empty_vehicle_route  # 更新路径
        mode = 'vtp' if force_vtp_mode else 'customer'
        # mode = 'customer'
        # print(f"  > [破坏模式]: 随机破坏 ({'VTP模式' if mode == 'vtp' else '客户模式'})")
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
                    # print(f"VTP节点 {vtp_node} 删除后不满足时间约束，跳过删除 (尝试 {attempt_count}/{max_attempts})")
                    continue
                
                # 6. 约束满足，执行实际删除操作
                # print(f"成功破坏VTP节点: 车辆{vehicle_id}的节点{vtp_node} (进度: {destroyed_vtp_count + 1}/{num_to_remove})")
                
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
                        orig_vehicle_id = assignment[4]
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
                                
                                # print(f"VTP链式删除客户点 {chain_customer}")
                                vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)
            
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
                    orig_vehicle_id = assignment[4]
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
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)

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
        # 清空上一轮迭代的破坏信息，确保每次破坏都是全新的
        new_state.destroyed_customers_info = {}
        current_customers = list(new_state.customer_plan.keys())
        vehicle_task_data = new_state.vehicle_task_data
        mode = 'vtp' if force_vtp_mode else 'customer'
        # print(f"  > [破坏模式]: 综合最差破坏 ({'VTP模式' if mode == 'vtp' else '客户模式'})")
        mode = 'customer'
        # mode = 'vtp'

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
            temp_vehicle_route = [route[:] for route in new_state.vehicle_routes]
            temp_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
            window_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(new_state.customer_plan, new_state.uav_cost, temp_vehicle_arrive_time, self.vehicle, self.customer_time_windows_h, 
            self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)

            # 寻找最忙车辆上表现最差的VTP作为种子
            vtp_performance = defaultdict(float)
            for customer, assignment in new_state.customer_plan.items():
                uav_id, launch_node, _, recovery_node, launch_veh, recovery_veh = assignment
                    
                # 只关注最忙车辆上的 VTP
                if launch_veh != v_max_id or recovery_veh != v_max_id:
                    continue
                    
                task_cost = total_cost_dict.get(customer, 0.0)
                    
                # 简单按 50/50 分摊，或者您可以复用之前的按距离比例分摊逻辑
                # 这里为了计算速度，假设两端对延误的贡献相当
                if launch_veh == v_max_id:
                    vtp_performance[(launch_veh, launch_node)] += task_cost * 1.0
                # if recovery_veh == v_max_id:
                #     vtp_performance[(recovery_veh, recovery_node)] += task_cost * 0.3

            # 2. 选择“种子 VTP” (Seed)
            #    优先选择最忙车辆上，归因成本最高的 VTP (即瓶颈节点)
            v_max_route_nodes = new_state.vehicle_routes[v_max_id - 1][1:-1]
            if not v_max_route_nodes:
                print("  > 警告: 最忙车辆没有可破坏的VTP节点。")
                return new_state
            # 构建权重
            weights = []
            candidates = []
            for node in v_max_route_nodes:
                key = (v_max_id, node)
                # 成本越高，权重越大 (平方放大差异)
                w = vtp_performance.get(key, 0) ** 2 
                weights.append(w + 1.0) # +1 防止为0
                candidates.append(node)
            probs = np.array(weights) / sum(weights)
            seed_vtp_node = self.rng.choice(candidates, p=probs)
            seed_key = (v_max_id, seed_vtp_node)  # 得到种子节点
            
            seed_node_obj = self.node[seed_vtp_node]
            seed_pos = (seed_node_obj.latDeg, seed_node_obj.lonDeg)  # 得到种子节点坐标
            
            print(f"  > Shaw重平衡(VTP): 种子 VTP {seed_vtp_node} (车辆 {v_max_id}), 归因惩罚: {vtp_performance.get(seed_key, 0):.2f}")
            # 预计算最大值
            max_dist_seed = 0
            max_dist_idle = 0
            max_badness = 0 # 记录最大的归因成本
            relatedness_scores = []
            # 收集所有其他 VTP
            all_other_vtps = []
            for v_idx, route in enumerate(new_state.vehicle_routes):
                vid = v_idx + 1
                for node in route[1:-1]:
                    if (vid, node) != seed_key:
                        all_other_vtps.append((vid, node))
                        
            if not all_other_vtps: return new_state

            # --- [Step 1] 第一次遍历：收集原始数据并统计最大值 ---
            max_dist_seed = 0.0
            max_dist_idle = 0.0
            max_badness = 0.0
            
            temp_data = []
            
            for v_id, vtp_node in all_other_vtps:
                node_obj = self.node[vtp_node]
                pos = (node_obj.latDeg, node_obj.lonDeg)
                
                # a. 与种子的地理距离
                d_seed = math.sqrt((pos[0] - seed_pos[0])**2 + (pos[1] - seed_pos[1])**2)
                
                # b. 与最闲车辆路线的最短距离
                d_idle = float('inf')
                if not v_min_positions:
                    d_idle = 0.0
                else:
                    for idle_pos in v_min_positions:
                        d_tmp = math.sqrt((pos[0] - idle_pos[0])**2 + (pos[1] - idle_pos[1])**2)
                        if d_tmp < d_idle: d_idle = d_tmp
                
                # c. 自身的“差劲”程度
                badness = vtp_performance.get((v_id, vtp_node), 0.0)
                
                # 更新全局最大值
                if d_seed > max_dist_seed: max_dist_seed = d_seed
                if d_idle > max_dist_idle: max_dist_idle = d_idle
                if badness > max_badness: max_badness = badness
                
                # 暂存原始数据
                temp_data.append({
                    'key': (v_id, vtp_node), 
                    'd_seed': d_seed, 
                    'd_idle': d_idle, 
                    'badness': badness
                })
            # --- [Step 2] 第二次遍历：使用全局最大值进行归一化和打分 ---
            w_seed = 0.4  # 聚类性权重
            w_idle = 0.4  # 转移倾向权重
            w_bad = 0.2   # 效率倾向权重
            epsilon = 1e-6 # 防止除以零
        
            for item in temp_data:
                # 现在 max_... 已经是全局最大值了，归一化是公平的
                norm_d_seed = item['d_seed'] / (max_dist_seed + epsilon)
                norm_d_idle = item['d_idle'] / (max_dist_idle + epsilon)
                norm_bad = item['badness'] / (max_badness + epsilon)
                
                # 公式：距离越近(小) + 离闲车越近(小) + 越差劲(大 -> 1-x 小)
                # 目标是分数越低越好
                score = w_seed * norm_d_seed + w_idle * norm_d_idle + w_bad * (1.0 - norm_bad)
                
                relatedness_scores.append({'key': item['key'], 'score': score})

            # 4. Top-K 选择并执行破坏
            relatedness_scores.sort(key=lambda x: x['score'])
            total_available_candidates = len(relatedness_scores) + 1
            num_to_remove = min(self.vtp_destroy_quantity['shaw'], total_available_candidates) 
            
            # 3. 构建移除列表
            # 先加入种子
            vtps_to_destroy = [seed_key]
            
            # 计算还需要移除多少个邻居
            num_neighbors_to_remove = num_to_remove - 1
            
            # 如果还需要移除邻居，且有邻居可选
            if num_neighbors_to_remove > 0 and relatedness_scores:
                # --- Top-K 随机选择逻辑 ---
                
                # 设定 K 值：我们从前 K 个最相关的邻居中进行选择
                # K 应该大于等于我们要移除的数量，以提供选择空间
                K = max(num_neighbors_to_remove + 2, 5) 
                
                # 截取前 K 个候选者
                top_k_candidates = relatedness_scores[:K]
                
                # 如果候选者数量刚好等于或少于我们要移除的数量，直接全选
                if len(top_k_candidates) <= num_neighbors_to_remove:
                    vtps_to_destroy.extend([item['key'] for item in top_k_candidates])
                else:
                    # 否则，进行带权随机选择
                    # 排名越靠前（索引越小），权重越大
                    weights = np.arange(len(top_k_candidates), 0, -1)
                    weight_sum = np.sum(weights)
                    probs = weights / weight_sum if weight_sum > 0 else None
                    
                    # 无放回抽取索引
                    chosen_indices = self.rng.choice(
                        len(top_k_candidates), 
                        size=num_neighbors_to_remove, 
                        p=probs, 
                        replace=False
                    )
                    
                    # 添加选中的邻居
                    vtps_to_destroy.extend([top_k_candidates[i]['key'] for i in chosen_indices])
            print(f"  > Shaw重平衡(VTP): 计划移除集群: {vtps_to_destroy}")

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
                    
                vehicle_id, vtp_node = candidate_vtps.pop(0)  # 按优先级顺序选择
                # vehicle_id = vehicle_index + 1
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
                        orig_vehicle_id = assignment[4]
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
                                
                                # print(f"VTP链式删除客户点 {chain_customer}")
                                vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)
            
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
            print("  > [破坏模式]: 综合shaw破坏策略")
        
            # 3.1 收集所有已服务客户
            if not current_customers:
                print("  > 警告: 没有已服务的客户可供破坏。")
                return new_state
                
            # 1. 预计算：获取包含时间窗惩罚的成本信息
            try:
                temp_vehicle_route = [route[:] for route in new_state.vehicle_routes]
                temp_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
                _, _, total_cost_dict = calculate_window_cost(new_state.customer_plan, new_state.uav_cost, temp_vehicle_arrive_time, self.vehicle, self.customer_time_windows_h, 
                self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
            except Exception:
                total_cost_dict = new_state.uav_cost # 退化

            current_customers = list(new_state.customer_plan.keys())
            if not current_customers: return new_state

            # ------------------------------------------------------------------
            # 2. 选择种子客户 (基于惩罚成本的带权随机)
            #    逻辑：惩罚越高，越是“病灶中心”，越该被选中
            # ------------------------------------------------------------------
            weights = []
            for k in current_customers:
                # cost 越高，权重越大 (使用平方放大差异)
                w = total_cost_dict.get(k, 0) ** 2
                weights.append(w + 1.0)
                
            probs = np.array(weights) / sum(weights)
            seed_customer = self.rng.choice(current_customers, p=probs)
            
            # 获取种子信息
            seed_node = self.node[seed_customer]
            seed_pos = (seed_node.latDeg, seed_node.lonDeg)
            seed_tw = self.customer_time_windows_h.get(seed_customer)
            # 解析时间窗 (start, end)
            s_start = float(seed_tw['ready_h']) if seed_tw else 0.0
            s_end = float(seed_tw['due_h']) if seed_tw else 24.0
            
            print(f"  > 高冲突Shaw破坏: 种子 {seed_customer}, 惩罚成本: {total_cost_dict.get(seed_customer,0):.2f}")

            # ------------------------------------------------------------------
            # 3. 计算相关性分数 (Score)
            #    我们要找：离得近 + 时间重叠大 + 自己也烂 的邻居
            # ------------------------------------------------------------------
            relatedness_list = []
            all_other = [c for c in current_customers if c != seed_customer]
            
            # 预计算用于归一化的最大值
            max_dist = 0.0
            max_penalty = 0.0
            
            temp_data = []
            
            for k in all_other:
                # a. 空间距离
                k_node = self.node[k]
                d = math.sqrt((k_node.latDeg - seed_pos[0])**2 + (k_node.lonDeg - seed_pos[1])**2)
                max_dist = max(max_dist, d)
                
                # b. 惩罚成本
                p = total_cost_dict.get(k, 0)
                max_penalty = max(max_penalty, p)
                
                # c. 时间重叠度 (Time Window Overlap)
                #    Overlap = max(0, min(End1, End2) - max(Start1, Start2))
                k_tw = self.customer_time_windows_h.get(k)
                k_start = float(k_tw['ready_h']) if k_tw else 0.0
                k_end = float(k_tw['due_h']) if k_tw else 24.0
                
                overlap = max(0.0, min(s_end, k_end) - max(s_start, k_start))
                # 归一化重叠：重叠比例 (相对于较短的那个时间窗)
                # 这样可以避免因时间窗本身很长而导致的虚假高重叠
                min_len = min(s_end - s_start, k_end - k_start) + 0.01
                norm_overlap = overlap / min_len # 范围 [0, 1]，1表示完全包含或完全重合
                
                temp_data.append({'k': k, 'd': d, 'p': p, 'o': norm_overlap})
                
            # 计算分数 (越低越相关)
            epsilon = 1e-6
            w_dist = 0.4    # 空间越近越好
            w_overlap = 0.4 # 时间重叠越大越好 (重叠大 -> 冲突大 -> 一起移走)
            w_penalty = 0.2 # 惩罚越高越好
            
            for item in temp_data:
                n_dist = item['d'] / (max_dist + epsilon)
                n_penalty = item['p'] / (max_penalty + epsilon)
                
                # Score = 距离(正相关) - 重叠(负相关) - 惩罚(负相关)
                # 我们希望：距离小，重叠大(1.0)，惩罚大(1.0)
                # score = w_dist * n_dist + w_overlap * (1 - item['o']) + w_penalty * (1 - n_penalty)
                
                # 或者更直接的写法：
                score = w_dist * n_dist - w_overlap * item['o'] - w_penalty * n_penalty
                
                relatedness_list.append({'key': item['k'], 'score': score})

            # ------------------------------------------------------------------
            # 4. Top-K 选择并执行
            # ------------------------------------------------------------------
            # 按分数升序排序 (越小越好)
            relatedness_list.sort(key=lambda x: x['score'])
            
            # 确定移除数量
            n = len(current_customers)
            num_to_remove = self.rng.integers(
                max(1, int(n * self.customer_destroy_ratio[0])),
                max(2, int(n * self.customer_destroy_ratio[1])) + 1
            )
            num_to_remove = min(num_to_remove, n)
            
            # Top-K 随机选择
            customers_to_destroy = [seed_customer]
            if num_to_remove > 1:
                neighbors_needed = num_to_remove - 1
                # 候选池大小
                K_pool = max(neighbors_needed + 5, 10)
                top_candidates = relatedness_list[:K_pool]
                
                # 简单截取或带权随机 (这里简单截取即可，因为分数本身已经包含了复杂的权衡)
                customers_to_destroy.extend([x['key'] for x in top_candidates[:neighbors_needed]])

            print(f"  > 计划移除 {len(customers_to_destroy)} 个高冲突聚类客户: {customers_to_destroy}")

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
                    orig_vehicle_id = assignment[4]
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
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)
                            if 12 not in vehicle_task_data[1][144].drone_list or 12 not in vehicle_task_data[1][142].drone_list:
                                print(f'12 not in vehicle_task_data[1][144].drone_list or 12 not in vehicle_task_data[1][142].drone_list')

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
        # mode = 'customer'
        print(f"  > [破坏模式]: 最差破坏 ({'VTP模式' if mode == 'vtp' else '客户模式'})")
        vehicle_task_data = new_state.vehicle_task_data
        if mode == 'vtp':
            # 收集所有活跃的VTP节点并计算成本效益比
            active_vtps_with_cost_ratio = []
            destroyed_vts_info = {}

            # 获得带违背时间窗口的信息内容
            temp_vehicle_route = [route[:] for route in new_state.vehicle_routes]
            temp_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
            window_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(new_state.customer_plan, new_state.uav_cost, temp_vehicle_arrive_time, self.vehicle, self.customer_time_windows_h, 
            self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
        
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
                            # 得到vtp节点发射产生的延迟违背时间窗成本
                            uav_violate_cost = uav_tw_violation_cost.get(customer, 0) if uav_tw_violation_cost else 0
                            total_cost = total_cost + uav_violate_cost
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
                        orig_vehicle_id = assignment[4]
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
                                
                                # print(f"VTP链式删除客户点 {chain_customer}")
                                vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)
            
            print(f"VTP最差破坏策略完成：成功破坏 {destroyed_vtp_count}/{len(vtps_to_destroy)} 个VTP节点")
            
            # 更新状态
            new_state.destroyed_vts_info = destroyed_vts_info
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            # 更新空跑节点等状态
            new_state.vehicle_routes = [route.copy() for route in new_state.rm_empty_vehicle_route]  # vtp节点被破坏后重更新
            new_state.rm_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(new_state.vehicle_routes)
            # new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            new_state.destroyed_node_cost = new_state.win_total_objective()
            print(f"破坏后剩余VTP节点: {sum(len(route) - 2 for route in new_state.vehicle_routes)}")  # 减去起点和终点
            # print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== VTP破坏阶段完成 ===\n")
            print(f"破坏后总成本: {new_state.destroyed_node_cost}")
        else:
            # 开始执行客户点层面的破坏策略
            # 1. 计算每个客户点的成本
            # customer_costs = []
            # for customer in current_customers:
            #     # 从uav_cost中获取该客户点的成本
            #     cost = new_state.uav_cost.get(customer, 0) if new_state.uav_cost else 0
            #     customer_costs.append((customer, cost))
            # 获得带违背时间窗口的信息内容
            temp_vehicle_route = [route[:] for route in new_state.vehicle_routes]
            temp_vehicle_arrive_time = new_state.calculate_rm_empty_vehicle_arrive_time(temp_vehicle_route)
            window_total_cost, uav_tw_violation_cost, total_cost_dict = calculate_window_cost(new_state.customer_plan, new_state.uav_cost, temp_vehicle_arrive_time, self.vehicle, self.customer_time_windows_h, 
            self.early_arrival_cost, self.late_arrival_cost, self.uav_travel, self.node)
            customer_costs = []
            for customer in current_customers:
                # 从uav_cost中获取该客户点的成本
                cost = total_cost_dict.get(customer, 0) if total_cost_dict else 0
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
                    orig_vehicle_id = assignment[4]
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
                            vehicle_task_data = deep_remove_vehicle_task(vehicle_task_data, chain_assignment, new_state.vehicle_routes, orig_vehicle_id)

            # 5. 更新空跑节点等状态
            # new_state.destroyed_node_cost = new_state.update_calculate_plan_cost(new_state.uav_cost, new_state.vehicle_routes)
            new_state.destroyed_node_cost = new_state.win_total_objective()
            # 将破坏的客户节点信息存储到状态中，供修复阶段使用
            new_state.destroyed_customers_info = destroyed_customers_info
            new_state.vehicle_task_data = vehicle_task_data
            print(f"破坏后剩余客户点: {len(new_state.customer_plan)}")
            print("=== 破坏阶段完成 ===\n")
            print(f"破坏后总成本: {new_state.destroyed_node_cost}")
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
                        
                        # print(f"链式删除客户点 {chain_customer}")

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
                            
                            # print(f"VTP破坏：链式删除客户点 {chain_customer}")
                    
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

    def validate_customer_plan(self, vehicle_routes, customer_plan, base_drone_assignment=None):
        """
        Validate customer plan consistency.

        Returns:
            tuple[bool, dict]:
                - is_valid
                - error_report (contains vehicle/node/drone/customer details)
        """
        if base_drone_assignment is None:
            base_drone_assignment = self.base_drone_assignment

        return validate_customer_plan_constraints(
            vehicle_routes=vehicle_routes,
            customer_plan=customer_plan,
            base_drone_assignment=base_drone_assignment,
            verbose=False,
        )
    # def validate_customer_plan(self, vehicle_routes, customer_plan, base_drone_assignment):
    #     """
    #     验证给定的 customer_plan 是否符合无人机的顺序和状态约束 (增强版)。
    #     模拟无人机在车辆路线上的状态变化，检测多种逻辑冲突并清晰报告。

    #     Args:
    #         vehicle_routes (list or dict): 车辆路线列表/字典。
    #         customer_plan (dict): {customer: (drone_id, ln, cn, rn, lv, rv)}
    #         base_drone_assignment (dict): {vehicle_id: [drone_id1, drone_id2, ...]}

    #     Returns:
    #         bool: 如果计划有效则返回 True，否则返回 False 并打印详细错误信息。
    #     """
    #     is_valid = True # 初始假设计划有效

    #     # ----------------------------------------------------------------------
    #     # 1. 初始化无人机状态 (使用深拷贝以隔离验证过程)
    #     # ----------------------------------------------------------------------
    #     # drone_state: 记录每个无人机的详细状态
    #     # 'location': vehicle_id (在车上) 或 'flying' (飞行中) 或 'depot' (初始在仓库)
    #     # 'last_event_node': (vehicle_id, node_id) 上次发生事件的节点
    #     # 'current_task': customer_id 正在执行的任务 (None 如果不在执行任务)
    #     drone_state = {}
    #     all_drones = set(d for drones in base_drone_assignment.values() for d in drones)
        
    #     # 尝试从 customer_plan 中也获取无人机，以防 base_assignment 不全
    #     try:
    #         drones_in_plan = set(assignment[0] for assignment in customer_plan.values())
    #         all_drones.update(drones_in_plan)
    #     except (TypeError, IndexError):
    #         print("  > 警告: customer_plan 格式可能不完全正确，无法提取所有无人机ID。")
            
    #     for drone_id in all_drones:
    #         drone_state[drone_id] = {'location': 'depot', 'last_event_node': None, 'current_task': None}

    #     for vehicle_id, drones_on_vehicle in base_drone_assignment.items():
    #         for drone_id in drones_on_vehicle:
    #             if drone_id in drone_state:
    #                 drone_state[drone_id]['location'] = vehicle_id # 初始在对应的车上
    #             else:
    #                 print(f"  > 警告: 基础分配中的无人机 {drone_id} 未在状态字典中初始化。")

    #     # ----------------------------------------------------------------------
    #     # 2. 构建任务查找表 (按节点组织)
    #     # ----------------------------------------------------------------------
    #     launch_tasks_at_node = {}    # {(vehicle_id, vtp_node): [(drone_id, customer, assignment_tuple), ...]}
    #     recovery_tasks_at_node = {} # {(vehicle_id, vtp_node): [(drone_id, customer, assignment_tuple), ...]}

    #     for customer, assignment in customer_plan.items():
    #         try:
    #             # 检查 assignment 结构是否有效
    #             if len(assignment) != 6:
    #                 raise ValueError("Assignment tuple length mismatch")
    #             drone_id, ln, _, rn, lv, rv = assignment
                
    #             # 检查无人机ID是否存在
    #             if drone_id not in drone_state:
    #                 print(f"  > 错误: 客户 {customer} 的任务引用了未知的无人机 ID: {drone_id}。")
    #                 is_valid = False
    #                 continue # 跳过这个无效任务

    #             launch_key = (lv, ln)
    #             if launch_key not in launch_tasks_at_node: launch_tasks_at_node[launch_key] = []
    #             launch_tasks_at_node[launch_key].append((drone_id, customer, assignment))
                
    #             recovery_key = (rv, rn)
    #             if recovery_key not in recovery_tasks_at_node: recovery_tasks_at_node[recovery_key] = []
    #             recovery_tasks_at_node[recovery_key].append((drone_id, customer, assignment))
    #         except (TypeError, ValueError, IndexError) as e:
    #             print(f"  > 错误: customer_plan 中客户 {customer} 的任务数据格式无效: {assignment}。错误: {e}")
    #             is_valid = False
    #             # return False # 可以选择提前退出

    #     if not is_valid: return False

    #     # ----------------------------------------------------------------------
    #     # 3. 模拟车辆行驶并验证无人机状态变化
    #     #    【重要】: 此模拟基于节点顺序，不考虑精确时间，检查的是逻辑顺序冲突。
    #     # ----------------------------------------------------------------------
    #     # 处理 vehicle_routes 是列表还是字典
    #     processed_routes = []
    #     # ... (与上一版本相同的代码，将 routes 转换为 [(vid, route_list), ...]) ...

    #     for vehicle_id, route in processed_routes:
    #         if len(route) < 2: continue
    #         print(f"\n--- 正在验证车辆 {vehicle_id} 的路线: {route} ---")
            
    #         # 遍历路线中的每个 VTP 节点 (跳过起点和终点 Depot)
    #         for node_idx in range(1, len(route) - 1):
    #             vtp_node = route[node_idx]
    #             node_key = (vehicle_id, vtp_node)
                
    #             print(f"  节点 {vtp_node} (索引 {node_idx}):")

    #             # --- 3.1 处理在该节点的【回收】任务 (必须先于发射处理) ---
    #             if node_key in recovery_tasks_at_node:
    #                 for drone_id, customer, assignment in recovery_tasks_at_node[node_key]:
    #                     print(f"    - 检查回收: 无人机 {drone_id} (来自客户 {customer})")
    #                     state = drone_state[drone_id]
                        
    #                     # 【验证规则 1】: 无人机必须处于飞行状态 ('flying')
    #                     if state['location'] != 'flying':
    #                         error_msg = (f"    -> !! 回收冲突 !! 无人机 {drone_id} 试图在节点 {vtp_node} (车辆 {vehicle_id}) 回收，"
    #                                     f"但其当前状态是 '{state['location']}' (应为 'flying')。")
    #                         if state['last_event_node']:
    #                             error_msg += f" 上次事件发生在 {state['last_event_node']}."
    #                         print(error_msg)
    #                         is_valid = False
                        
    #                     # 【验证规则 2】: 回收的任务必须是当前正在执行的任务
    #                     elif state['current_task'] != customer:
    #                         print(f"    -> !! 任务不匹配 !! 无人机 {drone_id} 试图回收服务客户 {customer} 的任务，"
    #                             f"但记录显示它正在执行的任务是 {state['current_task']}。")
    #                         is_valid = False
                        
    #                     else:
    #                         # 更新状态：无人机现在在这辆车上
    #                         state['location'] = vehicle_id
    #                         state['last_event_node'] = node_key
    #                         state['current_task'] = None # 任务完成
    #                         print(f"      状态更新: 无人机 {drone_id} 已回收至车辆 {vehicle_id}。")

    #             # --- 3.2 处理在该节点的【发射】任务 ---
    #             if node_key in launch_tasks_at_node:
    #                 for drone_id, customer, assignment in launch_tasks_at_node[node_key]:
    #                     print(f"    - 检查发射: 无人机 {drone_id} (飞往客户 {customer})")
    #                     state = drone_state[drone_id]

    #                     # 【验证规则 3】: 无人机必须在当前车辆上才能被发射
    #                     if state['location'] != vehicle_id:
    #                         error_msg = (f"    -> !! 发射冲突 !! 无人机 {drone_id} 试图从节点 {vtp_node} (车辆 {vehicle_id}) 发射，"
    #                                     f"但其当前状态是 '{state['location']}' (应在车辆 {vehicle_id} 上)。")
    #                         if state['last_event_node']:
    #                             error_msg += f" 上次事件发生在 {state['last_event_node']}."
    #                         print(error_msg)
    #                         is_valid = False
                        
    #                     # 【验证规则 4】: 无人机不能已经在执行任务（即上次发射后未回收）
    #                     elif state['current_task'] is not None:
    #                         print(f"    -> !! 状态冲突 !! 无人机 {drone_id} 试图发射新任务 (客户 {customer})，"
    #                             f"但它仍在执行上一个任务 (客户 {state['current_task']})。")
    #                         is_valid = False

    #                     else:
    #                         # 更新状态：无人机现在处于飞行状态，并记录当前任务
    #                         state['location'] = 'flying'
    #                         state['last_event_node'] = node_key
    #                         state['current_task'] = customer
    #                         print(f"      状态更新: 无人机 {drone_id} 已发射，状态为 'flying'，目标客户 {customer}。")

    #     # ----------------------------------------------------------------------
    #     # 4. 最终全局检查：所有任务是否都已完成？
    #     # ----------------------------------------------------------------------
    #     unfinished_drones = []
    #     for drone_id, state in drone_state.items():
    #         if state['location'] == 'flying' or state['current_task'] is not None:
    #             unfinished_drones.append((drone_id, state['current_task']))

    #     if unfinished_drones:
    #         print(f"\n  > 警告: 验证结束时，以下无人机仍处于飞行状态或有未完成的任务:")
    #         for d_id, c_id in unfinished_drones:
    #             print(f"    - 无人机 {d_id} (目标客户: {c_id})")
    #         # is_valid = False # 取决于您的业务规则是否允许任务不闭环

    #     # ----------------------------------------------------------------------
    #     # 5. 返回最终验证结果
    #     # ----------------------------------------------------------------------
    #     if is_valid:
    #         print("\n=== customer_plan 约束验证通过 ===")
    #     else:
    #         print("\n=== customer_plan 存在约束冲突 ===")
            
    #     return is_valid




# --- 核心：定义概率选择函数 ---
def weighted_choice_sub(candidates, k_limit):
    """
    从候选列表中，基于排名权重选择一个方案。
    排名越靠前（成本越低），权重越大。
    """
    if not candidates:
        return None, []
        
    # 1. 截断：只看前 K 个
    limit = min(len(candidates), k_limit)
    pool = candidates[:limit]
    backup = candidates[limit:] # 备选池
    
    # 2. 计算权重：使用简单的线性排名权重或指数权重
    # 方案 A (线性): 排名第1权重为K, 第2为K-1...
    # weights = [limit - i for i in range(limit)]
    
    # 方案 B (指数 - 推荐): 强化头部效应，比如 [1.0, 0.5, 0.25...]
    # 这样能保证大概率选最优，小概率选次优，非常靠谱
    weights = [math.exp(-0.5 * i) for i in range(limit)]
    
    # 3. 归一化并选择
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    
    # 按概率随机选择一个索引
    r = random.random()
    cumulative_p = 0.0
    selected_index = 0
    for i, p in enumerate(probs):
        cumulative_p += p
        if r <= cumulative_p:
            selected_index = i
            break
    
    # 4. 构建尝试队列
    # 队列顺序：[被选中的那个] + [RCS里剩下的(按原序)] + [备选池]
    # 这样如果"幸运儿"失败了，我们立刻回退到最稳妥的贪婪顺序
    
    chosen_one = pool[selected_index]
    
    # 构建剩余的 RCS 成员 (排除被选中的)
    remaining_pool = [c for i, c in enumerate(pool) if i != selected_index]
    
    # 最终执行队列
    execution_queue = [chosen_one] + remaining_pool + backup
    
    return execution_queue
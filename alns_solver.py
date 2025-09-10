#!/usr/bin/env python
"""
mFSTSP问题的ALNS求解框架
"""

import copy
import numpy as np
import numpy.random as rnd
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from alns import ALNS
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.select import RouletteWheel, RandomSelect
from alns.stop import MaxRuntime, MaxIterations

from cost_y import calculate_plan_cost
from task_data import *
from call_function import *
from initialize import *
from cbs_plan import *
from rm_node_sort_node import rm_empty_node


class MfstspState:
    """
    mFSTSP问题的解状态类
    包含：车辆路径、无人机任务分配、时间安排等
    """
    
    def __init__(self, vehicle_routes, uav_assignments, customer_plan, 
                 vehicle_task_data, global_reservation_table, total_cost=None):
        self.vehicle_routes = vehicle_routes          # 车辆路径 {vehicle_id: [node_ids]}
        self.uav_assignments = uav_assignments        # 无人机任务分配 {uav_id: [(launch_node, customer, recovery_node, launch_veh, recovery_veh)]}
        self.customer_plan = customer_plan            # 客户服务计划 {customer_id: assignment_tuple}
        self.vehicle_task_data = vehicle_task_data    # 车辆任务数据
        self.global_reservation_table = global_reservation_table  # 全局预留表
        self._total_cost = total_cost                 # 缓存的总成本
        
    def objective(self) -> float:
        """
        目标函数：计算总成本
        包括：车辆行驶成本 + 无人机飞行成本 + 时间成本
        """
        if self._total_cost is not None:
            return self._total_cost
            
        # 计算车辆成本
        vehicle_cost = 0
        for vehicle_id, route in self.vehicle_routes.items():
            if len(route) > 1:
                for i in range(len(route) - 1):
                    node_i, node_j = route[i], route[i+1]
                    # 这里需要根据您的距离矩阵计算成本
                    # vehicle_cost += distance_matrix[node_i][node_j]
                    vehicle_cost += 1  # 简化处理
                    
        # 计算无人机成本
        uav_cost = 0
        for uav_id, assignments in self.uav_assignments.items():
            for assignment in assignments:
                # 计算单个无人机任务的成本
                uav_cost += self._calculate_single_uav_cost(uav_id, assignment)
                
        self._total_cost = vehicle_cost + uav_cost
        return self._total_cost
    
    def _calculate_single_uav_cost(self, uav_id, assignment):
        """计算单个无人机任务的成本"""
        # 简化处理，实际应该根据飞行距离、时间等计算
        return 10.0
    
    def copy(self):
        """深拷贝当前解状态"""
        return MfstspState(
            copy.deepcopy(self.vehicle_routes),
            copy.deepcopy(self.uav_assignments),
            copy.deepcopy(self.customer_plan),
            copy.deepcopy(self.vehicle_task_data),
            copy.deepcopy(self.global_reservation_table),
            self._total_cost
        )
    
    def get_context(self):
        """返回上下文向量，用于上下文感知的选择器"""
        # 返回一个简单的特征向量
        num_vehicles = len(self.vehicle_routes)
        num_uavs = len(self.uav_assignments)
        num_customers = len(self.customer_plan)
        total_cost = self.objective()
        
        return [num_vehicles, num_uavs, num_customers, total_cost]


class MfstspALNS:
    """mFSTSP问题的ALNS求解器"""
    
    def __init__(self, node, vehicle, uav_travel, veh_distance, veh_travel, 
                 G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types,
                 V, T, DEPOT_nodeID, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, A_c, xeee):
        """
        初始化ALNS求解器
        
        Args:
            node: 节点信息
            vehicle: 车辆信息
            uav_travel: 无人机旅行时间矩阵
            veh_distance: 车辆距离矩阵
            veh_travel: 车辆旅行时间矩阵
            G_air: 空中网络图
            G_ground: 地面网络图
            air_matrix: 空中距离矩阵
            ground_matrix: 地面距离矩阵
            air_node_types: 空中节点类型
            ground_node_types: 地面节点类型
            V: 无人机ID列表
            T: 车辆ID列表
            DEPOT_nodeID: 仓库节点ID
            N: VTP节点列表
            N_zero: 包含仓库的VTP节点列表
            N_plus: 包含仓库的VTP节点列表（反向）
            A_total: 所有无人机节点
            A_cvtp: 除中继点外的无人机节点
            A_vtp: VTP节点
            A_aerial_relay_node: 空中中继节点
            A_c: 客户节点
            xeee: 无人机续航时间
        """
        self.node = node
        self.vehicle = vehicle
        self.uav_travel = uav_travel
        self.veh_distance = veh_distance
        self.veh_travel = veh_travel
        self.G_air = G_air
        self.G_ground = G_ground
        self.air_matrix = air_matrix
        self.ground_matrix = ground_matrix
        self.air_node_types = air_node_types
        self.ground_node_types = ground_node_types
        self.V = V
        self.T = T
        self.DEPOT_nodeID = DEPOT_nodeID
        self.N = N
        self.N_zero = N_zero
        self.N_plus = N_plus
        self.A_total = A_total
        self.A_cvtp = A_cvtp
        self.A_vtp = A_vtp
        self.A_aerial_relay_node = A_aerial_relay_node
        self.A_c = A_c
        self.xeee = xeee
        
        # 初始化ALNS
        self.alns = ALNS(rnd.default_rng(seed=42))
        
        # 添加破坏算子
        self.alns.add_destroy_operator(self._random_customer_removal)
        self.alns.add_destroy_operator(self._worst_customer_removal)
        self.alns.add_destroy_operator(self._vehicle_route_removal)
        self.alns.add_destroy_operator(self._uav_assignment_removal)
        
        # 添加修复算子
        self.alns.add_repair_operator(self._greedy_customer_insertion)
        self.alns.add_repair_operator(self._vehicle_route_repair)
        self.alns.add_repair_operator(self._uav_assignment_repair)
        self.alns.add_repair_operator(self._local_search_repair)
        
        # 配置选择策略
        self.select = RouletteWheel([3, 2, 1, 0.5], 0.8, 3, 1)
        
        # 配置接受准则
        self.accept = SimulatedAnnealing(initial_temperature=100, cooling_rate=0.95)
        
        # 配置停止条件
        self.stop = MaxRuntime(300)  # 5分钟
    
    def solve(self, initial_solution: MfstspState, max_runtime=300):
        """
        使用ALNS求解mFSTSP问题
        
        Args:
            initial_solution: 初始解
            max_runtime: 最大运行时间（秒）
            
        Returns:
            tuple: (best_solution, best_objective, result)
        """
        # 设置停止条件
        self.stop = MaxRuntime(max_runtime)
        
        # 运行ALNS
        result = self.alns.iterate(initial_solution, self.select, self.accept, self.stop)
        
        return result.best_state, result.best_objective, result
    
    # ==================== 破坏算子 ====================
    
    def _random_customer_removal(self, current: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        随机移除客户点
        """
        destroyed = current.copy()
        
        # 获取所有客户点
        all_customers = list(destroyed.customer_plan.keys())
        if not all_customers:
            return destroyed
            
        # 随机选择要移除的客户点数量（20%-40%）
        num_to_remove = rng.integers(
            max(1, len(all_customers) // 5), 
            max(2, len(all_customers) // 3)
        )
        
        # 随机选择要移除的客户点
        customers_to_remove = rng.choice(all_customers, num_to_remove, replace=False)
        
        # 移除选中的客户点
        for customer in customers_to_remove:
            if customer in destroyed.customer_plan:
                assignment = destroyed.customer_plan.pop(customer)
                
                # 从无人机分配中移除相关任务
                uav_id, _, _, _, _ = assignment
                if uav_id in destroyed.uav_assignments:
                    destroyed.uav_assignments[uav_id] = [
                        task for task in destroyed.uav_assignments[uav_id]
                        if task[1] != customer  # 移除包含该客户的任务
                    ]
        
        # 重置成本缓存
        destroyed._total_cost = None
        return destroyed
    
    def _worst_customer_removal(self, current: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        移除成本最高的客户点
        """
        destroyed = current.copy()
        
        # 计算每个客户点的成本（简化处理）
        customer_costs = {}
        for customer, assignment in destroyed.customer_plan.items():
            # 这里应该根据实际成本计算，简化处理
            customer_costs[customer] = rng.random() * 100
        
        # 按成本排序，移除成本最高的20%-30%
        sorted_customers = sorted(customer_costs.items(), key=lambda x: x[1], reverse=True)
        num_to_remove = max(1, len(sorted_customers) // 4)
        
        for i in range(num_to_remove):
            customer = sorted_customers[i][0]
            assignment = destroyed.customer_plan.pop(customer)
            
            # 从无人机分配中移除
            uav_id, _, _, _, _ = assignment
            if uav_id in destroyed.uav_assignments:
                destroyed.uav_assignments[uav_id] = [
                    task for task in destroyed.uav_assignments[uav_id]
                    if task[1] != customer
                ]
        
        destroyed._total_cost = None
        return destroyed
    
    def _vehicle_route_removal(self, current: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        移除部分车辆路径
        """
        destroyed = current.copy()
        
        # 随机选择车辆
        vehicles = list(destroyed.vehicle_routes.keys())
        if not vehicles:
            return destroyed
            
        num_vehicles_to_affect = rng.integers(1, min(3, len(vehicles)))
        selected_vehicles = rng.choice(vehicles, num_vehicles_to_affect, replace=False)
        
        for vehicle_id in selected_vehicles:
            route = destroyed.vehicle_routes[vehicle_id]
            
            # 随机移除路径中的部分节点（保留起点和终点）
            if len(route) > 2:
                # 随机选择要保留的节点数量
                keep_count = rng.integers(2, len(route))
                indices_to_keep = rng.choice(len(route), keep_count, replace=False)
                indices_to_keep = sorted(indices_to_keep)
                
                # 重建路径
                new_route = [route[i] for i in indices_to_keep]
                destroyed.vehicle_routes[vehicle_id] = new_route
        
        destroyed._total_cost = None
        return destroyed
    
    def _uav_assignment_removal(self, current: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        移除部分无人机任务分配
        """
        destroyed = current.copy()
        
        # 随机选择无人机
        uavs = list(destroyed.uav_assignments.keys())
        if not uavs:
            return destroyed
            
        num_uavs_to_affect = rng.integers(1, min(3, len(uavs)))
        selected_uavs = rng.choice(uavs, num_uavs_to_affect, replace=False)
        
        for uav_id in selected_uavs:
            assignments = destroyed.uav_assignments[uav_id]
            if len(assignments) > 1:
                # 随机移除部分任务
                num_to_remove = rng.integers(1, len(assignments))
                indices_to_remove = rng.choice(len(assignments), num_to_remove, replace=False)
                
                # 移除选中的任务
                new_assignments = [assignments[i] for i in range(len(assignments)) if i not in indices_to_remove]
                destroyed.uav_assignments[uav_id] = new_assignments
                
                # 同时从客户计划中移除
                for idx in indices_to_remove:
                    assignment = assignments[idx]
                    customer = assignment[1]
                    if customer in destroyed.customer_plan:
                        destroyed.customer_plan.pop(customer)
        
        destroyed._total_cost = None
        return destroyed
    
    # ==================== 修复算子 ====================
    
    def _greedy_customer_insertion(self, destroyed: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        贪婪插入未分配的客户点
        """
        repaired = destroyed.copy()
        
        # 找出未分配的客户点（这里需要根据实际情况获取）
        # 简化处理：假设所有客户点都已分配
        unassigned_customers = []
        
        for customer in unassigned_customers:
            # 找到最佳的无人机和车辆组合
            best_assignment = self._find_best_assignment(customer, repaired)
            if best_assignment:
                uav_id, launch_node, recovery_node, launch_veh, recovery_veh = best_assignment
                
                # 添加到解中
                repaired.customer_plan[customer] = best_assignment
                if uav_id not in repaired.uav_assignments:
                    repaired.uav_assignments[uav_id] = []
                repaired.uav_assignments[uav_id].append(best_assignment)
        
        repaired._total_cost = None
        return repaired
    
    def _vehicle_route_repair(self, destroyed: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        修复车辆路径
        """
        repaired = destroyed.copy()
        
        for vehicle_id, route in repaired.vehicle_routes.items():
            # 检查路径的完整性
            if not self._is_route_feasible(vehicle_id, route):
                # 使用启发式方法修复路径
                new_route = self._repair_vehicle_route(vehicle_id, route)
                repaired.vehicle_routes[vehicle_id] = new_route
        
        repaired._total_cost = None
        return repaired
    
    def _uav_assignment_repair(self, destroyed: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        修复无人机任务分配
        """
        repaired = destroyed.copy()
        
        # 检查每个无人机任务的可行性
        for uav_id, assignments in repaired.uav_assignments.items():
            valid_assignments = []
            for assignment in assignments:
                if self._is_assignment_feasible(uav_id, assignment):
                    valid_assignments.append(assignment)
                else:
                    # 尝试修复不可行的分配
                    fixed_assignment = self._fix_assignment(uav_id, assignment)
                    if fixed_assignment:
                        valid_assignments.append(fixed_assignment)
            
            repaired.uav_assignments[uav_id] = valid_assignments
        
        repaired._total_cost = None
        return repaired
    
    def _local_search_repair(self, destroyed: MfstspState, rng: rnd.Generator) -> MfstspState:
        """
        使用局部搜索修复解
        """
        repaired = destroyed.copy()
        
        # 执行2-opt局部搜索
        for vehicle_id, route in repaired.vehicle_routes.items():
            if len(route) > 3:
                improved_route = self._two_opt_search(route)
                repaired.vehicle_routes[vehicle_id] = improved_route
        
        repaired._total_cost = None
        return repaired
    
    # ==================== 辅助方法 ====================
    
    def _find_best_assignment(self, customer, state):
        """找到客户的最佳分配方案"""
        # 简化实现，实际应该考虑所有可能的无人机和车辆组合
        return None
    
    def _is_route_feasible(self, vehicle_id, route):
        """检查车辆路径是否可行"""
        return len(route) >= 2
    
    def _repair_vehicle_route(self, vehicle_id, route):
        """修复车辆路径"""
        # 简化实现，实际应该使用更复杂的修复策略
        if len(route) < 2:
            return [self.DEPOT_nodeID, self.DEPOT_nodeID]
        return route
    
    def _is_assignment_feasible(self, uav_id, assignment):
        """检查无人机分配是否可行"""
        # 简化实现，实际应该检查各种约束
        return True
    
    def _fix_assignment(self, uav_id, assignment):
        """修复不可行的无人机分配"""
        # 简化实现，实际应该使用更复杂的修复策略
        return assignment
    
    def _two_opt_search(self, route):
        """2-opt局部搜索"""
        # 简化实现，实际应该实现完整的2-opt算法
        return route


def create_initial_state(init_total_cost, init_uav_plan, init_customer_plan, 
                        init_time_uav_task_dict, init_vehicle_route, 
                        init_vehicle_plan_time, init_vehicle_task_data, 
                        init_global_reservation_table):
    """
    从初始解创建MfstspState对象
    
    Args:
        init_total_cost: 初始总成本
        init_uav_plan: 初始无人机计划
        init_customer_plan: 初始客户计划
        init_time_uav_task_dict: 初始无人机任务字典
        init_vehicle_route: 初始车辆路线
        init_vehicle_plan_time: 初始车辆计划时间
        init_vehicle_task_data: 初始车辆任务数据
        init_global_reservation_table: 初始全局预留表
        
    Returns:
        MfstspState: 初始解状态
    """
    # 转换车辆路线格式
    vehicle_routes = {}
    for i, route in enumerate(init_vehicle_route):
        vehicle_id = i + 1
        vehicle_routes[vehicle_id] = route
    
    return MfstspState(
        vehicle_routes=vehicle_routes,
        uav_assignments=init_time_uav_task_dict,
        customer_plan=init_customer_plan,
        vehicle_task_data=init_vehicle_task_data,
        global_reservation_table=init_global_reservation_table,
        total_cost=init_total_cost
    )


def solve_with_alns(node, vehicle, air_matrix, ground_matrix, air_node_types, ground_node_types,
                   numUAVs, numTrucks, uav_travel, veh_travel, veh_distance, G_air, G_ground,
                   V, T, DEPOT_nodeID, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, A_c, xeee,
                   initial_solution, max_runtime=300):
    """
    使用ALNS求解mFSTSP问题的主函数
    
    Args:
        node: 节点信息
        vehicle: 车辆信息
        air_matrix: 空中距离矩阵
        ground_matrix: 地面距离矩阵
        air_node_types: 空中节点类型
        ground_node_types: 地面节点类型
        numUAVs: 无人机数量
        numTrucks: 车辆数量
        uav_travel: 无人机旅行时间矩阵
        veh_travel: 车辆旅行时间矩阵
        veh_distance: 车辆距离矩阵
        G_air: 空中网络图
        G_ground: 地面网络图
        V: 无人机ID列表
        T: 车辆ID列表
        DEPOT_nodeID: 仓库节点ID
        N: VTP节点列表
        N_zero: 包含仓库的VTP节点列表
        N_plus: 包含仓库的VTP节点列表（反向）
        A_total: 所有无人机节点
        A_cvtp: 除中继点外的无人机节点
        A_vtp: VTP节点
        A_aerial_relay_node: 空中中继节点
        A_c: 客户节点
        xeee: 无人机续航时间
        initial_solution: 初始解
        max_runtime: 最大运行时间（秒）
        
    Returns:
        tuple: (best_solution, best_objective, result)
    """
    # 创建ALNS求解器
    alns_solver = MfstspALNS(
        node, vehicle, uav_travel, veh_distance, veh_travel,
        G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types,
        V, T, DEPOT_nodeID, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, A_c, xeee
    )
    
    # 使用ALNS求解
    best_solution, best_objective, result = alns_solver.solve(initial_solution, max_runtime)
    
    return best_solution, best_objective, result 
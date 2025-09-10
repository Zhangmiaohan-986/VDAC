#!/usr/bin/env python
"""
mFSTSP问题的简化ALNS求解框架
"""

import copy
import numpy as np
import numpy.random as rnd
from collections import defaultdict
import time


class SimpleMfstspState:
    """
    简化的mFSTSP解状态类
    """
    
    def __init__(self, vehicle_routes, uav_assignments, customer_plan, 
                 vehicle_task_data, global_reservation_table, total_cost=None):
        self.vehicle_routes = vehicle_routes
        self.uav_assignments = uav_assignments
        self.customer_plan = customer_plan
        self.vehicle_task_data = vehicle_task_data
        self.global_reservation_table = global_reservation_table
        self._total_cost = total_cost
        
    def objective(self):
        """目标函数：计算总成本"""
        if self._total_cost is not None:
            return self._total_cost
        
        # 简化成本计算
        vehicle_cost = len(self.vehicle_routes) * 10
        uav_cost = sum(len(assignments) for assignments in self.uav_assignments.values()) * 5
        self._total_cost = vehicle_cost + uav_cost
        return self._total_cost
    
    def copy(self):
        """深拷贝当前解状态"""
        return SimpleMfstspState(
            copy.deepcopy(self.vehicle_routes),
            copy.deepcopy(self.uav_assignments),
            copy.deepcopy(self.customer_plan),
            copy.deepcopy(self.vehicle_task_data),
            copy.deepcopy(self.global_reservation_table),
            self._total_cost
        )


class SimpleALNS:
    """简化的ALNS求解器"""
    
    def __init__(self, max_iterations=1000, max_runtime=60):
        self.max_iterations = max_iterations
        self.max_runtime = max_runtime
        self.rng = rnd.default_rng(42)
        
    def solve(self, initial_state):
        """
        使用简化的ALNS算法求解
        
        Args:
            initial_state: 初始解状态
            
        Returns:
            tuple: (best_solution, best_objective, statistics)
        """
        current_state = initial_state.copy()
        best_state = current_state.copy()
        best_objective = best_state.objective()
        
        start_time = time.time()
        iteration = 0
        
        print(f"开始ALNS求解，初始成本: {best_objective}")
        
        while iteration < self.max_iterations and (time.time() - start_time) < self.max_runtime:
            # 破坏阶段
            destroyed_state = self._destroy(current_state)
            
            # 修复阶段
            repaired_state = self._repair(destroyed_state)
            
            # 接受准则（爬山法）
            if repaired_state.objective() < current_state.objective():
                current_state = repaired_state
                
                # 更新最优解
                if current_state.objective() < best_objective:
                    best_state = current_state.copy()
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
        
        print(f"ALNS求解完成，最终成本: {best_objective}, 迭代次数: {iteration}, 运行时间: {elapsed_time:.2f}秒")
        
        return best_state, best_objective, statistics
    
    def _destroy(self, state):
        """破坏算子：随机移除部分客户点"""
        destroyed = state.copy()
        
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
                
                # 从无人机分配中移除相关任务
                uav_id, _, _, _, _ = assignment
                if uav_id in destroyed.uav_assignments:
                    destroyed.uav_assignments[uav_id] = [
                        task for task in destroyed.uav_assignments[uav_id]
                        if task[1] != customer
                    ]
        
        destroyed._total_cost = None
        return destroyed
    
    def _repair(self, destroyed_state):
        """修复算子：贪婪重新分配客户点"""
        repaired = destroyed_state.copy()
        
        # 这里简化处理，实际应该实现更复杂的修复策略
        # 对于被移除的客户点，可以重新分配到最佳的无人机和车辆组合
        
        repaired._total_cost = None
        return repaired


def create_simple_initial_state(init_total_cost, init_uav_plan, init_customer_plan, 
                               init_time_uav_task_dict, init_vehicle_route, 
                               init_vehicle_plan_time, init_vehicle_task_data, 
                               init_global_reservation_table):
    """
    从初始解创建SimpleMfstspState对象
    """
    # 转换车辆路线格式
    vehicle_routes = {}
    for i, route in enumerate(init_vehicle_route):
        vehicle_id = i + 1
        vehicle_routes[vehicle_id] = route
    
    return SimpleMfstspState(
        vehicle_routes=vehicle_routes,
        uav_assignments=init_time_uav_task_dict,
        customer_plan=init_customer_plan,
        vehicle_task_data=init_vehicle_task_data,
        global_reservation_table=init_global_reservation_table,
        total_cost=init_total_cost
    )


def solve_with_simple_alns(initial_solution, max_iterations=1000, max_runtime=60):
    """
    使用简化ALNS求解mFSTSP问题
    
    Args:
        initial_solution: 初始解
        max_iterations: 最大迭代次数
        max_runtime: 最大运行时间（秒）
        
    Returns:
        tuple: (best_solution, best_objective, statistics)
    """
    # 创建ALNS求解器
    alns_solver = SimpleALNS(max_iterations=max_iterations, max_runtime=max_runtime)
    
    # 使用ALNS求解
    best_solution, best_objective, statistics = alns_solver.solve(initial_solution)
    
    return best_solution, best_objective, statistics 
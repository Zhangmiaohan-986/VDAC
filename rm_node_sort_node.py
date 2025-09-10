from collections import defaultdict
from typing import List, Dict, Tuple
import copy
from collections import defaultdict
from task_data import *
from cost_y import *
from call_function import *
from initialize import *
from cbs_plan import *
# from insert_plan import *
from down_data import *

def rm_empty_node(customer_plan, vehicle_route):
    """
    移除车辆路线中不存在无人机任务的节点
    
    Args:
        customer_plan: 客户计划字典 {mission_tuple: assignment_info}
        vehicle_route: 车辆路线列表 [vehicle_route1, vehicle_route2, ...]
    
    Returns:
        tuple: (filtered_vehicle_route, empty_nodes_by_vehicle)
            - filtered_vehicle_route: 过滤后的车辆路线
            - empty_nodes_by_vehicle: 按车辆ID组织的空节点列表
    """
    # 初始化结果数据结构
    filtered_vehicle_route = []
    empty_nodes_by_vehicle = defaultdict(list)
    
    # 遍历每个车辆的路线
    for num_index, route in enumerate(vehicle_route):
        vehicle_id = num_index + 1

        # 获取该车辆路线中的节点（不包括起点和终点）
        task_route = route[1:-1] if len(route) > 2 else []
        
        # 收集该车辆上所有无人机任务的发射和回收节点
        mission_nodes = set()
        
        # 遍历所有无人机任务
        for mission_tuple in customer_plan.values():
            drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle = mission_tuple
            
            # 检查发射车辆和回收车辆是否相同，且都等于当前车辆
            if launch_vehicle == vehicle_id or recovery_vehicle == vehicle_id:
                # 如果发射节点在车辆路线上，则保留
                if launch_node in task_route:
                    mission_nodes.add(launch_node)
                
                # 如果回收节点在车辆路线上，则保留
                if recovery_node in task_route:
                    mission_nodes.add(recovery_node)
        
        # 找出不存在无人机任务的节点
        empty_nodes = [node for node in task_route if node not in mission_nodes]
        empty_nodes_by_vehicle[vehicle_id] = empty_nodes
        
        # 构建过滤后的车辆路线
        # 保留起点
        filtered_route = [route[0]]
        
        # 只保留有无人机任务的节点
        for node in task_route:
            if node in mission_nodes:
                filtered_route.append(node)
        
        # 保留终点
        if len(route) > 1:
            filtered_route.append(route[-1])
        
        filtered_vehicle_route.append(filtered_route)
    
    return filtered_vehicle_route, dict(empty_nodes_by_vehicle)
            

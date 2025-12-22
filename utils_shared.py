
def find_keys_and_indices(dictionary, target_value):
    """
    查找字典中值对应的键及其索引位置。
    
    参数:
    dictionary (dict): 要查找的字典
    target_value: 要查找的值
    
    返回:
    list: 包含元组 (键, 索引) 的列表，按键在字典中出现的顺序排列
    """
    result = []
    for index, (key, value) in enumerate(dictionary.items()):
        if value == target_value:
            result.append((key, index, value))
    return result

def find_same_xy_different_z(positions_dict, target_position):
    """
    在positions_dict中找到与target_position具有相同x和y坐标，但z坐标不同的项。
    
    参数:
    positions_dict (dict): 一个键从0到n的字典，每个值是一个(x,y,z)坐标。
    target_position (tuple): 目标坐标(x,y,z)。
    
    返回:
    tuple: (key, key, position)，包含找到项的键和坐标。
    如果没有找到匹配的项，返回None。
    """
    target_x, target_y, target_z = target_position
    
    for key, position in positions_dict.items():
        # 跳过相同的位置
        if position == target_position:
            continue
        
        x, y, z = position
        # 检查xy是否相同且z不同
        if x == target_x and y == target_y and z != target_z:
            # 返回匹配的项(键，键，坐标)格式
            return (key, key, position)
    
    # 如果没有找到匹配项
    return None



def merge_and_renumber_dicts(air_node_types, ground_node_types):
    """
    将两个字典合并并重新编号，键从0开始递增。
    先处理 air_node_types，再处理 ground_node_types，保持顺序。
    
    参数:
    air_node_types (dict): 空中节点类型字典
    ground_node_types (dict): 地面节点类型字典
    
    返回:
    dict: 合并后重新编号的字典
    """
    merged = {}
    current_index = 0
    
    # 处理 air_node_types，保持原始顺序
    for key in sorted(air_node_types.keys()):
        merged[current_index] = air_node_types[key]
        current_index += 1
    
    # 处理 ground_node_types，保持原始顺序
    for key in sorted(ground_node_types.keys()):
        merged[current_index] = ground_node_types[key]
        current_index += 1
    
    return merged

def find_chain_tasks(assignment, customer_plan, vehicle_routes, vehicle_task_data):
    """
    删除某个任务后，沿无人机链找到后续需要删除的任务：
    - 一直删到无人机“回到原始发射车辆”之后的第一次再次发射之前（不包含那次发射）
      也就是：回到原车后，看到 93（2@143->3@130）就停，不把 93/96 加入删除列表。
    """
    drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle = assignment

    # 同车任务不触发链删
    if launch_vehicle == recovery_vehicle:
        return []

    origin_vehicle = launch_vehicle

    # 快速查找：通过 (drone_id, launch_vehicle, launch_node) 找到对应客户任务
    launch_map = {}
    for cid, a in customer_plan.items():
        d, ln, _, rn, lv, rv = a
        launch_map[(d, lv, ln)] = (cid, a)

    # 从“被删任务”的回收车&回收点开始向后追
    cur_vehicle = recovery_vehicle
    cur_route = vehicle_routes[cur_vehicle - 1]
    if recovery_node not in cur_route:
        print(f"回收节点 {recovery_node} 不在回收车辆 {recovery_vehicle} 路线中")
        return []

    cur_index = cur_route.index(recovery_node)

    need = []
    returned_to_origin = False

    # 防死循环：记录 (vehicle, index, returned_flag)
    visited = set()

    while True:
        state = (cur_vehicle, cur_index, returned_to_origin)
        if state in visited:
            # 出现循环/异常链，停止
            break
        visited.add(state)

        route = vehicle_routes[cur_vehicle - 1]
        found_next = False

        for i in range(cur_index, len(route)):
            node = route[i]

            # 这个节点是否存在该无人机的发射标记
            if (node in vehicle_task_data.get(cur_vehicle, {}) and
                hasattr(vehicle_task_data[cur_vehicle][node], "launch_drone_list") and
                drone_id in vehicle_task_data[cur_vehicle][node].launch_drone_list):

                # ✅ 核心停条件：已经回到原车后，原车的下一次发射（例如 93）不删，直接停止
                if returned_to_origin and cur_vehicle == origin_vehicle:
                    found_next = True  # 找到“停点”
                    break

                key = (drone_id, cur_vehicle, node)
                if key not in launch_map:
                    # 有 launch 标记但 customer_plan 里没对应 assignment，跳过继续扫
                    continue

                cid, a = launch_map[key]
                need.append((cid, a))

                # 跳到该任务的回收车辆/回收节点继续追
                _, _, _, rec_node, _, rec_vehicle = a

                if rec_vehicle == origin_vehicle:
                    returned_to_origin = True

                cur_vehicle = rec_vehicle
                rec_route = vehicle_routes[cur_vehicle - 1]
                if rec_node not in rec_route:
                    print(f"回收节点 {rec_node} 不在回收车辆 {cur_vehicle} 路线中")
                    return _dedupe_keep_order(need)

                cur_index = rec_route.index(rec_node)
                found_next = True
                break

        if not found_next:
            break

        # 如果上面命中了“停点”（回到原车后的下一次发射），就退出 while
        if returned_to_origin and cur_vehicle == origin_vehicle:
            # 注意：这时我们没有 append 93
            break

    return _dedupe_keep_order(need)


def _dedupe_keep_order(pairs):
    """按 customer_id 去重，保留第一次出现的顺序"""
    out = []
    seen = set()
    for cid, a in pairs:
        if cid not in seen:
            out.append((cid, a))
            seen.add(cid)
    return out

# def find_chain_tasks(assignment, customer_plan, vehicle_routes, vehicle_task_data):
#     """
#     通过链式找到这个无人机后续的所有服务任务，跟踪无人机任务链直到返回原始发射车辆
    
#     Args:
#         assignment: 被删除的任务 (drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle)
#         customer_plan: 当前客户计划
#         vehicle_routes: 车辆路线
#         vehicle_task_data: 车辆任务数据
    
#     Returns:
#         list: 需要删除的任务列表 [(customer, assignment), ...]
#     """
#     drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle = assignment
#     need_to_remove_tasks = []
    
#     # 如果发射车辆和回收车辆相同，则无需删除后续任务
#     if launch_vehicle == recovery_vehicle:
#         # print(f"无人机 {drone_id} 任务为同车任务，无需删除后续任务")
#         return need_to_remove_tasks
    
#     # print(f"无人机 {drone_id} 任务为异车任务，开始查找后续任务链")
#     # print(f"原始发射车辆: {launch_vehicle}, 当前回收车辆: {recovery_vehicle}")
    
#     # 使用递归函数跟踪无人机任务链
#     def track_drone_chain(current_vehicle, current_node_index, original_launch_vehicle, visited_vehicles=None):
#         """
#         递归跟踪无人机任务链
        
#         Args:
#             current_vehicle: 当前车辆ID
#             current_node_index: 当前节点在路线中的索引
#             original_launch_vehicle: 原始发射车辆ID
#             visited_vehicles: 已访问的车辆集合（防止循环）
#         """
#         if visited_vehicles is None:
#             visited_vehicles = set()
        
#         # 防止无限循环
#         if current_vehicle in visited_vehicles:
#             print(f"检测到循环，停止跟踪车辆 {current_vehicle}")
#             return
        
#         visited_vehicles.add(current_vehicle)
        
#         # 获取当前车辆路线
#         if current_vehicle - 1 >= len(vehicle_routes):
#             print(f"车辆 {current_vehicle} 索引超出范围")
#             return
        
#         current_route = vehicle_routes[current_vehicle - 1]
#         finish_flag = False
#         # 从当前节点开始遍历后续节点
#         for i in range(current_node_index, len(current_route)):
#             node = current_route[i]
            
#             # 检查该节点是否有该无人机的发射任务
#             if (node in vehicle_task_data[current_vehicle] and 
#                 hasattr(vehicle_task_data[current_vehicle][node], 'launch_drone_list') and 
#                 drone_id in vehicle_task_data[current_vehicle][node].launch_drone_list):
                
#                 # print(f"在车辆 {current_vehicle} 的节点 {node} 发现无人机 {drone_id} 的发射任务")
                
#                 # 查找该发射任务对应的客户点
#                 for customer_id, customer_assignment in customer_plan.items():
#                     cust_drone_id, cust_launch_node, cust_customer, cust_recovery_node, cust_launch_vehicle, cust_recovery_vehicle = customer_assignment
                    
#                     # 如果找到匹配的无人机和发射节点
#                     if (cust_drone_id == drone_id and cust_launch_node == node and 
#                         cust_launch_vehicle == current_vehicle):
                        
#                         # print(f"找到需要删除的客户任务: 客户点 {customer_id}, 从车辆 {current_vehicle} 发射到车辆 {cust_recovery_vehicle}")
#                         need_to_remove_tasks.append((customer_id, customer_assignment))
                        
#                         # 如果回收车辆是原始发射车辆，则停止删除后续任务
#                         if cust_recovery_vehicle == original_launch_vehicle:
#                             # print(f"客户点 {customer_id} 的回收车辆 {cust_recovery_vehicle} 是原始发射车辆，停止删除后续任务")
#                             finish_flag = True
#                             break
                        
#                         # 如果回收车辆不是原始发射车辆，继续跟踪
#                         if cust_launch_vehicle != cust_recovery_vehicle:
#                             # print(f"客户点 {customer_id} 的回收车辆 {cust_recovery_vehicle} 不是原始发射车辆，继续跟踪")
                            
#                             # 找到回收节点在回收车辆路线中的位置
#                             if cust_recovery_vehicle - 1 < len(vehicle_routes):
#                                 recovery_route = vehicle_routes[cust_recovery_vehicle - 1]
#                                 recovery_node_index = recovery_route.index(cust_recovery_node) if cust_recovery_node in recovery_route else -1
                                
#                                 if recovery_node_index != -1:
#                                     # 递归跟踪回收车辆的任务链
#                                     track_drone_chain(cust_recovery_vehicle, recovery_node_index, original_launch_vehicle, visited_vehicles.copy())
#                                 else:
#                                     print(f"回收节点 {cust_recovery_node} 不在回收车辆 {cust_recovery_vehicle} 的路线中")
#                         break
#                 if finish_flag: # 如果已经找到原始发射车辆，则停止跟踪
#                     break
                
    
#     # 开始跟踪任务链
#     # 找到回收节点在回收车辆路线中的位置
#     recovery_vehicle_index = recovery_vehicle - 1
#     if recovery_vehicle_index >= len(vehicle_routes):
#         print(f"回收车辆 {recovery_vehicle} 索引超出范围")
#         return need_to_remove_tasks
    
#     recovery_route = vehicle_routes[recovery_vehicle_index]
#     recovery_node_index = recovery_route.index(recovery_node) if recovery_node in recovery_route else -1
    
#     if recovery_node_index == -1:
#         print(f"回收节点 {recovery_node} 不在回收车辆 {recovery_vehicle} 的路线中")
#         return need_to_remove_tasks
    
#     # 从回收节点开始跟踪任务链
#     track_drone_chain(recovery_vehicle, recovery_node_index, launch_vehicle)
    
#     # 去重（避免重复删除）
#     unique_tasks = []
#     seen_customers = set()
#     for customer_id, assignment in need_to_remove_tasks:
#         if customer_id not in seen_customers:
#             unique_tasks.append((customer_id, assignment))
#             seen_customers.add(customer_id)
    
#     # print(f"无人机 {drone_id} 的链式删除任务总数: {len(unique_tasks)}")
#     # for customer_id, _ in unique_tasks:
#     #     print(f"  - 客户点 {customer_id}")
    
#     return unique_tasks

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
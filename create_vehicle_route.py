import numpy as np
import random
from collections import defaultdict
from sklearn.cluster import KMeans
import copy
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB, quicksum
from task_data import *
from cost_y import *
from call_function import *
from initialize import *
from cbs_plan import *
# from insert_plan import *
from down_data import *
# 定义任务类型常量
TASK_IDLE = 10            # 空闲待命
TASK_LOADING = 11         # 货物装载
TASK_UNLOADING = 12       # 货物卸载
TASK_DRONE_LAUNCH = 13    # 无人机发射
TASK_DRONE_RECOVERY = 14  # 无人机回收
TASK_SERVICE = 15         # 客户服务
TASK_CHARGING = 16        # 电池充电
TASK_MAINTENANCE = 17     # 车辆维护

NODE_TYPE_DEPOT	= 0
NODE_TYPE_CUST	= 1

TYPE_TRUCK 		= 1
TYPE_UAV 		= 2


# 定义任务类型常量
TASK_IDLE = 10            # 空闲待命
TASK_LOADING = 11         # 货物装载
TASK_UNLOADING = 12       # 货物卸载
TASK_DRONE_LAUNCH = 13    # 无人机发射
TASK_DRONE_RECOVERY = 14  # 无人机回收
TASK_SERVICE = 15         # 客户服务
TASK_CHARGING = 16        # 电池充电
TASK_MAINTENANCE = 17     # 车辆维护
TASK_DRONE_FLIGHT = 18    # 无人机飞行-前往客户点
TASK_DRONE_FLIGHT_BACK = 19    # 无人机飞行-返回回收点

# 任务类型名称映射
TASK_NAMES = {
    TASK_IDLE: "空闲待命",
    TASK_LOADING: "货物装载",
    TASK_UNLOADING: "货物卸载",
    TASK_DRONE_LAUNCH: "无人机发射",
    TASK_DRONE_RECOVERY: "无人机回收",
    TASK_SERVICE: "客户服务",
    TASK_CHARGING: "电池充电",
    TASK_MAINTENANCE: "车辆维护",
    TASK_DRONE_FLIGHT: "无人机飞行-前往客户点",
    TASK_DRONE_FLIGHT_BACK: "无人机飞行-返回回收点"
}
# 生成可行的初始解任务
def initial_route(node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee):
    vtp_index = A_vtp
    # 提取VTP节点坐标用于聚类
    vtp_coords = np.array([node[i].position for i in vtp_index])
    
   # K-means聚类，将VTP节点均衡分配到与车辆数相同的簇中
    num_trucks = len(T)
    num_clusters = min(num_trucks, len(vtp_index))
    # 初始化路径生成器
    generator = DiverseRouteGenerator(node, DEPOT_nodeID, A_vtp, V, T, vehicle, uav_travel, veh_distance, veh_travel, vtp_coords, num_clusters, G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee)

    vehicle_task_data = generator._create_initial_vehicle_task_data()
    files = list_saved_files()
    # 删除文件
    # delete_saved_file("input_data_20240319_123456")
    # # 备份文件
    # backup_saved_file("input_data_20240319_123456")
    # # 使用自定义名称备份
    # backup_saved_file("input_data_20240319_123456", "important_result")

    # input_filename = get_latest_saved_file()
    # input_filename = "my_special_result_20num_3v_6d_200n"
    # input_filename = None
    input_filename = "my_special_result_20num_3v_6d_100n"
    if input_filename is None:
        # 生成多个候选解
        # num_solutions = 20  # 生成5个候选解
        num_solutions = 20  # 生成5个候选解
        air_vtp_solutions, vehicle_candidate_solutions, total_important_vtps = generator.generate_diverse_solutions(num_solutions)# 该操作生成了多种不同样的车辆路线
        best_customer_plan, best_uav_plan, best_plan_cost, best_vehicle_route, vehicle_task_data, vehicle_arrival_time, best_total_important_vtps = generator.generate_uav_solutions(vehicle_candidate_solutions, vehicle_task_data, total_important_vtps)# 该任务生成不同的无人机路线任务
        uav_task_dict = defaultdict(dict)
        for customer_id in best_customer_plan:
            drone_id, launch_node, customer_id, recovery_node, launch_vehicle, recovery_vehicle = best_customer_plan[customer_id]
            if drone_id not in uav_task_dict:
                uav_task_dict[drone_id] = []
            uav_task_dict[drone_id].append(best_customer_plan[customer_id])
        # 保存输入数据
        input_data = {
        'uav_task_dict': uav_task_dict,
        'best_customer_plan': best_customer_plan,
        'best_uav_plan': best_uav_plan,
        'best_vehicle_route': best_vehicle_route,
        'vehicle_task_data': vehicle_task_data,
        'vehicle_arrival_time': vehicle_arrival_time,
        'node': node,
        'DEPOT_nodeID': DEPOT_nodeID,
        'V': V,
        'T': T,
        'vehicle': vehicle,
        'uav_travel': uav_travel,
        'veh_distance': veh_distance,
        'veh_travel': veh_travel,
        'N': N,
        'N_zero': N_zero,
        'N_plus': N_plus,
        'A_total': A_total,
        'A_cvtp': A_cvtp,
        'A_vtp': A_vtp,
        'A_aerial_relay_node': A_aerial_relay_node,
        'G_air': G_air,
        'G_ground': G_ground,
        'air_matrix': air_matrix,
        'ground_matrix': ground_matrix,
        'air_node_types': air_node_types,
        'ground_node_types': ground_node_types,
        'A_c': A_c,
        'xeee': xeee,
        'best_total_important_vtps': best_total_important_vtps
        }
        # 保存输入数据
        # 这里需要指定你之前保存的文件名
        # 使用自定义名称保存数据
        custom_name = "my_special_result_20num_3v_6d_100n"
        input_filename = save_input_data_with_name(input_data, custom_name)
        # input_filename = save_input_data(input_data)  # 替换为你实际保存的文件名
    # input_filename = save_input_data(input_data)
    uav_task_dict, best_customer_plan, best_uav_plan, best_vehicle_route, vehicle_task_data, vehicle_arrival_time,node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee = run_low_update_time_from_saved(input_filename)
    # 得到粗略的初始方案后，先对粗略的初始方案规划正确的无人机及车辆的时间，并进行简单的任务分配情况描述uav_task_dict, best_uav_plan, best_vehicle_route, vehicle_task_data, vehicle_arrival_time, node, V, T, vehicle, uav_travel
    time_uav_task_dict, time_customer_plan, time_uav_plan, vehicle_plan_time, vehicle_task_data = low_update_time(uav_task_dict, best_uav_plan, 
    best_vehicle_route, vehicle_task_data, vehicle_arrival_time, node, V, T, vehicle, uav_travel)
    # solution_filename = save_solution_data(
    #     time_uav_task_dict,
    #     time_customer_plan,
    #     time_uav_plan,
    #     vehicle_plan_time,
    #     vehicle_task_data
    # )
    # time_uav_task_dict, time_customer_plan, time_uav_plan, vehicle_plan_time, vehicle_task_data = load_solution_data(solution_filename)
    # 针对已有的车辆路线及无人机任务，计算详细的时间分配，运用结合时间迭代的cbs算法规划无避障的空中路径。
    best_uav_plan, best_uav_cost, vehicle_plan_time, best_vehicle_task_data, global_reservation_table = rolling_time_cbs(vehicle_arrival_time, 
    best_vehicle_route, time_uav_task_dict, time_customer_plan, time_uav_plan, vehicle_plan_time, vehicle_task_data, node, DEPOT_nodeID, 
    V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, 
    air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee)
    
    # 根据重新规划好的新方案，重新计算总成本
    best_total_cost = calculate_plan_cost(best_uav_cost, best_vehicle_route, vehicle, T, V, veh_distance)
    # best_total_cost = update_calculate_plan_cost(best_uav_plan, best_vehicle_route, vehicle, T, V, veh_distance)
    
    return  best_total_cost, best_uav_plan, best_customer_plan, time_uav_task_dict, best_uav_cost, best_vehicle_route, vehicle_plan_time, best_vehicle_task_data, global_reservation_table

# 设计生成多样化的车辆路径，以及生成无人机插入，组成完成的车辆+无人机初始线路
class DiverseRouteGenerator:
    """生成多样化高质量车辆路径的类"""
    def __init__(self, node, depot_id, vtp_indices, uav_ids, truck_ids, vehicle, uav_travel, veh_distance, veh_travel, vtp_coords, num_clusters, G_air, G_ground, air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee):   
        self.node = node
        self.depot_id = depot_id
        self.vtp_indices = vtp_indices
        self.uav_ids = uav_ids
        self.truck_ids = truck_ids
        self.vehicle = vehicle
        self.uav_travel = uav_travel
        self.veh_distance = veh_distance
        self.veh_travel = veh_travel    
        self.num_trucks = len(truck_ids)
        self.num_uavs = len(uav_ids)
        self.num_clusters = num_clusters
        self.vtp_coords = vtp_coords
        self.G_air = G_air
        self.G_ground = G_ground
        self.air_matrix = air_matrix
        self.ground_matrix = ground_matrix
        self.air_node_types = air_node_types
        self.ground_node_types = ground_node_types
        self.A_c = A_c
        self.vtp_index = {}
        self.xeee = xeee
        self.save_path = r'saved_solutions'
        
        # 提取坐标
        self.depot_pos = np.array([node[depot_id].latDeg, node[depot_id].lonDeg, node[depot_id].altMeters])
        self.vtp_positions = vtp_coords
        # 构建vtp与距离矩阵之间的映射索引
        self.depot_index = 0
        vtp_index = 0
        # 搭建得到的距离矩阵与实际node节点之间的映射关系索引
        for node_id in self.node:
            if self.node[node_id].nodeType == 'VTP Takeoff/Landing Node':
                self.vtp_index[node_id] = vtp_index
                vtp_index += 1
        
        # 计算距离矩阵
        self._compute_distance_matrices()
    
    # 快速生成可行的车辆-无人机初始路径方案
    def generate_uav_solutions(self, vehicle_candidate_solutions, vehicle_task_data, total_important_vtps):
        uav_solutions = defaultdict(dict)
        insert_costs = []
        customer_uav_plans = {}
        diversity_costs = []
        best_uav_solution = None
        best_insert_cost = float('inf')
        task_arrival_times = defaultdict(dict)
        customer_plan = []
        uav_plan = []
        plan_cost = []
        total_vehicle_task = []
        total_vehicle_route = []
        feasible_plans = []
        num_plan_cost = []
        vehicle_arrival_time = []
        # total_important_vtps = []
        for num_index, solution in enumerate(vehicle_candidate_solutions):
            veh_arrival_times = defaultdict(dict)
            # 1. 根据车辆路线，计算车辆到达各个节点的时间
            # y_cijkdu = defaultdict(dict)
            for index, route in enumerate(solution, start=0):
                vehicle_id = self.truck_ids[index]
                for route_index, node_j in enumerate(route):
                    if route_index == 0:
                        veh_arrival_times[vehicle_id][node_j] = 0
                        continue
                    else:
                        node_i = route[route_index-1]
                        veh_arrival_times[vehicle_id][node_j] = veh_arrival_times[vehicle_id][node_i] + self.veh_travel[vehicle_id][node_i][node_j]
            # 2. 为每个客户点找到可行的无人机配送方案,车辆的路径随着vehicle——task_data更新
            current_vehicle_task = self._create_initial_vehicle_task_data()
            best_customer_plan, best_uav_plan, best_plan_cost, update_vehicle_task_data = self._find_feasible_uav_plans(solution, veh_arrival_times, current_vehicle_task)
            # 根据选择的最优调度方案，更新车辆和无人机在各个节点的状态
            customer_plan.append(best_customer_plan)
            uav_plan.append(best_uav_plan)
            plan_cost.append(best_plan_cost)
            total_vehicle_task.append(update_vehicle_task_data)
            total_vehicle_route.append(solution)
            vehicle_arrival_time.append(veh_arrival_times)
            # total_important_vtps.append(total_important_vtps[num_index])
            # 计算每种方案的总成本
            total_cost = calculate_plan_cost(best_plan_cost, solution, self.vehicle, self.truck_ids, self.uav_ids, self.veh_distance)
            num_plan_cost.append(total_cost)
        # 找到代价最小的方案组合
        num_plan_cost = np.array(num_plan_cost)
        min_plan_cost_index = np.argmin(num_plan_cost)
        best_customer_plan = customer_plan[min_plan_cost_index]
        best_uav_plan = uav_plan[min_plan_cost_index]
        best_plan_cost = plan_cost[min_plan_cost_index]
        best_vehicle_task = total_vehicle_task[min_plan_cost_index]
        best_vehicle_route = total_vehicle_route[min_plan_cost_index]
        best_vehicle_arrival_time = vehicle_arrival_time[min_plan_cost_index]
        best_total_important_vtps = total_important_vtps[min_plan_cost_index]
        return best_customer_plan, best_uav_plan, best_plan_cost, best_vehicle_route, best_vehicle_task, best_vehicle_arrival_time, best_total_important_vtps

    def _create_initial_vehicle_task_data(self):
            """
            工厂方法：创建一个全新的、原始状态的 vehicle_task_data 对象。
            这个方法每次被调用时，都会返回一个“干净”的实例。
            """
                # 修改defaultdict的创建方式
            def create_dict():
                return defaultdict(lambda: None)
            vehicle_task_data = defaultdict(create_dict)
            
            # 第一遍：初始化所有对象并设置卡车的无人机列表
            # 这样做可以确保在计算无人机归属时，卡车状态已经就绪
            for vehicle_id, veh in self.vehicle.items():
                for node_id in self.node:
                    vehicle_task_data[vehicle_id][node_id] = vehicle_task(
                        vehicle_id, veh.vehicleType, node_id, self.node
                    )
                    if veh.vehicleType == TYPE_TRUCK:
                        drone_list = list(veh.drones.keys())
                        vehicle_task_data[vehicle_id][node_id].update_drone_list(drone_list)

            for vehicle_id, veh in self.vehicle.items():
                if veh.vehicleType == TYPE_TRUCK:
                    drone_list = list(veh.drones.keys())
                    belong_vehicle_id = vehicle_id
                    for drone_id in drone_list:
                        for node_id in self.node:
                            vehicle_task_data[drone_id][node_id].update_drone_belong(
                                drone_id, belong_vehicle_id
                            )
                            # 确保无人机的任务数据已经创建
                            if vehicle_task_data[drone_id][node_id] is None:
                                drone_task = vehicle_task(drone_id, TYPE_UAV, node_id, self.node)
                                vehicle_task_data[drone_id][node_id] = drone_task
                            # 更新无人机在该车辆上的信息
                            drone_task = vehicle_task_data[drone_id][node_id]
                            if vehicle_id in drone_task.dict_vehicle:
                                drone_task.dict_vehicle[vehicle_id]['drone_belong'] = vehicle_id
                                drone_task.dict_vehicle[vehicle_id]['precise_arrive_time'] = 0
                                drone_task.dict_vehicle[vehicle_id]['precise_departure_time'] = 0
                                drone_task.dict_vehicle[vehicle_id]['launch_drone_list'] = []
                                drone_task.dict_vehicle[vehicle_id]['recovery_drone_list'] = []
                            

            return vehicle_task_data

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

    # 处理跨地面车辆发射和回收的客户点任务的所有情况及可行方案
    def find_cross_total_customer_plan(self, customer, launch_vehicle_segment, recovery_vehicle_segment, drone_id, launch_vehicle, recovery_vehicle, vehicle_arrival_time):
        y_plan = []
        y_cost = []
        y_time = []
        y_uav_route = []
        launch_vehicle_index = launch_vehicle - 1
        recovery_vehicle_index = recovery_vehicle - 1
        # 原车辆发射情况
        for i in range(len(launch_vehicle_segment)):
            for j in range(i+1, len(launch_vehicle_segment)):
                if launch_vehicle_segment[i] == launch_vehicle_segment[j]:
                    continue
                # 计算方案是否可行
                air_i = self.node[launch_vehicle_segment[i]].map_key
                air_j = self.node[launch_vehicle_segment[j]].map_key
                energy = self.xeee[drone_id][air_i][customer][air_j]
                if energy > 0 and energy is not None:
                    y = (drone_id, launch_vehicle_segment[i], customer, launch_vehicle_segment[j], launch_vehicle, launch_vehicle)
                    y_plan.append(y)
                    cost, time, uav_route = cal_low_cost(launch_vehicle_segment[i], customer, launch_vehicle_segment[j], launch_vehicle, launch_vehicle, drone_id, self.uav_travel, self.veh_distance, self.veh_travel, self.node, self.vehicle, 2, self.xeee)
                    y_cost.append(cost)
                    y_time.append(time)
                    y_uav_route.append(uav_route)
        # 回收车辆情况
        for i in range(len(recovery_vehicle_segment)):
            for j in range(i+1, len(recovery_vehicle_segment)):
                if recovery_vehicle_segment[i] == recovery_vehicle_segment[j]:
                    continue
                # 计算方案是否可行
                air_i = self.node[recovery_vehicle_segment[i]].map_key
                air_j = self.node[recovery_vehicle_segment[j]].map_key
                energy = self.xeee[drone_id][air_i][customer][air_j]
                if energy > 0 and energy is not None:
                    y = (drone_id, recovery_vehicle_segment[i], customer, recovery_vehicle_segment[j], recovery_vehicle, recovery_vehicle)
                    y_plan.append(y)
                    cost, time, uav_route = cal_low_cost(recovery_vehicle_segment[i], customer, recovery_vehicle_segment[j], recovery_vehicle, recovery_vehicle, drone_id, self.uav_travel, self.veh_distance, self.veh_travel, self.node, self.vehicle, 2, self.xeee)
                    y_cost.append(cost)
                    y_time.append(time)
                    y_uav_route.append(uav_route)
        # 原车辆发射，回收车辆回收
        for i in range(len(launch_vehicle_segment)):
            for j in range(len(recovery_vehicle_segment)):
                if launch_vehicle_segment[i] == recovery_vehicle_segment[j]:
                    continue
                # 计算方案是否可行
                air_i = self.node[launch_vehicle_segment[i]].map_key
                air_j = self.node[recovery_vehicle_segment[j]].map_key
                energy = self.xeee[drone_id][air_i][customer][air_j]
                if energy > 0 and energy is not None:
                    if vehicle_arrival_time[launch_vehicle][launch_vehicle_segment[i]] < vehicle_arrival_time[recovery_vehicle][recovery_vehicle_segment[j]]:
                        y = (drone_id, launch_vehicle_segment[i], customer, recovery_vehicle_segment[j], launch_vehicle, recovery_vehicle)
                        y_plan.append(y)
                        cost, time, uav_route = cal_low_cost(launch_vehicle_segment[i], customer, recovery_vehicle_segment[j], launch_vehicle, recovery_vehicle, drone_id, self.uav_travel, self.veh_distance, self.veh_travel, self.node, self.vehicle, 2, self.xeee)
                        y_cost.append(cost)
                        y_time.append(time)
                        y_uav_route.append(uav_route)
        # 发射车辆发射，原车辆回收
        for j in range(len(recovery_vehicle_segment)):
            for i in range(len(launch_vehicle_segment)):
                if launch_vehicle_segment[i] == recovery_vehicle_segment[j]:
                    continue
                # 计算方案是否可行
                air_i = self.node[recovery_vehicle_segment[j]].map_key
                air_j = self.node[launch_vehicle_segment[i]].map_key
                energy = self.xeee[drone_id][air_i][customer][air_j]
                if energy > 0 and energy is not None:
                    if vehicle_arrival_time[recovery_vehicle][recovery_vehicle_segment[j]] < vehicle_arrival_time[launch_vehicle][launch_vehicle_segment[i]]:
                        y = (drone_id, recovery_vehicle_segment[j], customer, launch_vehicle_segment[i], recovery_vehicle, launch_vehicle)
                        y_plan.append(y)
                        cost, time, uav_route = cal_low_cost(recovery_vehicle_segment[j], customer, launch_vehicle_segment[i], recovery_vehicle, launch_vehicle, drone_id, self.uav_travel, self.veh_distance, self.veh_travel, self.node, self.vehicle, 2, self.xeee)
                        y_cost.append(cost)
                        y_time.append(time)
                        y_uav_route.append(uav_route)
        return y_plan, y_cost, y_time, y_uav_route



    def find_total_customer_plan(self, customer, segment, drone_id, vehicle_id):
         # 初始化存储结构
        y_plan = []  # 方案详细信息
        y_cost = []  # 方案成本
        y_time = []  # 方案时间
        y_uav_route = []  # 方案无人机路径


        for i in range(len(segment)):
            for j in range(i+1, len(segment)):
                if segment[i] == segment[j]:
                    continue
                # 计算方案是否可行
                air_i = self.node[segment[i]].map_key
                air_j = self.node[segment[j]].map_key
                energy = self.xeee[drone_id][air_i][customer][air_j]
                if energy > 0 and energy is not None:
                    y = (drone_id, segment[i], customer, segment[j], vehicle_id, vehicle_id)
                    y_plan.append(y)
                    cost, time, uav_route = cal_low_cost(segment[i], customer, segment[j], vehicle_id, vehicle_id, drone_id, self.uav_travel, self.veh_distance, self.veh_travel, self.node, self.vehicle, 2, self.xeee)
                    y_cost.append(cost)
                    y_time.append(time)
                    y_uav_route.append(uav_route)
        return y_plan, y_cost, y_time, y_uav_route
            


    def find_nearest_customer_plan(self, customer, solution_route, solution, vehicle_task_data):
        y_cijkdu = {uav_id: [] for uav_id in self.uav_ids}
        y_plan = {}
        y_cost = {}
        best_y_cost = float('inf')
        best_y_cijkdu_plan = None
        xeee = self.xeee.copy()
        # remain_customer = copy.copy(customer)

        for drone_id in xeee:
            for i in xeee[drone_id]:
                i_vtp = self.node[i].map_key
                if customer not in xeee[drone_id][i]:
                    continue
                
                for j in xeee[drone_id][i][customer]:
                    j_vtp = self.node[j].map_key
                    energy = xeee[drone_id][i][customer][j]
                    if not (energy > 0 and energy is not None):
                        continue

                    for v_id, route_info in solution.items():
                        # 检查发射节点是否在车v_id的路线上，且车上携带此无人机
                        if i_vtp not in route_info or drone_id not in vehicle_task_data[v_id][i_vtp].drone_list:
                            continue

                        # 预计算发射时间
                        launch_time = route_info[i_vtp] + abs(self.vehicle[v_id].launchTime)

                        for recv_v_id, recv_route_info in solution.items():
                            # 检查回收节点是否在车recv_v_id的路线上
                            if j_vtp not in recv_route_info:
                                continue
                            
                            # 预计算回收时间
                            recovery_time = recv_route_info[j_vtp] + abs(self.vehicle[recv_v_id].recoveryTime)

                            # 基本时间检查：发射必须早于回收
                            if launch_time >= recovery_time:
                                continue

                            # ==========================================================================
                            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心改进：调用外部函数进行冲突检测 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                            # ==========================================================================
                            is_valid_plan = False
                            if v_id == recv_v_id:
                                # 场景一：同一辆车发射和回收
                                is_valid_plan = check_same_vehicle_conflict(
                                    v_id=v_id,
                                    drone_id=drone_id,
                                    i_vtp=i_vtp,
                                    j_vtp=j_vtp,
                                    solution_route=solution_route,
                                    solution=solution,
                                    vehicle_task_data=vehicle_task_data,    
                                    vehicle = self.vehicle
                                )
                            else:
                                # 场景二：跨车辆发射和回收
                                is_valid_plan = check_cross_vehicle_conflict_fixed(
                                    launch_v_id=v_id,
                                    recover_v_id=recv_v_id,
                                    drone_id=drone_id,
                                    i_vtp=i_vtp,
                                    j_vtp=j_vtp,
                                    solution_route=solution_route,
                                    solution=solution,
                                    vehicle_task_data=vehicle_task_data,
                                    launch_time=launch_time,
                                    recovery_time=recovery_time,
                                    vehicle=self.vehicle
                                )
                            
                            # 如果方案无效，则跳过当前组合
                            if not is_valid_plan:
                                continue

                            # 如果所有检查都通过，则这是一个可行的方案
                            cost, time, uav_route = cal_low_cost(i_vtp, customer, j_vtp, v_id, recv_v_id, drone_id, self.uav_travel, self.veh_distance, self.veh_travel, self.node, self.vehicle, 2, self.xeee)
                            
                            plan = {
                                'drone_id': drone_id,
                                'launch_vehicle': v_id,
                                'recovery_vehicle': recv_v_id,
                                'launch_node': i_vtp,
                                'recovery_node': j_vtp,
                                'customer': customer,
                                'launch_time': launch_time,
                                'recovery_time': recovery_time,
                                'energy': energy,
                                'cost': cost,
                                'time': time,
                                'uav_route': uav_route
                            }
                            
                            key = (drone_id, i_vtp, customer, j_vtp, v_id, recv_v_id)
                            
                            # if isinstance(remain_customer, list):
                            #     remain_customer.remove(customer)
                            # else:
                            #     remain_customer = [remain_customer]
                            #     remain_customer.remove(customer)
                            y_cijkdu[drone_id].append(plan)
                            y_plan[key] = plan
                            y_cost[key] = cost

                            if cost < best_y_cost:
                                best_y_cost = cost
                                best_y_cijkdu_plan = plan

        return y_cijkdu, y_plan, y_cost, best_y_cijkdu_plan, best_y_cost


    def _find_feasible_uav_plans(self, vehicle_route, vehicle_arrival_time, vehicle_task_data):
        """找到客户点c的可行无人机配送方案"""
        # 1. 统计每个客户点的服务需求数量
        # customer表示车辆路线，solution表示到达时间
        best_customer_plan = defaultdict(dict)
        best_uav_plan = defaultdict(dict)
        best_plan_cost = defaultdict(dict)
        customer_service_count = defaultdict(int)
        for c in self.A_c:
            for drone_id in self.xeee:
                for vtp_i in self.xeee[drone_id]:
                    for vtp_j in self.xeee[drone_id][vtp_i][c]:
                        if vtp_i == vtp_j:
                            continue
                        if self.xeee[drone_id][vtp_i][c][vtp_j] is not None:  # 代表存在前往该客户点的配送方式
                            customer_service_count[c] += 1

        # 2. 按服务数量从大到小排序客户点        
        sorted_customers = sorted(customer_service_count.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
        # 3. 遍历每个客户点，找到可行的无人机配送方案,完成了当前方案中客户点c的无人机配送方案
        for customer, service_count in sorted_customers:
            feasible_plans = []
            # 选择当前成本最低，航程最短的无人机配送方案，随后更新车辆和无人机在各个节点的约束状态,根据当前车辆无人机在各个节点状态，选择当前距离最近的无人机配送方案，符合约束条件
            [y_cijkdu, y_plan, y_cost, best_y_cijkdu_plan, best_y_cost] = self.find_nearest_customer_plan(customer, vehicle_route, vehicle_arrival_time, vehicle_task_data)  # 输入客户点及车辆，筛选出所有可能的决策方案，并获得距离最优最近的方案
            # 上述方案为寻找贪婪最优解方案，如果当前全部任务被无人机占据，则需要重新寻找
            if best_y_cijkdu_plan is None:
                best_orig_y, best_new_y, best_orig_cost, best_new_cost, best_orig_y_cijkdu_plan, best_new_y_cijkdu_plan = self.greedy_insert_feasible_plan(customer, vehicle_route, vehicle_arrival_time, vehicle_task_data, best_customer_plan)
                orig_drone_id, orig_launch_node, orig_customer, orig_recovery_node, orig_launch_vehicle, orig_recovery_vehicle = best_orig_y
                new_drone_id, new_launch_node, new_customer, new_recovery_node, new_launch_vehicle, new_recovery_vehicle = best_new_y
                if orig_customer == customer:
                    remove_customer = new_customer
                else:
                    remove_customer = orig_customer
                # 删除被对应拆除的方案
                y = best_customer_plan[remove_customer]
                del best_customer_plan[remove_customer]
                del best_plan_cost[remove_customer]
                del best_uav_plan[y]
                vehicle_task_data = remove_vehicle_task(vehicle_task_data, y, vehicle_route)
                # 更新新的记录方案,优先拆谁，优先补充谁
                if best_new_y_cijkdu_plan['launch_time'] < best_orig_y_cijkdu_plan['launch_time']:
                    best_customer_plan[new_customer] = best_new_y
                    best_plan_cost[new_customer] = best_new_cost
                    best_uav_plan[best_new_y] = best_new_y_cijkdu_plan
                    vehicle_task_data = update_vehicle_task(vehicle_task_data, best_new_y, vehicle_route)
                else:
                    best_customer_plan[orig_customer] = best_orig_y
                    best_plan_cost[orig_customer] = best_orig_cost
                    best_uav_plan[best_orig_y] = best_orig_y_cijkdu_plan
                    vehicle_task_data = update_vehicle_task(vehicle_task_data, best_orig_y, vehicle_route)
            else: 
                drone_id = best_y_cijkdu_plan['drone_id']
                launch_node = best_y_cijkdu_plan['launch_node']
                recovery_node = best_y_cijkdu_plan['recovery_node']
                launch_vehicle = best_y_cijkdu_plan['launch_vehicle']
                recovery_vehicle = best_y_cijkdu_plan['recovery_vehicle']
                y = (drone_id, launch_node, customer,recovery_node, launch_vehicle, recovery_vehicle)
                # 记录最优方案
                best_customer_plan[customer] = y
                best_plan_cost[customer] = best_y_cost
                best_uav_plan[y] = best_y_cijkdu_plan
                # 根据选择的最优调度方案，更新车辆和无人机在各个节点的状态
                vehicle_task_data = update_vehicle_task(vehicle_task_data, y, vehicle_route)
                
        return best_customer_plan, best_uav_plan, best_plan_cost, vehicle_task_data
                

    
    def air_route_convert_ground_route(self, solution, node):  # 将空中的vtp路径转换为地面路径
        """将空中路径转换为地面路径"""
        ground_route = []
        for vtp_index in solution:
            ground_route.append(node[vtp_index].map_key)
        return ground_route

    
    def _compute_distance_matrices(self):
        """计算各种距离矩阵"""
        # VTP之间的距离
        self.vtp_distances = cdist(self.vtp_positions, self.vtp_positions)
        
        # 仓库到VTP的距离
        self.depot_vtp_distances = cdist([self.depot_pos], self.vtp_positions)[0]
        
        # 客户节点位置（用于评估VTP重要性）
        customer_nodes = [n for n in self.node if self.node[n].nodeType == 'CUSTOMER']
        if customer_nodes:
            self.customer_positions = np.array([[self.node[i].latDeg, self.node[i].lonDeg, self.node[i].altMeters] 
                                               for i in customer_nodes])
            self.vtp_customer_distances = cdist(self.vtp_positions, self.customer_positions)
        else:
            self.customer_positions = np.array([])
            self.vtp_customer_distances = np.array([])
    
    def generate_diverse_solutions(self, num_solutions):
        """生成多样化的解决方案"""
        solutions = []
        vtp_solutions = []
        total_important_vtps = []
        # 1. 使用不同的聚类参数生成基础解
        clustering_params = self._generate_clustering_params(max(1, num_solutions // 3))
        for params in clustering_params:
            clusters_route = []
            clusters = self._cluster_vtps(**params)  # 通过**将字典参数解包
            routes = self._solve_tsp_for_clusters(clusters)
            # 将得到的路线转换为地面路径
            solutions.append(routes)
            # self.plot_vehicle_routes(routes)
            for route in routes:
                ground_route = self.air_route_convert_ground_route(route, self.node)
                clusters_route.append(ground_route)
            vtp_solutions.append(clusters_route)
        
        # 2. 基于节点重要性重新分配
        if solutions:
            important_vtps = self._evaluate_vtp_importance()  # 得到了每个vtp的重要性得分,键为vtp的id,值为得分
            base_solution = solutions.copy()
            diversified_solutions = self._redistribute_important_nodes(base_solution, important_vtps)
            # 将多样化的解决方案添加到解决方案列表中
            solutions.extend(diversified_solutions)
            for num_index, solution in enumerate(diversified_solutions):
                diversified_vtp_solutions = []
                for route in solution:
                    ground_route = self.air_route_convert_ground_route(route, self.node)
                    diversified_vtp_solutions.append(ground_route)
                vtp_solutions.append(diversified_vtp_solutions)
                total_important_vtps.extend(important_vtps)
        return solutions, vtp_solutions, total_important_vtps

    
    def _generate_clustering_params(self, num_params):
        """生成不同的聚类参数"""
        params_list = []
        
        init_methods = ['k-means++', 'random']
        random_states = [42, 123, 456, 789, 1000, 1234, 1456, 1678, 1890, 2000]
        use_depot_distance = [True, False]
        
        for i in range(num_params):
            params = {
                'init_method': init_methods[i % len(init_methods)],
                'random_state': random_states[i % len(random_states)],
                'use_depot_distance': use_depot_distance[i % len(use_depot_distance)]
            }
            params_list.append(params)
        
        return params_list
    
    def _cluster_vtps(self, init_method='k-means++', random_state=42, use_depot_distance=True):
        """对VTP节点进行聚类"""
        if len(self.vtp_indices) == 0:
            return {i: [] for i in range(self.num_trucks)}
        
        # 构建特征矩阵
        if use_depot_distance:
            features = np.column_stack([
                self.vtp_positions,
                self.depot_vtp_distances.reshape(-1, 1)
            ])
        else:
            features = self.vtp_positions
        
        # 确定聚类数量
        n_clusters = min(self.num_trucks, len(self.vtp_indices))
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=random_state)
        labels = kmeans.fit_predict(features)
        
        # 整理聚类结果
        clusters = {i: [] for i in range(self.num_trucks)}
        for idx, (vtp_idx, cluster_id) in enumerate(zip(self.vtp_indices, labels)):
            clusters[cluster_id].append(vtp_idx)
        
        return clusters
    
    def _solve_tsp_for_clusters(self, clusters):
        """为每个聚类求解TSP"""
        routes = []
        
        for truck_idx in range(self.num_trucks):
            vtp_indices = clusters.get(truck_idx, [])
            
            if len(vtp_indices) == 0:
                # 空路径
                route = []
            elif len(vtp_indices) == 1:
                # 只有一个节点,添加仓库作为所有车辆的出发点和终止点
                route = vtp_indices
                route.insert(0, self.depot_index)
                # route.append(self.depot_index)
            else:
                # 使用Gurobi求解TSP
                vtp_indices.insert(0, self.depot_index)
                # vtp_indices.append(self.depot_index)
                route = self._solve_tsp_gurobi(vtp_indices)
            routes.append(route)
        
        return routes
    
    def _solve_tsp_gurobi(self, vtp_indices):
        """使用Gurobi求解TSP"""
        # 创建节点列表
        nodes = vtp_indices.copy()
        n = len(nodes)
        
        # 构建距离矩阵
        dist_matrix = np.zeros((n, n))
        
       # 填充距离矩阵,根据仓库及vtp节点分布生成新的距离矩阵
        for i in range(n):
            for j in range(n):
                if i != j:
                    node_i = nodes[i]
                    node_j = nodes[j]
                    
                    if node_i == 0 and node_j != 0:  # 仓库到VTP
                        vtp_pos_idx = self.vtp_index[node_j]
                        dist_matrix[i, j] = self.depot_vtp_distances[vtp_pos_idx]
                    elif node_i != 0 and node_j == 0:  # VTP到仓库
                        vtp_pos_idx = self.vtp_index[node_i]
                        dist_matrix[i, j] = self.depot_vtp_distances[vtp_pos_idx]
                    else:  # VTP之间
                        if node_i != 0 and node_j != 0:
                            vtp_i_idx = self.vtp_index[node_i]
                            vtp_j_idx = self.vtp_index[node_j]
                            dist_matrix[i, j] = self.vtp_distances[vtp_i_idx, vtp_j_idx]
        # 创建Gurobi模型
        model = Model("TSP")
        
        # 设置不打印日志
        model.params.OutputFlag = 0
        
        # 设置求解时间限制（秒）
        model.params.timeLimit = 600
        
        # 创建决策变量
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.addVar(lb=0, ub=1, obj=float(dist_matrix[i, j]), 
                                        vtype=GRB.BINARY, name=f"x.{i}.{j}")
        
        model.update()
        
        # 添加约束：每个节点只能出发一次
        for i in range(n):
            model.addConstr(quicksum(x[i, j] for j in range(n) if j != i) == 1, f"out_{i}")
        
        # 添加约束：每个节点只能到达一次
        for j in range(n):
            model.addConstr(quicksum(x[i, j] for i in range(n) if i != j) == 1, f"in_{j}")
        
        # 定义DFS函数来查找连通分量
        def dfs(nodes, graph):
            visited = set()
            components = {}
            comp_id = 0
            
            def visit(node, comp):
                visited.add(node)
                components[comp].append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visit(neighbor, comp)
            
            for node in nodes:
                if node not in visited:
                    components[comp_id] = []
                    visit(node, comp_id)
                    comp_id += 1
            
            return components
        
        # 回调函数：用延迟约束消除子回路
        def subtourelim(model, where):
            if where == GRB.Callback.MIPSOL:
                # 提取当前解中选择的边
                from collections import defaultdict
                delta_dfs = defaultdict(list)
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            if (i, j) in x:
                                x_sol = model.cbGetSolution(x[i, j])
                                if x_sol > 0.5:
                                    delta_dfs[i].append(j)
                                    delta_dfs[j].append(i)
                
                # 使用DFS查找连通分量
                components = dfs(range(n), delta_dfs)
                
                # 如果有多个连通分量，添加子回路消除约束
                if len(components) > 1:
                    for k in components:
                        comp = components[k]
                        if len(comp) < n:
                            model.cbLazy(quicksum(x[i, j] 
                                                for i in comp 
                                                for j in comp 
                                                if i != j) <= len(comp) - 1)
        
        # 启用延迟约束
        model.params.lazyConstraints = 1
        
        # 优化模型
        model.optimize(subtourelim)
        
        # 提取最优路径
        if model.status == GRB.OPTIMAL:
            route = [self.depot_id]  # 从仓库开始
            current = 0
            
            while len(route) < n:
                for j in range(n):
                    if j != current and (current, j) in x and x[current, j].x > 0.5:
                        route.append(nodes[j])
                        current = j
                        break
            route.append(self.depot_id)
            return route
        else:
            # 如果没有找到最优解，返回贪心解
            print("Gurobi无法找到最优解，使用贪心算法")
            route = [self.depot_id]  # 从仓库开始
            unvisited = set(range(1, n))
            
            while unvisited:
                current = route[-1]
                current_idx = nodes.index(current)
                next_idx = min(unvisited, key=lambda j: dist_matrix[current_idx, j])
                route.append(nodes[next_idx])
                unvisited.remove(next_idx)
            route.append(self.depot_id)
            return route

    def plot_vehicle_routes(self, routes):
        """
        绘制车辆路径
        
        参数:
        routes: 列表或嵌套列表，包含节点访问顺序
        coordinates: 字典或列表，包含节点坐标
        
        例如:
        routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
        coordinates = {0: (0, 0), 1: (1, 2), 2: (3, 1), 3: (5, 2), 4: (4, 0)}
        或
        coordinates = [(0, 0), (1, 2), (3, 1), (5, 2), (4, 0)]
        """
        # plt.figure(figsize=(10, 8))
        # 创建一个新窗口
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,        
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,   
        'axes.titlesize': 16,   
        'xtick.labelsize': 12,  
        'ytick.labelsize': 12,
        'legend.fontsize': 10,  
        'legend.frameon': True,
        'legend.framealpha': 0.7,
        'legend.edgecolor': 'k'
    })
        # 检查routes是否为嵌套列表
        if not isinstance(routes[0], list):
            routes = [routes]  # 如果不是嵌套列表，将其转换为嵌套列表

        # 1. 首先绘制空中路网连接
        for edge in self.G_air.edges():
            node1, node2 = edge
            x1, y1 = self.node[node1].latDeg, self.node[node1].lonDeg
            x2, y2 = self.node[node2].latDeg, self.node[node2].lonDeg
            
            # 使用灰色虚线绘制空中连接
            plt.plot([x1, x2], [y1, y2], 
                    color='#666666',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.4,
                    zorder=1)  # 确保空中连接在最底层
            
        # 为每条路径分配不同的颜色
        colors = plt.cm.jet(np.linspace(0, 1, len(routes)))
        
        # 获取所有节点的坐标
        all_nodes = set()
        for route in routes:
            all_nodes.update(route)
        
        # 绘制每条路径
        for i, route in enumerate(routes):
            x_coords = []
            y_coords = []
            
            for node in route:
                # 获取节点坐标
                x, y = self.node[node].latDeg, self.node[node].lonDeg
                
                x_coords.append(x)
                y_coords.append(y)
            
            # 绘制路径线
            plt.plot(x_coords, y_coords, 'o-', color=colors[i], linewidth=2, markersize=8, label=f'route {i+1}')
            
            # 标记节点
            for j, node in enumerate(route):
                x, y = self.node[node].latDeg, self.node[node].lonDeg
                
                # 特别标记仓库节点
                if self.node[node].nodeType == 'DEPOT':
                    plt.plot(x, y, 'ks', markersize=12)  # 黑色方形表示仓库
                    plt.text(x, y+0.1, f'depot', fontsize=12, ha='center')
                else:
                    plt.text(x, y+0.1, f'{node}', fontsize=10, ha='center')

        # 整理客户节点坐标，并在图中用*标记
        # customer_nodes = [node for node in self.A_c]
        for node in self.A_c:
            x, y = self.node[node].latDeg, self.node[node].lonDeg
            plt.plot(x, y, 'b*', markersize=15)
        
        # 设置图的标题和标签
        plt.title('Vehicle Route Visualization', fontsize=15)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        # plt.legend()
        plt.tight_layout()
        plt.ion()  # 打开交互模式
        plt.show()
        # 保存图像，使用相同的DPI确保一致性
        if self.save_path:
            plt.savefig(self.save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close('all')  # 关闭所有打开的图形窗口
        print('路径可视化完成')
    
    def _greedy_tsp(self, vtp_indices):
        """贪婪TSP算法作为备选"""
        if not vtp_indices:
            return []
        
        route = []
        remaining = set(vtp_indices)
        
        # 从离仓库最近的点开始
        current_pos = self.depot_pos
        
        while remaining:
            best_dist = float('inf')
            best_node = None
            
            for vtp in remaining:
                vtp_pos = np.array([self.node[vtp].latDeg, self.node[vtp].lonDeg])
                dist = np.linalg.norm(current_pos - vtp_pos)
                
                if dist < best_dist:
                    best_dist = dist
                    best_node = vtp
            
            if best_node is not None:
                route.append(best_node)
                remaining.remove(best_node)
                current_pos = np.array([self.node[best_node].latDeg, self.node[best_node].lonDeg])
        
        return route
    
    def _evaluate_vtp_importance(self):
        """评估空中的VTP节点的重要性"""
        importance_scores = {}
        
        # 无人机飞行范围（假设值，实际应从vehicle参数获取）
        drone_range = 10  
        
        # 1. 初始化评分指标的权重
        weights = {
            'connectivity': 0.3,    # 连接性权重
            'centrality': 0.2,      # 中心性权重
            'customer_proximity': 0.4,  # 与客户节点距离权重
            'depot_proximity': 0.1     # 与仓库距离权重
        }
        
        # 存储所有分数以便后续归一化
        all_scores = {metric: [] for metric in weights.keys()}

        for vtp_id in self.vtp_indices:
            air_index = self.node[vtp_id].map_index
            # 计算连接性分数
            air_degree = sum(self.air_matrix[air_index] > 0)
            connectivity_score = air_degree / len(self.air_matrix)
            all_scores['connectivity'].append(connectivity_score)
            # 计算中心性分数
            # air_centrality = 1 / (np.sum(self.air_matrix[air_index]) + 1e-10)
            # all_scores['centrality'].append(air_centrality)
            reachable_vtp_count = 0
            total_vtp_nodes = len(self.vtp_indices)
            for other_vtp_id in self.vtp_indices:
                if other_vtp_id != vtp_id:
                    distance = np.linalg.norm(
                        np.array([self.node[vtp_id].latDeg, self.node[vtp_id].lonDeg]) -
                        np.array([self.node[other_vtp_id].latDeg, self.node[other_vtp_id].lonDeg])
                    )
                    if distance <= drone_range:
                        reachable_vtp_count += 1
            # 归一化：可达VTP节点数量除以总可能的连接数
            centrality_score = reachable_vtp_count / (total_vtp_nodes - 1) if total_vtp_nodes > 1 else 0
            all_scores['centrality'].append(centrality_score)
                        
            # 计算与客户节点的距离
            customer_distances = []
            for air_node_id in self.A_c:
                dist = np.linalg.norm(
                    np.array([self.node[vtp_id].latDeg, self.node[vtp_id].lonDeg]) -
                    np.array([self.node[air_node_id].latDeg, self.node[air_node_id].lonDeg])
                )
                customer_distances.append(dist)
            customer_proximity_score = 1 / (np.mean(customer_distances) + 1e-10)
            all_scores['customer_proximity'].append(customer_proximity_score)
            # 计算与仓库的proximity
            depot_distance = np.linalg.norm(
                np.array([self.node[vtp_id].latDeg, self.node[vtp_id].lonDeg]) -
                np.array([self.node[self.depot_id].latDeg, self.node[self.depot_id].lonDeg])
            )
            depot_proximity_score = 1 / (depot_distance + 1e-10)
            all_scores['depot_proximity'].append(depot_proximity_score)
            # 存储初步计算的分数
            importance_scores[vtp_id] = {
                'connectivity': connectivity_score,
                'centrality': centrality_score,
                'customer_proximity': customer_proximity_score,
                'depot_proximity': depot_proximity_score
            }
        # 3. 归一化所有分数
        for metric in weights.keys():
            max_val = max(all_scores[metric])
            min_val = min(all_scores[metric])
            range_val = max_val - min_val
            
            if range_val > 0:
                for vtp_id in importance_scores:
                    importance_scores[vtp_id][metric] = (importance_scores[vtp_id][metric] - min_val) / range_val
            else:
                # 如果所有值相同，则归一化为1
                for vtp_id in importance_scores:
                    importance_scores[vtp_id][metric] = 1.0
        
        # 4. 计算加权总分
        final_scores = {}
        for vtp_id in importance_scores:
            final_scores[vtp_id] = sum(importance_scores[vtp_id][metric] * weights[metric] for metric in weights)
        
        return final_scores
    
    def _redistribute_important_nodes(self, base_solutions, vtp_scores, insertion_prob_base=0.7):
        """基于重要性重新分配节点"""
        # new_routes = [route.copy() for route in base_solutions]
        diversified_solutions = []
        # 找出最大分数用于归一化
        max_score = max(vtp_scores.values()) if vtp_scores else 1.0
        for solution in base_solutions:
        # 创建解决方案的副本进行修改
            new_solution =[]
            add_solution = []
            tsp_solution = {}
            vehicle_id = 0  
            # 找出当前解决方案中已存在的所有VTP节点
            existing_vtps = set()
            for route in solution:
                current_route = route.copy()
                # 删除仓库节点
                current_route.pop(0)
                current_route.pop()
                existing_vtps.update(node for node in route if node in vtp_scores)
            
                # 获取未包含在任何路径中的VTP节点
                missing_vtps = [vtp for vtp in vtp_scores if vtp not in existing_vtps]
                
                # 按照评分从高到低排序未包含的VTP节点
                missing_vtps.sort(key=lambda x: vtp_scores[x], reverse=True)
        
                # 尝试插入每个缺失的VTP节点
                for vtp in missing_vtps:
                    # 根据归一化评分计算插入概率
                    normalized_score = vtp_scores[vtp] / max_score
                    insertion_prob = insertion_prob_base * normalized_score
                    
                    if random.random() < insertion_prob:
                        # 从当前路线中插入VTP节点
                        if current_route:  # 确保有可用路径
                            current_route.append(vtp)
                new_solution.append(current_route)
                tsp_solution[vehicle_id] = current_route
                vehicle_id += 1
            add_solution = self._solve_tsp_for_clusters(tsp_solution)
            # self.plot_vehicle_routes(add_solution)
            diversified_solutions.append(add_solution)
        return diversified_solutions
        
    def _find_best_insertion_position(self, route, vtp):
        """找到最佳插入位置"""
        if not route:
            return 0
        
        min_increase = float('inf')
        best_pos = 0
        
        vtp_pos = np.array([self.node[vtp].latDeg, self.node[vtp].lonDeg])
        
        for pos in range(len(route) + 1):
            # 计算插入后的距离增加
            if pos == 0:
                # 插入在开头
                if route:
                    first_pos = np.array([self.node[route[0]].latDeg, self.node[route[0]].lonDeg])
                    increase = (np.linalg.norm(self.depot_pos - vtp_pos) + 
                              np.linalg.norm(vtp_pos - first_pos) - 
                              np.linalg.norm(self.depot_pos - first_pos))
                else:
                    increase = np.linalg.norm(self.depot_pos - vtp_pos)
            elif pos == len(route):
                # 插入在末尾
                last_pos = np.array([self.node[route[-1]].latDeg, self.node[route[-1]].lonDeg])
                increase = (np.linalg.norm(last_pos - vtp_pos) + 
                          np.linalg.norm(vtp_pos - self.depot_pos) - 
                          np.linalg.norm(last_pos - self.depot_pos))
            else:
                # 插入在中间
                prev_pos = np.array([self.node[route[pos-1]].latDeg, self.node[route[pos-1]].lonDeg])
                next_pos = np.array([self.node[route[pos]].latDeg, self.node[route[pos]].lonDeg])
                increase = (np.linalg.norm(prev_pos - vtp_pos) + 
                          np.linalg.norm(vtp_pos - next_pos) - 
                          np.linalg.norm(prev_pos - next_pos))
            
            if increase < min_increase:
                min_increase = increase
                best_pos = pos
        
        return best_pos
    
    def _apply_local_search(self, routes):
        """应用局部搜索"""
        new_routes = [route.copy() for route in routes]
        
        # 随机选择一种操作
        operations = ['swap_within', 'swap_between', 'relocate', '2-opt']
        operation = random.choice(operations)
        
        if operation == 'swap_within':
            # 路径内交换
            truck_idx = random.randint(0, self.num_trucks - 1)
            if len(new_routes[truck_idx]) >= 2:
                i, j = random.sample(range(len(new_routes[truck_idx])), 2)
                new_routes[truck_idx][i], new_routes[truck_idx][j] = \
                    new_routes[truck_idx][j], new_routes[truck_idx][i]
        
        elif operation == 'swap_between':
            # 路径间交换
            if self.num_trucks >= 2:
                truck1, truck2 = random.sample(range(self.num_trucks), 2)
                if new_routes[truck1] and new_routes[truck2]:
                    i = random.randint(0, len(new_routes[truck1]) - 1)
                    j = random.randint(0, len(new_routes[truck2]) - 1)
                    new_routes[truck1][i], new_routes[truck2][j] = \
                        new_routes[truck2][j], new_routes[truck1][i]
        
        elif operation == 'relocate':
            # 重定位节点
            # 找到非空路径
            non_empty = [i for i in range(self.num_trucks) if len(new_routes[i]) > 0]
            if non_empty:
                src_truck = random.choice(non_empty)
                if len(new_routes[src_truck]) > 0:
                    node_idx = random.randint(0, len(new_routes[src_truck]) - 1)
                    node = new_routes[src_truck].pop(node_idx)
                    
                    # 选择目标路径
                    dst_truck = random.randint(0, self.num_trucks - 1)
                    if dst_truck == src_truck and new_routes[dst_truck]:
                        # 在同一路径内重定位
                        new_pos = random.randint(0, len(new_routes[dst_truck]))
                        new_routes[dst_truck].insert(new_pos, node)
                    else:
                        # 移到另一路径
                        pos = self._find_best_insertion_position(new_routes[dst_truck], node)
                        new_routes[dst_truck].insert(pos, node)
        
        elif operation == '2-opt':
            # 2-opt优化
            truck_idx = random.randint(0, self.num_trucks - 1)
            if len(new_routes[truck_idx]) >= 4:
                i = random.randint(0, len(new_routes[truck_idx]) - 3)
                j = random.randint(i + 2, len(new_routes[truck_idx]) - 1)
                new_routes[truck_idx][i+1:j+1] = new_routes[truck_idx][i+1:j+1][::-1]
        
        return new_routes
    
    def _create_empty_solution(self):
        """创建空解"""
        return [[] for _ in range(self.num_trucks)]
    
    def _create_random_solution(self):
        """创建随机解"""
        routes = [[] for _ in range(self.num_trucks)]
        remaining = self.vtp_indices.copy()
        random.shuffle(remaining)
        
        for i, vtp in enumerate(remaining):
            truck_idx = i % self.num_trucks
            routes[truck_idx].append(vtp)
        
        return routes
    
    def select_best_solution(self, solutions):
        """选择最佳解"""
        best_solution = None
        best_cost = float('inf')
        
        for solution in solutions:
            cost = self._evaluate_solution(solution)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
        
        return best_solution if best_solution is not None else self._create_empty_solution()
    
    def _evaluate_solution(self, solution):
        """评估解的质量"""
        total_cost = 0
        
        for route in solution:
            if not route:
                continue
            
            # 计算路径距离
            route_cost = 0
            
            # 从仓库到第一个节点
            if route:
                first_pos = np.array([self.node[route[0]].latDeg, self.node[route[0]].lonDeg])
                route_cost += np.linalg.norm(self.depot_pos - first_pos)
            
            # 路径中的距离
            for i in range(len(route) - 1):
                pos1 = np.array([self.node[route[i]].latDeg, self.node[route[i]].lonDeg])
                pos2 = np.array([self.node[route[i+1]].latDeg, self.node[route[i+1]].lonDeg])
                route_cost += np.linalg.norm(pos1 - pos2)
            
            # 从最后一个节点回到仓库
            if route:
                last_pos = np.array([self.node[route[-1]].latDeg, self.node[route[-1]].lonDeg])
                route_cost += np.linalg.norm(last_pos - self.depot_pos)
            
            total_cost += route_cost
        
        # 添加平衡性惩罚
        route_lengths = [len(route) for route in solution]
        if route_lengths:
            balance_penalty = np.std(route_lengths) * 10
            total_cost += balance_penalty
        
        return total_cost
    

def make_dict():# 设计实现一个可以无线嵌套的dict
	return defaultdict(make_dict)
    








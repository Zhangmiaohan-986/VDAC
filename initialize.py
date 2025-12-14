# 该文件通过类初始化车辆和无人机的各操作，包括接受，状态，任务，路径，等待时间，成本等
import numpy as np
import networkx as nx
import copy

def init_agent(node, vehicle, V, uav_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, xP, sigmaprime, sL, sR):
	trucks = []
	drones = []
	for vehicle_id, vehicle_type in vehicle.items():
		if vehicle_type == 'truck':
			trucks.append(Vehicle(vehicle_id, vehicle[vehicle_id].capacity, vehicle[vehicle_id].start_node_id))
		else:
			drones.append(Drone(vehicle_id, vehicle[vehicle_id].endurance, vehicle[vehicle_id].task_speed, vehicle[vehicle_id].speed))
    
	
	return trucks, drones

def initialize_drone_vehicle_assignments(vehicle, V, T):
    """
    初始化车辆携带无人机的关系，均匀分配无人机给车辆
    
    参数:
    vehicle - 车辆和无人机字典，键为ID
    V - 无人机ID列表
    T - 车辆ID列表
    
    返回:
    更新后的vehicle字典
    """
    num_drones = len(V)
    num_vehicles = len(T)
    
    print(f"分配 {num_drones} 架无人机给 {num_vehicles} 辆车辆")
    
    # 计算基本分配数量和余数
    drones_per_vehicle = num_drones // num_vehicles
    remaining_drones = num_drones % num_vehicles
    
    # 初始化每辆车的分配数量
    vehicle_assignments = {vehicle_id: drones_per_vehicle for vehicle_id in T}
    
    # 处理剩余的无人机（分配给前remaining_drones辆车）
    for i in range(remaining_drones):
        vehicle_assignments[T[i]] += 1
    
    # 打印分配情况
    for vehicle_id in T:
        print(f"车辆 {vehicle_id} 将被分配 {vehicle_assignments[vehicle_id]} 架无人机")
    
    # 执行分配
    drone_index = 0
    for vehicle_id in T:
        drones_to_assign = vehicle_assignments[vehicle_id]
        for _ in range(drones_to_assign):
            if drone_index < len(V):
                drone_id = V[drone_index]
                # 添加无人机到车辆
                vehicle[vehicle_id].add_drone(vehicle[drone_id])
                # 更新无人机状态
                vehicle[drone_id].is_available = True
                print(f"已将无人机 {drone_id} 分配给车辆 {vehicle_id}")
                drone_index += 1
            else:
                break
    
    # 统计分配结果
    for vehicle_id in T:
        num_assigned = len(vehicle[vehicle_id].drones)
        print(f"车辆 {vehicle_id} 现在携带 {num_assigned} 架无人机: {list(vehicle[vehicle_id].drones.keys())}")
    
    return vehicle


class Vehicle:
    def __init__(self, vehicle_id, capacity, start_node_id, speed=60):
        """
        初始化车辆
        :param vehicle_id: 车辆编号
        :param capacity: 车辆容量（携带无人机数量）
        :param start_node_id: 车辆起始节点编号
        """
        self.id = vehicle_id
        self.capacity = capacity
        self.route = []  # 车辆路径，存储节点编号的列表eeee
        self.real_route = []  # 车辆实际路径，存储节点编号的列表
        self.drones = []  # 车辆携带的无人机列表
        self.start_node_id = start_node_id
        self.speed = speed  # 车辆速度为12.5km/h
        self.work_time = 3/60  # 辅助收放电时间为3min-转变为小时
        self.completion_time = 0  # 完成时间
        self.type = 'vehicle'
        self.launch_records = []  # 发射记录，键为节点编号，值为发射的无人机列表
        self.recover_records = []  # 回收记录，键为节点编号，值为回收的无人机列表
        self.launch_recover = {}  # 同时起降并发射，即原地停靠策略产生的情况
        self.current_time = 0  # 记录当前时间
        self.current_node_index = 0  # 当前路径索引
        self.node_entry_time = {}  # 记录进入每个节点的时间
        self.node_entry_exit_time = {}  # 记录离开每个节点的时间
        self.drone_capacity = capacity  # 当前可用的无人机数量
        # self.vehicle_available_drones = self.drones.copy()
        self.vehicle_available_drones = copy.deepcopy(self.drones)
        self.route_refresh = False  # 路径是否刷新
        self.current_node = None  # 车辆实时全局节点
        self.current_time = 0  # 车辆的实时全局时间
        self.end_time = 0  # 车辆离开节点的时间
        self.available_drones = self.drones.copy()  # 可用的无人机列表
        self.available_capacity = self.capacity
        self.task_finish = False
        self.mission_route = []  # 任务路径
        # self.launch_record = {}

        # 新增属性
        self.waiting_times = {}  # 键：节点ID，值：在该节点等待的总时间
    def update_entry_exit_time(self, node_id, entry_time, exit_time):  # 记录vehicle进入每个节点的时间
        self.node_entry_time[node_id] = entry_time
        self.node_exit_time[node_id] = exit_time

    def can_enter_node(self, node_id, global_time, other_vehicles):
        """
        检查在 global_time 时刻，是否可以进入节点 node_id
        :param node_id: 节点编号
        :param global_time: 全局时间
        :param other_vehicles: 其他车辆的列表
        :return: True 或 False
        """
        for vehicle in other_vehicles:
            if node_id in vehicle.node_entry_time:
                if vehicle.node_entry_time[node_id] <= global_time < vehicle.node_exit_time[node_id]:
                    return False
        return True

    def launch_drone(self, drone, current_node):
        """
        发射无人机，更新无人机和车辆的状态
        :param drone: Drone 对象
        :param current_node: 当前节点编号
        """
        if self.drone_capacity > 0:
            # 更新无人机状态
            drone.is_available = False
            drone.current_node = current_node
            drone.start_time = self.current_time + self.work_time  # 考虑发射时间
            # 更新车辆状态
            self.drone_capacity -= 1
            if current_node not in self.launch:
                self.launch[current_node] = []
            self.launch[current_node].append(drone.id)
            # 更新车辆时间
            self.current_time += self.work_time
            return True
        else:
            return False

    def recover_drone(self, drone, current_node):
        """
        回收无人机，更新无人机和车辆的状态
        :param drone: Drone 对象
        :param current_node: 当前节点编号
        """
        # 更新无人机状态
        drone.is_available = True
        drone.current_node = current_node
        drone.end_time = self.current_time + self.work_time  # 考虑回收时间
        # 更新车辆状态
        self.drone_capacity += 1
        if current_node not in self.recover:
            self.recover[current_node] = []
        self.recover[current_node].append(drone.id)
        # 更新车辆时间
        self.current_time += self.work_time

    def add_drone(self, drone):
        """
        添加无人机到车辆
        :param drone: Drone对象
        """
        self.drones.append(drone)
        # self.drones[drone.id] = drone\

    def set_route(self, route):
        """
        设置车辆路径
        :param route: 节点编号列表
        """
        self.route = route.copy()

    def add_node_to_route(self, node_id):
        """
        向车辆路径中添加节点
        :param node_id: 节点编号
        """
        self.route.append(node_id)

    def encode_route(self):
        """
        编码车辆路径
        :return: 编码后的路径
        """
        return self.route.copy()

    def decode_route(self, encoded_route):
        """
        解码车辆路径
        :param encoded_route: 编码后的路径
        """
        self.route = encoded_route.copy()

    def get_completion_time(self, ground_graph):
        """
        计算车辆完成任务的时间
        :param ground_graph: 地面网络Graph对象
        :return: 完成时间
        """
        total_distance = 0
        for i in range(len(self.route) - 1):
            node1 = self.route[i]
            node2 = self.route[i + 1]
            edge = ground_graph.edges.get((node1, node2))
            if edge:
                total_distance += edge
            else:
                # 如果两个节点之间没有边，假设距离为无穷大
                total_distance += np.inf
        self.completion_time = total_distance
        return self.completion_time

class Drone:
    def __init__(self, drone_id, endurance=0.77, task_speed=36, speed=61.2):  # 所有无人机的状态参数，按km/h换算。
        """
        初始化无人机
        :param drone_id: 无人机编号
        :param endurance: 无人机续航时间（距离）
        :param speed: 无人机速度（距离/时间）
        """
        self.id = drone_id
        self.endurance = endurance
        self.task_speed = task_speed
        self.speed = speed
        self.route = []  # 无人机路径，键为发射，降落节点编号，值为路径
        self.time = []
        self.completion_time = 0  # 完成时间
        self.type = 'drone'
        self.launch = {}
        self.recover = {}
        self.launch_recover = {}
        self.is_available = True
        self.current_node = None
        self.start_time = 0  # 发射时间
        self.end_time = 0  # 回收时间
        self.task_finish = False
        self.remaining_endurance = endurance
        self.current_node = None
        self.assigned_vehicle = None
        self.launch_time = None
        self.recover_time = None
        self.task_assigned = None
        self.current_time = 0
        self.need_back = False
        self.returnable_nodes = []  # 可返回的节点列表
        self.inspection_nodes = []  # 巡检节点集合
        self.inspection_time = 0  # 巡检总时间
        self.inspection_nodes_index = 0  # 无人机当前巡检节点索引
        self.violate = False  # 判断是否违背了约束条件
        self.recover_vehicle = None
        self.no_task = False

    def assign_task(self, path):
        """
        分配巡检任务
        :param path: 巡检路径（节点编号列表）
        """
        self.route = path.copy()
        self.is_available = False

    def update_status(self, time_spent):
        """
        更新无人机状态
        :param time_spent: 花费的时间
        """
        self.remaining_time -= time_spent
        if self.remaining_time <= 0:
            self.is_available = False  # 无人机需要回收

    def set_route(self, route):
        """
        设置无人机路径
        :param route: 节点编号列表
        """
        self.route = route.copy()

    def add_node_to_route(self, node_id):
        """
        向无人机路径中添加节点
        :param node_id: 节点编号
        """
        self.route.append(node_id)

    def encode_route(self):
        """
        编码无人机路径
        :return: 编码后的路径
        """
        return self.route.copy()

    def decode_route(self, encoded_route):
        """
        解码无人机路径
        :param encoded_route: 编码后的路径
        """
        self.route = encoded_route.copy()

    def get_completion_time(self, air_graph):
        """
        计算无人机完成任务的时间
        :param air_graph: 空中网络Graph对象
        :return: 完成时间
        """
        total_distance = 0
        for i in range(len(self.route) - 1):
            node1 = self.route[i]
            node2 = self.route[i + 1]
            edge = air_graph.edges.get((node1, node2))
            if edge:
                total_distance += edge
            else:
                # 如果两个节点之间没有边，假设距离为无穷大
                total_distance += np.inf
        self.completion_time = total_distance / self.speed
        return self.completion_time
    
from collections import defaultdict

from collections import defaultdict
import copy


# 假设 make_dict 函数在当前作用域内可用
def make_dict():
    return defaultdict(make_dict)


def deep_copy_vehicle_task_data(original_data):
    """
    极致优化的嵌套结构复制。
    利用原生的 .copy() 保留 defaultdict 类型和 factory，
    仅手动处理最底层的 vehicle_task 对象隔离。
    """
    # 1. 复制最外层容器 
    # .copy() 会自动保留 defaultdict 的 default_factory，不需要手动提取
    copied_data = original_data.copy()
    
    # 2. 遍历外层字典
    for vehicle_id, inner_dict in copied_data.items():
        
        # 3. 复制内层容器 (同样保留了类型和 factory)
        # 如果 inner_dict 是 defaultdict，.copy() 后依然是，且 factory 一致
        # 如果 inner_dict 是普通 dict，.copy() 后依然是普通 dict
        new_inner = inner_dict.copy()
        
        # 4. 遍历内层，将 value 替换为全新的对象
        # 这一步确保了 task 对象不指向同一内存地址
        for node_id, task_object in new_inner.items():
            # 调用之前优化过的 vehicle_task.fast_copy()
            new_inner[node_id] = task_object.fast_copy()
            
        # 5. 将处理好的新内层字典回写到外层
        copied_data[vehicle_id] = new_inner
        
    return copied_data
# def deep_copy_vehicle_task_data(original_data):
#     """
#     快速、高效地复制嵌套的 vehicle_task_data 结构。
#     """
#     # 1. 创建一个新的顶层 defaultdict，使用与原始对象相同的工厂函数
#     copied_data = defaultdict(original_data.default_factory)
    
#     # 2. 遍历外层字典 (vehicle_id -> inner_dict)
#     for vehicle_id, inner_dict in original_data.items():
        
#         # # 3. 为每个 vehicle_id 创建一个新的内层 defaultdict
#         # new_inner_dict = defaultdict(inner_dict.default_factory)
#         # 检查是否是defaultdict
#         if isinstance(inner_dict, defaultdict):
#             new_inner_dict = defaultdict(inner_dict.default_factory)
#         else:
#             # 如果是普通dict，创建一个新的defaultdict
#             new_inner_dict = defaultdict(lambda: None)
        
#         # 4. 遍历内层字典 (node_id -> vehicle_task object)
#         for node_id, task_object in inner_dict.items():
#             # 5. 使用我们定义的 fast_copy 方法来复制 vehicle_task 对象
#             new_inner_dict[node_id] = task_object.fast_copy()
            
#         # 6. 将新创建的内层字典赋值给顶层字典
#         copied_data[vehicle_id] = new_inner_dict
        
#     return copied_data


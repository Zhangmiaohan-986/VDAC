import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import copy
from collections import defaultdict

NODE_TYPE_DEPOT	= 0
NODE_TYPE_CUST	= 1

TYPE_TRUCK 		= 1
TYPE_UAV 		= 2

NUM_POINTS = 50
SEED = 6
Z_COORD = 5
UAV_DISTANCE = 15

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

class Task:
    """简单的任务类"""
    def __init__(self, task_type, start_time, end_time, details=None):
        self.task_type = task_type
        self.start_time = start_time
        self.end_time = end_time
        self.details = details or {}
    
    def __str__(self):
        task_name = TASK_NAMES.get(self.task_type, "未知任务")
        return f"{task_name} ({self.start_time:.2f}-{self.end_time:.2f})"
    
    def to_dict(self):
        """将任务转换为字典"""
        return {
            'task_type': self.task_type,
            'task_name': TASK_NAMES.get(self.task_type, "未知任务"),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time,
            'details': self.details
        }
    
    def copy(self):
        """创建任务的副本"""
        return Task(
            task_type=self.task_type,
            start_time=self.start_time,
            end_time=self.end_time,
            details=self.details.copy() if self.details else None
        )
    
class VehicleInfo:
    def __init__(self):
        self.drone_belong = None
        self.precise_arrive_time = 0
        self.precise_departure_time = 0
        self.launch_time = []
        self.recovery_time = []
        # 假设 Task 类有 .copy() 方法
        self.task = {}
        # 【【【新增】】】为 VehicleInfo 添加 copy 方法
    def copy(self):
        new_info = VehicleInfo()
        new_info.drone_belong = self.drone_belong
        new_info.precise_arrive_time = self.precise_arrive_time
        new_info.precise_departure_time = self.precise_departure_time
        new_info.launch_time = self.launch_time.copy()
        new_info.recovery_time = self.recovery_time.copy()
        # 假设 self.task 的值（可能是Task对象）也有 .copy() 方法
        new_info.task = {k: v.copy() for k, v in self.task.items()}
        return new_info 

def create_vehicle_info_dict():
    """
    代替 VehicleInfo 类的构造函数。
    返回一个用于存储无人机-车辆交互信息的标准字典。
    """
    return {
        "drone_belong": None,
        "prcise_arrive_time": 0,
        "prcise_departure_time": 0,
        "arrive_times": [],
        "departure_times": [],
        "task": {}  # 注意：这里也需要一个自定义的 add_task 逻辑
    }

def copy_vehicle_info_dict(d):
    """一个专门用于复制 vehicle_info 字典的函数"""
    # 假设 Task 对象也有 .copy() 方法
    # tasks_copy = {node_id: [task.copy() for task in task_list] 
    #               for node_id, task_list in d["task"].items()}
                  
    return {
        "drone_belong": d["drone_belong"],
        "precise_arrive_time": d["precise_arrive_time"],
        "precise_departure_time": d["precise_departure_time"],
        "arrive_times": d["arrive_times"].copy(),          # 浅拷贝列表
        "departure_times": d["departure_times"].copy(),      # 浅拷贝列表
        "task": []  # 使用上面创建的任务深拷贝
    }

class vehicle_task:
    def __init__(self, id, vehicleType, node_id, node):
        self.vehicleType = vehicleType
        self.id = id  # 车辆ID
        self.node = node
        self.node_id = node_id
        self.tasks = {}  # 节点ID -> 任务列表
        self.arrive_times = []  # 节点ID -> 到达时间
        self.departure_times = []  # 节点ID -> 离开时间
        self.prcise_arrive_time = None
        self.prcise_departure_time = None
        self.dict_vehicle = {}
        if node[node_id].nodeType == 'DEPOT':
            self.is_task = True  # 是否真实执行了任务
        else:
            self.is_task = False
        self.drone_list = None
        self.launch_drone_list = None
        self.recovery_drone_list = None
        self.drone_belong = None
        if self.vehicleType == TYPE_TRUCK:
            # 初始化携带无人机的列表
            self.drone_list = []
            self.launch_drone_list = []  # 初始化在当前节点发射无人机的列表
            self.recovery_drone_list = []  # 初始化在当前节点回收无人机的列表
        else:
            # 初始化无人机在当前位置归属的车辆
            self.drone_belong = None
            # 新增：用于存储无人机在不同车辆上的信息的字典
            self.dict_vehicle = {}
            # 为每个无人机初始化dict_vehicle
            self.init_dict_vehicle()

    def init_dict_vehicle(self):
        """初始化无人机的dict_vehicle"""
        # 为每个可能的车辆ID创建一个预先定义好的 VehicleInfo 类的实例
        for vehicle_id in range(1, 4):  # 注意：这里硬编码了车辆数量，可能需要优化
            self.dict_vehicle[vehicle_id] = create_vehicle_info_dict()
            # self.dict_vehicle[vehicle_id] = VehicleInfo()

    # 更新车辆携带无人机列表
    def update_drone_list(self, drone_id):
        # 判断是否为列表
        if  not isinstance(drone_id, list):
            self.drone_list.append(drone_id)
        else:
            self.drone_list.extend(drone_id)

    # 删除无人机携带的无人机列表
    def delete_drone_list(self, drone_id):
        if not isinstance(drone_id, list):
            self.drone_list.remove(drone_id)
        else:
            for i in drone_id:
                self.drone_list.remove(i)

    # 更新无人机在当前位置归属的车辆
    def update_drone_belong(self, drone_id, vehicle_id):
        if drone_id == self.id:
            self.drone_belong = vehicle_id

    # 删除无人机在当前位置归属的车辆
    def delete_drone_belong_to_vehicle(self, drone_id):
        if not isinstance(drone_id, list):
            self.drone_belong_to_vehicle.pop(drone_id)
        else:
            for i in drone_id:
                self.drone_belong_to_vehicle.pop(i)

    def add_node(self, node_id, arrive_time=0, departure_time=0):
        """添加节点访问记录"""
        self.tasks[node_id] = []  # 初始化该节点的任务列表
        self.arrive_times.append(arrive_time)
        self.departure_times.append(departure_time)
        
    def add_task(self, node_id, task_type, start_time, end_time, details=None):
        """在指定节点添加任务"""
        if node_id not in self.tasks:
            self.add_node(node_id)
            
        task = Task(task_type, start_time, end_time, details)
        self.tasks[node_id].append(task)
        return task
    
    def get_node_tasks(self, node_id):
        """获取指定节点的所有任务"""
        return self.tasks.get(node_id, [])
    
    def set_arrive_time(self, time):
        """设置到达时间"""
        self.arrive_times.append(time)
        
    def set_departure_time(self, time):
        """设置离开时间"""
        self.departure_times.append(time)
    
    # 查找任务状况
    # 你可以把这个方法添加到你的 vehicle_task 类里面
    def find_task(self, node_id, task_type):
        """
        在指定节点查找特定类型的第一个任务。

        Args:
            node_id (int): 要搜索的节点ID。
            task_type (int): 要搜索的任务类型常量 (例如 TASK_DRONE_RECOVERY)。

        Returns:
            Task: 如果找到，返回第一个匹配的 Task 对象。
            None: 如果没有找到。
        """
        # 使用 .get(node_id, []) 可以安全地处理节点不存在的情况，返回一个空列表而不是报错
        tasks_at_node = self.tasks.get(node_id, [])
        
        # 遍历该节点的所有任务
        for task in tasks_at_node:
            if task.task_type == task_type:
                return task  # 找到后立即返回该任务对象
                
        return None # 如果循环结束都没找到，返回 None
    
    # 删除任务状况
    def delete_task(self, node_id, task_type):
        """
        删除指定节点上特定类型的第一个任务。

        Args:
            node_id (int): 任务所在的节点ID。
            task_type (int): 要删除的任务类型。

        Returns:
            bool: 如果成功删除了一个任务，返回 True，否则返回 False。
        """
        if node_id not in self.tasks:
            return False # 如果节点本身都不存在，直接返回False

        original_tasks = self.tasks[node_id]
        
        # 找出要删除的任务的第一个实例
        task_to_delete = None
        for task in original_tasks:
            if task.task_type == task_type:
                task_to_delete = task
                break # 找到第一个就停止

        if task_to_delete:
            # 从原始列表中移除找到的任务对象
            original_tasks.remove(task_to_delete)
            return True # 成功删除
            
        return False # 没有找到可删除的任务

    def to_dict(self):
        """将整个车辆任务对象转换为字典"""
        result = {
            'vehicle_id': self.id,
            'vehicle_type': self.vehicleType,
            'nodes': {}
        }
        
        for node_id in self.tasks:
            node_dict = {
                'arrive_time': self.arrive_times.get(node_id, 0),
                'departure_time': self.departure_times.get(node_id, 0),
                'tasks': [task.to_dict() for task in self.tasks[node_id]]
            }
            result['nodes'][str(node_id)] = node_dict
            
        return result
    
    def fast_copy(self):
        """
        创建一个 vehicle_task 对象的快速、独立的副本。
        - 不可变属性直接赋值。
        - 共享数据（如 self.node）保持引用。
        - 实例独有的可变属性（列表、字典）创建新的浅拷贝。
        """
        # 1. 创建一个新实例，传递初始的不可变或共享的参数
        new_task = vehicle_task(self.id, self.vehicleType, self.node_id, self.node)

        # 2. 复制简单的、实例独有的属性
        new_task.is_task = self.is_task
        
        # 3. 复制可变的字典和列表属性
        # new_task.tasks = {node_id: task_list.copy() for node_id, task_list in self.tasks.items()}
        new_task.tasks = {node_id: [task.copy() for task in task_list] 
                          for node_id, task_list in self.tasks.items()}
        new_task.arrive_times = self.arrive_times.copy()
        new_task.departure_times = self.departure_times.copy()
        # new_task.arrive_times = self.arrive_times.copy()
        # new_task.departure_times = self.departure_times.copy()

        # 4. 根据车辆类型，安全地复制特定属性
        if self.vehicleType == TYPE_TRUCK:
            new_task.drone_list = self.drone_list.copy()
            new_task.launch_drone_list = self.launch_drone_list.copy()
            new_task.recovery_drone_list = self.recovery_drone_list.copy()
        
        elif self.vehicleType == TYPE_UAV:
            new_task.drone_belong = self.drone_belong
            # # 复制每个VehicleInfo实例到新对象的dict_vehicle中
            # for vehicle_id, vehicle_info in self.dict_vehicle.items():
            #     new_task.dict_vehicle[vehicle_id] = vehicle_info.copy()
                    # 【重要】复制 dict_vehicle 字典
            # new_task.dict_vehicle = {
            #     vehicle_id: vehicle_info.copy() 
            #     for vehicle_id, vehicle_info in self.dict_vehicle.items()
            # }
            new_task.dict_vehicle = {v_id: copy_vehicle_info_dict(info_dict) 
                                    for v_id, info_dict in self.dict_vehicle.items()}
            

        return new_task

    # def fast_copy(self):
    #     """
    #     创建一个 vehicle_task 对象的快速、独立的副本。
    #     - 不可变属性直接赋值。
    #     - 共享数据（如 self.node）保持引用。
    #     - 实例独有的可变属性（列表、字典）创建新的浅拷贝。
    #     """
    #     # 1. 创建一个新实例，传递初始的不可变或共享的参数
    #     new_task = vehicle_task(self.id, self.vehicleType, self.node_id, self.node)

    #     # 2. 复制简单的、实例独有的属性
    #     new_task.is_task = self.is_task
        
    #     # 3. 复制可变的字典和列表属性
    #     #    - tasks 是一个 {node_id: [task_obj1, ...]} 结构
    #     #    - 我们复制字典，并复制每个节点下的任务列表
    #     #    - 假设 Task 对象本身不需要深拷贝，这通常是安全的
    #     new_task.tasks = {node_id: task_list.copy() for node_id, task_list in self.tasks.items()}
        
    #     new_task.arrive_times = self.arrive_times.copy()
    #     new_task.departure_times = self.departure_times.copy()

    #     # 4. 根据车辆类型，安全地复制特定属性
    #     if self.vehicleType == TYPE_TRUCK:
    #         # self.drone_list 等属性在 TRUCK 类型的对象上保证存在
    #         new_task.drone_list = self.drone_list.copy()
    #         new_task.launch_drone_list = self.launch_drone_list.copy()
    #         new_task.recovery_drone_list = self.recovery_drone_list.copy()
        
    #     elif self.vehicleType == TYPE_UAV:
    #         # self.drone_belong 属性在 UAV 类型的对象上保证存在
    #         new_task.drone_belong = self.drone_belong

    #     return new_task
    
    def print_tasks(self):
        """打印车辆任务详细信息"""
        print(f"\n===== 车辆 ID: {self.id} ({self.vehicleType}) 任务报告 =====")
        
        if not self.tasks:
            print("该车辆没有分配任务")
            return
            
        for node_id in sorted(self.tasks.keys()):
            print(f"\n节点 {node_id}:")
            print(f"  到达时间: {self.arrive_times.get(node_id, 0):.2f}")
            print(f"  离开时间: {self.departure_times.get(node_id, 0):.2f}")
            
            if not self.tasks[node_id]:
                print("  无任务执行")
                continue
                
            print("  任务列表:")
            for i, task in enumerate(sorted(self.tasks[node_id], key=lambda t: t.start_time), 1):
                task_name = TASK_NAMES.get(task.task_type, "未知任务")
                print(f"    {i}. {task_name} ({task.start_time:.2f} → {task.end_time:.2f})")
                
                if task.details:
                    details_str = ", ".join(f"{k}={v}" for k, v in task.details.items())
                    print(f"       详情: {details_str}")


def find_belong_vehicle_id(vehicle, uav, node_id): # 输入的vehicle是task_data数据
    for vehicle_id in vehicle:
        drone_list = list(vehicle[vehicle_id][node_id].drone_list)
        if uav.id in drone_list:
            return vehicle_id
    print(f"无人机 {uav.id} 在节点 {node_id} 没有归属车辆")
    return None

# 根据车辆路径任务，更新车辆和无人机在各个节点的状态,最后返回更新后的所有车辆在各个节点状态变化，以及无人机在各个节点的状态变化，同时将修改后的精细化时间输出
# def update_vehicle_task(vehicle_task_data, y, vehicle_route, uav_travel, vehicle_arrival_time, vehicle):
#     drone_id, vtp_i, c, vtp_j, v_id, recv_v_id = y
#     v_id_index = v_id - 1
#     recv_v_id_index = recv_v_id - 1
#     v_id_route = vehicle_route[v_id_index]
#     recv_v_id_route = vehicle_route[recv_v_id_index]
#     v_id_arrive_time = vehicle_arrival_time[v_id_index][vtp_i]
#     recv_v_id_arrive_time = vehicle_arrival_time[recv_v_id_index][vtp_j]
#     v_id_depart_time_list = vehicle_task_data[v_id][vtp_i].departure_times  # 获得车辆在该节点任务时间的节点列表
#     recv_v_id_depart_time_list = vehicle_task_data[recv_v_id][vtp_j].departure_times
#     uav_task_time = uav_travel[drone_id][vtp_i][c] + uav_travel[drone_id][c][vtp_j]
#     return vehicle_task_data


def remove_vehicle_task(vehicle_task_data, y, vehicle_route):
    drone_id, vtp_i, customer, vtp_j, v_id, recv_v_id = y
    veh_launch_index = v_id -1
    veh_recovery_index = recv_v_id -1
    update_vheicle_route = vehicle_route.copy()
    # 更新vehicle_task中车辆携带无人机状态更新
    if v_id == recv_v_id:
        task_route = vehicle_route[veh_launch_index]
        task_route_launch_index = task_route.index(vtp_i)
        task_route_recovery_index = task_route.index(vtp_j)
        remove_uav_route = task_route[task_route_launch_index:task_route_recovery_index+1]  # 包含了发射和回收的车辆路线
        for node_index, node in enumerate(remove_uav_route):
            if node_index == 0:  # 代表车辆在该点发射无人机
                vehicle_task_data[v_id][node].launch_drone_list.remove(drone_id)
                vehicle_task_data[v_id][node].drone_list.append(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
            elif node_index == len(remove_uav_route)-1:
                vehicle_task_data[v_id][node].recovery_drone_list.remove(drone_id)
                if drone_id not in vehicle_task_data[v_id][node].launch_drone_list:  # 关键判断约束，防止无人机无限制在该节点发射导致的报错
                    vehicle_task_data[v_id][node].drone_list.append(drone_id)
                    vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                    vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                else:
                    vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
            else:
                vehicle_task_data[v_id][node].drone_list.append(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
    else:
        task_launch_route = vehicle_route[veh_launch_index]
        task_recovery_route = vehicle_route[veh_recovery_index]
        task_launch_index = task_launch_route.index(vtp_i)
        task_recovery_index = task_recovery_route.index(vtp_j)
        remove_launch_uav_route = task_launch_route[task_launch_index:]  # 包含了发射和回收的车辆路线
        remove_recovery_uav_route = task_recovery_route[task_recovery_index:]  # 包含了发射和回收的车辆路线
        for node_index, node in enumerate(remove_launch_uav_route):
            if node_index == 0:
                vehicle_task_data[v_id][node].launch_drone_list.remove(drone_id)
                vehicle_task_data[v_id][node].drone_list.append(drone_id)
                vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                continue
            vehicle_carry_drone_list = vehicle_task_data[v_id][node].drone_list
            if drone_id not in vehicle_task_data[v_id][node].recovery_drone_list and drone_id not in vehicle_task_data[v_id][node].drone_list:
                vehicle_task_data[v_id][node].drone_list.append(drone_id)  # 判断车辆在节点上是否携带其型号无人机
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
            elif drone_id in vehicle_task_data[v_id][node].recovery_drone_list:
                # vehicle_task_data[v_id][node].recovery_drone_list.remove(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                break
        for node_index, node in enumerate(remove_recovery_uav_route):
            if node_index == 0:
                vehicle_task_data[recv_v_id][node].recovery_drone_list.remove(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = None
                # 判断这个节点是否有发射任务,若有发射任务，则回归列表不添加
                if drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list:    
                    vehicle_task_data[recv_v_id][node].drone_list.remove(drone_id)
                    vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                # continue
                elif drone_id in vehicle_task_data[recv_v_id][node].launch_drone_list:
                    vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                    break
            # vehicle_carry_drone_list = vehicle_task_data[recv_v_id][node].drone_list
            if drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list and drone_id in vehicle_task_data[recv_v_id][node].drone_list:
                vehicle_task_data[recv_v_id][node].drone_list.remove(drone_id)
                vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = None
            elif drone_id in vehicle_task_data[recv_v_id][node].launch_drone_list:
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                break

    return vehicle_task_data

def deep_remove_vehicle_task(vehicle_task_data, y, vehicle_route):
    drone_id, vtp_i, customer, vtp_j, v_id, recv_v_id = y
    veh_launch_index = v_id -1
    veh_recovery_index = recv_v_id -1
    update_vheicle_route = vehicle_route.copy()
    # 更新vehicle_task中车辆携带无人机状态更新
    if v_id == recv_v_id:
        task_route = vehicle_route[veh_launch_index]
        task_route_launch_index = task_route.index(vtp_i)
        task_route_recovery_index = task_route.index(vtp_j)
        remove_uav_route = task_route[task_route_launch_index:task_route_recovery_index+1]  # 包含了发射和回收的车辆路线
        for node_index, node in enumerate(remove_uav_route):
            if node_index == 0:  # 代表车辆在该点发射无人机
                vehicle_task_data[v_id][node].launch_drone_list.remove(drone_id)
                # vehicle_task_data[v_id][node].drone_list.append(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
            elif node_index == len(remove_uav_route)-1:
                vehicle_task_data[v_id][node].recovery_drone_list.remove(drone_id)
                if drone_id in vehicle_task_data[v_id][node].drone_list:
                    vehicle_task_data[v_id][node].drone_list.remove(drone_id)
                    vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                    vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
                # if drone_id not in vehicle_task_data[v_id][node].launch_drone_list:  # 关键判断约束，防止无人机无限制在该节点发射导致的报错
                #     vehicle_task_data[v_id][node].drone_list.append(drone_id)
                #     vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                #     vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                # else:
                #     vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
            else:
                # vehicle_task_data[v_id][node].drone_list.append(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
        # 获取同车发射后的所有节点
        rm_route_list = task_route[task_route_recovery_index:]
        is_launch_task = False
        # 查找是否存在发射任务
        for index, node in enumerate(rm_route_list):
            if index == 0:
                if v_id in vehicle_task_data[v_id][node].launch_drone_list:
                    is_launch_task = True
                    break
            else:
                if v_id in vehicle_task_data[v_id][node].launch_drone_list:
                    is_launch_task = True
                    break
        if not is_launch_task:  # 后续任务未删除
            for index, node in enumerate(rm_route_list):
                if index == 0:
                    continue
                else:
                    if drone_id in vehicle_task_data[v_id][node].drone_list:
                        vehicle_task_data[v_id][node].drone_list.remove(drone_id)
                        # vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                        vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
    else:
        task_launch_route = vehicle_route[veh_launch_index]
        task_recovery_route = vehicle_route[veh_recovery_index]
        task_launch_index = task_launch_route.index(vtp_i)
        task_recovery_index = task_recovery_route.index(vtp_j)
        remove_launch_uav_route = task_launch_route[task_launch_index:]  # 包含了发射和回收的车辆路线
        remove_recovery_uav_route = task_recovery_route[task_recovery_index:]  # 包含了发射和回收的车辆路线
        for node_index, node in enumerate(remove_launch_uav_route):
            if node_index == 0:
                vehicle_task_data[v_id][node].launch_drone_list.remove(drone_id)
                # vehicle_task_data[v_id][node].drone_list.append(drone_id)
                # vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
                continue
            vehicle_carry_drone_list = vehicle_task_data[v_id][node].drone_list
            if drone_id not in vehicle_task_data[v_id][node].recovery_drone_list and drone_id not in vehicle_task_data[v_id][node].drone_list:
                # vehicle_task_data[v_id][node].drone_list.append(drone_id)  # 判断车辆在节点上是否携带其型号无人机
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
            # if drone_id not in vehicle_task_data[v_id][node].launch_drone_list and drone_id in vehicle_task_data[v_id][node].drone_list:
            if drone_id not in vehicle_task_data[v_id][node].recovery_drone_list and drone_id in vehicle_task_data[v_id][node].drone_list:
                vehicle_task_data[v_id][node].drone_list.remove(drone_id)
                # vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
            if drone_id in vehicle_task_data[v_id][node].recovery_drone_list:
                # vehicle_task_data[v_id][node].recovery_drone_list.remove(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
            if drone_id in vehicle_task_data[v_id][node].launch_drone_list:
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                break
                
        for node_index, node in enumerate(remove_recovery_uav_route):
            if node_index == 0:
                vehicle_task_data[recv_v_id][node].recovery_drone_list.remove(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                if drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list and drone_id in vehicle_task_data[recv_v_id][node].drone_list:
                    vehicle_task_data[recv_v_id][node].drone_list.remove(drone_id)
                    # vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                    vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = None
                if drone_id in vehicle_task_data[recv_v_id][node].launch_drone_list:
                    vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                    break
                # 判断这个节点是否有发射任务,若有发射任务，则回归列表不添加
                # if drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list and drone_id in vehicle_task_data[recv_v_id][node].drone_list:    
                #     vehicle_task_data[recv_v_id][node].drone_list.remove(drone_id)
                #     vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                # continue
            # vehicle_carry_drone_list = vehicle_task_data[recv_v_id][node].drone_list
            if drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list and drone_id in vehicle_task_data[recv_v_id][node].drone_list:
                vehicle_task_data[recv_v_id][node].drone_list.remove(drone_id)
                # vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
            elif drone_id in vehicle_task_data[recv_v_id][node].launch_drone_list:
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                break

    return vehicle_task_data



# 根据车辆路径任务，更新车辆和无人机在各个节点的状态,最后返回更新后的所有车辆在各个节点状态变化，以及无人机在各个节点的状态变化，同时将修改后的精细化时间输出,仅更新车辆无人机携带状态
def update_vehicle_task(vehicle_task_data, y, vehicle_route):
    drone_id, vtp_i, customer, vtp_j, v_id, recv_v_id = y
    # vtp_i = best_y_cijkdu_plan['launch_node']
    # vtp_j = best_y_cijkdu_plan['recovery_node']
    # v_id = best_y_cijkdu_plan['launch_vehicle']
    # recv_v_id = best_y_cijkdu_plan['recovery_vehicle']
    # drone_id = best_y_cijkdu_plan['drone_id']
    # drone_use_time = best_y_cijkdu_plan['time']
    # customer = best_y_cijkdu_plan['customer']
    # 根据无人机的发射状态，更新车辆携带及节点信息
    veh_launch_index = v_id -1
    veh_recovery_index = recv_v_id -1
    update_vheicle_route = vehicle_route.copy()
    # 更新vehicle_task中车辆携带无人机状态更新
    if v_id == recv_v_id:
        task_route = vehicle_route[veh_launch_index]
        task_route_launch_index = task_route.index(vtp_i)
        task_route_recovery_index = task_route.index(vtp_j)
        remove_uav_route = task_route[task_route_launch_index:task_route_recovery_index+1]  # 包含了发射和回收的车辆路线
        for node_index, node in enumerate(remove_uav_route):
            if node_index == 0:  # 代表车辆在该点发射无人机
                vehicle_task_data[v_id][node].launch_drone_list.append(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                vehicle_task_data[v_id][node].drone_list.remove(drone_id)
            elif node_index == len(remove_uav_route)-1:
                vehicle_task_data[v_id][node].recovery_drone_list.append(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                # 判断这个节点是否有发射任务,若有发射任务，则回归列表不添加
                if drone_id not in vehicle_task_data[v_id][node].launch_drone_list: 
                    vehicle_task_data[v_id][node].drone_list.append(drone_id)
                    # vehicle_task_data[drone_id][node].dict_vehicle[v_id].drone_belong = v_id
                    vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                # else:
                #     vehicle_task_data[drone_id][node].dict_vehicle[v_id].drone_belong = v_id
            else:
                vehicle_task_data[v_id][node].drone_list.remove(drone_id)
                # vehicle_task_data[drone_id][node].dict_vehicle[v_id].drone_belong = v_id
    else:
        task_launch_route = vehicle_route[veh_launch_index]
        task_recovery_route = vehicle_route[veh_recovery_index]
        task_launch_index = task_launch_route.index(vtp_i)
        task_recovery_index = task_recovery_route.index(vtp_j)
        remove_launch_uav_route = task_launch_route[task_launch_index:]  # 包含了发射和回收的车辆路线
        remove_recovery_uav_route = task_recovery_route[task_recovery_index:]  # 包含了发射和回收的车辆路线
        for node_index, node in enumerate(remove_launch_uav_route):
            if node_index == 0:
                vehicle_task_data[v_id][node].launch_drone_list.append(drone_id)
                vehicle_task_data[v_id][node].drone_list.remove(drone_id)
                vehicle_task_data[v_id][node].drone_list = remove_duplicates(vehicle_task_data[v_id][node].drone_list)
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                continue
            vehicle_carry_drone_list = vehicle_task_data[v_id][node].drone_list
            if drone_id not in vehicle_task_data[v_id][node].recovery_drone_list:
                vehicle_task_data[v_id][node].drone_list.remove(drone_id)  # 判断车辆在节点上是否携带其型号无人机
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = None
            else:
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                break
        for node_index, node in enumerate(remove_recovery_uav_route):
            if node_index == 0:
                vehicle_task_data[recv_v_id][node].recovery_drone_list.append(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                # 判断这个节点是否有发射任务,若有发射任务，则回归列表不添加
                if drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list:    
                    vehicle_task_data[recv_v_id][node].drone_list.append(drone_id)
                    vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                    # vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id].drone_belong = recv_v_id
                continue
            vehicle_carry_drone_list = vehicle_task_data[recv_v_id][node].drone_list
            if drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list:
                vehicle_task_data[recv_v_id][node].drone_list.append(drone_id)
                vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
            else:
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                break

        # 根据无人机的作业时间，更新车辆在各个节点的到达，离开时间
        # # 根据无人机的作业时间，更新车辆在各个节点的到达，离开时间
        # vehicle_arrival_node_time = veh_arrival_times[v_id][vtp_i]
        # vehicle_task[v_id][vtp_i].arrive_times.append(vehicle_arrival_node_time)
        # vehicle_task[v_id][vtp_i].departure_times.append(vehicle_arrival_node_time + vehicle[drone_id].launchTime)
        # vehicle_departure_node_time = max(vehicle_arrival_node_time + vehicle[drone_id].launchTime, vehicle_task[v_id][vtp_j].arrive_times)
        # vehicle_stay_time = vehicle_departure_node_time - vehicle_arrival_node_time

    # 更新车辆在各个节点的状态
    return vehicle_task_data

def remove_duplicates(lst):  # 去除重复的数字后同时维持原列表顺序
    result = []
    seen = set()
    for num in lst:
        if num not in seen:
            result.append(num)
            seen.add(num)
    return result

from typing import List, Union, Any

# 输入数据标准化函数，用于将输入数据转换为一维列表
def normalize_input_data(data: Union[int, float, List[Any]]) -> List[Any]:

    # 1. 判断输入是否为列表
    if not isinstance(data, list):
        # 如果不是列表，判断是否为单一数值
        if isinstance(data, (int, float)):
            return [data]
        # 否则，抛出类型错误异常
        else:
            raise TypeError(f"输入类型不受支持，期望是 int, float 或 list，但得到的是 {type(data).__name__}")

    # 2. 判断输入是否为嵌套列表
    is_nested = any(isinstance(item, list) for item in data)

    if not is_nested:
        return data
    else:
        return [
            item 
            for sublist_or_item in data 
            for item in (sublist_or_item if isinstance(sublist_or_item, list) else [sublist_or_item])
        ]


# def update_delta_time(delta_time, detailed_vehicle_task_data, vehicle_route, y_ijkd, vehicle_arrival_time, vehicle):
#     drone_id, vtp_i, customer, vtp_j, v_id, recv_v_id = y_ijkd
#     veh_launch_index = v_id -1
#     veh_recovery_index = recv_v_id -1
#     # 更新vehicle_task中车辆携带无人机状态更新
#     if v_id == recv_v_id:
#         task_route = vehicle_route[veh_launch_index]
#         remain_route = task_route[veh_launch_index+1:]
#         detailed_vehicle_task_data = update_delta_route(delta_time, detailed_vehicle_task_data, vehicle_arrival_time, y_ijkd, remain_route)


# (假设你已经在外部维护了这个字典)
# precise_mission_times = {(drone_id, ...): {'launch_end': 123.45}, ...}

def update_delta_time(delta_time, detailed_vehicle_task_data, vehicle_route, y_ijkd, vehicle):
    """
    高层函数，用于启动时间延迟的传播。
    """
    drone_id, vtp_i, customer, vtp_j, v_id, recv_v_id = y_ijkd
    
    # 只处理同一个车辆发射和回收的情况，因为这是最直接的延迟传播路径
    if v_id == recv_v_id:
        task_route = vehicle_route[v_id - 1] # 假设 vehicle_route 是从0开始的列表
        
        # 找到发射节点在路线中的索引
        try:
            launch_node_index = task_route.index(vtp_i)
        except ValueError:
            # 如果发射节点不在路线上，这是一个逻辑错误，直接返回
            print(f"错误：发射节点 {vtp_i} 不在车辆 {v_id} 的路线中。")
            return detailed_vehicle_task_data

        # 延迟影响的是发射节点之后的所有节点
        remain_route = task_route[launch_node_index + 1:]
        
        detailed_vehicle_task_data = update_delta_route(
            delta_time, 
            detailed_vehicle_task_data, 
            v_id, 
            remain_route,
            vehicle
        )
    else:  # 更新两个车辆路线中的所有状况
        v_id_task_route = vehicle_route[v_id - 1]
        recv_v_id_task_route = vehicle_route[recv_v_id - 1]
        launch_node_index = v_id_task_route.index(vtp_i)
        recovery_node_index = recv_v_id_task_route.index(vtp_j)
        launch_remain_route = v_id_task_route[launch_node_index + 1:]
        recovery_remain_route = recv_v_id_task_route[recovery_node_index:] # 从回收当前节点开始，进而更新后续剩余路线
        detailed_vehicle_task_data = update_delta_route(
            delta_time, 
            detailed_vehicle_task_data, 
            v_id, 
            launch_remain_route,
            vehicle
        )
        detailed_vehicle_task_data = update_delta_route(
            0, 
            detailed_vehicle_task_data, 
            recv_v_id, 
            recovery_remain_route,
            vehicle
        )
    # 更新车辆在各个节点的状态
    return detailed_vehicle_task_data


def update_delta_route(
    delta_time: float, 
    detailed_vehicle_task_data: dict, 
    v_id: int, 
    remain_route: list,
    vehicle: dict, # 需要传入vehicle对象以获取回收时间
):
    """
    将一个时间延迟(delta_time)传播到指定车辆(v_id)的剩余路线(remain_route)上。
    并精确处理回收节点的排队调度。
    """
    # 延迟会从上一个节点传递下来
    accumulated_delay = delta_time

    for node_id in remain_route:
        task_obj = detailed_vehicle_task_data[v_id][node_id]
        task_obj_prcise_arrive_time = task_obj.prcise_arrive_time
        task_obj_prcise_departure_time = task_obj.prcise_departure_time

        # 1. 更新本节点的精确到达时间
        # 到达时间 = 原来的到达时间 + 累积的延迟
        task_obj.prcise_arrive_time = task_obj_prcise_arrive_time + accumulated_delay
        task_obj_prcise_arrive_time += accumulated_delay
        if task_obj_prcise_departure_time < task_obj_prcise_arrive_time:
            task_obj.prcise_departure_time = task_obj_prcise_arrive_time
            task_obj_prcise_departure_time = task_obj_prcise_arrive_time

        # 为了数据一致性，同步更新arrive_times和departure_times的初始值
        # 注意：这里的departure_times只是一个临时值，如果后面有任务，它会被覆盖
        task_obj.arrive_times.append(task_obj.prcise_arrive_time)
        task_obj.departure_times.append(task_obj.prcise_departure_time)

        # 获取在该节点需要回收的无人机列表
        drones_to_recover = task_obj.recovery_drone_list
        drones_to_launch = task_obj.launch_drone_list
        drones_to_carry = task_obj.drone_list
        copy_drones_to_carry = (drones_to_carry or []) + (drones_to_launch or [])
        # 2. 判断：如果本节点没有回收任务，逻辑非常简单
        if not drones_to_recover:  # 如果没有回收无人机的任务，则直接更新
            if not copy_drones_to_carry:  # 没有回收任务，没有发射任务，更新车辆离开时间的延迟后直接下一个点
                task_obj.prcise_departure_time = task_obj.prcise_arrive_time
                continue
            else:
                for drone_to_carry_id in copy_drones_to_carry:
                    drone_task_obj = detailed_vehicle_task_data[drone_to_carry_id][node_id]
                    drone_task_obj.prcise_arrive_time = task_obj.prcise_arrive_time
                    drone_task_obj.dict_vehicle[v_id]['prcise_arrive_time'] = task_obj.prcise_arrive_time
                    if drone_task_obj.prcise_departure_time < drone_task_obj.prcise_arrive_time:
                        drone_task_obj.prcise_departure_time = drone_task_obj.prcise_arrive_time
                        drone_task_obj.dict_vehicle[v_id]['prcise_departure_time'] = task_obj_prcise_departure_time
                continue
        else:  # 后续节点存在有无人机到达的情况
            # ================================================================
            # 3. 核心逻辑：处理有多个回收任务的复杂情况
            # ================================================================
            
            # 3.1 收集所有回收事件，并计算每个无人机的到达时间
            pending_recoveries = []
            for drone_to_recover_id in drones_to_recover:
                # 为了找到该无人机的任务信息，我们需要反向查找
                # (这是一个可以优化的点，可以提前构建一个 drone_id -> mission 的映射)
                drone_task_obj = detailed_vehicle_task_data[drone_to_recover_id][node_id]
                # drone_arrival_time = drone_task_obj.prcise_arrive_time  # 无人机到达节点的精确时间
                drone_arrival_time = detailed_vehicle_task_data[drone_to_recover_id][node_id].dict_vehicle[v_id]['prcise_arrive_time']
                
                pending_recoveries.append({
                    "drone_id": drone_to_recover_id,
                    "arrival_time": drone_arrival_time
                })

            # 3.2 按无人机到达时间排序，实现“先到先回收”
            pending_recoveries.sort(key=lambda x: x['arrival_time'])

            # 3.3 模拟车辆在该节点的服务时间线
            # 车辆的服务时间从它精确到达该节点时开始
            node_service_timeline = task_obj.prcise_arrive_time

            for recovery_event in pending_recoveries:
                drone_id = recovery_event['drone_id']
                drone_arrival = recovery_event['arrival_time']
                
                # 车辆开始回收的时间点，必须同时满足两个条件：
                # 1. 车辆已经完成上一个任务（由 node_service_timeline 表示）
                # 2. 无人机已经飞抵当前节点（由 drone_arrival 表示）
                # 因此，取两者中的最大值
                recovery_start_time = max(node_service_timeline, drone_arrival)
                
                # 获取回收操作需要的时间
                recovery_duration = vehicle[drone_id].recoveryTime
                
                # 计算回收操作的结束时间
                recovery_end_time = recovery_start_time + recovery_duration
                
                # 更新服务时间线，为下一个回收任务做准备
                # 现在，车辆直到 recovery_end_time 才有空
                node_service_timeline = recovery_end_time
                # 判断是否添加了该任务，若添加了，则删除后更新，未添加，则添加
                uav_task_found = drone_task_obj.find_task(node_id, 14)
                if uav_task_found:
                    drone_task_obj.delete_task(node_id, 14)
                drone_task_obj.add_task(node_id, 14, drone_arrival, recovery_end_time)
                # 记录回收后-离开时间列表
                drone_task_obj.departure_times.append(recovery_end_time)
                drone_task_obj.dict_vehicle[v_id]['departure_times'].append(recovery_end_time)

            # 3.4 更新车辆在该节点的最终离开时间
            # 当所有回收任务都完成后，服务时间线上的最终时间就是车辆的精确离开时间
            task_obj.prcise_departure_time = node_service_timeline
            for drone_to_recover_id in drones_to_recover:  # 将无人机全部回收完成后，统一处理。 
                drone_task_obj = detailed_vehicle_task_data[drone_to_recover_id][node_id]
                drone_task_obj.prcise_departure_time = node_service_timeline
                drone_task_obj.dict_vehicle[v_id]['prcise_departure_time'] = node_service_timeline
            # 进一步，更新承担或者待发射无人机的情况
            for drone_to_launch_id in drones_to_launch:
                drone_task_obj = detailed_vehicle_task_data[drone_to_launch_id][node_id]
                drone_task_obj.prcise_departure_time = node_service_timeline
                drone_task_obj.dict_vehicle[v_id]['prcise_departure_time'] = node_service_timeline
            
            # 4. 计算并更新传递到下一个节点的累积延迟
            # 新的延迟 = 最终离开时间 - 最初无延迟时的到达时间
            # (这里假设 vehicle_arrival_time 是未被修改的原始计划时间)
            if node_service_timeline > task_obj.prcise_arrive_time:
                original_arrival_time = task_obj.prcise_arrive_time
                accumulated_delay += (task_obj.prcise_departure_time - original_arrival_time)
            else:
                original_arrival_time = task_obj.prcise_arrive_time
                accumulated_delay += (task_obj.prcise_departure_time - original_arrival_time)

    return detailed_vehicle_task_data

def update_uav_plan(detailed_vehicle_task_data, best_uav_plan):
    """
    更新无人机计划时间
    """
    uav_plan_time = copy.copy(best_uav_plan)
    for y_ijkd in best_uav_plan.keys():
        uav_id, launch_node, customer, recovery_node, launch_veh_id, recovery_veh_id = y_ijkd
        # uav_plan_time[y_ijkd]['launch_time'] = detailed_vehicle_task_data[uav_id][launch_node].prcise_departure_time
        uav_plan_time[y_ijkd]['launch_time'] = detailed_vehicle_task_data[uav_id][launch_node].dict_vehicle[launch_veh_id]['prcise_departure_time']
        # uav_plan_time[y_ijkd]['recovery_time'] = detailed_vehicle_task_data[uav_id][recovery_node].prcise_arrive_time
        uav_plan_time[y_ijkd]['recovery_time'] = detailed_vehicle_task_data[uav_id][recovery_node].dict_vehicle[recovery_veh_id]['prcise_arrive_time']
    return uav_plan_time

def update_vehicle_arrive_time(detailed_vehicle_task_data, vehicle_arrival_time):
    update_vehicle_arrive_time = defaultdict(lambda: defaultdict(list))
    for v_id, key_values in vehicle_arrival_time.items():
        for key in key_values.keys():
            arrive_time = detailed_vehicle_task_data[v_id][key].prcise_arrive_time
            departure_time = detailed_vehicle_task_data[v_id][key].prcise_departure_time
            update_vehicle_arrive_time[v_id][key] = [arrive_time, departure_time]
    return update_vehicle_arrive_time


# def update_delta_route(delta_time, detailed_vehicle_task_data, vehicle_arrival_time, y_ijkd, remain_route):
#     drone_id, vtp_i, customer, vtp_j, v_id, recv_v_id = y_ijkd
#     for node_id in remain_route:
#         # 判断车辆在该节点是否存在回收的无人机
#         recovery_drone_list = detailed_vehicle_task_data[v_id][node_id].recovery_drone_list
#         # 更新该车辆的到达时间
#         detailed_vehicle_task_data[v_id][node_id].prcise_arrive_time = detailed_vehicle_task_data[v_id][node_id].prcise_arrive_time + delta_time
#         # 更新车辆的初试时间和离开时间
#         detailed_vehicle_task_data[v_id][node_id].arrive_times[0] = detailed_vehicle_task_data[v_id][node_id].prcise_arrive_time
#         detailed_vehicle_task_data[v_id][node_id].departure_times[0] = detailed_vehicle_task_data[v_id][node_id].prcise_arrive_time
#         # 如果车辆在该节点没有回收的无人机，则更新车辆的离开时间等于到达时间
#         if not recovery_drone_list:
#             detailed_vehicle_task_data[v_id][node_id].prcise_departure_time = detailed_vehicle_task_data[v_id][node_id].prcise_arrive_time
#         else:
#             # 如果车辆在该节点存在回收的无人机，则根据回收无人机的到达时间来精确判定
#             # 整合回收无人机列表的到达时间
#             recovery_drone_list_arrive_time = []


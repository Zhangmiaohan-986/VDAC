import numpy as np
from collections import defaultdict
from typing import Any, List, Dict, Tuple
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
        for vehicle_id in range(1, 11):  # 注意：这里硬编码了车辆数量，可能需要优化.当前代码最多允许的车辆数目为10
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
    #     # new_task.tasks = {node_id: task_list.copy() for node_id, task_list in self.tasks.items()}
    #     new_task.tasks = {node_id: [task.copy() for task in task_list] 
    #                       for node_id, task_list in self.tasks.items()}
    #     new_task.arrive_times = self.arrive_times.copy()
    #     new_task.departure_times = self.departure_times.copy()
    #     # new_task.arrive_times = self.arrive_times.copy()
    #     # new_task.departure_times = self.departure_times.copy()

    #     # 4. 根据车辆类型，安全地复制特定属性
    #     if self.vehicleType == TYPE_TRUCK:
    #         new_task.drone_list = self.drone_list.copy()
    #         new_task.launch_drone_list = self.launch_drone_list.copy()
    #         new_task.recovery_drone_list = self.recovery_drone_list.copy()
        
    #     elif self.vehicleType == TYPE_UAV:
    #         new_task.drone_belong = self.drone_belong
    #         # # 复制每个VehicleInfo实例到新对象的dict_vehicle中
    #         # for vehicle_id, vehicle_info in self.dict_vehicle.items():
    #         #     new_task.dict_vehicle[vehicle_id] = vehicle_info.copy()
    #                 # 【重要】复制 dict_vehicle 字典
    #         # new_task.dict_vehicle = {
    #         #     vehicle_id: vehicle_info.copy() 
    #         #     for vehicle_id, vehicle_info in self.dict_vehicle.items()
    #         # }
    #         new_task.dict_vehicle = {v_id: copy_vehicle_info_dict(info_dict) 
    #                                 for v_id, info_dict in self.dict_vehicle.items()}
            
    #     return new_task

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
        #    - tasks 是一个 {node_id: [task_obj1, ...]} 结构
        #    - 我们复制字典，并复制每个节点下的任务列表
        #    - 假设 Task 对象本身不需要深拷贝，这通常是安全的
        new_task.tasks = {node_id: task_list.copy() for node_id, task_list in self.tasks.items()}
        
        new_task.arrive_times = self.arrive_times.copy()
        new_task.departure_times = self.departure_times.copy()

        # 4. 根据车辆类型，安全地复制特定属性,需要独立来复制出来，不能使用copy()方法,需要指向不同对象
        if self.vehicleType == TYPE_TRUCK:
            # self.drone_list 等属性在 TRUCK 类型的对象上保证存在
            # new_task.drone_list = self.drone_list.copy()
            # new_task.launch_drone_list = self.launch_drone_list.copy()
            # new_task.recovery_drone_list = self.recovery_drone_list.copy()
            new_task.drone_list = [d_id for d_id in self.drone_list]
            # 针对 launch_drone_list，如果它确定是列表（哪怕是空列表），直接用推导式即可
            new_task.launch_drone_list = [d_id for d_id in self.launch_drone_list]
            new_task.recovery_drone_list = [d_id for d_id in self.recovery_drone_list]
        elif self.vehicleType == TYPE_UAV:
            # self.drone_belong 属性在 UAV 类型的对象上保证存在
            new_task.drone_belong = self.drone_belong

        return new_task
    
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
from initialize import deep_copy_vehicle_task_data
def restore_vehicle_task_data_for_vehicles(
    temp_vehicle_task_data,
    original_vehicle_task_data,
    changed_vehicle_ids
):
    """
    将 temp_vehicle_task_data 中指定车辆的任务数据
    恢复为 original_vehicle_task_data 中的“初始版本”。

    参数:
        temp_vehicle_task_data: 当前被修改过的任务数据 (dict / defaultdict)
        original_vehicle_task_data: 原始的任务数据 (dict / defaultdict)
        changed_vehicle_ids: 需要恢复的车辆 ID 列表/集合，比如 {2, 3}
    """
    from collections import defaultdict

    # 为了安全，统一用 set
    changed_vehicle_ids = set(changed_vehicle_ids)

    # 确保 temp_vehicle_task_data 是一个字典
    if temp_vehicle_task_data is None:
        temp_vehicle_task_data = {}

    # 创建一个新的字典来保存恢复后的数据
    restored_data = temp_vehicle_task_data.copy()
    # restored_data = deep_copy_vehicle_task_data(original_vehicle_task_data)
    for veh_id in changed_vehicle_ids:
        # 如果原始数据里没有这辆车，就直接跳过/或者删掉
        if veh_id not in original_vehicle_task_data:
            # 如果希望删除 temp 里的这辆车任务，可以用下面这行
            # restored_data.pop(veh_id, None)
            continue

        inner_dict = original_vehicle_task_data[veh_id]

        # 保留原始内层 dict/defaultdict 的类型和 default_factory
        new_inner = inner_dict.copy()

        # 遍历节点任务，逐个调用 vehicle_task.fast_copy()
        for node_id, task_obj in inner_dict.items():
            new_inner[node_id] = task_obj.fast_copy()

        # 用“恢复后的”新内层字典覆盖 restored_data 中对应车辆的任务数据
        restored_data[veh_id] = new_inner

    return restored_data


def deep_remove_vehicle_task(vehicle_task_data, y, vehicle_route, orig_vehicle_id):
    drone_id, vtp_i, customer, vtp_j, v_id, recv_v_id = y
    veh_launch_index = v_id -1
    veh_recovery_index = recv_v_id -1
    # update_vheicle_route = vehicle_route.copy()
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
                if drone_id in vehicle_task_data[v_id][node].launch_drone_list:
                    is_launch_task = True
                    break
            else:
                if drone_id not in vehicle_task_data[v_id][node].launch_drone_list and drone_id in vehicle_task_data[v_id][node].drone_list and drone_id not in vehicle_task_data[v_id][node].recovery_drone_list:
                    vehicle_task_data[v_id][node].drone_list.remove(drone_id)
                if drone_id in vehicle_task_data[v_id][node].launch_drone_list:
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
                break   # 修改因为该点承担后续回收任务，会把后面的节点内容的承担也删掉的错误
            if drone_id in vehicle_task_data[v_id][node].launch_drone_list:
                vehicle_task_data[drone_id][node].dict_vehicle[v_id]['drone_belong'] = v_id
                break
                
        for node_index, node in enumerate(remove_recovery_uav_route):
            if node_index == 0:
                vehicle_task_data[recv_v_id][node].recovery_drone_list.remove(drone_id)
                vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id
                if orig_vehicle_id == recv_v_id:
                    vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = recv_v_id # 如果原始车辆是接收车辆，则无人机归属为接收车辆
                    break                
                elif orig_vehicle_id != recv_v_id and drone_id not in vehicle_task_data[recv_v_id][node].launch_drone_list and drone_id in vehicle_task_data[recv_v_id][node].drone_list:
                    vehicle_task_data[recv_v_id][node].drone_list.remove(drone_id)
                    # vehicle_task_data[recv_v_id][node].drone_list = remove_duplicates(vehicle_task_data[recv_v_id][node].drone_list)
                    vehicle_task_data[drone_id][node].dict_vehicle[recv_v_id]['drone_belong'] = None
                elif drone_id in vehicle_task_data[recv_v_id][node].launch_drone_list:
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
    # uav_plan_time = {}
    for y_ijkd in best_uav_plan:
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

def update_re_uav_plan(state, re_uav_plan):
    """
    更新并重构 re_uav_plan
    
    参数:
    state: 包含 customer_plan, uav_travel, rm_empty_vehicle_arrive_time 的状态对象
    xeee_matrix: 对应原本的 self.xeee，能耗矩阵 [drone][l_map][cust][r_map]
    node_collection: 对应原本的 self.node，可以通过 node_id 获取 map_key 的对象列表或字典
    """
    
    # 1. 提取 State 中的数据源
    vehicle_arrive_time = state.rm_empty_vehicle_arrive_time
    customer_plan = state.customer_plan
    customer_cost = state.uav_cost
    uav_travel = state.uav_travel
    node_collection = state.node
    xeee_matrix = state.xeee
    # 临时列表，用于存储 (排序时间, key, value)
    temp_tasks = []

    # 2. 遍历 customer_plan 以重建最新的任务数据
    # customer_plan 结构: key=customer_id, value=[drone, launch, customer, recovery, l_veh, r_veh]
    for customer_id, plan in customer_plan.items():
        
        # 解包 plan 信息
        drone_id = plan[0]
        launch_node = plan[1]
        target_customer = plan[2]
        recovery_node = plan[3]
        launch_vehicle = plan[4]
        recovery_vehicle = plan[5]

        # 3. 构造任务 Key (Tuple)
        mission_key = (drone_id, launch_node, target_customer, recovery_node, launch_vehicle, recovery_vehicle)

        # 4. 获取 Map Key (用于矩阵索引)
        # 对应原 self.node[id].map_key
        l_map_key = node_collection[launch_node].map_key
        r_map_key = node_collection[recovery_node].map_key

        # 5. 获取车辆到达时间 (作为新的 launch/recovery time)
        # 对应 state.vehicle_arrive_time
        try:
            l_arrival_time = vehicle_arrive_time[launch_vehicle][launch_node]
        except KeyError:
            l_arrival_time = float('inf') # 防止数据缺失报错

        try:
            r_arrival_time = vehicle_arrive_time[recovery_vehicle][recovery_node]
        except KeyError:
            r_arrival_time = float('inf')

        # 6. 获取 UAV 飞行数据 (Route, Cost, Time)
        # 对应 state.uav_travel[drone][l_map][r_map]
        travel_obj = uav_travel[drone_id][l_map_key][r_map_key]
        
        # 获取路径
        uav_route_path = travel_obj.path
        
        # 假设 travel_obj 对象中有对应的 cost 和 time 属性
        # 如果属性名不同（例如是 total_cost 或 duration），请在此处修改
        # route_cost = getattr(travel_obj, 'travel_cost', 0) 
        route_time = getattr(travel_obj, 'totalTime', 0)

        # 7. 获取能量消耗
        # 对应 self.xeee[drone][l_map][cust][r_map]
        energy_val = xeee_matrix[drone_id][l_map_key][target_customer][r_map_key]

        # 8. 组装 Value 字典
        task_info = {
            'drone_id': drone_id,
            'launch_vehicle': launch_vehicle,
            'recovery_vehicle': recovery_vehicle,
            'launch_node': launch_node,
            'recovery_node': recovery_node,
            'customer': target_customer,
            'launch_time': l_arrival_time,   # 更新为车辆到达发射点时间
            'recovery_time': r_arrival_time, # 更新为车辆到达回收点时间
            'energy': energy_val,            # 更新能耗
            'cost': customer_cost[target_customer],              # 更新路径成本
            'time': route_time,              # 更新飞行时间
            'uav_route': uav_route_path      # 更新路径列表
        }

        # 将数据存入临时列表，第一个元素用于排序
        temp_tasks.append((l_arrival_time, mission_key, task_info))

    # 9. 排序：按照发射车辆到达时间 (升序)
    temp_tasks.sort(key=lambda x: x[0])

    # 10. 重组字典 (保持顺序)
    sorted_re_uav_plan = {item[1]: item[2] for item in temp_tasks}

    # 11. 更新回 State
    state.re_uav_plan = sorted_re_uav_plan
    
    return sorted_re_uav_plan

import os
import json
from typing import Any, Dict, List
import pandas as pd


def save_alns_results(
    instance_name: str,
    y_best,
    y_cost,
    win_cost,
    uav_route_cost,
    vehicle_route_cost,
    strategy_weights,
    operator_weights,
    elapsed_time,
    best_objective,
    best_vehicle_max_times,
    best_global_max_time,
    best_arrive_time,
    best_window_total_cost,
    best_uav_tw_violation_cost,
    best_total_cost_dict,
    best_state,
    # === 新增: 最终方案相关标量指标 ===
    best_final_uav_cost=None,
    best_final_objective=None,
    best_final_win_cost=None,
    best_total_win_cost=None,
    # === [新增] 在此处添加 best_final_vehicle_route_cost ===
    best_final_vehicle_route_cost=None, 
    
    # === 新增: 最终方案全过程曲线 ===
    final_uav_cost=None,
    final_total_list=None,
    final_win_cost=None,
    final_total_objective=None,
    final_vehicle_route_cost=None,
    # === 新增: 完成时间相关 ===
    best_final_vehicle_max_times=None,      
    best_final_global_max_time=None,        
    work_time=None,                         
    final_work_time=None,
    
    # === [新增] 在此处添加 best_final_state ===
    best_final_state=None,              
    
    # base_dir: str = r"VDAC\saved_solutions",
    base_dir = None,
) -> None:
    """
    将 ALNS 求解过程与最优解信息保存为 txt 和 Excel 文件。
    """

    # =========================
    # 1. 创建目录
    # =========================
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    case_dir = os.path.join(base_dir, instance_name)
    os.makedirs(case_dir, exist_ok=True)

    prefix = os.path.join(case_dir, instance_name)

    # =========================
    # 2. 构造 summary_data（数据预处理）
    # =========================

    # --- 处理 best_uav_tw_violation_cost ---
    if best_uav_tw_violation_cost is None:
        buv_for_summary = None
    elif hasattr(best_uav_tw_violation_cost, "items"):
        buv_for_summary = {k: float(v) for k, v in best_uav_tw_violation_cost.items()}
    else:
        buv_for_summary = float(best_uav_tw_violation_cost)

    # --- 处理 best_total_cost_dict ---
    if best_total_cost_dict is None:
        btcd_for_summary = None
    elif hasattr(best_total_cost_dict, "items"):
        btcd_for_summary = {k: float(v) for k, v in best_total_cost_dict.items()}
    else:
        btcd_for_summary = best_total_cost_dict
        
    # --- 处理 best_vehicle_max_times ---
    if best_vehicle_max_times is None:
        bvt_for_summary = None
    elif hasattr(best_vehicle_max_times, "items"):
        bvt_for_summary = {k: float(v) for k, v in best_vehicle_max_times.items()}
    else:
        bvt_for_summary = float(best_vehicle_max_times)

    # --- 处理 best_final_vehicle_max_times ---
    if best_final_vehicle_max_times is None:
        bfvt_for_summary = None
    elif hasattr(best_final_vehicle_max_times, "items"):
        bfvt_for_summary = {k: float(v) for k, v in best_final_vehicle_max_times.items()}
    else:
        bfvt_for_summary = float(best_final_vehicle_max_times)

    summary_data = {
        "instance_name": instance_name,
        "best_objective": float(best_objective) if best_objective is not None else None,
        "elapsed_time": float(elapsed_time) if elapsed_time is not None else None,
        "best_global_max_time": float(best_global_max_time) if best_global_max_time is not None else None,
        "len_y_cost": len(y_cost) if y_cost is not None else 0,
        "len_y_best": len(y_best) if y_best is not None else 0,
        "len_win_cost": len(win_cost) if win_cost is not None else 0,
        "len_uav_route_cost": len(uav_route_cost) if uav_route_cost is not None else 0,
        "len_vehicle_route_cost": len(vehicle_route_cost) if vehicle_route_cost is not None else 0,
        "strategy_weights": strategy_weights,
        "operator_weights": operator_weights,
        "best_vehicle_max_times": bvt_for_summary,
        "best_arrive_time_brief": {
            "num_vehicles": len(best_arrive_time),
            "total_records": sum(len(v) for v in best_arrive_time.items())
            if hasattr(best_arrive_time, "items")
            else sum(len(v) for v in best_arrive_time.values()),
        },
        "best_window_total_cost": float(best_window_total_cost) if best_window_total_cost is not None else None,
        "best_total_cost_dict": btcd_for_summary,
        "best_uav_tw_violation_cost": buv_for_summary,
        "best_state_total_cost_attr": getattr(best_state, "_total_cost", None),

        # === 新增: 最终方案标量指标写入 summary ===
        "best_final_uav_cost": float(best_final_uav_cost) if best_final_uav_cost is not None else None,
        "best_final_objective": float(best_final_objective) if best_final_objective is not None else None,
        "best_final_win_cost": float(best_final_win_cost) if best_final_win_cost is not None else None,
        "best_total_win_cost": float(best_total_win_cost) if best_total_win_cost is not None else None,
        
        # === [新增] ===
        "best_final_vehicle_route_cost": float(best_final_vehicle_route_cost) if best_final_vehicle_route_cost is not None else None,
        "best_final_state_total_cost_attr": getattr(best_final_state, "_total_cost", None) if best_final_state else None,

        "best_final_vehicle_max_times": bfvt_for_summary,
        "best_final_global_max_time": float(best_final_global_max_time) if best_final_global_max_time is not None else None,

        # === 新增: 最终方案曲线长度信息 ===
        "len_final_uav_cost": len(final_uav_cost) if final_uav_cost is not None else 0,
        "len_final_total_list": len(final_total_list) if final_total_list is not None else 0,
        "len_final_win_cost": len(final_win_cost) if final_win_cost is not None else 0,
        "len_final_total_objective": len(final_total_objective) if final_total_objective is not None else 0,
        "len_final_vehicle_route_cost": len(final_vehicle_route_cost) if final_vehicle_route_cost is not None else 0,
        "len_work_time": len(work_time) if work_time is not None else 0,
        "len_final_work_time": len(final_work_time) if final_work_time is not None else 0,
    }
    
    # =========================
    # 3. 保存概览到 TXT
    # =========================
    summary_txt_path = f"{prefix}_summary.txt"

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("==== ALNS Result Summary ====\n")
        for k, v in summary_data.items():
            if isinstance(v, (dict, list)):
                v_clean = make_json_friendly(v)
                f.write(f"{k}:\n{json.dumps(v_clean, ensure_ascii=False, indent=2)}\n\n")
            else:
                f.write(f"{k}: {v}\n")

    # =========================
    # 4. 保存曲线到 Excel
    # =========================
    curves_xlsx_path = f"{prefix}_curves.xlsx"

    def _pad_list(lst: List[Any], target_len: int) -> List[Any]:
        if lst is None:
            return [None] * target_len
        lst = list(lst)
        if len(lst) < target_len:
            lst = lst + [None] * (target_len - len(lst))
        return lst

    max_len = max(
        len(y_cost),
        len(y_best),
        len(win_cost),
        len(uav_route_cost),
        len(vehicle_route_cost),
        len(final_uav_cost) if final_uav_cost is not None else 0,
        len(final_total_list) if final_total_list is not None else 0,
        len(final_win_cost) if final_win_cost is not None else 0,
        len(final_total_objective) if final_total_objective is not None else 0,
        len(final_vehicle_route_cost) if final_vehicle_route_cost is not None else 0,
        len(work_time) if work_time is not None else 0,
        len(final_work_time) if final_work_time is not None else 0,
    )

    df_curves = pd.DataFrame({
        "iteration": list(range(max_len)),
        "y_cost": _pad_list(y_cost, max_len),
        "y_best": _pad_list(y_best, max_len),
        "win_cost": _pad_list(win_cost, max_len),
        "uav_route_cost": _pad_list(uav_route_cost, max_len),
        "vehicle_route_cost": _pad_list(vehicle_route_cost, max_len),
        # === 新增: 最终方案相关曲线 ===
        "final_uav_cost": _pad_list(final_uav_cost, max_len),
        "final_total_list": _pad_list(final_total_list, max_len),
        "final_win_cost": _pad_list(final_win_cost, max_len),
        "final_total_objective": _pad_list(final_total_objective, max_len),
        "final_vehicle_route_cost": _pad_list(final_vehicle_route_cost, max_len),
        # === 新增: 完成时间曲线 ===
        "work_time": _pad_list(work_time, max_len),
        "final_work_time": _pad_list(final_work_time, max_len),
    })

    # 策略权重
    df_strategy_weights = pd.DataFrame(
        [{"strategy": k, "weight": v} for k, v in strategy_weights.items()]
    )

    # 算子权重展开
    op_rows = []
    for strat, group in operator_weights.items():
        for op_type, w_dict in group.items():
            for op_name, w in w_dict.items():
                op_rows.append({
                    "strategy": strat,
                    "op_type": op_type,
                    "operator": op_name,
                    "weight": w,
                })
    df_operator_weights = pd.DataFrame(op_rows)

    with pd.ExcelWriter(curves_xlsx_path) as writer:
        df_curves.to_excel(writer, sheet_name="curves", index=False)
        df_strategy_weights.to_excel(writer, sheet_name="strategy_weights", index=False)
        df_operator_weights.to_excel(writer, sheet_name="operator_weights", index=False)

    # =========================
    # 5. 保存 summary + 违约成本 + total_cost_dict
    # =========================
    summary_xlsx_path = f"{prefix}_summary.xlsx"

    # 每辆车完成时间
    if best_vehicle_max_times is not None and hasattr(best_vehicle_max_times, "items"):
        df_vehicle_finish = pd.DataFrame(
            [{"vehicle_id": vid, "completion_time": t} for vid, t in best_vehicle_max_times.items()]
        )
    else:
        df_vehicle_finish = pd.DataFrame([{"vehicle_id": "all", "completion_time": best_vehicle_max_times}])

    # 时间窗惩罚细节
    if best_uav_tw_violation_cost is not None and hasattr(best_uav_tw_violation_cost, "items"):
        df_tw_violation = pd.DataFrame(
            [{"key": k, "violation_cost": float(v)} for k, v in best_uav_tw_violation_cost.items()]
        )
    else:
        df_tw_violation = pd.DataFrame(columns=["key", "violation_cost"])

    # total_cost_dict
    if best_total_cost_dict is not None and hasattr(best_total_cost_dict, "items"):
        df_total_cost_dict = pd.DataFrame(
            [{"key": k, "value": float(v)} for k, v in best_total_cost_dict.items()]
        )
    else:
        df_total_cost_dict = pd.DataFrame(columns=["key", "value"])

    df_summary_scalar = pd.DataFrame(
        [
            {
                "instance_name": instance_name,
                "best_objective": best_objective,
                "elapsed_time": elapsed_time,
                "best_global_max_time": best_global_max_time,
                "best_window_total_cost": best_window_total_cost,
                "best_state_total_cost_attr": getattr(best_state, "_total_cost", None),
                # === 新增: 最终方案标量指标 ===
                "best_final_uav_cost": best_final_uav_cost,
                "best_final_objective": best_final_objective,
                "best_final_win_cost": best_final_win_cost,
                "best_total_win_cost": best_total_win_cost,
                
                "best_final_vehicle_route_cost": best_final_vehicle_route_cost,
                "best_final_state_total_cost_attr": getattr(best_final_state, "_total_cost", None) if best_final_state else None,
                
                "best_final_vehicle_max_times": str(best_final_vehicle_max_times),
                "best_final_global_max_time": best_final_global_max_time,
            }
        ]
    )

    with pd.ExcelWriter(summary_xlsx_path) as writer:
        df_summary_scalar.to_excel(writer, sheet_name="summary", index=False)
        df_vehicle_finish.to_excel(writer, sheet_name="vehicle_finish_time", index=False)
        df_tw_violation.to_excel(writer, sheet_name="tw_violation", index=False)
        df_total_cost_dict.to_excel(writer, sheet_name="total_cost_detail", index=False)

    # =========================
    # 6. best_arrive_time
    # =========================
    arrive_xlsx_path = f"{prefix}_best_arrive_time.xlsx"
    arrive_rows = []
    if best_arrive_time:
        for vid, node_times in best_arrive_time.items():
            for node_id, t in node_times.items():
                arrive_rows.append(
                    {"vehicle_id": vid, "node_id": node_id, "arrive_time": t}
                )
    df_arrive = pd.DataFrame(arrive_rows)
    df_arrive.to_excel(arrive_xlsx_path, index=False, sheet_name="arrive_time")

    # =========================
    # 7. best_state 核心结构 (原始最优解)
    # =========================
    ### [修正] 这里不再调用 _save_state_to_excel，而是直接展开保存逻辑 ###
    best_state_xlsx_path = f"{prefix}_best_state.xlsx"
    
    # 7.1 customer_plan
    cp_rows = []
    for cid, assign in best_state.customer_plan.items():
        row = {"customer_id": cid}
        if isinstance(assign, (list, tuple)):
            for i, v in enumerate(assign):
                row[f"field_{i}"] = v
        else:
            row["assignment"] = assign
        cp_rows.append(row)
    df_customer_plan = pd.DataFrame(cp_rows)

    # 7.2 uav_cost
    uav_cost = getattr(best_state, "uav_cost", {})
    if isinstance(uav_cost, dict):
        df_uav_cost = pd.DataFrame(
            [{"uav_id": k, "cost": v} for k, v in uav_cost.items()]
        )
    else:
        df_uav_cost = pd.DataFrame(columns=["uav_id", "cost"])

    # 7.3 vehicle_routes
    vr = getattr(best_state, "vehicle_routes", [])
    vr_rows = []
    if isinstance(vr, dict):
        for vid, route in vr.items():
            for idx, node in enumerate(route):
                vr_rows.append(
                    {"vehicle_id": vid, "seq": idx, "node_id": node}
                )
    else:
        for i, route in enumerate(vr):
            vid = i + 1
            for idx, node in enumerate(route):
                vr_rows.append(
                    {"vehicle_id": vid, "seq": idx, "node_id": node}
                )
    df_vehicle_routes = pd.DataFrame(vr_rows)

    # 7.4 uav_plan
    uav_plan = getattr(best_state, "uav_plan", None)
    try:
        uav_plan_json = json.dumps(make_json_friendly(uav_plan), ensure_ascii=False)
    except TypeError:
        uav_plan_json = repr(uav_plan)
    df_uav_plan = pd.DataFrame([{"uav_plan_json": uav_plan_json}])

    # 7.5 scalar
    df_state_scalar = pd.DataFrame(
        [
            {
                "_total_cost_attr": getattr(best_state, "_total_cost", None),
                "objective_now": best_state.objective()
                if hasattr(best_state, "objective")
                else None,
            }
        ]
    )
    
    # 7.6 final_uav_plan (如果 best_state 中已经包含此信息)
    final_uav_plan_data_bs = getattr(best_state, "final_uav_plan", None)

    with pd.ExcelWriter(best_state_xlsx_path) as writer:
        df_customer_plan.to_excel(writer, sheet_name="customer_plan", index=False)
        df_uav_cost.to_excel(writer, sheet_name="uav_cost", index=False)
        df_vehicle_routes.to_excel(writer, sheet_name="vehicle_routes", index=False)
        df_uav_plan.to_excel(writer, sheet_name="uav_plan_raw", index=False)
        df_state_scalar.to_excel(writer, sheet_name="state_scalar", index=False)

        if final_uav_plan_data_bs is not None:
            final_rows_bs = []
            for key, info in final_uav_plan_data_bs.items():
                row = {}
                if isinstance(key, tuple) and len(key) >= 6:
                    row["key_drone_id"] = key[0]
                    row["key_launch_node"] = key[1]
                    row["key_customer"] = key[2]
                    row["key_recovery_node"] = key[3]
                    row["key_launch_vehicle"] = key[4]
                    row["key_recovery_vehicle"] = key[5]
                else:
                    row["key_raw"] = str(key)

                if isinstance(info, dict):
                    for field in ["drone_id", "launch_vehicle", "recovery_vehicle", 
                                  "launch_node", "recovery_node", "customer", 
                                  "launch_time", "recovery_time", "energy", 
                                  "cost", "time", "uav_route_cost", "uav_time_cost"]:
                        row[field] = info.get(field)
                    
                    route = info.get("uav_route")
                    try:
                        row["uav_route"] = json.dumps(make_json_friendly(route), ensure_ascii=False)
                    except Exception:
                        row["uav_route"] = str(route)
                    
                    try:
                        row["uav_route_len"] = len(route) if route is not None else 0
                    except TypeError:
                        row["uav_route_len"] = None
                else:
                    row["info_raw"] = str(info)
                final_rows_bs.append(row)
            
            df_final_uav_bs = pd.DataFrame(final_rows_bs)
            df_final_uav_bs.to_excel(writer, sheet_name="final_uav_plan", index=False)


    # =========================
    # 8. [新增] best_final_state 核心结构 (最终修正解)
    # =========================
    if best_final_state is not None:
        best_final_state_xlsx_path = f"{prefix}_best_final_state.xlsx"
        
        # 8.1 customer_plan
        cp_rows = []
        for cid, assign in best_final_state.customer_plan.items():
            row = {"customer_id": cid}
            if isinstance(assign, (list, tuple)):
                for i, v in enumerate(assign):
                    row[f"field_{i}"] = v
            else:
                row["assignment"] = assign
            cp_rows.append(row)
        df_customer_plan = pd.DataFrame(cp_rows)

        # 8.2 uav_cost
        uav_cost = getattr(best_final_state, "uav_cost", {})
        if isinstance(uav_cost, dict):
            df_uav_cost = pd.DataFrame(
                [{"uav_id": k, "cost": v} for k, v in uav_cost.items()]
            )
        else:
            df_uav_cost = pd.DataFrame(columns=["uav_id", "cost"])

        # 8.3 vehicle_routes
        vr = getattr(best_final_state, "vehicle_routes", [])
        vr_rows = []
        if isinstance(vr, dict):
            for vid, route in vr.items():
                for idx, node in enumerate(route):
                    vr_rows.append(
                        {"vehicle_id": vid, "seq": idx, "node_id": node}
                    )
        else:
            for i, route in enumerate(vr):
                vid = i + 1
                for idx, node in enumerate(route):
                    vr_rows.append(
                        {"vehicle_id": vid, "seq": idx, "node_id": node}
                    )
        df_vehicle_routes = pd.DataFrame(vr_rows)

        # 8.4 uav_plan
        uav_plan = getattr(best_final_state, "uav_plan", None)
        try:
            uav_plan_json = json.dumps(make_json_friendly(uav_plan), ensure_ascii=False)
        except TypeError:
            uav_plan_json = repr(uav_plan)
        df_uav_plan = pd.DataFrame([{"uav_plan_json": uav_plan_json}])

        # 8.5 scalar
        df_state_scalar = pd.DataFrame(
            [
                {
                    "_total_cost_attr": getattr(best_final_state, "_total_cost", None),
                    "objective_now": best_final_state.objective()
                    if hasattr(best_final_state, "objective")
                    else None,
                }
            ]
        )
        
        # 8.6 final_uav_plan (如果 final_state 里有这个属性)
        final_uav_plan_data = getattr(best_final_state, "final_uav_plan", None)

        with pd.ExcelWriter(best_final_state_xlsx_path) as writer:
            df_customer_plan.to_excel(writer, sheet_name="customer_plan", index=False)
            df_uav_cost.to_excel(writer, sheet_name="uav_cost", index=False)
            df_vehicle_routes.to_excel(writer, sheet_name="vehicle_routes", index=False)
            df_uav_plan.to_excel(writer, sheet_name="uav_plan_raw", index=False)
            df_state_scalar.to_excel(writer, sheet_name="state_scalar", index=False)

            if final_uav_plan_data is not None:
                final_rows = []
                for key, info in final_uav_plan_data.items():
                    row = {}
                    if isinstance(key, tuple) and len(key) >= 6:
                        row["key_drone_id"] = key[0]
                        row["key_launch_node"] = key[1]
                        row["key_customer"] = key[2]
                        row["key_recovery_node"] = key[3]
                        row["key_launch_vehicle"] = key[4]
                        row["key_recovery_vehicle"] = key[5]
                    else:
                        row["key_raw"] = str(key)

                    if isinstance(info, dict):
                        # 尝试提取常用字段
                        for field in ["drone_id", "launch_vehicle", "recovery_vehicle", 
                                      "launch_node", "recovery_node", "customer", 
                                      "launch_time", "recovery_time", "energy", 
                                      "cost", "time", "uav_route_cost", "uav_time_cost"]:
                            row[field] = info.get(field)

                        route = info.get("uav_route")
                        try:
                            row["uav_route"] = json.dumps(make_json_friendly(route), ensure_ascii=False)
                        except Exception:
                            row["uav_route"] = str(route)
                        
                        try:
                            row["uav_route_len"] = len(route) if route is not None else 0
                        except TypeError:
                            row["uav_route_len"] = None
                    else:
                        row["info_raw"] = str(info)

                    final_rows.append(row)

                df_final_uav = pd.DataFrame(final_rows)
                df_final_uav.to_excel(writer, sheet_name="final_uav_plan", index=False)
        
        print(f"[save_alns_results] best_final_state 已保存到: {best_final_state_xlsx_path}")

    print(f"[save_alns_results] 结果已保存到目录: {case_dir}")


# def save_alns_results(
#     instance_name: str,
#     y_best,
#     y_cost,
#     win_cost,
#     uav_route_cost,
#     vehicle_route_cost,
#     strategy_weights,
#     operator_weights,
#     elapsed_time,
#     best_objective,
#     best_vehicle_max_times,
#     best_global_max_time,
#     best_arrive_time,
#     best_window_total_cost,
#     best_uav_tw_violation_cost,
#     best_total_cost_dict,
#     best_state,
#     # === 新增: 最终方案相关标量指标 ===
#     best_final_uav_cost=None,
#     best_final_objective=None,
#     best_final_win_cost=None,
#     best_total_win_cost=None,
#     # === 新增: 最终方案全过程曲线 ===
#     final_uav_cost=None,
#     final_total_list=None,
#     final_win_cost=None,
#     final_total_objective=None,
#     final_vehicle_route_cost=None,
#     # === 新增: 完成时间相关 ===
#     best_final_vehicle_max_times=None,      # 最终方案下车辆完成时间（标量）
#     best_final_global_max_time=None,        # 最终方案下全局最大完成时间（标量）
#     work_time=None,                         # 每一代当前解完成时间 list
#     final_work_time=None,                   # 每一代最终方案完成时间 list
#     base_dir: str = r"VDAC\saved_solutions",
# ) -> None:
#     """
#     将 ALNS 求解过程与最优解信息保存为 txt 和 Excel 文件。
#     """

#     # =========================
#     # 1. 创建目录
#     # =========================
#     base_dir = os.path.abspath(base_dir)
#     os.makedirs(base_dir, exist_ok=True)

#     case_dir = os.path.join(base_dir, instance_name)
#     os.makedirs(case_dir, exist_ok=True)

#     prefix = os.path.join(case_dir, instance_name)

#     # =========================
#     # 2. 构造 summary_data（注意类型消毒）
#     # =========================

#     # best_uav_tw_violation_cost：可能是 dict / defaultdict / 标量
#     if best_uav_tw_violation_cost is None:
#         buv_for_summary = None
#     elif hasattr(best_uav_tw_violation_cost, "items"):
#         # dict-like
#         buv_for_summary = {k: float(v) for k, v in best_uav_tw_violation_cost.items()}
#     else:
#         # 标量
#         buv_for_summary = float(best_uav_tw_violation_cost)

#     # best_total_cost_dict：你给的是 defaultdict(float, {...})
#     if best_total_cost_dict is None:
#         btcd_for_summary = None
#     elif hasattr(best_total_cost_dict, "items"):
#         btcd_for_summary = {k: float(v) for k, v in best_total_cost_dict.items()}
#     else:
#         btcd_for_summary = best_total_cost_dict
        
#     if best_vehicle_max_times is None:
#         bvt_for_summary = None
#     elif hasattr(best_vehicle_max_times, "items"):
#         # dict-like: {veh_id: time}
#         bvt_for_summary = {k: float(v) for k, v in best_vehicle_max_times.items()}
#     else:
#         # 标量：比如你现在这个 18
#         bvt_for_summary = float(best_vehicle_max_times)
    
#     # --- 处理 best_vehicle_max_times (中间最优解) ---
#     if best_vehicle_max_times is None:
#         bvt_for_summary = None
#     elif hasattr(best_vehicle_max_times, "items"):
#         bvt_for_summary = {k: float(v) for k, v in best_vehicle_max_times.items()}
#     else:
#         bvt_for_summary = float(best_vehicle_max_times)

#     # --- [修改] 处理 best_final_vehicle_max_times (最终解) --- 
#     # 原代码直接 float() 导致报错，这里增加了字典判断逻辑
#     if best_final_vehicle_max_times is None:
#         bfvt_for_summary = None
#     elif hasattr(best_final_vehicle_max_times, "items"):    ### <--- [修改] 2. 增加字典判断
#         # 如果是字典，保留字典结构
#         bfvt_for_summary = {k: float(v) for k, v in best_final_vehicle_max_times.items()}
#     else:
#         # 如果是标量，才转 float
#         bfvt_for_summary = float(best_final_vehicle_max_times)

#     # summary_data = {
#     #     "instance_name": instance_name,
#     #     "best_objective": float(best_objective) if best_objective is not None else None,
#     #     "elapsed_time": float(elapsed_time) if elapsed_time is not None else None,
#     #     "best_global_max_time": float(best_global_max_time) if best_global_max_time is not None else None,
#     #     "len_y_cost": len(y_cost) if y_cost is not None else 0,
#     #     "len_y_best": len(y_best) if y_best is not None else 0,
#     #     "len_win_cost": len(win_cost) if win_cost is not None else 0,
#     #     "len_uav_route_cost": len(uav_route_cost) if uav_route_cost is not None else 0,
#     #     "len_vehicle_route_cost": len(vehicle_route_cost) if vehicle_route_cost is not None else 0,
#     #     "strategy_weights": strategy_weights,
#     #     "operator_weights": operator_weights,
#     #     "best_vehicle_max_times": bvt_for_summary,
#     #     "best_arrive_time_brief": {
#     #         "num_vehicles": len(best_arrive_time),
#     #         "total_records": sum(len(v) for v in best_arrive_time.items())
#     #         if hasattr(best_arrive_time, "items")
#     #         else sum(len(v) for v in best_arrive_time.values()),
#     #     },
#     #     "best_window_total_cost": float(best_window_total_cost) if best_window_total_cost is not None else None,
#     #     "best_total_cost_dict": btcd_for_summary,
#     #     "best_uav_tw_violation_cost": buv_for_summary,
#     #     "best_state_total_cost_attr": getattr(best_state, "_total_cost", None),
#     # }
#     # === 新增: 最终方案曲线如果传进来是 numpy array / generator, 这里不强制转换，交给 _pad_list 统一处理 ===
#     summary_data = {
#         "instance_name": instance_name,
#         "best_objective": float(best_objective) if best_objective is not None else None,
#         "elapsed_time": float(elapsed_time) if elapsed_time is not None else None,
#         "best_global_max_time": float(best_global_max_time) if best_global_max_time is not None else None,
#         "len_y_cost": len(y_cost) if y_cost is not None else 0,
#         "len_y_best": len(y_best) if y_best is not None else 0,
#         "len_win_cost": len(win_cost) if win_cost is not None else 0,
#         "len_uav_route_cost": len(uav_route_cost) if uav_route_cost is not None else 0,
#         "len_vehicle_route_cost": len(vehicle_route_cost) if vehicle_route_cost is not None else 0,
#         "strategy_weights": strategy_weights,
#         "operator_weights": operator_weights,
#         "best_vehicle_max_times": bvt_for_summary,
#         "best_arrive_time_brief": {
#             "num_vehicles": len(best_arrive_time),
#             "total_records": sum(len(v) for v in best_arrive_time.items())
#             if hasattr(best_arrive_time, "items")
#             else sum(len(v) for v in best_arrive_time.values()),
#         },
#         "best_window_total_cost": float(best_window_total_cost) if best_window_total_cost is not None else None,
#         "best_total_cost_dict": btcd_for_summary,
#         "best_uav_tw_violation_cost": buv_for_summary,
#         "best_state_total_cost_attr": getattr(best_state, "_total_cost", None),

#         # === 新增: 最终方案标量指标写入 summary ===
#         "best_final_uav_cost": float(best_final_uav_cost) if best_final_uav_cost is not None else None,
#         "best_final_objective": float(best_final_objective) if best_final_objective is not None else None,
#         "best_final_win_cost": float(best_final_win_cost) if best_final_win_cost is not None else None,
#         "best_total_win_cost": float(best_total_win_cost) if best_total_win_cost is not None else None,
#         # === [新增] ===
#         "len_final_vehicle_route_cost": len(final_vehicle_route_cost) if final_vehicle_route_cost is not None else 0,

#         # === 新增: 完成时间相关最终标量 ===
#         # "best_final_vehicle_max_times": float(best_final_vehicle_max_times) if best_final_vehicle_max_times is not None else None,
#         "best_final_vehicle_max_times": bfvt_for_summary,   ### <--- [修改] 3. 使用处理后的变量
#         "best_final_global_max_time": float(best_final_global_max_time) if best_final_global_max_time is not None else None,

#         # === 新增: 最终方案曲线长度信息 ===
#         "len_final_uav_cost": len(final_uav_cost) if final_uav_cost is not None else 0,
#         "len_final_total_list": len(final_total_list) if final_total_list is not None else 0,
#         "len_final_win_cost": len(final_win_cost) if final_win_cost is not None else 0,
#         "len_final_total_objective": len(final_total_objective) if final_total_objective is not None else 0,
#         "len_final_vehicle_route_cost": len(final_vehicle_route_cost) if final_vehicle_route_cost is not None else 0, ### <--- [新增] 4. 记录长度
#         "len_work_time": len(work_time) if work_time is not None else 0,
#         "len_final_work_time": len(final_work_time) if final_work_time is not None else 0,
#     }
#     # =========================
#     # 3. 保存概览到 TXT（统一 json-friendly）
#     # =========================
#     summary_txt_path = f"{prefix}_summary.txt"

#     with open(summary_txt_path, "w", encoding="utf-8") as f:
#         f.write("==== ALNS Result Summary ====\n")
#         for k, v in summary_data.items():
#             if isinstance(v, (dict, list)):
#                 v_clean = make_json_friendly(v)
#                 f.write(f"{k}:\n{json.dumps(v_clean, ensure_ascii=False, indent=2)}\n\n")
#             else:
#                 f.write(f"{k}: {v}\n")

#     # =========================
#     # 4. 保存曲线到 Excel
#     # =========================
#     curves_xlsx_path = f"{prefix}_curves.xlsx"

#     def _pad_list(lst: List[Any], target_len: int) -> List[Any]:
#         if lst is None:
#             return [None] * target_len
#         lst = list(lst)
#         if len(lst) < target_len:
#             lst = lst + [None] * (target_len - len(lst))
#         return lst

#     # max_len = max(
#     #     len(y_cost),
#     #     len(y_best),
#     #     len(win_cost),
#     #     len(uav_route_cost),
#     #     len(vehicle_route_cost),
#     # )
#     # === 新增: 把最终方案曲线也考虑进最大长度 ===
#     max_len = max(
#         len(y_cost),
#         len(y_best),
#         len(win_cost),
#         len(uav_route_cost),
#         len(vehicle_route_cost),
#         len(final_uav_cost) if final_uav_cost is not None else 0,
#         len(final_total_list) if final_total_list is not None else 0,
#         len(final_win_cost) if final_win_cost is not None else 0,
#         len(final_total_objective) if final_total_objective is not None else 0,
#         # === [新增] ===
#         # len(final_vehicle_route_cost) if final_vehicle_route_cost is not None else 0,
#         len(final_vehicle_route_cost) if final_vehicle_route_cost is not None else 0, ### <--- [新增] 5. 加入最大长度计算
#         len(work_time) if work_time is not None else 0,
#         len(final_work_time) if final_work_time is not None else 0,
#     )

#     df_curves = pd.DataFrame({
#         "iteration": list(range(max_len)),
#         "y_cost": _pad_list(y_cost, max_len),
#         "y_best": _pad_list(y_best, max_len),
#         "win_cost": _pad_list(win_cost, max_len),
#         "uav_route_cost": _pad_list(uav_route_cost, max_len),
#         "vehicle_route_cost": _pad_list(vehicle_route_cost, max_len),
#         # === 新增: 最终方案相关曲线 ===
#         "final_uav_cost": _pad_list(final_uav_cost, max_len),
#         "final_total_list": _pad_list(final_total_list, max_len),
#         "final_win_cost": _pad_list(final_win_cost, max_len),
#         "final_total_objective": _pad_list(final_total_objective, max_len),
#         # === [新增] ===
#         # "final_vehicle_route_cost": _pad_list(final_vehicle_route_cost, max_len),
#         "final_vehicle_route_cost": _pad_list(final_vehicle_route_cost, max_len), ### <--- [新增] 6. 写入 DataFrame
#         # === 新增: 完成时间曲线 ===
#         "work_time": _pad_list(work_time, max_len),
#         "final_work_time": _pad_list(final_work_time, max_len),
#     })

#     # 策略权重
#     df_strategy_weights = pd.DataFrame(
#         [{"strategy": k, "weight": v} for k, v in strategy_weights.items()]
#     )

#     # 算子权重展开
#     op_rows = []
#     for strat, group in operator_weights.items():
#         for op_type, w_dict in group.items():
#             for op_name, w in w_dict.items():
#                 op_rows.append({
#                     "strategy": strat,
#                     "op_type": op_type,
#                     "operator": op_name,
#                     "weight": w,
#                 })
#     df_operator_weights = pd.DataFrame(op_rows)

#     with pd.ExcelWriter(curves_xlsx_path) as writer:
#         df_curves.to_excel(writer, sheet_name="curves", index=False)
#         df_strategy_weights.to_excel(writer, sheet_name="strategy_weights", index=False)
#         df_operator_weights.to_excel(writer, sheet_name="operator_weights", index=False)

#     # =========================
#     # 5. 保存 summary + 违约成本 + total_cost_dict
#     # =========================
#     summary_xlsx_path = f"{prefix}_summary.xlsx"

#     # 每辆车完成时间：兼容 dict / 标量
#     if best_vehicle_max_times is not None and hasattr(best_vehicle_max_times, "items"):
#         # dict: {veh_id: time}
#         df_vehicle_finish = pd.DataFrame(
#             [
#                 {"vehicle_id": vid, "completion_time": t}
#                 for vid, t in best_vehicle_max_times.items()
#             ]
#         )
#     else:
#         # 标量：只知道“全局最大完成时间”，没有按车分开
#         df_vehicle_finish = pd.DataFrame(
#             [
#                 {"vehicle_id": "all", "completion_time": best_vehicle_max_times}
#             ]
#         )

#     # 时间窗惩罚细节（dict / defaultdict）
#     if best_uav_tw_violation_cost is not None and hasattr(best_uav_tw_violation_cost, "items"):
#         df_tw_violation = pd.DataFrame(
#             [{"key": k, "violation_cost": float(v)} for k, v in best_uav_tw_violation_cost.items()]
#         )
#     else:
#         df_tw_violation = pd.DataFrame(columns=["key", "violation_cost"])

#     # total_cost_dict：defaultdict(float)
#     if best_total_cost_dict is not None and hasattr(best_total_cost_dict, "items"):
#         df_total_cost_dict = pd.DataFrame(
#             [{"key": k, "value": float(v)} for k, v in best_total_cost_dict.items()]
#         )
#     else:
#         df_total_cost_dict = pd.DataFrame(columns=["key", "value"])

#     df_summary_scalar = pd.DataFrame(
#         [
#             {
#                 "instance_name": instance_name,
#                 "best_objective": best_objective,
#                 "elapsed_time": elapsed_time,
#                 "best_global_max_time": best_global_max_time,
#                 "best_window_total_cost": best_window_total_cost,
#                 "best_state_total_cost_attr": getattr(best_state, "_total_cost", None),
#                 # === 新增: 最终方案标量指标也写到 summary sheet 里 ===
#                 "best_final_uav_cost": best_final_uav_cost,
#                 "best_final_objective": best_final_objective,
#                 "best_final_win_cost": best_final_win_cost,
#                 "best_total_win_cost": best_total_win_cost,
#                 # === 新增: 完成时间相关最终标量 ===
#                 # "best_final_vehicle_max_times": best_final_vehicle_max_times,
#                 "best_final_vehicle_max_times": str(best_final_vehicle_max_times), ### <--- [修改] 7. 字典转字符串以避免写入错误
#                 "best_final_global_max_time": best_final_global_max_time,
#             }
#         ]
#     )

#     with pd.ExcelWriter(summary_xlsx_path) as writer:
#         df_summary_scalar.to_excel(writer, sheet_name="summary", index=False)
#         df_vehicle_finish.to_excel(writer, sheet_name="vehicle_finish_time", index=False)
#         df_tw_violation.to_excel(writer, sheet_name="tw_violation", index=False)
#         df_total_cost_dict.to_excel(writer, sheet_name="total_cost_detail", index=False)

#     # =========================
#     # 6. best_arrive_time
#     # =========================
#     arrive_xlsx_path = f"{prefix}_best_arrive_time.xlsx"
#     arrive_rows = []
#     for vid, node_times in best_arrive_time.items():
#         for node_id, t in node_times.items():
#             arrive_rows.append(
#                 {"vehicle_id": vid, "node_id": node_id, "arrive_time": t}
#             )
#     df_arrive = pd.DataFrame(arrive_rows)
#     df_arrive.to_excel(arrive_xlsx_path, index=False, sheet_name="arrive_time")

#     # =========================
#     # 7. best_state 核心结构
#     # =========================
#     best_state_xlsx_path = f"{prefix}_best_state.xlsx"

#     # 7.1 customer_plan
#     cp_rows = []
#     for cid, assign in best_state.customer_plan.items():
#         row = {"customer_id": cid}
#         if isinstance(assign, (list, tuple)):
#             for i, v in enumerate(assign):
#                 row[f"field_{i}"] = v
#         else:
#             row["assignment"] = assign
#         cp_rows.append(row)
#     df_customer_plan = pd.DataFrame(cp_rows)

#     # 7.2 uav_cost
#     uav_cost = getattr(best_state, "uav_cost", {})
#     if isinstance(uav_cost, dict):
#         df_uav_cost = pd.DataFrame(
#             [{"uav_id": k, "cost": v} for k, v in uav_cost.items()]
#         )
#     else:
#         df_uav_cost = pd.DataFrame(columns=["uav_id", "cost"])

#     # 7.3 vehicle_routes：兼容 list / dict
#     vr = getattr(best_state, "vehicle_routes", [])
#     vr_rows = []
#     if isinstance(vr, dict):
#         for vid, route in vr.items():
#             for idx, node in enumerate(route):
#                 vr_rows.append(
#                     {"vehicle_id": vid, "seq": idx, "node_id": node}
#                 )
#     else:
#         for i, route in enumerate(vr):
#             vid = i + 1
#             for idx, node in enumerate(route):
#                 vr_rows.append(
#                     {"vehicle_id": vid, "seq": idx, "node_id": node}
#                 )
#     df_vehicle_routes = pd.DataFrame(vr_rows)

#     # 7.4 uav_plan：结构不管，直接 json / repr
#     uav_plan = getattr(best_state, "uav_plan", None)
#     try:
#         uav_plan_json = json.dumps(make_json_friendly(uav_plan), ensure_ascii=False)
#     except TypeError:
#         uav_plan_json = repr(uav_plan)
#     df_uav_plan = pd.DataFrame([{"uav_plan_json": uav_plan_json}])

#     # 7.5 scalar
#     df_state_scalar = pd.DataFrame(
#         [
#             {
#                 "_total_cost_attr": getattr(best_state, "_total_cost", None),
#                 "objective_now": best_state.objective()
#                 if hasattr(best_state, "objective")
#                 else None,
#             }
#         ]
#     )
#     # === 新增: 7.6 保存最终方案的 final_uav_plan 详细信息 ===
#     final_uav_plan = getattr(best_state, "final_uav_plan", None)

#     with pd.ExcelWriter(best_state_xlsx_path) as writer:
#         df_customer_plan.to_excel(writer, sheet_name="customer_plan", index=False)
#         df_uav_cost.to_excel(writer, sheet_name="uav_cost", index=False)
#         df_vehicle_routes.to_excel(writer, sheet_name="vehicle_routes", index=False)
#         df_uav_plan.to_excel(writer, sheet_name="uav_plan_raw", index=False)
#         df_state_scalar.to_excel(writer, sheet_name="state_scalar", index=False)

#         if final_uav_plan is not None:
#             final_rows = []
#             for key, info in final_uav_plan.items():
#                 row = {}

#                 # key 是 tuple: (drone_id, launch_node, customer, recovery_node, launch_vehicle, recovery_vehicle)
#                 if isinstance(key, tuple) and len(key) >= 6:
#                     row["key_drone_id"] = key[0]
#                     row["key_launch_node"] = key[1]
#                     row["key_customer"] = key[2]
#                     row["key_recovery_node"] = key[3]
#                     row["key_launch_vehicle"] = key[4]
#                     row["key_recovery_vehicle"] = key[5]
#                 else:
#                     row["key_raw"] = str(key)

#                 if isinstance(info, dict):
#                     row["drone_id"] = info.get("drone_id")
#                     row["launch_vehicle"] = info.get("launch_vehicle")
#                     row["recovery_vehicle"] = info.get("recovery_vehicle")
#                     row["launch_node"] = info.get("launch_node")
#                     row["recovery_node"] = info.get("recovery_node")
#                     row["customer"] = info.get("customer")
#                     row["launch_time"] = info.get("launch_time")
#                     row["recovery_time"] = info.get("recovery_time")
#                     row["energy"] = info.get("energy")
#                     row["cost"] = info.get("cost")
#                     row["time"] = info.get("time")
#                     row["uav_route_cost"] = info.get("uav_route_cost")
#                     row["uav_time_cost"] = info.get("uav_time_cost")

#                     route = info.get("uav_route")
#                     try:
#                         row["uav_route"] = json.dumps(make_json_friendly(route), ensure_ascii=False)
#                     except Exception:
#                         row["uav_route"] = str(route)

#                     try:
#                         row["uav_route_len"] = len(route) if route is not None else 0
#                     except TypeError:
#                         row["uav_route_len"] = None
#                 else:
#                     row["info_raw"] = str(info)

#                 final_rows.append(row)

#             df_final_uav = pd.DataFrame(final_rows)
#             df_final_uav.to_excel(writer, sheet_name="final_uav_plan", index=False)
#     # with pd.ExcelWriter(best_state_xlsx_path) as writer:
#     #     df_customer_plan.to_excel(writer, sheet_name="customer_plan", index=False)
#     #     df_uav_cost.to_excel(writer, sheet_name="uav_cost", index=False)
#     #     df_vehicle_routes.to_excel(writer, sheet_name="vehicle_routes", index=False)
#     #     df_uav_plan.to_excel(writer, sheet_name="uav_plan_raw", index=False)
#     #     df_state_scalar.to_excel(writer, sheet_name="state_scalar", index=False)

#     print(f"[save_alns_results] 结果已保存到目录: {case_dir}")


def update_init_vehicle_task_data(new_vehicle_task_data, state):
    """
    更新车辆的prcise_time内容，根据state的相关信息内容
    """
    customer_plan = state.customer_plan
    vehicle_routes = state.vehicle_routes
    vehicle_id = state.T
    uav_id = state.V
    vehicle_arrive_time = state.rm_empty_vehicle_arrive_time
    # 更新后续的vehicle_task_data的精确时间
    for veh_id in vehicle_id:
        vehicle_route = vehicle_routes[veh_id-1]
        for node_id in vehicle_route:
            if node_id in new_vehicle_task_data[veh_id]:
                new_vehicle_task_data[veh_id][node_id].prcise_arrive_time = vehicle_arrive_time[veh_id][node_id]
                new_vehicle_task_data[veh_id][node_id].prcise_departure_time = vehicle_arrive_time[veh_id][node_id]
            else:
                print(f"node_id {node_id} not in new_vehicle_task_data[veh_id]")
                new_vehicle_task_data[veh_id][node_id].prcise_arrive_time = 0
                new_vehicle_task_data[veh_id][node_id].prcise_departure_time = 0
    return new_vehicle_task_data

def update_init_sorted_mission_keys(state):
    """
    按照车辆到达时间排序任务表格，并返回排序后的任务表格
    """
    # 1. 获取数据源
    # 注意：根据你提供的框架，这里使用 state.rm_empty_vehicle_arrive_time
    # 逻辑上它对应你给出的 vehicle_arrive_time 数据结构
    vehicle_arrive_time = state.rm_empty_vehicle_arrive_time
    customer_plan = state.customer_plan

    # 2. 构建临时列表 (到达时间, 任务数据元组)
    # plan 索引含义: [0:无人机, 1:发射点, 2:客户点, 3:回收点, 4:发射车, 5:回收车]
    temp_list = []
    
    for plan in customer_plan.values():
        launch_node = plan[1]      # 获取发射节点
        launch_vehicle = plan[4]   # 获取发射车辆ID
        
        # 从时间表中获取时间
        # 结构: {vehicle_id: {node_id: time}}
        try:
            arrive_time = vehicle_arrive_time[launch_vehicle][launch_node]
        except KeyError:
            # 防御性编程：如果找不到对应的时间，赋予无穷大，将其排到最后
            arrive_time = float('inf')
            
        # 将 plan (list) 转换为 tuple，符合输出格式要求
        temp_list.append((arrive_time, tuple(plan)))

    # 3. 根据到达时间（元组的第0个元素）进行升序排序
    temp_list.sort(key=lambda x: x[0])

    # 4. 提取排序后的任务详情，生成最终列表
    sorted_mission_keys = [item[1] for item in temp_list]

    return sorted_mission_keys

import numpy as np


import os
import json
from typing import Any, Dict, List
import pandas as pd

try:
    import numpy as np
except ImportError:
    np = None


def make_json_friendly(obj: Any) -> Any:
    """
    递归把对象转成 json 友好类型：
    - numpy 标量 -> Python 标量
    - defaultdict / dict -> 普通 dict，key 统一转 str
    - 其他奇怪类型 -> str
    """

    # 基础类型直接返回
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # numpy 标量
    if np is not None and isinstance(obj, np.generic):
        # np.int32 / np.int64 / np.float64 等
        if "int" in type(obj).__name__.lower():
            return int(obj)
        if "float" in type(obj).__name__.lower():
            return float(obj)
        return obj.item()

    # 容器类型
    if isinstance(obj, (list, tuple, set)):
        return [make_json_friendly(x) for x in obj]

    # 字典或 defaultdict
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            k_clean = make_json_friendly(k)
            v_clean = make_json_friendly(v)
            new_dict[str(k_clean)] = v_clean
        return new_dict

    # 其他统统用字符串兜底
    return str(obj)

from collections import defaultdict
from typing import Dict, Any

def extract_arrive_time_from_plan(
    vehicle_plan_time: Dict[Any, Dict[Any, list]]
) -> Dict[int, Dict[int, float]]:
    """
    将 final_vehicle_plan_time 这种结构：
        {veh_id: {node_id: [start_time, end_time]}}
    转换为：
        {veh_id: {node_id: arrive_time}}
    其中 arrive_time 取列表的第一个元素（start_time）。

    参数
    ----
    vehicle_plan_time : dict-like
        类似 defaultdict(int -> defaultdict(node -> [start, end])) 的结构。

    返回
    ----
    dict[int, dict[int, float]]
        普通 dict，key 为 int 的车和节点，值为到达时间（float）。
    """
    result: Dict[int, Dict[int, float]] = {}

    if vehicle_plan_time is None:
        return result

    # 兼容 defaultdict / 普通 dict
    for veh_id, node_dict in vehicle_plan_time.items():
        # 车 id 转成 int，防止后面 JSON / Excel 之类乱套
        try:
            veh_key = int(veh_id)
        except (TypeError, ValueError):
            veh_key = veh_id

        result[veh_key] = {}

        # node_dict 是 {node_id: [start, end]}
        for node_id, times in node_dict.items():
            try:
                node_key = int(node_id)
            except (TypeError, ValueError):
                node_key = node_id

            # times 正常是 [start, end]，我们取第一个
            if isinstance(times, (list, tuple)) and len(times) > 0:
                arrive_time = float(times[0])
            else:
                # 防御式兜底：如果不是 list，就直接转成 float 试试
                try:
                    arrive_time = float(times)
                except (TypeError, ValueError):
                    # 实在不行就跳过/置 None，这里选 None 更安全
                    arrive_time = None

            result[veh_key][node_key] = arrive_time

    return result

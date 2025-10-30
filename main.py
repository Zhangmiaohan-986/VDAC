#!/usr/bin/env python

# INSTALLING ADDITIONAL PYTHON MODULES:
#
# pandas:
#	sudo pip install pandas



# RUNNING THIS SCRIPT:

# python main.py <problemName> <vehicleFileID> <UAVSpeedType> <numUAVs> <numTrucks>

# problemName: Name of the folder containing the data for a particular problem instance
# vehicleFileID: 101, 102, 103, 104 (Chooses a particular UAV type depending on the file ID)
# UAVSpeedType: 1 (variable), 2 (maximum), or 3 (maximum-range)
# numUAVs: Number of UAVs available in the problem
# numTrucks: Number of trucks available in the problem (currently only solvable for 1 truck)

# An example - Solving the mFSTSP-VDS:
#    python main.py 20170608T121632668184 101 1 3 -1
#		numTrucks is ignored
#		The UAVs are defined in tbl_vehicles.  If you ask for more UAVs than are defined, you'll get a warning.


import sys
import datetime
import time
import math
from collections import defaultdict
import pandas as pd

from parseCSV import *
from parseCSVstring import *

from gurobipy import *
import os
import os.path
from subprocess import call		# allow calling an external command in python.  See http://stackoverflow.com/questions/89228/calling-an-external-command-in-python
from task_data import *
from cost_y import *
from call_function import *
from initialize import *
from cbs_plan import *
# from insert_plan import *
from down_data import *
from solve_mfstsp_heuristic import *

import distance_functions

from generate_test_problems import *
# =============================================================
startTime 		= time.time()

METERS_PER_MILE = 1609.34


UAVSpeedTypeString = {1: 'variable', 2: 'maximum', 3: 'maximum-range'}


NODE_TYPE_DEPOT	= 0
NODE_TYPE_CUST	= 1

TYPE_TRUCK 		= 1
TYPE_UAV 		= 2

# NUM_POINTS = 50
NUM_POINTS = 100
SEED = 6
Z_COORD = 0.05  # 规划无人机空中高度情况
UAV_DISTANCE = 15

# =============================================================

def make_dict():# 设计实现一个可以无线嵌套的dict
	return defaultdict(make_dict)

	# Usage:
	# tau = defaultdict(make_dict)
	# v = 17
	# i = 3
	# j = 12
	# tau[v][i][j] = 44

class make_node: 
	def __init__(self, index, nodeType, latDeg, lonDeg, altMeters, parcelWtLbs, serviceTimeTruck, serviceTimeUAV, map_key, map_type, map_position, map_index):
		# Set node[nodeID]
		self.index				= index
		self.nodeType 			= nodeType
		self.latDeg 			= latDeg # 对应x坐标
		self.lonDeg				= lonDeg # 对应y坐标
		self.altMeters			= altMeters # 对应z坐标
		self.position			= (latDeg, lonDeg, altMeters)  # 对应坐标
		self.parcelWtLbs 		= parcelWtLbs # 对应包裹重量
		self.serviceTimeTruck	= serviceTimeTruck	# [seconds]
		self.serviceTimeUAV 	= serviceTimeUAV	# [seconds]
		if nodeType == 'DEPOT':  # 如果节点类型为DEPOT
			self.map_key 			= index			# Might be None
			self.map_type 			= 'DEPOT'			# Might be None
			self.map_position 		= (latDeg, lonDeg, altMeters)		# Might be None
			self.map_index 			= map_index			# Might be None
		else:
			self.map_key 			= map_key			# Might be None
			self.map_type 			= map_type			# Might be None
			self.map_position 		= map_position		# Might be None
			self.map_index 			= map_index			# Might be None

class make_vehicle:  # 添加车辆和无人机的各类属性集合
	def __init__(self, id, vehicleType, takeoffSpeed, cruiseSpeed, landingSpeed, yawRateDeg, cruiseAlt, capacityLbs, launchTime, recoveryTime, serviceTime, batteryPower, flightRange, vehicleSpeed, maxDrones, per_cost,fix_cost=10):
		# Set vehicle[vehicleID]
		self.vehicleType	= vehicleType
		# 将速度转变为Km/h
		self.takeoffSpeed	= takeoffSpeed * 3.6
		self.cruiseSpeed	= cruiseSpeed * 3.6
		self.landingSpeed	= landingSpeed * 3.6
		self.yawRateDeg		= yawRateDeg
		self.cruiseAlt		= cruiseAlt
		self.capacityLbs	= capacityLbs
		self.launchTime		= launchTime / 3600 	# [seconds].
		self.recoveryTime	= recoveryTime / 3600	# [seconds].
		self.serviceTime	= serviceTime / 60	# [minutes].转变为小时
		self.batteryPower	= batteryPower	# [joules].
		self.flightRange	= flightRange	# 'high' or 'low'
		self.vehicleSpeed	= vehicleSpeed*3.6  # 无人机速度为-1，卡车速度为10
		self.per_cost = per_cost
		self.fix_cost = fix_cost
		self.time_cost = 2  # 定义无人机单位飞行成本价格
		# 增改
		self.id = id # 给出车辆ID
		self.route = defaultdict(make_dict)
		self.current_time = 0
		self.current_node = None
		self.entry_node_time = defaultdict(make_dict)
		self.exit_node_time = defaultdict(make_dict)
		self.dict_vehicle = {}
		if self.vehicleType == TYPE_TRUCK:  # 如何判断车辆类型
			self.drones = defaultdict(make_dict)  # 车辆在节点携带的无人机列表
			self.launch_records = defaultdict(make_dict)  # 发射记录，键为节点编号，值为发射的无人机列表
			self.recover_records = defaultdict(make_dict)  # 回收记录，键为节点编号，值为回收的无人机列表
			self.available_drones = defaultdict(make_dict)  # 可用无人机列表
			self.maxDrones = maxDrones  # 无人机最大数量
		elif self.vehicleType == TYPE_UAV:
			self.is_available = True  # 无人机是否可用
			self.returnable_nodes = []  # 可返回的节点列表
			self.violate = False  # 判断是否违背了约束条件
	# 		self.init_dict_vehicle()
	# def init_dict_vehicle(self):
	# 	"""初始化无人机的dict_vehicle"""
	# 	if self.vehicleType == TYPE_UAV:
	# 		class VehicleInfo:
	# 			def __init__(self):
	# 				self.drone_belong = None  # 无人机属于哪辆车
	# 				self.precise_arrive_time = 0     # 精确时间
	# 				self.precise_departure_time = 0
	# 				self.launch_time = None   # 发射时间
	# 				self.recovery_time = [] # 回收时间
	# 		# 为每个可能的车辆ID创建一个VehicleInfo实例
	# 	for vehicle_id in range(1, 3):  # 假设最多100辆车，可以根据实际情况调整, 后期修改车辆数据这里需要进一步修改
	# 		self.dict_vehicle[vehicle_id] = VehicleInfo()

	def add_drone(self, drone):
		self.drones[drone.id] = drone
	def remove_drone(self, drone):
		del self.drones[drone.id]

class make_travel:
	def __init__(self, takeoffTime, flyTime, landTime, totalTime, takeoffDistance, flyDistance, landDistance, totalDistance, path):
		# Set travel[vehicleID][fromID][toID]
		self.takeoffTime 	 = takeoffTime
		self.flyTime 		 = flyTime
		self.landTime 		 = landTime
		self.totalTime 		 = totalTime
		self.takeoffDistance = takeoffDistance
		self.flyDistance	 = flyDistance
		self.landDistance	 = landDistance
		self.totalDistance	 = totalDistance
		self.path			 = path

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

# 输出:
# {0: "airplane", 1: "drone", 2: "tank", 3: "truck"}

class missionControl():
	def __init__(self):
		
		timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
		# 获取基准路径（建议在类外部定义）
		self.base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件绝对路径的目录
		if (len(sys.argv) == 6):  # 给出各种参数，包括无人机数量
			problemName 		= sys.argv[1]
			vehicleFileID		= int(sys.argv[2])
			UAVSpeedType 		= int(sys.argv[3])
			# numUAVs				= int(sys.argv[4])
			# numTrucks			= int(sys.argv[5])
			# 使用 os.path.join 构建跨平台兼容路径
			self.locationsFile = os.path.join(self.base_dir, 'Problems', problemName, 'tbl_locations.csv')
			# self.vehiclesFile = os.path.join(self.base_dir, 'Problems', f'tbl_vehicles_{vehicleFileID}.csv')
			self.vehiclesFile = os.path.join(self.base_dir, 'Problems', f'tbl_vehicles_tits.csv')
			self.distmatrixFile = os.path.join(self.base_dir, 'Problems', problemName, 'tbl_truck_travel_data_PG.csv')
			# self.solutionSummaryFile = os.path.join(
			# 	self.base_dir, 'Problems', problemName, 
			# 	f'tbl_solutions_{vehicleFileID}_{numUAVs}_{UAVSpeedTypeString[UAVSpeedType]}.csv'
			# )
		else:
			print(f'ERROR: Expected 5 parameters, got {len(sys.argv)-1}.')
			quit()	
		# 参数输入设置
		num_points = 100
		seed = 6
		Z_coord = 0.05  # 空中走廊高度
		uav_distance = 15  # 无人机最远飞行距离
		numUAVs = 6  # 无人机数量
		UAVSpeedType = 1
		numTrucks = 3  # 卡车数量，在此处修改卡车数量和无人机
		per_uav_cost = 1  # 每公里成本
		per_vehicle_cost = 1  # 每公里成本
		max_Drones = 10  # 无人机最大数量
		# Define data structures
		self.node = {}
		self.vehicle = {}
		self.uav_travel = defaultdict(make_dict) # 创建一个默认字典，用于存储车辆之间的旅行时间矩阵
		self.veh_travel = defaultdict(make_dict) # 创建一个默认字典，用于存储车辆之间的旅行时间矩阵
		self.veh_distance = defaultdict(make_dict) # 创建一个默认字典，用于存储车辆之间的距离矩阵

		# Read data for node locations, vehicle properties, and travel time matrix of truck:
		self.readData(numUAVs, numTrucks, per_uav_cost, per_vehicle_cost, max_Drones) # 读取节点位置、车辆属性以及卡车旅行时间矩阵
		# 计算无人机之间的旅行时间矩阵
		for vehicleID in self.vehicle:
			if (self.vehicle[vehicleID].vehicleType == TYPE_UAV):
				for i in self.node:
					if self.node[i].nodeType == 'VTP Takeoff/Landing Node' or self.node[i].nodeType == 'CUSTOMER Takeoff/Landing Node':
						for j in self.node:
							if self.node[j].nodeType == 'VTP Takeoff/Landing Node' or self.node[j].nodeType == 'CUSTOMER Takeoff/Landing Node':
								if (j == i):
									self.uav_travel[vehicleID][i][j] = main.make_travel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
								else:
									init_node = self.node[i]
									target_node = self.node[j]
									[takeoffTime, flyTime, landTime, totalTime, takeoffDistance, flyDistance, landDistance, totalDistance, path] = distance_functions.calcMultirotorTravelTime(self.vehicle[vehicleID].takeoffSpeed, self.vehicle[vehicleID].cruiseSpeed, self.vehicle[vehicleID].landingSpeed, self.vehicle[vehicleID].yawRateDeg, self.node[i].altMeters, self.vehicle[vehicleID].cruiseAlt, self.node[j].altMeters, self.node[i].latDeg, self.node[i].lonDeg, self.node[j].latDeg, self.node[j].lonDeg, -361, -361, self.G_air, self.G_ground, init_node, target_node)
									self.uav_travel[vehicleID][i][j] = main.make_travel(takeoffTime, flyTime, landTime, totalTime, takeoffDistance, flyDistance, landDistance, totalDistance, path)
		# 计算卡车之间的旅行时间矩阵 
		for vehicleID in self.vehicle:
			if (self.vehicle[vehicleID].vehicleType == TYPE_TRUCK):
				for i in self.node:
					if self.node[i].nodeType == 'VTP' or self.node[i].nodeType == 'DEPOT':
						for j in self.node:
							if self.node[j].nodeType == 'VTP' or self.node[j].nodeType == 'DEPOT':
								if (j == i):
									self.veh_distance[vehicleID][i][j] = 0
									self.veh_distance[vehicleID][j][i] = 0
									self.veh_travel[vehicleID][i][j] = 0
									self.veh_travel[vehicleID][j][i] = 0
								else:
									init_node =  self.node[i].map_index
									target_node = self.node[j].map_index 
									distance = self.ground_matrix[init_node][target_node]
									self.veh_distance[vehicleID][i][j] = distance
									self.veh_travel[vehicleID][i][j] = distance / self.vehicle[vehicleID].vehicleSpeed

		
		# Now, call the heuristic:
		print('Calling a Heuristic to solve VDCD-AC...')
		# 调用启发式算法解决mFSTSP-VDS问题
		# [objVal, assignments, packages, waitingTruck, waitingUAV] = solve_mfstsp_heuristic(self.node, self.vehicle, self.travel, problemName, vehicleFileID, numUAVs, UAVSpeedType)
		[objVal, assignments, packages, costTruck, costUAV] = solve_mfstsp_heuristic(self.node, self.vehicle, self.air_matrix, self.ground_matrix, self.air_node_types, self.ground_node_types, numUAVs, numTrucks, self.uav_travel, self.veh_travel, self.veh_distance, self.G_air, self.G_ground)
		print('The mFSTSP-VDS Heuristic is Done.  It returned something')


		numUAVcust			= 0
		numTruckCust		= 0		
		for nodeID in packages:
			if (self.node[nodeID].nodeType == NODE_TYPE_CUST):
				if (packages[nodeID].packageType == TYPE_UAV):
					numUAVcust		+= 1
				else:
					numTruckCust	+= 1

			
		# Write in the performance_summary file:
		total_time = time.time() - startTime
		print("Total time for the whole process: %f" % (total_time))
		print("Objective Function Value: %f" % (objVal))

		runString = ' '.join(sys.argv[0:])

		myFile = open('performance_summary.csv','a')
		str = '%s, %d, %d, %s, %d,' % (problemName, vehicleFileID, numUAVs, UAVSpeedTypeString[UAVSpeedType], numTrucks)
		myFile.write(str)

		numCustomers = len(self.node) - 2
		str = '%d, %s, %f, %f, %d, %d, %f, %f \n' % (numCustomers, timestamp, objVal, total_time, numUAVcust, numTruckCust, waitingTruck, waitingUAV)
		myFile.write(str)
					
		myFile.close()
		print("\nSee 'performance_summary.csv' for statistics.\n")


		# Write in the solution file:
		myFile = open(self.solutionSummaryFile, 'a')
		myFile.write('problemName,vehicleFileID,UAVSpeedType,numUAVs,numTrucks \n')
		str = '%s, %d, %s, %d, %d \n\n' % (problemName, vehicleFileID, UAVSpeedTypeString[UAVSpeedType], numUAVs, numTrucks)
		myFile.write(str)

		myFile.write('Objective Function Value: %f \n\n' % (objVal))
		myFile.write('Assignments: \n')

		myFile.close()

		# Create a dataframe to sort assignments according to their start times:
		assignDF = pd.DataFrame(columns=['vehicleID', 'vehicleType', 'activityType', 'startTime', 'startNode', 'endTime', 'endNode', 'Description', 'Status'])
		indexDF = 1

		for v in assignments:
			for statusID in assignments[v]:
				for statusIndex in assignments[v][statusID]:
					if (assignments[v][statusID][statusIndex].vehicleType == TYPE_TRUCK):
						vehicleType = 'Truck'
					else:
						vehicleType = 'UAV'
					if (statusID == TRAVEL_UAV_PACKAGE):
						status = 'UAV travels with parcel'
					elif (statusID == TRAVEL_UAV_EMPTY):
						status = 'UAV travels empty'
					elif (statusID == TRAVEL_TRUCK_W_UAV):
						status = 'Truck travels with UAV(s) on board'
					elif (statusID == TRAVEL_TRUCK_EMPTY):
						status = 'Truck travels with no UAVs on board'
					elif (statusID == VERTICAL_UAV_EMPTY):
						status = 'UAV taking off or landing with no parcels'
					elif (statusID == VERTICAL_UAV_PACKAGE):
						status = 'UAV taking off or landing with a parcel'
					elif (statusID == STATIONARY_UAV_EMPTY):
						status = 'UAV is stationary without a parcel'
					elif (statusID == STATIONARY_UAV_PACKAGE):
						status = 'UAV is stationary with a parcel'
					elif (statusID == STATIONARY_TRUCK_W_UAV):
						status = 'Truck is stationary with UAV(s) on board'
					elif (statusID == STATIONARY_TRUCK_EMPTY):
						status = 'Truck is stationary with no UAVs on board'
					else:
						print('UNKNOWN statusID.')
						quit()

					
					if (assignments[v][statusID][statusIndex].ganttStatus == GANTT_IDLE):
						ganttStr = 'Idle'
					elif (assignments[v][statusID][statusIndex].ganttStatus == GANTT_TRAVEL):
						ganttStr = 'Traveling'
					elif (assignments[v][statusID][statusIndex].ganttStatus == GANTT_DELIVER):
						ganttStr = 'Making Delivery'
					elif (assignments[v][statusID][statusIndex].ganttStatus == GANTT_RECOVER):
						ganttStr = 'UAV Recovery'
					elif (assignments[v][statusID][statusIndex].ganttStatus == GANTT_LAUNCH):
						ganttStr = 'UAV Launch'
					elif (assignments[v][statusID][statusIndex].ganttStatus == GANTT_FINISHED):
						ganttStr = 'Vehicle Tasks Complete'
					else:
						print('UNKNOWN ganttStatus')
						quit()
					
					assignDF.loc[indexDF] = [v, vehicleType, status, assignments[v][statusID][statusIndex].startTime, assignments[v][statusID][statusIndex].startNodeID, assignments[v][statusID][statusIndex].endTime, assignments[v][statusID][statusIndex].endNodeID, assignments[v][statusID][statusIndex].description, ganttStr]	
					indexDF += 1
		
		assignDF = assignDF.sort_values(by=['vehicleID', 'startTime'])

		# Add this assignment dataframe to the solution file:
		assignDF.to_csv(self.solutionSummaryFile, mode='a', header=True, index=False)
		
		print("\nSee '%s' for solution summary.\n" % (self.solutionSummaryFile))

	# 读取车辆数据
	def readData(self, numUAVs, numTrucks, per_uav_cost, per_vehicle_cost, max_Drones): # 读取车辆数据
		# b)  tbl_vehicles.csv
		tmpUAVs = 0
		tmpTrucks = 0
		rawData = parseCSVstring(self.vehiclesFile, returnJagged=False, fillerValue=-1, delimiter=',', commentChar='%')
		for i in range(0,len(rawData)):
			vehicleID 			= int(rawData[i][0])
			vehicleType			= int(rawData[i][1])
			takeoffSpeed		= float(rawData[i][2])
			cruiseSpeed			= float(rawData[i][3])
			landingSpeed		= float(rawData[i][4])
			yawRateDeg			= float(rawData[i][5])  # 偏航率（度 / 秒）
			cruiseAlt			= float(rawData[i][6])  # 巡航高度（米）
			capacityLbs			= float(rawData[i][7])  # 载重能力（磅）
			launchTime			= float(rawData[i][8])	# [seconds].
			recoveryTime		= float(rawData[i][9])	# [seconds].
			serviceTime			= float(rawData[i][10])	# [seconds].
			batteryPower		= float(rawData[i][11])	# [joules].
			flightRange			= str(rawData[i][12])	# 'high' or 'low'
			maxDrones			= max_Drones	# 无人机最大数量
			if (vehicleType == TYPE_UAV):
				tmpUAVs += 1
				if (tmpUAVs <= numUAVs): # 判断应用无人机的个数，生成无人机的性能状态数据结构参数
					vehicleSpeed = -1
					self.vehicle[vehicleID] = make_vehicle(vehicleID, vehicleType, takeoffSpeed, cruiseSpeed, landingSpeed, yawRateDeg, cruiseAlt, capacityLbs, launchTime, recoveryTime, serviceTime, batteryPower, flightRange, vehicleSpeed, maxDrones, per_uav_cost)
				else:  # 如果无人机数量超过numUAVs，则跳出循环
					break
			else:
				tmpTrucks += 1
				if (tmpTrucks <= numTrucks):
					vehicleSpeed = 10
					self.vehicle[vehicleID] = make_vehicle(vehicleID, vehicleType, takeoffSpeed, cruiseSpeed, landingSpeed, yawRateDeg, cruiseAlt, capacityLbs, launchTime, recoveryTime, serviceTime, batteryPower, flightRange, vehicleSpeed, maxDrones, per_vehicle_cost)

		if (tmpUAVs < numUAVs):
			print("WARNING: You requested %d UAVs, but we only have data on %d UAVs." % (numUAVs, tmpUAVs))
			print("\t We'll solve the problem with %d UAVs.  Sorry." % (tmpUAVs))

		# a)  tbl_locations.csv
		# 获得空中地面图，空地距离矩阵，位置和节点类型，及所有数据集合。
		G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions, all_data, air_node_types, ground_node_types = generate_complex_network(NUM_POINTS, SEED, Z_COORD, UAV_DISTANCE)
		# 读取节点位置，类型数据
		# 使用字典构造函数
		air_ground_node_types =  merge_and_renumber_dicts(air_node_types, ground_node_types)
		air_ground_positions = merge_and_renumber_dicts(air_positions, ground_positions)
		len_air_ground_node_types = len(air_ground_node_types)
		len_air_node_types = len(air_node_types)
		len_ground_node_types = len(ground_node_types)
		serviceTimeTruck = 0
		for index, node_type in air_ground_node_types.items():
			nodeID = index
			nodeType = node_type
			latDeg = air_ground_positions[index][0]
			lonDeg = air_ground_positions[index][1]
			altMeters = air_ground_positions[index][2]
			# 如果是顾客点，随机生成1-5kg的货物物资需求
			if nodeType == 'CUSTOMER':
				parcelWtLbs = random.randint(1, 6)  # 随机生成1-5kg的货物物资需求
				serviceTimeUAV = 5  # 无人机服务时间
			else:
				parcelWtLbs = 0
				serviceTimeUAV = 0
			address = ''
			# if nodeType == 'VTP':  # 找到对应的空中映射节点
			map_result = find_same_xy_different_z(air_ground_positions, air_ground_positions[index])
			# 考虑空中中继点没有对应地面中继点的情况
			if map_result == None:
				map_key = None
				map_type = None
				map_position = None
				if index < len_air_node_types:
					map_index = index
				else:
					map_index = index - len_air_node_types  # map_index代表其对应空中和地面的分别的索引
			else:
				map_key = map_result[0]
				# map_index = map_result[1]
				map_position = map_result[2]
				map_type = air_ground_node_types[map_key]
				if index < len_air_node_types:
					map_index = index
				else:
					map_index = index - len_air_node_types

			self.node[nodeID] = make_node(index, nodeType, latDeg, lonDeg, altMeters, parcelWtLbs, serviceTimeTruck, serviceTimeUAV, map_key, map_type, map_position, map_index)
		# 计算各个类型节点间的距离矩阵
		self.air_matrix = air_adj_matrix
		self.ground_matrix = ground_adj_matrix
		self.air_node_types = air_node_types
		self.ground_node_types = ground_node_types
		self.G_air = G_air
		self.G_ground = G_ground

if __name__ == '__main__':
	try:
		missionControl()
	except: 
		print("There was a problem.  Sorry things didn't work out.  Bye.")
		raise
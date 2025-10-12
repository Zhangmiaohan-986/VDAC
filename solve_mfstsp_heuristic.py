#!/usr/bin/env python

import sys
import time
import datetime
import math
from parseCSV import *
from gurobipy import *
from collections import defaultdict
import copy
from initialize import init_agent, initialize_drone_vehicle_assignments
from create_vehicle_route import *

import os
# from main import find_keys_and_indices
from mfstsp_heuristic_1_partition import *
from mfstsp_heuristic_2_asgn_uavs import *
from mfstsp_heuristic_3_timing import *

from local_search import *
from rm_node_sort_node import rm_empty_node

import main
import endurance_calculator
import distance_functions

import random

# 
NODE_TYPE_DEPOT	= 0
NODE_TYPE_CUST	= 1

TYPE_TRUCK 		= 1
TYPE_UAV 		= 2
#

METERS_PER_MILE = 1609.34

# http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python
def make_dict():
	return defaultdict(make_dict)

class make_node:
	def __init__(self, nodeType, latDeg, lonDeg, altMeters, parcelWtLbs, serviceTimeTruck, serviceTimeUAV, address): 
		# Set node[nodeID]
		self.nodeType 			= nodeType
		self.latDeg 			= latDeg
		self.lonDeg				= lonDeg
		self.altMeters			= altMeters
		self.parcelWtLbs 		= parcelWtLbs
		self.serviceTimeTruck	= serviceTimeTruck	# [seconds]
		self.serviceTimeUAV 	= serviceTimeUAV	# [seconds]
		self.address 			= address			# Might be None...need MapQuest to give us this info later.

class make_assignments:
	def __init__(self, vehicleType, startTime, startNodeID, startLatDeg, startLonDeg, startAltMeters, endTime, endNodeID, endLatDeg, endLonDeg, endAltMeters, icon, description, UAVsOnBoard, ganttStatus):
		# Set assignments[v][statusID][statusIndex]
		self.vehicleType 	= vehicleType
		self.startTime 		= startTime
		self.startNodeID	= startNodeID
		self.startLatDeg 	= startLatDeg
		self.startLonDeg 	= startLonDeg
		self.startAltMeters = startAltMeters
		self.endTime 		= endTime
		self.endNodeID 		= endNodeID
		self.endLatDeg		= endLatDeg
		self.endLonDeg		= endLonDeg
		self.endAltMeters 	= endAltMeters
		self.icon			= icon
		self.description 	= description
		self.UAVsOnBoard 	= UAVsOnBoard
		self.ganttStatus	= ganttStatus

class make_packages:
	def __init__(self, packageType, latDeg, lonDeg, deliveryTime, icon):
		# Set packages[nodeID]
		self.packageType 	= packageType
		self.latDeg 		= latDeg
		self.lonDeg 		= lonDeg
		self.deliveryTime 	= deliveryTime
		self.icon 			= icon


# def solve_mfstsp_heuristic(node, vehicle, travel, problemName, vehicleFileID, numUAVs, UAVSpeedType):
def solve_mfstsp_heuristic(node, vehicle, air_matrix, ground_matrix, air_node_types, ground_node_types, numUAVs, numTrucks, uav_travel, veh_travel, veh_distance, G_air, G_ground):
	# 建立系统参数：
	C 			= [] # 客户列表
	tau			= defaultdict(make_dict) # 卡车旅行时间
	xtauprimeF	= defaultdict(make_dict) # 初始无人机旅行时间（带包裹）
	xtauprimeE	= defaultdict(make_dict) # 初始无人机旅行时间（不带包裹）
	xspeedF		= defaultdict(make_dict) # 初始无人机速度（带包裹）
	xspeedE 		= defaultdict(make_dict) # 初始无人机速度（不带包裹）
	xeee 		= defaultdict(make_dict) # 初始化无人机续航时间（in seconds）
	xeeePrime 	= defaultdict(make_dict) # 初始化无人机最大续航时间（in seconds）
	V			= []		# Set of UAVs.
	T = []
	N = []
	sL			= defaultdict(make_dict) # 无人机发射时间
	sR			= defaultdict(make_dict) # 无人机回收时间

	sigma		= {} # 卡车交付客户时间
	sigmaprime	= {} # 无人机交付客户时间


	for nodeID in node:	
		if (node[nodeID].nodeType == 'CUSTOMER'):
			C.append(nodeID)		# C is the vector of customer nodes.  C = [1, 2, ..., c],整理出是客户点的节点
			
	# 整理车辆及无人机的状态参数列表		
	for vehicleID in vehicle:
		if (vehicle[vehicleID].vehicleType == TYPE_UAV):
			V.append(vehicleID)
		else:
			T.append(vehicleID)
												
	c = len(C)				# c is the number of customers
	for key, nodeID in node.items():
		if nodeID.nodeType == 'DEPOT':
			DEPOT_nodeID = nodeID.index
			break

	for nodeID in node:
		if (node[nodeID].nodeType == 'VTP'):
			N.append(nodeID)		# N is the vector of customer nodes.  N = [1, 2, ..., c]
	N_zero = [DEPOT_nodeID] + N.copy()  # DEPOT_nodeID在开头
	N_plus = N.copy() + [DEPOT_nodeID]  # DEPOT_nodeID在末尾
	N_total = [DEPOT_nodeID] + N.copy() + [DEPOT_nodeID]  # 头尾都添加,后期试验最好前后返回的坐标一样，但是编号不一样
	
	A_vtp = []
	A_cust = []
	A_aerial_relay_node = []

	for key, nodeID in node.items():
		if node[key].nodeType == 'DEPOT':
			DEPOT_nodeID = key
		if node[key].nodeType == 'Aerial Relay Node':
			A_aerial_relay_node.append(key)
		if node[key].nodeType == 'VTP Takeoff/Landing Node':
			A_vtp.append(key)
		if node[key].nodeType == 'CUSTOMER Takeoff/Landing Node':
			A_cust.append(key)

	A_total = A_aerial_relay_node+A_vtp+A_cust  # 所有无人机节点
	A_cvtp = A_vtp+A_cust # 整理除中继点外其他的所有无人机节点
	A_c = A_cust
										
	# Build the set of all possible sorties: [v, i, j, k]
	# xP = {}
	# for sr in initializedSpeeds:
	# 	xP[sr] = []
	xP = {}	
	for v in V:		# 每个无人机可服务的范围
		xP[v] = []
		for i in A_vtp:
			for j in A_cust:
				if ((j != i) and (node[j].parcelWtLbs <= vehicle[v].capacityLbs)):
					for k in A_vtp:
						if (k != i) and (k != j):
							# xeee[v][i][j][k] = endurance_calculator.give_endurance(node, vehicle, uav_travel, v, i, j, k, xtauprimeF[v][i][j], xtauprimeE[v][j][k], xspeedF[v][i][j], xspeedE[v][j][k])
							# 返回任务完成后的剩余续航时间
							xeee[v][i][j][k] = endurance_calculator.give_endurance(node, vehicle, uav_travel, v, i, j, k, uav_travel[v][i][j].totalTime, uav_travel[v][j][k].totalTime, vehicle[v].cruiseSpeed, vehicle[v].cruiseSpeed)
							if (xeee[v][i][j][k] >= 0):
								xP[v].append([v,i,j,k])  # 得到所有可能的sortied,符合各个约束条件									

	# Build the launch service times:
	for v in V:
		for i in N:
			sL[v][i] = vehicle[v].launchTime
			
	# Build the recovery service times:
	for v in V:
		for k in N:
			sR[v][k] = vehicle[v].recoveryTime
	
	# 更新车辆携带无人机的方案
	vehicle = initialize_drone_vehicle_assignments(vehicle, V, T)

	# Build the customer service times:
	sigma[0] = 0.0
	for k in A_cust:
		sigmaprime[k] = node[k].serviceTimeUAV/60
	
	# 上述内容将基础参数全部处理完成，随后开始仿真实验处理
	iter_num = 10 # 仿真实验次数,便于计算平均值等方案
	# 初始化多次试验的仿真实验结果
	best_total_cost_list = np.array([])
	best_uav_plan_list = []
	best_customer_plan_list = []
	best_time_uav_task_dict_list = []
	best_vehicle_plan_time_list = []
	best_vehicle_task_data_list = []
	best_global_reservation_table_list = []
	# 是否允许无人机拥堵调度
	allow_uav_congestion = False
	for iter in range(iter_num):
		# 初始化仿真实验结果
		# best_total_cost_list = np.append(best_total_cost_list, float('inf'))
		# best_uav_plan_list.append([])
		# best_customer_plan_list.append([])
		# best_time_uav_task_dict_list.append([])
		# best_vehicle_plan_time_list.append([])
		# best_vehicle_task_data_list.append([])
		# best_global_reservation_table_list.append([])

		# 获得高质量初始卡车路径分配方案
		init_total_cost, init_uav_plan, init_customer_plan, init_time_uav_task_dict, init_uav_cost, init_vehicle_route, init_vehicle_plan_time, init_vehicle_task_data, init_global_reservation_table=initial_route(node, DEPOT_nodeID,
		 V, T, vehicle, uav_travel, veh_distance, veh_travel, 
		N, N_zero, N_plus, A_total, A_cvtp, A_vtp, 
		A_aerial_relay_node, G_air, G_ground,air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee)
		# # 处理空跑节点
		# rm_empty_vehicle_route, empty_nodes_by_vehicle = rm_empty_node(init_customer_plan, init_vehicle_route)
		# rm_empty_node_cost = calculate_plan_cost(init_uav_plan, rm_empty_vehicle_route, vehicle, T, V, veh_distance)
		# 搭建高效ALNS算法求解框架
		from fast_alns_solver import create_fast_initial_state, solve_with_fast_alns
		
		# 创建初始解状态
		initial_state = create_fast_initial_state(
			init_total_cost, init_uav_plan, init_customer_plan, init_uav_cost,
			init_time_uav_task_dict, init_vehicle_route, 
			init_vehicle_plan_time, init_vehicle_task_data, 
			init_global_reservation_table, node, DEPOT_nodeID, 
			V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, 
			air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee
		)
		
		# 使用高效ALNS求解（增量式算法，避免深拷贝）
		best_solution, best_objective, statistics = solve_with_fast_alns(
		initial_state, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, 
		A_aerial_relay_node, G_air, G_ground,air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee, 
		max_iterations=50, max_runtime=30, use_incremental=True
		)
		
		print(f"ALNS求解结果 - 总成本: {best_objective}")
		
		# # 提取ALNS优化后的结果
		# alns_vehicle_routes = best_solution.vehicle_routes
		# alns_uav_assignments = best_solution.uav_assignments
		# alns_customer_plan = best_solution.customer_plan
		# alns_vehicle_task_data = best_solution.vehicle_task_data
		# alns_global_reservation_table = best_solution.global_reservation_table
		

		

		
		
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
import json
import numpy as np
import pandas as pd
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
SAVE_DIR_TOTAL = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions\data_total"

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


# def solve_mfstsp_heuristic(node, vehicle, air_matrix, ground_matrix, air_node_types, ground_node_types, numPoints, numUAVs, numTrucks, uav_travel, veh_travel, veh_distance, G_air, G_ground, customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName, max_iterations, loop_iterations):
def solve_mfstsp_heuristic(
    node, vehicle, air_matrix, ground_matrix, air_node_types, ground_node_types, numPoints,
    numUAVs, numTrucks, uav_travel, veh_travel, veh_distance, G_air, G_ground,
    customer_time_windows_h, early_arrival_cost, late_arrival_cost,
    problemName, max_iterations, loop_iterations,
    run_tag=None, algorithm=None, seed=None, algo_seed=None, instance_name=None):
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

	if seed is not None:
		try:
			import random
			import numpy as np
			random.seed(int(seed))
			np.random.seed(int(seed))
		except Exception:
			pass


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
	# 车辆-无人机数目
	numTrucks = len(T)
	numUAVs = len(V)
	numCustomers = len(C)

	solveAlns = algorithm  # 此处改为算子情况
	# 判断当前阶段任务是否有文件夹了，没有就建立D:\NKU\VDAC_PAP\VDAC\saved_solutions
	ROOT_BASE_DIR = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"
	# ROOT_BASE_DIR = r"D:\NKU\VDAC_PAP\VDAC\saved_solutions"
	# SUMMARY_PARENT = f"{solveAlns}求解{numCustomers}客户节点结果汇总"
	# SUMMARY_FOLDER = f"{numTrucks}车{numUAVs}机配送{numCustomers}节点任务汇总"
	# ROOT_BASE_DIR = r"D:\NKU\VDAC_PAP\VDAC\saved_solutions"  # 按你的新路径要求

	# op_suffix = f"__D-{destroy_op}__R-{repair_op}" if destroy_op and repair_op else ""
	SUMMARY_PARENT = f"{solveAlns}_C{numCustomers}result"
	SUMMARY_FOLDER = f"{solveAlns}_T{numUAVs}_U{numCustomers}"

	# 1) 确保 ROOT_BASE_DIR 存在
	os.makedirs(ROOT_BASE_DIR, exist_ok=True)

	# 2) ROOT_BASE_DIR 下的 SUMMARY_PARENT
	parent_dir = os.path.join(ROOT_BASE_DIR, SUMMARY_PARENT)
	if not os.path.isdir(parent_dir):
		os.makedirs(parent_dir, exist_ok=True)  # 不存在就创建
	# 存在就不管

	# 3) SUMMARY_PARENT 下的 SUMMARY_FOLDER
	summary_dir = os.path.join(parent_dir, SUMMARY_FOLDER)
	if not os.path.isdir(summary_dir):
		os.makedirs(summary_dir, exist_ok=True)  # 不存在就创建


	# 上述内容将基础参数全部处理完成，随后开始仿真实验处理
	# results_all = init_results_framework(["H_ALNS"])  # 后续加别的算法名即可
	results_all = init_results_framework([algorithm])
	iter_num = 1 # 仿真实验次数,便于计算平均值等方案
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
	for iter in range(loop_iterations):
		# 获得高质量初始卡车路径分配方案
		init_total_cost, init_uav_plan, init_customer_plan, init_time_uav_task_dict, init_uav_cost, init_vehicle_route, init_vehicle_plan_time, init_vehicle_task_data, init_global_reservation_table=initial_route(node, DEPOT_nodeID,
		 V, T, vehicle, uav_travel, veh_distance, veh_travel, 
		N, N_zero, N_plus, A_total, A_cvtp, A_vtp, 
		A_aerial_relay_node, G_air, G_ground,air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee, customer_time_windows_h, early_arrival_cost, late_arrival_cost, numPoints, numTrucks, numUAVs)
		# # 处理空跑节点
		# rm_empty_vehicle_route, empty_nodes_by_vehicle = rm_empty_node(init_customer_plan, init_vehicle_route)
		# rm_empty_node_cost = calculate_plan_cost(init_uav_plan, rm_empty_vehicle_route, vehicle, T, V, veh_distance)
		# 搭建高效ALNS算法求解框架
		from fast_alns_solver import create_fast_initial_state, solve_with_fast_alns, solve_with_T_alns, solve_with_T_I_alns
		
		# 创建初始解状态
		initial_state = create_fast_initial_state(
			init_total_cost, init_uav_plan, init_customer_plan, init_uav_cost,
			init_time_uav_task_dict, init_vehicle_route, 
			init_vehicle_plan_time, init_vehicle_task_data, 
			init_global_reservation_table, node, DEPOT_nodeID, 
			V, T, vehicle, uav_travel, veh_distance, veh_travel, N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground, 
			air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee, 
			customer_time_windows_h, early_arrival_cost, late_arrival_cost
		)
		H_alns_initial_state = initial_state.fast_copy()
		T_alns_initial_state = initial_state.fast_copy()
		# 使用高效ALNS求解（增量式算法，避免深拷贝）,输出最佳方案结果，并保存到文件中
		if algorithm == "H_ALNS":
			(H_alns_best_state, H_alns_best_final_state, H_alns_best_objective, H_alns_best_final_objective, H_alns_best_final_uav_cost, 
			H_alns_best_final_win_cost, H_alns_best_total_win_cost, H_alns_best_final_global_max_time, H_alns_best_global_max_time, H_alns_best_window_total_cost, 
			H_alns_best_total_uav_tw_violation_cost, H_alns_best_total_vehicle_cost, H_alns_elapsed_time, H_alns_win_cost, H_alns_uav_route_cost, H_alns_vehicle_route_cost, 
			H_alns_final_uav_cost, H_alns_final_total_list, H_alns_final_win_cost, H_alns_final_total_objective, H_alns_y_cost, H_alns_y_best, H_alns_work_time, 
			H_alns_final_work_time) = solve_with_fast_alns(
				H_alns_initial_state, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel,
				N, N_zero, N_plus, A_total, A_cvtp, A_vtp,
				A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix,
				air_node_types, ground_node_types, A_c, xeee,
				customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName,
				iter=iter, max_iterations=max_iterations, max_runtime=30, summary_dir=summary_dir, use_incremental=True, seed=seed, algo_seed=algo_seed
			)
			record_one_run(H_alns_best_state, H_alns_best_final_state, H_alns_best_objective, H_alns_best_final_objective, H_alns_best_final_uav_cost, 
			results_all[algorithm],
			# --- state（只存不导出）---
			best_state_list=H_alns_best_state,
			best_final_state_list=H_alns_best_final_state,

			# --- 标量 ---
			best_objective_list=H_alns_best_objective,
			best_final_objective_list=H_alns_best_final_objective,
			best_final_uav_cost_list=H_alns_best_final_uav_cost,
			best_final_win_cost_list=H_alns_best_final_win_cost,
			best_total_win_cost_list=H_alns_best_total_win_cost,
			best_final_global_max_time_list=H_alns_best_final_global_max_time,
			best_global_max_time_list=H_alns_best_global_max_time,
			best_window_total_cost_list=H_alns_best_window_total_cost,
			best_total_uav_tw_violation_cost_list=H_alns_best_total_uav_tw_violation_cost,
			best_total_vehicle_cost_list=H_alns_best_total_vehicle_cost,
			elapsed_time_list=H_alns_elapsed_time,

			# --- 每代曲线（list/np.array 都可以）---
			win_cost_curve_list=H_alns_win_cost,
			uav_route_cost_curve_list=H_alns_uav_route_cost,
			vehicle_route_cost_curve_list=H_alns_vehicle_route_cost,
			final_uav_cost_curve_list=H_alns_final_uav_cost,
			final_total_list_curve_list=H_alns_final_total_list,
			final_win_cost_curve_list=H_alns_final_win_cost,
			final_total_objective_curve_list=H_alns_final_total_objective,
			y_cost_curve_list=H_alns_y_cost,
			y_best_curve_list=H_alns_y_best,
			work_time_curve_list=H_alns_work_time,
			final_work_time_curve_list=H_alns_final_work_time,
			)
			print(f"H-ALNS_finished_{iter}_iterations")
		elif algorithm == "T_ALNS":
			T_alns_initial_state = initial_state.fast_copy()
			# 使用传统ALNS求解,输出最佳方案结果，并保存到文件中
			(T_alns_best_state, T_alns_best_final_state, T_alns_best_objective, T_alns_best_final_objective, T_alns_best_final_uav_cost, 
			T_alns_best_final_win_cost, T_alns_best_total_win_cost, T_alns_best_final_global_max_time, T_alns_best_global_max_time, T_alns_best_window_total_cost, 
			T_alns_best_total_uav_tw_violation_cost, T_alns_best_total_vehicle_cost, T_alns_elapsed_time, T_alns_win_cost, T_alns_uav_route_cost, T_alns_vehicle_route_cost, 
			T_alns_final_uav_cost, T_alns_final_total_list, T_alns_final_win_cost, T_alns_final_total_objective, T_alns_y_cost, T_alns_y_best, T_alns_work_time, 
			T_alns_final_work_time) = solve_with_T_alns(
				T_alns_initial_state, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel,
				N, N_zero, N_plus, A_total, A_cvtp, A_vtp,
				A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix,
				air_node_types, ground_node_types, A_c, xeee,
				customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName,
				iter=iter, max_iterations=max_iterations, max_runtime=30, use_incremental=True, seed=seed, algo_seed=algo_seed
			)
			record_one_run(T_alns_best_state, T_alns_best_final_state, T_alns_best_objective, T_alns_best_final_objective, T_alns_best_final_uav_cost, 
			results_all[algorithm],
			# --- state（只存不导出）---
			best_state_list=T_alns_best_state,
			best_final_state_list=T_alns_best_final_state,
			# --- 标量 ---
			best_objective_list=T_alns_best_objective,
			best_final_objective_list=T_alns_best_final_objective,
			best_final_uav_cost_list=T_alns_best_final_uav_cost,
			best_final_win_cost_list=T_alns_best_final_win_cost,
			best_total_win_cost_list=T_alns_best_total_win_cost,
			best_final_global_max_time_list=T_alns_best_final_global_max_time,
			best_global_max_time_list=T_alns_best_global_max_time,
			best_window_total_cost_list=T_alns_best_window_total_cost,
			best_total_uav_tw_violation_cost_list=T_alns_best_total_uav_tw_violation_cost,
			best_total_vehicle_cost_list=T_alns_best_total_vehicle_cost,
			elapsed_time_list=T_alns_elapsed_time,
			# --- 每代曲线（list/np.array 都可以）---
			win_cost_curve_list=T_alns_win_cost,
			uav_route_cost_curve_list=T_alns_uav_route_cost,
			vehicle_route_cost_curve_list=T_alns_vehicle_route_cost,
			final_uav_cost_curve_list=T_alns_final_uav_cost,
			final_total_list_curve_list=T_alns_final_total_list,
			final_win_cost_curve_list=T_alns_final_win_cost,
			final_total_objective_curve_list=T_alns_final_total_objective,
			y_cost_curve_list=T_alns_y_cost,
			y_best_curve_list=T_alns_y_best,
			work_time_curve_list=T_alns_work_time,
			final_work_time_curve_list=T_alns_final_work_time,
			)
			print(f"T-I-ALNS_finished_{iter}_iterations")
		elif algorithm == "T_I_ALNS":
			from fast_alns_solver import solve_with_T_I_alns
			T_I_alns_initial_state = initial_state.fast_copy()
			# 使用传统增量式ALNS求解,输出最佳方案结果，并保存到文件中
			(T_I_alns_best_state, T_I_alns_best_final_state, T_I_alns_best_objective, T_I_alns_best_final_objective, T_I_alns_best_final_uav_cost, 
			T_I_alns_best_final_win_cost, T_I_alns_best_total_win_cost, T_I_alns_best_final_global_max_time, T_I_alns_best_global_max_time, T_I_alns_best_window_total_cost, 
			T_I_alns_best_total_uav_tw_violation_cost, T_I_alns_best_total_vehicle_cost, T_I_alns_elapsed_time, T_I_alns_win_cost, T_I_alns_uav_route_cost, T_I_alns_vehicle_route_cost, 
			T_I_alns_final_uav_cost, T_I_alns_final_total_list, T_I_alns_final_win_cost, T_I_alns_final_total_objective, T_I_alns_y_cost, T_I_alns_y_best, T_I_alns_work_time, 
			T_I_alns_final_work_time) = solve_with_T_I_alns(
				T_I_alns_initial_state, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel,
				N, N_zero, N_plus, A_total, A_cvtp, A_vtp,
				A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix,
				air_node_types, ground_node_types, A_c, xeee,
				customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName,
				iter=iter, max_iterations=max_iterations, max_runtime=30, use_incremental=True, seed=seed, algo_seed=algo_seed
			)
			record_one_run(T_I_alns_best_state, T_I_alns_best_final_state, T_I_alns_best_objective, T_I_alns_best_final_objective, T_I_alns_best_final_uav_cost, 
			results_all[algorithm],
			# --- state（只存不导出）---
			best_state_list=T_I_alns_best_state,
			best_final_state_list=T_I_alns_best_final_state,
			# --- 标量 ---
			best_objective_list=T_I_alns_best_objective,
			best_final_objective_list=T_I_alns_best_final_objective,
			best_final_uav_cost_list=T_I_alns_best_final_uav_cost,
			best_final_win_cost_list=T_I_alns_best_final_win_cost,
			best_total_win_cost_list=T_I_alns_best_total_win_cost,
			best_final_global_max_time_list=T_I_alns_best_final_global_max_time,
			best_global_max_time_list=T_I_alns_best_global_max_time,
			best_window_total_cost_list=T_I_alns_best_window_total_cost,
			best_total_uav_tw_violation_cost_list=T_I_alns_best_total_uav_tw_violation_cost,
			best_total_vehicle_cost_list=T_I_alns_best_total_vehicle_cost,
			elapsed_time_list=T_I_alns_elapsed_time,
			# --- 每代曲线（list/np.array 都可以）---
			win_cost_curve_list=T_I_alns_win_cost,
			uav_route_cost_curve_list=T_I_alns_uav_route_cost,
			vehicle_route_cost_curve_list=T_I_alns_vehicle_route_cost,
			final_uav_cost_curve_list=T_I_alns_final_uav_cost,
			final_total_list_curve_list=T_I_alns_final_total_list,
			final_win_cost_curve_list=T_I_alns_final_win_cost,
			final_total_objective_curve_list=T_I_alns_final_total_objective,
			y_cost_curve_list=T_I_alns_y_cost,
			y_best_curve_list=T_I_alns_y_best,
			work_time_curve_list=T_I_alns_work_time,
			final_work_time_curve_list=T_I_alns_final_work_time,
			)
			print(f"T-I-ALNS_finished_{iter}_iterations")
		elif algorithm == "TA_I_ALNS":  # 参考对应论文A multi-visit flexible-docking vehicle routing problem with drones for simultaneous pickup and delivery services
			from fast_alns_solver import solve_with_TA_I_alns
			TA_I_alns_initial_state = initial_state.fast_copy()
			# 使用传统增量式ALNS求解,输出最佳方案结果，并保存到文件中
			(TA_I_alns_best_state, TA_I_alns_best_final_state, TA_I_alns_best_objective, TA_I_alns_best_final_objective, TA_I_alns_best_final_uav_cost, 
			TA_I_alns_best_final_win_cost, TA_I_alns_best_total_win_cost, TA_I_alns_best_final_global_max_time, TA_I_alns_best_global_max_time, TA_I_alns_best_window_total_cost, 
			TA_I_alns_best_total_uav_tw_violation_cost, TA_I_alns_best_total_vehicle_cost, TA_I_alns_elapsed_time, TA_I_alns_win_cost, TA_I_alns_uav_route_cost, TA_I_alns_vehicle_route_cost, 
			TA_I_alns_final_uav_cost, TA_I_alns_final_total_list, TA_I_alns_final_win_cost, TA_I_alns_final_total_objective, TA_I_alns_y_cost, TA_I_alns_y_best, TA_I_alns_work_time, 
			TA_I_alns_final_work_time) = solve_with_TA_I_alns(
				TA_I_alns_initial_state, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel,
				N, N_zero, N_plus, A_total, A_cvtp, A_vtp,
				A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix,
				air_node_types, ground_node_types, A_c, xeee,
				customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName,
				iter=iter, max_iterations=max_iterations, max_runtime=30, use_incremental=True, seed=seed, algo_seed=algo_seed
			)
			record_one_run(TA_I_alns_best_state, TA_I_alns_best_final_state, TA_I_alns_best_objective, TA_I_alns_best_final_objective, TA_I_alns_best_final_uav_cost, 
			results_all[algorithm],
			# --- state（只存不导出）---
			best_state_list=TA_I_alns_best_state,
			best_final_state_list=TA_I_alns_best_final_state,
			# --- 标量 ---
			best_objective_list=TA_I_alns_best_objective,
			best_final_objective_list=TA_I_alns_best_final_objective,
			best_final_uav_cost_list=TA_I_alns_best_final_uav_cost,
			best_final_win_cost_list=TA_I_alns_best_final_win_cost,
			best_total_win_cost_list=TA_I_alns_best_total_win_cost,
			best_final_global_max_time_list=TA_I_alns_best_final_global_max_time,
			best_global_max_time_list=TA_I_alns_best_global_max_time,
			best_window_total_cost_list=TA_I_alns_best_window_total_cost,
			best_total_uav_tw_violation_cost_list=TA_I_alns_best_total_uav_tw_violation_cost,
			best_total_vehicle_cost_list=TA_I_alns_best_total_vehicle_cost,
			elapsed_time_list=TA_I_alns_elapsed_time,
			# --- 每代曲线（list/np.array 都可以）---
			win_cost_curve_list=TA_I_alns_win_cost,
			uav_route_cost_curve_list=TA_I_alns_uav_route_cost,
			vehicle_route_cost_curve_list=TA_I_alns_vehicle_route_cost,
			final_uav_cost_curve_list=TA_I_alns_final_uav_cost,
			final_total_list_curve_list=TA_I_alns_final_total_list,
			final_win_cost_curve_list=TA_I_alns_final_win_cost,
			final_total_objective_curve_list=TA_I_alns_final_total_objective,
			y_cost_curve_list=TA_I_alns_y_cost,
			y_best_curve_list=TA_I_alns_y_best,
			work_time_curve_list=TA_I_alns_work_time,
			final_work_time_curve_list=TA_I_alns_final_work_time,
			)
			print(f"TA_I_ALNS_finished_{iter}_iterations")
		elif algorithm == "A_I_ALNS":  # 参考对应论文A multi-visit flexible-docking vehicle routing problem with drones for simultaneous pickup and delivery services
			from fast_alns_solver import solve_with_A_I_alns
			A_I_alns_initial_state = initial_state.fast_copy()
			# 使用传统增量式ALNS求解,输出最佳方案结果，并保存到文件中
			(A_I_alns_best_state, A_I_alns_best_final_state, A_I_alns_best_objective, A_I_alns_best_final_objective, A_I_alns_best_final_uav_cost, 
			A_I_alns_best_final_win_cost, A_I_alns_best_total_win_cost, A_I_alns_best_final_global_max_time, A_I_alns_best_global_max_time, A_I_alns_best_window_total_cost, 
			A_I_alns_best_total_uav_tw_violation_cost, A_I_alns_best_total_vehicle_cost, A_I_alns_elapsed_time, A_I_alns_win_cost, A_I_alns_uav_route_cost, A_I_alns_vehicle_route_cost, 
			A_I_alns_final_uav_cost, A_I_alns_final_total_list, A_I_alns_final_win_cost, A_I_alns_final_total_objective, A_I_alns_y_cost, A_I_alns_y_best, A_I_alns_work_time, 
			A_I_alns_final_work_time) = solve_with_A_I_alns(
				A_I_alns_initial_state, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel,
				N, N_zero, N_plus, A_total, A_cvtp, A_vtp,
				A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix,
				air_node_types, ground_node_types, A_c, xeee,
				customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName,
				iter=iter, max_iterations=max_iterations, max_runtime=30, use_incremental=True, seed=seed, algo_seed=algo_seed
			)
			record_one_run(A_I_alns_best_state, A_I_alns_best_final_state, A_I_alns_best_objective, A_I_alns_best_final_objective, A_I_alns_best_final_uav_cost, 
			results_all[algorithm],
			# --- state（只存不导出）---
			best_state_list=A_I_alns_best_state,
			best_final_state_list=A_I_alns_best_final_state,
			# --- 标量 ---
			best_objective_list=A_I_alns_best_objective,
			best_final_objective_list=A_I_alns_best_final_objective,
			best_final_uav_cost_list=A_I_alns_best_final_uav_cost,
			best_final_win_cost_list=A_I_alns_best_final_win_cost,
			best_total_win_cost_list=A_I_alns_best_total_win_cost,
			best_final_global_max_time_list=A_I_alns_best_final_global_max_time,
			best_global_max_time_list=A_I_alns_best_global_max_time,
			best_window_total_cost_list=A_I_alns_best_window_total_cost,
			best_total_uav_tw_violation_cost_list=A_I_alns_best_total_uav_tw_violation_cost,
			best_total_vehicle_cost_list=A_I_alns_best_total_vehicle_cost,
			elapsed_time_list=A_I_alns_elapsed_time,
			# --- 每代曲线（list/np.array 都可以）---
			win_cost_curve_list=A_I_alns_win_cost,
			uav_route_cost_curve_list=A_I_alns_uav_route_cost,
			vehicle_route_cost_curve_list=A_I_alns_vehicle_route_cost,
			final_uav_cost_curve_list=A_I_alns_final_uav_cost,
			final_total_list_curve_list=A_I_alns_final_total_list,
			final_win_cost_curve_list=A_I_alns_final_win_cost,
			final_total_objective_curve_list=A_I_alns_final_total_objective,
			y_cost_curve_list=A_I_alns_y_cost,
			y_best_curve_list=A_I_alns_y_best,
			work_time_curve_list=A_I_alns_work_time,
			final_work_time_curve_list=A_I_alns_final_work_time,
			)
			print(f"A_I_ALNS_finished_{iter}_iterations")
		elif algorithm == "DA_I_ALNS":  # 参考对应论文The drone-assisted simultaneous pickup and delivery problem with time windows
			from fast_alns_solver import solve_with_DA_I_alns
			DA_I_alns_initial_state = initial_state.fast_copy()
			# 使用传统增量式ALNS求解,输出最佳方案结果，并保存到文件中
			(DA_I_alns_best_state, DA_I_alns_best_final_state, DA_I_alns_best_objective, DA_I_alns_best_final_objective, DA_I_alns_best_final_uav_cost, 
			DA_I_alns_best_final_win_cost, DA_I_alns_best_total_win_cost, DA_I_alns_best_final_global_max_time, DA_I_alns_best_global_max_time, DA_I_alns_best_window_total_cost, 
			DA_I_alns_best_total_uav_tw_violation_cost, DA_I_alns_best_total_vehicle_cost, DA_I_alns_elapsed_time, DA_I_alns_win_cost, DA_I_alns_uav_route_cost, DA_I_alns_vehicle_route_cost, 
			DA_I_alns_final_uav_cost, DA_I_alns_final_total_list, DA_I_alns_final_win_cost, DA_I_alns_final_total_objective, DA_I_alns_y_cost, DA_I_alns_y_best, DA_I_alns_work_time, 
			DA_I_alns_final_work_time) = solve_with_DA_I_alns(
				DA_I_alns_initial_state, node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel,
				N, N_zero, N_plus, A_total, A_cvtp, A_vtp,
				A_aerial_relay_node, G_air, G_ground, air_matrix, ground_matrix,
				air_node_types, ground_node_types, A_c, xeee,
				customer_time_windows_h, early_arrival_cost, late_arrival_cost, problemName,
				iter=iter, max_iterations=max_iterations, max_runtime=30, use_incremental=True, seed=seed, algo_seed=algo_seed
			)
			record_one_run(DA_I_alns_best_state, DA_I_alns_best_final_state, DA_I_alns_best_objective, DA_I_alns_best_final_objective, DA_I_alns_best_final_uav_cost, 
			results_all[algorithm],
			# --- state（只存不导出）---
			best_state_list=DA_I_alns_best_state,
			best_final_state_list=DA_I_alns_best_final_state,
			# --- 标量 ---
			best_objective_list=DA_I_alns_best_objective,
			best_final_objective_list=DA_I_alns_best_final_objective,
			best_final_uav_cost_list=DA_I_alns_best_final_uav_cost,
			best_final_win_cost_list=DA_I_alns_best_final_win_cost,
			best_total_win_cost_list=DA_I_alns_best_total_win_cost,
			best_final_global_max_time_list=DA_I_alns_best_final_global_max_time,
			best_global_max_time_list=DA_I_alns_best_global_max_time,
			best_window_total_cost_list=DA_I_alns_best_window_total_cost,
			best_total_uav_tw_violation_cost_list=DA_I_alns_best_total_uav_tw_violation_cost,
			best_total_vehicle_cost_list=DA_I_alns_best_total_vehicle_cost,
			elapsed_time_list=DA_I_alns_elapsed_time,
			# --- 每代曲线（list/np.array 都可以）---
			win_cost_curve_list=DA_I_alns_win_cost,
			uav_route_cost_curve_list=DA_I_alns_uav_route_cost,
			vehicle_route_cost_curve_list=DA_I_alns_vehicle_route_cost,
			final_uav_cost_curve_list=DA_I_alns_final_uav_cost,
			final_total_list_curve_list=DA_I_alns_final_total_list,
			final_win_cost_curve_list=DA_I_alns_final_win_cost,
			final_total_objective_curve_list=DA_I_alns_final_total_objective,
			y_cost_curve_list=DA_I_alns_y_cost,
			y_best_curve_list=DA_I_alns_y_best,
			work_time_curve_list=DA_I_alns_work_time,
			final_work_time_curve_list=DA_I_alns_final_work_time,
			)
			print(f"DA_I_ALNS_finished_{iter}_iterations")

	base_dir_route = SAVE_DIR_TOTAL  # 基础目录
	# 目标子目录：按客户数 + 算子组合分组
	folder_name = f"{len(C)}customers{algorithm}"
	target_dir = os.path.join(base_dir_route, folder_name)
	# 没有就创建
	os.makedirs(target_dir, exist_ok=True)
	# 文件名带 algo_seed
	export_name = f"{algorithm}_{problemName}_C_{len(C)}_{run_tag}"
	# 保存到新目录
	export_results_to_excel(results_all, export_name, save_dir=target_dir)
	print(f"H-ALNS求解完成，保存到文件中。")
		
# ========= 工具函数 =========
def _to_py(obj):
    """把 numpy 类型 / ndarray / list 统一转成 python 可序列化对象"""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
	
def _jsonify(obj):
    """list/ndarray -> JSON string；标量保持标量"""
    obj = _to_py(obj)
    if isinstance(obj, (list, dict)):
        return json.dumps(obj, ensure_ascii=False)
    return obj

def init_results_framework(algo_names):
    """
    algo_names: 例如 ["H_ALNS", "GA", "MyAlgo"]
    返回：results[algo] 是一个 dict，内部所有字段都是 list，用于按 iter 追加
    """
    results = {}
    for algo in algo_names:
        results[algo] = {
            # --- state：只存，不导出 ---
            "best_state_list": [],
            "best_final_state_list": [],

            # --- 关键标量（每次仿真实验一条） ---
            "best_objective_list": [],
            "best_final_objective_list": [],
            "best_final_uav_cost_list": [],
            "best_final_win_cost_list": [],
            "best_total_win_cost_list": [],
            "best_final_global_max_time_list": [],
            "best_global_max_time_list": [],
            "best_window_total_cost_list": [],
            "best_total_uav_tw_violation_cost_list": [],
            "best_total_vehicle_cost_list": [],
            "elapsed_time_list": [],

            # --- 每代曲线（每次仿真实验一条“列表/数组”） ---
            "win_cost_curve_list": [],
            "uav_route_cost_curve_list": [],
            "vehicle_route_cost_curve_list": [],
            "final_uav_cost_curve_list": [],
            "final_total_list_curve_list": [],
            "final_win_cost_curve_list": [],
            "final_total_objective_curve_list": [],
            "y_cost_curve_list": [],
            "y_best_curve_list": [],
            "work_time_curve_list": [],
            "final_work_time_curve_list": [],
        }
    return results

def record_one_run(results_for_algo, **kwargs):
    """
    往某个算法的 results dict 里追加一次仿真实验结果。
    kwargs 支持你传任何字段；只要在 results_for_algo 里存在就会 append。
    """
    for k, v in kwargs.items():
        if k not in results_for_algo:
            # 你后续加新指标时，不用改框架：自动创建新列表字段
            results_for_algo[k] = []
        results_for_algo[k].append(_to_py(v))

def export_results_to_excel(results_by_algo, problemName, save_dir=SAVE_DIR_TOTAL):
    """
    把所有算法导出到同一个 Excel：
      - 每个算法两个 sheet：{algo}_summary 和 {algo}_curves
      - summary：每次 iter 一行（曲线字段写成 JSON 字符串，便于后续解析）
      - curves：长表(run_id, gen, 指标...) 方便画迭代曲线
    """
    os.makedirs(save_dir, exist_ok=True)
    xlsx_path = os.path.join(save_dir, f"{problemName}_data_total.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for algo, data in results_by_algo.items():
            # ===== summary（忽略 state）=====
            n_runs = len(data.get("best_objective_list", []))
            summary_rows = []
            for i in range(n_runs):
                row = {"run_id": i, "algo": algo}
                for key, lst in data.items():
                    if key in ("best_state_list", "best_final_state_list"):
                        continue  # 忽略 state
                    if not isinstance(lst, list) or i >= len(lst):
                        continue
                    row[key] = _jsonify(lst[i])
                summary_rows.append(row)
            df_summary = pd.DataFrame(summary_rows)

            # ===== curves（长表）=====
            curve_keys = [
                "win_cost_curve_list",
                "uav_route_cost_curve_list",
                "vehicle_route_cost_curve_list",
                "final_uav_cost_curve_list",
                "final_total_list_curve_list",
                "final_win_cost_curve_list",
                "final_total_objective_curve_list",
                "y_cost_curve_list",
                "y_best_curve_list",
                "work_time_curve_list",
                "final_work_time_curve_list",
            ]

            curve_rows = []
            for run_id in range(n_runs):
                # 找这次 run 的最大代数（取所有曲线里最长的那个）
                max_len = 0
                run_curves = {}
                for ck in curve_keys:
                    curves_all_runs = data.get(ck, [])
                    curve = curves_all_runs[run_id] if run_id < len(curves_all_runs) else None
                    curve = _to_py(curve)
                    if curve is None:
                        run_curves[ck] = None
                        continue
                    if not isinstance(curve, list):
                        curve = [curve]  # 兜底：单值也变成 list
                    run_curves[ck] = curve
                    max_len = max(max_len, len(curve))

                for gen in range(max_len):
                    row = {"run_id": run_id, "gen": gen, "algo": algo}
                    for ck, curve in run_curves.items():
                        # 列名更好看：去掉 _curve_list
                        col = ck.replace("_curve_list", "")
                        if curve is None or gen >= len(curve):
                            row[col] = np.nan
                        else:
                            row[col] = _to_py(curve[gen])
                    curve_rows.append(row)

            df_curves = pd.DataFrame(curve_rows)

            # 写入 Excel
            sheet_summary = f"{algo}_summary"
            sheet_curves = f"{algo}_curves"
            df_summary.to_excel(writer, sheet_name=sheet_summary[:31], index=False)
            df_curves.to_excel(writer, sheet_name=sheet_curves[:31], index=False)

    print(f"[OK] 导出完成：{xlsx_path}")
    return xlsx_path
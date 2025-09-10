#!/usr/bin/env python

import sys
import time
import datetime
import math
from parseCSV import *
from gurobipy import *
from heapq import *
from collections import defaultdict
import operator
import itertools
import endurance_calculator
import copy

# =============================================================
NODE_TYPE_DEPOT		= 0
NODE_TYPE_CUST		= 1

TYPE_TRUCK 			= 1
TYPE_UAV 			= 2

TRAVEL_UAV_PACKAGE		= 1
TRAVEL_UAV_EMPTY		= 2
TRAVEL_TRUCK_W_UAV		= 3
TRAVEL_TRUCK_EMPTY		= 4

VERTICAL_UAV_EMPTY		= 5
VERTICAL_UAV_PACKAGE	= 6

STATIONARY_UAV_EMPTY	= 7
STATIONARY_UAV_PACKAGE	= 8
STATIONARY_TRUCK_W_UAV	= 9
STATIONARY_TRUCK_EMPTY	= 10

GANTT_IDLE		= 1
GANTT_TRAVEL	= 2
GANTT_DELIVER	= 3
GANTT_RECOVER	= 4
GANTT_LAUNCH	= 5
GANTT_FINISHED	= 6

# There's a package color that corresponds to the VEHICLE that delivered the package.
# Right now we only have 5 boxes (so we can have at most 5 trucks).
packageIcons		= ['box_yellow_centered.gltf', 'box_blue_centered.gltf', 'box_orange_centered.gltf', 'box_green_centered.gltf', 'box_gray_centered.gltf', 'box_brown_centered.gltf']
# =============================================================


# http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python
def make_dict():
	return defaultdict(make_dict)

	# Usage:
	# tau = defaultdict(make_dict)
	# v = 17
	# i = 3
	# j = 12
	# tau[v][i][j] = 44


def mfstsp_heuristic_3_timing(x, y, z, node, inieee, N, P, V, c, sigma, sigmaprime, tau, iniTauprimeE, iniTauprimeF, minDistance, sR, sL, vehicle, travel, optLowBnd, iniSpeedE, iniSpeedF, P3Type, UAVSpeedType):

	# Helper parameters:
	B = defaultdict(make_dict)  # 无人机v在k点回收时服务的客户j
	A = defaultdict(make_dict)  # 无人机v在i点发射时服务的客户j
	AB = defaultdict(make_dict)  # 无人机v在k点回收时的发射点i
	ABC = defaultdict(make_dict)  # 无人机v在i点发射时的回收点k
	for [v,i,j,k] in y:
		B[v][k] = j
		A[v][i] = j
		AB[v][k] = i
		ABC[v][i] = k		
	
	# Reset C -- Should only include customers NOT visited by the truck   # 重置C为仅包含无人机服务的客户
	C = []
	for [v,i,j,k] in y:	
		if (j not in C):
			C.append(j)

	# Capture all UAVs that land at a particular node # 初始化发射/回收节点跟踪
	# Capture all UAVs that launch from a particular node
	launchesfrom = {}  # 字典: 节点i -> 从i发射的无人机列表 [v1, v2, ...]
	landsat = {}  # 节点i -> 在i回收的无人机列表 [v3, v4, ...]
	for i in N:
		launchesfrom[i] = []
		landsat[i] = []
	for [v,i,j,k] in y:
		launchesfrom[i].append(v)  # 节点i发射的无人机列表 # 将无人机v添加到发射节点i的列表
		landsat[k].append(v)  # 节点k回收的无人机列表 将无人机v添加到回收节点k的列表
	

	UAVtau = defaultdict(make_dict) # Travel time for v from i->j->k  # 字典: v -> i -> j -> k -> 无人机v从i飞到j再到k的总时间
	for [v,i,j,k] in y:  # 计算了无人机v从i到j再到k的飞行时间，包括是否携带包裹等情况考虑
		UAVtau[v][i][j][k] = iniTauprimeF[v][i][j] + sigmaprime[j] + iniTauprimeE[v][j][k]  # 计算总任务时间 = 去程飞行时间 + 客户点服务时间 + 返程飞行时间

	availVehicles = {} # List of available UAVs at each truck customer node # 初始化可用无人机列表 字典: 卡车路径上的节点i -> 在该节点可用的无人机列表 (包括代表卡车的'1')
	TSPtour = []   # 卡车路径节点序列
	for [i,k] in x:   # x 假设是卡车路径解，形式为 [(起点, 终点), ...] 的弧段列表
		TSPtour.append(i)  # 添加卡车路径节点
		availVehicles[i] = list(V)  # 初始化可用无人机列表 V 可能是所有无人机的集合 {uav1, uav2, ...}。初始假设所有无人机在节点i都可用
		availVehicles[i].append(1)  # 添加无人机1 V 可能是所有无人机的集合 {uav1, uav2, ...}。初始假设所有无人机在节点i都可用

	availVehicles[c+1] = list(V)  # 在终点，所有无人机也假定可用
	availVehicles[c+1].append(1)  # 在终点，所有无人机也假定可用

	TSPtour.append(c+1)  # 在终点，所有无人机也假定可用
	#  # 更新可用无人机状态
	for [v,i,j,k] in y:
		tempp = TSPtour.index(i) # 发射点索引# 找到发射点i在卡车路径中的索引
		tempq = TSPtour.index(k)  # 回收点索引# 找到回收点k在卡车路径中的索引
		 # 在发射到回收的路径段移除该无人机
		for tempindex in range(tempp+1, tempq+1):
			availVehicles[TSPtour[tempindex]].remove(v)  # 更新可用无人机状态 将无人机v从这些节点的可用列表中移除，因为它正在执行任务

	# Decision Variables of Table 3:
	decvarchecktprime = defaultdict(make_dict)  # 字典: v -> k -> 无人机v到达回收点k的时间
	decvarhattprime = defaultdict(make_dict) # 字典: v -> i -> 无人机v离开发射点i的时间 (或客户点?) 根据后续使用判断是发射点
	decvarbart = defaultdict(make_dict)   # 卡车在节点i完成服务/发射活动的时间 (?) 可能是卡车服务完成时间
	decvarhatt = defaultdict(make_dict)   # 卡车离开节点i的时间
	decvarcheckt = defaultdict(make_dict) # 卡车离开节点i的时间

	ArrTime = defaultdict(make_dict) # 字典: k -> v -> 无人机v计算出的到达回收点k的时间
	EndTime = defaultdict(make_dict) # 字典: k -> v -> 无人机v必须在回收点k被回收的最晚时间 (由续航决定)

	serviceOrder = {} # 每个节点服务顺序  # 字典: 节点i -> 在该节点执行的活动顺序列表 [(v1, 'l'), (1, 's'), (v2, 'r'), ...] (l:launch, r:retrieve, s:service)
	for i in TSPtour:
		serviceOrder[i] = []

	#-------------------------------------ALGORITHM 9 (Determine the order of activities at each truck location) STARTS HERE--------------------------------------#
	curTime = 0  # 当前时间
	Status = 'Feasible'  # 状态

	for i in TSPtour:  # 遍历卡车路径的每个节点
		# 记录卡车到达当前节点i的时间
		decvarcheckt[i] = curTime  

		# Sort retrievals in the order of their arrival  处理无人机回收逻辑（仅非仓库节点且有回收需求时）
		if (i == 0) or (len(landsat[i]) == 0):  # 如果节点是仓库或没有回收需求 如果是起始点(假设为0)或没有无人机在此回收
			Retrievals = []  # 无需安排回收
		else:# 阶段1：基础可行性检查 # 有无人机需要在此节点i回收
			isFeasible = True  # 先假设按到达时间回收是可行的

			# Check for Phase 3 feasibility:
			Retrievals = sorted(ArrTime[i], key=ArrTime[i].get)# 按无人机到达时间升序排序  将计划在i点降落的无人机，按它们预计到达时间(ArrTime)升序排序
			checkTime = curTime  # 初始化时间检查点（卡车到达当前节点的时间） 从卡车到达i的时间开始检查

			# Check if this best retrieval strategy is feasible:  模拟按此默认顺序进行回收
			for v in Retrievals:    #  对每个要回收的无人机v
				if checkTime < EndTime[i][v]:   # 检查时间窗口约束：卡车操作时间是否在无人机可用时间窗内  检查时间窗口约束：卡车可用的时间(checkTime)必须早于无人机的续航结束时间(EndTime[i][v])
					checkTime = max(checkTime, ArrTime[i][v])  # 更新时间检查点（卡车到达时间）  # 实际回收开始时间 = max(卡车空闲时间, 无人机到达时间)
					checkTime += sR[v][i]  # 更新时间检查点（卡车操作时间） # 加上回收v所需的操作时间 sR[v][i]
				else: # 卡车到达太晚或之前的操作耗时太长，导致来不及回收v
					isFeasible = False  # 标记为不可行
					break # 停止检查这个顺序

			# If not, check all possible permutations of retrievals to find the best one:  # 阶段2：全排列穷举搜索
			if isFeasible == False:   # 初始化最佳完成时间为无穷大   # 如果按到达时间排序不可行

				bestTime = float('inf')  # 初始化最佳完成时间为无穷大  # 初始化最佳完成时间为无穷大
				Retrievals = []  # 重置可行回收顺序  # 重置可行回收顺序列表
				# 生成所有可能的无人机回收顺序排列
				Rset = list(itertools.permutations(list(landsat[i])))   # 全排列组合  生成所有可能的无人机回收顺序的排列组合

				for Rcand in Rset: # 遍历所有可能的回收顺序
					candFeasible = True # 当前排列可行性标记 # 假设当前候选顺序可行
					checkTime = curTime  # 重置时间检查点  # 重置检查时间

					for v in Rcand: # 验证当前排列可行性 模拟按当前候选顺序 Rcand 进行回收
						if checkTime < EndTime[i][v]:  # 同样的检查：卡车可用时间 < 无人机续航结束时间
							checkTime = max(checkTime, ArrTime[i][v])  # 更新时间检查点（卡车到达时间）  # 实际回收开始时间 = max(卡车空闲时间, 无人机到达时间)
							checkTime += sR[v][i]  # 更新时间检查点（卡车操作时间） # 加上回收v所需的操作时间 sR[v][i]
						else:
							candFeasible = False
							break

					if candFeasible == True:  # 如果当前候选顺序可行
						if checkTime < bestTime:  # 如果当前候选顺序的完成时间 < 最佳完成时间
							bestTime = checkTime  # 更新最佳完成时间
							Retrievals = list(Rcand)  # 更新可行回收顺序

				if len(Retrievals) == 0:  # 检查它是否比当前找到的最佳顺序完成得更早
					# Phase 3 is infeasible for this heuristic
					Status = 'Infeasible'  # 对于这个启发式算法，该节点的调度不可行 -> 整个方案不可行
					break   # 跳出外层 for i in TSPtour 循环

		# 按无人机旅行时间、无人机续航时间或剩余续航时间的降序对服务和发射进行排序:
		Launches = [1]
		# 准备将在节点i进行的发射和卡车服务任务列表  # '1' 代表卡车服务任务 (如果 sigma[i] > 0)
		zzz = {} # 临时字典，用于存储排序依据的值
		# 三种不同的发射排序策略
		if P3Type == 1:
			for v in launchesfrom[i]:  # 按实际任务时间排序 策略1: 按无人机实际任务时长(UAVtau)降序排列 (耗时长的优先?)
				zzz[v] = UAVtau[v][i][A[v][i]][ABC[v][i]]  # UAVtau存储实际飞行时间

			Lzzz = sorted(zzz, key=zzz.get, reverse=True)
			for v in Lzzz:
				Launches.append(v) # 将排序后的无人机v加入Launches列表

		elif P3Type == 2:  # 策略2: 按无人机任务所需总续航时间(inieee)降序排列
			for v in launchesfrom[i]: # 获取v从i出发完成任务所需的总续航
				zzz[v] = inieee[v][i][A[v][i]][ABC[v][i]]   # 假设inieee存储续航值

			Lzzz = sorted(zzz, key=zzz.get, reverse=True)
			for v in Lzzz:
				Launches.append(v)

		else:  # 策略3 (默认): 按任务完成后剩余续航时间升序排列 (剩余少的优先，即相对耗时长的优先)
			for v in launchesfrom[i]:  # 计算续航余量 = 总续航 - 实际任务时间
				zzz[v] = inieee[v][i][A[v][i]][ABC[v][i]] - UAVtau[v][i][A[v][i]][ABC[v][i]]  

			Lzzz = sorted(zzz, key=zzz.get, reverse=True)  # 按续航余量降序排列	# 按续航余量降序排序 (原文reverse=True是余量大的优先，若要余量小的优先，应去掉reverse=True，需确认逻辑)
			for v in Lzzz:
				Launches.append(v)					

		# 按照先前确定的 `Retrievals` 顺序处理回收任务，并尝试在等待时插入 `Launches` 中的任务
		# Schedule retrievals:
		while len(Retrievals) != 0:  # 处理回收任务队列，直到队列为空

			v = Retrievals[0]  # 获取下一个要回收的无人机v

			if curTime < ArrTime[i][v]: # If truck has to wait before UAV v can arrive at i  情况1: 卡车到达时间 `curTime` 早于 无人机v的到达时间 `ArrTime[i][v]`
				 # 检查当前节点是否有可用无人机可调度
				if len(set(availVehicles[i]).intersection(set(Launches))) != 0: # 找出当前可用且计划要发射/服务的任务
					# 遍历发射队列寻找可调度无人机
					for v2 in Launches:  	# 检查在等待期间是否有可用的车辆（无人机或卡车'1'）可以执行 `Launches` 列表中的任务 # 按优先级遍历计划的发射/服务任务 v2

						if v2 in availVehicles[i]: # Try to schedule the first launch/service that's available  # 如果执行该任务的车辆v2当前可用

							if v2 == 1:  # 记录卡车服务或无人机发射    # 如果是卡车服务
								ServiceLaunch = sigma[i]    # 卡车服务结束时间  # 如果是卡车服务
							else:
								ServiceLaunch = sL[v2][i]    # # 无人机v2的发射准备时间
							# 关键检查: 执行任务v2会不会导致 `curTime` 超过等待回收的无人机v的续航结束时间 `EndTime[i][v]`?
							if curTime + ServiceLaunch < EndTime[i][v]: # 检查无人机发射是否在回收时间窗内  # 可行！可以插入任务v2
								curTime += ServiceLaunch  # 更新当前时间

								if v2 == 1:  # 记录任务v2的完成信息
									decvarbart[i] = curTime  # 记录任务v2的完成信息  
									serviceOrder[i].append((v2, 's'))  # 加入服务顺序列表
								else:  # 如果是无人机发射
									decvarhattprime[v2][i] = curTime   # 记录无人机v2的发射完成时间
									serviceOrder[i].append((v2, 'l'))   # 加入服务顺序列表
									ArrTime[ABC[v2][i]][v2] = curTime + UAVtau[v2][i][A[v2][i]][ABC[v2][i]]  # 计算v2到达回收点的时间
									EndTime[ABC[v2][i]][v2] = curTime + inieee[v2][i][A[v2][i]][ABC[v2][i]]   # 计算v2必须被回收的最晚时间

									decvarchecktprime[v2][A[v2][i]] = curTime + iniTauprimeF[v2][i][A[v2][i]]  # 到达客户点时间
									decvarhattprime[v2][A[v2][i]] = curTime + iniTauprimeF[v2][i][A[v2][i]] + sigmaprime[A[v2][i]]  # 离开客户点时间

								availVehicles[i].remove(v2)   # 将v2从当前节点i的可用车辆中移除

								break  # 插入一个任务后，停止尝试插入更多任务，重新评估（因为时间已更新）

							else: # If a launch is not possible before retrieval, just schedule the arrival  # 如果没有可插入的任务 (或者上面尝试插入失败) # 如果没有可插入的任务 (或者上面尝试插入失败)
								curTime = float(ArrTime[i][v])   # 将当前时间推进到v的到达时间
								curTime += sR[v][i]   # 将当前时间推进到v的到达时间
								decvarchecktprime[v][i] = curTime  # 记录v回收完成的时间
								availVehicles[i].append(v)   # 回收后，无人机v在节点i变为可用状态
								Retrievals.remove(v)  # 从待回收列表中 移除v
								serviceOrder[i].append((v, 'r'))  # 记录回收活动
								break

					continue

				else: # If there aren't any launch/service possibility, just schedule the arrival  
					curTime = float(ArrTime[i][v])  # 
					curTime += sR[v][i] # 执行回收，更新当前时间
					decvarchecktprime[v][i] = curTime
					availVehicles[i].append(v)
					Retrievals.remove(v)
					serviceOrder[i].append((v, 'r'))

			else: # UAV has to wait for truck
				# 检查现在开始回收是否来得及 (在无人机v的续航结束时间 `EndTime[i][v]` 之前完成)
				if curTime <= EndTime[i][v]: # If there is enough endurance left

					curTime += sR[v][i]  # 执行回收，更新当前时间
					decvarchecktprime[v][i] = curTime # 执行回收，更新当前时间
					availVehicles[i].append(v) # 无人机v变为可用
					Retrievals.remove(v)   # 移出待回收列表
					serviceOrder[i].append((v, 'r'))  # 记录回收活动
				# 情况3: 回收不可行 - 卡车太晚或前面任务延误过多
				else: # If not enough endurance, then take enough launches out of the service order so that the retrieval becomes feasible
					dummyLaunch = []   # 临时存储被撤销的发射/服务活动
					dummyTime = curTime   # 复制当前时间，用于模拟计算
					dummyOrder = list(serviceOrder[i])   # 复制当前时间，用于模拟计算
					dummyOrder.append((v, 'r'))  # 假设性地将不可行的回收v加到最后，用于时间检查
					# 从后往前遍历当前节点已安排的活动 `serviceOrder[i]`
					for revIndex in range(len(serviceOrder[i])-1, -1, -1):
						dv = serviceOrder[i][revIndex][0]  # 活动涉及的车辆/无人机
						dt = serviceOrder[i][revIndex][1]  # 活动类型 (发射/服务/回收)

						if dt in ('l','s'):  # 只考虑撤销发射('l')或服务('s')活动，不撤销之前的回收('r')
							if dt == 'l':  # 从模拟时间中减去该活动所花费的时间
								dummyTime -= sL[dv][i]  # 减去发射时间
							elif dt == 's':
								dummyTime -= sigma[i]  # 减去服务时间
							dummyOrder.remove((dv,dt))  # 从模拟顺序中移除该活动，并加入到临时撤销列表
							dummyLaunch.append((dv,dt))

							if dummyTime <= EndTime[i][v]:# 检查撤销这个活动后，回收v是否变得可行# 回收最早开始时间是 max(撤销后的时间dummyTime, v的到达时间ArrTime[i][v])# 检查这个开始时间 + 回收时间 是否 <= v的续航结束时间
								pivot = revIndex - 1 # 检查撤销这个活动后，回收v是否变得可行
								break# 检查撤销这个活动后，回收v是否变得可行

					dummyLaunch.reverse()  # 将被撤销的活动恢复原始相对顺序

					serviceOrder[i] = list(dummyOrder + dummyLaunch)  # 更新实际的活动顺序: 回溯点之前的活动 + 现在可行的回收v + 被撤销的活动(按原序放回)
					availVehicles[i].append(v)  #  无人机v现在被回收了，变为可用
					Retrievals.remove(v)    # 从待回收列表移除v

					# Set back the current time, so that re-scheduling can happen:  
					if serviceOrder[i][pivot][1] == 'l':  # 重置为上一个发射的完成时间
						curTime = decvarhattprime[serviceOrder[i][pivot][0]][i]
					elif serviceOrder[i][pivot][1] == 'r':  # 重置为上一个回收的完成时间
						curTime = decvarchecktprime[serviceOrder[i][pivot][0]][i]
					elif serviceOrder[i][pivot][1] == 's':  # 重置为上一个服务的完成时间
						curTime = decvarbart[i]

					# Re-schedule:  # 根据新的 `serviceOrder[i]`，重新模拟计算从 `pivot+1` 开始的所有活动的时间
					for newIndex in range(pivot+1, len(serviceOrder[i])):
						dv = serviceOrder[i][newIndex][0]   # 车辆
						dt = serviceOrder[i][newIndex][1]   # 活动类型

						if dt == 's':  # 重新安排服务
							curTime += sigma[i]
							decvarbart[i] = curTime

						elif dt == 'l':  # 重新安排服务
							curTime += sL[dv][i]
							decvarhattprime[dv][i] = curTime
							# 重新计算这个发射无人机的下游时间点
							decvarchecktprime[dv][A[dv][i]] = curTime + iniTauprimeF[dv][i][A[dv][i]]
							decvarhattprime[dv][A[dv][i]] = curTime + iniTauprimeF[dv][i][A[dv][i]] + sigmaprime[A[dv][i]]

							ArrTime[ABC[dv][i]][dv] = curTime + UAVtau[dv][i][A[dv][i]][ABC[dv][i]]
							EndTime[ABC[dv][i]][dv] = curTime + inieee[dv][i][A[dv][i]][ABC[dv][i]]

						elif dt == 'r':  # 重新安排回收 (这个就是刚才导致回溯的那个v)  # 回收开始时间必须 >= 当前时间和无人机到达时间
							curTime = max(curTime, ArrTime[i][dv])
							curTime += sR[dv][i]   # 执行回收
							decvarchecktprime[dv][i] = curTime



		# Schedule launches/service:  安排剩余的发射任务
		for v in Launches:  # 遍历优先发射/服务列表
			if v in availVehicles[i]:  # 如果v当前可用

				if v == 1:  # 卡车服务
					curTime += sigma[i]  # 加上服务时间
					decvarbart[i] = curTime  # 加上服务时间
					serviceOrder[i].append((v, 's'))  # 加上服务时间
					availVehicles[i].remove(v)  # 标记卡车服务已完成 (概念上)
				else:
					curTime += sL[v][i]  # 加上发射准备时间
					decvarhattprime[v][i] = curTime  # 加上发射准备时间
					serviceOrder[i].append((v, 'l'))  # 记录发射完成时间
					ArrTime[ABC[v][i]][v] = curTime + UAVtau[v][i][A[v][i]][ABC[v][i]]  # 计算无人机v到达回收点的时间
					EndTime[ABC[v][i]][v] = curTime + inieee[v][i][A[v][i]][ABC[v][i]]  # 计算无人机v必须被回收的最晚时间
					availVehicles[i].remove(v)  # 标记无人机v已发射

					decvarchecktprime[v][A[v][i]] = curTime + iniTauprimeF[v][i][A[v][i]]  # 计算无人机v到达客户点的时间
					decvarhattprime[v][A[v][i]] = curTime + iniTauprimeF[v][i][A[v][i]] + sigmaprime[A[v][i]]  # 计算无人机v离开客户点的时间


		# Leave:# --- 第7阶段: 前往下一节点 --- # 如果当前节点i不是最后一个节点 (c+1是终点)
		decvarhatt[i] = curTime

		if i != c+1:  # 获取卡车路径上的下一个节点k
			k = TSPtour[TSPtour.index(i)+1]
			curTime += tau[i][k]# 获取卡车路径上的下一个节点k


	#####################################################################################		

	# Values that are input parameters to the local search procedure:
	ls_hatt = {}
	ls_checkt = {}
	ls_checktprime = {}

	# Final UAV travel times and UAV speeds, to be calculated after the speed modification:
	finTauprimeF = defaultdict(make_dict)
	finTauprimeE = defaultdict(make_dict)
	fineee 		 = defaultdict(make_dict)
	finSpeedF 	 = defaultdict(make_dict)
	finSpeedE 	 = defaultdict(make_dict)
	

	if (Status == 'Infeasible'):
		# NO FEASIBLE SOLUTION

		assignmentsArray 	= []
		packagesArray 		= []
		p3isFeasible 		= False
		p3OFV				= float('inf')
		waitingTruck		= -1
		waitingUAV			= -1
		waitingArray 		= {}
		oldWaitingArray 	= {}						
	
	else:
		# A feasible solution is FOUND

		p3isFeasible 			= True
		p3OFV 					= decvarhatt[c+1]
		waitingTruck			= 0.0
		waitingUAV				= 0.0

		for [v,i,j,k] in y:
			finTauprimeF[v][i][j] = float(iniTauprimeF[v][i][j])
			finTauprimeE[v][j][k] = float(iniTauprimeE[v][j][k])
			finSpeedF[v][i][j] = float(iniSpeedF[v][i][j])
			finSpeedE[v][j][k] = float(iniSpeedE[v][j][k])
			fineee[v][i][j][k] = float(inieee[v][i][j][k])

		#-------------------------------------ALGORITHM 10 (Modify UAV cruise speeds to reduce truck waiting) STARTS HERE--------------------------------------#

		# New timing variables, whose values are obtained after the speed modification:
		newcheckt 	= defaultdict(make_dict)
		newbart 	= defaultdict(make_dict)
		newhatt 	= defaultdict(make_dict)
		newchecktprime 	= defaultdict(make_dict)
		newhattprime 	= defaultdict(make_dict)

		if UAVSpeedType == 1: # Solving for mFSTSP-VDS (speed modification required)
			currentTime = 0

			for [i,k] in x:
				
				if i == 0: # Schedule the launches at the depot
					newcheckt[i] = currentTime
					newbart[i] = currentTime

					activities = []

					for v in launchesfrom[i]:
						heappush(activities, (decvarhattprime[v][i], ['hattprime',v,i]))

					while len(activities) > 0:
						actTime, act = heappop(activities)

						if act[0] == 'hattprime':
							currentTime += sL[act[1]][act[2]]
							newhattprime[act[1]][act[2]] = currentTime

					newhatt[i] = currentTime

				currentTime += tau[i][k]
				newcheckt[k] = currentTime

				activities = []

				for v in landsat[k]:
					heappush(activities, (decvarchecktprime[v][k], ['checktprime',v,k]))

				for v in launchesfrom[k]:
					heappush(activities, (decvarhattprime[v][k], ['hattprime',v,k]))

				heappush(activities, (decvarbart[k], ['bart',k]))

				while len(activities) > 0:
					actTime, act = heappop(activities)

					if act[0] == 'bart': # The timings do not change corresponding to the service
						currentTime += sigma[k]
						newbart[k] = currentTime

					elif act[0] == 'hattprime': # # The timings do not change corresponding to the launch
						currentTime += sL[act[1]][act[2]]
						newhattprime[act[1]][act[2]] = currentTime

					elif act[0] == 'checktprime': # Modify UAV timings corresponding to retrievals, if possible
						potCurrentTime = currentTime + sR[act[1]][act[2]]
						vcur = act[1]
						icur = AB[act[1]][act[2]]
						jcur = B[act[1]][act[2]]
						kcur = act[2]

						## First check if there is truck waiting, then only do the below procedure:
						if ((decvarchecktprime[vcur][kcur] - decvarhattprime[vcur][icur]) - (potCurrentTime - newhattprime[vcur][icur]) > 0.0001) and ((newhattprime[vcur][icur] + iniTauprimeF[vcur][icur][jcur] + sigmaprime[jcur] + iniTauprimeE[vcur][jcur][kcur] + sR[vcur][kcur]) > potCurrentTime):
							if kcur == c+1:
								[newchecktprime[vcur][jcur], newhattprime[vcur][jcur], currentTime, finTauprimeF[vcur][icur][jcur], finTauprimeE[vcur][jcur][kcur], finSpeedF[vcur][icur][jcur], finSpeedE[vcur][jcur][kcur]] = endurance_calculator.modify_speed(decvarhattprime[vcur][icur], decvarchecktprime[vcur][kcur], newhattprime[vcur][icur], potCurrentTime, vcur, icur, jcur, 0, node, vehicle, travel, iniTauprimeF[vcur][icur][jcur], iniTauprimeE[vcur][jcur][kcur], iniSpeedF[vcur][icur][jcur], iniSpeedE[vcur][jcur][kcur])						
							else:
								[newchecktprime[vcur][jcur], newhattprime[vcur][jcur], currentTime, finTauprimeF[vcur][icur][jcur], finTauprimeE[vcur][jcur][kcur], finSpeedF[vcur][icur][jcur], finSpeedE[vcur][jcur][kcur]] = endurance_calculator.modify_speed(decvarhattprime[vcur][icur], decvarchecktprime[vcur][kcur], newhattprime[vcur][icur], potCurrentTime, vcur, icur, jcur, kcur, node, vehicle, travel, iniTauprimeF[vcur][icur][jcur], iniTauprimeE[vcur][jcur][kcur], iniSpeedF[vcur][icur][jcur], iniSpeedE[vcur][jcur][kcur])						

							# Update eee:
							if kcur == c+1:
								fineee[vcur][icur][jcur][kcur] = endurance_calculator.give_endurance(node, vehicle, travel, vcur, icur, jcur, 0, finTauprimeF[vcur][icur][jcur], finTauprimeE[vcur][jcur][kcur], finSpeedF[vcur][icur][jcur], finSpeedE[vcur][jcur][kcur])
							else:
								fineee[vcur][icur][jcur][kcur] = endurance_calculator.give_endurance(node, vehicle, travel, vcur, icur, jcur, kcur, finTauprimeF[vcur][icur][jcur], finTauprimeE[vcur][jcur][kcur], finSpeedF[vcur][icur][jcur], finSpeedE[vcur][jcur][kcur])

							newchecktprime[vcur][kcur] = currentTime
						else:
							currentTime = potCurrentTime
							newchecktprime[vcur][kcur] = currentTime
							newchecktprime[vcur][jcur] = newhattprime[vcur][icur] + iniTauprimeF[vcur][icur][jcur]
							newhattprime[vcur][jcur] = newchecktprime[vcur][jcur] + sigmaprime[jcur]

				newhatt[k] = max(currentTime, newbart[k])

		else: # Solving for mFSTSP (No speed modification required)
			for i in decvarcheckt:
				newcheckt[i] = float(decvarcheckt[i])
			for i in decvarbart:
				newbart[i] = float(decvarbart[i])
			for i in decvarhatt:
				newhatt[i] = float(decvarhatt[i])
			for v in decvarchecktprime:
				for i in decvarchecktprime[v]:
					newchecktprime[v][i] = float(decvarchecktprime[v][i])
			for v in decvarhattprime:
				for i in decvarhattprime[v]:
					newhattprime[v][i] = float(decvarhattprime[v][i])

		####################################################################################################################################

		#-------------------------------------ALGORITHM 11 (Store the final activity timings, and update the incumbent) STARTS HERE--------------------------------------#

		# Update the objective function:
		p3OFV = newhatt[c+1]  # 使用优化后(或未优化时直接复制的)卡车离开终点站的时间作为最终的目标函数值 (总时长)

		# BUILD ASSIGNMENTS AND PACKAGES DICTIONARIES:  (构建任务分配和包裹信息字典)

		prevTime = {}  # 字典: 记录每个载具 (1=卡车, v=UAV) 上一个活动完成的时间，用于计算空闲
		assignmentsArray = {}  # 字典: 存储每个载具的详细活动列表 (key: 1=卡车, v=UAV; value: list of activities)
		packagesArray = {}   # 字典: 存储每个客户节点的包裹交付信息 (key: 客户j; value: [类型, lat, lon, 时间, 图标])
		waitingArray = {}  # 字典: 存储优化 *后* 的等待时间 (key: 节点i/客户j; value: 等待秒数)
		oldWaitingArray = {}   # 字典: 存储优化 *前* 的等待时间 (key: 节点i/客户j; value: 等待秒数)
		waitingArray[0] = 0  # 初始化起点等待时间为0
		oldWaitingArray[0] = 0   # 初始化起点等待时间为0
		
		prevTime[1] = 0.0	# truck  # truck - 初始化卡车上一个活动完成时间为0
		assignmentsArray[1] = []   # 初始化卡车的活动列表为空列表
		for v in V:  # 遍历所有无人机 V
			prevTime[v] = 0.0	# UAVs  # UAVs - 初始化无人机 v 上一个活动完成时间为0
			assignmentsArray[v] = []  # 初始化无人机 v 的活动列表为空列表
			
		# Are there any UAVs on the truck?
		uavRiders = []  # 列表: 记录当前在卡车上的无人机编号
		for v in V:   # 初始状态，所有无人机都在起点 (假设在卡车上)
			uavRiders.append(v)
			
			
		tmpIcon = 'ub_truck_%d.gltf' % (1)  # 为卡车设置一个临时图标路径				
		for [i,j] in x:  # 遍历卡车行驶的每一段路径 (从 i 到 j)
			# Capture the waiting time  # 卡车总等待时间 += (实际在j到达时间 - 实际在i到达时间) - (理论最短行驶+服务时间)
			waitingTruck += ((newcheckt[j] - newcheckt[i]) - (tau[i][j] + sigma[j]))

			dummy_1 = 0   # 计算节点i总的发射准备时间
			for v in launchesfrom[i]:
				dummy_1 += sL[v][i]

			dummy_2 = 0   # 计算节点i总的回收操作时间
			for v in landsat[i]:
				dummy_2 += sR[v][i]
  
			waitingArray[i] = (newcheckt[j] - newcheckt[i]) - (tau[i][j] + sigma[i] + dummy_1 + dummy_2)  # 优化后的等待  # 这代表了卡车在i点停留期间，除去必要操作之外的空闲时间。
			oldWaitingArray[i] = (decvarcheckt[j] - decvarcheckt[i]) - (tau[i][j] + sigma[i] + dummy_1 + dummy_2)  # 优化前的等待 # 计算节点 i 的等待时间 (优化前) - 使用 decvar 时间点
			
			# Are there any UAVs on the truck when the truck leaves i?  (更新卡车离开 i 时，车上还有哪些无人机)
			for v in V:
				if ((v in landsat[i]) and (v not in uavRiders)):  # 如果无人机 v 在 i 点降落且之前不在车上
					uavRiders.append(v)   # 添加到车上列表
				if ((v in launchesfrom[i]) and (v in uavRiders)):   # 如果无人机 v 在 i 点发射且之前在车上
					uavRiders.remove(v)  # 从车上列表移除
	
			# These activities need to be sorted by time (ascending)  这些活动需要按开始时间升序排序)
			tmpTimes = []  # 临时列表，存储与当前段 i->j 相关的所有卡车活动
						
			if (i == 0):  # 活动 1: 起点 (i=0) 的无人机发射准备 (如果 i 是起点)
				for v in launchesfrom[i]:   # 遍历在起点发射的无人机
					if (len(uavRiders) > 0):  # 判断卡车状态 (是否载有其他无人机)
						A_statusID = STATIONARY_TRUCK_W_UAV   # 静止卡车 - 有无人机
					else:
						A_statusID = STATIONARY_TRUCK_EMPTY  # 静止卡车 - 空
					A_vehicleType = TYPE_TRUCK  # 载具类型: 卡车
					A_startTime = newhattprime[v][i] - sL[v][i]  # 开始时间 = 发射完成时间 - 发射准备时长
					A_startNodeID = i  # 开始节点
					A_startLatDeg = node[i].latDeg  # 开始纬度
					A_startLonDeg = node[i].lonDeg  # 开始经度
					A_startAltMeters = 0.0  # 开始高度
					A_endTime = newhattprime[v][i]  # 结束时间 = 发射完成时间
					A_endNodeID = i  # 结束节点
					A_endLatDeg = node[i].latDeg  # 结束纬度
					A_endLonDeg = node[i].lonDeg  # 结束经度
					A_endAltMeters = 0.0  # 结束高度
					A_icon = tmpIcon  # 图标路径
					A_description = 'Launching UAV %d' % (v)  # 描述: 发射无人机 v
					A_UAVsOnBoard = uavRiders # 车上无人机列表 ([:] 创建副本)
					A_ganttStatus = GANTT_LAUNCH  # 甘特图状态: 发射
					# 将活动记录添加到临时列表
					tmpTimes.append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		


			if (len(uavRiders) > 0):  # 活动 2: 卡车从 i 行驶到 j
				A_statusID = TRAVEL_TRUCK_W_UAV  # 行驶卡车 - 有无人机
			else:
				A_statusID = TRAVEL_TRUCK_EMPTY  # 行驶卡车 - 空
			A_vehicleType = TYPE_TRUCK  # 载具类型: 卡车	
			A_startTime = newhatt[i]  # 开始时间 = 卡车离开 i 的时间
			A_startNodeID = i
			A_startLatDeg = node[i].latDeg
			A_startLonDeg = node[i].lonDeg
			A_startAltMeters = 0.0
			A_endTime = newhatt[i] + tau[i][j]  # 结束时间 = 离开 i 时间 + 行驶时间
			A_endNodeID = j  # 结束节点 j
			A_endLatDeg = node[j].latDeg
			A_endLonDeg = node[j].lonDeg
			A_endAltMeters = 0.0
			A_icon = tmpIcon
			A_description = 'Travel from node %d to node %d' % (i, j)  # 描述: 从 i 到 j
			A_UAVsOnBoard = uavRiders
			A_ganttStatus = GANTT_TRAVEL  # 甘特图状态: 行驶

	
			tmpTimes.append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
	
			if (newcheckt[j] - newhatt[i] - tau[i][j] > 0.01):  # 活动 3: 卡车到达 j 后可能的等待 (Idle Time)  如果卡车实际到达 j 的时间 (newcheckt[j]) 晚于 (离开i时间 + 行驶时间)，说明有等待
				if (len(uavRiders) > 0):
					A_statusID = STATIONARY_TRUCK_W_UAV
				else:
					A_statusID = STATIONARY_TRUCK_EMPTY
				A_vehicleType = TYPE_TRUCK
				A_startTime = (newhatt[i] + tau[i][j])  # 等待开始时间 = 理论到达时间
				A_startNodeID = j
				A_startLatDeg = node[j].latDeg
				A_startLonDeg = node[j].lonDeg
				A_startAltMeters = 0.0
				A_endTime = newcheckt[j]  # 等待结束时间 = 实际到达时间 (也是后续活动开始时间)
				A_endNodeID = j
				A_endLatDeg = node[j].latDeg
				A_endLonDeg = node[j].lonDeg
				A_endAltMeters = 0.0
				A_icon = tmpIcon
				A_description = 'Idle for %3.0f seconds at node %d' % (A_endTime - A_startTime, j)
				A_UAVsOnBoard = uavRiders 
				A_ganttStatus = GANTT_IDLE   # 甘特图状态: 空闲
	
				tmpTimes.append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
				
			# 活动 4: 卡车在节点 j 的服务活动
			if (j == c+1): # 如果 j 是终点站
				myMin, mySec	= divmod(newhatt[j], 60)   # 计算总时间的小时:分钟:秒
				myHour, myMin 	= divmod(myMin, 60)
				A_description	= 'At the Depot.  Total Time = %d:%02d:%02d' % (myHour, myMin, mySec)   # 描述: 到达终点及总时间
				A_endTime		= -1  # 结束时间设为-1表示流程结束
				A_ganttStatus	= GANTT_FINISHED  # 甘特图状态: 完成
			else:
				A_description		= 'Dropping off package to Customer %d' % (j)  # 描述: 为客户 j 送货
				A_endTime 		= newbart[j]  # 结束时间 = 服务完成时间
				A_ganttStatus 	= GANTT_DELIVER  # 甘特图状态: 交付
				packagesArray[j] = [TYPE_TRUCK, node[j].latDeg, node[j].lonDeg, newbart[j], packageIcons[1]]  # 记录包裹交付信息

		
			if (len(uavRiders) > 0):  # 服务活动的状态
				A_statusID = STATIONARY_TRUCK_W_UAV
			else:
				A_statusID = STATIONARY_TRUCK_EMPTY
			A_vehicleType = TYPE_TRUCK
			if (j == c+1):  # 如果是终点，服务开始时间=终点到达时间 (假设终点服务时间为0?)
				A_startTime = newhatt[j] - sigma[j]
			else:	  # 如果是客户点，服务开始时间 = 服务完成时间 - 服务时长
				A_startTime = newbart[j] - sigma[j]
			A_startNodeID = j
			A_startLatDeg = node[j].latDeg
			A_startLonDeg = node[j].lonDeg
			A_startAltMeters = 0.0
			A_endNodeID = j
			A_endLatDeg = node[j].latDeg
			A_endLonDeg = node[j].lonDeg
			A_endAltMeters = 0.0
			A_icon = tmpIcon
			A_UAVsOnBoard = uavRiders
	
			tmpTimes.append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
			if (j <= c+1):  # 活动 5: 在节点 j 的无人机回收准备 (如果 j <= c+1, 即非虚拟终点?)
				# We're NOT going to ignore UAVs that land at the depot.
				for v in landsat[j]:   # 遍历在 j 点降落的无人机
					if (len(uavRiders) > 0):
						A_statusID = STATIONARY_TRUCK_W_UAV
					else:
						A_statusID = STATIONARY_TRUCK_EMPTY
					A_vehicleType = TYPE_TRUCK
					A_startTime = newchecktprime[v][j] - sR[v][j]
					A_startNodeID = j
					A_startLatDeg = node[j].latDeg
					A_startLonDeg = node[j].lonDeg
					A_startAltMeters = 0.0
					A_endTime = newchecktprime[v][j]
					A_endNodeID = j
					A_endLatDeg = node[j].latDeg
					A_endLonDeg = node[j].lonDeg
					A_endAltMeters = 0.0
					A_icon = tmpIcon
					A_description = 'Retrieving UAV %d' % (v)
					A_UAVsOnBoard = uavRiders
					A_ganttStatus = GANTT_RECOVER
			
					tmpTimes.append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
			# 活动 6: 在节点 j 的无人机发射准备
			for v in launchesfrom[j]:
				if (len(uavRiders) > 0):
					A_statusID = STATIONARY_TRUCK_W_UAV
				else:
					A_statusID = STATIONARY_TRUCK_EMPTY
				A_vehicleType = TYPE_TRUCK
				A_startTime = newhattprime[v][j] - sL[v][j]
				A_startNodeID = j
				A_startLatDeg = node[j].latDeg
				A_startLonDeg = node[j].lonDeg
				A_startAltMeters = 0.0
				A_endTime = newhattprime[v][j]
				A_endNodeID = j
				A_endLatDeg = node[j].latDeg
				A_endLonDeg = node[j].lonDeg
				A_endAltMeters = 0.0
				A_icon = tmpIcon
				A_description = 'Launching UAV %d' % (v)
				A_UAVsOnBoard = uavRiders
				A_ganttStatus = GANTT_LAUNCH
		
				tmpTimes.append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		

	
			# Now, sort the tmpTimes array based on ascending start times.  
			# Along the way, check for truck idle times. 现在，基于开始时间升序排列 tmpTimes 数组。同时检查卡车空闲时间。
			unasgnInd = list(range(0, len(tmpTimes)))		# 创建包含 tmpTimes 所有索引的列表，用于追踪未分配的活动 1. 创建一个索引列表，代表 tmpTimes 中所有待处理的活动。
			while (len(unasgnInd) > 0):   # 当还有未分配的活动时循环
				tmpMin = 2*newhatt[j]	# Set to a large number (初始化最早开始时间为一个很大的数)
				# Find the minimum unassigned time 找到未分配活动中的最早开始时间)
				for tmpIndex in unasgnInd:  # 遍历未分配活动的索引
					if (tmpTimes[tmpIndex][2] < tmpMin):   # 如果当前活动的开始时间 [2] 更早
						tmpMin = tmpTimes[tmpIndex][2]	# This is the "startTime" component of tmpTimes  (更新最早开始时间)
						myIndex = tmpIndex   # 记录下这个最早开始活动的索引
						
				# Was there idle time in the assignments? 检查是否有空闲时间?  如果最早活动开始时间 tmpMin 比 上一个活动结束时间 prevTime[1] 晚超过 0.01 秒
				if (tmpMin - prevTime[1] > 0.01):
					# MAKE ASSIGNMENT:	
					if (len(tmpTimes[myIndex][14]) > 0):
						A_statusID = STATIONARY_TRUCK_W_UAV
					else:
						A_statusID = STATIONARY_TRUCK_EMPTY
					A_vehicleType = TYPE_TRUCK
					A_startTime = prevTime[1]
					A_startNodeID = tmpTimes[myIndex][3]
					A_startLatDeg = node[tmpTimes[myIndex][3]].latDeg
					A_startLonDeg = node[tmpTimes[myIndex][3]].lonDeg
					A_startAltMeters = 0.0
					A_endTime = prevTime[1] + (tmpMin - prevTime[1])
					A_endNodeID = tmpTimes[myIndex][3]
					A_endLatDeg = node[tmpTimes[myIndex][3]].latDeg
					A_endLonDeg = node[tmpTimes[myIndex][3]].lonDeg
					A_endAltMeters = 0.0
					A_icon = tmpIcon
					A_description = 'Idle for %3.0f seconds' % (A_endTime - A_startTime)
					A_UAVsOnBoard = tmpTimes[myIndex][14]
					A_ganttStatus = GANTT_IDLE
			
					assignmentsArray[1].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
											
				# MAKE ASSIGNMENT:
				assignmentsArray[1].append(tmpTimes[myIndex])
							
				prevTime[1] = tmpTimes[myIndex][7] 		# This is the "endTime" component of tmpTimes
				unasgnInd.remove(myIndex)

				
			# Also, is there idle time before leaving node j?  Check prevTime[1] and decvarhatt[j].x
			if ((j != c+1) and (prevTime[1] - newhatt[j] < -0.01)):
				# MAKE ASSIGNMENT:			
				if (len(tmpTimes[myIndex][14]) > 0):
					A_statusID = STATIONARY_TRUCK_W_UAV
				else:
					A_statusID = STATIONARY_TRUCK_EMPTY
				A_vehicleType = TYPE_TRUCK
				A_startTime = tmpMin
				A_startNodeID = tmpTimes[myIndex][3]
				A_startLatDeg = node[tmpTimes[myIndex][3]].latDeg
				A_startLonDeg = node[tmpTimes[myIndex][3]].lonDeg
				A_startAltMeters = 0.0
				A_endTime = newhatt[j]
				A_endNodeID = tmpTimes[myIndex][3]
				A_endLatDeg = node[tmpTimes[myIndex][3]].latDeg
				A_endLonDeg = node[tmpTimes[myIndex][3]].lonDeg
				A_endAltMeters = 0.0
				A_icon = tmpIcon
				A_description = 'Idle for %3.0f seconds before departing' % (A_endTime - A_startTime)
				A_UAVsOnBoard = tmpTimes[myIndex][14]
				A_ganttStatus = GANTT_IDLE
		
				assignmentsArray[1].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
				
	
			# Update the previous time value:					
			prevTime[1] = newhatt[j]
			
	
	
		for [v,i,j,k] in y:
			# Capture waiting time for UAVs
			waitingUAV += ((newchecktprime[v][k] - newcheckt[i]) - (finTauprimeF[v][i][j] + finTauprimeE[v][j][k] + sigmaprime[j] + sL[v][i] + sR[v][k]))
			
			waitingArray[j] = ((newchecktprime[v][k] - newcheckt[i]) - (finTauprimeF[v][i][j] + finTauprimeE[v][j][k] + sigmaprime[j] + sL[v][i] + sR[v][k]))
			oldWaitingArray[j] = ((decvarchecktprime[v][k] - decvarcheckt[i]) - (iniTauprimeF[v][i][j] + iniTauprimeE[v][j][k] + sigmaprime[j] + sL[v][i] + sR[v][k]))
			
			# Launch Prep (on ground, with package)		
			A_statusID = STATIONARY_UAV_PACKAGE
			A_vehicleType = TYPE_UAV
			A_startTime = newhattprime[v][i] - sL[v][i]
			A_startNodeID = i
			A_startLatDeg = node[i].latDeg
			A_startLonDeg = node[i].lonDeg
			A_startAltMeters = 0.0
			A_endTime = newhattprime[v][i]
			A_endNodeID = i
			A_endLatDeg = node[i].latDeg
			A_endLonDeg = node[i].lonDeg
			A_endAltMeters = 0.0
			A_icon = 'iris_black_blue_plus_box_yellow.gltf'
			A_description = 'Prepare to launch from truck'
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_LAUNCH
	
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
	
			# Takoff (vertical, with package)
			A_statusID = VERTICAL_UAV_PACKAGE
			A_vehicleType = TYPE_UAV
			A_startTime = newhattprime[v][i]
			A_startNodeID = i
			A_startLatDeg = node[i].latDeg
			A_startLonDeg = node[i].lonDeg
			A_startAltMeters = 0.0
			A_endTime = newhattprime[v][i] + travel[v][i][j].takeoffTime
			A_endNodeID = i
			A_endLatDeg = node[i].latDeg
			A_endLonDeg = node[i].lonDeg
			A_endAltMeters = vehicle[v].cruiseAlt
			A_icon = 'iris_black_blue_plus_box_yellow.gltf'
			if (i == 0):
				A_description = 'Takeoff from Depot'
			else:	
				A_description = 'Takeoff from truck at Customer %d' % (i)
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_TRAVEL
	
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
			
			tmpStart = newhattprime[v][i] + travel[v][i][j].takeoffTime
			# Idle (i --> j)?
			if (newchecktprime[v][j] - newhattprime[v][i] - finTauprimeF[v][i][j] > 0.01):
				tmpIdle = newchecktprime[v][j] - (newhattprime[v][i] + finTauprimeF[v][i][j])
				tmpEnd = tmpStart + tmpIdle
				A_statusID = STATIONARY_UAV_PACKAGE
				A_vehicleType = TYPE_UAV
				A_startTime = tmpStart
				A_startNodeID = i
				A_startLatDeg = node[i].latDeg
				A_startLonDeg = node[i].lonDeg
				A_startAltMeters = vehicle[v].cruiseAlt
				A_endTime = tmpEnd
				A_endNodeID = i
				A_endLatDeg = node[i].latDeg
				A_endLonDeg = node[i].lonDeg
				A_endAltMeters = vehicle[v].cruiseAlt
				A_icon = 'iris_black_blue_plus_box_yellow.gltf'
				A_description = 'Idle at initial takeoff (node %d)' % (i)
				A_UAVsOnBoard = []
				A_ganttStatus = GANTT_IDLE
				
				assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
				tmpStart = tmpEnd
	
				
			# Fly to customer j (with package)
			A_statusID = TRAVEL_UAV_PACKAGE
			A_vehicleType = TYPE_UAV
			A_startTime = tmpStart
			A_startNodeID = i
			A_startLatDeg = node[i].latDeg
			A_startLonDeg = node[i].lonDeg
			A_startAltMeters = vehicle[v].cruiseAlt
			A_endTime = tmpStart + (finTauprimeF[v][i][j] - travel[v][i][j].takeoffTime - travel[v][i][j].landTime)
			A_endNodeID = j
			A_endLatDeg = node[j].latDeg
			A_endLonDeg = node[j].lonDeg
			A_endAltMeters = vehicle[v].cruiseAlt
			A_icon = 'iris_black_blue_plus_box_yellow.gltf'
			A_description = 'Fly to UAV customer %d' % (j)
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_TRAVEL
			
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
			
			
			# Land at customer (Vertical, with package)
			A_statusID = VERTICAL_UAV_PACKAGE
			A_vehicleType = TYPE_UAV
			A_startTime = newchecktprime[v][j] - travel[v][i][j].landTime
			A_startNodeID = j
			A_startLatDeg = node[j].latDeg
			A_startLonDeg = node[j].lonDeg
			A_startAltMeters = vehicle[v].cruiseAlt
			A_endTime = newchecktprime[v][j]
			A_endNodeID = j
			A_endLatDeg = node[j].latDeg
			A_endLonDeg = node[j].lonDeg
			A_endAltMeters = 0.0
			A_icon = 'iris_black_blue_plus_box_yellow.gltf'
			A_description = 'Land at UAV customer %d' % (j)
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_TRAVEL
			
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
			
	
			# Serve customer (on ground, with package)
			A_statusID = STATIONARY_UAV_PACKAGE
			A_vehicleType = TYPE_UAV
			A_startTime = newchecktprime[v][j]
			A_startNodeID = j
			A_startLatDeg = node[j].latDeg
			A_startLonDeg = node[j].lonDeg
			A_startAltMeters = 0.0
			A_endTime = newhattprime[v][j]
			A_endNodeID = j
			A_endLatDeg = node[j].latDeg
			A_endLonDeg = node[j].lonDeg
			A_endAltMeters = 0.0
			A_icon = 'iris_black_blue_plus_box_yellow.gltf'
			A_description = 'Serving UAV customer %d' % (j)
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_DELIVER
			
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		

			packagesArray[j] = [TYPE_UAV, node[j].latDeg, node[j].lonDeg, newhattprime[v][j], packageIcons[0]]
	
			
			# Takeoff (vertical, empty)
			A_statusID = VERTICAL_UAV_EMPTY
			A_vehicleType = TYPE_UAV
			A_startTime = newhattprime[v][j]
			A_startNodeID = j
			A_startLatDeg = node[j].latDeg
			A_startLonDeg = node[j].lonDeg
			A_startAltMeters = 0.0
			if (k == c+1):
				# We didn't define "travel" ending at the depot replica.
				A_endTime = newhattprime[v][j] + travel[v][j][0].takeoffTime
			else:
				A_endTime = newhattprime[v][j] + travel[v][j][k].takeoffTime
			A_endNodeID = j
			A_endLatDeg = node[j].latDeg
			A_endLonDeg = node[j].lonDeg
			A_endAltMeters = vehicle[v].cruiseAlt
			A_icon = 'iris_with_props_black_blue.gltf'
			A_description = 'Takeoff from UAV customer %d' % (j)
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_TRAVEL
			
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
			
			# Fly to truck (empty)
			A_statusID = TRAVEL_UAV_EMPTY
			A_vehicleType = TYPE_UAV
			if (k == c+1):
				# We didn't define "travel" ending at the depot replica.
				A_startTime = newhattprime[v][j] + travel[v][j][0].takeoffTime
			else:
				A_startTime = newhattprime[v][j] + travel[v][j][k].takeoffTime
			A_startNodeID = j
			A_startLatDeg = node[j].latDeg
			A_startLonDeg = node[j].lonDeg
			A_startAltMeters = vehicle[v].cruiseAlt
			if (k == c+1):
				A_endTime = newhattprime[v][j] + (finTauprimeE[v][j][k] - travel[v][j][0].landTime)
			else:
				A_endTime = newhattprime[v][j] + (finTauprimeE[v][j][k] - travel[v][j][k].landTime)
			A_endNodeID = k
			A_endLatDeg = node[k].latDeg
			A_endLonDeg = node[k].lonDeg
			A_endAltMeters = vehicle[v].cruiseAlt
			A_icon = 'iris_with_props_black_blue.gltf'
			if (k == c+1):
				A_description = 'Fly to depot'
			else:
				A_description = 'Fly to truck at customer %d' % (k)
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_TRAVEL
			
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
			
			# Idle (j --> k)?
			if (k == c+1):
				tmpStart =  newhattprime[v][j] +  (finTauprimeE[v][j][k] - travel[v][j][0].landTime)
			else:
				tmpStart =  newhattprime[v][j] +  (finTauprimeE[v][j][k] - travel[v][j][k].landTime)
			tmpIdle = newchecktprime[v][k] - sR[v][k] - newhattprime[v][j] - finTauprimeE[v][j][k]
				
			if (tmpIdle > 0.01):				
				tmpEnd = tmpStart + tmpIdle
				A_statusID = STATIONARY_UAV_EMPTY
				A_vehicleType = TYPE_UAV
				A_startTime = tmpStart
				A_startNodeID = k
				A_startLatDeg = node[k].latDeg
				A_startLonDeg = node[k].lonDeg
				A_startAltMeters = vehicle[v].cruiseAlt
				A_endTime = tmpEnd
				A_endNodeID = k
				A_endLatDeg = node[k].latDeg
				A_endLonDeg = node[k].lonDeg
				A_endAltMeters = vehicle[v].cruiseAlt
				A_icon = 'iris_with_props_black_blue.gltf'
				if (k == c+1):
					A_description = 'Idle above depot location'
				else:
					A_description = 'Idle above rendezvous location (customer %d)' % (k)					
				A_UAVsOnBoard = []
				A_ganttStatus = GANTT_IDLE
				
				assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
	
				tmpStart = tmpEnd
			
			
			# Land at k (vertical, empty)
			A_statusID = VERTICAL_UAV_EMPTY
			A_vehicleType = TYPE_UAV
			A_startTime = tmpStart
			A_startNodeID = k
			A_startLatDeg = node[k].latDeg
			A_startLonDeg = node[k].lonDeg
			A_startAltMeters = vehicle[v].cruiseAlt
			if (k == c+1):
				# We didn't define "travel" ending at the depot replica.
				A_endTime = tmpStart + travel[v][j][0].landTime
			else:
				A_endTime = tmpStart + travel[v][j][k].landTime
			A_endNodeID = k
			A_endLatDeg = node[k].latDeg
			A_endLonDeg = node[k].lonDeg
			A_endAltMeters = 0.0
			A_icon = 'iris_with_props_black_blue.gltf'
			if (k == c+1):
				A_description = 'Land at depot'
			else:
				A_description = 'Land at truck rendezvous location (customer %d)' % (k)
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_TRAVEL
			
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		
			
			
			# Recovery (on ground, empty)
			A_statusID = STATIONARY_UAV_EMPTY
			A_vehicleType = TYPE_UAV
			A_startTime = newchecktprime[v][k] - sR[v][k]
			A_startNodeID = k
			A_startLatDeg = node[k].latDeg
			A_startLonDeg = node[k].lonDeg
			A_startAltMeters = 0.0
			A_endTime = newchecktprime[v][k]
			A_endNodeID = k
			A_endLatDeg = node[k].latDeg
			A_endLonDeg = node[k].lonDeg
			A_endAltMeters = 0.0
			A_icon = 'iris_with_props_black_blue.gltf'
			if (k == c+1):
				A_description = 'Recovered at depot'
			else:
				A_description = 'Recovered by truck at customer %d' % k
			A_UAVsOnBoard = []
			A_ganttStatus = GANTT_RECOVER
			
			assignmentsArray[v].append([A_statusID, A_vehicleType, A_startTime, A_startNodeID, A_startLatDeg, A_startLonDeg, A_startAltMeters, A_endTime, A_endNodeID, A_endLatDeg, A_endLonDeg, A_endAltMeters, A_icon, A_description, A_UAVsOnBoard, A_ganttStatus])		


		# Capture the values of hatt, checkt and checktprime for use in the local search:
		for k in N:
			if k not in C:
				ls_hatt[k] = decvarhatt[k]
				ls_checkt[k] = decvarcheckt[k]
			for v in V:
				if ((v in landsat[k]) or (v in launchesfrom[k])):
					if (k != 0):
						ls_checktprime[v,k] = decvarchecktprime[v][k]
	  # 布尔值: 调度是否可行, # 浮点数: 最终目标函数值 (总时长),  # 字典: {载具ID: [活动列表]},# 字典: {客户ID: [交付信息]},# 浮点数: 卡车总等待时间 (优化后), # 浮点数: 无人机总等待时间 (优化后) # 字典: {节点/客户ID: 优化前等待时间},# 字典: {节点k: [在该点降落的UAV列表]} (从输入或前序算法传入),# 字典: {节点i: [在该点发射的UAV列表]} (从输入或前序算法传入),
	return (p3isFeasible, p3OFV, assignmentsArray, packagesArray, waitingTruck, waitingUAV, oldWaitingArray, landsat, launchesfrom, ls_checkt, ls_hatt, ls_checktprime, finTauprimeE, finTauprimeF, finSpeedE, finSpeedF, fineee)# 字典: {节点k: 原始卡车到达时间} (用于局部搜索) # 字典: {节点k: 原始卡车离开时间} (用于局部搜索) # 字典: {(v,k): 原始无人机回收时间} (用于局部搜索) # 字典: 最终无人机空载飞行时间
# 字典: 最终无人机带货飞行时间
# 字典: 最终无人机空载速度
# 字典: 最终无人机带货速度
# 字典: 最终无人机任务所需续航
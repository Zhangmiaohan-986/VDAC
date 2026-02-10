#!/usr/bin/env python

import sys
import time
import datetime
import math
from parseCSV import *
# from gurobipy import *
from collections import defaultdict

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

# There's a package color that corresponds to the VEHICLE that delivered the package.
# Right now we only have 5 boxes (so we can have at most 5 trucks).
packageIcons		= ['', 'box_blue_centered.gltf', 'box_orange_centered.gltf', 'box_green_centered.gltf', 'box_gray_centered.gltf', 'box_brown_centered.gltf']
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


def mfstsp_heuristic_2_asgn_uavs(node, eee, eeePrime, N_zero, N_plus, C, V, c, assignments, customersUAV, customersTruck, sigma, sigmaprime, tau, tauprimeE, tauprimeF, vehicle, sL, sR, prevTSPtours):

	#-------------------------------------ALGORITHM 6 (Find the set of feasible sorties for each UAV customer) STARTS HERE--------------------------------------#
	UAVCustSupport = defaultdict(make_dict) # 存储无人机客户的支持起降点
		
	# Get the truck arrival times and build the ordered list of truck visits.
	t = {}	
	myX = []  # 卡车的路径弧集合[i,j]
	myOrderedList = []  # 按时间顺序排列的卡车停靠点（含仓库0）
	myOrderedList.append(0)	# Initialize with depot.
	for vehicleID in assignments:
		for statusID in assignments[vehicleID]:
			for asgnIndex in assignments[vehicleID][statusID]:
				if (assignments[vehicleID][statusID][asgnIndex].vehicleType == TYPE_TRUCK):		
					if (statusID in [TRAVEL_TRUCK_W_UAV, TRAVEL_TRUCK_EMPTY]):
						i = assignments[vehicleID][statusID][asgnIndex].startNodeID
						j = assignments[vehicleID][statusID][asgnIndex].endNodeID
						
						if (j not in customersTruck):
							# When we solve the TSP, we only use a subset of nodes.  
							# This causes the TSP solution to return a final node that's not c+1.
							j = c+1
							bigM = assignments[vehicleID][statusID][asgnIndex].endTime

						myX.append([i,j])
						if i == 0:
							t[i] = assignments[vehicleID][statusID][asgnIndex].startTime
						t[j] = assignments[vehicleID][statusID][asgnIndex].endTime
						
						if (j in C):
							myOrderedList.append(j)	
	myOrderedList.append(c+1)  # 卡车路径终点
	
	# Reset N_zero -- Should only include customers visted by the truck, plus depot 0.	
	# Reset N_plus -- Should only include customers visted by the truck, plus depot c+1.	
	N_zero = []
	N_plus = []
	N_zero.append(0)
	for i in customersTruck:
		N_zero.append(i)
		N_plus.append(i)
	
	N_plus.append(c+1)
	
	# Reset C -- Should only include customers NOT visited by the truck
	C = []
	for i in customersUAV:
		C.append(i)
		
	
	# Find the cost of inserting truck customers:  # 计算将无人机客户点插入卡车路径的代价
	insertCost = {}
	for j in C:
		insertCost[j] = float('inf')
		for [i,k] in myX:
			tmpCost = max(0, (tau[i][j] + sigma[j] + tau[j][k] - tau[i][k]))
			
			if (tmpCost < insertCost[j]):
				insertCost[j] = tmpCost	

	# 在这段代码中，无人机的实时被更改了
	# Calculate the number of skips allowed in total for a given truck route and number of customers:
	totalNumSkipsAllowed = len(V)*(len(myOrderedList) - 1) - len(customersUAV)

	maxSkip = min(3 , totalNumSkipsAllowed) # 每个无人机在每个卡车路径段（相邻节点间）可尝试一次跳过，总跳过次数上限为 无人机数×路径段数。减去已分配的无人机客户数，避免重复服务。maxSkip 限制单次操作最多跳过3次，防止计算爆炸。
	# 跳过节点数
	supportCount = {}
	for i in range(0,maxSkip+1):
		supportCount[i] = {}
		for j in C:
			UAVCustSupport[i][j] = []  # 存储每个无人机客户 j 在跳跃距离 i 下的支持 sortie
			supportCount[i][j] = 0  # 存储每个无人机客户 j 在跳跃距离 i 下的支持 sortie 数量
	# 无人机回收点范围计算
	for v in V:
		for tmpi in range(0,len(myOrderedList)-1):
			i = myOrderedList[tmpi]
			tmpkUpper = min(tmpi+2+maxSkip, len(myOrderedList)) # 计算当前发射点之后允许的最大回收点范围
			for tmpk in range(tmpi+1,tmpkUpper):  # 不能在当前点回收无人机，因此从当前点+1开始,直到最大回收点范围
				k = myOrderedList[tmpk]  # 当前回收点
				if (t[k] - t[i] - sigma[i] > eeePrime[v][i][k]):  # 如果回收点k与发射点i之间的飞行时间减去发射点i的等待时间大于无人机的最大飞行时间，则跳过当前回收点
					break	# exit out of k loop				
				for j in customersUAV:	# 整理每个无人机v可以在节点i发射，在节点k回收的所有可能的客户节点位置j						 
					if (tauprimeF[v][i][j] + node[j].serviceTimeUAV + tauprimeE[v][j][k] <= eee[v][i][j][k]) and (t[k] - t[i] - sigma[i] < eee[v][i][j][k]):					
						UAVCustSupport[tmpk-(tmpi+1)][j].append([v,i,k]) # List of potential sorties for each UAV customer 表示发射点和回收点之间跳过的卡车停靠点数量


	#-------------------------------------ALGORITHM 7 (Assign a sortie to each UAV customer) STARTS HERE--------------------------------------#
	myY = {} # 存储最终为每个客户分配的 sortie (架次)。键可能是 tmpMaxSkip，值是 sortie 列表 [[v,i,j,k], ...]
	bigZ = {} # 存储在此算法迭代中未能成功分配 sortie 的客户。键可能是 tmpMaxSkip，值是客户列表 [j1, j2, ...]

	SortedCust = {} # 存储按“分配难度”（潜在 sortie 数量）排序的无人机客户。键是 tmpMaxSkip，值是排序后的客户列表。

	# 为每个客户 j 计算其拥有的、跳跃距离小于等于 k 的总支持 sortie 数量。
	for i in UAVCustSupport:
		for j in C:
			for k in range(i,maxSkip+1):
				supportCount[k][j] += len(UAVCustSupport[i][j])  # 计算任务累计,找出每个客户j在跳跃距离i下拥有的支持sortie数量

	# Sort UAV customers in the ascending order of number of potential sorties they have: 对于每个 k，将无人机客户 j 按其拥有的、跳跃距离 <= k 的总 sortie 数量进行升序排序。
	for i in supportCount:
		SortedCust[i] = []
		for j in sorted(supportCount[i], key=lambda j: supportCount[i][j]):
			SortedCust[i].append(j)


	# Main loop of the sortie assignment process (in which UAV waiting is preferred) starts here:
	for tmpMaxSkip in SortedCust:# 遍历所有可能的跳跃次数（0到maxSkip）
		# 初始化当前跳跃次数下的任务容器
		myY[tmpMaxSkip] = [] # 存储已分配的无人机任务[v,i,j,k]
		bigZ[tmpMaxSkip] = [] # 存储未分配的无人机任务

		 # 计算允许跳过的总次数（动态调整）
		totalNumSkipsAllowed = len(V)*(len(myOrderedList) - 1) - len(customersUAV)
		# 初始化无人机可用性跟踪字典
		availUAVs = {}
		for i in N_zero:
			availUAVs[i] = list(V)   # 每个卡车节点初始都有全部无人机可用

		# Create UAV sorties for each UAV customer: # 	用于跟踪哪个无人机在卡车路线的哪个段上是空闲的。
		for j in SortedCust[tmpMaxSkip]:  # 根据不同的跳跃程度，选择最好的sories,对于客户节点j

			Waiting = 10*bigM  # 初始化等待时间

			sortie = []  # 存储当前客户j的最佳sortie
			# 尝试不同跳跃次数（0到允许最大值）
			for abc in range(0 , min(tmpMaxSkip,totalNumSkipsAllowed)+1):
				# 遍历当前客户j在跳跃距离abc下的所有支持sortie
				for [v,i,k] in UAVCustSupport[abc][j]:
					tempp = myOrderedList.index(i)
					tempq = myOrderedList.index(k)
					availability = True  # 检查无人机可用性
					for tempindex in range(tempp,tempq):
						if v not in availUAVs[myOrderedList[tempindex]]:
							availability = False
							break # 无人机在该路径段不可用
					 # 计算等待时间（无人机任务时间 - 卡车行程时间差）
					if availability == True:  
						tempWaiting = (tauprimeF[v][i][j] + node[j].serviceTimeUAV + tauprimeE[v][j][k]) - (t[k] - t[i])

						if tempWaiting >= 0:# 无人机需要等待
							if tempWaiting < Waiting: # 找到更小等待
								Waiting = tempWaiting
								sortie = [v,i,j,k]  # 更新最佳任务
						# 卡车需要等待
						else:
							if Waiting >= 0:  # 优先选择卡车等待的方案
								Waiting = tempWaiting
								sortie = [v,i,j,k]
							else:
								if Waiting < tempWaiting:  # 无人机等待时间更短
									Waiting = tempWaiting
									sortie = [v,i,j,k]
			# 上述代码尝试将所有的可能的j放入，在固定的跳跃次数中，从而得到最少等待方案，对每个j，都尝试了所有可能的跳跃次数
			# 如果当前客户j没有找到可行的sortie，将其添加到bigZ列表中
			if len(sortie) == 0:
				if tmpMaxSkip == 0: # 当前跳跃次数为0（最低要求），即走最少的路缺仍没有回收
					bigZ[tmpMaxSkip].append(j) # 将客户j标记为无法分配
				else:
					del myY[tmpMaxSkip] # 删除当前跳跃层的结果容器
					del bigZ[tmpMaxSkip]
					break # 跳出当前客户j的循环
			# 如果找到可行的sortie，将其添加到myY列表中
			else:
				myY[tmpMaxSkip].append(sortie) # 记录当前任务[v,i,j,k],对于每一个客户j，尝试不用的跳跃步数，选择最好的sortie更新
				tempp = myOrderedList.index(sortie[1]) # 发射点i在路径中的位置索引
				tempq = myOrderedList.index(sortie[3]) # 回收点k在路径中的位置索引
				# 更新剩余允许跳跃次数
				totalNumSkipsAllowed = totalNumSkipsAllowed - (tempq-tempp-1)
				# 更新无人机可用性
				for tempindex in range(tempp,tempq):
					availUAVs[myOrderedList[tempindex]].remove(sortie[0])


	# Main loop of the sortie assignment process (in which truck waiting is preferred) starts here:
	for tmpMaxSkip in SortedCust:

		myY[tmpMaxSkip+maxSkip+1] = []
		bigZ[tmpMaxSkip+maxSkip+1] = []

		# Calculate the number of skips allowed in total for a given truck route and number of customers:
		totalNumSkipsAllowed = len(V)*(len(myOrderedList) - 1) - len(customersUAV)

		availUAVs = {}
		for i in N_zero:
			availUAVs[i] = list(V)

		# Create UAV sorties for each UAV customer:
		for j in SortedCust[tmpMaxSkip]:

			Waiting = -10*bigM

			sortie = []

			for abc in range(0 , min(tmpMaxSkip,totalNumSkipsAllowed)+1):

				for [v,i,k] in UAVCustSupport[abc][j]:
					tempp = myOrderedList.index(i)
					tempq = myOrderedList.index(k)
					availability = True
					for tempindex in range(tempp,tempq):
						if v not in availUAVs[myOrderedList[tempindex]]:
							availability = False
							break

					if availability == True:
						tempWaiting = (tauprimeF[v][i][j] + node[j].serviceTimeUAV + tauprimeE[v][j][k]) - (t[k] - t[i])

						if tempWaiting < 0:
							if tempWaiting > Waiting:
								Waiting = tempWaiting
								sortie = [v,i,j,k]

						else:
							if Waiting < 0:
								Waiting = tempWaiting
								sortie = [v,i,j,k]
							else:
								if Waiting > tempWaiting:
									Waiting = tempWaiting
									sortie = [v,i,j,k]

			# If no sortie is found for customer j, append it to the list bigZ:
			if len(sortie) == 0:
				if tmpMaxSkip == 0:
					bigZ[tmpMaxSkip+maxSkip+1].append(j)
				else:
					del myY[tmpMaxSkip+maxSkip+1]
					del bigZ[tmpMaxSkip+maxSkip+1]
					break

			else:
				myY[tmpMaxSkip+maxSkip+1].append(sortie)
				tempp = myOrderedList.index(sortie[1])
				tempq = myOrderedList.index(sortie[3])

				totalNumSkipsAllowed = totalNumSkipsAllowed - (tempq-tempp-1)

				for tempindex in range(tempp,tempq):
					availUAVs[myOrderedList[tempindex]].remove(sortie[0])

	# 上述分别考虑了不同步数中车辆等待无人机及无人机等待车辆的情况.
	# Only pass unique UAV tours:
	UniqueMyY = []  # 存储去重后的可行任务分配方案
	UniqueBigZ = []   # 存储对应的未分配客户列表

	for i in myY:  # 遍历所有可能的跳跃配置
		if myY[i] not in UniqueMyY:  # 检查当前配置是否已存在
			UniqueMyY.append(list(myY[i]))
			UniqueBigZ.append(list(bigZ[i]))


	#------------------------ALGORITHM 8 (Find a UAV customer to insert into truck tour if no feasible sortie assignments found) STARTS HERE-----------------------#

	# Reset bigZ 合并所有未分配客户到bigZ列表
	bigZ = []
	for i in UniqueBigZ:
		for j in i:
			if j not in bigZ:
				bigZ.append(j)
		
	# If there are infeasible customers (len(bigZ) > 0),				
	# we're only going to return only one bigZ customer with the lowest insertCost # 初始化结果存储列表
	insertTuple = []  # 存储插入操作参数，格式为字典{'j':客户, 'i':前节点, 'k':后节点}
	myInsertCost = []  # 存储每个插入操作的成本
	myZ = [] # 存储插入后的未分配客户列表
	# 处理无未分配客户的情况
	if (len(bigZ) == 0):	# No infeasible customers
		for i in range(0,len(UniqueMyY)):
			myInsertCost.append(0)
			insertTuple.append({})
			myZ.append([])

	else:
		cInsert = defaultdict(make_dict) # 插入成本，格式cInsert[客户i][位置p]
		cWait = defaultdict(make_dict)	 # 等待成本，格式cWait[客户i][位置p][客户j]
		cFail = defaultdict(make_dict)  # 失败成本，格式cFail[客户i][位置p][客户j]

		for i in C:
			for p in range(1, len(myOrderedList)):  # 遍历所有可能的插入位置（从第1个位置到倒数第2个位置）
				# Create a TSP route by inserting UAV customer i into the input TSP tour:
				tmpRoute = myOrderedList[0:p] + [i] + myOrderedList[p:len(myOrderedList)]  # 构建临时路径：在位置p插入客户i
				
				# Is this TSP route unique?  # 检查路径是否唯一
				if (tmpRoute not in prevTSPtours):
					 # 计算插入成本（路径长度变化）
					cInsert[i][p] = tau[myOrderedList[p-1]][i] + tau[i][myOrderedList[p]] - tau[myOrderedList[p-1]][myOrderedList[p]]
					# 处理每个未分配客户j
					for j in bigZ:
						isFeas = False
						tmp = float('inf')
						
						if (j == i): # 插入自身的情况
							# We're just going to insert j into 
							isFeas = True
							tmp = 0.0
							cWait[i][p][j] = 0.0
							cFail[i][p][j] = 0.0
						else:
							# Find the maximum possible endurance for this v,i,j combination: # 前向检查：从插入点i出发，检查无人机v在i,j,k路径上的最大续航
							eeeMax = max(eee[v][i][j].values())
							# 计算从i到路径终点的行驶时间
							# Could we launch from i?
							truckTime = 0.0		# We'll find the total time to travel from i to some changing k
							iprime = i
							prevk = 0
							for pprime in range(p, len(myOrderedList)):
								k = myOrderedList[pprime]
								truckTime += tau[iprime][k]		# Time to travel from iprime to k
								iprime = k
								if (prevk):
									truckTime += sigma[prevk]	# Time to serve intermediate customers
								prevk = k
								 # 检查是否超出续航时间
								if truckTime <= eeeMax:
									if ((tauprimeF[v][i][j] + sigmaprime[j] + tauprimeE[v][j][k] <= eee[v][i][j][k]) and (truckTime <= eee[v][i][j][k])):
										isFeas = True
										
										if (max(0, tauprimeF[v][i][j] + sigmaprime[j] + tauprimeE[v][j][k] - truckTime) < tmp):
											tmp = max(0, tauprimeF[v][i][j] + sigmaprime[j] + tauprimeE[v][j][k] - truckTime)
								else:
									break
							# 验证任务可行性
							# Find the maximum possible endurance for this v,i,j combination:
							eeeMax = 0
							for k in eee[v]:
								if ((k != i) and (k != j) and (node[j].parcelWtLbs <= vehicle[v].capacityLbs)):
									if eeeMax < eee[v][k][j][i]:
										eeeMax = eee[v][k][j][i]

							# Could we launch from k?
							truckTime = 0.0		# We'll find the total time to travel from some changing k to i
							iprime = i
							prevk = 0						
							for pprime in range(p-1, 0-1, -1):
								k = myOrderedList[pprime]
								truckTime += tau[k][iprime]
								iprime = k
								if (prevk):
									truckTime += sigma[prevk]
								prevk = k
								
								if truckTime <= eeeMax:							
									if ((tauprimeF[v][k][j] + sigmaprime[j] + tauprimeE[v][j][i] <= eee[v][k][j][i]) and (truckTime <= eee[v][k][j][i])):
										isFeas = True
										
										if (max(0, tauprimeF[v][k][j] + sigmaprime[j] + tauprimeE[v][j][i] - truckTime) < tmp):
											tmp = max(0, tauprimeF[v][k][j] + sigmaprime[j] + tauprimeE[v][j][i] - truckTime)
								else:
									break
	
							# Update costs
							if (isFeas):
								cWait[i][p][j] = tmp
								cFail[i][p][j] = 0.0
							else:
								cWait[i][p][j] = 0.0
								cFail[i][p][j] = insertCost[j]


		bigZ = None
		for bigZ in UniqueBigZ:
			if len(bigZ) == 0:
				myInsertCost.append(0)
				insertTuple.append({})
				myZ.append([])

			else:
				bestCost = float('inf')
				bestIPcombo = []

				for i in cInsert:
					for p in cInsert[i]:

						if i in bigZ:
							tmpcInsert = cInsert[i][p]
						else:
							tmpcInsert = cInsert[i][p]*1.5

						tmpcWait = 0.0
						tmpcFail = 0.0

						for j in bigZ:
							tmpcWait +=	cWait[i][p][j]
							tmpcFail += cFail[i][p][j]		
		
						if (tmpcInsert + tmpcWait + tmpcFail < bestCost):
							bestCost = tmpcInsert + tmpcWait + tmpcFail
							bestIPcombo = [i, p]			

				# We're going to insert customer j in the truck's route between customers i and k:
				p = bestIPcombo[1]		# Position
				j = bestIPcombo[0]		# Inserted customer
				i = myOrderedList[p-1]	# Customer before j
				k = myOrderedList[p]	# Customer after j
				
				myInsertCost.append(insertCost[j])
				insertTuple.append({'j': j, 'i': i, 'k': k})
				myZ.append([j])
	# 输出结果，包括插入成本、插入位置、可行任务分配方案、未分配客户列表和插入操作参数
	return (myInsertCost, myX, UniqueMyY, myZ, insertTuple)
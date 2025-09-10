#!/usr/bin/env python

import sys
import time
import datetime
import math
from parseCSV import *
from gurobipy import *
from collections import defaultdict

from solve_tsp_callback import *
from checkP2Feasibility import *


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
		self.address 			= address			# Might be None

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

# Function to generate TSP assignments for a given TSP tour:
def generateTSPinfo(myTour, c, C, node, tau, sigma):
	tmpAssignments = defaultdict(make_dict)

	vehicleType = TYPE_TRUCK
	UAVsOnBoard = []				
	startAltMeters = 0.0
	endAltMeters = 0.0

	i = 0					# Start at the depot
	mayEnd = 0
	tmpDepart = 0.0
	icon = 'ub_truck_1.gltf'
	for myIndex in range(1,len(myTour)):
		j = myTour[myIndex]
		# We are traveling from i to j
		# Capture the "traveling" component:
		statusID 	= TRAVEL_TRUCK_EMPTY
		ganttStatus	= GANTT_TRAVEL
		startTime 	= tmpDepart	# When we departed from i
		startNodeID = i
		startLatDeg = node[i].latDeg
		startLonDeg	= node[i].lonDeg
		endTime		= startTime + tau[i][j]	# This is when we arrive at j
		endNodeID	= j
		endLatDeg	= node[j].latDeg
		endLonDeg	= node[j].lonDeg
		if ((i in C) and (j in C)):	
			description 	= 'Driving from Customer %d to Customer %d' % (i,j)						
		elif ((i == 0) and (j in C)):
			description 	= 'Driving from Depot to Customer %d' % (j)
		elif ((i in C) and (j == c+1)):
			description 	= 'Returning to the Depot from Customer %d' % (i)
		elif ((i == 0) and (j == c+1)):
			description 	= 'Truck 1 was not used' 
		else:
			print('WE HAVE A PROBLEM.  What is the proper description?')
			print('\t Quitting Now.')
			exit()

		if (0 in tmpAssignments[1][statusID]):
			statusIndex = len(tmpAssignments[1][statusID])
		else:
			statusIndex = 0
		
		tmpAssignments[1][statusID][statusIndex] = make_assignments(vehicleType, startTime, startNodeID, startLatDeg, startLonDeg, startAltMeters, endTime, endNodeID, endLatDeg, endLonDeg, endAltMeters, icon, description, UAVsOnBoard, ganttStatus)


		# Now, capture the "service" component:
		startTime 		= endTime		# When we arrived at j
		startNodeID 	= j
		startLatDeg 	= node[j].latDeg
		startLonDeg		= node[j].lonDeg
		endTime			= startTime + sigma[j]	# This is when we finish up at j
		endNodeID		= j
		endLatDeg		= node[j].latDeg
		endLonDeg		= node[j].lonDeg
		objVal 			= endTime
		if (j == c+1):
			statusID		= STATIONARY_TRUCK_EMPTY
			ganttStatus		= GANTT_FINISHED
			description		= 'Arrived at the Depot.'
		else:
			statusID		= STATIONARY_TRUCK_EMPTY
			ganttStatus		= GANTT_DELIVER
			description		= 'Dropping off package to Customer %d' % (j)

		if (0 in tmpAssignments[1][statusID]):
			statusIndex = len(tmpAssignments[1][statusID])
		else:
			statusIndex = 0
		
		tmpAssignments[1][statusID][statusIndex] = make_assignments(vehicleType, startTime, startNodeID, startLatDeg, startLonDeg, startAltMeters, endTime, endNodeID, endLatDeg, endLonDeg, endAltMeters, icon, description, UAVsOnBoard, ganttStatus)

		tmpDepart = endTime
		if (j != c+1):
			# Go to the next arc
			i = j

	return (objVal, tmpAssignments, myTour)

# Function to generate TSP assignments and packages for a given TSP tour:
def make_TSP_package(myTour, c, C, node, tau, sigma):
	# We want to return this collection of assignments and packages:
	assignments	= defaultdict(make_dict)
	packages	= defaultdict(make_dict)

	# Build the assignment
	vehicleType = TYPE_TRUCK
	UAVsOnBoard = []
	startAltMeters = 0.0
	endAltMeters = 0.0

	i = 0					# Start at the depot
	mayEnd = 0
	tmpDepart = 0.0
	icon = 'ub_truck_1.gltf'
	for myIndex in range(1,len(myTour)):
		j = myTour[myIndex]
		# We are traveling from i to j
		# Capture the "traveling" component:
		statusID 	= TRAVEL_TRUCK_EMPTY
		ganttStatus	= GANTT_TRAVEL
		startTime 	= tmpDepart	# When we departed from i
		startNodeID = i
		startLatDeg = node[i].latDeg
		startLonDeg	= node[i].lonDeg
		endTime		= startTime + tau[i][j]	# This is when we arrive at j
		endNodeID	= j
		endLatDeg	= node[j].latDeg
		endLonDeg	= node[j].lonDeg
		if ((i in C) and (j in C)):
			description 	= 'Driving from Customer %d to Customer %d' % (i,j)
		elif ((i == 0) and (j in C)):
			description 	= 'Driving from Depot to Customer %d' % (j)
		elif ((i in C) and (j == c+1)):
			description 	= 'Returning to the Depot from Customer %d' % (i)
		elif ((i == 0) and (j == c+1)):
			description 	= 'Truck 1 was not used'
		else:
			print('WE HAVE A PROBLEM.  What is the proper description?')
			print('\t Quitting Now.')
			exit()

		if (0 in assignments[1][statusID]):
			statusIndex = len(assignments[1][statusID])
		else:
			statusIndex = 0

		assignments[1][statusID][statusIndex] = make_assignments(vehicleType, startTime, startNodeID, startLatDeg, startLonDeg, startAltMeters, endTime, endNodeID, endLatDeg, endLonDeg, endAltMeters, icon, description, UAVsOnBoard, ganttStatus)


		# Now, capture the "service" component:
		startTime 		= endTime		# When we arrived at j
		startNodeID 	= j
		startLatDeg 	= node[j].latDeg
		startLonDeg		= node[j].lonDeg
		endTime			= startTime + sigma[j]	# This is when we finish up at j
		endNodeID		= j
		endLatDeg		= node[j].latDeg
		endLonDeg		= node[j].lonDeg
		objVal 			= endTime
		if (j == c+1):
			statusID		= STATIONARY_TRUCK_EMPTY
			ganttStatus		= GANTT_FINISHED
			tmpMin, tmpSec	= divmod(endTime, 60)
			tmpHour, tmpMin = divmod(tmpMin, 60)
			description		= 'Arrived at the Depot.  Total Time = %d:%02d:%02d' % (tmpHour, tmpMin, tmpSec)
			endTime			= -1
		else:
			statusID		= STATIONARY_TRUCK_EMPTY
			ganttStatus		= GANTT_DELIVER
			description		= 'Dropping off package to Customer %d' % (j)
			packageType 	= TYPE_TRUCK
			pkgIcon 		= packageIcons[1]
			packages[j] 	= make_packages(packageType, endLatDeg, endLonDeg, endTime, pkgIcon)

		if (0 in assignments[1][statusID]):
			statusIndex = len(assignments[1][statusID])
		else:
			statusIndex = 0

		assignments[1][statusID][statusIndex] = make_assignments(vehicleType, startTime, startNodeID, startLatDeg, startLonDeg, startAltMeters, endTime, endNodeID, endLatDeg, endLonDeg, endAltMeters, icon, description, UAVsOnBoard, ganttStatus)

		tmpDepart = endTime
		if (j != c+1):
			# Go to the next arc
			i = j

	return (objVal, assignments, packages, myTour)


def getTSP(c, tmpTruckCustomers, node, vehicle, travel):
	# Build nodes:
	newNode = {}
	newNode[0] = make_node(node[0].nodeType, node[0].latDeg, node[0].lonDeg, node[0].altMeters, node[0].parcelWtLbs, node[0].serviceTimeTruck, node[0].serviceTimeUAV, node[0].address)
	for j in tmpTruckCustomers:
		newNode[j] = make_node(node[j].nodeType, node[j].latDeg, node[j].lonDeg, node[j].altMeters, node[j].parcelWtLbs, node[j].serviceTimeTruck, node[j].serviceTimeUAV, node[j].address)	

	# Solve TSP using Gurobi callback:
	[TSPobjVal, TSPassignments, TSPpackages, TSPtour] = solve_tsp_callback(newNode, vehicle, travel)  # 使用Gurobi回调函数求解TSP,返回详细的目标值、分配、包裹和路线

	# Add depot at the end of the tour:
	fixedTSPtour = []
	for i in TSPtour[0:-1]:
		fixedTSPtour.append(i)
	fixedTSPtour.append(c+1)
	
	return (fixedTSPtour, TSPobjVal)  # 返回固定TSP路线和目标值


def getTotalCost(TSPtour, tau, sigma, V, C, sL, sR):
		
	totalCost = 0
	i = 0
	for j in TSPtour[1:len(TSPtour)]:
		totalCost += (tau[i][j] + sigma[j])
		i = j
	v = min(V)	
	for j in C:
		if j not in TSPtour:
			totalCost += (sL[v][j] + sR[v][j])

	return totalCost

def getTruckMoves(V, C, TSPtour, tau, sigma, sL, sR):	
	# Find customers currently served via UAV that could be served via truck.
	v = min(V)
	moreTruck = []
	moreTruckSavings = []
	for j in C:
		# We want to know which UAV customers results in positive savings when added to the list of truck customers.
		if j not in TSPtour:
			for iii in range(0, len(TSPtour)-1):
				i = TSPtour[iii]
				k = TSPtour[iii+1]
				tmpSavings = (sL[v][j] + sR[v][j]) - (tau[i][j] + tau[j][k] + sigma[j] - tau[i][k])
				if (tmpSavings > 0):	
					# print("Cheaper to serve %d via truck (save %f)" % (j, tmpSavings))
					if (j not in moreTruck):
						moreTruck.append(j)
						moreTruckSavings.append(tmpSavings)
							
	return (list(moreTruck), list(moreTruckSavings))


def getUAVmoves(TSPtour, xxxTruckOnly, sL, sR, tau, sigma, V, Pprime):	
	# Find customers currently served via truck that would be cheaper with UAV.
	v = min(V)
	moreUAV = []
	moreUAVsavings = []
	for iii in range(0, len(TSPtour)-2):
		i = TSPtour[iii]
		j = TSPtour[iii+1]
		k = TSPtour[iii+2]
		if (j not in xxxTruckOnly):
			if ([i,k] in Pprime[v][j]):
				tmpSavings = (sigma[j] + tau[i][j] + tau[j][k]) - (sL[v][j] + sR[v][j] + tau[i][k]) 
				if (tmpSavings > 0):
					# print("Cheaper to serve %d via UAV (save %f)" % (j, tmpSavings))
					if (j not in moreUAV):
						moreUAV.append(j)
						moreUAVsavings.append(tmpSavings)
					
	return (list(moreUAV), list(moreUAVsavings))


def mfstsp_heuristic_1_partition(node, vehicle, travel, N, N_zero, N_plus, C, P, tau, tauprimeE, tauprimeF, sigma, sigmaprime, sL, sR, lowerTruckLimit, requireUniqueTSP, prevTSPtours, bestOFV, p1_previousTSP, p1_FEASobjVal):

	V = []  # 存储UAV车辆ID
	for vehicleID in vehicle:
		if (vehicle[vehicleID].vehicleType == TYPE_UAV):
			V.append(vehicleID)
	
	c = len(C)
	
	# Define Cprime 至少能被一个无人机从某个可行架次 P 中服务的客户
	Cprime = []
	
	# Define Pprime 存储无人机 v 服务客户 j 的所有可行起飞点 i 和回收点 k 的配对列表。
	Pprime = defaultdict(make_dict)
	for v in V:
		for j in C:
			Pprime[v][j] = []
			
	for [v,i,j,k] in P:	
		if (j not in Cprime):
			Cprime.append(j)

		Pprime[v][j].append([i,k])

	
	# We want to return these arrays:
	customersUAV = []  # 存储UAV客户
	customersTruck = []  # 存储卡车客户


	# Initial addition/removal of customers to the truck tour purely based on savings:
	if len(p1_previousTSP) == 0:	# This loop is being run for the first time. Therefore, go through the entire initial tour building process.

		#-------------------------------------------------ALGORITHM 2 (Initialization) STARTS HERE---------------------------------------------------------#
		# Create a list of "truck-must" customers:  找出仅仅能被卡车服务的客户
		xxxTruckOnly = []
		for i in C:
			if (i not in Cprime):
				xxxTruckOnly.append(i)

		# Get TSP tour:  获取TSP路线
		[TSPtour, TSPobjVal] = getTSP(c, xxxTruckOnly, node, vehicle, travel)
	
		# Calculate total cost of this tour (TSP + UAV launch & recovery)
		totalCost = getTotalCost(TSPtour, tau, sigma, V, C, sL, sR)		

		# Initialize the variables corresponding to best TSP obtained so far:
		bestCost = totalCost
		currentTSP = list(TSPtour)
		
		#------------------------ALGORITHM 3 (Move customers between truckCust & UAVcust based on a savings metric) STARTS HERE----------------------------#
		# Add and remove customers from the truck route sequentially (20 times):
		for xyz in range(0,20):
			# Are there any cases where it's cheaper to serve via truck?
			[moreTruck, moreTruckSavings] = getTruckMoves(V, C, TSPtour, tau, sigma, sL, sR)
		
			if (len(moreTruck) > 0):
				# Add the customers in moreTruck to the list of truck customers:
				tmpTruckCustomers = list(TSPtour + moreTruck)		# also includes nodes 0 and c+1
				tmpTruckCustomers = list( set(tmpTruckCustomers) - set([0, c+1]) )		# remove 0 and c+1
		
				# Get TSP tour:			
				[TSPtour, TSPobjVal] = getTSP(c, tmpTruckCustomers, node, vehicle, travel)
				
				# Calculate total cost of this tour (TSP + UAV launch & recovery)
				totalCost = getTotalCost(TSPtour, tau, sigma, V, C, sL, sR)		
				
				if (totalCost < bestCost):
					bestCost = totalCost
					currentTSP = list(TSPtour)
			
			# Any candidates for moving back to drone?
			[moreUAV, moreUAVsavings] = getUAVmoves(TSPtour, xxxTruckOnly, sL, sR, tau, sigma, V, Pprime)
			
			if (len(moreUAV) > 0):
				# Add the customers in moreUAV to the list of UAV customers:
				tmpTruckCustomers = list(set(TSPtour) - set(moreUAV))	# also includes nodes 0 and c+1
				tmpTruckCustomers = list( set(tmpTruckCustomers) - set([0, c+1]) )		# remove 0 and c+1

				# Get TSP tour:
				[TSPtour, TSPobjVal] = getTSP(c, tmpTruckCustomers, node, vehicle, travel)
				
				# Calculate total cost of this tour (TSP + UAV launch & recovery)
				totalCost = getTotalCost(TSPtour, tau, sigma, V, C, sL, sR)		

				if (totalCost < bestCost):
					bestCost = totalCost
					currentTSP = list(TSPtour)					

	else:	# The initial tour building process was already performed in the first LTL loop. Therefore, just copy that tour here.
		currentTSP = list(p1_previousTSP)
		bestCost = float(p1_FEASobjVal)


	#---------------------------------ALGORITHM 4 (Check feasibility. Modify customer partitions as necessary.) STARTS HERE----------------------------------#

	isFailed = True		# Assume true until shown otherwise
	
	# Move customers to truck for feasibility, and to satisfy LTL requirements:
	while isFailed:

		failed2reach = []	# list of customers we can't reach right now
		insertCost = defaultdict(make_dict)

		maxSupport = 0
		support = defaultdict(make_dict)
		mostHelpful = None
		
		cheapestCost = float('inf')
		cheapestCostInfo = None

		bestRatio = float('inf')
		bestRatioInfo = None

		# create a list of drone customers (C setminus currentTSP)
		droners = list( set(C) - set(currentTSP) )

		# 1a)  Check for unreachable drone customers,如果无人机无法到达客户j，则将j添加到failed2reach列表中
		for j in droners:
			# Can we serve j?
			canServe = False
			for iii in range(0, len(currentTSP)-1):
				i = currentTSP[iii]
				for kkk in [iii+1]:
					k = currentTSP[kkk]
					for v in V:
						if ([i,k] in Pprime[v][j]):
							canServe = True
							break
			if not canServe:			
				failed2reach.append(j)


		# If there are no unreachable drone customers, check if there are drone customers that may result in Phase 2 infeasibility,
		# or if there are at least lowerTruckLimit number of customers
		if (len(failed2reach) == 0):
			if len(V) >= 1:
				[p3_status, p3_infeas_cust] = checkP2Feasibility(droners, currentTSP, V, Pprime)  # 判断无人机是否满足约束，并整理出无法满足约束的客户
				if (p3_status == 1) and (len(currentTSP) - 2 >= lowerTruckLimit):
					isFailed = False
				else:
					isFailed = True
					if (p3_status == 0):
						failed2reach = list(p3_infeas_cust)
			else:
				if (len(currentTSP) - 2 >= lowerTruckLimit):
					isFailed = False
				else:
					isFailed = True
		else:
			isFailed = True

		# If infeasible, add a drone customer to truck that looks the most promising:
		if (isFailed):
			# If we insert j into the truck route, which customers can now be reached?
			# Also, what is the cost?
			for j in droners:
				for iii in range(0, len(currentTSP)-1):
					i = currentTSP[iii]
					k = currentTSP[iii+1]
	
					v = min(V)
					# negative cost is a savings
					tmpCost = (tau[i][j] + tau[j][k] + sigma[j] - tau[i][k]) - (sL[v][j] + sR[v][j])
					insertCost[j][iii+1] = tmpCost	# insert customer j *after* customer i (position iii+1)	
					if (tmpCost < cheapestCost):
						cheapestCost = tmpCost
						cheapestCostInfo = [j, iii+1, tmpCost]

					# Have we seen this tour before:
					dummyTSP = list(currentTSP)
					dummyTSP.insert(iii+1, j)

					if dummyTSP not in prevTSPtours:  # 如果当前TSP路线未出现过
							
						# Our truck *would* travel q - i - j - k - r	[ but q or r might not exist (iii == 0 or iii == len(currentTSP) - 2) ]
						support[j][iii+1] = []		# [list of unreachable customers that could be reached if we insert j after i]  # 记录因插入客户j后，可以到达的客户
	
						if (j in failed2reach):
							support[j][iii+1].append(j)
						
						for l in list( set(failed2reach) - set([j]) ):
							foundIt = False
							for v in V:
								if ([i,j] in Pprime[v][l]):
									# i - l - j
									support[j][iii+1].append(l)
									foundIt = True
								
								elif ([j,k] in Pprime[v][l]):
									# j - l - k
									support[j][iii+1].append(l)
									foundIt = True
									
								if (foundIt):
									break
	
						if (len(support[j][iii+1]) > maxSupport):
							maxSupport = len(support[j][iii+1])  # 记录因插入客户j后，可以到达的客户数量最多的客户
							mostHelpful = [j, iii+1, len(support[j][iii+1])]  # 记录因插入客户j后，可以到达的客户数量最多的客户
						
						# find best cost/number ratio  # 计算因插入客户j后，可以到达的客户数量最多的客户
						if (len(support[j][iii+1]) > 0):
							if (tmpCost < 0):
								# We're actually saving time.  Multiply by the number of customers reached.  # 如果插入客户j后，可以到达的客户数量大于0，则计算因插入客户j后，可以到达的客户数量最多的客户
								tmpRatio = tmpCost * len(support[j][iii+1])
							else:
								# This is costing us time. Find cost per customer.  # 如果插入客户j后，可以到达的客户数量大于0，则计算因插入客户j后，可以到达的客户数量最多的客户
								tmpRatio = tmpCost / len(support[j][iii+1])
						
							if (tmpRatio < bestRatio):		
								bestRatio = tmpRatio
								bestRatioInfo = [j, iii+1, bestRatio]
					

			# Update the TSP:
			if ( (len(currentTSP) - 2 < lowerTruckLimit) and (len(failed2reach) == 0) ):  #  路径较短且无失败客户 
				# If we only need to get to LTL, choose cheapest insertion.   # 如果只需要满足LTL约束，则选择最便宜的插入
				j = cheapestCostInfo[0]
				tmpTruckCustomers = list(currentTSP + [j])		# includes nodes 0 and c+1
				tmpTruckCustomers = list( set(tmpTruckCustomers) - set([0, c+1]) )		# remove 0 and c+1

				# Get TSP tour:
				[currentTSP, TSPobjVal] = getTSP(c, tmpTruckCustomers, node, vehicle, travel)

			elif ( (len(currentTSP) - 2 >= lowerTruckLimit) and (len(failed2reach) > 0) and (bestRatioInfo != None) ):  # 路径较长且有失败客户，且有最佳比率
				# If we only need to address unreachable customers, choose best ratio
				currentTSP.insert(bestRatioInfo[1], bestRatioInfo[0])
				
			elif (bestRatioInfo != None):  
				# If we need to do both, choose best ratio?
				j = bestRatioInfo[0]
				tmpTruckCustomers = list(currentTSP + [j])		# includes nodes 0 and c+1
				tmpTruckCustomers = list( set(tmpTruckCustomers) - set([0, c+1]) )		# remove 0 and c+1

				# Get TSP tour:
				[currentTSP, TSPobjVal] = getTSP(c, tmpTruckCustomers, node, vehicle, travel)

			else:
				# If we only need to get to LTL, choose cheapest insertion.  
				j = cheapestCostInfo[0]
				tmpTruckCustomers = list(currentTSP + [j])		# includes nodes 0 and c+1
				tmpTruckCustomers = list( set(tmpTruckCustomers) - set([0, c+1]) )		# remove 0 and c+1

				# Get TSP tour:
				[currentTSP, TSPobjVal] = getTSP(c, tmpTruckCustomers, node, vehicle, travel)				
	
			
			# Re-calculate total cost of new TSP (TSP + UAV launch & recovery)
			totalCost = getTotalCost(currentTSP, tau, sigma, V, C, sL, sR)		
			
		# end "isFailed" if loop
	# end "while" loop

	# Saving a copy of the current TSP for the next loop:
	basicPreviousTSP = list(currentTSP)

	# Need to build the TSP solution in order
	customersTruck = list(currentTSP[1:-1])
	customersUAV = list(set(C) - set(currentTSP))
	

	newNode = {}
	newNode[0] = make_node(node[0].nodeType, node[0].latDeg, node[0].lonDeg, node[0].altMeters, node[0].parcelWtLbs, node[0].serviceTimeTruck, node[0].serviceTimeUAV, node[0].address)
	for j in customersTruck:
		newNode[j] = make_node(node[j].nodeType, node[j].latDeg, node[j].lonDeg, node[j].altMeters, node[j].parcelWtLbs, node[j].serviceTimeTruck, node[j].serviceTimeUAV, node[j].address)

	newNode[c+1] = make_node(node[0].nodeType, node[0].latDeg, node[0].lonDeg, node[0].altMeters, node[0].parcelWtLbs, node[0].serviceTimeTruck, node[0].serviceTimeUAV, node[0].address)	

	[TSPobjVal, TSPassignments, TSPpackages, TSPtour] = make_TSP_package(currentTSP, c, C, newNode, tau, sigma)


	totalCost = getTotalCost(TSPtour, tau, sigma, V, C, sL, sR)


	#-------------------------------------------------ALGORITHM 5 (Ensure unique TSP solution) STARTS HERE---------------------------------------------------------#

	# Have we seen this tour before?
	if (TSPtour not in prevTSPtours):					
		# This is a new TSP tour
		prevTSPtours.append(TSPtour)
		foundTSP = True	
		
	elif ((requireUniqueTSP) and (TSPtour in prevTSPtours)):
		# Need to create a different TSP tour

		# Create a different tour using one of the following three options:
		# (i) Swap a truck customer and a UAV customer
		# (ii) Perform a subtour reversal
		# (iii) Reverse the entire TSP tour
		
		foundTSP = False
		action = None
		
		if (not foundTSP):

			# Truck/UAV swap?
			newTruckCust = None  # 要加入卡车路径的客户ID
			newUAVcust = None # 要移出卡车路径转为无人机服务的客户ID
			minCost = float('inf') # 记录最小成本变化
			for tmpIndex in range(0,len(TSPtour)-3):  # 遍历当前TSP路径中可能的插入位置（排除最后两个节点）
				i = TSPtour[tmpIndex]  # 获取当前连续三个节点i -> j -> k
				j = TSPtour[tmpIndex+1]
				k = TSPtour[tmpIndex+2]
				foundEligible = False  # 检查j节点是否可由无人机服务（载重是否达标）
				for v in V:
					if node[j].parcelWtLbs <= vehicle[v].capacityLbs:  # 如果j节点的载重小于无人机v的载重
						foundEligible = True
						break
				if (foundEligible):  # 如果j节点可由无人机服务,尝试用无人机客户l替换j
					for l in customersUAV:  # customersUAV是当前无人机客户列表
						# If we insert customer l into TSP tour, have we already seen this tour?
						tmpTSPtour = list(TSPtour)  # 创建当前TSP路径的副本
						tmpTSPtour[tmpIndex + 1] = l  # 将l插入到当前位置,尝试用l来替换j
						if (tmpTSPtour not in prevTSPtours): 	
							tmpCost = (tau[i][l] + tau[l][k] - (tau[i][j] + tau[j][k]))
							if (tmpCost < minCost):

								# Do the following check (P2 feasiblity) when it is worth doing it (meaning if all the previous checks are satisfied):
								failed2reach = []
								# create a list of drone customers (C setminus currentTSP)
								droners = list( set(C) - set(tmpTSPtour) )

								# 1a)  Check for unreachable drone customers
								for temp_j in droners:
									# Can we serve j?
									canServe = False
									for iii in range(0, len(tmpTSPtour)-1):
										temp_i = tmpTSPtour[iii]
										for kkk in [iii+1]:
											temp_k = tmpTSPtour[kkk]
											for temp_v in V:
												if ([temp_i,temp_k] in Pprime[temp_v][temp_j]):
													canServe = True
													break
									if not canServe:			
										failed2reach.append(temp_j)	

								if (len(failed2reach) == 0):
									if len(V) >= 1:
										[p3_status, p3_infeas_cust] = checkP2Feasibility(droners, tmpTSPtour, V, Pprime)
										if (p3_status == 1) and (len(tmpTSPtour) - 2 >= lowerTruckLimit):

											minCost = tmpCost
											newTruckCust = l
											newUAVcust = j
											insertionIndex = tmpIndex + 1
											action = 'truckuavswap'
									else:
										if (len(tmpTSPtour) - 2 >= lowerTruckLimit):

											minCost = tmpCost
											newTruckCust = l
											newUAVcust = j
											insertionIndex = tmpIndex + 1
											action = 'truckuavswap'										
								
			# Subtour reversal?
			bestTour = []  # 存储当前找到的最优路径
			for tmpIndex in range(1,len(TSPtour)-2):  # 遍历当前TSP路径中可能的插入位置（排除最后两个节点）
				i = TSPtour[tmpIndex-1]  # 获取当前连续三个节点i -> j -> k
				j = TSPtour[tmpIndex]
				k = TSPtour[tmpIndex+1]
				l = TSPtour[tmpIndex+2]

				tmpTSPtour = list(TSPtour)  # 创建当前TSP路径的副本
				tmpTSPtour[tmpIndex] = k  # 将k插入到当前位置,尝试用k来替换j
				tmpTSPtour[tmpIndex+1] = j  # 将j插入到当前位置,尝试用j来替换k
				if (tmpTSPtour not in prevTSPtours):  # 如果当前TSP路径未出现过
					tmpCost = tau[i][k] + tau[k][j] + tau[j][l] - tau[i][j] - tau[j][k] - tau[k][l]
					
					if (tmpCost < minCost):  # 如果当前TSP路径的成本变化小于最小成本

						# Do the following check (P2 feasiblity) when it is worth doing it (meaning if all the previous checks are satisfied):
						failed2reach = []   # 检查每个无人机客户
						# create a list of drone customers (C setminus currentTSP)
						droners = list( set(C) - set(tmpTSPtour) )

						# 1a)  Check for unreachable drone customers
						for temp_j in droners:
							# Can we serve j?
							canServe = False
							for iii in range(0, len(tmpTSPtour)-1):
								temp_i = tmpTSPtour[iii]
								for kkk in [iii+1]:
									temp_k = tmpTSPtour[kkk]
									for temp_v in V:
										if ([temp_i,temp_k] in Pprime[temp_v][temp_j]):
											canServe = True
											break
							if not canServe:			
								failed2reach.append(temp_j)	  # 如果无人机客户j无法被服务，则将其添加到failed2reach列表中

						if (len(failed2reach) == 0):  # 如果无人机客户j可以被服务
							if len(V) >= 1:
								[p3_status, p3_infeas_cust] = checkP2Feasibility(droners, tmpTSPtour, V, Pprime)
								if (p3_status == 1) and (len(tmpTSPtour) - 2 >= lowerTruckLimit):

									bestTour = list(tmpTSPtour)  # 更新最优路径
									minCost = tmpCost  # 更新最小成本
									action = 'subtour'  # 更新操作类型
							else:
								if (len(tmpTSPtour) - 2 >= lowerTruckLimit):  # 如果当前TSP路径满足LTL约束

									bestTour = list(tmpTSPtour)  # 更新最优路径
									minCost = tmpCost  # 更新最小成本
									action = 'subtour'  # 更新操作类型															
								
			# Reverse entire TSP tour?
			tmpTSPtour = list(reversed(TSPtour))
			tmpTSPtour[0] = 0
			tmpTSPtour[-1] = c+1
			if (tmpTSPtour not in prevTSPtours):

				costOld = TSPobjVal
				[costNew, JUNKassignments, JUNKtour] = generateTSPinfo(tmpTSPtour, c, C, node, tau, sigma)	
				
				if (costNew - costOld < minCost):

					# Do the following check (P2 feasiblity) when it is worth doing it (meaning if all the previous checks are satisfied):
					failed2reach = []
					# create a list of drone customers (C setminus currentTSP)
					droners = list( set(C) - set(tmpTSPtour) )

					# 1a)  Check for unreachable drone customers
					for temp_j in droners:
						# Can we serve j?
						canServe = False
						for iii in range(0, len(tmpTSPtour)-1):
							temp_i = tmpTSPtour[iii]
							for kkk in [iii+1]:
								temp_k = tmpTSPtour[kkk]
								for temp_v in V:
									if ([temp_i,temp_k] in Pprime[temp_v][temp_j]):
										canServe = True
										break
						if not canServe:			
							failed2reach.append(temp_j)	

					if (len(failed2reach) == 0):
						if len(V) >= 1:
							[p3_status, p3_infeas_cust] = checkP2Feasibility(droners, tmpTSPtour, V, Pprime)
							if (p3_status == 1) and (len(tmpTSPtour) - 2 >= lowerTruckLimit):
						
								entireTSPtour = list(tmpTSPtour)
								minCost = costNew - costOld
								action = 'entire'
						else:
							if (len(tmpTSPtour) - 2 >= lowerTruckLimit):
						
								entireTSPtour = list(tmpTSPtour)
								minCost = costNew - costOld
								action = 'entire'							

			# 客户列表更新：将客户从卡车移动到无人机，反向亦然
			if (action == 'truckuavswap'):
				foundTSP = True
				customersTruck.remove(newUAVcust)  # 从卡车客户中移除某客户
				customersTruck.append(newTruckCust) # 向卡车客户添加新客户
				customersUAV.remove(newTruckCust)  # 从无人机客户中移除某客户
				customersUAV.append(newUAVcust)  # 向无人机客户添加新客户

				tmpTSPtour = list(TSPtour)
				tmpTSPtour[insertionIndex] = newTruckCust  # 在插入点替换客户
				
				prevTSPtours.append(tmpTSPtour)
				[TSPobjVal, TSPassignments, TSPtour] = generateTSPinfo(tmpTSPtour, c, C, node, tau, sigma)					
			
			elif (action == 'subtour'):
				prevTSPtours.append(bestTour)
				[TSPobjVal, TSPassignments, TSPtour] = generateTSPinfo(bestTour, c, C, node, tau, sigma)
					
				if (TSPobjVal < bestOFV):
					foundTSP = True
			
			elif (action == 'entire'):
				prevTSPtours.append(entireTSPtour)
				[TSPobjVal, TSPassignments, TSPtour] = generateTSPinfo(entireTSPtour, c, C, node, tau, sigma)

				if (TSPobjVal < bestOFV):
					foundTSP = True

			else:
				# We couldn't find a new/unique TSP tour
				foundTSP = False
				
	else:
		# We've seen this TSP tour before, but we are not going to modify it
		foundTSP = False

	return (bestCost, customersTruck, customersUAV, TSPobjVal, TSPassignments, TSPpackages, TSPtour, foundTSP, prevTSPtours, basicPreviousTSP)
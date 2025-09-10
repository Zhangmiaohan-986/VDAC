#!/usr/bin/env python

from __future__ import division
from math import *
from collections import defaultdict # 导入 defaultdict 类。它是一种特殊的字典，当访问一个不存在的键时，会自动使用一个默认工厂函数（如 int, list）来创建该键的值。
import os
import numpy as np
import networkx as nx

# =================================================================
radius_of_earth = 6378100.0	# [meters] # [meters] 定义地球半径的常量，单位是米。这个值用于后续计算球面距离。
DIST_TOL = 1.0	# [meters]	If we're within DIST_TOL meters of the target, just assume we're actually at the target.定义一个距离容差常量，单位是米。如果起点和终点的水平距离小于这个值，代码会认为它们在同一水平位置。
ALT_TOL = 1.0 # [meters] 定义一个距离容差常量，单位是米。如果起点和终点的水平距离小于这个值，代码会认为它们在同一水平位置。

MODE_CAR 		= 1 # 定义常量，代表交通方式：汽车
MODE_BIKE 		= 2 # 定义常量，代表交通方式：自行车
MODE_WALK 		= 3 # 定义常量，代表交通方式：步行
MODE_FLY 		= 4 # 定义常量，代表交通方式：飞行


METERS_PER_MILE = 1609.34 # 定义一个常量，表示1英里等于1609.34米。这个值用于将距离从英里转换为米。
# =================================================================


# This file contains a function for calculating UAV travel time.

# 输入参数:
# takeoffSpeed: 起飞速度
# cruiseSpeed: 巡航速度
# landSpeed: 降落速度
# yawRateDeg: 偏航率
# initAlt: 初始高度，初始点高度
# flightAlt: 飞行高度，飞行点高度
# goalAlt: 目标高度，目标点高度
# initLatDeg: 初始点纬度
# initLongDeg: 初始点经度
# goalLatDeg: 目标点纬度
# goalLongDeg: 目标点经度
# initHeadingDeg: 初始点航向
# goalHeadingDeg: 目标点航向

def calcMultirotorTravelTime(takeoffSpeed, cruiseSpeed, landSpeed, yawRateDeg, initAlt, flightAlt, goalAlt, initLatDeg, initLongDeg, goalLatDeg, goalLongDeg, initHeadingDeg, goalHeadingDeg, G_air, G_ground, init_node, goal_node):

	initx = initLatDeg
	inity = initLongDeg
	initz = initAlt
	goalx = goalLatDeg
	goaly = goalLongDeg
	goalz = goalAlt
	
	
	takeoffTime = 0.0 # 起飞时间初始化为0
	flyTime = 0.0 # 飞行时间初始化为0
	landTime = 0.0 # 降落时间初始化为0
	totalTime = 0.0 # 总时间初始化为0
	
	takeoffDistance = 0.0 # 起飞距离初始化为0
	flyDistance = 0.0 # 飞行距离初始化为0
	landDistance = 0.0 # 降落距离初始化为0
	totalDistance = 0.0 # 总距离初始化为0
	totalpower = 0.0 # 总功率初始化为0
	# 检查是否已经接近目标位置
	# distanceToGoal = groundDistanceStraight(initLatRad, initLongRad, goalLatRad, goalLongRad)
	distanceToGoal = AirDistance(initx, inity, initz, goalx, goaly, goalz)
	if (distanceToGoal <= DIST_TOL):
		# 如果距离足够小，则认为初始位置和目标位置相同
		initx = goalx
		inity = goaly
		initz = goalz

	# if (abs(initAlt - goalAlt) <= ALT_TOL):
	# 	# 如果高度足够小，则认为初始位置和目标位置相同
	# 	initAlt = goalAlt 

	# 注意：可能我们的初始位置和目标位置相同。
	# 在这种情况下，我们不需要任何旅行（特别是起飞/降落时间）
	if (([initx, inity, initz] == [goalx, goaly, goalz])):
		totalTime = 0.0
	else:
		# 1) 调整高度（例如，起飞）
		#	 NOTE: 只有当我们的初始/目标坐标不同（否则我们不会准备起飞）
		if ([initx, inity] != [goalx, goaly]):
			init_deltaAlt = initz
			init_node_alt = init_node.map_position[2]
			init_deltaAlt = init_deltaAlt - init_node_alt
			totalTime += (abs(init_deltaAlt)/takeoffSpeed)
			takeoffTime += (abs(init_deltaAlt)/takeoffSpeed)
			takeoffDistance += abs(init_deltaAlt)
			totalDistance += abs(init_deltaAlt)
	
		# 3) 飞向目标,通过所得到的空中G_air(networdx)结构获得无向图距离i-j
		# myDistance = groundDistanceStraight(initLatRad, initLongRad, goalLatRad, goalLongRad)
		init_key = init_node.index  # 获得初始节点索引
		goal_key = goal_node.index  # 获得目标节点索引
		myDistance = nx.shortest_path_length(G_air, source=init_key, target=goal_key, weight='weight')  # 获得初始节点到目标节点的最短距离
		myPath = nx.shortest_path(G_air, source=init_key, target=goal_key, weight='weight')  # 得到类似list的节点序列[16,46,17]

		totalTime += myDistance/cruiseSpeed
		flyTime += myDistance/cruiseSpeed
		flyDistance += myDistance
		totalDistance += myDistance
		
		# # 4) Rotate at target to desired heading (if applicable)
		# if (goalHeadingDeg <= -361):
		# 	# 我们不关心目标航向
		# 	totalTime += 0.0
		# 	landTime += 0.0
			
		# else:
		# 	# 从当前位置到目标位置需要旋转多少角度？
		# 	#	 See http://www.movable-type.co.uk/scripts/latlong.html
		# 	y = sin(goalLongRad - initLongRad) * cos(goalLatRad)
		# 	x = cos(initLatRad)*sin(goalLatRad) - sin(initLatRad)*cos(goalLatRad)*cos(goalLongRad-initLongRad)
		# 	headingStrRad = (atan2(y, x) + 2*pi) % (2*pi)	# In the range [0,2*pi]

		# 	# 当前航向和直线航向之间的差异是多少？
		# 	# 以下公式告诉我们如果顺时针旋转（在弧度中）需要旋转多少：
		# 	deltaHeadingRad = (headingStrRad - goalHeadingRad) % (2*pi)

		# 	# 我们应该顺时针还是逆时针旋转？
		# 	if (deltaHeadingRad <= pi):
		# 		# 最短的旋转将是逆时针。
		# 		rotateCCW = 1		# Set to true
		# 	else:
		# 		# 我们更喜欢顺时针旋转。
		# 		rotateCCW = 0		# Set to false
		# 		# 当顺时针旋转时，实际需要的旋转角度小于pi弧度：
		# 		deltaHeadingRad = 2*pi - deltaHeadingRad
	
		# 	# 旋转deltaHeadingRad弧度
		# 	totalTime += deltaHeadingRad/yawRateRad
		# 	landTime += deltaHeadingRad/yawRateRad
	
		# 5) 调整高度（例如，降落）
		# 我们还没有到达目的地，所以我们需要从飞行高度变为目标高度
		goal_node_alt = goal_node.map_position[2]
		goal_deltaAlt = goalz - goal_node_alt
		totalTime += (abs(goal_deltaAlt)/landSpeed)
		landTime += (abs(goal_deltaAlt)/landSpeed)
		landDistance += abs(goal_deltaAlt)
		totalDistance += abs(goal_deltaAlt)
		# 将地面基站节点也插入到路径中
		myPath.insert(0, init_node.map_key)
		myPath.append(goal_node.map_key)
		# if ([initLatRad, initLongRad] != [goalLatRad, goalLongRad]):
		# 	# 我们还没有到达目的地，所以我们需要从飞行高度变为目标高度
		# 	deltaAlt = flightAlt - goalAlt
		# 	totalTime += (abs(deltaAlt)/landSpeed)
		# 	landTime += (abs(deltaAlt)/landSpeed)
		# 	landDistance += abs(deltaAlt)
		# 	totalDistance += abs(deltaAlt)
		# else:
		# 	# 我们已经到达目的地，我们可能已经开始降落
		# 	# initAlt 描述了“当前”高度
		# 	deltaAlt = initAlt - goalAlt
		# 	totalTime += (abs(deltaAlt)/landSpeed)
		# 	landTime += (abs(deltaAlt)/landSpeed)
		# 	landDistance += abs(deltaAlt)
		# 	totalDistance += abs(deltaAlt)

	# 返回结果 起飞时间，飞行时间，降落时间，总时间，起飞距离，飞行距离，降落距离，总距离
	return [takeoffTime, flyTime, landTime, totalTime, takeoffDistance, flyDistance, landDistance, totalDistance, myPath]

	
	
def groundDistanceStraight(lat1, long1, lat2, long2):
	# 计算从点1到点2的距离，单位为[米]：
	# 这是一个直线距离，忽略高度变化和转弯

	# 注意事项:
	#	* 纬度/经度值的单位为 [弧度]
	# Haversine公式计算两个点之间的距离
	distance = 2*radius_of_earth*asin( sqrt( pow(sin((lat2 - lat1)/2),2) + cos(lat1)*cos(lat2)*pow(sin((long2-long1)/2),2) ))

	return (distance)

def AirDistance(initx, inity, initz, goalx, goaly, goalz):
	distance = np.linalg.norm(np.array([initx, inity, initz]) - np.array([goalx, goaly, goalz]))
	return distance

def GroundDistance(initx, inity, goalx, goaly):
	distance = np.linalg.norm(np.array([initx, inity]) - np.array([goalx, goaly]))
	return distance

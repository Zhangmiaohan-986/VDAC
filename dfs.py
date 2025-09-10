from collections import defaultdict
import random


def dfs(V, delta):
	#Main Code:
	color = {}  # 记录节点访问状态（WHITE=未访问，GREY=正在访问，BLACK=已访问）。
	parent = {}  # 记录每个节点的父节点。
	discovered = {} 
	finished = {}

	for u in V:
		color[u] = 'WHITE'
		parent[u] = None

	time = 0

	Q = []
	for i in V:
		Q.append(i)

	counter = 0

	component = {}

	def DFS_VISIT(x):
		#time += 1
		#discovered[x] = time
		color[x] = 'GREY'

		for y in delta[x]:
			if color[y] == 'WHITE':
				parent[y] = x
				DFS_VISIT(y)

		color[x] = 'BLACK'
		#time += 1
		#finished[x] = time
		Q.remove(x)
		component[counter].append(x)

	while len(Q) != 0:
		u = random.choice(Q)
		counter += 1
		component[counter] = []

		DFS_VISIT(u)

	return component
'''
Created on April 2, 2014

@author: Jacob Conaway
@author: David Jones 
'''

import csv

def parseCSVstring(filename, returnJagged=False, fillerValue=-1, delimiter=',', commentChar='%'):
	with open(filename, "U") as csvfile: 
		csvFile = csvfile.readlines()
		matrix = []
		maxSize = 0

		for line in csvFile:
			if(line.startswith(commentChar)): # 检查是否为注释行
				# Got to next line
				continue
			# line.rstrip() 删除行尾的空白字符（包括换行符）
			# line.rstrip().split(delimiter) 将行按指定分隔符分割成列表
			# filter(None, line.rstrip().split(delimiter)) 过滤掉空值
			# map(str, filter(None, line.rstrip().split(delimiter))) 将列表中的每个元素转换为字符串
			row = list(map(str, filter(None, line.rstrip().split(delimiter)))) # 将行中的每个元素转换为字符串，并过滤掉空值	
			matrix.append(row)
			if (len(row) > maxSize):
				maxSize = len(row)
		# 如果矩阵不是锯齿状的，则将矩阵中的每一行填充到最大长度
		# 如果矩阵是锯齿状的，则不进行填充
		if(not(returnJagged)):
			for row in matrix:    
				row += [fillerValue] * (maxSize - len(row))

		#if (len(matrix) == 1):
			# This is a vector, just return a 1-D vector
			#matrix = matrix[0]			

		return matrix
        
# 打印矩阵
# 遍历矩阵的每一行，然后遍历每一行的每个元素，并打印出来
def printMatrix(matrix):
	for row in matrix:
		for cell in row:
			print(cell), # 打印每个元素，不换行
		print() # 换行

# 测试
if __name__ == "__main__":
	matrix = parseCSVstring("data/test.csv")
	printMatrix(matrix)


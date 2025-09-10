'''
Created on April 2, 2014

@author: Jacob Conaway
@author: David Jones 
'''

import csv

def parseCSV(filename, returnJagged=False, fillerValue=-1, delimiter=',', commentChar='%'):
	with open(filename, "r", encoding='utf-8') as csvfile:
		csvFile = csvfile.readlines()
		matrix = []
		maxSize = 0

		for line in csvFile:
			if(line.startswith(commentChar)):
				# Got to next line
				continue

			row = list(map(float, filter(None, line.rstrip().split(delimiter))))
			matrix.append(row)
			if (len(row) > maxSize):
				maxSize = len(row)

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
	matrix = parseCSV("data/test.csv")
	printMatrix(matrix)
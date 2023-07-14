import numpy as np
from scipy.signal import convolve

# 假设存在一个三维矩阵 matrix，其中包含原始的数据
matrix = np.random.randint(0, 10, size=(5, 5, 3))

# 定义卷积核，用于计算均值
kernel = np.ones((3, 1, 3)) / 9
print(kernel)

# 使用卷积操作计算新的三维矩阵
new_matrix = convolve(matrix, kernel, mode='same')

print("原始矩阵：")
print(matrix)
print("更新后的矩阵：")
print(new_matrix)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 假设存在一个二维矩阵 ReflectMatrix，表示反射率的矩阵
ReflectMatrix = np.random.rand(10, 10)  # 替换为实际的 ReflectMatrix 数据

# 绘制伪彩图
sns.heatmap(ReflectMatrix)

# 添加标题和标签
plt.title('Pseudocolor Plot of ReflectMatrix')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示伪彩图
plt.show()
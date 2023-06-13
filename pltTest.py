import matplotlib.pyplot as plt
import numpy as np

# 打开交互模式
plt.ion()

# 创建初始图形
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

# 动态更新图形
for i in range(100):
    # 更新数据
    y = np.sin(x + i * 0.1)
    
    # 清除当前图形
    plt.clf()
    
    # 绘制新图形
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # 暂停一小段时间
    plt.pause(0.1)

# 关闭交互模式
plt.ioff()

# 显示最终图形
plt.show()
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建Matplotlib的Figure和Axes对象
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)

        # 创建一个绘制图形的画布
        self.canvas = FigureCanvas(self.figure)

        # 将画布放置在主窗口的中央
        self.setCentralWidget(self.canvas)

        # 连接鼠标移动事件
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # 设置窗口标题和大小
        self.setWindowTitle("Mouse Hover Example")
        self.setGeometry(200, 200, 600, 400)

        # 绘制曲线
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.axes.plot(x, y)

        # 显示窗口
        self.show()

    def on_mouse_move(self, event):
        if event.inaxes == self.axes:
            # 获取鼠标在Axes上的坐标
            x = event.xdata

            # 清除之前的标注
            self.axes.lines[1].remove() if len(self.axes.lines) > 1 else None

            # 找到最接近的点
            xdata = self.axes.lines[0].get_xdata()
            ydata = self.axes.lines[0].get_ydata()
            index = np.argmin(np.abs(xdata - x))
            closest_x = xdata[index]
            closest_y = ydata[index]

            # 在最接近的点上添加标注
            self.axes.plot(closest_x, closest_y, 'ro')

            # 刷新画布
            self.canvas.draw()

            # 在状态栏显示点的信息
            self.statusBar().showMessage(f"点: ({closest_x:.2f}, {closest_y:.2f})")
        else:
            self.statusBar().clearMessage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())

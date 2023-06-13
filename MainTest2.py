import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和初始大小
        self.setWindowTitle("激活十字光标")
        self.setGeometry(100, 100, 500, 500)

        # 初始化鼠标位置
        self.mouse_x = 0
        self.mouse_y = 0

        # 初始化标志位
        self.show_crosshair = False

        # 创建一个按钮来激活/关闭十字光标
        self.button = QPushButton("激活十字光标", self)
        self.button.clicked.connect(self.toggle_crosshair)
        self.button.setGeometry(10, 10, 150, 30)

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.show_crosshair:
            # 绘制十字光标
            pen = QPen(Qt.red, 1, Qt.SolidLine)
            painter.setPen(pen)

            # 绘制竖线
            painter.drawLine(self.mouse_x, 0, self.mouse_x, self.height())

            # 绘制横线
            painter.drawLine(0, self.mouse_y, self.width(), self.mouse_y)

            # 绘制空心点
            radius = 5
            painter.drawEllipse(self.mouse_x - radius, self.mouse_y - radius, radius * 2, radius * 2)
            
    def mouseMoveEvent(self, event):
        # 更新鼠标位置
        self.mouse_x = event.x()
        self.mouse_y = event.y()

        # 重新绘制窗口
        self.update()

    def toggle_crosshair(self):
        # 切换标志位的状态
        self.show_crosshair = not self.show_crosshair

        if self.show_crosshair:
            self.button.setText("关闭十字光标")
        else:
            self.button.setText("激活十字光标")

        # 清空窗口内容
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
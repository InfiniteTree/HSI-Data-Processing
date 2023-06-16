from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPen
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsPathItem


class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # 开启鼠标追踪
        self.cursor_pos = QPointF(0, 0)  # 光标位置

        # 创建十字光标路径项
        self.crosshair_item = QGraphicsPathItem()
        self.crosshair_item.setPen(QPen(Qt.red))
        self.scene().addItem(self.crosshair_item)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateCrosshair()

    def mouseMoveEvent(self, event):
        self.cursor_pos = event.pos()
        self.updateCrosshair()

    def updateCrosshair(self):
        view_width = self.viewport().width()
        view_height = self.viewport().height()

        path = QPainterPath()
        # 绘制竖线
        path.moveTo(self.cursor_pos.x(), 0)
        path.lineTo(self.cursor_pos.x(), view_height)
        # 绘制横线
        path.moveTo(0, self.cursor_pos.y())
        path.lineTo(view_width, self.cursor_pos.y())

        self.crosshair_item.setPath(path)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    view = CustomGraphicsView()
    scene = QGraphicsScene()
    view.setScene(scene)
    scene.addEllipse(50, 50, 200, 200)  # 在场景中添加一个椭圆，用于测试

    view.show()
    sys.exit(app.exec_())
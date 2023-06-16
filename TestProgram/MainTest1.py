from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QPushButton, QGraphicsRectItem
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor

class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.selection_rect = None
        self.selecting = False

    def startSelection(self):
        self.selecting = True
        self.selection_rect = QGraphicsRectItem()
        self.selection_rect.setPen(Qt.blue)
        self.scene().addItem(self.selection_rect)

    def stopSelection(self):
        if self.selection_rect is not None:
            selected_items = self.scene().items(self.selection_rect.rect(), Qt.IntersectsItemShape)
            # 执行框选后的操作，例如获取被选中的项

            # 打印选择框的 x 和 y 坐标
            rect = self.selection_rect.rect()
            x = rect.x()
            y = rect.y()
            print("x:", x)
            print("y:", y)

            self.scene().removeItem(self.selection_rect)
            self.selection_rect = None
        self.selecting = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.selecting:
            pos_in_view = event.pos()
            pos_in_scene = self.mapToScene(pos_in_view)
            self.selection_rect.setRect(QRectF(pos_in_scene, pos_in_scene))
            self.scene().addItem(self.selection_rect)

    def mouseMoveEvent(self, event):
        if self.selecting and self.selection_rect is not None:
            pos_in_view = event.pos()
            pos_in_scene = self.mapToScene(pos_in_view)
            rect = QRectF(self.selection_rect.rect().topLeft(), pos_in_scene)
            self.selection_rect.setRect(rect.normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.selecting:
            self.stopSelection()


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main")
        self.setGeometry(100, 100, 500, 500)

        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.button = QPushButton("Start Selection", self)
        self.button.setGeometry(10, 10, 120, 30)
        self.button.clicked.connect(self.startSelection)

    def startSelection(self):
        self.view.startSelection()


if __name__ == '__main__':
    app = QApplication([])
    main = Main()
    main.show()
    app.exec_()
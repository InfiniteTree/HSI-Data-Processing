import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Image Selection")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # 设置要加载的图像路径
        image_path = "path_to_your_image.jpg"
        self.image = QPixmap(image_path)

        self.selection_rect = None
        self.selection_start = None
        self.selection_end = None

        self.image_label.setPixmap(self.image)
        self.setCentralWidget(self.image_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_start = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_end = event.pos()
            self.selection_rect = QRect(self.selection_start, self.selection_end).normalized()
            self.update()

            # 获取选中区域的信息
            if self.selection_rect.isValid():
                selected_region = self.image.copy(self.selection_rect)
                selected_region.save("selected_region.jpg")  # 保存选中的区域

                # 打印选中区域的位置和大小
                print("Selected region:")
                print("X:", self.selection_rect.x())
                print("Y:", self.selection_rect.y())
                print("Width:", self.selection_rect.width())
                print("Height:", self.selection_rect.height())

    def paintEvent(self, event):
        if self.selection_rect is not None:
            painter = QPainter(self)
            painter.setPen(QColor(255, 0, 0))
            painter.drawRect(self.selection_rect)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
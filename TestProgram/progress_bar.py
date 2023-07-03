from PyQt5.QtWidgets import QMainWindow, QProgressBar, QPushButton, QApplication
from PyQt5.QtCore import QTimer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, 200, 25)

        self.start_button = QPushButton('Start', self)
        self.start_button.setGeometry(30, 80, 75, 30)
        self.start_button.clicked.connect(self.start_progress)

    def start_progress(self):
        self.progress_value = 0
        self.progress_bar.setValue(self.progress_value)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # 每100毫秒更新一次进度条

    def update_progress(self):
        self.progress_value += 1
        self.progress_bar.setValue(self.progress_value)

        if self.progress_value >= 100:
            self.timer.stop()

    def closeEvent(self, event):
        # 在窗口关闭时停止定时器
        self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QDialog, QVBoxLayout, QProgressBar, QDialogButtonBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 300, 200)

        button = QPushButton("Open Progress Window")
        button.clicked.connect(self.open_progress_window)

        layout = QVBoxLayout()
        layout.addWidget(button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_progress_window(self):
        progress_window = ProgressWindow(self)
        progress_window.show()

class WorkerThread(QThread):
    progress_updated = pyqtSignal(int)

    def run(self):
        counter = 0
        while counter <= 100:
            self.progress_updated.emit(counter)
            counter += 1
            self.msleep(100)  # 模拟耗时任务

class ProgressWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Progress Window")
        self.setMinimumWidth(400)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)

        button_box = QDialogButtonBox()
        #button_box.setStandardButtons(QDialogButtonBox.Cancel)

        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        layout.addWidget(button_box)

        self.setLayout(layout)

        self.worker_thread = WorkerThread()
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
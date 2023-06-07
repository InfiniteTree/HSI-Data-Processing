from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
import sys

class FileDialogExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('File Dialog Example')

        self.button = QPushButton('Open File', self)
        self.button.clicked.connect(self.showFileDialog)
        self.button.setGeometry(50, 50, 200, 30)

    def showFileDialog(self):
        file_dialog = QFileDialog()
        file_dialog.exec_()
        selected_file = file_dialog.selectedFiles()
        if selected_file:
            print('Selected file:', selected_file[0])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FileDialogExample()
    window.show()
    sys.exit(app.exec_())
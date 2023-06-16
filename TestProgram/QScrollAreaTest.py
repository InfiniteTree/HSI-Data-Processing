from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollArea, QLabel


class ScrollableWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 添加要显示的内容，这里使用 QLabel 作为示例
        for i in range(20):
            label = QLabel(f"Item {i+1}")
            content_layout.addWidget(label)
        
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication([])
    window = ScrollableWidget()
    window.show()
    app.exec_()








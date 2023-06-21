import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit


class FormulaInputWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.returnPressed.connect(self.evaluate_formula)
        layout.addWidget(self.input_edit)

        self.setLayout(layout)

    def evaluate_formula(self):
        formula = self.input_edit.text()
        try:
            result = eval(formula)
            print("Result:", result)
        except Exception as e:
            print("Error:", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = FormulaInputWidget()
    widget.show()
    sys.exit(app.exec_())
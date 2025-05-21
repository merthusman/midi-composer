from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
import sys

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("Test")
window.setGeometry(100, 100, 200, 100)
window.show()
sys.exit(app.exec()) 
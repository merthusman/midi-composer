import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                          QVBoxLayout, QWidget, QLabel, QMessageBox)
from PyQt6.QtCore import Qt

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Uygulamasi")
        self.setGeometry(100, 100, 300, 200)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add a welcome label
        self.label = QLabel("Test Uygulamasina Hosgeldiniz!")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; margin: 10px;")
        layout.addWidget(self.label)
        
        # Add counter label
        self.counter_label = QLabel("Tiklama Sayisi: 0")
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.counter_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.counter_label)
        
        # Add button with better styling
        self.counter = 0
        self.button = QPushButton("Bana Tikla!")
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.button.clicked.connect(self.on_button_click)
        layout.addWidget(self.button)
    
    def on_button_click(self):
        self.counter += 1
        self.counter_label.setText(f"Tiklama Sayisi: {self.counter}")
        
        if self.counter % 5 == 0:
            QMessageBox.information(
                self,
                "Basari!",
                f"Tebrikler! {self.counter} kez tikladiniz!"
            )

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 
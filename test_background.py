#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basit PyQt6 arka plan resmi testi
"""

import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Ana pencere ayarları
        self.setWindowTitle("Arka Plan Testi")
        self.setGeometry(100, 100, 800, 600)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout
        self.layout = QVBoxLayout(self.central_widget)
        
        # Test etiketi
        self.label = QLabel("Bu bir test etiketidir. Arka plan resmi görünüyor mu?")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background-color: rgba(0, 0, 0, 100);")
        self.layout.addWidget(self.label)
        
        # Arka plan resmini doğrudan QSS ile ayarla
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1f1f1f; /* Yedek renk */
                background-image: url("resources/images/background.jpg");
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }
            
            QWidget {
                background-color: transparent;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

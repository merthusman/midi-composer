import sys
import os
import logging
import traceback
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox

# Configure both file and console logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

print("Script starting...")
logging.info("Logging initialized")

class TestWindow(QMainWindow):
    def __init__(self):
        print("Initializing window...")
        try:
            super().__init__()
            print("Window parent initialized")
            
            self.setWindowTitle("Console Test App")
            self.setGeometry(100, 100, 400, 200)
            
            # Print current working directory
            print(f"Current working directory: {os.getcwd()}")
            
            # Add test button
            self.button = QPushButton("Click to Test (Check Console)", self)
            self.button.setGeometry(50, 30, 300, 40)
            self.button.clicked.connect(self.show_test_message)
            
            print("Window setup complete")
            
        except Exception as e:
            print(f"ERROR in window initialization: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            self.show_error_message(str(e))

    def show_test_message(self):
        try:
            print("Button clicked!")
            QMessageBox.information(self, "Test", "Check the console window!")
            print("Test message shown")
        except Exception as e:
            print(f"ERROR showing message: {str(e)}")
            traceback.print_exc()
            self.show_error_message(str(e))

    def show_error_message(self, error):
        print(f"Showing error dialog: {error}")
        QMessageBox.critical(self, "Error", f"An error occurred: {error}")

def main():
    try:
        print("Application starting...")
        
        # Print Python version and path
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        
        app = QApplication(sys.argv)
        print("QApplication created")
        
        window = TestWindow()
        print("Window created")
        
        window.show()
        print("Window shown")
        
        print("Entering main event loop...")
        return app.exec()
        
    except Exception as e:
        print(f"CRITICAL ERROR in main: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    print("Script entry point reached")
    exit_code = main()
    print(f"Application exiting with code: {exit_code}")
    sys.exit(exit_code) 
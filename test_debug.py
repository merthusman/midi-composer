import sys
import logging
import traceback
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox

# Configure logging
logging.basicConfig(
    filename='app_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TestWindow(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            logging.info("Initializing TestWindow")
            
            self.setWindowTitle("Test App")
            self.setGeometry(100, 100, 300, 150)
            
            # Add test button
            self.button = QPushButton("Click to Test", self)
            self.button.setGeometry(50, 30, 200, 40)
            self.button.clicked.connect(self.show_test_message)
            
            logging.info("TestWindow initialized successfully")
            
        except Exception as e:
            logging.error(f"Error in TestWindow initialization: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message(str(e))

    def show_test_message(self):
        try:
            QMessageBox.information(self, "Test", "Application is working!")
            logging.info("Test message shown successfully")
        except Exception as e:
            logging.error(f"Error showing test message: {str(e)}")
            self.show_error_message(str(e))

    def show_error_message(self, error):
        QMessageBox.critical(self, "Error", f"An error occurred: {error}")

def main():
    try:
        logging.info("Starting application")
        app = QApplication(sys.argv)
        
        # Set application details
        app.setApplicationName("TestDebugApp")
        app.setOrganizationName("TestOrg")
        
        window = TestWindow()
        window.show()
        
        logging.info("Application started successfully")
        return app.exec()
        
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
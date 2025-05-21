# src/gui/panels/settings_panel.py
import logging
import json
from src.utils.serialization import object_to_dict, to_json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QFileDialog, QGroupBox, QSizePolicy, QGridLayout,
    QFrame, QSpacerItem, QFormLayout, QLineEdit, QSpinBox, 
    QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, Qt

logger = logging.getLogger(__name__)

class SettingsPanel(QWidget):
    """Panel for application settings functionality."""
    
    # Signals
    settings_saved = pyqtSignal(dict)  # Emitted when settings are saved
    settings_loaded = pyqtSignal()  # Emitted when settings are loaded
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface components."""
        # Set up QSS styles
        self.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                margin-top: 20px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QLabel {
                font-size: 12px;
                color: #333;
                background-color: rgba(255, 255, 255, 0.2);
                padding: 5px 10px;
                border-radius: 5px;
            }
            
            QSpinBox, QDoubleSpinBox {
                font-size: 12px;
                min-width: 120px;
                height: 30px;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            
            QCheckBox {
                font-size: 12px;
                color: #333;
                padding: 5px;
            }
            
            QPushButton {
                font-size: 12px;
                min-width: 120px;
                height: 30px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
            }
            
            QPushButton:hover {
                background-color: #45a049;
            }
            
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Settings Group
        self.settings_group = QGroupBox("Uygulama Ayarları")
        self.settings_group.setObjectName("settings_group")
        self.settings_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        settings_layout = QVBoxLayout(self.settings_group)
        settings_layout.setContentsMargins(15, 20, 15, 15)
        settings_layout.setSpacing(15)
        
        # Training Parameters Group
        training_group = QGroupBox("Model Eğitimi Parametreleri")
        training_group.setObjectName("training_group")
        training_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        training_layout = QGridLayout()
        training_layout.setSpacing(10)
        training_layout.setContentsMargins(15, 15, 15, 15)
        
        # Epochs
        self.epochs_label = QLabel("Epoch Sayısı:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setMinimumWidth(100)
        training_layout.addWidget(self.epochs_label, 0, 0)
        training_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Batch Size
        self.batch_label = QLabel("Batch Boyutu:")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        self.batch_spin.setMinimumWidth(100)
        training_layout.addWidget(self.batch_label, 1, 0)
        training_layout.addWidget(self.batch_spin, 1, 1)
        
        # Learning Rate
        self.lr_label = QLabel("Öğrenme Oranı:")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setMinimumWidth(100)
        training_layout.addWidget(self.lr_label, 2, 0)
        training_layout.addWidget(self.lr_spin, 2, 1)
        
        # Early Stopping
        self.early_stopping_label = QLabel("Early Stopping:")
        self.early_stopping_check = QCheckBox()
        self.early_stopping_check.setChecked(True)
        training_layout.addWidget(self.early_stopping_label, 3, 0)
        training_layout.addWidget(self.early_stopping_check, 3, 1)
        
        # Training Button
        self.train_button = QPushButton("Modeli Eğit")
        self.train_button.setObjectName("train_button")
        self.train_button.setMinimumWidth(150)
        self.train_button.clicked.connect(self.start_training)
        training_layout.addWidget(self.train_button, 4, 0, 1, 2)
        
        training_group.setLayout(training_layout)
        settings_layout.addWidget(training_group)
        
        # Settings display and edit
        self.settings_text = QTextEdit()
        self.settings_text.setObjectName("settings_text")
        self.settings_text.setReadOnly(False)  # Allow editing
        self.settings_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.settings_text.setMinimumHeight(300)
        self.settings_text.setPlaceholderText("Ayarlar JSON formatında görüntülenir ve düzenlenebilir.")
        
        # Buttons for settings operations
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.save_button = QPushButton("Ayarları Kaydet")
        self.save_button.setObjectName("save_settings_button")
        self.save_button.setMinimumWidth(150)
        self.save_button.clicked.connect(self.save_settings)
        
        self.load_button = QPushButton("Ayarları Yeniden Yükle")
        self.load_button.setObjectName("load_settings_button")
        self.load_button.setMinimumWidth(150)
        self.load_button.clicked.connect(self.load_settings)
        
        self.export_button = QPushButton("Ayarları Dışa Aktar")
        self.export_button.setObjectName("export_settings_button")
        self.export_button.setMinimumWidth(150)
        self.export_button.clicked.connect(self.export_settings)
        
        self.import_button = QPushButton("Ayarları İçe Aktar")
        self.import_button.setObjectName("import_settings_button")
        self.import_button.setMinimumWidth(150)
        self.import_button.clicked.connect(self.import_settings)
        
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.import_button)
        
        # Add components to settings layout
        settings_layout.addWidget(training_group)
        settings_layout.addWidget(self.settings_text)
        settings_layout.addLayout(buttons_layout)
        
        # Paths Group
        self.paths_group = QGroupBox("Dosya Yolları")
        self.paths_group.setObjectName("paths_group")
        self.paths_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.paths_group.setMinimumHeight(200)
        
        paths_layout = QFormLayout(self.paths_group)
        paths_layout.setContentsMargins(15, 20, 15, 15)
        paths_layout.setSpacing(10)
        paths_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        paths_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Model directory path
        self.model_dir_label = QLabel("Model Dizini:")
        self.model_dir_label.setObjectName("path_label")
        self.model_dir_label.setMinimumWidth(120)
        
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setObjectName("model_dir_edit")
        self.model_dir_edit.setMinimumWidth(300)
        self.model_dir_edit.setMinimumHeight(30)
        self.model_dir_edit.setReadOnly(True)
        
        self.model_dir_button = QPushButton("Gözat...")
        self.model_dir_button.setObjectName("browse_button")
        self.model_dir_button.clicked.connect(lambda: self.browse_directory("model_dir"))
        
        model_dir_layout = QHBoxLayout()
        model_dir_layout.addWidget(self.model_dir_edit)
        model_dir_layout.addWidget(self.model_dir_button)
        
        # Memory file path
        self.memory_file_label = QLabel("Hafıza Dosyası:")
        self.memory_file_label.setObjectName("path_label")
        self.memory_file_label.setMinimumWidth(120)
        
        self.memory_file_edit = QLineEdit()
        self.memory_file_edit.setObjectName("memory_file_edit")
        self.memory_file_edit.setMinimumWidth(300)
        self.memory_file_edit.setMinimumHeight(30)
        self.memory_file_edit.setReadOnly(True)
        
        self.memory_file_button = QPushButton("Gözat...")
        self.memory_file_button.setObjectName("browse_button")
        self.memory_file_button.clicked.connect(lambda: self.browse_file("memory_file"))
        
        memory_file_layout = QHBoxLayout()
        memory_file_layout.addWidget(self.memory_file_edit)
        memory_file_layout.addWidget(self.memory_file_button)
        
        # Output directory path
        self.output_dir_label = QLabel("Çıktı Dizini:")
        self.output_dir_label.setObjectName("path_label")
        self.output_dir_label.setMinimumWidth(120)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setObjectName("output_dir_edit")
        self.output_dir_edit.setMinimumWidth(300)
        self.output_dir_edit.setMinimumHeight(30)
        self.output_dir_edit.setReadOnly(True)
        
        self.output_dir_button = QPushButton("Gözat...")
        self.output_dir_button.setObjectName("browse_button")
        self.output_dir_button.clicked.connect(lambda: self.browse_directory("output_dir"))
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        
        # Add path fields to paths layout
        paths_layout.addRow(self.model_dir_label, model_dir_layout)
        paths_layout.addRow(self.memory_file_label, memory_file_layout)
        paths_layout.addRow(self.output_dir_label, output_dir_layout)
        
        # Add groups to main layout
        main_layout.addWidget(self.settings_group, 2)  # 2/3 of the space
        main_layout.addWidget(self.paths_group, 1)     # 1/3 of the space
        
        # Set size policies for responsive layout
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
    def update_settings_display(self):
        """Update the settings text display with current settings."""
        try:
            # Gelişmiş serileştirme yardımcısını kullan
            settings_json = to_json(self.settings)
            self.settings_text.setText(settings_json)
            
            # Update path fields
            if 'model_dir_path' in self.settings:
                self.model_dir_edit.setText(self.settings['model_dir_path'])
                
            if 'memory_file_full_path' in self.settings:
                self.memory_file_edit.setText(self.settings['memory_file_full_path'])
                
            if 'output_dir' in self.settings:
                self.output_dir_edit.setText(self.settings['output_dir'])
                
            logger.info("Ayarlar görüntüleme güncellendi")
        except Exception as e:
            logger.error(f"Ayarlar görüntülenirken hata oluştu: {e}")
            self.settings_text.setText(f"Ayarlar görüntülenirken hata oluştu: {e}")
        
    def set_settings(self, settings):
        """Set the settings and update the display."""
        self.settings = settings
        self.update_settings_display()
        
        # Update training parameters
        if 'epochs' in settings:
            self.epochs_spin.setValue(settings['epochs'])
        if 'batch_size' in settings:
            self.batch_spin.setValue(settings['batch_size'])
        if 'learning_rate' in settings:
            self.lr_spin.setValue(settings['learning_rate'])
        if 'early_stopping' in settings:
            self.early_stopping_check.setChecked(settings['early_stopping'])
        
    def save_settings(self):
        """Save the edited settings."""
        try:
            # Parse the JSON from the text edit
            settings_json = self.settings_text.toPlainText()
            new_settings = json.loads(settings_json)
            
            # Update path fields from UI if they were changed
            new_settings['model_dir_path'] = self.model_dir_edit.text()
            new_settings['memory_file_full_path'] = self.memory_file_edit.text()
            new_settings['output_dir'] = self.output_dir_edit.text()
            
            # Update internal settings
            self.settings = new_settings
            
            # Emit signal with new settings
            self.settings_saved.emit(new_settings)
            logger.info("Ayarlar kaydedildi")
        except json.JSONDecodeError as e:
            logger.error(f"Ayarlar JSON formatında değil: {e}")
            # Show error in the text edit
            current_text = self.settings_text.toPlainText()
            self.settings_text.setText(f"HATA: JSON formatında değil - {e}\n\n{current_text}")
        except Exception as e:
            logger.error(f"Ayarlar kaydedilirken hata oluştu: {e}")
            # Show error in the text edit
            current_text = self.settings_text.toPlainText()
            self.settings_text.setText(f"HATA: {e}\n\n{current_text}")
        
    def load_settings(self):
        """Reload the settings from the application."""
        self.settings_loaded.emit()
        logger.info("Ayarlar yeniden yükleme istendi")
        
    def export_settings(self):
        """Export settings to a JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Ayarları Dışa Aktar", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=4, sort_keys=True, ensure_ascii=False)
                logger.info(f"Ayarlar dışa aktarıldı: {file_path}")
            except Exception as e:
                logger.error(f"Ayarlar dışa aktarılırken hata oluştu: {e}")
        
    def import_settings(self):
        """Import settings from a file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Ayarları İçe Aktar", "", "JSON Files (*.json);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.set_settings(settings)
                    self.settings_saved.emit(settings)
        except Exception as e:
            logger.error(f"Ayarlar içe aktarılırken hata oluştu: {e}")
            QMessageBox.critical(self, "Hata", f"Ayarlar içe aktarılırken hata oluştu: {e}")

    def start_training(self):
        """Start model training with current parameters."""
        try:
            # Get training parameters
            epochs = self.epochs_spin.value()
            batch_size = self.batch_spin.value()
            learning_rate = self.lr_spin.value()
            early_stopping = self.early_stopping_check.isChecked()
            
            # Update settings with training parameters
            self.settings['epochs'] = epochs
            self.settings['batch_size'] = batch_size
            self.settings['learning_rate'] = learning_rate
            self.settings['early_stopping'] = early_stopping
            
            # Save settings
            self.save_settings()
            
            # Emit training parameters to the main window
            self.settings_saved.emit({
                'task': 'train_model',
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'early_stopping': early_stopping
            })
            
            # Show progress message
            QMessageBox.information(self, "Bilgi", 
                f"Model eğitimi başlatılıyor...\n\n"
                f"Epoch: {epochs}\n"
                f"Batch Boyutu: {batch_size}\n"
                f"Öğrenme Oranı: {learning_rate}\n"
                f"Early Stopping: {'Evet' if early_stopping else 'Hayır'}")
            
        except Exception as e:
            logger.error(f"Model eğitimi başlatılırken hata oluştu: {e}")
            QMessageBox.critical(self, "Hata", f"Model eğitimi başlatılırken hata oluştu: {e}")
        
    def browse_directory(self, field_name):
        """Open a directory browser dialog for the specified field."""
        directory = QFileDialog.getExistingDirectory(self, f"{field_name.replace('_', ' ').title()} Seç")
        
        if directory:
            if field_name == "model_dir":
                self.model_dir_edit.setText(directory)
                self.settings['model_dir_path'] = directory
            elif field_name == "output_dir":
                self.output_dir_edit.setText(directory)
                self.settings['output_dir'] = directory
                
            # Update the settings display
            self.update_settings_display()
            logger.info(f"{field_name} dizini güncellendi: {directory}")
        
    def browse_file(self, field_name):
        """Open a file browser dialog for the specified field."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"{field_name.replace('_', ' ').title()} Seç", "", "All Files (*)"
        )
        
        if file_path:
            if field_name == "memory_file":
                self.memory_file_edit.setText(file_path)
                self.settings['memory_file_full_path'] = file_path
                
            # Update the settings display
            self.update_settings_display()
            logger.info(f"{field_name} dosyası güncellendi: {file_path}")

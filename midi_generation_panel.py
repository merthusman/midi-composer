# src/gui/panels/midi_generation_panel.py
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
    QGroupBox, QSizePolicy, QGridLayout, QFormLayout, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt

logger = logging.getLogger(__name__)

class MIDIGenerationPanel(QWidget):
    """Panel for MIDI generation functionality."""
    
    # Signals
    generation_requested = pyqtSignal(dict)  # Emitted when generation is requested with parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # MIDI Generation Group
        self.generation_group = QGroupBox("MIDI Üretimi")
        self.generation_group.setObjectName("midi_uretim_group")
        self.generation_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.generation_group.setMinimumHeight(300)
        self.generation_group.setMinimumWidth(500)
        
        generation_layout = QGridLayout(self.generation_group)
        generation_layout.setContentsMargins(15, 20, 15, 15)
        generation_layout.setSpacing(15)  # Increased spacing for better readability
        
        # Create a form layout for parameters
        params_form = QFormLayout()
        params_form.setContentsMargins(0, 0, 0, 0)
        params_form.setSpacing(10)
        params_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        params_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Bar count parameter
        self.bar_count_label = QLabel("Bar Sayısı:")
        self.bar_count_label.setObjectName("param_label")
        self.bar_count_label.setMinimumWidth(120)
        
        self.bar_count_spin = QSpinBox()
        self.bar_count_spin.setObjectName("bar_count_spin")
        self.bar_count_spin.setMinimumWidth(150)
        self.bar_count_spin.setMinimumHeight(30)
        self.bar_count_spin.setRange(1, 32)
        self.bar_count_spin.setValue(4)
        self.bar_count_spin.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        
        # Tempo parameter
        self.tempo_label = QLabel("Tempo (BPM):")
        self.tempo_label.setObjectName("param_label")
        self.tempo_label.setMinimumWidth(120)
        
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setObjectName("tempo_spin")
        self.tempo_spin.setMinimumWidth(150)
        self.tempo_spin.setMinimumHeight(30)
        self.tempo_spin.setRange(60, 200)
        self.tempo_spin.setValue(120)
        self.tempo_spin.setSingleStep(5)
        self.tempo_spin.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        
        # Temperature parameter
        self.temperature_label = QLabel("Yaratıcılık:")
        self.temperature_label.setObjectName("param_label")
        self.temperature_label.setMinimumWidth(120)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setObjectName("temperature_spin")
        self.temperature_spin.setMinimumWidth(150)
        self.temperature_spin.setMinimumHeight(30)
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setValue(1.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        
        # Music style parameter
        self.style_label = QLabel("Müzik Stili:")
        self.style_label.setObjectName("param_label")
        self.style_label.setMinimumWidth(120)
        
        self.style_combo = QComboBox()
        self.style_combo.setObjectName("style_combo")
        self.style_combo.setMinimumWidth(150)
        self.style_combo.setMinimumHeight(30)
        self.style_combo.addItems(["Pop", "Rock", "Klasik", "Jazz", "Hip-Hop", "Elektronik"])
        self.style_combo.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        
        # Add parameters to form layout
        params_form.addRow(self.bar_count_label, self.bar_count_spin)
        params_form.addRow(self.tempo_label, self.tempo_spin)
        params_form.addRow(self.temperature_label, self.temperature_spin)
        params_form.addRow(self.style_label, self.style_combo)
        
        # Generate button
        self.generate_button = QPushButton("MIDI Üret")
        self.generate_button.setObjectName("generate_button")
        self.generate_button.setMinimumWidth(250)
        self.generate_button.setMinimumHeight(40)
        self.generate_button.clicked.connect(self.request_generation)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("generation_progress")
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Hazır")
        
        # Status label
        self.status_label = QLabel("Üretim için parametreleri ayarlayın ve 'MIDI Üret' butonuna tıklayın.")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        
        # Add components to generation layout
        generation_layout.addLayout(params_form, 0, 0, 4, 3)
        generation_layout.addWidget(self.generate_button, 4, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        generation_layout.addWidget(self.progress_bar, 5, 0, 1, 3)
        generation_layout.addWidget(self.status_label, 6, 0, 1, 3)
        
        # Set column stretches to ensure proper layout
        generation_layout.setColumnStretch(0, 0)  # Label column
        generation_layout.setColumnStretch(1, 0)  # Input column
        generation_layout.setColumnStretch(2, 1)  # Stretch column
        
        # Generated MIDI Group
        self.result_group = QGroupBox("Üretilen MIDI")
        self.result_group.setObjectName("result_group")
        self.result_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.result_group.setMinimumHeight(200)
        
        result_layout = QVBoxLayout(self.result_group)
        result_layout.setContentsMargins(15, 20, 15, 15)
        result_layout.setSpacing(10)
        
        # Result information
        self.result_info = QLabel("Henüz MIDI üretilmedi")
        self.result_info.setObjectName("result_info")
        self.result_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_info.setWordWrap(True)
        
        # Piano roll display for generated MIDI
        self.piano_roll_display = QLabel()
        self.piano_roll_display.setObjectName("generated_piano_roll")
        self.piano_roll_display.setMinimumSize(400, 250)
        self.piano_roll_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.piano_roll_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.piano_roll_display.setStyleSheet("background-color: rgba(0, 0, 0, 0.2); border-radius: 5px;")
        
        # Add components to result layout
        result_layout.addWidget(self.result_info)
        result_layout.addWidget(self.piano_roll_display)
        
        # Add groups to main layout
        main_layout.addWidget(self.generation_group)
        main_layout.addWidget(self.result_group)
        
        # Set size policies for responsive layout
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
    def request_generation(self):
        """Request MIDI generation with current parameters."""
        params = {
            'bar_count': self.bar_count_spin.value(),
            'tempo': self.tempo_spin.value(),
            'temperature': self.temperature_spin.value(),
            'style': self.style_combo.currentText()
        }
        
        self.generation_requested.emit(params)
        self.status_label.setText("MIDI üretiliyor...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Üretiliyor... %p%")
        logger.info(f"MIDI üretimi istendi: {params}")
        
    def update_progress(self, value, status_text=None):
        """Update the progress bar and status text."""
        self.progress_bar.setValue(value)
        if status_text:
            self.status_label.setText(status_text)
            self.progress_bar.setFormat(f"{status_text} %p%")
        
    def display_result(self, file_path, pixmap=None):
        """Display the generated MIDI result information."""
        if file_path:
            filename = file_path.split("/")[-1].split("\\")[-1]
            self.result_info.setText(f"Üretilen MIDI: {filename}\nDosya yolu: {file_path}")
            
            # Display piano roll if available
            if pixmap:
                scaled_pixmap = pixmap.scaled(
                    self.piano_roll_display.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.piano_roll_display.setPixmap(scaled_pixmap)
            
            # Reset progress and status
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Tamamlandı")
            self.status_label.setText(f"MIDI başarıyla üretildi: {filename}")
        else:
            self.result_info.setText("MIDI üretimi başarısız oldu.")
            self.piano_roll_display.clear()
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Hata")
            self.status_label.setText("MIDI üretimi sırasında bir hata oluştu.")
            
    def clear_displays(self):
        """Clear all displays."""
        self.result_info.setText("Henüz MIDI üretilmedi")
        self.piano_roll_display.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Hazır")
        self.status_label.setText("Üretim için parametreleri ayarlayın ve 'MIDI Üret' butonuna tıklayın.")

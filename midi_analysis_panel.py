# src/gui/panels/midi_analysis_panel.py
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QTextEdit, QGroupBox, QSizePolicy, QGridLayout,
    QFrame, QSpacerItem
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap

logger = logging.getLogger(__name__)

class MIDIAnalysisPanel(QWidget):
    """Panel for MIDI file selection and analysis functionality."""
    
    # Signals
    analysis_requested = pyqtSignal(str)  # Emitted when analysis is requested for a MIDI file
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.midi_file_path = ""
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # MIDI File Selection Group
        self.file_group = QGroupBox("MIDI Dosyası Seçimi")
        self.file_group.setObjectName("midi_file_group")
        self.file_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.file_group.setMinimumHeight(120)  # Daha kompakt yükseklik
        
        file_layout = QGridLayout(self.file_group)
        file_layout.setContentsMargins(12, 25, 12, 12)  # Daha az yatay padding, üstte daha fazla boşluk
        file_layout.setSpacing(8)  # Daha az boşluk
        
        # File selection components - First row
        self.file_path_label = QLabel("Seçilen Dosya:")
        self.file_path_label.setObjectName("file_path_label")
        self.file_path_label.setMinimumWidth(100)  # Daha dar etiket
        self.file_path_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        
        self.file_path_display = QLabel("Henüz bir MIDI dosyası seçilmedi")
        self.file_path_display.setObjectName("file_path_display")
        self.file_path_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.file_path_display.setWordWrap(True)
        self.file_path_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Butonlar için yatay layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)  # Butonlar arası daha az boşluk
        
        self.browse_button = QPushButton("Dosya Seç")
        self.browse_button.setObjectName("browse_button")
        self.browse_button.setMinimumWidth(120)
        self.browse_button.clicked.connect(self.browse_midi_file)
        
        self.analyze_button = QPushButton("Analiz Et")
        self.analyze_button.setObjectName("analyze_button")
        self.analyze_button.setMinimumWidth(120)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.request_analysis)
        
        button_layout.addWidget(self.browse_button)
        button_layout.addWidget(self.analyze_button)
        button_layout.addStretch()  # Butonları sola yasla
        
        # Add components to file layout
        file_layout.addWidget(self.file_path_label, 0, 0, 1, 1)
        file_layout.addWidget(self.file_path_display, 0, 1, 1, 2)
        file_layout.addLayout(button_layout, 1, 1, 1, 2)
        
        # Sütun genişlik ayarları
        file_layout.setColumnStretch(0, 0)  # Etiket sabit genişlikte
        file_layout.setColumnStretch(1, 0)  # Butonlar sabit genişlikte
        file_layout.setColumnStretch(2, 1)  # Boşluk genişleyebilir
        
        # Satır yükseklik ayarları
        file_layout.setRowStretch(0, 1)  # İlk satır (etiket ve dosya yolu)
        file_layout.setRowStretch(1, 1)  # İkinci satır (butonlar)
        
        # Analysis Results Group
        self.analysis_group = QGroupBox("Analiz Sonuçları")
        self.analysis_group.setObjectName("analysis_group")
        self.analysis_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.analysis_group.setMinimumHeight(300)  # Daha yüksek minimum yükseklik
        
        analysis_layout = QVBoxLayout(self.analysis_group)
        analysis_layout.setContentsMargins(12, 25, 12, 12)  # Daha az padding
        analysis_layout.setSpacing(8)  # Daha az boşluk
        
        # Results text area with scroll
        self.results_text = QTextEdit()
        self.results_text.setObjectName("results_text")
        self.results_text.setReadOnly(True)
        self.results_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.results_text.setMinimumHeight(120)
        
        # Piano roll section
        piano_roll_container = QVBoxLayout()
        piano_roll_container.setSpacing(4)
        
        # Piano roll label with a subtle separator
        self.piano_roll_label = QLabel("PİYANO RULOSU")
        self.piano_roll_label.setObjectName("piano_roll_label")
        self.piano_roll_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Piano roll display with border
        self.piano_roll_display = QLabel()
        self.piano_roll_display.setObjectName("piano_roll_display")
        self.piano_roll_display.setMinimumSize(400, 200)
        self.piano_roll_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        self.piano_roll_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add to piano roll container
        piano_roll_container.addWidget(self.piano_roll_label)
        piano_roll_container.addWidget(self.piano_roll_display)
        
        # Add all to main analysis layout
        analysis_layout.addWidget(self.results_text, 1)  # 1: stretch factor
        analysis_layout.addLayout(piano_roll_container, 2)  # 2: stretch factor
        
        # Set stretch factors
        analysis_layout.setStretch(0, 0)  # Title (handled by QGroupBox)
        analysis_layout.setStretch(1, 1)  # Results text (takes 1 part)
        analysis_layout.setStretch(2, 2)  # Piano roll (takes 2 parts)
        
        # Add groups to main layout
        main_layout.addWidget(self.file_group)
        main_layout.addWidget(self.analysis_group)
        
        # Set size policies for responsive layout
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
    def browse_midi_file(self):
        """Open a file dialog to select a MIDI file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "MIDI Dosyası Seç", "", "MIDI Files (*.mid *.midi);;All Files (*)"
        )
        
        if file_path:
            self.midi_file_path = file_path
            # Display only the filename to save space
            filename = file_path.split("/")[-1].split("\\")[-1]
            self.file_path_display.setText(filename)
            self.file_path_display.setToolTip(file_path)  # Show full path on hover
            self.analyze_button.setEnabled(True)
            logger.info(f"MIDI dosyası seçildi: {file_path}")
        
    def request_analysis(self):
        """Request analysis of the selected MIDI file."""
        if self.midi_file_path:
            self.analysis_requested.emit(self.midi_file_path)
            logger.info(f"MIDI dosyası analizi istendi: {self.midi_file_path}")
        
    def display_analysis_results(self, analysis_text):
        """Display the analysis results in the text area."""
        self.results_text.setPlainText(analysis_text)
        
    def display_piano_roll(self, pixmap):
        """Display the piano roll image."""
        if isinstance(pixmap, QPixmap):
            # Scale pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.piano_roll_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.piano_roll_display.setPixmap(scaled_pixmap)
        else:
            logger.error("Piano roll display failed: Invalid pixmap")
            self.piano_roll_display.clear()
            self.piano_roll_display.setText("Piano roll görüntülenemedi")
            
    def clear_displays(self):
        """Clear all displays."""
        self.results_text.clear()
        self.piano_roll_display.clear()

# src/gui/panels/memory_panel.py
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QListWidgetItem, QTextEdit, QComboBox,
    QGroupBox, QSizePolicy, QGridLayout, QFrame, QSplitter, QLayout,
    QLineEdit, QScrollArea
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)

class MemoryPanel(QWidget):
    """Panel for managing MIDI patterns in memory."""
    
    # Signals
    pattern_selected = pyqtSignal(str)  # Emitted when a pattern is selected
    memory_search_requested = pyqtSignal(str)  # Emitted when memory search is requested
    
    def __init__(self, parent=None):
        """Initialize the memory panel."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Create main layout with optimized spacing and margins
        self.layout = QVBoxLayout()
        self.layout.setSpacing(5)  # Reduced spacing
        self.layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        self.layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        
        # Set size policies for the panel
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create splitter with optimized sizes and size policies
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Pattern List Group (Left Side)
        self.pattern_group = QGroupBox("Hafıza Desenleri")
        self.pattern_group.setObjectName("pattern_group")
        self.pattern_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.pattern_layout = QVBoxLayout()
        self.pattern_layout.setSpacing(5)
        self.pattern_layout.setContentsMargins(5, 5, 5, 5)
        self.pattern_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.pattern_layout.setSpacing(2)
        
        # Search components
        self.search_layout = QHBoxLayout()
        self.search_layout.setSpacing(5)
        self.search_layout.setContentsMargins(5, 5, 5, 5)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Desenleri ara...")
        self.search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search_layout.addWidget(self.search_input)
        
        # Category filter
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Tüm Kategoriler", "Melodik", "Harmonik", "Ritmik"])
        self.category_combo.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.search_layout.addWidget(self.category_combo)
        
        # Pattern list
        self.pattern_list = QScrollArea()
        self.pattern_list.setWidgetResizable(True)
        self.pattern_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add search layout to pattern layout
        self.pattern_layout.addLayout(self.search_layout)
        self.pattern_layout.addWidget(self.pattern_list)
        
        self.pattern_group.setLayout(self.pattern_layout)
        
        # Pattern Detail Group (Right Side)
        self.detail_group = QGroupBox("Desen Detayları")
        self.detail_group.setObjectName("detail_group")
        self.detail_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.detail_layout = QVBoxLayout()
        self.detail_layout.setSpacing(5)
        self.detail_layout.setContentsMargins(5, 5, 5, 5)
        self.detail_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.detail_layout.setSpacing(2)
        
        # Pattern details
        self.detail_text = QLabel()
        self.detail_text.setWordWrap(True)
        self.detail_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.detail_layout.addWidget(self.detail_text)
        
        # Piano roll display for pattern
        self.piano_roll_label = QLabel("Desen Piyano Rulosu")
        self.piano_roll_label.setObjectName("piano_roll_label")
        self.piano_roll_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.piano_roll_display = QLabel()
        self.piano_roll_display.setObjectName("pattern_piano_roll")
        self.piano_roll_display.setMinimumSize(350, 200)
        self.piano_roll_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add components to detail layout
        self.detail_layout.addWidget(self.piano_roll_label)
        self.detail_layout.addWidget(self.piano_roll_display)
        self.piano_roll_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.piano_roll_display.setStyleSheet("background-color: rgba(0, 0, 0, 0.2); border-radius: 5px;")
        
        # Add controls
        self.control_layout = QHBoxLayout()
        self.control_layout.setSpacing(5)
        self.control_layout.setContentsMargins(5, 5, 5, 5)
        
        # Play button
        self.play_button = QPushButton("Oynat")
        self.play_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.control_layout.addWidget(self.play_button)
        
        # Save button
        self.save_button = QPushButton("Kaydet")
        self.save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.control_layout.addWidget(self.save_button)
        
        # Delete button
        self.delete_button = QPushButton("Sil")
        self.delete_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.control_layout.addWidget(self.delete_button)
        
        self.detail_layout.addLayout(self.control_layout)
        
        self.detail_group.setLayout(self.detail_layout)
        
        # Add groups to splitter
        self.splitter.addWidget(self.pattern_group)
        self.splitter.addWidget(self.detail_group)
        
        # Set initial sizes and stretch factors
        self.splitter.setSizes([300, 600])  # Adjust initial sizes
        self.splitter.setStretchFactor(0, 1)  # First widget (pattern group) can expand
        self.splitter.setStretchFactor(1, 2)  # Second widget (detail group) can expand more
        
        # Add splitter to main layout
        self.layout.addWidget(self.splitter)
        
        # Set layout to widget
        self.setLayout(self.layout)
        
    def search_memory(self):
        """Search memory patterns based on selected category."""
        category = self.category_combo.currentText()
        self.memory_search_requested.emit(category)
        logger.info(f"Hafıza araması istendi: {category}")
        
    def on_pattern_selected(self, item):
        """Handle pattern selection from the list."""
        if item:
            pattern_id = item.data(Qt.ItemDataRole.UserRole)
            self.pattern_selected.emit(pattern_id)
            logger.info(f"Desen seçildi: {pattern_id}")
            
    def update_pattern_list(self, patterns):
        """Update the pattern list with the provided patterns."""
        self.pattern_list.clear()
        
        if not patterns:
            item = QListWidgetItem("Desen bulunamadı")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.pattern_list.addItem(item)
            return
            
        for pattern_id, pattern_info in patterns.items():
            item = QListWidgetItem(pattern_info.get('name', f"Desen {pattern_id}"))
            item.setData(Qt.ItemDataRole.UserRole, pattern_id)
            item.setToolTip(pattern_info.get('description', ''))
            self.pattern_list.addItem(item)
            
        logger.info(f"{len(patterns)} desen listelendi")
        
    def display_pattern_details(self, pattern_info):
        """Display the details of the selected pattern."""
        try:
            if not pattern_info:
                # Clear all displays
                self.detail_text.clear()
                self.piano_roll_display.clear()
                self.piano_roll_display.setText("Desen seçili değil")
                self.piano_roll_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
                return
                
            # Format pattern details as text
            details = f"İsim: {pattern_info.get('name', 'Bilinmiyor')}\n"
            details += f"Kategori: {pattern_info.get('category', 'Bilinmiyor')}\n"
            details += f"Uzunluk: {pattern_info.get('length', 'Bilinmiyor')} bar\n"
            details += f"Tempo: {pattern_info.get('tempo', 'Bilinmiyor')} BPM\n"
            details += f"Oluşturulma: {pattern_info.get('created_at', 'Bilinmiyor')}\n\n"
            details += f"Açıklama: {pattern_info.get('description', '')}"
            
            # Update text display
            self.detail_text.setText(details)
            self.detail_text.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 0.2);
                    border-radius: 5px;
                    padding: 10px;
                    color: #e0e0e0;
                }
            """)
            
            # Update piano roll display if available
            if 'piano_roll_widget' in pattern_info and pattern_info['piano_roll_widget'] is not None:
                # Clear existing layout if any
                if self.piano_roll_display.layout():
                    layout = self.piano_roll_display.layout()
                    while layout.count():
                        child = layout.takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
                
                # Create new layout
                layout = QVBoxLayout()
                layout.addWidget(pattern_info['piano_roll_widget'])
                self.piano_roll_display.setLayout(layout)
                
                # Set minimum size for better visibility
                self.piano_roll_display.setMinimumSize(600, 350)
                
                # Update style
                self.piano_roll_display.setStyleSheet("""
                    QLabel {
                        background-color: rgba(0, 0, 0, 0.2);
                        border-radius: 5px;
                        padding: 10px;
                    }
                """)
            else:
                # If no piano roll widget, show a message
                self.piano_roll_display.setText("Piyano rulosu mevcut değil")
                self.piano_roll_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.piano_roll_display.setStyleSheet("""
                    QLabel {
                        background-color: rgba(0, 0, 0, 0.2);
                        border-radius: 5px;
                        padding: 20px;
                        color: #ff6b6b;
                    }
                """)
                
        except Exception as e:
            logger.error(f"Error displaying pattern details: {e}")
            self.piano_roll_display.setText(f"Hata: {str(e)}")
            self.piano_roll_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.piano_roll_display.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 0, 0, 0.1);
                    border-radius: 5px;
                    padding: 20px;
                    color: #ff6b6b;
                }
            """)
            
    def clear_displays(self):
        """Clear all displays."""
        self.pattern_list.clear()
        self.piano_roll_display.clear()
        self.detail_label.setText("Desen Bilgileri")

# src/gui/main_window_new.py
import logging
import os
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import io
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QStatusBar, QSplitter, QSizePolicy, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

# Import the panel components
from src.gui.panels import (
    MIDIAnalysisPanel,
    MIDIGenerationPanel,
    MemoryPanel,
    SettingsPanel
)

# Import core components
try:
    from src.core.settings import Settings
    from src.midi.processor import MIDIProcessor, MIDIAnalysis
    from src.midi.midi_memory import MIDIMemory, MIDIPattern
    from src.model.midi_model import MIDIModel
    from src.midi.instrument_library import InstrumentLibrary, MusicStyle
    _core_modules_imported = True
except ImportError as e:
    _core_modules_imported = False
    print(f"Error importing core modules: {e}")

# Worker thread for background tasks
class Worker(QThread):
    """Worker thread for long-running tasks."""
    
    # Signals
    analysis_finished = pyqtSignal(object)  # MIDIAnalysis or None
    generation_finished = pyqtSignal(object)  # Generated sequence or None
    training_progress = pyqtSignal(int, int, float)  # epoch, total_epochs, loss
    training_finished = pyqtSignal(object)  # Trained model or None
    memory_search_finished = pyqtSignal(dict)  # Dict of patterns
    
    def __init__(self, task, data=None, settings=None, processor=None, midi_model=None, midi_memory=None):
        """Initialize the worker thread."""
        super().__init__()
        self.task = task
        self.data = data or {}
        self.settings = settings
        self.processor = processor
        self.midi_model = midi_model
        self.midi_memory = midi_memory
        
    def run(self):
        """Execute the task."""
        logger = logging.getLogger(__name__)
        logger.info(f"Worker thread started: {self.task}")
        
        try:
            if self.task == "analyze_midi":
                # Analyze MIDI file
                midi_file = self.data.get("midi_file")
                if not midi_file or not os.path.exists(midi_file):
                    logger.error(f"MIDI dosyası bulunamadı: {midi_file}")
                    self.analysis_finished.emit(None)
                    return
                    
                if not self.processor:
                    logger.error("MIDI işlemcisi başlatılmamış")
                    self.analysis_finished.emit(None)
                    return
                    
                analysis_result = self.processor.analyze_file(midi_file)
                self.analysis_finished.emit(analysis_result)
                
            elif self.task == "generate_midi":
                # Generate MIDI sequence
                if not self.midi_model:
                    logger.error("MIDI modeli başlatılmamış")
                    self.generation_finished.emit(None)
                    return
                
                # Get generation parameters
                bar_count = self.data.get("bar_count", 4)
                temperature = self.data.get("temperature", 1.0)
                
                # Create seed sequence or use provided one
                seed_sequence = self.data.get("seed_sequence")
                if seed_sequence is None:
                    # Create random seed
                    seed_shape = self.midi_model.get_input_shape()
                    seed_sequence = np.random.rand(*seed_shape) * 0.1
                
                # Generate sequence
                try:
                    generated_sequence = self.midi_model.generate_sequence(
                        seed_sequence, 
                        bar_count=bar_count,
                        temperature=temperature
                    )
                    self.generation_finished.emit(generated_sequence)
                except Exception as e:
                    logger.error(f"MIDI üretimi sırasında hata: {e}")
                    self.generation_finished.emit(None)
                
            elif self.task == "train_model":
                # Train the model
                if not self.midi_model:
                    logger.error("MIDI modeli başlatılmamış")
                    self.training_finished.emit(None)
                    return
                    
                # Get training parameters
                patterns = self.data.get("patterns", [])
                epochs = self.data.get("epochs", 10)
                batch_size = self.data.get("batch_size", 32)
                
                if not patterns:
                    logger.error("Eğitim için desen bulunamadı")
                    self.training_finished.emit(None)
                    return
                
                # Create progress callback
                class ProgressCallback:
                    def __init__(self, worker_thread, total_epochs):
                        self.worker_thread = worker_thread
                        self.total_epochs = total_epochs
                        
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        loss = logs.get('loss', 0.0)
                        self.worker_thread.training_progress.emit(epoch + 1, self.total_epochs, loss)
                
                # Train model
                try:
                    callback = ProgressCallback(self, epochs)
                    self.midi_model.train(
                        patterns,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[callback]
                    )
                    self.training_finished.emit(self.midi_model)
                except Exception as e:
                    logger.error(f"Model eğitimi sırasında hata: {e}")
                    self.training_finished.emit(None)
                
            elif self.task == "search_memory":
                # Search memory patterns
                if not self.midi_memory:
                    logger.error("MIDI hafızası başlatılmamış")
                    self.memory_search_finished.emit({})
                    return
                    
                # Get search parameters
                category = self.data.get("category", "Tüm Kategoriler")
                
                # Search patterns
                try:
                    if category == "Tüm Kategoriler":
                        patterns = self.midi_memory.get_all_patterns()
                    else:
                        patterns = self.midi_memory.search_patterns_by_category(category)
                    
                    # Convert patterns to dict for UI display
                    pattern_dict = {}
                    for pattern in patterns:
                        pattern_dict[pattern.id] = {
                            'name': pattern.name,
                            'category': pattern.category,
                            'description': pattern.description,
                            'length': pattern.length,
                            'tempo': pattern.tempo,
                            'created_at': pattern.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(pattern, 'created_at') else "Unknown"
                        }
                    
                    self.memory_search_finished.emit(pattern_dict)
                except Exception as e:
                    logger.error(f"Hafıza araması sırasında hata: {e}")
                    self.memory_search_finished.emit({})
            
            else:
                logger.error(f"Bilinmeyen görev: {self.task}")
        
        except Exception as e:
            logger.error(f"Worker thread çalışırken hata: {e}")
            
            # Emit appropriate signal based on task
            if self.task == "analyze_midi":
                self.analysis_finished.emit(None)
            elif self.task == "generate_midi":
                self.generation_finished.emit(None)
            elif self.task == "train_model":
                self.training_finished.emit(None)
            elif self.task == "search_memory":
                self.memory_search_finished.emit({})
        
        logger.info(f"Worker thread tamamlandı: {self.task}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, settings):
        """Initialize the main window."""
        super().__init__()
        
        # Store settings
        self.settings = settings
        
        # Initialize core components
        self.initialize_core_components()
        
        # Set up the user interface
        self.setup_ui()
        
        # Connect signals and slots
        self.connect_signals()
        
        # Set window properties
        self.setWindowTitle("MIDI Composer")
        self.resize(1000, 800)  # Set initial size
        self.setMinimumSize(900, 700)  # Set minimum size to ensure all content is visible
        
        # Log initialization
        logger = logging.getLogger(__name__)
        logger.info("MainWindow initialized")
        
    def initialize_core_components(self):
        """Initialize the core components of the application."""
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize MIDI processor
            self.processor = MIDIProcessor()
            logger.info("MIDI processor initialized")
            
            # Initialize instrument library
            self.instrument_library = InstrumentLibrary()
            logger.info("Instrument library initialized")
            
            # Initialize MIDI model
            model_settings = self.settings.model_settings if hasattr(self.settings, 'model_settings') else None
            self.midi_model = MIDIModel(settings=model_settings)
            logger.info("MIDI model initialized")
            
            # Initialize MIDI memory
            memory_settings = self.settings.memory_settings if hasattr(self.settings, 'memory_settings') else None
            self.midi_memory = MIDIMemory(settings=memory_settings, instrument_library=self.instrument_library)
            self.midi_memory.load()  # Load patterns from file
            logger.info("MIDI memory initialized and loaded")
            
            # Set flag for successful initialization
            self._core_components_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing core components: {e}")
            self._core_components_initialized = False
            
            # Show error message
            QMessageBox.critical(
                self,
                "Başlatma Hatası",
                f"Uygulama bileşenleri başlatılırken hata oluştu:\n\n{e}\n\nBazı özellikler çalışmayabilir."
            )
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget
        self.central_widget = QWidget()
        self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create header with title
        header_layout = QHBoxLayout()
        
        self.title_label = QLabel("MIDI Composer")
        self.title_label.setObjectName("title_label")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #2196F3;")
        
        header_layout.addWidget(self.title_label)
        
        # Create tab widget for main content
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("main_tab_widget")
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setTabShape(QTabWidget.TabShape.Rounded)
        self.tab_widget.setMinimumHeight(500)
        
        # Create panels
        self.analysis_panel = MIDIAnalysisPanel()
        self.generation_panel = MIDIGenerationPanel()
        
        # Add panels to tabs
        self.tab_widget.addTab(self.analysis_panel, "MIDI Analizi")
        self.tab_widget.addTab(self.generation_panel, "MIDI Üretimi")
        
        # Create bottom tab widget for memory and settings
        self.bottom_tab_widget = QTabWidget()
        self.bottom_tab_widget.setObjectName("bottom_tab_widget")
        self.bottom_tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.bottom_tab_widget.setTabShape(QTabWidget.TabShape.Rounded)
        self.bottom_tab_widget.setMinimumHeight(300)
        
        # Create memory and settings panels
        self.memory_panel = MemoryPanel()
        self.settings_panel = SettingsPanel()
        
        # Add panels to bottom tabs
        self.bottom_tab_widget.addTab(self.memory_panel, "Hafıza/Desenler")
        self.bottom_tab_widget.addTab(self.settings_panel, "Ayarlar")
        
        # Create a splitter for resizable sections
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setObjectName("main_splitter")
        self.main_splitter.setChildrenCollapsible(False)
        
        # Add widgets to splitter
        self.main_splitter.addWidget(self.tab_widget)
        self.main_splitter.addWidget(self.bottom_tab_widget)
        
        # Set initial splitter sizes (60% top, 40% bottom)
        self.main_splitter.setSizes([600, 400])
        
        # Add layouts and widgets to main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(self.main_splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("status_bar")
        self.setStatusBar(self.status_bar)
        
        # Set initial status
        self.status_bar.showMessage("Uygulama hazır")
        
    def connect_signals(self):
        """Connect signals and slots."""
        # Analysis panel signals
        self.analysis_panel.analysis_requested.connect(self.analyze_midi_file)
        
        # Generation panel signals
        self.generation_panel.generation_requested.connect(self.generate_midi_sequence)
        
        # Memory panel signals
        self.memory_panel.pattern_selected.connect(self.display_pattern_details)
        self.memory_panel.memory_search_requested.connect(self.search_memory)
        
        # Settings panel signals
        self.settings_panel.settings_saved.connect(self.save_settings)
        self.settings_panel.settings_loaded.connect(self.load_settings)
        
        # Initialize settings panel with current settings
        self.settings_panel.set_settings(vars(self.settings))
        
    def analyze_midi_file(self, midi_file_path):
        """Analyze the selected MIDI file."""
        logger = logging.getLogger(__name__)
        logger.info(f"Analyzing MIDI file: {midi_file_path}")
        
        # Update status
        self.status_bar.showMessage(f"MIDI dosyası analiz ediliyor: {os.path.basename(midi_file_path)}")
        
        # Create worker thread
        self.worker_thread = Worker(
            task="analyze_midi",
            data={"midi_file": midi_file_path},
            processor=self.processor
        )
        
        # Connect signals
        self.worker_thread.analysis_finished.connect(self.on_analysis_finished)
        
        # Start worker thread
        self.worker_thread.start()
        
    def on_analysis_finished(self, analysis_result):
        """Handle analysis results."""
        logger = logging.getLogger(__name__)
        
        if analysis_result:
            logger.info("MIDI analysis completed successfully")
            
            # Format analysis results as text
            analysis_text = self.format_analysis_results(analysis_result)
            
            # Display analysis results
            self.analysis_panel.display_analysis_results(analysis_text)
            
            # Generate and display piano roll
            self.generate_piano_roll(analysis_result.midi_file, self.analysis_panel.piano_roll_display)
            
            # Update status
            self.status_bar.showMessage("MIDI analizi tamamlandı")
        else:
            logger.error("MIDI analysis failed")
            self.analysis_panel.display_analysis_results("MIDI dosyası analiz edilemedi.")
            self.status_bar.showMessage("MIDI analizi başarısız oldu")
            
    def format_analysis_results(self, analysis_result):
        """Format analysis results as text."""
        if not analysis_result:
            return "Analiz sonuçları mevcut değil."
            
        text = f"Dosya: {os.path.basename(analysis_result.midi_file)}\n"
        text += f"Uzunluk: {analysis_result.length_seconds:.2f} saniye\n"
        text += f"Tempo: {analysis_result.tempo} BPM\n"
        text += f"Zaman İmzası: {analysis_result.time_signature}\n"
        text += f"Ton: {analysis_result.key}\n\n"
        
        text += "Enstrümanlar:\n"
        for i, instrument in enumerate(analysis_result.instruments):
            text += f"  {i+1}. {instrument.name} (Program: {instrument.program})\n"
            text += f"     Nota Sayısı: {len(instrument.notes)}\n"
            
        text += "\nNotlar:\n"
        text += f"  Toplam Nota Sayısı: {analysis_result.total_notes}\n"
        text += f"  Ortalama Nota Hızı: {analysis_result.average_velocity:.2f}\n"
        text += f"  Ortalama Nota Uzunluğu: {analysis_result.average_note_duration:.2f} saniye\n"
        
        if hasattr(analysis_result, 'detected_chords') and analysis_result.detected_chords:
            text += "\nTespit Edilen Akorlar:\n"
            for chord in analysis_result.detected_chords[:10]:  # Show first 10 chords
                text += f"  {chord}\n"
            if len(analysis_result.detected_chords) > 10:
                text += f"  ... ve {len(analysis_result.detected_chords) - 10} akor daha\n"
                
        return text
        
    def generate_piano_roll(self, midi_file, display_label):
        """Generate and display a piano roll for the MIDI file."""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Plot piano roll
            plt.subplot(111)
            plot_piano_roll(midi_data)
            plt.title(f"Piano Roll: {os.path.basename(midi_file)}")
            
            # Save figure to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert buffer to QPixmap
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            
            # Display pixmap
            display_label.setPixmap(pixmap.scaled(
                display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Close figure to free memory
            plt.close()
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating piano roll: {e}")
            display_label.clear()
            display_label.setText("Piano roll oluşturulamadı")
            
    def generate_midi_sequence(self, params):
        """Generate a MIDI sequence with the specified parameters."""
        logger = logging.getLogger(__name__)
        logger.info(f"Generating MIDI sequence with parameters: {params}")
        
        # Update status
        self.status_bar.showMessage("MIDI üretiliyor...")
        
        # Update progress
        self.generation_panel.update_progress(10, "Model hazırlanıyor...")
        
        # Create worker thread
        self.worker_thread = Worker(
            task="generate_midi",
            data=params,
            midi_model=self.midi_model
        )
        
        # Connect signals
        self.worker_thread.generation_finished.connect(self.on_generation_finished)
        
        # Start worker thread
        self.worker_thread.start()
        
    def on_generation_finished(self, generated_sequence):
        """Handle generated MIDI sequence."""
        logger = logging.getLogger(__name__)
        
        if generated_sequence is not None:
            logger.info("MIDI generation completed successfully")
            
            # Update progress
            self.generation_panel.update_progress(50, "MIDI dosyası oluşturuluyor...")
            
            try:
                # Convert sequence to MIDI
                midi_data = self.midi_model.sequence_to_midi(generated_sequence)
                
                # Set tempo from parameters
                tempo = self.generation_panel.tempo_spin.value()
                midi_data.initial_tempo = tempo
                
                # Get style
                style = self.generation_panel.style_combo.currentText()
                
                # Create output filename with timestamp, tempo, and style
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                bar_count = self.generation_panel.bar_count_spin.value()
                filename = f"generated_{timestamp}_{bar_count}bar_{tempo}bpm_{style}.mid"
                
                # Get output directory from settings
                output_dir = getattr(self.settings, 'output_dir', os.path.join(os.path.dirname(__file__), '..', '..', 'generated_midi'))
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Save MIDI file
                output_path = os.path.join(output_dir, filename)
                midi_data.write(output_path)
                
                # Generate piano roll
                self.generate_piano_roll(output_path, self.generation_panel.piano_roll_display)
                
                # Display result
                self.generation_panel.display_result(output_path)
                
                # Update status
                self.status_bar.showMessage(f"MIDI üretildi: {filename}")
                
                # Update progress
                self.generation_panel.update_progress(100, "Tamamlandı")
                
            except Exception as e:
                logger.error(f"Error saving generated MIDI: {e}")
                self.generation_panel.display_result(None)
                self.status_bar.showMessage("MIDI dosyası oluşturulurken hata oluştu")
        else:
            logger.error("MIDI generation failed")
            self.generation_panel.display_result(None)
            self.status_bar.showMessage("MIDI üretimi başarısız oldu")
            
    def search_memory(self, category):
        """Search memory patterns by category."""
        logger = logging.getLogger(__name__)
        logger.info(f"Searching memory patterns with category: {category}")
        
        # Update status
        self.status_bar.showMessage(f"Hafıza desenleri aranıyor: {category}")
        
        # Create worker thread
        self.worker_thread = Worker(
            task="search_memory",
            data={"category": category},
            midi_memory=self.midi_memory
        )
        
        # Connect signals
        self.worker_thread.memory_search_finished.connect(self.on_memory_search_finished)
        
        # Start worker thread
        self.worker_thread.start()
        
    def on_memory_search_finished(self, patterns):
        """Handle memory search results."""
        logger = logging.getLogger(__name__)
        
        # Update pattern list
        self.memory_panel.update_pattern_list(patterns)
        
        # Update status
        pattern_count = len(patterns)
        self.status_bar.showMessage(f"Hafıza araması tamamlandı: {pattern_count} desen bulundu")
        
        logger.info(f"Memory search completed: {pattern_count} patterns found")
        
    def display_pattern_details(self, pattern_id):
        """Display details of the selected pattern."""
        logger = logging.getLogger(__name__)
        logger.info(f"Displaying pattern details: {pattern_id}")
        
        # Get pattern from memory
        pattern = self.midi_memory.get_pattern(pattern_id)
        
        if pattern:
            # Convert pattern to dict for display
            pattern_info = {
                'name': pattern.name,
                'category': pattern.category,
                'description': pattern.description,
                'length': pattern.length,
                'tempo': pattern.tempo,
                'created_at': pattern.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(pattern, 'created_at') else "Unknown"
            }
            
            # Generate piano roll if sequence exists
            if hasattr(pattern, 'sequence') and pattern.sequence is not None:
                try:
                    # Convert sequence to MIDI
                    midi_data = self.midi_model.sequence_to_midi(pattern.sequence)
                    
                    # Create a temporary file
                    temp_file = os.path.join(os.path.dirname(__file__), 'temp_pattern.mid')
                    midi_data.write(temp_file)
                    
                    # Generate piano roll
                    plt.figure(figsize=(8, 4))
                    plot_piano_roll(midi_data)
                    plt.title(f"Piano Roll: {pattern.name}")
                    
                    # Save figure to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    
                    # Convert buffer to QPixmap
                    image = QImage.fromData(buf.getvalue())
                    pixmap = QPixmap.fromImage(image)
                    
                    # Add piano roll to pattern info
                    pattern_info['piano_roll'] = pixmap
                    
                    # Close figure to free memory
                    plt.close()
                    
                    # Remove temporary file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    logger.error(f"Error generating pattern piano roll: {e}")
            
            # Display pattern details
            self.memory_panel.display_pattern_details(pattern_info)
            
        else:
            logger.error(f"Pattern not found: {pattern_id}")
            self.memory_panel.display_pattern_details(None)
            
    def save_settings(self, settings_dict):
        """Save settings to the application."""
        logger = logging.getLogger(__name__)
        logger.info("Saving settings")
        
        try:
            # Update settings object with new values
            for key, value in settings_dict.items():
                setattr(self.settings, key, value)
                
            # Save settings to file
            self.settings.save()
            
            # Update status
            self.status_bar.showMessage("Ayarlar kaydedildi")
            
            logger.info("Settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self.status_bar.showMessage("Ayarlar kaydedilirken hata oluştu")
            
    def load_settings(self):
        """Load settings from file."""
        logger = logging.getLogger(__name__)
        logger.info("Loading settings")
        
        try:
            # Load settings from file
            self.settings.load()
            
            # Update settings panel
            self.settings_panel.set_settings(vars(self.settings))
            
            # Update status
            self.status_bar.showMessage("Ayarlar yüklendi")
            
            logger.info("Settings loaded successfully")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self.status_bar.showMessage("Ayarlar yüklenirken hata oluştu")


def plot_piano_roll(midi_data, start_pitch=21, end_pitch=109, fs=100):
    """Plot a piano roll visualization of the MIDI data."""
    # Get the piano roll
    piano_roll = midi_data.get_piano_roll(fs=fs)
    
    # Trim to the specified pitch range
    piano_roll = piano_roll[start_pitch:end_pitch]
    
    # Plot the piano roll
    plt.imshow(
        piano_roll,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        cmap='Blues'
    )
    
    # Set labels
    plt.ylabel('Pitch')
    plt.xlabel('Time (s)')
    
    # Add grid
    plt.grid(alpha=0.3)
    
    return plt


# Standalone test
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting standalone test")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create dummy settings
    class DummySettings:
        def __init__(self):
            self.model_dir_path = "models"
            self.memory_file_full_path = "memory/patterns.json"
            self.output_dir = "generated_midi"
            
        def save(self):
            logger.info("Dummy settings save called")
            
        def load(self):
            logger.info("Dummy settings load called")
    
    # Create main window
    main_window = MainWindow(DummySettings())
    main_window.show()
    
    # Start event loop
    sys.exit(app.exec())

# src/gui/main_window.py
# --- KONTROL SATIRI: BU MESAJI KONSOLDA GÖRÜRSENIZ, DOĞRU main_window.py ÇALIŞIYOR DEMEKTİR --
# Bu satır sadece test amaçlıdır, isterseniz silebilirsiniz.
print("--- Running the verified and corrected main_window.py ---\n") # Added newline for clarity
# ------------------------------------------------------------------------------------------

# Yeni oluşturulan yardımcı modülleri import et
from src.utils.ui_helpers import create_styled_label, create_styled_button, UIStyles
from src.utils.error_handlers import handle_error
from src.utils.event_handlers import MIDIEventHandler
from src.utils.paths import (
    get_project_root, get_logs_dir, get_config_dir, 
    get_resources_dir, get_temp_dir, get_memory_dir, get_model_dir
)

# Standard library imports
import logging
import os
import numpy as np
import pretty_midi
import tensorflow as tf # TensorFlow hala model için gerekli
import matplotlib.pyplot as plt
import io
import random
from dataclasses import dataclass, field, asdict # Added asdict for settings serialization in test
from typing import List, Tuple, Optional, Dict, Any, Set # Added Set and ensured all others are here
import uuid # Import uuid
import sys # Import sys for sys.path check, though main.py sets it
import shutil # Import shutil for file moving or copying
from datetime import datetime # Used for date/time formatting if needed [cite: 3]
import json # Needed for asdict/json.dumps in Settings display and dummy settings

# PyQt6 imports
# Import only necessary modules and classes
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout,
    QWidget, QLabel, QLineEdit, QListWidget, QListWidgetItem, QTextEdit,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar, QTabWidget,
    QMessageBox, QGroupBox, QFormLayout, QHBoxLayout, QCheckBox, QRadioButton,
    QPlainTextEdit, QSplitter, QSizePolicy, QGridLayout, QFrame, QSpacerItem, QAbstractItemView, QStackedWidget # QStackedWidget eklendi
)
from PyQt6.QtGui import QIcon, QPixmap, QPixmapCache, QImage, QFont, QPalette, QBrush, QPainter, qRgb, QColor, QPen, QPainterPath, QLinearGradient # QPixmapCache eklendi
from PyQt6.QtCore import QPoint, QRect # QPoint ve QRect eklendi
import math # math modülü eklendi
import random # random modülü eklendi
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QTimer # Added QTimer for delayed actions if needed


# --- KRİTİK DÜZELTME: Çekirdek Modülleri Doğru Import Et ---
# main.py sys.path'e src dizinini eklediği için, modüllere doğrudan erişebilmeliyiz. [cite: 4]
# Karmaşık try...except bloklarını ve placeholder sınıflarını kaldırıyoruz.
# Import the necessary modules and classes from src package [cite: 5]
try:
    # Assuming src.core.settings is correct and includes the necessary typing imports (List, Dict, Optional, Any, Set, Tuple) and logger setup
    from src.core.settings import Settings, ModelSettings, GeneralSettings, MemorySettings # Import necessary settings dataclasses
    # Assuming src.midi package structure
    from src.midi.instrument_library import InstrumentLibrary, MusicStyle, InstrumentCategory
    from src.midi.processor import MIDIProcessor, MIDIAnalysis # Import MIDIProcessor and MIDIAnalysis
    from src.midi.midi_memory import MIDIMemory, MIDIPattern, PatternCategory # Import MIDIMemory and related classes
    # Assuming src.model package structure [cite: 6]
    from src.model.midi_model import MIDIModel # Import MIDIModel (TensorFlow model wrapper)

    _core_modules_imported = True
    # Logger should be obtained AFTER logging is configured in main.py
    # Delay getting the logger instance until later in __init__ or methods if __name__ != "__main__"
    # However, if main.py configures logging early and imports this, getting it here is fine.
    # Get logger for this module [cite: 7]
    logger = logging.getLogger(__name__)
    logger.info("Core modules imported successfully.")

except ImportError as e:
    # This is a critical failure if these core modules cannot be imported. [cite: 8]
    # The application cannot run without them.
    # Ensure logger exists even if imports fail to log the critical error. [cite: 9]
    # This logger setup might be redundant if main.py sets it up, but good for standalone test or early errors.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s: %(message)s") # Basic config if no handlers exist [cite: 10]
    logger = logging.getLogger(__name__) # Get logger even if imports fail
    logger.critical(f"FATAL ERROR: Failed to import core application modules: {e}. Application cannot start.", exc_info=True)
    _core_modules_imported = False
    # Since imports failed, we cannot proceed. The main() function in main.py should handle this check. [cite: 11]
    # No need for complex placeholders if the app will exit.

# --- Worker Thread for Long-Running Tasks --- [cite: 12]
# Tasks like MIDI analysis, model training, sequence generation, memory search
# should be run in a separate thread to keep the GUI responsive.
class Worker(QThread): # [cite: 13]
    # Define signals to communicate results back to the main thread (GUI)
    # --- FIX: Removed Optional[] from pyqtSignal definitions ---
    # pyqtSignal does not directly support complex type hints like Optional[] (Union)
    # The slot receiving the signal should handle the possibility of receiving None if applicable.
    analysis_finished = pyqtSignal(MIDIAnalysis) # Removed Optional[] [cite: 14]
    # Signal for sequence generation result: emits generated numpy array or None
    generation_finished = pyqtSignal(np.ndarray) # Removed Optional[]
    # Signal for training progress: emits (epoch, total_epochs, current_loss)
    training_progress = pyqtSignal(int, int, float) # epoch, total_epochs, current_loss
    # Signal for training finished: emits training history or None
    training_finished = pyqtSignal(object) # Use object instead of Optional[Any] or history type
    # Signal for memory search finished: emits list of similar patterns
    memory_search_finished = pyqtSignal(list) # Use list instead of List[MIDIPattern] [cite: 15]
    # Signal for generic task finished (e.g., saving/loading model/memory)
    task_finished = pyqtSignal(str, bool, str) # task_name, success, message # Assuming message is str
    # Signal for errors during a task
    error = pyqtSignal(str, str) # task_name, error_message


    def __init__(self, task: str, data: Optional[Dict[str, Any]] = None, settings=None, processor=None, midi_model=None, midi_memory=None):
        super().__init__()
        self.task = task
        self.data = data if data is not None else {} # [cite: 16]
        # Pass necessary objects to the worker
        self.settings = settings
        self.processor = processor # Pass the MIDIProcessor instance
        self.midi_model = midi_model # Pass the MIDIModel instance
        self.midi_memory = midi_memory # Pass the MIDIMemory instance


    def run(self):
        """
        The main method where the worker thread's task is executed. [cite: 17]
        """
        logger.debug(f"Worker thread started for task: {self.task}")
        try:
            if self.task == "analyze_midi":
                midi_path = self.data.get("midi_path")
                if midi_path and self.processor:
                    logger.info(f"Worker: Analyzing MIDI file: {midi_path}") # [cite: 18]
                    analysis_result = self.processor.analyze_midi_file(midi_path)
                    # Emit the result (can be None if analyze_midi_file returns None)
                    # The slot should handle receiving None if analyze_midi_file can return None
                    self.analysis_finished.emit(analysis_result) # [cite: 19]
                    logger.info(f"Worker: MIDI analysis finished for: {midi_path}")
                else:
                     error_msg = "Worker: Cannot analyze MIDI: path or processor not provided."
                     logger.error(error_msg) # [cite: 20]
                     self.error.emit(self.task, error_msg)
                     # Explicitly emit None if analysis failed due to missing input/processor
                     # self.analysis_finished.emit(None) # Only emit if signature allowed None


            elif self.task == "generate_sequence":
                seed_sequence = self.data.get("seed_sequence") # [cite: 21]
                num_steps = self.data.get("num_steps", 100)
                temperature = self.data.get("temperature", 1.0)
                
                # Yeni parametreleri al
                tempo = self.data.get("tempo", 120) # [cite: 22]
                style = self.data.get("style", "Otomatik")
                bar_count = self.data.get("bar_count", 4)
                
                if seed_sequence is not None and self.midi_model:
                    logger.info(f"Worker: Generating sequence for {num_steps} steps ({bar_count} bars) at {tempo} BPM with {style} style...") # [cite: 23]
                    generated_sequence = self.midi_model.generate_sequence(seed_sequence, num_steps, temperature)
                    
                    # Verileri kaydet - on_generation_finished'da kullanılacak
                    self.data["tempo"] = tempo # [cite: 24]
                    self.data["style"] = style
                    self.data["bar_count"] = bar_count
                    
                    # Emit the result (can be None if generate_sequence returns None) [cite: 25]
                    self.generation_finished.emit(generated_sequence)
                    logger.info(f"Worker: Sequence generation finished: {bar_count} bars, {tempo} BPM, {style} style")
                else:
                     error_msg = "Worker: Cannot generate sequence: seed or midi_model not provided."
                     logger.error(error_msg) # [cite: 26]
                     self.error.emit(self.task, error_msg)
                     # Explicitly emit None if generation failed
                     # self.generation_finished.emit(None) # Only emit if signature allowed None


            elif self.task == "train_model": # [cite: 27]
                 training_data = self.data.get("training_data")
                 epochs = self.data.get("epochs", 10)
                 validation_data = self.data.get("validation_data")
                 if training_data is not None and self.midi_model:
                      logger.info(f"Worker: Starting model training for {epochs} epochs...") # [cite: 28]
                      # Training requires a callback to report progress
                      # Keras Model.fit can take callbacks. Need to pass a custom callback [cite: 29]
                      # that emits the training_progress signal.
                      class ProgressCallback(tf.keras.callbacks.Callback): # [cite: 30]
                           def __init__(self, worker_thread, total_epochs):
                                super().__init__()
                                self.worker_thread = worker_thread
                                self.total_epochs = total_epochs # [cite: 31]

                           def on_epoch_end(self, epoch, logs=None):
                                logs = logs or {}
                                loss = logs.get('loss', -1.0) # Get loss, default to -1 if not available [cite: 32]
                                # Emit progress signal (epoch is 0-indexed, add 1 for display)
                                # Ensure float conversion for loss [cite: 33]
                                self.worker_thread.training_progress.emit(epoch + 1, self.total_epochs, float(loss))


                      callback = ProgressCallback(self, epochs) # Pass worker and total epochs
                      # The train method in MIDIModel needs to accept a callbacks list [cite: 34]
                      # Modify MIDIModel.train to accept callbacks=[callback]
                      # Check if train method exists and accepts 'callbacks' argument
                      if hasattr(self.midi_model, 'train') and callable(self.midi_model.train): # [cite: 35]
                           import inspect
                           sig = inspect.signature(self.midi_model.train)
                           if 'callbacks' in sig.parameters:
                                logger.debug("MIDIModel.train accepts 'callbacks'. Passing progress callback.") # [cite: 36]
                                history = self.midi_model.train(training_data, epochs, validation_data, callbacks=[callback]) # [cite: 37]
                           else:
                                logger.warning("MIDIModel.train does not accept 'callbacks'. Cannot report progress via callback.") # [cite: 38]
                                # Run training without callback
                                history = self.midi_model.train(training_data, epochs, validation_data)
                      else: # [cite: 39]
                            logger.error("MIDIModel object does not have a callable 'train' method.")
                            history = None # Indicate training failed


                      # Emit the result (can be None if train returns None or method not callable) [cite: 40]
                      # The slot should handle receiving None
                      self.training_finished.emit(history)
                      logger.info("Worker: Model training finished.")
                 else:
                      error_msg = "Worker: Cannot train model: training data or midi_model not provided." # [cite: 41]
                      logger.error(error_msg)
                      self.error.emit(self.task, error_msg)
                      # Explicitly emit None if training failed [cite: 42]
                      # self.training_finished.emit(None) # Only emit if signature allowed None


            elif self.task == "save_model":
                 file_path = self.data.get("file_path")
                 if file_path and self.midi_model:
                      logger.info(f"Worker: Saving model to {file_path}...") # [cite: 43]
                      success = self.midi_model.save_model(file_path)
                      # Emit success and message (assuming message is a string)
                      # The slot should handle receiving bool and str [cite: 44]
                      self.task_finished.emit(self.task, success, f"Model saved successfully." if success else f"Failed to save model.")
                      logger.info(f"Worker: Model save finished (Success: {success}).")
                 else:
                      error_msg = "Worker: Cannot save model: file path or midi_model not provided." # [cite: 45]
                      logger.error(error_msg)
                      self.error.emit(self.task, error_msg)
                      self.task_finished.emit(self.task, False, error_msg)


            elif self.task == "load_model":
                 file_path = self.data.get("file_path") # [cite: 46]
                 if file_path and self.midi_model:
                      logger.info(f"Worker: Loading model from {file_path}...")
                      success = self.midi_model.load_model(file_path)
                      # Emit success and message [cite: 47]
                      # The slot should handle receiving bool and str
                      self.task_finished.emit(self.task, success, f"Model loaded successfully." if success else f"Failed to load model.")
                      logger.info(f"Worker: Model load finished (Success: {success}).")
                 else: # [cite: 48]
                      error_msg = "Worker: Cannot load model: file path or midi_model not provided."
                      logger.error(error_msg)
                      self.error.emit(self.task, error_msg)
                      self.task_finished.emit(self.task, False, error_msg) # [cite: 49]


            elif self.task == "add_pattern_to_memory":
                 midi_path = self.data.get("midi_path")
                 analysis = self.data.get("analysis")
                 category = self.data.get("category")
                 tags = self.data.get("tags") # [cite: 50]
                 if midi_path and self.midi_memory: # Analysis can be None
                      logger.info(f"Worker: Adding pattern from {midi_path} to memory...")
                      # Assuming add_pattern returns pattern_id (str) or None
                      pattern_id = self.midi_memory.add_pattern(midi_path, analysis, category, tags) # [cite: 51]
                      success = pattern_id is not None
                      # Emit success and message (assuming message is a string)
                      self.task_finished.emit(self.task, success, f"Pattern added with ID: {pattern_id[:6]}..." if success else f"Failed to add pattern from {os.path.basename(midi_path)}") # [cite: 52]
                      logger.info(f"Worker: Add pattern to memory finished (Success: {success}).")
                 else:
                      error_msg = "Worker: Cannot add pattern to memory: path or midi_memory not provided."
                      logger.error(error_msg) # [cite: 53]
                      self.error.emit(self.task, error_msg)
                      self.task_finished.emit(self.task, False, error_msg)


            elif self.task == "find_similar_patterns":
                 reference_analysis = self.data.get("reference_analysis")
                 if reference_analysis and self.midi_memory: # [cite: 54]
                      logger.info(f"Worker: Searching for similar patterns to reference (Tempo: {reference_analysis.tempo:.2f})...")
                      # Assuming find_similar_patterns returns List[MIDIPattern]
                      similar_patterns = self.midi_memory.find_similar_patterns(reference_analysis)
                      # Emit the list of patterns (can be empty list) [cite: 55]
                      # The slot should handle receiving a list
                      self.memory_search_finished.emit(similar_patterns)
                      logger.info(f"Worker: Similar pattern search finished. Found {len(similar_patterns)} matches.") # [cite: 56]
                 else:
                      error_msg = "Worker: Cannot search for similar patterns: reference analysis or midi_memory not provided."
                      logger.error(error_msg)
                      self.error.emit(self.task, error_msg) # [cite: 57]
                      self.task_finished.emit(self.task, False, error_msg)


            elif self.task == "save_memory":
                 if self.midi_memory: # [cite: 58]
                      logger.info("Worker: Saving memory...")
                      # Assuming MIDIMemory has a public save method or _save_memory is accessible/callable
                      try:
                           # Assuming save() is the public method that handles saving
                           self.midi_memory.save() # Call the public save method [cite: 59]
                           success = True
                           message = "Memory saved successfully."
                           logger.info("MIDIMemory saved successfully.")
                      except Exception as e: # [cite: 60]
                           success = False
                           message = f"Error saving memory: {e}"
                           logger.error(message, exc_info=True) # [cite: 61]

                      # Emit success and message
                      self.task_finished.emit(self.task, success, message)
                      logger.info(f"Worker: Memory save finished (Success: {success}).")

                 else: # [cite: 62]
                      error_msg = "Worker: Cannot save memory: midi_memory not provided."
                      logger.error(error_msg)
                      self.error.emit(self.task, error_msg)
                      self.task_finished.emit(self.task, False, error_msg) # [cite: 63]


            else:
                error_msg = f"Worker: Unknown task specified: {self.task}"
                logger.error(error_msg)
                self.error.emit(self.task, error_msg)


        except Exception as e:
            # Catch any unhandled exceptions within the worker thread [cite: 64]
            error_msg = f"Worker: Unhandled exception during task '{self.task}': {e}"
            logger.critical(error_msg, exc_info=True)
            self.error.emit(self.task, error_msg)

        logger.debug(f"Worker thread finished for task: {self.task}")


# --- Main Application Window ---
class MainWindow(QMainWindow):
    # --- FIX: Pass settings object to MainWindow __init__ ---
    def __init__(self, settings):
        super().__init__()

        # Check if core modules were imported successfully in the main module [cite: 65]
        # This check should ideally be done in main.py before creating MainWindow
        if not _core_modules_imported:
             # If core imports failed, show a critical message box and exit. [cite: 66]
             # QApplication must exist before showing a QMessageBox.
             # This part might be redundant if main.py already checks and exits
             app = QApplication.instance() # Check if QApplication exists
             if app is None: # Create QApplication if it doesn't exist
                  app = QApplication(sys.argv)
             QMessageBox.critical(self, "Critical Error", "Uygulama başlatılamadı: Gerekli çekirdek modüller bulunamadı.") # [cite: 67]
             sys.exit(1) # Exit application


        # Store the settings object
        self.settings = settings

        # --- Initialize core components ---
        # Pass settings and other necessary objects to core components
        self.instrument_library = None
        self.processor = None
        self.midi_model = None # [cite: 68]
        self.midi_memory = None

        # --- FIX: Initialize components correctly using settings and passed objects ---
        # These initializations were causing TypeErrors before due to incorrect parameters
        # Added try...except blocks here to catch potential initialization errors
        try:
            # InstrumentLibrary might not need settings or other objects initially, but pass if its __init__ changes. [cite: 69]
            # Assuming InstrumentLibrary() does not take parameters based on its __init__ in instrument_library.py
            if 'InstrumentLibrary' in globals() and issubclass(InstrumentLibrary, object):
                 self.instrument_library = InstrumentLibrary()
                 logger.info("InstrumentLibrary initialized successfully.")
            else:
                 logger.error("InstrumentLibrary class not available for initialization.") # [cite: 70]
                 self.instrument_library = None


        except Exception as e:
             logger.error(f"Failed to initialize InstrumentLibrary: {e}", exc_info=True)
             self.instrument_library = None # Ensure it's None if initialization failed


        try:
            # MIDIProcessor needs only settings parameter
            # Check if processor import was successful before initializing [cite: 71]
            if 'MIDIProcessor' in globals() and issubclass(MIDIProcessor, object): # Basic check if class exists
                 self.processor = MIDIProcessor(settings=self.settings) # Pass only settings parameter
                 logger.info("MIDIProcessor initialized successfully.")
            else:
                 logger.error("MIDIProcessor class not available for initialization.") # [cite: 72]
                 self.processor = None # Ensure it's None if class is missing


        except Exception as e:
            # This catch block should catch any errors during MIDIProcessor __init__
            logger.error(f"Failed to initialize MIDIProcessor: {e}", exc_info=True)
            self.processor = None # Ensure it's None if initialization failed [cite: 73]


        try:
            # MIDIModel needs settings
            # Removed note_range_size and other individual parameters from __init__ call
            # Added settings parameter as per updated midi_model.py
            # Check if MIDIModel import was successful before initializing [cite: 74]
            if 'MIDIModel' in globals() and issubclass(MIDIModel, object): # Basic check if class exists
                 self.midi_model = MIDIModel(settings=self.settings) # Pass settings object
                 # --- FIX: Build the model after initialization ---
                 # Model is not built in __init__, call build_model separately.
                 if self.midi_model: # Check if initialization was successful [cite: 75]
                     self.midi_model.build_model() # Build the Keras model after object creation
                     logger.info("MIDIModel built successfully.")
                 else:
                      logger.error("MIDIModel object is None after initialization attempt.") # [cite: 76]
            else:
                 logger.error("MIDIModel class not available for initialization.")
                 self.midi_model = None # Ensure it's None if class is missing


        except Exception as e:
             # This catch block should catch any errors during MIDIModel __init__
             logger.error(f"Failed to initialize MIDIModel: {e}", exc_info=True) # [cite: 77]
             self.midi_model = None # Ensure it's None if initialization failed


        try:
            # MIDIMemory needs settings and InstrumentLibrary
            # Added settings and instrument_library parameters as per updated midi_memory.py
            # Check if MIDIMemory import was successful before initializing [cite: 78]
            if 'MIDIMemory' in globals() and issubclass(MIDIMemory, object): # Basic check if class exists
                 self.midi_memory = MIDIMemory(settings=self.settings, instrument_library=self.instrument_library) # Pass settings and instrument_library
                 logger.info("MIDIMemory initialized successfully.")
            else:
                 logger.error("MIDIMemory class not available for initialization.") # [cite: 79]
                 self.midi_memory = None # Ensure it's None if class is missing


        except Exception as e:
             # This catch block should catch any errors during MIDIMemory __init__
             logger.error(f"Failed to initialize MIDIMemory: {e}", exc_info=True)
             self.midi_memory = None # Ensure it's None if initialization failed [cite: 80]


        # --- GUI Setup ---
        self.setWindowTitle("MIDI Composer AI")
        # Get project root path relative to main_window.py for icon
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        # QSS stil dosyasını yükle
        try:
            # Stil dosyasının yolunu bul [cite: 81]
            style_path = os.path.join(script_dir, 'styles.qss')
            
            # Dosyanın var olduğunu kontrol et
            if os.path.exists(style_path):
                # Dosyayı aç ve içeriğini oku
                with open(style_path, 'r', encoding='utf-8') as style_file: # [cite: 82]
                    style_content = style_file.read()
                    # Stili uygula
                    self.setStyleSheet(style_content)
                    logger.info(f"Stylesheet loaded from: {style_path}")
            else: # [cite: 83]
                logger.warning(f"Stylesheet file not found at: {style_path}")
        except Exception as e:
            logger.error(f"Error loading stylesheet: {e}", exc_info=True)

        # Use the generate icon from resources/icons/
        icon_path = os.path.join(project_root_path, 'resources', 'icons', 'up-arrow.svg')

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path)) # [cite: 84]
            logger.debug(f"Application icon set from: {icon_path}")
        else:
            # Fallback to icons.svg in gui resources
            fallback_icon = os.path.join(script_dir, 'resources', 'icons.svg')
            if os.path.exists(fallback_icon):
                self.setWindowIcon(QIcon(fallback_icon))
                logger.debug(f"Application icon set from fallback: {fallback_icon}") # [cite: 85]
            else:
                logger.warning(f"No application icon found at: {icon_path} or {fallback_icon}")

        # Pencere boyutunu ve konumunu ayarla - içeriğin düzgün görünmesi için yeterli
        self.setGeometry(100, 100, 1200, 900) # Başlangıç pencere boyutu artırıldı
        self.setMinimumSize(1000, 800) # Minimum pencere boyutu artırıldı

        # Ana widget oluştur
        self.central_widget = QWidget()
        self.central_widget.setObjectName("central_widget") # QSS için objectName
        self.setCentralWidget(self.central_widget)
        
        # Arka plan resmi için QLabel kullan - bu en güvenilir yöntemdir
        # Önce resim dosyasının tam yolunu belirle - kullanıcının belirttiği konum
        background_path = os.path.join(os.path.dirname(__file__), 'resources', 'background.jpg')
        logger.info(f"Arka plan resmi yolu: {background_path}")
        logger.info(f"Dosya var mı: {os.path.exists(background_path)}")
        logger.info(f"Dosya boyutu: {os.path.getsize(background_path) if os.path.exists(background_path) else 'Dosya yok'}")
        
        if os.path.exists(background_path) and os.path.getsize(background_path) > 1000:
            try:
                # Önce önbelleği temizle
                QPixmapCache.clear()
                
                # Arka plan resmi için QLabel oluştur
                self.background_label = QLabel(self.central_widget)
                self.background_label.setGeometry(0, 0, self.width(), self.height())
                
                # Etiketin içeriğinin boyutunu ayarla
                self.background_label.setScaledContents(True) # Bu, resmin etiket boyutuna göre otomatik ölçeklenmesini sağlar
                
                # Resmi doğrudan dosyadan yükle
                background_pixmap = QPixmap()
                load_success = background_pixmap.load(background_path)
                
                if load_success and not background_pixmap.isNull():
                    logger.info(f"Resim başarıyla yüklendi. Boyut: {background_pixmap.width()}x{background_pixmap.height()}")
                    
                    # Resmi etikete ayarla
                    self.background_label.setPixmap(background_pixmap)
                    
                    # Arka plan etiketini en alt katmana yerleştir
                    self.background_label.lower()
                    
                    # Pencere boyutu değiştiğinde resmi yeniden boyutlandırmak için
                    self.resizeEvent = self.on_resize
                    
                    logger.info(f"Arka plan resmi başarıyla ayarlandı: {background_path}")
                else:
                    logger.error(f"Resim yüklenemedi. Yükleme başarısı: {load_success}, Boş mu: {background_pixmap.isNull()}")
            except Exception as e:
                logger.error(f"Arka plan resmi yüklenirken hata: {e}", exc_info=True)
        else:
            logger.warning(f"Arka plan resmi bulunamadı veya geçersiz: {background_path}")
            
        # Stil ayarları - QSS dosyasından yükleniyor, burada ek stil gerekmez
        
        # Gölge efektlerini uygula
        self.apply_shadow_effects()
        
        # Arka plan resmi artık QSS üzerinden yükleniyor
        # self.setup_background_image()
        
        # Ana düzen yapısı - modern ve organize görünüm için üç ana bölüm
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20) # Daha geniş kenar boşlukları
        self.main_layout.setSpacing(15) # Elemanlar arası boşluk
        
        # Üç ana bölüm oluştur
        self.top_section = QWidget()
        self.top_section.setObjectName("top_section")
        self.top_layout = QHBoxLayout(self.top_section)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setSpacing(15)
        
        self.middle_section = QWidget()
        self.middle_section.setObjectName("middle_section")
        self.middle_layout = QVBoxLayout(self.middle_section)
        self.middle_layout.setContentsMargins(0, 0, 0, 0)
        self.middle_layout.setSpacing(15)
        
        self.bottom_section = QWidget()
        self.bottom_section.setObjectName("bottom_section")
        self.bottom_layout = QVBoxLayout(self.bottom_section)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.setSpacing(15)
        
        # Ana bölümler için boyut politikaları
        self.top_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.middle_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bottom_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Minimum yükseklikleri ayarla
        self.top_section.setMinimumHeight(100)
        self.middle_section.setMinimumHeight(400)
        self.bottom_section.setMinimumHeight(300)
        
        # Bölümleri ana düzene ekle
        self.main_layout.addWidget(self.top_section)
        self.main_layout.addWidget(self.middle_section)
        self.main_layout.addWidget(self.bottom_section)
        
        # --- MIDI Dosya Seçimi Bölümü ---
        # MIDI Dosya Seçimi grup kutusu
        self.midi_file_group = QGroupBox("MIDI Dosya Seçimi")
        self.midi_file_group.setObjectName("midi_file_group")
        self.midi_file_layout = QHBoxLayout(self.midi_file_group)
        self.midi_file_layout.setContentsMargins(15, 25, 15, 15)
        self.midi_file_layout.setSpacing(10)
        
        # Dosya yolu gösterimi
        self.midi_path_edit = QLineEdit()
        self.midi_path_edit.setPlaceholderText("MIDI dosyası seçin...")
        self.midi_path_edit.setReadOnly(True)
        
        # Göz at butonu - yardımcı fonksiyon kullan
        self.browse_button = create_styled_button(
            text="Göz At",
            object_name="browse_button",
            fixed_width=100,
            connect_to=self.browse_midi_file
        )
        
        # Analiz butonu - yardımcı fonksiyon kullan
        self.analyze_button = create_styled_button(
            text="Analiz Et",
            object_name="analyze_button",
            fixed_width=100,
            connect_to=self.on_analyze_button_clicked
        )
        
        # Öğeleri düzene ekle
        self.midi_file_layout.addWidget(self.midi_path_edit, 1)
        self.midi_file_layout.addWidget(self.browse_button, 0)
        self.midi_file_layout.addWidget(self.analyze_button, 0)
        
        # Grup kutusunu üst bölüme ekle
        self.top_layout.addWidget(self.midi_file_group)

        # MIDI Analizi Görüntüleme Bölümü - Daha kompakt


        # MIDI Analizi Görüntüleme Bölümü - Daha kompakt
        self.analysis_group = QGroupBox("MIDI Analizi")
        self.analysis_group.setObjectName("analysis_group") # QSS için objectName
        self.analysis_group.setMinimumHeight(220) # Minimum yükseklik daha da azaltıldı
        # Expanding boyut politikası - genişleyebilir
        self.analysis_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Analiz grubu için ana dikey layout - daha kompakt
        self.analysis_layout = QVBoxLayout(self.analysis_group)
        self.analysis_layout.setContentsMargins(8, 8, 8, 8) # Kenar boşlukları daha da azaltıldı
        self.analysis_layout.setSpacing(5) # Elemanlar arası boşluk daha da azaltıldı
        
        # Analiz grubu için yatay layout (sol taraf ve piyano rulosu yan yana)
        self.analysis_details_layout = QHBoxLayout()
        self.analysis_details_layout.setSpacing(8) # Elemanlar arası boşluk azaltıldı
        
        # Sol taraf için dikey layout (analiz metni ve butonlar)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(5) # Elemanlar arası boşluk daha da azaltıldı
        
        # Analiz metin alanı - daha kompakt
        self.analysis_text_edit = QPlainTextEdit() # Düz metin için PlainTextEdit
        self.analysis_text_edit.setReadOnly(True) # Salt okunur
        self.analysis_text_edit.setObjectName("analysis_text_edit") # QSS için objectName
        self.analysis_text_edit.setMinimumWidth(220) # Minimum genişlik daha da azaltıldı
        self.analysis_text_edit.setMinimumHeight(150) # Minimum yükseklik daha da azaltıldı
        # Expanding boyut politikası - genişleyebilir
        self.analysis_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Yazı tipi boyutunu ayarla - daha küçük
        analysis_font = QFont()
        analysis_font.setPointSize(9) # Yazı tipi boyutu azaltıldı
        self.analysis_text_edit.setFont(analysis_font)
        
        # Analiz butonları için yatay layout - daha kompakt
        button_layout_analysis = QHBoxLayout()
        button_layout_analysis.setSpacing(8) # Butonlar arası boşluk daha da azaltıldı
        button_layout_analysis.setContentsMargins(0, 3, 0, 0) # Boşluklar daha da azaltıldı
        
        # Analiz butonu - yardımcı fonksiyon kullan
        # Hafızalarda belirtildiği gibi daha belirgin ve kullanışlı butonlar
        button_font = QFont()
        button_font.setPointSize(10)  # Daha büyük yazı tipi
        button_font.setBold(True)
        
        self.btn_analyze = create_styled_button(
            text="Analiz",
            object_name="btn_analyze",
            min_width=120,  # Daha geniş buton
            min_height=30,  # Daha yüksek buton
            font=button_font,
            connect_to=self.analyze_current_midi_file
        )
        button_layout_analysis.addWidget(self.btn_analyze)
        
        # Eğitim butonu - yardımcı fonksiyon kullan
        self.btn_train = create_styled_button(
            text="Eğit",
            object_name="btn_train",
            min_width=120,  # Daha geniş buton
            min_height=30,  # Daha yüksek buton
            font=button_font,
            connect_to=self.start_training
        )
        button_layout_analysis.addWidget(self.btn_train)
        
        # Boşluk ekleyerek butonları sola yasla
        button_layout_analysis.addStretch(1)
        
        # Metin alanı ve butonları sol layout'a ekle
        left_layout.addWidget(self.analysis_text_edit, 1) # Stretch faktörü 1
        left_layout.addLayout(button_layout_analysis) # Stretch faktörü yok
        
        # Piyano rulosu görüntüsü - yardımcı fonksiyon kullan
        # Hafızalarda belirtildiği gibi daha büyük ve belirgin piyano roll görüntüsü
        self.piano_roll_label = create_styled_label(
            text="MIDI dosyası yüklenip analiz edildiğinde\npiyano rulosu burada görüntülenecektir",
            object_name="piano_roll_label",
            alignment=Qt.AlignmentFlag.AlignCenter,
            min_width=400,  # Minimum genişlik
            min_height=250  # Minimum yükseklik
        )
        # Expanding boyut politikası - genişleyebilir
        self.piano_roll_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Piyano rulosu için stil - boş alan daha belirgin
        self.piano_roll_label.setStyleSheet("""
            QLabel#piano_roll_label {
                background-color: #2C3E50;
                color: #BDC3C7;
                border: 2px dashed #7F8C8D;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        
        # Sol layout ve piyano rulosunu ana yatay layout'a ekle
        self.analysis_details_layout.addLayout(left_layout, 1) # Stretch faktörü 1
        self.analysis_details_layout.addWidget(self.piano_roll_label, 2) # Stretch faktörü 2
        
        # Detaylar layout'unu ana layout'a ekle
        self.analysis_layout.addLayout(self.analysis_details_layout)

        self.top_layout.addWidget(self.analysis_group) # Add the analysis group box to the top layout [cite: 97]


        # --- MIDI Üretimi Bölümü ---
        # MIDI Üretimi grup kutusu
        self.midi_uretim_group = QGroupBox("MIDI Üretimi")
        self.midi_uretim_group.setObjectName("midi_uretim_group")
        self.midi_uretim_layout = QGridLayout(self.midi_uretim_group)
        self.midi_uretim_layout.setContentsMargins(15, 25, 15, 15)
        self.midi_uretim_layout.setHorizontalSpacing(15)
        self.midi_uretim_layout.setVerticalSpacing(15)
        
        # Parametre etiketleri ve giriş alanları
        # Ölçü Sayısı
        self.bar_count_label = create_styled_label(
            text="Ölçü Sayısı:",
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.bar_count_spin = QSpinBox()
        self.bar_count_spin.setRange(1, 16)
        self.bar_count_spin.setValue(4)
        self.bar_count_spin.setFixedWidth(150)
        
        # Tempo
        self.tempo_label = create_styled_label(
            text="Tempo (BPM):",
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(40, 240)
        self.tempo_spin.setValue(120)
        self.tempo_spin.setFixedWidth(150)
        
        # Yaratıcılık
        self.creativity_label = create_styled_label(
            text="Yaratıcılık (0.1-2.0):",
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.creativity_spin = QDoubleSpinBox()
        self.creativity_spin.setRange(0.1, 2.0)
        self.creativity_spin.setValue(1.0)
        self.creativity_spin.setSingleStep(0.1)
        self.creativity_spin.setFixedWidth(150)
        
        # Müzik Stili
        self.style_label = create_styled_label(
            text="Müzik Stili:",
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Otomatik", "Pop", "Rock", "Klasik", "Jazz", "Electronic"])
        self.style_combo.setFixedWidth(150)
        
        # Parametreleri grid düzene ekle
        self.midi_uretim_layout.addWidget(self.bar_count_label, 0, 0)
        self.midi_uretim_layout.addWidget(self.bar_count_spin, 0, 1)
        self.midi_uretim_layout.addWidget(self.tempo_label, 1, 0)
        self.midi_uretim_layout.addWidget(self.tempo_spin, 1, 1)
        self.midi_uretim_layout.addWidget(self.creativity_label, 2, 0)
        self.midi_uretim_layout.addWidget(self.creativity_spin, 2, 1)
        self.midi_uretim_layout.addWidget(self.style_label, 3, 0)
        self.midi_uretim_layout.addWidget(self.style_combo, 3, 1)
        
        # Üret butonu - yardımcı fonksiyon kullan
        # Hafızalarda belirtildiği gibi daha büyük ve belirgin olmalı
        generate_font = QFont()
        generate_font.setPointSize(14)  # Daha büyük yazı tipi
        generate_font.setBold(True)    # Kalın yazı tipi
        
        self.generate_button = create_styled_button(
            text="MIDI ÜRET",  # Büyük harflerle daha dikkat çekici
            object_name="generate_button",
            fixed_size=(250, 70),  # Daha büyük buton
            font=generate_font,
            connect_to=self.generate_midi_sequence
        )
        
        # Buton için ortalanmış bir konteyner
        self.button_container = QWidget()
        self.button_layout = QHBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(0, 10, 0, 0)
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.generate_button)
        self.button_layout.addStretch(1)
        
        # Buton konteynerini grid'e ekle
        self.midi_uretim_layout.addWidget(self.button_container, 4, 0, 1, 2)
        
        # Grup kutusunu orta bölüme ekle
        self.middle_layout.addWidget(self.midi_uretim_group)
        
        # --- Hafıza ve Ayarlar Bölümü ---
        # Tab widget - Hafızalarda belirtildiği gibi daha belirgin ve kullanımı kolay
        self.bottom_tab_widget = QTabWidget()
        self.bottom_tab_widget.setObjectName("bottom_tab_widget")
        self.bottom_tab_widget.setDocumentMode(True)
        self.bottom_tab_widget.setMovable(True)
        # Minimum boyut ayarla - Hafızalarda belirtilen sorunları çözmek için
        self.bottom_tab_widget.setMinimumHeight(350)  # Daha yüksek minimum yükseklik
        self.bottom_tab_widget.setMinimumWidth(600)   # Daha geniş minimum genişlik
        
        # Hafıza tab'ı
        self.memory_tab = QWidget()
        self.memory_tab.setObjectName("memory_tab")
        self.memory_layout = QVBoxLayout(self.memory_tab)
        self.memory_layout.setContentsMargins(20, 20, 20, 20)  # Daha fazla kenar boşluğu
        self.memory_layout.setSpacing(20)  # Elemanlar arası daha fazla boşluk
        
        # Hafıza listesi başlığı - yardımcı fonksiyon kullan
        self.memory_list_label = create_styled_label(
            text="Hafızadaki MIDI Desenleri",
            object_name="memory_list_label",
            alignment=Qt.AlignmentFlag.AlignCenter,
            min_height=30  # Daha yüksek etiket
        )
        
        # Hafıza listesi
        self.memory_patterns_list = QListWidget()
        self.memory_patterns_list.setObjectName("memory_patterns_list")
        self.memory_patterns_list.setAlternatingRowColors(True)
        self.memory_patterns_list.setMinimumHeight(250)  # Daha yüksek liste
        # Boyut politikası - MinimumExpanding kullan
        self.memory_patterns_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        
        # Hafıza butonları için konteyner
        self.memory_buttons_container = QWidget()
        self.memory_buttons_layout = QHBoxLayout(self.memory_buttons_container)
        self.memory_buttons_layout.setContentsMargins(0, 10, 0, 0)  # Üst kısımda daha fazla boşluk
        self.memory_buttons_layout.setSpacing(20)  # Butonlar arası daha fazla boşluk
        
        # Hafızaya Ekle butonu - yardımcı fonksiyon kullan
        self.add_to_memory_button = create_styled_button(
            text="Hafızaya Ekle",
            object_name="add_to_memory_button",
            min_width=150,  # Daha geniş buton
            min_height=40,  # Daha yüksek buton
            connect_to=self.add_to_memory
        )
        
        # Benzer Ara butonu - yardımcı fonksiyon kullan
        self.search_memory_button = create_styled_button(
            text="Benzer Ara",
            object_name="search_memory_button",
            min_width=150,  # Daha geniş buton
            min_height=40,  # Daha yüksek buton
            connect_to=self.search_memory
        )
        
        # Butonları konteyner'a ekle
        self.memory_buttons_layout.addStretch(1)
        self.memory_buttons_layout.addWidget(self.add_to_memory_button)
        self.memory_buttons_layout.addWidget(self.search_memory_button)
        self.memory_buttons_layout.addStretch(1)
        
        # Öğeleri hafıza düzene ekle
        self.memory_layout.addWidget(self.memory_list_label)
        self.memory_layout.addWidget(self.memory_patterns_list)
        self.memory_layout.addWidget(self.memory_buttons_container)
        
        # Ayarlar tab'ı - Hafızalarda belirtilen sorunları çözmek için düzenlemeler
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName("settings_tab")
        self.settings_layout = QVBoxLayout(self.settings_tab)
        self.settings_layout.setContentsMargins(20, 20, 20, 20)  # Daha fazla kenar boşluğu
        self.settings_layout.setSpacing(20)  # Elemanlar arası daha fazla boşluk
        
        # Ayarlar içeriği - Minimum boyut ayarla
        self.settings_text_edit = QTextEdit()
        self.settings_text_edit.setObjectName("settings_text_edit")
        self.settings_text_edit.setReadOnly(True)
        self.settings_text_edit.setMinimumHeight(300)  # Daha yüksek metin alanı
        # Boyut politikası - MinimumExpanding kullan
        self.settings_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.settings_layout.addWidget(self.settings_text_edit)
        
        # Ayarlar butonları için konteyner
        self.settings_buttons_container = QWidget()
        self.settings_buttons_layout = QHBoxLayout(self.settings_buttons_container)
        self.settings_buttons_layout.setContentsMargins(0, 10, 0, 0)  # Üst kısımda daha fazla boşluk
        self.settings_buttons_layout.setSpacing(20)  # Butonlar arası daha fazla boşluk
        
        # Ayarları Kaydet butonu - yardımcı fonksiyon kullan
        self.save_settings_button = create_styled_button(
            text="Ayarları Kaydet",
            object_name="save_settings_button",
            min_width=150,  # Daha geniş buton
            min_height=40,  # Daha yüksek buton
            connect_to=self.save_settings
        )
        
        # Ayarları Yükle butonu - yardımcı fonksiyon kullan
        self.load_settings_button = create_styled_button(
            text="Ayarları Yükle",
            object_name="load_settings_button",
            min_width=150,  # Daha geniş buton
            min_height=40,  # Daha yüksek buton
            connect_to=self.load_settings
        )
        
        # Butonları konteyner'a ekle
        self.settings_buttons_layout.addStretch(1)
        self.settings_buttons_layout.addWidget(self.save_settings_button)
        self.settings_buttons_layout.addWidget(self.load_settings_button)
        self.settings_buttons_layout.addStretch(1)
        
        # Butonları ayarlar düzene ekle
        self.settings_layout.addWidget(self.settings_buttons_container)
        
        # Tab'ları tab widget'a ekle
        self.bottom_tab_widget.addTab(self.memory_tab, "Hafıza ve Desenler")
        self.bottom_tab_widget.addTab(self.settings_tab, "Ayarlar")
        
        # Tab widget'ı alt bölüme ekle
        self.bottom_layout.addWidget(self.bottom_tab_widget)
        
        # Etiketler için stil - modern ve belirgin
        label_style = """font-weight: bold; 
                        color: #FFFFFF; 
                        font-size: 12pt; 
                        background-color: rgba(41, 128, 185, 0.7); 
                        padding: 8px 12px; 
                        border-radius: 6px; 
                        margin-right: 15px;
                        min-width: 180px;
                        text-align: left;
                        border: 1px solid rgba(41, 128, 185, 0.9);
                     """
        
        # Bar/Ölçü sayısı ayarı
        self.bar_count_label = QLabel("Ölçü Sayısı:")
        self.bar_count_label.setStyleSheet(label_style)
        self.bar_count_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) # Sola hizala
        # Fixed boyut politikası - etiket boyutunun sabit kalması için
        self.bar_count_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.bar_count_label.setMinimumWidth(200) # Minimum genişlik belirlendi
        
        # SpinBox - modern görünüm
        self.bar_count_spinbox = QSpinBox()
        self.bar_count_spinbox.setRange(1, 32)  # 1-32 bar arası
        self.bar_count_spinbox.setObjectName("bar_count_spinbox")
        self.bar_count_spinbox.setValue(4)  # Varsayılan 4 bar
        self.bar_count_spinbox.setSuffix(" bar")  # Birim ekle
        self.bar_count_spinbox.setMinimumWidth(180) # Minimum genişlik daha da artırıldı
        self.bar_count_spinbox.setMinimumHeight(40) # Minimum yükseklik daha da artırıldı
        
        # SpinBox için modern stil
        self.bar_count_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #2C3E50;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #3498DB;
                font-weight: bold;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3498DB;
                border-radius: 4px;
                width: 20px;
                height: 15px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #2980B9;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background-color: #1B4F72;
            }
        """)
        
        # Yazı tipi
        spinbox_font = QFont()
        spinbox_font.setPointSize(10)
        spinbox_font.setBold(True)
        self.bar_count_spinbox.setFont(spinbox_font)
        
        # Fixed boyut politikası - spinbox boyutunun sabit kalması için
        self.bar_count_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Tempo ayarı
        self.tempo_label = QLabel("Tempo (BPM):")
        self.tempo_label.setStyleSheet(label_style)
        self.tempo_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) # Sola hizala
        self.tempo_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.tempo_label.setMinimumWidth(180) # Minimum genişlik belirlendi
        
        # Tempo SpinBox - modern görünüm
        self.tempo_spinbox = QSpinBox()
        self.tempo_spinbox.setRange(40, 240)  # 40-240 BPM arası
        self.tempo_spinbox.setObjectName("tempo_spinbox")
        self.tempo_spinbox.setValue(120)  # Varsayılan 120 BPM
        self.tempo_spinbox.setSuffix(" BPM")  # Birim ekle
        self.tempo_spinbox.setMinimumWidth(180) # Minimum genişlik daha da artırıldı
        self.tempo_spinbox.setMinimumHeight(40) # Minimum yükseklik daha da artırıldı
        
        # SpinBox için modern stil - tutarlılık için aynı stil
        self.tempo_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #2C3E50;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #3498DB;
                font-weight: bold;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3498DB;
                border-radius: 4px;
                width: 20px;
                height: 15px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #2980B9;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background-color: #1B4F72;
            }
        """)
        
        # Yazı tipi
        self.tempo_spinbox.setFont(spinbox_font)
        
        # Genişleyebilir boyut politikası
        self.tempo_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Yaratıcılık ayarı
        self.temperature_label = QLabel("Yaratıcılık (0.1-2.0):")
        self.temperature_label.setStyleSheet(label_style)
        self.temperature_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) # Sola hizala
        self.temperature_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.temperature_label.setMinimumWidth(220) # Minimum genişlik belirlendi
        
        # Yaratıcılık double spinbox - modern görünüm
        self.temperature_double_spinbox = QDoubleSpinBox()
        self.temperature_double_spinbox.setRange(0.1, 2.0)
        self.temperature_double_spinbox.setSingleStep(0.1)
        self.temperature_double_spinbox.setValue(1.0)
        self.temperature_double_spinbox.setObjectName("temperature_double_spinbox")
        self.temperature_double_spinbox.setToolTip("Düşük değerler daha tahmin edilebilir, yüksek değerler daha yaratıcı sonuçlar üretir")
        self.temperature_double_spinbox.setMinimumWidth(180) # Minimum genişlik daha da artırıldı
        self.temperature_double_spinbox.setMinimumHeight(40) # Minimum yükseklik daha da artırıldı
        
        # DoubleSpinBox için modern stil - tutarlılık için aynı stil
        self.temperature_double_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2C3E50;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #3498DB;
                font-weight: bold;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #3498DB;
                border-radius: 4px;
                width: 20px;
                height: 15px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #2980B9;
            }
            QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #1B4F72;
            }
        """)
        
        # Yazı tipi
        self.temperature_double_spinbox.setFont(spinbox_font) # Tutarlı yazı tipi
        self.temperature_double_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Stil seçimi
        self.style_label = QLabel("Müzik Stili:")
        self.style_label.setStyleSheet(label_style)
        self.style_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) # Sola hizala
        self.style_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.style_label.setMinimumWidth(200) # Minimum genişlik belirlendi
        
        # Stil combo box - modern görünüm
        self.style_combo = QComboBox()
        self.style_combo.setObjectName("style_combo")
        self.style_combo.addItems(["Otomatik", "Pop", "Rock", "Klasik", "Jazz", "Hip-Hop", "Elektronik"])
        self.style_combo.setMinimumWidth(200) # Minimum genişlik daha da artırıldı
        self.style_combo.setMinimumHeight(40) # Minimum yükseklik daha da artırıldı
        
        # ComboBox için modern stil - tutarlılık için benzer stil
        self.style_combo.setStyleSheet("""
            QComboBox {
                background-color: #2C3E50;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 8px;
                padding: 8px;
                min-width: 6em;
                font-weight: bold;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left: 1px solid #3498DB;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox::down-arrow {
                width: 14px;
                height: 14px;
            }
            QComboBox:hover {
                border: 1px solid #2980B9;
            }
            QComboBox QAbstractItemView {
                background-color: #2C3E50;
                color: white;
                selection-background-color: #3498DB;
                selection-color: white;
                border: 1px solid #3498DB;
                border-radius: 0px 0px 8px 8px;
                padding: 5px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 25px;
                padding: 5px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #34495E;
            }
        """)
        
        # Yazı tipi - tutarlılık için
        self.style_combo.setFont(spinbox_font) # Tutarlı yazı tipi
        self.style_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Elemanları grid layout'a ekle
        self.midi_uretim_layout.addWidget(self.bar_count_label, 0, 0)
        self.midi_uretim_layout.addWidget(self.bar_count_spinbox, 0, 1)
        
        self.midi_uretim_layout.addWidget(self.tempo_label, 1, 0)
        self.midi_uretim_layout.addWidget(self.tempo_spinbox, 1, 1)
        
        self.midi_uretim_layout.addWidget(self.temperature_label, 2, 0)
        self.midi_uretim_layout.addWidget(self.temperature_double_spinbox, 2, 1)
        
        self.midi_uretim_layout.addWidget(self.style_label, 3, 0)
        self.midi_uretim_layout.addWidget(self.style_combo, 3, 1)
        
        # MIDI Üretim butonu - Modern, dikkat çekici ve profesyonel tasarım
        self.generate_button = QPushButton("MIDI ÜRET")
        self.generate_button.setObjectName("generate_button")
        self.generate_button.setMinimumHeight(70)  # Yükseklik daha da artırıldı
        self.generate_button.setMinimumWidth(350) # Genişlik daha da artırıldı
        
        # Yazı tipi - Daha büyük ve belirgin
        generate_font = QFont()
        generate_font.setPointSize(18) # Yazı tipi boyutu daha da artırıldı
        generate_font.setBold(True)   # Kalın yazı tipi
        self.generate_button.setFont(generate_font)
        
        # Özel stil - Daha profesyonel ve modern görünüm
        self.generate_button.setStyleSheet("""
            QPushButton#generate_button {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ECC71, stop:1 #27AE60); /* Gradyan arka plan */
                color: white;
                border-radius: 12px;
                padding: 15px;
                margin-top: 25px;
                border: none;
                font-weight: bold;
                letter-spacing: 1px; /* Harfler arası boşluk */
            }
            QPushButton#generate_button:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27AE60, stop:1 #219653); /* Hover durumunda daha koyu gradyan */
                color: #FFFFFF;
            }
            QPushButton#generate_button:pressed {
                background: #1E8449; /* Basılı durumda düz renk */
                padding: 17px 13px 13px 17px; /* Basılı efekti */
            }
        """)
        
        # Butonun boyut politikası - KRİTİK: Sabit boyut
        self.generate_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed) # Sabit boyut - asla küçülmeyecek
        self.generate_button.clicked.connect(self.generate_midi_sequence)
        
        # KRİTİK: Buton için daha fazla boşluk - üst kısım
        spacer_item = QSpacerItem(10, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.midi_uretim_layout.addItem(spacer_item, 4, 0, 1, 2)
        
        # KRİTİK: Buton için ayrı bir hücre, iki sütunu kaplayacak şekilde ve ortalanmış
        self.midi_uretim_layout.addWidget(self.generate_button, 5, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        
        # KRİTİK: Butonun altına boşluk ekleyerek sıkışmasını önle
        bottom_spacer = QSpacerItem(10, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.midi_uretim_layout.addItem(bottom_spacer, 6, 0, 1, 2)
        
        # ÇOK KRİTİK: Satır stretch faktörlerini optimize et
        # Tüm satırlara 0 stretch faktörü ver - sabit kalsınlar
        for i in range(7):
            self.midi_uretim_layout.setRowStretch(i, 0)
            
        # ÇOK KRİTİK: SADECE butondan sonraki satıra yüksek stretch faktörü ver
        # Bu, butonun ve parametrelerin görünür kalmasını sağlar
        self.midi_uretim_layout.setRowStretch(7, 5) # Butondan sonraki boş satıra çok yüksek stretch faktörü
        
        # Üretim grubunu ana layout'a ekle
        self.top_layout.addWidget(self.midi_uretim_group)
        
        # KRİTİK: Üst bölümün en altına stretch ekle - içeriğin yukarıya itilmesini sağlar
        self.top_layout.addStretch(1)
        
        # --- Alt Bölüm (Hafıza ve Ayarlar) - Tab yapısı ---
        self.bottom_widget = QWidget()
        self.bottom_widget.setObjectName("bottom_widget") # QSS için objectName
        # Fixed - alt bölüm sabit boyutta kalsın
        self.bottom_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.bottom_widget.setMinimumHeight(350) # Minimum yükseklik artırıldı
        
        # Alt bölüm için dikey layout
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0) # Kenar boşluklarını kaldır - tab widget tam dolduracak
        self.bottom_layout.setSpacing(0) # Boşlukları kaldır
        
        # Alt bölümü ana düzene ekle - splitter yerine doğrudan ana düzene ekliyoruz
        self.main_layout.addWidget(self.bottom_widget)
        
        # Tab Widget - daha kompakt modern görünüm
        self.bottom_tab_widget = QTabWidget()
        self.bottom_tab_widget.setObjectName("bottom_tab_widget") # QSS için objectName
        self.bottom_tab_widget.setMinimumHeight(320) # Minimum yükseklik azaltıldı
        self.bottom_tab_widget.setDocumentMode(True) # Daha modern görünüm
        self.bottom_tab_widget.setTabPosition(QTabWidget.TabPosition.North) # Tabları üste yerleştir
        self.bottom_tab_widget.setMovable(True) # Tabların yerini değiştirilebilir yap
        self.bottom_tab_widget.setTabsClosable(False) # Kapanabilir tabları devre dışı bırak
        
        # Tab yazı tipi - daha küçük
        tab_font = QFont()
        tab_font.setPointSize(10) # Yazı tipi boyutu azaltıldı
        tab_font.setBold(True)
        self.bottom_tab_widget.setFont(tab_font)
        
        # Tab widget'ı alt bölüme ekle - tam olarak doldursun
        self.bottom_layout.addWidget(self.bottom_tab_widget)
        
        # --- Hafıza/Desenler Tab'ı ---
        self.memory_tab = QWidget()
        self.memory_tab.setObjectName("memory_tab") # QSS için objectName
        # Tab için minimum boyut ve boyut politikası
        self.memory_tab.setMinimumWidth(700)  # Minimum genişlik azaltıldı
        self.memory_tab.setMinimumHeight(300) # Minimum yükseklik azaltıldı
        self.memory_tab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Tab'a ikon ekle
        memory_icon = QIcon(os.path.join(project_root_path, 'resources', 'icons', 'memory.svg'))
        if not memory_icon.isNull():
            self.bottom_tab_widget.addTab(self.memory_tab, memory_icon, "Hafıza ve Desenler")
        else:
            self.bottom_tab_widget.addTab(self.memory_tab, "Hafıza ve Desenler")
        
        # ÇOK KRİTİK: Hafıza/Desenler için dikey layout - daha iyi kenar boşlukları
        self.memory_layout = QVBoxLayout(self.memory_tab)
        self.memory_layout.setContentsMargins(20, 20, 20, 20) # Kenar boşlukları
        self.memory_layout.setSpacing(20) # Elemanlar arası boşluk daha da artırıldı
        
        # Görsel hata ayıklama için geçici kenar ve arka plan rengi
        self.memory_tab.setStyleSheet("""
            QWidget#memory_tab {
                border: 3px solid #9B59B6; /* Mor kenar - daha belirgin */
                background-color: rgba(155, 89, 182, 20); /* Hafif saydam arka plan */
            }
        """)
        
        # --- Ayarlar Tab'ı ---
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName("settings_tab") # QSS için objectName
        # Tab için minimum boyut ve boyut politikası
        self.settings_tab.setMinimumWidth(700)  # Minimum genişlik
        self.settings_tab.setMinimumHeight(350) # Minimum yükseklik artırıldı
        # Fixed boyut politikası - sabit boyutta kalsın
        self.settings_tab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Tab'a ikon ekle
        settings_icon = QIcon(os.path.join(project_root_path, 'resources', 'icons', 'settings.svg'))
        if not settings_icon.isNull():
            self.bottom_tab_widget.addTab(self.settings_tab, settings_icon, "Ayarlar")
        else:
            self.bottom_tab_widget.addTab(self.settings_tab, "Ayarlar")
        
        # Ayarlar için dikey layout - daha iyi kenar boşlukları
        self.settings_layout = QVBoxLayout(self.settings_tab)
        self.settings_layout.setContentsMargins(20, 20, 20, 20) # Kenar boşlukları artırıldı
        self.settings_layout.setSpacing(15) # Elemanlar arası boşluk artırıldı

        # Hafıza/Desenler Tab'ı içeriği - Daha kompakt yapı
        # 1. Hafıza Başlığı - Modern ve belirgin
        self.memory_header_label = QLabel("Hafızadaki MIDI Desenleri")
        self.memory_header_label.setObjectName("memory_header_label") # QSS için objectName
        
        # Başlık için özel yazı tipi
        memory_header_font = QFont()
        memory_header_font.setPointSize(14) # Daha büyük yazı tipi
        memory_header_font.setBold(True)
        self.memory_header_label.setFont(memory_header_font)
        self.memory_header_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Merkeze hizala
        self.memory_header_label.setMinimumHeight(40) # Minimum yükseklik artırıldı
        
        # Başlık için modern stil
        self.memory_header_label.setStyleSheet("""
            QLabel#memory_header_label {
                color: white;
                background-color: #3498DB;
                border-radius: 6px;
                padding: 8px;
                margin-bottom: 10px;
                font-weight: bold;
            }
        """)
        
        self.memory_layout.addWidget(self.memory_header_label)
        
        # Hafıza Listesi - Modern ve belirgin görünüm
        self.memory_patterns_list = QListWidget()
        self.memory_patterns_list.setObjectName("memory_patterns_list") # QSS için objectName
        self.memory_patterns_list.setMinimumHeight(250) # Minimum yükseklik daha da artırıldı
        self.memory_patterns_list.setAlternatingRowColors(True) # Alternatif satır renkleri
        self.memory_patterns_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection) # Çoklu seçim
        self.memory_patterns_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove) # Sürükle-bırak
        
        # Liste için boyut politikası - dikey olarak genişleyebilir
        self.memory_patterns_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.memory_patterns_list.setMinimumHeight(250)  # Minimum yükseklik belirlendi
        
        # Liste için boş durum mesajı - QListWidget için özel yöntem
        # QListWidget setPlaceholderText desteklemediği için boş durumu kontrol eden bir fonksiyon ekleyeceğiz
        self.empty_list_label = QLabel("Hafızada kayıtlı MIDI deseni bulunmamaktadır\n\nYeni üretilen MIDI'leri hafızaya eklemek için\n'Hafızaya Ekle' butonunu kullanın")
        self.empty_list_label.setObjectName("empty_list_label")
        self.empty_list_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_list_label.setWordWrap(True)
        self.empty_list_label.setStyleSheet("""
            QLabel#empty_list_label {
                color: #7F8C8D;
                font-style: italic;
                font-size: 11pt;
                padding: 20px;
            }
        """)
        
        # Boş liste mesajını göstermek için bir layout oluştur
        self.memory_list_container = QStackedWidget()
        self.memory_list_container.addWidget(self.memory_patterns_list)  # 0. indeks
        self.memory_list_container.addWidget(self.empty_list_label)      # 1. indeks
        
        # Başlangıçta boş ise mesajı göster
        if self.midi_memory and self.midi_memory.get_all_patterns() and len(self.midi_memory.get_all_patterns()) > 0:
            self.memory_list_container.setCurrentIndex(0)  # Liste göster
        else:
            self.memory_list_container.setCurrentIndex(1)  # Boş mesajı göster
        
        # Liste yazı tipi - daha okunakıl
        list_font = QFont()
        list_font.setPointSize(11)  # Yazı tipi boyutu artırıldı
        list_font.setBold(True)  # Kalın yazı tipi
        self.memory_patterns_list.setFont(list_font)
        
        # Liste için modern ve profesyonel stil
        self.memory_patterns_list.setStyleSheet("""
            QListWidget#memory_patterns_list {
                border: 1px solid #3498DB; /* İnce mavi kenar */
                border-radius: 10px; /* Daha yuvarlak köşeler */
                padding: 12px; /* Daha fazla iç boşluk */
                background-color: #1E2B3C; /* Daha koyu arka plan */
            }
            QListWidget#memory_patterns_list::item {
                padding: 12px 15px; /* Daha fazla iç boşluk */
                border-radius: 6px; /* Yuvarlak köşeler */
                margin: 5px 0; /* Üst ve alt kenar boşluğu artırıldı */
                color: white; /* Beyaz yazı rengi */
                background-color: #2C3E50; /* Öğe arka plan rengi */
                border-left: 3px solid #3498DB; /* Sol kenar vurgusu */
            }
            QListWidget#memory_patterns_list::item:selected {
                background-color: #3498DB; /* Mavi seçim rengi */
                color: white;
                font-weight: bold;
                border-left: 3px solid #2ECC71; /* Yeşil sol kenar vurgusu */
            }
            QListWidget#memory_patterns_list::item:hover {
                background-color: #34495E; /* Hover durumunda daha açık renk */
                color: white;
                border-left: 3px solid #F1C40F; /* Sarı sol kenar vurgusu */
            }
            QListWidget#memory_patterns_list[count="0"] {
                color: #7F8C8D; /* Boş liste durumunda daha belirgin yazı rengi */
                font-style: italic;
                font-size: 11pt; /* Boş durum mesajı için daha büyük yazı tipi */
            }
        """)
        
        # 2. Önce liste container'ı ekle - daha fazla alan ver
        self.memory_layout.addWidget(self.memory_list_container, 3) # Stretch faktörü 3
        
        # 3. ÇOK KRİTİK: Liste ile butonlar arasına daha fazla boşluk ekle
        self.memory_layout.addSpacing(30) # Daha fazla boşluk
        
        # 4. Hafıza Kontrolleri - Butonlar için ayrı bir layout
        # Modern görünüm için buton container'ı oluştur
        self.memory_buttons_container = QWidget()
        self.memory_buttons_container.setObjectName("memory_buttons_container")
        self.memory_buttons_container.setMinimumHeight(80) # Minimum yükseklik artırıldı
        self.memory_buttons_container.setStyleSheet("""
            QWidget#memory_buttons_container {
                border: 2px solid #3498DB; /* Mavi kenar - tutarlılık için */
                background-color: rgba(52, 152, 219, 0.1); /* Hafif saydam arka plan */
                border-radius: 8px;
                padding: 5px;
            }
        """)
        
        # Buton container'a yatay layout ekle - daha iyi boşluklar
        self.memory_controls_layout = QHBoxLayout(self.memory_buttons_container)
        self.memory_controls_layout.setSpacing(25) # Butonlar arası boşluk artırıldı
        self.memory_controls_layout.setContentsMargins(15, 15, 15, 15) # Kenar boşlukları artırıldı
        
        # Butonlar için ortak yazı tipi - daha belirgin
        memory_button_font = QFont()
        memory_button_font.setPointSize(11) # Yazı tipi boyutu artırıldı
        memory_button_font.setBold(True)
        
        # Butonları ortalamak için sol tarafta boşluk
        self.memory_controls_layout.addStretch(1)
        
        # Hafızaya Ekle butonu - Modern ve belirgin
        self.add_to_memory_button = QPushButton("Hafızaya Ekle")
        self.add_to_memory_button.setObjectName("add_to_memory_button") # QSS için objectName
        self.add_to_memory_button.clicked.connect(self.add_selected_file_to_memory)
        self.add_to_memory_button.setMinimumHeight(50) # Minimum yükseklik daha da artırıldı
        self.add_to_memory_button.setMinimumWidth(200) # Minimum genişlik daha da artırıldı
        self.add_to_memory_button.setFont(memory_button_font)
        self.add_to_memory_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed) # Sabit boyut
        
        # Modern ve profesyonel görünüm için stil
        self.add_to_memory_button.setStyleSheet("""
            QPushButton#add_to_memory_button {
                background-color: #27AE60; /* Yeşil arka plan */
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton#add_to_memory_button:hover {
                background-color: #2ECC71; /* Hover durumunda daha açık yeşil */
            }
            QPushButton#add_to_memory_button:pressed {
                background-color: #1E8449; /* Basılı durumda daha koyu yeşil */
                padding: 11px 9px 9px 11px; /* Basılı efekti */
            }
        """)
        
        # Buton için ikon ekle
        add_icon = QIcon(os.path.join(project_root_path, 'resources', 'icons', 'add.svg'))
        if not add_icon.isNull():
            self.add_to_memory_button.setIcon(add_icon)
            self.add_to_memory_button.setIconSize(QSize(20, 20))
        
        self.memory_controls_layout.addWidget(self.add_to_memory_button)

        # Benzer Desenleri Ara butonu - Modern ve belirgin
        self.search_memory_button = QPushButton("Benzer Ara")
        self.search_memory_button.setObjectName("search_memory_button") # QSS için objectName
        self.search_memory_button.clicked.connect(self.search_similar_patterns)
        self.search_memory_button.setMinimumHeight(50) # Minimum yükseklik daha da artırıldı
        self.search_memory_button.setMinimumWidth(180) # Minimum genişlik daha da artırıldı
        self.search_memory_button.setFont(memory_button_font)
        self.search_memory_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed) # Sabit boyut
        
        # Modern ve profesyonel görünüm için stil
        self.search_memory_button.setStyleSheet("""
            QPushButton#search_memory_button {
                background-color: #3498DB; /* Mavi arka plan */
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton#search_memory_button:hover {
                background-color: #2980B9; /* Hover durumunda daha koyu mavi */
            }
            QPushButton#search_memory_button:pressed {
                background-color: #1B4F72; /* Basılı durumda daha koyu mavi */
                padding: 11px 9px 9px 11px; /* Basılı efekti */
            }
        """)
        
        # Buton için ikon ekle
        search_icon = QIcon(os.path.join(project_root_path, 'resources', 'icons', 'search.svg'))
        if not search_icon.isNull():
            self.search_memory_button.setIcon(search_icon)
            self.search_memory_button.setIconSize(QSize(16, 16)) # İkon boyutu
        
        self.memory_controls_layout.addWidget(self.search_memory_button)
        
        # Butonları ortalamak için sağ tarafta boşluk
        self.memory_controls_layout.addStretch(1)
        
        # 5. ÇOK KRİTİK: Buton container'ı layout'a ekle - liste altına
        self.memory_layout.addWidget(self.memory_buttons_container)
        
        # 6. Benzer Desenler Bölümü ile butonlar arasına KRİTİK: Daha fazla boşluk ekle
        self.memory_layout.addSpacing(35) # Boşluk daha da artırıldı
        
        # Modern görünüm için ayırıcı çizgi
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setObjectName("separator")
        separator.setLineWidth(2) # Çizgi kalınlığı
        separator.setMinimumHeight(20) # Yükseklik artırıldı
        separator.setStyleSheet("""
            QFrame#separator {
                color: #3498DB; /* Mavi renk - tutarlılık için */
                background-color: #3498DB;
                margin: 15px 0px; /* Margin artırıldı */
                border-radius: 2px;
            }
        """)
        self.memory_layout.addWidget(separator)
        
        # Benzer Desenler Başlığı - Modern ve belirgin
        self.similar_patterns_label = QLabel("Bulunan Benzer Desenler")
        self.similar_patterns_label.setObjectName("similar_patterns_label") # QSS için objectName
        
        # Başlık için yazı tipi - daha belirgin
        similar_patterns_font = QFont()
        similar_patterns_font.setPointSize(14) # Yazı tipi boyutu daha da artırıldı
        similar_patterns_font.setBold(True)
        self.similar_patterns_label.setFont(similar_patterns_font)
        self.similar_patterns_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Merkeze hizala
        self.similar_patterns_label.setMinimumHeight(40) # Minimum yükseklik artırıldı
        
        # Başlık için modern stil - tutarlılık için mavi renk
        self.similar_patterns_label.setStyleSheet("""
            QLabel#similar_patterns_label {
                color: white;
                background-color: #3498DB;
                border-radius: 6px;
                padding: 8px;
                margin-bottom: 10px;
                font-weight: bold;
            }
        """)
        
        # 8. Başlığı ekle
        self.memory_layout.addWidget(self.similar_patterns_label)
        
        # Benzer Desenler Metin Alanı - Modern ve okunaklı
        self.similar_patterns_text_edit = QPlainTextEdit()
        self.similar_patterns_text_edit.setObjectName("similar_patterns_text_edit") # QSS için objectName
        self.similar_patterns_text_edit.setReadOnly(True) # Salt okunur
        self.similar_patterns_text_edit.setMinimumHeight(250) # Minimum yükseklik daha da artırıldı
        # Genişleyebilir boyut politikası - dikey olarak genişleyebilir
        self.similar_patterns_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Boş olduğunda bilgilendirici mesaj göster
        self.similar_patterns_text_edit.setPlaceholderText("Benzer desenler aramak için 'Benzer Ara' butonuna tıklayın.")
        
        # Modern ve profesyonel görünüm için stil
        self.similar_patterns_text_edit.setStyleSheet("""
            QPlainTextEdit#similar_patterns_text_edit {
                border: 1px solid #3498DB; /* İnce mavi kenar */
                border-radius: 8px;
                background-color: #2a2a2a;
                color: #FFFFFF; /* Parlak yazı rengi */
                padding: 15px; /* Daha fazla padding */
                font-size: 12pt; /* Daha büyük yazı tipi */
                line-height: 1.5; /* Satır aralığı */
            }
            QPlainTextEdit#similar_patterns_text_edit:focus {
                border: 2px solid #3498DB; /* Odaklandığında daha belirgin kenar */
            }
        """)
        
        # Metin alanı için daha okunablır yazı tipi
        text_font = QFont()
        text_font.setPointSize(11) # Yazı tipi boyutu artırıldı
        self.similar_patterns_text_edit.setFont(text_font)
        
        # Metin alanı için stil zaten yukarıda tanımlandı
        
        # 10. ÇOK KRİTİK: Metin alanını ekle
        self.memory_layout.addWidget(self.similar_patterns_text_edit, 2) # Stretch faktörü 2 - daha fazla alan
        
        # 11. ÇOK KRİTİK: En alta stretch ekle - içeriğin yukarıya itilmesini sağlar
        self.memory_layout.addStretch(1)
        
        # 11. En alta kalan boşluğu doldurmak için stretch ekle
        self.memory_layout.addStretch(1)

        # Populate initial memory patterns list (if memory loaded successfully)
        # This is done after the UI is created
        # self.populate_memory_list() # Call populate after memory is initialized and UI exists


        # Ayarlar Tab'ı içeriği - Modern ve düzenli yapı
        # Ayarlar başlığı
        self.settings_header_label = QLabel("Uygulama Ayarları")
        self.settings_header_label.setObjectName("settings_header_label") # QSS için objectName
        
        # Başlık için özel yazı tipi
        settings_header_font = QFont()
        settings_header_font.setPointSize(13)
        settings_header_font.setBold(True)
        self.settings_header_label.setFont(settings_header_font)
        self.settings_header_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Başlık için özel stil
        self.settings_header_label.setStyleSheet("""
            QLabel#settings_header_label {
                color: #FFFFFF;
                padding: 5px 0px;
                border-bottom: 1px solid #3a3a3a;
                margin-bottom: 10px;
            }
        """)
        
        self.settings_layout.addWidget(self.settings_header_label)
        
        # Açıklama etiketi - Daha iyi okunabilirlik
        self.settings_description_label = QLabel("Aşağıdaki ayarlar uygulama davranışını ve model parametrelerini kontrol eder.")
        self.settings_description_label.setObjectName("settings_description_label")
        self.settings_description_label.setWordWrap(True)
        
        # Açıklama için özel yazı tipi
        description_font = QFont()
        description_font.setPointSize(11)
        self.settings_description_label.setFont(description_font)
        
        # Açıklama için özel stil
        self.settings_description_label.setStyleSheet("""
            QLabel#settings_description_label {
                color: #e0e0e0;
                padding: 5px 0px;
                margin-bottom: 15px;
            }
        """)
        
        self.settings_layout.addWidget(self.settings_description_label)
        
        # Ayarlar metin kutusu - Gelişmiş görünüm
        self.settings_text_edit = QPlainTextEdit() # Ayarları göster
        self.settings_text_edit.setObjectName("settings_text_edit") # QSS için objectName
        self.settings_text_edit.setReadOnly(True) # Salt okunur
        self.settings_text_edit.setMinimumHeight(320) # Minimum yükseklik artırıldı
        
        # Metin kutusu için özel yazı tipi
        settings_font = QFont("Consolas", 11) # Sabit genişlikli yazı tipi - ayarlar için daha uygun
        self.settings_text_edit.setFont(settings_font)
        
        # Metin kutusu için özel stil
        self.settings_text_edit.setStyleSheet("""
            QPlainTextEdit#settings_text_edit {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                padding: 10px;
                background-color: #2a2a2a;
                color: #e0e0e0;
                selection-background-color: #007acc;
            }
        """)
        
        self.settings_layout.addWidget(self.settings_text_edit, 4) # Daha fazla alan
        
        # Ayarlar butonları için container frame
        self.settings_buttons_frame = QFrame()
        self.settings_buttons_frame.setObjectName("settings_buttons_frame")
        self.settings_buttons_frame.setFrameShape(QFrame.Shape.NoFrame)
        
        # Butonlar için yatay layout
        self.settings_buttons_layout = QHBoxLayout(self.settings_buttons_frame)
        self.settings_buttons_layout.setContentsMargins(0, 15, 0, 0) # Üst boşluk artırıldı
        self.settings_buttons_layout.setSpacing(18) # Butonlar arası boşluk artırıldı
        
        # Butonlar için ortak yazı tipi
        settings_button_font = QFont()
        settings_button_font.setPointSize(11)
        settings_button_font.setBold(True)
        
        # Ayarları Kaydet butonu
        self.save_settings_button = QPushButton("Ayarları Kaydet")
        self.save_settings_button.setObjectName("save_settings_button") # QSS için objectName
        # self.save_settings_button.clicked.connect(self.save_settings) # Henüz uygulanmadı
        self.save_settings_button.setMinimumHeight(40) # Minimum yükseklik artırıldı
        self.save_settings_button.setMinimumWidth(200) # Minimum genişlik artırıldı
        self.save_settings_button.setFont(settings_button_font)
        
        # Buton için ikon ekle
        save_icon = QIcon(os.path.join(project_root_path, 'resources', 'icons', 'save.svg'))
        if not save_icon.isNull():
            self.save_settings_button.setIcon(save_icon)
            self.save_settings_button.setIconSize(QSize(20, 20))
        
        # Ayarları Yükle butonu
        self.load_settings_button = QPushButton("Ayarları Yükle")
        self.load_settings_button.setObjectName("load_settings_button") # QSS için objectName
        # self.load_settings_button.clicked.connect(self.load_settings) # Henüz uygulanmadı
        self.load_settings_button.setMinimumHeight(40) # Minimum yükseklik artırıldı
        self.load_settings_button.setMinimumWidth(200) # Minimum genişlik artırıldı
        self.load_settings_button.setFont(settings_button_font)
        
        # Buton için ikon ekle
        load_icon = QIcon(os.path.join(project_root_path, 'resources', 'icons', 'load.svg'))
        if not load_icon.isNull():
            self.load_settings_button.setIcon(load_icon)
            self.load_settings_button.setIconSize(QSize(20, 20))
        
        # Butonları layout'a ekle
        self.settings_buttons_layout.addWidget(self.save_settings_button)
        self.settings_buttons_layout.addWidget(self.load_settings_button)
        self.settings_buttons_layout.addStretch(1) # Butonları sola yasla
        
        # Buton frame'ini ana layout'a ekle
        self.settings_layout.addWidget(self.settings_buttons_frame)

        # Display current settings string using the save(file_path=None) method
        settings_string = "Ayarlar yüklenemedi." # Default message if settings object is None [cite: 110]
        if self.settings:
             try:
                 # Call the save method with file_path=None to get the string representation
                 settings_string_result = self.settings.save(file_path=None)
                 if isinstance(settings_string_result, str): # [cite: 111]
                     settings_string = settings_string_result
                 else:
                      # If save method returns None or other non-string on success
                      settings_string = "Ayarlar başarıyla yüklendi (metin gösterilemedi)." # [cite: 112]
             except Exception as e:
                  logger.error(f"Error getting settings string for display: {e}", exc_info=True)
                  settings_string = f"Ayarları gösterirken hata oluştu: {e}"


        self.settings_text_edit.setPlainText(settings_string)

        # --- Status Bar and Progress ---
        self.statusBar = self.statusBar() # Durum çubuğunu al [cite: 113]
        self.statusBar.setStyleSheet("QStatusBar { border-top: 1px solid #3F3F46; }") # [cite: 114]
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar() # İlerleme çubuğu oluştur
        self.progress_bar.setRange(0, 100) # Aralık belirle (0-100%)
        self.progress_bar.setVisible(False) # Başlangıçta gizle
        self.progress_bar.setMinimumWidth(200) # Minimum genişlik
        self.progress_bar.setMaximumWidth(300) # Maksimum genişlik
        self.progress_bar.setTextVisible(True) # Yüzde göstersin
        self.statusBar.addPermanentWidget(self.progress_bar) # Sağ tarafa ekle [cite: 115]

        # --- Worker Thread Setup ---
        self.worker = None # To hold the current worker thread instance
        self.worker_task = None # To track the current task being run by the worker


        # --- Other Attributes ---
        self.current_midi_file_path = None # Store the path of the currently selected MIDI file
        self.current_midi_analysis = None # Store the analysis result of the current MIDI file [cite: 116]


        # Apply dark theme (if styles.qss exists)
        self.apply_dark_theme()


        # Connect signals and slots after all widgets are created
        self.memory_patterns_list.itemSelectionChanged.connect(self.on_memory_pattern_selection_changed)
        # Connect other signals as needed
        # Connect analyze button if used: self.btn_analyze.clicked.connect(self.analyze_current_midi_file) # If btn_analyze should re-analyze current file
        # Connect generate button
        # self.generate_button.clicked.connect(self.generate_midi_sequence) # Already connected above [cite: 117]


        # Populate memory list after memory is initialized and UI is ready
        if self.midi_memory:
             self.populate_memory_list()


        logger.info("MainWindow created and shown.") # Log success here


    # --- File Handling ---
    def browse_midi_file(self):
        """Opens a file dialog to select a MIDI file."""
        logger.debug("Browse button clicked.") # [cite: 118]
        # Use settings to get initial directory? Or remember last directory? [cite: 119]
        # For now, default to current working directory or a common MIDI folder
        # QFileDialog.getOpenFileName returns a tuple: (selected_file_path, selected_filter)
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("MIDI Dosyası Seçin")
        # Set initial directory based on settings or a default
        # Example: start in the project root or a specific MIDI folder
        # Use settings if available and it has a suitable property like output_dir_path [cite: 120]
        initial_dir = self.settings.general_settings.output_dir_path if self.settings and hasattr(self.settings.general_settings, 'output_dir_path') and self.settings.general_settings.output_dir_path else os.path.expanduser("~") # Default to home dir
        # Ensure initial_dir exists, fallback to home if not
        if not os.path.exists(initial_dir):
             initial_dir = os.path.expanduser("~")
             logger.warning(f"Initial directory from settings not found: {self.settings.general_settings.output_dir_path if self.settings and hasattr(self.settings.general_settings, 'output_dir_path') else 'N/A'}. Falling back to home.") # [cite: 121]

        file_dialog.setDirectory(initial_dir)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("MIDI Files (*.mid *.midi)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.current_midi_file_path = selected_files[0]
                self.midi_path_edit.setText(self.current_midi_file_path)  # Dosya yolunu göster
                logger.info(f"MIDI file selected: {self.current_midi_file_path}")
                self.statusBar.showMessage("Dosya seçildi. Analiz için 'Analiz Et' butonuna tıklayın.", 5000)
                # Removed duplicate statusBar message [cite: 122]


    # --- Analysis ---
    def analyze_current_midi_file(self):
        """Analyzes the currently selected MIDI file using the processor."""
        if not _core_modules_imported: # [cite: 123]
             logger.error("Core modules not imported. Cannot analyze MIDI.") # [cite: 124]
             QMessageBox.critical(self, "Analiz Hatası", "Uygulama bileşenleri yüklenemedi. Analiz yapılamıyor.")
             return


        if self.current_midi_file_path and self.processor:
            logger.info(f"Analyzing selected MIDI file: {self.current_midi_file_path}")
            self.statusBar.showMessage("MIDI analizi yapılıyor...", 0) # Indeterminate message
            self.progress_bar.show()
            self.progress_bar.setRange(0, 0) # Indeterminate progress bar [cite: 125]

            # Run analysis in a worker thread to keep GUI responsive
            # Pass necessary components to the worker
            self.worker = Worker(task="analyze_midi", data={"midi_path": self.current_midi_file_path},
                                 settings=self.settings, processor=self.processor, midi_model=self.midi_model, midi_memory=self.midi_memory) # [cite: 126]
            # Connect signals
            # The signal definition was pyqtSignal(MIDIAnalysis) - so expect MIDIAnalysis or None
            self.worker.analysis_finished.connect(self.on_analysis_finished)
            self.worker.error.connect(self.on_worker_error)
            # Connect task_finished signal for generic completion logging/message
            # self.worker.task_finished.connect(self.on_worker_task_finished) # Connect if using generic signal for this task [cite: 127]

            self.worker_task = "analyze_midi"
            self.worker.start()

        elif not self.current_midi_file_path:
            logger.warning("No MIDI file selected for analysis.")
            self.analysis_text_edit.setPlainText("Analiz için MIDI dosyası seçilmedi.")
            self.piano_roll_label.setText("Piyano Rulosu (Seçilen Dosya Yok)")
            # Clear previous analysis results if any [cite: 128]
            self.current_midi_analysis = None
            self.analysis_text_edit.clear()
            self.piano_roll_label.clear()

        elif not self.processor:
            logger.error("MIDIProcessor is not initialized. Cannot analyze file.") # [cite: 129]
            self.analysis_text_edit.setPlainText("Hata: MIDI Analiz İşlemcisi başlatılamadı.")
            self.piano_roll_label.setText("Piyano Rulosu (Hata)")
            # Clear previous analysis results if any
            self.current_midi_analysis = None
            self.analysis_text_edit.clear()
            self.piano_roll_label.clear()


    # Slot receives MIDIAnalysis or None
    def on_analysis_finished(self, analysis_result: Optional[MIDIAnalysis]): # [cite: 130]
        """Slot to receive analysis results from the worker thread."""
        self.current_midi_analysis = analysis_result # Store the analysis result
        self.statusBar.clearMessage()
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100) # Reset to determinate range

        if analysis_result:
            logger.info("MIDI analysis finished successfully.")
            # Display analysis results using the helper function [cite: 131]
            self.display_analysis_results(analysis_result)

            # Display piano roll image - use the file_path from the analysis result
            if analysis_result.file_path and os.path.exists(analysis_result.file_path):
                 self.display_piano_roll(analysis_result.file_path)
            else:
                 logger.warning("Cannot display piano roll: Analysis result has no valid file path.") # [cite: 132]
                 self.piano_roll_label.setText("Piyano Rulosu (Dosya Bulunamadı)")
                 self.piano_roll_label.clear()


        else:
            logger.warning("MIDI analysis returned no results or encountered an error.")
            self.analysis_text_edit.setPlainText("MIDI analizi tamamlandı, ancak sonuç alınamadı veya bir hata oluştu.") # [cite: 133]
            self.piano_roll_label.setText("Piyano Rulosu (Analiz Başarısız)")
            self.piano_roll_label.clear() # Clear any previous piano roll


    def display_analysis_results(self, analysis_result: MIDIAnalysis):
        """Helper function to display MIDIAnalysis results in the text edit."""
        # This is similar to on_analysis_finished but takes a MIDIAnalysis object directly.
        if not analysis_result: # [cite: 134]
             self.analysis_text_edit.setPlainText("Analiz verisi bulunmuyor.")
             return

        analysis_text = f"Kaynak Dosya: {os.path.basename(analysis_result.file_path)}\n"
        analysis_text += f"Süre: {analysis_result.duration:.2f} saniye\n"
        analysis_text += f"Tempo: {analysis_result.tempo:.2f} BPM\n"
        analysis_text += f"Anahtar: {analysis_result.key}\n"
        analysis_text += f"Zaman İşareti: {analysis_result.time_signature}\n"
        analysis_text += f"Nota Sayısı: {analysis_result.note_count}\n" # [cite: 135]
        analysis_text += f"Ortalama Velocity: {analysis_result.average_velocity:.2f}\n"
        analysis_text += f"Pitch Aralığı: {analysis_result.pitch_range}\n"
        analysis_text += f"Ritim Karmaşıklığı: {analysis_result.rhythm_complexity:.4f}\n"

        analysis_text += "\nEnstrümanlar:\n"
        if analysis_result.instrument_names:
            # Sort instrument names alphabetically for consistent display
            sorted_instrument_names = sorted(analysis_result.instrument_names.items())
            for name, count in sorted_instrument_names: # [cite: 136]
                 analysis_text += f"- {name} ({count} nota)\n"
        elif analysis_result.instrument_programs: # Fallback to program numbers if names not resolved
            # Sort by program number
            sorted_instrument_programs = sorted(analysis_result.instrument_programs.items())
            for program, count in sorted_instrument_programs:
                 analysis_text += f"- Program {program} ({count} nota)\n" # [cite: 137]
        else:
             analysis_text += "  Tespit edilen enstrüman yok.\n"

        # Display a simplified polyphony profile (avoid flooding with every change)
        analysis_text += "\nPolifoni Profili (Başlangıç Zamanları):\n"
        if analysis_result.polyphony_profile:
             # Sort polyphony profile by time [cite: 138]
             sorted_polyphony_points = sorted(analysis_result.polyphony_profile.items())
             # Display up to N points or summary statistics
             num_points_to_display = 10
             display_interval = max(1, len(sorted_polyphony_points) // num_points_to_display)
             for i in range(0, len(sorted_polyphony_points), display_interval):
                 time, polyphony = sorted_polyphony_points[i] # [cite: 139]
                 analysis_text += f"  - Zaman {time:.2f}s: {polyphony} nota\n"
             if len(sorted_polyphony_points) > num_points_to_display:
                  # Calculate the actual remaining count to display correctly
                  remaining_count = len(sorted_polyphony_points) - (i + display_interval) # [cite: 140]
                  if remaining_count > 0:
                       analysis_text += f"  ... (ve {remaining_count} ek zaman noktası)\n" # Correct remaining count


        else:
             analysis_text += "  Polifoni verisi bulunamadı veya boş.\n"


        self.analysis_text_edit.setPlainText(analysis_text)


    def display_piano_roll(self, midi_path: str): # [cite: 141]
        """
        Generates and displays a piano roll image for the given MIDI file.
        Requires pretty_midi and matplotlib. [cite: 142]
        """
        if not os.path.exists(midi_path):
             logger.warning(f"Cannot display piano roll: MIDI file not found at {midi_path}")
             self.piano_roll_label.setText("Piyano Rulosu (Dosya Bulunamadı)")
             # Clear previous pixmap if any
             self.piano_roll_label.clear()
             return

        try: # [cite: 143]
            logger.info(f"Generating piano roll for: {os.path.basename(midi_path)}")
            midi_data = pretty_midi.PrettyMIDI(midi_path)

            # Create the piano roll image using pretty_midi's methods and matplotlib
            # Get a piano roll representation (pitch, time)
            # Can specify resolution (fs - sampling frequency), default is 100 Hz
            # Use the model's resolution setting to match if desired, but pretty_midi's default is usually good for visualization [cite: 144]
            # piano_roll = midi_data.get_piano_roll(fs=int(1/self.settings.model_settings.resolution)) # Use model resolution
            piano_roll = midi_data.get_piano_roll(fs=100) # Use default 100 Hz

            # Plot the piano roll using matplotlib
            fig, ax = plt.subplots(figsize=(8, 4)) # Use figure and axes objects [cite: 145]
            # Use pretty_midi's display function or matplotlib.imshow
            # pretty_midi.display.specshow(piano_roll, sr=100, x_axis='time', y_axis='cqt_note') # Example with pretty_midi display
            # Or use matplotlib.imshow directly
            # Transpose piano_roll to get (time, pitch) if needed for imshow orientation
            # The get_piano_roll output is (pitch, time). Transpose to (time, pitch) for imshow with origin='lower' [cite: 146]
            # Use extent to set axis limits correctly based on time and pitch range
            # Time extent: from 0 to duration (end_time_seconds)
            # Pitch extent: from min_pitch (or 21) to max_pitch (or 108)
            # Need min/max pitch from analysis if available, otherwise use default note range
            min_midi_pitch = self.current_midi_analysis.pitch_range[0] if self.current_midi_analysis and self.current_midi_analysis.pitch_range and self.current_midi_analysis.pitch_range[0] is not None else self.settings.model_settings.note_range[0] if self.settings and self.settings.model_settings else 21 # [cite: 147]
            max_midi_pitch = self.current_midi_analysis.pitch_range[1] if self.current_midi_analysis and self.current_midi_analysis.pitch_range and self.current_midi_analysis.pitch_range[1] is not None else self.settings.model_settings.note_range[1] if self.settings and self.settings.model_settings else 108
            duration_seconds = self.current_midi_analysis.duration if self.current_midi_analysis else midi_data.get_end_time()


            # Ensure pitch range is valid for plotting
            if min_midi_pitch is None or max_midi_pitch is None or min_midi_pitch >= max_midi_pitch: # [cite: 148]
                 min_midi_pitch = 21
                 max_midi_pitch = 108
                 logger.warning("Invalid or None pitch range from analysis. Using default (21-108) for piano roll display.") # [cite: 149]


            # Adjust extent to match MIDI pitches correctly.
            # imshow extent is (left, right, bottom, top).
            # For piano roll (time, pitch), this means (min_time, max_time, min_pitch, max_pitch)
            # The piano_roll array covers the pitches from 0 to 127. [cite: 150]
            # If we plot piano_roll, the y-axis is pitch index relative to 0-127 range.
            # We need to map these pitch indices to actual MIDI note numbers.
            # The y-axis limits should correspond to the MIDI note range used or found.
            # Let's plot the full 0-127 range but set y-limits to the relevant range.


            # Get the time vector for the piano roll [cite: 151]
            # The x-axis should represent time in seconds. pretty_midi.get_piano_roll(fs) gives columns per time step.
            # time for each column = column_index / fs
            # The x-axis limits should go from 0 to the total duration.
            # The number of time steps in piano_roll is piano_roll.shape[1]. [cite: 152]
            # The total time covered is piano_roll.shape[1] / fs.
            # Extent needs (left, right, bottom, top). For transpose (time, pitch): (0, total_time, min_pitch, max_pitch) [cite: 153]
            # For non-transpose (pitch, time): (0, total_time, min_pitch_index, max_pitch_index)
            # Let's plot non-transpose (pitch, time) and set y-limits to MIDI pitches. [cite: 154]
            # imshow with origin='lower' means index 0 is at the bottom. [cite: 155]
            # If piano_roll is (pitch, time), pitch 0 is at y=0, pitch 127 is at y=127. [cite: 156]
            # y-limits should be (min_midi_pitch, max_midi_pitch)


            # Plot the piano roll image
            # Use aspect='auto' to fill the area, origin='lower' for MIDI note axis
            # The y-axis of imshow (pitch index) needs to map to MIDI note number. [cite: 157]
            # If piano_roll covers 0-127 pitches, y-axis index I corresponds to MIDI note I.
            im = ax.imshow(piano_roll, aspect='auto', origin='lower', cmap='Blues', interpolation='nearest',
                           extent=[0, duration_seconds, 0, 127]) # X-axis: time in seconds, Y-axis: MIDI Pitch 0-127

            ax.set_title(f"Piyano Rulosu: {os.path.basename(midi_path)}")
            ax.set_xlabel("Zaman (saniye)")
            ax.set_ylabel("MIDI Nota") # [cite: 158]

            # Set y-axis limits to cover the relevant pitch range found or set
            ax.set_ylim(min_midi_pitch - 1, max_midi_pitch + 1) # Add padding


            # Add colorbar if piano roll represents velocity (get_piano_roll usually uses velocity values)
            # Check if piano_roll is not empty before adding colorbar
            if piano_roll.size > 0: # [cite: 159]
                 fig.colorbar(im, ax=ax, label="Velocity") # Add colorbar if piano roll represents velocity


            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # Save the plot to a BytesIO object (in-memory file) as a PNG image
            buf = io.BytesIO()
            plt.savefig(buf, format='png') # [cite: 160]
            buf.seek(0) # Rewind the buffer

            # Close the matplotlib figure to free memory
            plt.close(fig) # Close the specific figure
            logger.debug("Matplotlib figure closed.")


            # Load the image from the buffer into a QPixmap
            img = QImage.fromData(buf.getvalue()) # [cite: 161]
            pixmap = QPixmap.fromImage(img)

            # Scale the pixmap to fit the label while maintaining aspect ratio
            # Get the size of the label or a desired maximum size
            label_size = self.piano_roll_label.size() # Get the current size of the label
            # Handle case where label_size might be empty (e.g., before UI is fully rendered) [cite: 162]
            if not label_size.isNull() and label_size.width() > 0 and label_size.height() > 0:
                 scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                 logger.debug(f"Scaled piano roll to label size: {label_size.width()}x{label_size.height()}")
            else:
                 # If label size is not available or zero, scale to a default size or just use the original pixmap [cite: 163]
                 # Scaling to a fixed smaller size is better than full size or nothing
                 default_scaled_size = QSize(600, 300) # Example default size
                 scaled_pixmap = pixmap.scaled(default_scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation) # [cite: 164]
                 logger.warning(f"Piano roll label size is null or zero ({label_size.width()}x{label_size.height()}). Using default scaled size {default_scaled_size.width()}x{default_scaled_size.height()} for display.") # [cite: 165]


            # Set the pixmap to the label
            self.piano_roll_label.setPixmap(scaled_pixmap)
            self.piano_roll_label.setText("") # Clear the default text
            logger.info("Piano roll image displayed.")

        except Exception as e:
            logger.error(f"Error displaying piano roll for {os.path.basename(midi_path)}: {e}", exc_info=True)
            self.piano_roll_label.setText("Piyano Rulosu (Görüntülenemedi)") # [cite: 166]
            # Clear previous pixmap on error
            self.piano_roll_label.clear()


    # --- Generation ---
    def generate_midi_sequence(self):
        """Bar/ölçü sayısına göre MIDI dizisi üretimini başlatır."""
        if not _core_modules_imported:
             logger.error("Core modules not imported. Cannot generate MIDI.")
             QMessageBox.critical(self, "Üretim Hatası", "Uygulama bileşenleri yüklenemedi. Üretim yapılamıyor.") # [cite: 167]
             return

        if self.midi_model and self.processor and self.settings and self.settings.model_settings:
            # Model eğitilmiş mi kontrol et
            if self.midi_model.model is None or not self.midi_model._is_trained:
                 QMessageBox.warning(self, "Üretim Hatası", "Model henüz eğitilmemiş veya yüklenmemiş. Lütfen önce bir model eğitin veya yükleyin.") # [cite: 168]
                 logger.warning("Generation attempted but model is not built/trained/loaded.")
                 return

            logger.info("Initiating MIDI sequence generation.")
            
            # Bar/ölçü sayısından adım sayısını hesapla
            # Eğer bar_count_spinbox tanımlı değilse veya erişilemiyorsa, varsayılan değerleri kullan [cite: 169]
            try:
                bar_count = self.bar_count_spinbox.value() if hasattr(self, 'bar_count_spinbox') else 4
                tempo = self.tempo_spinbox.value() if hasattr(self, 'tempo_spinbox') else 120
                temperature = self.temperature_double_spinbox.value() if hasattr(self, 'temperature_double_spinbox') else 1.0
            except Exception as e: # [cite: 170]
                logger.warning(f"Could not access UI controls, using default values: {e}")
                bar_count = 4  # Varsayılan 4 bar
                tempo = 120    # Varsayılan 120 BPM
                temperature = 1.0  # Varsayılan sıcaklık [cite: 171]
            
            # Bar başına 16 adım (16th note resolution) varsayalım
            # 4/4'lük ölçüde, bir ölçü 16 adım içerir (16th note resolution ile)
            steps_per_bar = 16
            num_steps_to_generate = bar_count * steps_per_bar
            
            logger.info(f"Generating {bar_count} bars ({num_steps_to_generate} steps) at {tempo} BPM with temperature {temperature}") # [cite: 172]
            
            # Kullanıcıya bilgi ver
            self.statusBar.showMessage(f"{bar_count} ölçü ({num_steps_to_generate} adım) üretiliyor...", 0)

            # Get a seed sequence (e.g., from the end of the analyzed MIDI or a random start) [cite: 173]
            # For simplicity now, let's use a random seed or a dummy seed if no file is analyzed. [cite: 174]
            # In a real app, the seed could be from user input, a selected pattern, or the end of the last generated sequence.
            seed_sequence = None # [cite: 175]
            if self.current_midi_file_path and self.current_midi_analysis and self.processor:
                 # Option 1: Use the end of the analyzed MIDI file as seed
                 # This requires converting the end portion of the analyzed MIDI to sequence format
                 try:
                     # Ensure processor is available [cite: 176]
                     if self.processor:
                         # Convert the whole MIDI to sequence and take the end if long enough
                         full_sequence = self.processor.midi_to_sequence(self.current_midi_file_path) # [cite: 177]
                         if full_sequence is not None and full_sequence.shape[0] >= self.settings.model_settings.sequence_length:
                              seed_sequence = full_sequence[-self.settings.model_settings.sequence_length:, :, :] # Take the last 'sequence_length' steps
                              # Add batch dimension (batch size 1) [cite: 178]
                              seed_sequence = np.expand_dims(seed_sequence, axis=0)
                              logger.debug(f"Using end of '{os.path.basename(self.current_midi_file_path)}' as seed.")
                         elif full_sequence is not None: # [cite: 179]
                              logger.warning(f"Analyzed MIDI is too short ({full_sequence.shape[0]} steps) for seed sequence ({self.settings.model_settings.sequence_length} steps). Using random seed.") # [cite: 180]
                              seed_sequence = self.create_random_seed_sequence() # Fallback to random seed
                         else:
                              logger.warning("Failed to convert analyzed MIDI to full sequence. Using random seed.") # [cite: 181]
                              seed_sequence = self.create_random_seed_sequence() # Fallback to random seed
                     else:
                          logger.error("MIDIProcessor not available for seed sequence creation from analyzed MIDI. Using random seed.") # [cite: 182]
                          seed_sequence = self.create_random_seed_sequence()


                 except Exception as e:
                      logger.error(f"Error creating seed sequence from analyzed MIDI: {e}", exc_info=True)
                      logger.warning("Using random seed due to error.") # [cite: 183]
                      seed_sequence = self.create_random_seed_sequence() # Fallback to random seed

            else:
                 # Option 2: Use a random seed sequence
                 logger.info("No analyzed MIDI selected. Using random seed for generation.") # [cite: 184]
                 seed_sequence = self.create_random_seed_sequence()


            if seed_sequence is not None:
                 logger.debug(f"Seed sequence shape for generation: {seed_sequence.shape}")
                 # Ensure seed sequence batch size is 1 for consistency with model's expectation
                 if seed_sequence.ndim == 3: # [cite: 185]
                     seed_sequence = np.expand_dims(seed_sequence, axis=0)
                     logger.debug(f"Added batch dimension to seed sequence. New shape: {seed_sequence.shape}")
                 elif seed_sequence.ndim != 4 or seed_sequence.shape[0] != 1:
                      logger.error(f"Invalid seed sequence shape for generation: {seed_sequence.shape}. Expected (1, seq_len, note_range, features).") # [cite: 186]
                      QMessageBox.critical(self, "Üretim Hatası", f"Başlangıç (seed) dizisi geçersiz boyutta: {seed_sequence.shape}.")
                      return


                 self.progress_bar.show()
                 self.progress_bar.setRange(0, 0) # Indeterminate progress bar [cite: 187]

                 # Müzik stili bilgisini al
                 selected_style = self.style_combo.currentText()
                 
                 # Run generation in a worker thread
                 # Pass necessary components to the worker [cite: 188]
                 self.worker = Worker(task="generate_sequence",
                                   data={
                                       "seed_sequence": seed_sequence, # [cite: 189]
                                       "num_steps": num_steps_to_generate,
                                       "temperature": temperature,
                                       "tempo": tempo, # [cite: 190]
                                       "style": selected_style,
                                       "bar_count": bar_count # [cite: 191]
                                   },
                                   settings=self.settings,
                                   processor=self.processor, # [cite: 192]
                                   midi_model=self.midi_model,
                                   midi_memory=self.midi_memory)

                 self.worker.generation_finished.connect(self.on_generation_finished) # [cite: 193]
                 self.worker.error.connect(self.on_worker_error)

                 self.worker_task = "generate_sequence"
                 self.worker.start()

                 logger.info(f"Started sequence generation: {bar_count} bars, {tempo} BPM, {selected_style} style, temperature {temperature}")

            else:
                 logger.error("Failed to create a seed sequence for generation.") # [cite: 194]
                 QMessageBox.critical(self, "Üretim Hatası", "Üretim için bir başlangıç (seed) dizisi oluşturulamadı.")


        elif not self.midi_model:
            logger.error("MIDIModel is not initialized. Cannot generate sequence.") # [cite: 195]
            QMessageBox.critical(self, "Üretim Hatası", "Hata: MIDI Model başlatılamadı.")
        elif not self.processor:
            logger.error("MIDIProcessor is not initialized. Cannot convert sequence to MIDI.")
            QMessageBox.critical(self, "Üretim Hatası", "Hata: MIDI İşlemcisi başlatılamadı.")
        elif not (self.settings and self.settings.model_settings):
            logger.error("Settings or ModelSettings are not available. Cannot generate sequence.") # [cite: 196]
            QMessageBox.critical(self, "Üretim Hatası", "Hata: Ayarlar yüklenemedi.")



    # Creates a random seed sequence with the correct shape (batch_size=1)
    def create_random_seed_sequence(self) -> Optional[np.ndarray]:
         """Creates a random seed sequence with the correct shape (batch_size=1)."""
         if not self.settings or not self.settings.model_settings:
              logger.error("Settings or ModelSettings not available for random seed creation.")
              return None # Cannot create seed without settings [cite: 197]


         logger.info("Creating a random seed sequence.")
         # Shape: (1, sequence_length, note_range_size, input_features)
         seq_len = self.settings.model_settings.sequence_length
         note_range_size = self.settings.model_settings.note_range[1] - self.settings.model_settings.note_range[0] + 1
         input_feats = self.settings.model_settings.input_features

         # Ensure note_range_size is positive
         if note_range_size <= 0: # [cite: 198]
              logger.error(f"Invalid note range in settings: {self.settings.model_settings.note_range}. Cannot create random seed.") # [cite: 199]
              return None
         # Ensure sequence_length is positive
         if seq_len <= 0:
              logger.error(f"Invalid sequence length in settings: {seq_len}. Cannot create random seed.")
              return None
         # Ensure input_features is positive
         if input_feats <= 0: # [cite: 200]
              logger.error(f"Invalid input features in settings: {input_feats}. Cannot create random seed.")
              return None


         seed_sequence = np.zeros((1, seq_len, note_range_size, input_feats), dtype=np.float32)

         # Example: Add some random sparse notes to the seed
         num_random_notes = int(seq_len * note_range_size * 0.01) # Example: 1% sparse notes
         num_random_notes = max(5, min(num_random_notes, 50)) # Clamp between 5 and 50 for reasonable density [cite: 201]

         for _ in range(num_random_notes):
              random_step = random.randint(0, seq_len - 1)
              random_pitch_index = random.randint(0, note_range_size - 1)
              seed_sequence[0, random_step, random_pitch_index, 0] = 1.0 # Pitch activation
              if input_feats >= 2: # [cite: 202]
                   # Ensure velocity is within 0-1 range if input_features >= 2
                   seed_sequence[0, random_step, random_pitch_index, 1] = random.random() # Random normalized velocity

         return seed_sequence


    # Slot receives np.ndarray or None
    def on_generation_finished(self, generated_sequence: Optional[np.ndarray]):
        """Slot to receive generated sequence from the worker thread.""" # [cite: 203]
        self.statusBar.clearMessage()
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100) # Reset

        # Check if processor is available before trying to convert to MIDI
        if generated_sequence is not None and self.processor and self.settings and hasattr(self.settings, 'output_dir_path') and self.settings.output_dir_path:
            logger.info("MIDI sequence generation finished successfully.")
            
            # Worker'dan tempo ve stil bilgilerini al [cite: 204]
            tempo_to_use = 120.0  # Varsayılan değer
            style_name = ""      # Varsayılan boş
            bar_count = 4        # Varsayılan değer
            
            # Worker'dan gelen verileri kontrol et [cite: 205]
            if hasattr(self.worker, 'data') and self.worker.data:
                if 'tempo' in self.worker.data:
                    tempo_to_use = self.worker.data['tempo']
                if 'style' in self.worker.data:
                    style_name = self.worker.data['style'] # [cite: 206]
                if 'bar_count' in self.worker.data:
                    bar_count = self.worker.data['bar_count']
            
            # Dosya adı oluştur - tempo ve stil bilgilerini içersin
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            style_suffix = f"_{style_name}" if style_name and style_name != "Otomatik" else "" # [cite: 207]
            output_filename = f"generated_{bar_count}bar_{tempo_to_use}bpm{style_suffix}_{timestamp}.mid"
            output_path = os.path.join(self.settings.output_dir_path, output_filename)
            
            # Çıktı dizini yoksa oluştur
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): # [cite: 208]
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory for generated MIDI: {output_dir}")

            logger.info(f"Converting generated sequence to MIDI: {output_path} with tempo {tempo_to_use} BPM")
            success = self.processor.sequence_to_midi(generated_sequence, output_path, tempo=tempo_to_use)


            if success:
                logger.info(f"Generated MIDI saved to: {output_path}") # [cite: 209]
                QMessageBox.information(self, "Üretim Başarılı", f"MIDI dizisi başarıyla üretildi ve kaydedildi:\n{output_path}")
                # Optional: Add the generated MIDI to the file list or memory automatically
                # This would require analyzing the generated MIDI first
                # self.analyze_and_add_to_memory(output_path) # Requires a new method [cite: 210]


            else:
                logger.error(f"Failed to save generated MIDI to {output_path}")
                QMessageBox.critical(self, "Üretim Hatası", f"Üretilen MIDI dizisi kaydedilirken hata oluştu:\n{output_path}")
        # Handle case where generated_sequence is None due to worker error
        elif generated_sequence is None: # [cite: 211]
            logger.error("MIDI sequence generation failed in worker thread.")
            QMessageBox.critical(self, "Üretim Hatası", "MIDI dizisi üretimi sırasında bir hata oluştu.")
        # Handle missing components/settings
        else: # self.processor or settings or output_dir_path is missing
            logger.error("Required components (Processor, Settings, Output Dir) not available for saving generated MIDI.")
            QMessageBox.critical(self, "Üretim Hatası", "Hata: Gerekli bileşenler veya çıktı dizini yolu bulunamadı. Üretilen MIDI kaydedilemedi.") # [cite: 212]


    # --- Model Training ---
    def start_training(self):
        """Starts the model training process with data from memory patterns."""
        if not _core_modules_imported:
            logger.error("Core modules not imported. Cannot train model.")
            QMessageBox.critical(self, "Eğitim Hatası", "Uygulama bileşenleri yüklenemedi. Model eğitilemiyor.")
            return

        if not self.midi_model: # [cite: 214]
            logger.error("MIDIModel is not initialized. Cannot train model.")
            QMessageBox.critical(self, "Eğitim Hatası", "Model bileşeni başlatılamadı.")
            return

        if not self.midi_memory:
            logger.error("MIDIMemory is not initialized. Cannot get training data.")
            QMessageBox.critical(self, "Eğitim Hatası", "Hafıza bileşeni başlatılamadı.")
            return # [cite: 215]

        # Get training data from memory patterns
        all_patterns = self.midi_memory.get_all_patterns() if self.midi_memory else []
        
        if not all_patterns:
            logger.warning("No patterns in memory for training.")
            QMessageBox.warning(self, "Eğitim Verisi Yok", "Hafızada eğitim için desen bulunamadı. Lütfen önce MIDI dosyalarını hafızaya ekleyin.") # [cite: 216]
            return

        # Prepare training data from patterns
        logger.info(f"Preparing training data from {len(all_patterns)} patterns...")
        
        # Show a progress dialog while preparing data
        self.statusBar.showMessage("Eğitim verisi hazırlanıyor...", 0)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        try: # [cite: 217]
            # Convert patterns to sequences for training
            training_sequences = []
            for pattern in all_patterns:
                if pattern.file_path and os.path.exists(pattern.file_path) and self.processor:
                    try: # [cite: 218]
                        sequence = self.processor.midi_to_sequence(pattern.file_path)
                        if sequence is not None:
                            training_sequences.append(sequence)
                            logger.debug(f"Added sequence from {os.path.basename(pattern.file_path)} to training data") # [cite: 219]
                    except Exception as e:
                        logger.error(f"Error converting pattern to sequence: {e}", exc_info=True)
            
            if not training_sequences:
                logger.error("Failed to prepare any valid training sequences.") # [cite: 220]
                self.statusBar.clearMessage()
                self.progress_bar.hide()
                QMessageBox.critical(self, "Eğitim Hatası", "Eğitim dizileri hazırlanamadı. Lütfen geçerli MIDI dosyaları ekleyin.") # [cite: 221]
                return

            # Combine sequences into a single training dataset
            training_data = np.concatenate(training_sequences, axis=0)
            logger.info(f"Training data prepared with shape: {training_data.shape}")

            # --- Reshape training data to (num_samples, sequence_length, note_range_size, input_features) ---
            # Get necessary dimensions from settings and processor [cite: 222]
            sequence_length = self.settings.model_settings.sequence_length
            note_range_size = self.processor.note_range_size
            input_features = self.processor.input_features

            # Calculate the number of samples
            num_samples = training_data.shape[0] // sequence_length
            
            # Truncate the training data to fit the sequence length [cite: 223]
            truncated_length = num_samples * sequence_length
            training_data = training_data[:truncated_length]

            # Reshape the training data
            training_data = training_data.reshape((num_samples, sequence_length, note_range_size, input_features))
            logger.info(f"Reshaped training data to shape: {training_data.shape}") # [cite: 224]

            # Get training parameters (could be from UI in the future)
            epochs = 10  # Default number of epochs
            
            # Show training dialog with parameters
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Eğitim Başlat")
            msg_box.setText(f"Model eğitimi başlatılacak:\n\n" # [cite: 225]
                           f"- Eğitim veri boyutu: {training_data.shape}\n"
                           f"- Epoch sayısı: {epochs}\n\n"
                           f"Eğitim işlemi başlatılsın mı?") # [cite: 226]
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) # [cite: 227]
            msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

            if msg_box.exec() == QMessageBox.StandardButton.Yes:
                # Start training in worker thread
                self.worker = Worker(
                    task="train_model",
                    data={"training_data": training_data, "epochs": epochs}, # [cite: 228]
                    settings=self.settings,
                    processor=self.processor,
                    midi_model=self.midi_model,
                    midi_memory=self.midi_memory
                ) # [cite: 229]
                
                self.worker.training_progress.connect(self.on_training_progress)
                self.worker.training_finished.connect(self.on_training_finished)
                self.worker.error.connect(self.on_worker_error)
                
                self.worker_task = "train_model" # [cite: 230]
                self.worker.start()
                
                logger.info("Model eğitimi başlatıldı.")
                self.statusBar.showMessage("Eğitim başlatılıyor...", 0)
                self.progress_bar.show()
                self.progress_bar.setRange(0, epochs) # [cite: 231]
        except Exception as e:
            logger.error(f"Error preparing training data: {e}", exc_info=True)
            self.statusBar.clearMessage()
            self.progress_bar.hide()
            QMessageBox.critical(self, "Eğitim Hatası", f"Eğitim hazırlığı sırasında hata oluştu:\n{e}")
            return

    def on_training_progress(self, epoch: int, total_epochs: int, loss: float):
        """Updates the UI with training progress.""" # [cite: 232]
        logger.debug(f"Training progress: Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")
        
        # Update progress bar
        self.progress_bar.setRange(0, total_epochs)
        self.progress_bar.setValue(epoch)
        
        # Update status message
        self.statusBar.showMessage(f"Eğitim: Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}", 0)

    def on_training_finished(self, model):
        """Updates the UI when training is complete."""
        logger.info("Training finished.")
        self.statusBar.showMessage("Eğitim tamamlandı.", 5000)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.hide()
        
    def add_to_memory(self):
        """Adds the current MIDI analysis to memory."""
        if not self.current_midi_analysis or not self.current_midi_file_path:
            logger.warning("No MIDI analysis or file to add to memory.")
            QMessageBox.warning(self, "Hafıza Hatası", "Hafızaya eklemek için analiz edilmiş bir MIDI dosyası gereklidir.")
            return
            
        try:
            # Eğer midi_memory varsa ve doğru şekilde başlatıldıysa
            if hasattr(self, 'midi_memory') and self.midi_memory:
                # Analiz sonucunu hafızaya ekle
                pattern_name = os.path.basename(self.current_midi_file_path)
                success = self.midi_memory.add_pattern(pattern_name, self.current_midi_analysis)
                
                if success:
                    logger.info(f"Added pattern {pattern_name} to memory.")
                    self.statusBar.showMessage(f"\"{pattern_name}\" hafızaya eklendi.", 5000)
                    # Hafıza listesini güncelle
                    self.update_memory_list()
                else:
                    logger.warning(f"Failed to add pattern {pattern_name} to memory.")
                    QMessageBox.warning(self, "Hafıza Hatası", f"\"{pattern_name}\" hafızaya eklenemedi.")
            else:
                logger.error("MIDI Memory is not initialized.")
                QMessageBox.warning(self, "Hafıza Hatası", "Hafıza bileşeni başlatılamadı.")
        except Exception as e:
            error_msg = f"Hafızaya ekleme sırasında hata: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Hafıza Hatası", error_msg)
            
    def search_memory(self):
        """Searches for similar patterns in memory based on current MIDI analysis."""
        if not self.current_midi_analysis:
            logger.warning("No MIDI analysis to search with.")
            QMessageBox.warning(self, "Arama Hatası", "Benzer desen aramak için analiz edilmiş bir MIDI dosyası gereklidir.")
            return
            
        try:
            # Eğer midi_memory varsa ve doğru şekilde başlatıldıysa
            if hasattr(self, 'midi_memory') and self.midi_memory:
                # Arama işlemi uzun sürebilir, worker thread'de çalıştır
                self.statusBar.showMessage("Benzer desenler aranıyor...", 0)
                self.progress_bar.show()
                self.progress_bar.setRange(0, 0)  # Belirsiz ilerleme çubuğu
                
                # Worker thread'i başlat
                self.worker = Worker(task="search_memory", 
                                     data={"analysis": self.current_midi_analysis},
                                     settings=self.settings, 
                                     processor=self.processor, 
                                     midi_model=self.midi_model, 
                                     midi_memory=self.midi_memory)
                
                # Sinyalleri bağla
                self.worker.memory_search_finished.connect(self.on_memory_search_finished)
                self.worker.error.connect(self.on_worker_error)
                
                self.worker_task = "search_memory"
                self.worker.start()
            else:
                logger.error("MIDI Memory is not initialized.")
                QMessageBox.warning(self, "Arama Hatası", "Hafıza bileşeni başlatılamadı.")
        except Exception as e:
            error_msg = f"Hafıza araması sırasında hata: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Arama Hatası", error_msg)
            
    def on_memory_search_finished(self, results):
        """Handles the results of a memory search."""
        self.statusBar.showMessage("Benzer desen araması tamamlandı.", 5000)
        self.progress_bar.hide()
        
        if not results or len(results) == 0:
            logger.info("No similar patterns found in memory.")
            QMessageBox.information(self, "Arama Sonucu", "Hafızada benzer desen bulunamadı.")
            return
            
        # Sonuçları göster
        result_text = "Benzer Desenler:\n\n"
        for i, (pattern_name, similarity) in enumerate(results, 1):
            result_text += f"{i}. {pattern_name} - Benzerlik: {similarity:.2f}\n"
            
        # Sonuçları göstermek için bir dialog oluştur
        results_dialog = QMessageBox(self)
        results_dialog.setWindowTitle("Arama Sonuçları")
        results_dialog.setText(result_text)
        results_dialog.setIcon(QMessageBox.Icon.Information)
        results_dialog.exec()
        
    def update_memory_list(self):
        """Updates the memory patterns list with current patterns."""
        if not hasattr(self, 'midi_memory') or not self.midi_memory:
            return
            
        # Listeyi temizle
        self.memory_patterns_list.clear()
        
        # Desenleri al ve listeye ekle
        patterns = self.midi_memory.get_all_patterns()
        for pattern_name in patterns:
            item = QListWidgetItem(pattern_name)
            self.memory_patterns_list.addItem(item)
            
    def save_settings(self):
        """Saves the current settings to a file."""
        try:
            if hasattr(self, 'settings') and self.settings:
                # Ayarları JSON formatında göster
                settings_json = json.dumps(asdict(self.settings), indent=4, ensure_ascii=False)
                self.settings_text_edit.setPlainText(settings_json)
                
                # Ayarları dosyaya kaydet
                config_dir = get_config_dir()
                os.makedirs(config_dir, exist_ok=True)
                settings_file = os.path.join(config_dir, "settings.json")
                
                with open(settings_file, "w", encoding="utf-8") as f:
                    f.write(settings_json)
                    
                logger.info(f"Settings saved to {settings_file}")
                self.statusBar.showMessage(f"Ayarlar kaydedildi: {settings_file}", 5000)
            else:
                logger.error("Settings object is not initialized.")
                QMessageBox.warning(self, "Ayarlar Hatası", "Ayarlar bileşeni başlatılamadı.")
        except Exception as e:
            error_msg = f"Ayarları kaydetme sırasında hata: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Ayarlar Hatası", error_msg)
            
    def load_settings(self):
        """Loads settings from a file."""
        try:
            config_dir = get_config_dir()
            settings_file = os.path.join(config_dir, "settings.json")
            
            if not os.path.exists(settings_file):
                logger.warning(f"Settings file not found: {settings_file}")
                QMessageBox.warning(self, "Ayarlar Hatası", "Ayarlar dosyası bulunamadı.")
                return
                
            with open(settings_file, "r", encoding="utf-8") as f:
                settings_json = f.read()
                
            # Ayarları JSON formatında göster
            self.settings_text_edit.setPlainText(settings_json)
            
            # Ayarları yükle
            if hasattr(self, 'settings') and self.settings:
                settings_dict = json.loads(settings_json)
                
                # Ayarları güncelle
                # Not: Gerçek bir uygulamada, burada daha karmaşık bir güncelleme mantığı olabilir
                # Örneğin, her bir alt ayar sınıfını ayrı ayrı güncelleme gibi
                
                logger.info("Settings loaded successfully.")
                self.statusBar.showMessage("Ayarlar yüklendi.", 5000)
            else:
                logger.error("Settings object is not initialized.")
                QMessageBox.warning(self, "Ayarlar Hatası", "Ayarlar bileşeni başlatılamadı.")
        except Exception as e:
            error_msg = f"Ayarları yükleme sırasında hata: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Ayarlar Hatası", error_msg)
            
    def on_analyze_button_clicked(self):
        """Handles the analyze button click event."""
        if not self.current_midi_file_path:
            logger.warning("No MIDI file selected for analysis.")
            QMessageBox.warning(self, "Analiz Hatası", "Analiz için önce bir MIDI dosyası seçmelisiniz.")
            return
            
        # Analiz işlemini başlat
        logger.info(f"Analyzing MIDI file: {self.current_midi_file_path}")
        self.analyze_current_midi_file()
        
        # Piyano rulosunu görüntüle
        if self.current_midi_file_path:
            self.display_piano_roll(self.current_midi_file_path)
            # Optionally save the trained model [cite: 234]
            self.save_model_after_training()
        else:
            logger.warning("Model training completed but returned no history.")
            QMessageBox.warning(self, "Eğitim Tamamlandı", "Model eğitimi tamamlandı ancak eğitim geçmişi alınamadı.")
        
        self.worker_task = None

    def save_model_after_training(self): # [cite: 235]
        """Asks the user if they want to save the trained model."""
        if not self.midi_model:
            return
            
        try:
            # Create trained_models directory if it doesn't exist
            trained_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trained_models')
            os.makedirs(trained_models_dir, exist_ok=True) # [cite: 236]
            
            # Generate a unique filename based on current time
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(trained_models_dir, f'midi_model_{timestamp}.h5')
            
            logger.info(f"Saving model to {model_path}")
            
            # Ensure model is built and trained before saving [cite: 237]
            if not self.midi_model.model:
                logger.error("Model is not built. Cannot save.") # [cite: 238]
                QMessageBox.critical(self, "Model Kaydetme Hatası", 
                                 "Model henüz oluşturulmamış. Lütfen önce eğitimi başlatın.")
                return
            
            if not self.midi_model._is_trained: # [cite: 239]
                logger.error("Model is not trained. Cannot save.")
                QMessageBox.critical(self, "Model Kaydetme Hatası", 
                                 "Model henüz eğitilmemiş. Lütfen önce eğitimi tamamlayın.")
                return
            
            # Save the model using a worker thread [cite: 240]
            self.worker = Worker(
                task="save_model",
                data={"file_path": model_path},
                settings=self.settings,
                processor=self.processor, # [cite: 241]
                midi_model=self.midi_model,
                midi_memory=self.midi_memory
            )
            
            self.worker.task_finished.connect(self.on_model_save_finished) # Corrected to connect to the single instance
            self.worker.error.connect(self.on_worker_error)
            
            self.worker_task = "save_model" # [cite: 242]
            self.worker.start()
            
            logger.info(f"Saving model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            QMessageBox.critical(self, "Model Kaydetme Hatası", f"Model kaydedilirken hata oluştu:\n{e}") # [cite: 243]

    # Removed duplicate on_model_save_finished, on_training_progress, on_training_finished, save_model_after_training definitions.
    # The first instance of each method is kept.

    def on_model_save_finished(self, task_name: str, success: bool, message: str): # Modified to match task_finished signal
        """Handles completion of model saving."""
        # The task_name argument comes from the task_finished signal
        if task_name == "save_model":
            if success:
                logger.info(f"Model successfully saved. Message: {message}")
                QMessageBox.information(self, "Model Kaydedildi", message)
            else:
                logger.error(f"Model saving failed. Message: {message}") # [cite: 244]
                QMessageBox.critical(self, "Model Kaydetme Hatası", message)
        
        self.worker_task = None


    # --- Memory Operations ---
    def populate_memory_list(self):
        """Populates the memory patterns list widget."""
        self.memory_patterns_list.clear() # Clear existing items [cite: 423]
        if self.midi_memory:
            logger.debug("Populating memory list.")
            all_patterns = self.midi_memory.get_all_patterns()
            if all_patterns:
                # Sort patterns by creation date or filename if desired
                # all_patterns.sort(key=lambda p: p.creation_date) [cite: 424]
                for pattern in all_patterns:
                    # Display pattern information (e.g., filename, ID, category)
                    display_text = f"{os.path.basename(pattern.file_path)} (ID: {pattern.pattern_id[:6]})" # Display first 6 chars of ID
                    if pattern.category: # [cite: 425]
                        display_text += f" - Kategori: {pattern.category}"
                    # Add other info like creation date or tags if space allows
                    item = QListWidgetItem(display_text)
                    # Store the full pattern_id in the item's data role for easy retrieval [cite: 426]
                    item.setData(Qt.ItemDataRole.UserRole, pattern.pattern_id)
                    self.memory_patterns_list.addItem(item)
                logger.info(f"Memory list populated with {len(all_patterns)} patterns.")
                self.update_memory_list_visibility()  # Update visibility based on list count
            else:
                logger.info("Memory is empty. No patterns to display.") # [cite: 427]
                self.memory_patterns_list.addItem("Hafıza Boş.")
                self.update_memory_list_visibility()  # Update visibility based on list count
        else:
            logger.error("MIDIMemory is not initialized. Cannot populate list.")
            self.memory_patterns_list.addItem("Hata: Hafıza bileşeni başlatılamadı.")
            
        # Her durumda görünürlüğü güncelle
        self.update_memory_list_visibility()


    def update_memory_list_visibility(self):
        """Hafıza listesinin boş olup olmadığını kontrol eder ve uygun görünümü gösterir."""
        if self.memory_patterns_list.count() > 0:
            # Liste dolu, normal listeyi göster
            self.memory_list_container.setCurrentIndex(0)
        else:
            # Liste boş, bilgi mesajını göster
            self.memory_list_container.setCurrentIndex(1)
    
    def on_memory_pattern_selection_changed(self):
        """Handles selection changes in the memory patterns list."""
        selected_items = self.memory_patterns_list.selectedItems()
        if selected_items and self.midi_memory: # [cite: 428]
            selected_item = selected_items[0] # Assuming single selection mode
            # Retrieve the pattern_id from the item's data
            pattern_id = selected_item.data(Qt.ItemDataRole.UserRole)
            if pattern_id:
                logger.debug(f"Memory pattern selected: ID {pattern_id}")
                # Retrieve the full pattern object from memory [cite: 429]
                selected_pattern = self.midi_memory.get_pattern(pattern_id)
                if selected_pattern and selected_pattern.analysis:
                    logger.info(f"Displaying analysis for selected memory pattern: {os.path.basename(selected_pattern.file_path)}")
                    # Display the analysis results of the selected pattern [cite: 430]
                    self.display_analysis_results(selected_pattern.analysis)
                    # Display piano roll for the selected memory pattern's file
                    if os.path.exists(selected_pattern.file_path):
                        self.display_piano_roll(selected_pattern.file_path) # [cite: 431]
                    else:
                         logger.warning(f"Original file for memory pattern ID {pattern_id} not found at {selected_pattern.file_path}. Cannot display piano roll.") # [cite: 432]
                         self.piano_roll_label.setText("Piyano Rulosu (Dosya Bulunamadı)")
                         self.piano_roll_label.clear() # Clear previous pixmap


                elif selected_pattern: # Pattern found but no analysis
                    logger.warning(f"Selected memory pattern ID {pattern_id} has no analysis data.") # [cite: 433]
                    self.analysis_text_edit.setPlainText(f"Desen: {os.path.basename(selected_pattern.file_path)}\nID: {selected_pattern.pattern_id}\nKategori: {selected_pattern.category or 'Yok'}\nEtiketler: {', '.join(selected_pattern.tags) if selected_pattern.tags else 'Yok'}\nOluşturulma Tarihi: {selected_pattern.creation_date}\n\nAnaliz verisi bulunamadı.")
                    self.piano_roll_label.setText("Piyano Rulosu (Analiz Yok)")
                    self.piano_roll_label.clear() # Clear previous pixmap


                else: # Pattern ID somehow not found in memory data [cite: 434]
                    logger.error(f"Selected pattern ID {pattern_id} not found in memory data dictionary.")
                    self.analysis_text_edit.setPlainText("Hata: Seçilen desen bilgisi hafızada bulunamadı.")
                    self.piano_roll_label.setText("Piyano Rulosu (Hata)")
                    self.piano_roll_label.clear() # [cite: 435]


        elif not selected_items: # No items selected
             logger.debug("Memory pattern selection cleared.")
             # Clear analysis and piano roll display
             self.analysis_text_edit.clear()
             self.piano_roll_label.clear()
             self.piano_roll_label.setText("Piyano Rulosu (Seçilen Desen Yok)") # [cite: 436]


        else: # midi_memory is None
            logger.error("MIDIMemory is not initialized. Cannot retrieve pattern details.") # [cite: 437]
            self.analysis_text_edit.setPlainText("Hata: Hafıza bileşeni başlatılamadı.")
            self.piano_roll_label.setText("Piyano Rulosu (Hata)")
            self.piano_roll_label.clear()


    def add_selected_file_to_memory(self):
        """Adds the currently selected MIDI file and its analysis to memory."""
        if not _core_modules_imported:
             logger.error("Core modules not imported. Cannot add to memory.")
             QMessageBox.critical(self, "Hafıza Hatası", "Uygulama bileşenleri yüklenemedi. Hafızaya eklenemiyor.") # [cite: 438]
             return

        # Check if a file is selected and analyzed, and memory is initialized
        if self.current_midi_file_path and self.current_midi_analysis and self.midi_memory:
            logger.info(f"Attempting to add selected file '{os.path.basename(self.current_midi_file_path)}' to memory.")
            # Get optional category and tags from UI if available (TODO: Add UI elements for these) [cite: 439]
            # For now, add with no category and default tags
            category = None # Get from UI (e.g., a QLineEdit or QComboBox)
            tags = set() # Get from UI (e.g., a QTextEdit parsed into a set)

            # Run add to memory in a worker thread (optional, might be fast) [cite: 440]
            # For simplicity now, let's add directly and rely on MIDIMemory's internal save
            # start_worker("add_pattern_to_memory", {"midi_path": self.current_midi_file_path, "analysis": self.current_midi_analysis, "category": category, "tags": tags})
            try:
                 # Pass file_path and analysis object to add_pattern
                 pattern_id = self.midi_memory.add_pattern(self.current_midi_file_path, self.current_midi_analysis, category, tags) # [cite: 441]

                 if pattern_id:
                      logger.info(f"Selected file added to memory with ID: {pattern_id}")
                      QMessageBox.information(self, "Hafızaya Eklendi", f"'{os.path.basename(self.current_midi_file_path)}' dosyası başarıyla hafızaya eklendi (ID: {pattern_id[:6]}...).")
                      self.populate_memory_list() # Refresh the memory list display [cite: 442]
                      self.update_memory_list_visibility() # Hafıza listesinin görünürlüğünü güncelle
                      # Save memory after adding a pattern
                      try:
                           self.midi_memory.save()
                           logger.info("Memory saved automatically after adding pattern.") # [cite: 443]
                      except Exception as e:
                           logger.error(f"Automatic memory save failed after adding pattern: {e}", exc_info=True)
                           # QMessageBox.warning(self, "Hafıza Kayıt Hatası", "Desen eklendi ancak hafıza otomatik olarak kaydedilemedi.") [cite: 444]


                 else:
                      # add_pattern returns None if it failed (e.g., file not found, already exists based on internal logic)
                      logger.error(f"Failed to add selected file to memory: add_pattern returned None.") # [cite: 445]
                      QMessageBox.critical(self, "Hafıza Hatası", f"'{os.path.basename(self.current_midi_file_path)}' dosyası hafızaya eklenirken hata oluştu (add_pattern returned None).")

            except Exception as e:
                 logger.error(f"Error calling midi_memory.add_pattern: {e}", exc_info=True)
                 QMessageBox.critical(self, "Hafıza Hatası", f"'{os.path.basename(self.current_midi_file_path)}' dosyası hafızaya eklenirken beklenmeyen hata oluştu:\n{e}")


        elif not self.current_midi_file_path: # [cite: 446]
            logger.warning("No file selected to add to memory.")
            QMessageBox.warning(self, "Hafızaya Ekle", "Lütfen önce hafızaya eklemek istediğiniz MIDI dosyasını seçin.")
        elif not self.current_midi_analysis:
            logger.warning("Selected file has no analysis data to add to memory.")
            QMessageBox.warning(self, "Hafızaya Ekle", "Seçilen dosyanın analiz verisi bulunmuyor. Lütfen önce dosyayı analiz edin.") # [cite: 447]
        elif not self.midi_memory:
             logger.error("MIDIMemory is not initialized. Cannot add pattern.")
             QMessageBox.critical(self, "Hafıza Hatası", "Hata: Hafıza bileşeni başlatılamadı. Desen eklenemedi.")


    def search_similar_patterns(self):
        """Searches for similar patterns in memory based on the current analysis."""
        if not _core_modules_imported:
             logger.error("Core modules not imported. Cannot search memory.") # [cite: 448]
             QMessageBox.critical(self, "Hafıza Hatası", "Uygulama bileşenleri yüklenemedi. Arama yapılamıyor.")
             return

        if self.current_midi_analysis and self.midi_memory:
            logger.info("Initiating search for similar patterns in memory.")
            self.similar_patterns_text_edit.clear()
            self.similar_patterns_text_edit.insertPlainText("Benzer desenler aranıyor...\n")
            self.statusBar.showMessage("Benzer desenler aranıyor...", 0) # [cite: 449]
            self.progress_bar.show()
            self.progress_bar.setRange(0, 0) # Indeterminate


            # Run search in a worker thread
            # Pass necessary components to the worker
            self.worker = Worker(task="find_similar_patterns", data={"reference_analysis": self.current_midi_analysis},
                                 settings=self.settings, processor=self.processor, midi_model=self.midi_model, midi_memory=self.midi_memory) # [cite: 450]
            # Connect signals
            # The signal definition was pyqtSignal(list) - so expect list or None
            self.worker.memory_search_finished.connect(self.on_memory_search_finished)
            self.worker.error.connect(self.on_worker_error)
            # self.worker.task_finished.connect(self.on_worker_task_finished) # Connect if using generic signal for this task [cite: 451]

            self.worker_task = "find_similar_patterns"
            self.worker.start()


        elif not self.current_midi_analysis:
            logger.warning("No analysis available to search for similar patterns.")
            QMessageBox.warning(self, "Hafıza Araması", "Lütfen önce arama yapmak istediğiniz deseni (MIDI dosyasını) analiz edin.")
            self.similar_patterns_text_edit.setPlainText("Arama için analiz verisi bulunmuyor.") # [cite: 452]
        elif not self.midi_memory:
            logger.error("MIDIMemory is not initialized. Cannot search memory.") # [cite: 453]
            QMessageBox.critical(self, "Hafıza Hatası", "Hata: Hafıza bileşeni başlatılamadı. Arama yapılamadı.")


    # Slot receives List[MIDIPattern] or None
    def on_memory_search_finished(self, similar_patterns: Optional[List[MIDIPattern]]): # Signal signature was list, changing to List[MIDIPattern] based on worker emit
        """Slot to receive similar patterns list from the worker thread."""
        self.statusBar.clearMessage()
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100) # Reset

        self.similar_patterns_text_edit.clear() # Clear the "searching..." message [cite: 454]

        if similar_patterns is not None and self.current_midi_analysis and self.midi_memory: # Ensure list is not None and components are available
            logger.info(f"Memory search finished. Found {len(similar_patterns)} similar patterns.")
            self.similar_patterns_text_edit.appendPlainText(f"'{os.path.basename(self.current_midi_file_path or 'Seçili Dosya')}' desenine benzer bulunanlar:")

            # Recalculate score here to display it next to the pattern name
            # Sort patterns by score descending for consistent display [cite: 455]
            similar_patterns_with_score = []
            for pattern in similar_patterns:
                 # Ensure pattern and its analysis are valid before calculating score
                 if pattern and pattern.analysis:
                      # Use the midi_memory's calculate_similarity_score method [cite: 456]
                      # Ensure similarity_settings are available in midi_memory
                      if hasattr(self.midi_memory, 'similarity_settings') and self.midi_memory.similarity_settings:
                           score = self.midi_memory._calculate_similarity_score(self.current_midi_analysis, pattern.analysis, self.midi_memory.similarity_settings) # [cite: 457]
                           similar_patterns_with_score.append((pattern, score))
                      else:
                           logger.warning("MIDIMemory similarity settings not available. Cannot calculate scores.") # [cite: 458]
                           similar_patterns_with_score.append((pattern, 0.0)) # Add with score 0

            # Sort by score descending
            similar_patterns_with_score.sort(key=lambda item: item[1], reverse=True)

            # Display sorted results
            for pattern, score in similar_patterns_with_score:
                 display_text = f"- {os.path.basename(pattern.file_path)} (Benzerlik: {score:.2f}) (ID: {pattern.pattern_id[:6]}...)" # [cite: 459]
                 self.similar_patterns_text_edit.appendPlainText(display_text)

        elif similar_patterns is not None and not similar_patterns: # Empty list received
            logger.info("No similar patterns found in memory.")
            self.similar_patterns_text_edit.setPlainText(f"'{os.path.basename(self.current_midi_file_path or 'Seçili Dosya')}' desenine benzer desen bulunamadı.")

        else: # similar_patterns is None (error occurred in worker) [cite: 460]
            logger.error("Memory search failed in worker thread.")
            self.similar_patterns_text_edit.setPlainText("Hafıza araması sırasında bir hata oluştu.")


    # --- UI Helpers ---
    def apply_dark_theme(self):
        """Applies a dark theme from a QSS file if it exists."""
        # Get project root path relative to main_window.py for styles.qss
        # A simple relative path from src/gui to src/gui/styles.qss is './styles.qss'
        # Or use the settings object to get resource paths [cite: 473]
        if self.settings:
             project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Calculate project root directly
             qss_path = os.path.join(project_root_path, 'src', 'gui', 'styles.qss') # Path to the QSS file
             if os.path.exists(qss_path):
                  try:
                      with open(qss_path, "r", encoding='utf-8') as f: # Added encoding='utf-8' [cite: 474]
                          self.setStyleSheet(f.read())
                      logger.info(f"Applied QSS from {qss_path}")
                  except Exception as e:
                       logger.error(f"Error applying QSS from {qss_path}: {e}", exc_info=True) # [cite: 475]
             else:
                  logger.warning(f"QSS style file not found at: {qss_path}. Dark theme not applied.") # [cite: 476]
        else:
             logger.warning("Settings object not available. Cannot apply dark theme.")


    # --- Worker Error Handling ---
    def on_worker_error(self, task_name: str, error_message: str):
        """Slot to handle errors emitted by the worker thread."""
        logger.error(f"Error in worker during task '{task_name}': {error_message}")
        self.statusBar.clearMessage()
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100) # Reset [cite: 477]

        # Show an error message box to the user
        QMessageBox.critical(self, f"Görev Hatası: {task_name}", f"'{task_name}' görevi sırasında bir hata oluştu:\n{error_message}\n\nLütfen log dosyasına bakın.")

        # Reset worker task status
        self.worker_task = None


    # --- Event Handling ---
    def apply_shadow_effects(self):
        """Butonlara ve etiketlere gölge efektleri uygular."""
        try:
            from PyQt6.QtWidgets import QGraphicsDropShadowEffect
            from PyQt6.QtGui import QColor
            
            # Gölge efekti oluştur - butonlar için
            button_shadow = QGraphicsDropShadowEffect()
            button_shadow.setBlurRadius(15)
            button_shadow.setColor(QColor(0, 0, 0, 160))
            button_shadow.setOffset(3, 3)
            
            # Gölge efekti oluştur - başlıklar için
            label_shadow = QGraphicsDropShadowEffect()
            label_shadow.setBlurRadius(10)
            label_shadow.setColor(QColor(0, 0, 0, 160))
            label_shadow.setOffset(2, 2)
            
            # Önemli butonlara gölge efekti uygula
            if hasattr(self, 'generate_button'):
                self.generate_button.setGraphicsEffect(QGraphicsDropShadowEffect(self.generate_button))
                self.generate_button.graphicsEffect().setBlurRadius(15)
                self.generate_button.graphicsEffect().setColor(QColor(0, 0, 0, 160))
                self.generate_button.graphicsEffect().setOffset(3, 3)
            
            # Diğer önemli butonlara gölge efekti uygula
            important_buttons = [
                'btn_analyze', 'btn_train', 'add_to_memory_button', 
                'search_memory_button', 'save_settings_button', 'load_settings_button'
            ]
            
            for button_name in important_buttons:
                if hasattr(self, button_name):
                    button = getattr(self, button_name)
                    shadow_effect = QGraphicsDropShadowEffect(button)
                    shadow_effect.setBlurRadius(10)
                    shadow_effect.setColor(QColor(0, 0, 0, 160))
                    shadow_effect.setOffset(2, 2)
                    button.setGraphicsEffect(shadow_effect)
            
            # Önemli başlıklara gölge efekti uygula
            important_labels = [
                'title_label', 'generation_group', 'memory_group', 'settings_group'
            ]
            
            for label_name in important_labels:
                if hasattr(self, label_name):
                    label = getattr(self, label_name)
                    shadow_effect = QGraphicsDropShadowEffect(label)
                    shadow_effect.setBlurRadius(8)
                    shadow_effect.setColor(QColor(0, 0, 0, 160))
                    shadow_effect.setOffset(1, 1)
                    label.setGraphicsEffect(shadow_effect)
            
            logger.info("Gölge efektleri başarıyla uygulandı.")
        except Exception as e:
            logger.error(f"Gölge efektleri uygulanırken hata oluştu: {e}", exc_info=True)
    
    def setup_background_image(self):
        """Arka plan resmini yükler ve ölçeklendirme ayarlarını yapar."""
        try:
            # Resmi yeniden ölçeklendir - kullanıcının belirttiği konum
            background_path = os.path.join(os.path.dirname(__file__), 'resources', 'background.jpg')
            
            # Eğer resim dosyası yoksa veya boşsa, yeni bir görüntü oluştur
            if not os.path.exists(background_path) or os.path.getsize(background_path) < 1000:
                # resources/images klasörünün var olduğundan emin ol
                os.makedirs(os.path.dirname(background_path), exist_ok=True)
                
                # Varsayılan bir görüntü oluştur
                self.create_default_background(background_path)
                logger.info(f"Varsayılan arka plan resmi oluşturuldu: {background_path}")
            
            # Resmi yükle
            background = QPixmap(background_path)
            if background.isNull():
                logger.warning(f"Arka plan resmi yüklenemedi: {background_path}")
                # Varsayılan bir görüntü oluştur
                self.create_default_background(background_path)
                background = QPixmap(background_path)
                if background.isNull():
                    logger.error("Varsayılan arka plan resmi de yüklenemedi.")
                    return
            
            # Arka plan resmini ayarla
            self.set_background_image(background)
            
            # Pencere yeniden boyutlandırıldığında resmi güncellemek için olay bağlantısı
            self.resizeEvent = self.on_resize
            
            # Otomatik arka plan doldurma özelliğini etkinleştir
            self.setAutoFillBackground(True)
            
            logger.info(f"Arka plan resmi başarıyla ayarlandı: {background_path}")
        except Exception as e:
            logger.error(f"Arka plan resmi ayarlanırken hata oluştu: {e}", exc_info=True)
    
    def apply_shadow_effects(self):
        """Önemli UI bileşenlerine gölge efekti uygular."""
        try:
            # Gölge efekti için QGraphicsDropShadowEffect kullan
            # Önce widget'ların var olup olmadığını kontrol et
            widgets_to_shadow = []
            
            # Ana grup kutuları
            if hasattr(self, 'file_group'):
                widgets_to_shadow.append(self.file_group)
            
            if hasattr(self, 'midi_uretim_group'):
                widgets_to_shadow.append(self.midi_uretim_group)
            
            if hasattr(self, 'bottom_tab_widget'):
                widgets_to_shadow.append(self.bottom_tab_widget)
            
            # Butonlar
            if hasattr(self, 'generate_button'):
                widgets_to_shadow.append(self.generate_button)
            
            if hasattr(self, 'browse_button'):
                widgets_to_shadow.append(self.browse_button)
            
            if hasattr(self, 'analyze_button'):
                widgets_to_shadow.append(self.analyze_button)
            
            # Gölge efektlerini uygula
            for widget in widgets_to_shadow:
                shadow = QGraphicsDropShadowEffect(self)
                shadow.setBlurRadius(15)
                shadow.setColor(QColor(0, 0, 0, 180))
                shadow.setOffset(3, 3)
                widget.setGraphicsEffect(shadow)
                
            logger.info("Gölge efektleri başarıyla uygulandı")
        except Exception as e:
            logger.error(f"Gölge efektleri uygulanırken hata oluştu: {e}", exc_info=True)
    
    def create_default_background(self, save_path):
        """Müzik temalı bir arka plan resmi oluşturur ve kaydeder."""
        try:
            # Koyu arka plan oluştur - müzik stüdyosu teması
            width, height = 1920, 1080  # Yüksek çözünürlüklü arka plan
            image = QImage(width, height, QImage.Format.Format_ARGB32)
            
            # Koyu arka plan rengi
            background_color = qRgb(20, 22, 36)  # Koyu lacivert/siyah
            image.fill(background_color)
            
            # Resmi çizmek için painter oluştur
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            
            # Müzik notaları ve semboller çiz
            # Nota renkleri
            note_colors = [
                QColor(41, 128, 185),   # Mavi
                QColor(142, 68, 173),   # Mor
                QColor(192, 57, 43),    # Kırmızı
                QColor(211, 84, 0),     # Turuncu
                QColor(39, 174, 96),    # Yeşil
                QColor(241, 196, 15)    # Sarı
            ]
            
            # Rastgele notalar ve semboller çiz
            for i in range(50):
                # Rastgele pozisyon
                x = random.randint(0, width)
                y = random.randint(0, height)
                size = random.randint(30, 150)
                color = note_colors[random.randint(0, len(note_colors) - 1)]
                
                # Rastgele şekil seç
                shape_type = random.randint(0, 3)
                
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                
                # Farklı müzik sembolleri çiz
                if shape_type == 0:  # Daire (nota)
                    painter.drawEllipse(x, y, size, size)
                elif shape_type == 1:  # Kare (tuş)
                    painter.drawRect(x, y, size, size // 2)
                elif shape_type == 2:  # Üçgen (plektrum)
                    points = [QPoint(x, y), QPoint(x + size, y), QPoint(x + size // 2, y + size)]
                    painter.drawPolygon(points)
                else:  # Yarım daire (kulaklık)
                    painter.drawChord(x, y, size, size, 0, 180 * 16)
            
            # Parlak çizgiler ekle (müzik dalgaları)
            for i in range(15):
                y = random.randint(0, height)
                color = note_colors[random.randint(0, len(note_colors) - 1)]
                color.setAlpha(100)  # Yarı saydam
                
                pen = QPen(color, random.randint(1, 5))
                painter.setPen(pen)
                
                # Dalgalı çizgi çiz
                path = QPainterPath()
                path.moveTo(0, y)
                
                segment_count = 10
                segment_width = width / segment_count
                
                for j in range(1, segment_count + 1):
                    # Sinüs dalgası benzeri hareket
                    control_y = y + random.randint(-100, 100)
                    end_y = y + random.randint(-50, 50)
                    
                    path.quadTo(j * segment_width - segment_width/2, control_y, j * segment_width, end_y)
                
                painter.drawPath(path)
            
            # Merkeze "MIDI COMPOSER" yazısı ekle
            title_font = QFont("Arial", 120, QFont.Weight.Bold)
            painter.setFont(title_font)
            
            # Gölge efekti için önce koyu renkle yaz
            shadow_color = QColor(0, 0, 0, 180)
            painter.setPen(shadow_color)
            painter.drawText(QRect(5, 105, width, 200), Qt.AlignmentFlag.AlignCenter, "MIDI COMPOSER")
            
            # Ana yazı
            gradient = QLinearGradient(0, 0, width, 200)
            gradient.setColorAt(0, QColor(41, 128, 185))  # Mavi
            gradient.setColorAt(0.5, QColor(142, 68, 173))  # Mor
            gradient.setColorAt(1, QColor(192, 57, 43))  # Kırmızı
            
            painter.setPen(QPen(QBrush(gradient), 2))
            painter.drawText(QRect(0, 100, width, 200), Qt.AlignmentFlag.AlignCenter, "MIDI COMPOSER")
            
            # Resmi tamamla
            painter.end()
            
            # Resmi kaydet
            image.save(save_path, "JPG", 95)
            logger.info(f"Müzik temalı arka plan resmi başarıyla oluşturuldu: {save_path}")
            return True
        except Exception as e:
            logger.error(f"Arka plan oluşturulurken hata: {e}", exc_info=True)
            return False
    
    def set_background_image(self, pixmap):
        """Arka plan resmini ölçeklendirip uygulamaya ekler."""
        try:
            # Resmi pencere boyutuna göre ölçeklendir
            scaled_pixmap = pixmap.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Palette oluştur ve güncelle
            palette = self.palette()
            palette.setBrush(QPalette.ColorRole.Window, QBrush(scaled_pixmap))
            
            # Central widget'a palette'i uygula
            self.central_widget.setPalette(palette)
            self.central_widget.setAutoFillBackground(True)
            
            # Ana pencereye de palette'i uygula
            self.setPalette(palette)
            
            logger.debug("Arka plan resmi başarıyla ölçeklendirildi ve uygulandı.")
            return True
        except Exception as e:
            logger.error(f"Arka plan resmi ölçeklendirilirken hata: {e}", exc_info=True)
            return False
    
    def on_resize(self, event):
        """Pencere boyutu değiştiğinde arka plan resmini yeniden ölçeklendir."""
        try:
            # Orijinal QMainWindow.resizeEvent metodunu çağır
            super().resizeEvent(event)
            
            # Arka plan etiketinin boyutunu pencere boyutuna göre ayarla
            if hasattr(self, 'background_label'):
                # Sadece etiketin boyutunu güncelle, içerik otomatik olarak ölçeklenecek (setScaledContents(True) sayesinde)
                self.background_label.setGeometry(0, 0, self.width(), self.height())
                
                # Arka plan etiketini en alt katmana yerleştir
                self.background_label.lower()
                
                logger.debug(f"Arka plan etiketi yeniden boyutlandırıldı: {self.width()}x{self.height()}")
        except Exception as e:
            logger.error(f"Arka plan resmi yeniden ölçeklendirilirken hata oluştu: {e}", exc_info=True)
    
    def closeEvent(self, event):
        """Handles the window close event."""
        logger.info("Window close event triggered. Attempting to stop workers.") # [cite: 478]
        # Stop any running worker threads before closing
        if self.worker and self.worker.isRunning():
            logger.info(f"Stopping active worker task: {self.worker_task}")
            self.worker.requestInterruption() # Request the worker to stop gracefully
            # Give the worker a moment to finish clean-up after interruption request
            if not self.worker.wait(500): # Wait up to 500ms [cite: 479]
                logger.warning(f"Worker task '{self.worker_task}' did not respond to interruption request, attempting quit.")
                self.worker.quit() # Quit the worker thread event loop
                if not self.worker.wait(500): # Wait another 500ms
                     logger.warning(f"Worker task '{self.worker_task}' did not quit, attempting terminate.") # [cite: 480]
                     self.worker.terminate() # Force terminate (use as last resort)
                     self.worker.wait(500) # Wait briefly for termination


        logger.info("Application window closing.")
        event.accept() # Accept the close event
        
    def browse_midi_file(self):
        """MIDI dosyası seçmek için dosya tarayıcısını açar."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "MIDI Dosyası Seçin",
                "",
                "MIDI Files (*.mid *.midi);;All Files (*)"
            )
            
            if file_path:
                self.midi_path_edit.setText(file_path)
                self.current_midi_path = file_path
                logger.info(f"MIDI dosyası seçildi: {file_path}")
        except Exception as e:
            logger.error(f"MIDI dosyası seçilirken hata oluştu: {e}", exc_info=True)
            QMessageBox.critical(self, "Hata", f"MIDI dosyası seçilirken hata oluştu: {e}")
            
    def on_analyze_button_clicked(self):
        """Analiz butonu tıklandığında MIDI dosyasını analiz eder."""
        try:
            if not hasattr(self, 'current_midi_path') or not self.current_midi_path:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir MIDI dosyası seçin.")
                return
                
            # Analiz işlemi başladığını göster
            self.statusBar().showMessage("MIDI dosyası analiz ediliyor...")
            QApplication.processEvents()
            
            # MIDI dosyasını analiz et
            midi_data = self.midi_processor.load_midi(self.current_midi_path)
            analysis_result = self.midi_processor.analyze_midi(midi_data)
            
            # Analiz sonuçlarını görüntüle
            self.display_midi_analysis(analysis_result)
            
            # Piyano rulosunu görüntüle
            self.display_piano_roll(midi_data)
            
            # Analiz tamamlandığını göster
            self.statusBar().showMessage("MIDI dosyası analizi tamamlandı.", 5000)
            logger.info(f"MIDI dosyası analiz edildi: {self.current_midi_path}")
        except Exception as e:
            logger.error(f"MIDI dosyası analiz edilirken hata oluştu: {e}", exc_info=True)
            QMessageBox.critical(self, "Hata", f"MIDI dosyası analiz edilirken hata oluştu: {e}")
            self.statusBar().showMessage("MIDI dosyası analizi sırasında hata oluştu.", 5000)
            
    def display_midi_analysis(self, analysis_result):
        """MIDI analiz sonuçlarını metin alanında görüntüler."""
        try:
            # Analiz sonuçlarını biçimlendir
            analysis_text = "MIDI Analiz Sonuçları:\n\n"
            
            # Temel bilgiler
            if 'track_count' in analysis_result:
                analysis_text += f"Toplam Parça Sayısı: {analysis_result['track_count']}\n"
            if 'tempo' in analysis_result:
                analysis_text += f"Tempo: {analysis_result['tempo']} BPM\n"
            if 'time_signature' in analysis_result:
                analysis_text += f"Zaman İmzası: {analysis_result['time_signature']}\n"
            if 'key_signature' in analysis_result:
                analysis_text += f"Anahtar: {analysis_result['key_signature']}\n"
            if 'duration' in analysis_result:
                analysis_text += f"Süre: {analysis_result['duration']:.2f} saniye\n"
            
            # Enstrüman bilgileri
            if 'instruments' in analysis_result and analysis_result['instruments']:
                analysis_text += "\nEnstrümanlar:\n"
                for instrument in analysis_result['instruments']:
                    analysis_text += f"- {instrument}\n"
            
            # Not istatistikleri
            if 'note_stats' in analysis_result:
                stats = analysis_result['note_stats']
                analysis_text += "\nNot İstatistikleri:\n"
                analysis_text += f"- Toplam Not Sayısı: {stats.get('total_notes', 0)}\n"
                analysis_text += f"- Ortalama Not Süresi: {stats.get('avg_note_duration', 0):.2f} vuruş\n"
                analysis_text += f"- En Yüksek Nota: {stats.get('highest_note', 'Bilinmiyor')}\n"
                analysis_text += f"- En Düşük Nota: {stats.get('lowest_note', 'Bilinmiyor')}\n"
                analysis_text += f"- Nota Aralığı: {stats.get('note_range', 0)} semitone\n"
            
            # Metin alanına analiz sonuçlarını ekle
            self.analysis_text_edit.setText(analysis_text)
            logger.info("MIDI analiz sonuçları görüntülendi.")
        except Exception as e:
            logger.error(f"MIDI analiz sonuçları görüntülenirken hata oluştu: {e}", exc_info=True)
            self.analysis_text_edit.setText("Analiz sonuçları görüntülenirken hata oluştu.")
    
    def display_piano_roll(self, midi_data):
        """MIDI verilerinden piyano rulosu görüntüsü oluşturur ve gösterir."""
        try:
            # Geçici dosya oluştur
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, "piano_roll.png")
            
            # Piyano rulosu görüntüsü oluştur
            self.midi_processor.create_piano_roll_image(midi_data, temp_file)
            
            # Görüntüyü yükle ve göster
            if os.path.exists(temp_file):
                pixmap = QPixmap(temp_file)
                if not pixmap.isNull():
                    # Görüntüyü etiket boyutuna göre ölçeklendir
                    scaled_pixmap = pixmap.scaled(
                        self.piano_roll_label.width(),
                        self.piano_roll_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.piano_roll_label.setPixmap(scaled_pixmap)
                    logger.info("Piyano rulosu görüntüsü başarıyla oluşturuldu ve gösterildi.")
                else:
                    self.piano_roll_label.setText("Piyano rulosu görüntüsü oluşturulamadı.")
                    logger.error("Piyano rulosu görüntüsü yüklenemedi.")
            else:
                self.piano_roll_label.setText("Piyano rulosu görüntüsü oluşturulamadı.")
                logger.error(f"Piyano rulosu görüntü dosyası bulunamadı: {temp_file}")
        except Exception as e:
            logger.error(f"Piyano rulosu görüntülenirken hata oluştu: {e}", exc_info=True)
            self.piano_roll_label.setText("Piyano rulosu görüntülenirken hata oluştu.")



# --- Entry Point for Testing (if run directly, not via main.py) ---
# This block is for testing main_window.py in isolation and should not configure logging if main.py runs it. [cite: 481]
if __name__ == "__main__": # [cite: 482]
    # Configure basic logging only if running THIS file directly for testing
    # In a real application, logging is configured at the main entry point (main.py).
    # Check if root logger has handlers to avoid reconfiguring if main.py already did. [cite: 483]
    if not logging.getLogger('').handlers: # [cite: 484]
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(funcName)s: %(message)s")

    # Get logger for this module AFTER basic config if run directly
    logger = logging.getLogger(__name__)
    logger.info("Running main_window.py test (standalone).")

    # --- Dummy Settings for Standalone Test ---
    # Create a dummy settings object for standalone testing
    # In the main application, a real Settings object is created and passed by main.py
    @dataclass
    class DummyModelSettings:
        note_range: Tuple[int, int] = (21, 108) # [cite: 485]
        resolution: float = 0.125
        input_features: int = 1 # Match what sequence representation is
        sequence_length: int = 32 # Example model sequence length


    @dataclass
    class DummyGeneralSettings:
         output_dir: str = "generated_midi_test_standalone"
         model_dir: str = "trained_model_test_standalone"
         memory_dir: str = "memory_test_dir_standalone"
         memory_file: str = "midi_memory_test_standalone.json" # [cite: 486]


    @dataclass
    class DummyMemorySettings:
         similarity_settings: Dict[str, Any] = field(default_factory=dict) # Dummy similarity settings
         # Add other memory settings used by MIDIMemory if any


    @dataclass
    class DummySettings:
         model_settings: DummyModelSettings = field(default_factory=DummyModelSettings)
         general_settings: DummyGeneralSettings = field(default_factory=DummyGeneralSettings)
         memory_settings: DummyMemorySettings = field(default_factory=DummyMemorySettings)
         # Add other settings sections if needed [cite: 487]

         # Need a dummy _get_project_root or hardcode paths for standalone testing
         def _get_project_root(self):
              # Assume project root is two levels up from src/gui
              return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

         @property
         def output_dir_path(self):
              # Ensure general_settings is not None before accessing its attribute [cite: 488]
              return os.path.join(self._get_project_root(), self.general_settings.output_dir) if self.general_settings and self.general_settings.output_dir else os.path.join(self._get_project_root(), "generated_midi_test_default")


         @property
         def model_dir_path(self):
              # Ensure general_settings is not None before accessing its attribute
              return os.path.join(self._get_project_root(), self.general_settings.model_dir) if self.general_settings and self.general_settings.model_dir else os.path.join(self._get_project_root(), "trained_model_test_default") # [cite: 489]


         @property
         def memory_file_full_path(self):
              # Ensure general_settings is not None before accessing its attribute
              memory_dir = os.path.join(self._get_project_root(), self.general_settings.memory_dir) if self.general_settings and self.general_settings.memory_dir else os.path.join(self._get_project_root(), "memory_test_dir_default")
              memory_file = self.general_settings.memory_file if self.general_settings and self.general_settings.memory_file else "midi_memory_test_default.json"
              return os.path.join(memory_dir, memory_file) # [cite: 490]


         # Dummy save method (returns string if file_path is None, simulates file save otherwise)
         def save(self, file_path: Optional[str] = None):
              logger.debug(f"Dummy Settings: save called (path={file_path})")
              try:
                  settings_dict = asdict(self) # [cite: 491]
                  # Convert any sets to lists for JSON compatibility
                  # This might need to be recursive for nested structures
                  # For simplicity, handle known sets if they exist in memory_settings.similarity_settings
                  if 'memory_settings' in settings_dict and 'similarity_settings' in settings_dict['memory_settings']: # [cite: 492]
                       sim_settings = settings_dict['memory_settings']['similarity_settings']
                       # Example: if 'tags' is a set
                       if 'tags' in sim_settings and isinstance(sim_settings['tags'], set):
                           sim_settings['tags'] = list(sim_settings['tags']) # [cite: 493]
                       # Add handling for other potential sets here


                  if file_path is None:
                       return json.dumps(settings_dict, indent=4) # Return string [cite: 494]

                  # Simulate file save
                  output_dir = os.path.dirname(file_path)
                  if output_dir and not os.path.exists(output_dir):
                       os.makedirs(output_dir, exist_ok=True) # Use exist_ok=True
                  logger.info(f"Dummy Settings: Simulating save to {file_path}") # [cite: 495]
                  # Optionally write dummy content
                  # with open(file_path, 'w', encoding='utf-8') as f:
                  #      json.dump(settings_dict, f, indent=4)
                  return None # Indicate success [cite: 496]


              except Exception as e:
                  logger.error(f"Error serializing dummy settings for display/save: {e}", exc_info=True)
                  return f"Error serializing dummy settings: {e}" # Return error message string


         # Dummy load method (doesn't actually load from file)
         def load(self, file_path: Optional[str] = None): # [cite: 497]
              logger.debug(f"Dummy Settings: load called (path={file_path}) - Using default settings.")
              # In a real test, you might load from a dummy file content
              pass


    # --- Dummy MIDI Processor and Model for Standalone Test ---
    # Need minimal dummy implementations if testing MainWindow in isolation requires these objects to be non-None [cite: 498]
    # These dummy objects should mimic the expected interface (methods like analyze_midi_file, generate_sequence etc.)
    class DummyMIDIProcessor:
         def __init__(self, settings=None, instrument_library=None):
              logger.debug("DummyMIDIProcessor initialized.")
              self.settings = settings
              self.instrument_library = instrument_library
              # Add dummy properties if needed by MainWindow logic [cite: 499]
              self.note_range = settings.model_settings.note_range if settings and settings.model_settings else (21, 108)
              self.note_range_size = self.note_range[1] - self.note_range[0] + 1
              self.resolution = settings.model_settings.resolution if settings and settings.model_settings else 0.125
              self.input_features = settings.model_settings.input_features if settings and settings.model_settings else 1
              self._seconds_per_step = self.resolution * (60.0 / 120.0) # Dummy calculation [cite: 500]


         def analyze_midi_file(self, midi_file_path: str) -> Optional[MIDIAnalysis]:
              logger.debug(f"DummyMIDIProcessor: analyze_midi_file called with {midi_file_path}")
              # Return a dummy analysis result
              if os.path.exists(midi_file_path):
                   logger.debug("Dummy analysis: File exists, returning dummy analysis.") # [cite: 501]
                   # Create a minimal dummy MIDIAnalysis object
                   dummy_analysis = MIDIAnalysis(
                        file_path=midi_file_path,
                        duration=10.0, # [cite: 502]
                        tempo=120.0,
                        key="C Major (Dummy)",
                        time_signature="4/4 (Dummy)",
                        note_count=50, # [cite: 503]
                        average_velocity=80.0,
                        pitch_range=(60, 72),
                        polyphony_profile={0.0: 0, 1.0: 3, 2.0: 2},
                        rhythm_complexity=0.1, # [cite: 504]
                        instrument_programs={0: 50},
                        instrument_names={"Acoustic Grand Piano (Dummy)": 50}
                   )
                   return dummy_analysis
              else: # [cite: 505]
                   logger.warning("Dummy analysis: File not found, returning None.")
                   return None


         def midi_to_sequence(self, midi_file_path: str) -> Optional[np.ndarray]:
              logger.debug(f"DummyMIDIProcessor: midi_to_sequence called with {midi_file_path}")
              # Return a dummy sequence (e.g., 1 bar of silence) [cite: 506]
              if self.settings and self.settings.model_settings:
                   seq_len = self.settings.model_settings.sequence_length
                   note_range_size = self.settings.model_settings.note_range[1] - self.settings.model_settings.note_range[0] + 1
                   input_features = self.settings.model_settings.input_features
                   dummy_seq = np.zeros((seq_len, note_range_size, input_features), dtype=np.float32) # [cite: 507]
                   return dummy_seq
              else:
                   logger.warning("DummyMIDIProcessor: Settings not available for dummy sequence, returning None.")
                   return None


         def sequence_to_midi(self, sequence: np.ndarray, output_path: str, tempo: float = 120.0) -> bool: # [cite: 508]
              logger.debug(f"DummyMIDIProcessor: sequence_to_midi called, simulating save to {output_path}")
              # Simulate saving by creating the directory
              output_dir = os.path.dirname(output_path)
              if output_dir and not os.path.exists(output_dir):
                  os.makedirs(output_dir, exist_ok=True) # [cite: 509]
              logger.info(f"DummyMIDIProcessor: Simulated saving dummy MIDI to {output_path}")
              return True


    class DummyMIDIModel:
         def __init__(self, settings=None):
              logger.debug("DummyMIDIModel initialized.")
              self.settings = settings
              self.model = "Dummy Trained Model" # Placeholder to indicate it's 'built' [cite: 510]
              self._is_trained = True # Assume it's trained for testing generation


         def build_model(self):
              logger.debug("DummyMIDIModel: build_model called.")
              # In a real model, this would build the Keras model
              pass # [cite: 511]


         def generate_sequence(self, seed_sequence: np.ndarray, num_steps: int, temperature: float) -> Optional[np.ndarray]:
              logger.debug(f"DummyMIDIModel: generate_sequence called (seed shape: {seed_sequence.shape}, steps: {num_steps})")
              # Return a dummy generated sequence
              if self.settings and self.settings.model_settings:
                   seq_len = self.settings.model_settings.sequence_length # [cite: 512]
                   note_range_size = self.settings.model_settings.note_range[1] - self.settings.model_settings.note_range[0] + 1
                   input_features = self.settings.model_settings.input_features
                   # Create a dummy sequence of the requested length
                   dummy_generated_seq = np.random.rand(num_steps, note_range_size, input_features).astype(np.float32)
                   # For binary pitch, convert feature 0 to 0 or 1 [cite: 513]
                   dummy_generated_seq[:, :, 0] = (dummy_generated_seq[:, :, 0] > 0.9).astype(np.float32) # Sparse dummy notes
                   return dummy_generated_seq
              else:
                   logger.warning("DummyMIDIModel: Settings not available for dummy generation, returning None.") # [cite: 514]
                   return None

         def train(self, training_data, epochs: int, validation_data=None, callbacks=None):
              logger.debug(f"DummyMIDIModel: train called (epochs: {epochs})")
              # Simulate training progress and completion
              if callbacks: # [cite: 515]
                   class DummyLogs: # Minimal logs object
                        def get(self, key, default): return default
                   dummy_logs = DummyLogs()
                   for epoch in range(epochs): # [cite: 516]
                        logger.debug(f"Dummy training: Epoch {epoch + 1}/{epochs}")
                        if callbacks:
                             for callback in callbacks:
                                   if hasattr(callback, 'on_epoch_end'): # [cite: 517]
                                       callback.on_epoch_end(epoch, dummy_logs)
                                   import time # [cite: 518]
                                   time.sleep(0.1) # Simulate some work
              logger.info("Dummy training finished.")
              return "Dummy History" # Return a dummy history object


         def save_model(self, file_path: str) -> bool:
              logger.debug(f"DummyMIDIModel: save_model called, simulating save to {file_path}") # [cite: 519]
              # Simulate saving by creating the directory
              output_dir = os.path.dirname(file_path)
              if output_dir and not os.path.exists(output_dir):
                   os.makedirs(output_dir, exist_ok=True)
              logger.info(f"DummyMIDIModel: Simulated saving dummy model to {file_path}") # [cite: 520]
              return True

         def load_model(self, file_path: str) -> bool:
              logger.debug(f"DummyMIDIModel: load_model called, simulating load from {file_path}")
              # Simulate loading - check if file exists (optional)
              if os.path.exists(file_path):
                   logger.info(f"DummyMIDIModel: Simulated loading model from {file_path}") # [cite: 521]
                   self._is_trained = True # Assume loaded model is trained
                   return True
              else:
                   logger.warning(f"DummyMIDIModel: Model file not found at {file_path}, simulating load failure.") # [cite: 522]
                   self._is_trained = False
                   return False


    class DummyMIDIMemory:
         def __init__(self, settings=None, instrument_library=None):
              logger.debug("DummyMIDIMemory initialized.")
              self.settings = settings
              self.instrument_library = instrument_library # [cite: 523]
              # Simulate loading an empty memory initially
              self.patterns: Dict[str, MIDIPattern] = {} # Use the actual MIDIPattern dataclass if defined
              self.similarity_settings = settings.memory_settings.similarity_settings if settings and settings.memory_settings else {}


         def get_all_patterns(self) -> List[MIDIPattern]:
              logger.debug("DummyMIDIMemory: get_all_patterns called.") # [cite: 524]
              # Return a list of dummy patterns or the current patterns
              # For testing, return actual stored patterns
              return list(self.patterns.values())


         def get_pattern(self, pattern_id: str) -> Optional[MIDIPattern]:
              logger.debug(f"DummyMIDIMemory: get_pattern called with ID: {pattern_id}") # [cite: 525]
              # Return a specific pattern by ID if it exists
              return self.patterns.get(pattern_id)


         def add_pattern(self, midi_file_path: str, analysis: Optional[MIDIAnalysis], category: Optional[str] = None, tags: Optional[Set[str]] = None) -> Optional[str]:
              logger.debug(f"DummyMIDIMemory: add_pattern called with {os.path.basename(midi_file_path)}")
              # Simulate adding a pattern [cite: 526]
              if midi_file_path:
                   pattern_id = str(uuid.uuid4()) # Generate a unique ID
                   # Create a dummy MIDIPattern object
                   dummy_pattern = MIDIPattern(
                        pattern_id=pattern_id, # [cite: 527]
                        file_path=midi_file_path,
                        analysis=analysis, # Can be the real analysis if provided
                        category=category,
                        tags=tags if tags is not None else set(), # [cite: 528]
                        creation_date=datetime.now()
                   )
                   self.patterns[pattern_id] = dummy_pattern
                   logger.info(f"DummyMIDIMemory: Added dummy pattern ID {pattern_id[:6]}...") # [cite: 529]
                   return pattern_id # Return the new ID
              else:
                   logger.warning("DummyMIDIMemory: Cannot add pattern, file path is None.")
                   return None


         def find_similar_patterns(self, reference_analysis: MIDIAnalysis) -> List[MIDIPattern]: # [cite: 530]
              logger.debug(f"DummyMIDIMemory: find_similar_patterns called (ref tempo: {reference_analysis.tempo:.2f})")
              # Return a list of dummy similar patterns (e.g., the first 2 patterns in memory)
              logger.warning("DummyMIDIMemory: find_similar_patterns is a dummy implementation, returning first 2 patterns.")
              return list(self.patterns.values())[:2] # Return the first 2 stored patterns [cite: 531]


         def save(self):
              logger.debug("DummyMIDIMemory: save called, simulating save.")
              # Simulate saving memory (e.g., save the pattern dictionary to a dummy file path from settings)
              if self.settings and hasattr(self.settings, 'memory_file_full_path') and self.settings.memory_file_full_path:
                   memory_file_path = self.settings.memory_file_full_path
                   memory_dir = os.path.dirname(memory_file_path) # [cite: 532]
                   if memory_dir and not os.path.exists(memory_dir):
                        os.makedirs(memory_dir, exist_ok=True)
                   # In a real save, you would serialize self.patterns to JSON etc.
                   logger.info(f"DummyMIDIMemory: Simulated saving memory to {memory_file_path}") # [cite: 533]
                   # Example: Save just a marker file
                   # with open(memory_file_path, 'w') as f:
                   #      f.write("dummy memory saved\n")

              else: # [cite: 534]
                   logger.warning("DummyMIDIMemory: Memory file path not available in settings, cannot simulate save.")


         def load(self):
              logger.debug("DummyMIDIMemory: load called, simulating load.")
              # Simulate loading memory (e.g., load patterns from a dummy file)
              # For simplicity in dummy, it just starts with empty patterns [cite: 535]
              logger.warning("DummyMIDIMemory: Dummy load always starts with empty memory.")
              self.patterns = {}


    # Create QApplication instance only if running standalone
    # The main application in main.py creates the QApplication
    app = QApplication.instance() # Check if QApplication already exists
    if app is None: # Create QApplication if it doesn't exist [cite: 536]
         logger.debug("Creating QApplication instance (standalone).")
         app = QApplication(sys.argv)
    else:
         logger.debug("QApplication instance already exists.")


    try:
        # Create dummy core components for standalone test
        dummy_settings = DummySettings()
        logger.info("Dummy settings created for standalone test.")

        # Use dummy implementations of core components [cite: 537]
        # If real imports failed, these dummy classes will be used if defined in this file. [cite: 538]
        # If real imports succeeded, the real classes will be initialized by MainWindow __init__. [cite: 539]
        # The standalone test block should use REAL components if they can be initialized and functionality needs testing. [cite: 540]
        # The current MainWindow __init__ initializes components *inside* itself. [cite: 541]
        # So, we don't pass dummy processor/model/memory to MainWindow __init__ as it's written. [cite: 542]
        # We rely on the REAL classes being imported at the top. [cite: 543]
        # The issue was the indentation in the REAL MainWindow __init__ method. [cite: 544]
        # The current full code below has the REAL imports and the CORRECTED __init__ with buttons. [cite: 545]
        # Create dummy settings object (used by MainWindow init)
        dummy_settings = DummySettings() # Create dummy settings object
        # MainWindow will attempt to initialize real components using these settings
        main_window = MainWindow(settings=dummy_settings)


        main_window.show()

        # Start the Qt event loop
        logger.info("Starting Qt application event loop (standalone).")
        exit_code = app.exec()
        logger.info(f"Qt application event loop finished with exit code: {exit_code} (standalone).\n") # Added newline [cite: 546]

        # Clean up matplotlib figures potentially left open in tests
        plt.close('all')
        logger.debug("Closed all matplotlib figures.")


        sys.exit(exit_code)

    except Exception as e:
        # Catch any unhandled exceptions during standalone test execution
        logger.critical(f"Unhandled error during standalone main_window.py execution: {e}", exc_info=True)
        # Show a message box if QApplication is available [cite: 547]
        app = QApplication.instance() # Check if QApplication exists
        if app:
             # Use a QTimer to ensure the message box appears after the event loop might have started/crashed
             QTimer.singleShot(0, lambda: QMessageBox.critical(None, "Beklenmeyen Hata (Test)", f"Standalone test sırasında hata oluştu:\n\n{e}"))
        sys.exit(1)
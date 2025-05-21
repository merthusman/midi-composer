import os
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget, QGroupBox, QTextEdit, QSplitter,
    QFileDialog, QMessageBox, QProgressBar, QStatusBar, QSizePolicy, QScrollArea, QSlider,
    QGridLayout, QApplication, QSizePolicy, QFrame, QScrollBar, QCheckBox, QRadioButton,
    QProgressDialog, QDialog, QDialogButtonBox, QToolButton, QMenu, QMenuBar, QToolBar,
    QDockWidget, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QTableView, QTableWidget, QTableWidgetItem, QAbstractItemView, QStyledItemDelegate,
    QStyle, QStyleOptionViewItem, QStyleOptionButton, QStyleOption, QInputDialog,
    QColorDialog, QFontDialog, QFileSystemModel, QTreeView, QFormLayout, QSpacerItem
)
from PyQt6.QtCore import (
    Qt, QSize, QTimer, QThread, pyqtSignal, QObject, QRect, QPoint, QEvent, QUrl,
    QDateTime, QTime, QDate, QSizeF, QRectF, QLineF, QPointF, QEasingCurve, QPropertyAnimation,
    QParallelAnimationGroup, QSequentialAnimationGroup, QAbstractAnimation, pyqtProperty,
    QLibraryInfo, QTranslator, QLocale, QLibrary, QThreadPool, QRunnable, QMutex, QWaitCondition,
    QSemaphore, QProcess, QProcessEnvironment, QSettings, QStandardPaths, QDir, QFile, QFileInfo,
    QTextStream, QTextCodec, QIODevice, QDataStream, QBuffer, QByteArray, QMimeData, QUrlQuery,
    QRegularExpression, QRegularExpressionMatch, QRegularExpressionMatchIterator, QVariant,
    QMetaObject, QMetaMethod, QMetaType, QMetaEnum, Q_ARG, Q_RETURN_ARG, pyqtSlot, pyqtBoundSignal,
    QItemSelection, QItemSelectionModel, QModelIndex, QAbstractItemModel, QSortFilterProxyModel,
    QAbstractTableModel, QAbstractListModel, QIdentityProxyModel, QFileSystemWatcher, QTimerEvent,
    QChildEvent, QDynamicPropertyChangeEvent, QNativeGestureEvent, QObjectCleanupHandler
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QFont, QPainter, QColor, QBrush, QPen, QLinearGradient, QRadialGradient,
    QConicalGradient, QFontMetrics, QFontMetricsF, QTextDocument, QTextCursor, QTextCharFormat,
    QTextBlockFormat, QTextListFormat, QTextTableFormat, QTextFrameFormat, QTextImageFormat,
    QTextFormat, QTextLength, QTextObject, QTextOption, QSyntaxHighlighter, QTextDocumentFragment,
    QTextDocumentWriter, QTextTable, QTextFrame, QTextBlock, QTextList, QTextLine, QTextObjectInterface,
    QAction, QActionGroup, QShortcut, QKeySequence, QKeyEvent, QMouseEvent, QWheelEvent, QHoverEvent,
    QTabletEvent, QTouchEvent, QNativeGestureEvent, QContextMenuEvent, QInputMethodEvent, QDropEvent,
    QDragEnterEvent, QDragMoveEvent, QDragLeaveEvent, QDragResponseEvent, QHelpEvent, QWhatsThisClickedEvent,
    QStatusTipEvent, QWindowStateChangeEvent, QShowEvent, QHideEvent, QCloseEvent, QFileOpenEvent,
    QSessionManager, QClipboard, QCursor, QDesktopServices, QGuiApplication, QIconEngine, QImage,
    QImageReader, QImageWriter, QMovie, QPainterPath, QPainterPathStroker, QPalette, QPdfWriter,
    QPicture, QPixmapCache, QPolygon, QPolygonF, QRegion, QScreen, QSessionManager, QStandardItem,
    QStandardItemModel, QTextDocument, QTransform, QValidator, QVector2D, QVector3D, QVector4D
)

# Uygulama kök dizinini bul
if getattr(sys, 'frozen', False):
    # PyInstaller ile paketlenmiş uygulama
    PROJECT_ROOT = Path(sys._MEIPASS)
else:
    # Geliştirme ortamı
    PROJECT_ROOT = Path(__file__).parent.parent.parent

# Kaynaklar dizini
RESOURCES_DIR = PROJECT_ROOT / 'resources'
STYLES_DIR = RESOURCES_DIR / 'styles'
IMAGES_DIR = RESOURCES_DIR / 'images'

# Varsayılan stil dosyası yolu
DEFAULT_STYLESHEET = STYLES_DIR / 'app.qss'

def load_stylesheet(stylesheet_path: Union[str, Path] = None) -> str:
    """
    Stil dosyasını yükler ve içeriğini döndürür.
    
    Args:
        stylesheet_path: Stil dosyasının yolu. None ise varsayılan stil dosyası kullanılır.
        
    Returns:
        str: Stil dosyası içeriği
    """
    try:
        path = Path(stylesheet_path) if stylesheet_path else DEFAULT_STYLESHEET
        
        if not path.exists():
            logging.warning(f"Stil dosyası bulunamadı: {path}")
            return ""
            
        with open(path, 'r', encoding='utf-8') as f:
            style = f.read()
            
        if not style.strip():
            logging.warning(f"Stil dosyası boş: {path}")
            return ""
            
        logging.info(f"Stil dosyası yüklendi: {path}")
        return style
        
    except Exception as e:
        logging.error(f"Stil dosyası yüklenirken hata oluştu: {e}", exc_info=True)
        return ""

# Uygulama ayarlarını yükle
try:
    from src.settings import Settings
except ImportError:
    # Eğer settings modülü yüklenemezse, basit bir Settings sınıfı oluştur
    class Settings:
        def __init__(self):
            self.model_settings = type('ModelSettings', (), {})()
            self.general_settings = type('GeneralSettings', (), {})()
            self.memory_settings = type('MemorySettings', (), {})()
        
        def load(self, *args, **kwargs):
            pass
                
        def save(self, *args, **kwargs):
            pass


class MainWindow(QMainWindow):
    """Uygulamanın ana penceresi."""
    
    def __init__(self, settings: Optional[Settings] = None, parent: Optional[QWidget] = None):
        """
        Ana pencereyi başlatır.
        
        Args:
            settings: Uygulama ayarları
            parent: Üst widget
        """
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.settings = settings if settings is not None else Settings()
        
        # Pencere özelliklerini ayarla
        self.setWindowTitle("MIDI Composer")
        self.setMinimumSize(1200, 800)
        
        # Stilleri yükle ve uygula
        self._load_styles()
        
        # Merkezi widget'ı oluştur
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Ana layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # UI'ı başlat
        self._init_ui()
        
        # Ayarları yükle
        self._load_settings()
        
        # Pencere boyutunu ve konumunu ayarla
        self._restore_window_geometry()
    def _init_ui(self):
        """Kullanıcı arayüzünü başlatır."""
        # Ana tab widget'ı oluştur
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # MIDI Üretim sekmesi
        self.create_midi_production_tab()
        
        # Ayarlar sekmesi
        self.create_settings_tab()
        
        # Durum çubuğu
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar, 1)
        
        # Başlangıç mesajı
        self.status_bar.showMessage("Uygulama başlatıldı", 3000)
    
    def _show_about(self):
        """Hakkında diyaloğunu gösterir."""
        about_text = """
        <h2>MIDI Besteleyici</h2>
        <p>Versiyon 1.0.0</p>
        <p>Bu uygulama, yapay zeka destekli MIDI besteleme aracıdır.</p>
        <p>Geliştirici: [Geliştirici Adı]</p>
        <p>Telif Hakkı © 2025 Tüm hakları saklıdır.</p>
        """
        
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Hakkında - MIDI Besteleyici")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(about_text.strip())
        msg_box.setIcon(QMessageBox.Icon.Information)
        
        # Pencere boyutunu ayarla
        msg_box.setMinimumSize(400, 300)
        
        # Pencereyi modal yap (kullanıcı diğer pencerelerle etkileşime giremesin)
        msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        msg_box.exec()
    
    def _load_styles(self):
        """Uygulama stillerini yükler ve uygular."""
        try:
            # Varsayılan stilleri yükle
            style_sheet = load_stylesheet()
            
            if style_sheet:
                self.setStyleSheet(style_sheet)
                self.logger.info("Stiller başarıyla yüklendi")
            else:
                self.logger.warning("Stil dosyası yüklenemedi, varsayılan stiller kullanılacak")
                self._apply_default_styles()
                
        except Exception as e:
            self.logger.error(f"Stiller yüklenirken hata oluştu: {e}", exc_info=True)
            self._apply_default_styles()
    
    def _apply_default_styles(self):
        """Varsayılan stilleri uygular."""
        # Temel stiller
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1A1B35;
                color: #FFFFFF;
            }
            
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                min-width: 80px;
                min-height: 30px;
                font-weight: bold;
                font-size: 11pt;
            }
            
            QPushButton:hover {
                background-color: #357abd;
            }
            
            QPushButton:pressed {
                background-color: #2a5d8a;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            
            #generateButton {
                background-color: #4caf50;
                font-size: 12pt;
                min-width: 120px;
                min-height: 40px;
            }
            
            #generateButton:hover {
                background-color: #3d8b40;
            }
            
            #generateButton:pressed {
                background-color: #2e7d32;
            }
            
            QLabel {
                color: #FFFFFF;
                font-size: 11pt;
                padding: 5px;
                background-color: transparent;
                min-height: 20px;
            }
            
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: rgba(45, 45, 45, 0.9);
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px 10px;
                min-height: 30px;
                min-width: 150px;
                font-size: 10pt;
            }
            
            QGroupBox {
                border: 1px solid rgba(80, 80, 80, 0.7);
                border-radius: 8px;
                margin-top: 1.8em;
                padding: 15px;
                color: #FFFFFF;
                font-weight: normal;
                background-color: rgba(30, 30, 30, 0.85);
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #FFFFFF;
                font-size: 13px;
                font-weight: bold;
                background-color: transparent;
            }
        """)
    
    def _load_settings(self):
        """Ayarları yükler."""
        try:
            if hasattr(self.settings, 'load'):
                self.settings.load()
                self.logger.info("Ayarlar başarıyla yüklendi")
        except Exception as e:
            self.logger.error(f"Ayarlar yüklenirken hata: {e}", exc_info=True)
    
    def _restore_window_geometry(self):
        """Pencere boyutunu ve konumunu geri yükler."""
        try:
            if hasattr(self.settings.general_settings, 'window_geometry'):
                self.restoreGeometry(self.settings.general_settings.window_geometry)
            if hasattr(self.settings.general_settings, 'window_state'):
                self.restoreState(self.settings.general_settings.window_state)
        except Exception as e:
            self.logger.warning(f"Pencere boyutu geri yüklenirken hata: {e}")
    
    def _save_window_geometry(self):
        """Pencere boyutunu ve konumunu kaydeder."""
        try:
            if not hasattr(self.settings.general_settings, 'window_geometry'):
                self.settings.general_settings.window_geometry = self.saveGeometry()
            if not hasattr(self.settings.general_settings, 'window_state'):
                self.settings.general_settings.window_state = self.saveState()
        except Exception as e:
            self.logger.warning(f"Pencere boyutu kaydedilirken hata: {e}")
    
    def create_midi_production_tab(self):
        """MIDI üretim sekmesini oluşturur."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Üst kısım - Parametreler
        params_group = QGroupBox("Üretim Parametreleri")
        params_layout = QGridLayout(params_group)
        params_layout.setContentsMargins(10, 15, 10, 15)
        params_layout.setSpacing(15)
        
        # Parametreler
        self.bar_count_label = QLabel("Ölçü Sayısı:")
        self.bar_count_spin = QSpinBox()
        self.bar_count_spin.setRange(1, 32)
        self.bar_count_spin.setValue(8)
        self.bar_count_spin.setMinimumWidth(100)
        
        self.tempo_label = QLabel("Tempo (BPM):")
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(40, 240)
        self.tempo_spin.setValue(120)
        self.tempo_spin.setMinimumWidth(100)
        
        self.temp_label = QLabel("Yaratıcılık:")
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(10, 100)
        self.temp_slider.setValue(70)
        self.temp_value = QLabel("0.7")
        
        self.style_label = QLabel("Müzik Stili:")
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Pop", "Rock", "Klasik", "Jazz", "Hip-Hop", "Elektronik"])
        
        # Parametreleri layout'a ekle
        row = 0
        params_layout.addWidget(self.bar_count_label, row, 0)
        params_layout.addWidget(self.bar_count_spin, row, 1)
        row += 1
        
        params_layout.addWidget(self.tempo_label, row, 0)
        params_layout.addWidget(self.tempo_spin, row, 1)
        row += 1
        
        params_layout.addWidget(self.temp_label, row, 0)
        params_layout.addWidget(self.temp_slider, row, 1)
        params_layout.addWidget(self.temp_value, row, 2)
        row += 1
        
        params_layout.addWidget(self.style_label, row, 0)
        params_layout.addWidget(self.style_combo, row, 1, 1, 2)
        
        # Butonlar
        buttons_layout = QHBoxLayout()
        self.generate_btn = QPushButton("MIDI Üret")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setObjectName("generateButton")
        
        self.play_btn = QPushButton("Çal")
        self.stop_btn = QPushButton("Durdur")
        self.save_btn = QPushButton("Kaydet")
        
        buttons_layout.addWidget(self.generate_btn)
        buttons_layout.addWidget(self.play_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.save_btn)
        
        # Alt kısım - Çıktı ve önizleme
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Sol panel - Çıktı
        output_group = QGroupBox("Çıktı")
        output_layout = QVBoxLayout(output_group)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        
        # Sağ panel - Piyano rulosu önizleme
        preview_group = QGroupBox("Piyano Rulosu Önizleme")
        preview_layout = QVBoxLayout(preview_group)
        self.piano_roll_label = QLabel("Piyano rulosu burada görüntülenecek")
        self.piano_roll_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.piano_roll_label.setMinimumSize(400, 250)
        self.piano_roll_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #aaaaaa;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        preview_layout.addWidget(self.piano_roll_label)
        
        # Splitter'a ekle
        splitter.addWidget(output_group)
        splitter.addWidget(preview_group)
        splitter.setSizes([600, 400])
        
        # Ana layout'a ekle
        layout.addWidget(params_group)
        layout.addLayout(buttons_layout)
        layout.addWidget(splitter, 1)  # 1 = stretch factor
        
        # Sekmeye ekle
        self.tab_widget.addTab(tab, "MIDI Üretimi")
        
        # Sinyalleri bağla
        self.temp_slider.valueChanged.connect(self.update_temp_value)
        self.generate_btn.clicked.connect(self.on_generate_clicked)
    
    def create_settings_tab(self):
        """Ayarlar sekmesini oluşturur."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Ayarlar için metin editörü
        self.settings_edit = QTextEdit()
        self.settings_edit.setFont(QFont("Consolas", 10))
        
        # Kaydet butonu
        save_btn = QPushButton("Ayarları Kaydet")
        save_btn.clicked.connect(self.save_settings)
        
        layout.addWidget(QLabel("Uygulama Ayarları (JSON formatında):"))
        layout.addWidget(self.settings_edit, 1)
        layout.addWidget(save_btn)
        
        # Sekmeye ekle
        self.tab_widget.addTab(tab, "Ayarlar")
        
        # Ayarları yükle
        self.load_settings_to_ui()
    
    def update_temp_value(self, value):
        """Sıcaklık değerini günceller."""
        temp = value / 100.0
        self.temp_value.setText(f"{temp:.2f}")
    
    def on_generate_clicked(self):
        """MIDI üretme butonuna tıklandığında çağrılır."""
        try:
            # Parametreleri al
            bar_count = self.bar_count_spin.value()
            tempo = self.tempo_spin.value()
            temperature = float(self.temp_value.text())
            style = self.style_combo.currentText()
            
            # İşlemi başlat
            self.status_bar.showMessage("MIDI üretiliyor...")
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            
            # Burada gerçek MIDI üretim işlemi yapılacak
            # Şimdilik sadece simüle ediyoruz
            QTimer.singleShot(2000, self.on_generation_completed)
            
            # Çıktıya bilgi ekle
            self.output_text.append(f"Yeni MIDI üretiliyor...")
            self.output_text.append(f"Ölçü Sayısı: {bar_count}")
            self.output_text.append(f"Tempo: {tempo} BPM")
            self.output_text.append(f"Yaratıcılık: {temperature:.2f}")
            self.output_text.append(f"Stil: {style}\n")
            
        except Exception as e:
            self.logger.error(f"MIDI üretilirken hata: {e}", exc_info=True)
            QMessageBox.critical(self, "Hata", f"MIDI üretilirken bir hata oluştu:\n{str(e)}")
            self.status_bar.showMessage("Hata oluştu", 3000)
            self.progress_bar.hide()
    
    def on_generation_completed(self):
        """MIDI üretimi tamamlandığında çağrılır."""
        self.status_bar.showMessage("MIDI başarıyla oluşturuldu", 5000)
        self.progress_bar.setValue(100)
        self.output_text.append("MIDI başarıyla oluşturuldu!\n")
        
        # İlerleme çubuğunu gizle
        QTimer.singleShot(3000, self.progress_bar.hide)
    
    def load_settings_to_ui(self):
        """Ayarları kullanıcı arayüzüne yükler."""
        try:
            if hasattr(self.settings, 'to_dict'):
                settings_dict = self.settings.to_dict()
                self.settings_edit.setPlainText(json.dumps(settings_dict, indent=4, ensure_ascii=False))
        except Exception as e:
            self.logger.error(f"Ayarlar arayüze yüklenirken hata: {e}", exc_info=True)
    
    def save_settings(self):
        """Kullanıcı arayüzünden ayarları kaydeder."""
        try:
            settings_text = self.settings_edit.toPlainText()
            settings_dict = json.loads(settings_text)
            
            if hasattr(self.settings, 'from_dict'):
                self.settings.from_dict(settings_dict)
                if self._save_settings():
                    QMessageBox.information(self, "Başarılı", "Ayarlar başarıyla kaydedildi")
                    return True
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Hata", "Geçersiz JSON formatı")
        except Exception as e:
            self.logger.error(f"Ayarlar kaydedilirken hata: {e}", exc_info=True)
            QMessageBox.critical(self, "Hata", f"Ayarlar kaydedilirken bir hata oluştu:\n{str(e)}")
        
        return False
    
    def closeEvent(self, event):
        """Pencere kapatıldığında çağrılır."""
        self._save_window_geometry()
        self._save_settings()
        event.accept()
    
    def _setup_window_properties(self):
        """Set up the main window properties."""
        # Set window title and size
        self.setWindowTitle("MIDI Besteleyici")
        self.setMinimumSize(1024, 768)
        
        # Set window icon if available
        icon_path = os.path.join(os.path.dirname(__file__), "..", "resources", "icons", "midi_composer.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Set window flags
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # Center the window on screen
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)
        
        # Set status bar
        self.statusBar().showMessage("Hazır")
    
    def connect_signals(self):
        """Connect signals from UI elements to their respective slots."""
        try:
            # Connect menu actions
            # Note: Menu actions are already connected in _create_menu_bar
            
            # Connect other UI signals here
            # Örnek: self.some_button.clicked.connect(self.some_slot)
            
            logger.info("UI signals connected successfully")
        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
            raise
            
    def setup_ui(self):
        """Set up the user interface with a tabbed layout."""
        # Create the main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create tab widget with custom styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("main_tabs")
        self.tab_widget.setDocumentMode(True)  # More modern look
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setMovable(False)
        
        # Apply tab styles
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                background: rgba(30, 30, 30, 0.7);
                color: #b0b0b0;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: rgba(41, 128, 185, 0.5);
                color: white;
                border-bottom: 2px solid #3498db;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(60, 60, 60, 0.8);
            }
        """)
        
        # Create tabs
        self.setup_midi_analysis_tab()
        self.setup_midi_generation_tab()
        self.setup_memory_tab()
        self.setup_settings_tab()
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget, 1)  # 1 = stretch factor
        
        # Set the central widget
        self.setCentralWidget(main_widget)
        
        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır")
        
        # Apply styles
        self.apply_styles()
    
    def apply_styles(self):
        """Apply custom styles to the UI elements."""
        # Set object names for styling
        self.tab_widget.setObjectName("main_tabs")
        
        # Set object names for specific groups
        if hasattr(self, 'analysis_tab'):
            for widget in self.analysis_tab.findChildren(QGroupBox):
                if "Analiz" in widget.title():
                    widget.setObjectName("analysis_group")
        
        if hasattr(self, 'generation_tab'):
            for widget in self.generation_tab.findChildren(QGroupBox):
                if "Üretim" in widget.title():
                    widget.setObjectName("midi_uretim_group")
                elif "Sonuç" in widget.title():
                    widget.setObjectName("result_group")
        
        if hasattr(self, 'memory_tab'):
            for widget in self.memory_tab.findChildren(QGroupBox):
                if "Öğeler" in widget.title():
                    widget.setObjectName("pattern_group")
                elif "Detay" in widget.title():
                    widget.setObjectName("detail_group")
        
        if hasattr(self, 'settings_tab'):
            for widget in self.settings_tab.findChildren(QGroupBox):
                if "Genel" in widget.title():
                    widget.setObjectName("settings_group")
                elif "Yollar" in widget.title():
                    widget.setObjectName("paths_group")
        
        # Apply styles to all widgets
        self.setStyleSheet("""
            /* Ana Pencere */
            QMainWindow {
                background-color: #1A1B35;
                color: #FFFFFF;
                min-width: 900px;
                min-height: 700px;
            }
            
            /* Sekmeler */
            QTabWidget::pane {
                border: 1px solid rgba(80, 80, 80, 0.7);
                border-radius: 4px;
                background: rgba(30, 30, 30, 0.85);
                margin: 0px;
                padding: 0px;
            }
            
            QTabBar::tab {
                background: rgba(40, 40, 40, 0.9);
                color: #FFFFFF;
                border: 1px solid rgba(80, 80, 80, 0.7);
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            
            QTabBar::tab:selected {
                background: #00BCD4;
                color: #FFFFFF;
                font-weight: bold;
            }
            
            /* Grup Kutuları */
            #midi_file_group, #analysis_group,
            #midi_uretim_group, #result_group,
            #pattern_group, #detail_group,
            #settings_group, #paths_group {
                background-color: rgba(30, 30, 30, 0.85);
                border: 2px solid rgba(0, 188, 212, 0.7);
                border-radius: 6px;
                margin: 5px;
                padding: 10px;
            }
            
            /* Piyano Rulosu Görüntüleme */
            QLabel[objectName^="piano_roll"] {
                background-color: rgba(30, 30, 30, 0.7);
                border: 1px solid #555555;
                border-radius: 5px;
                min-width: 400px;
                min-height: 250px;
            }
            
            /* Durum Çubuğu */
            QStatusBar {
                background-color: rgba(30, 30, 30, 0.9);
                color: #FFFFFF;
                border-top: 1px solid #555555;
                padding: 2px 5px;
                min-height: 25px;
            }
            
            /* İlerleme Çubuğu */
            QProgressBar {
                border: 1px solid rgba(80, 80, 80, 0.7);
                border-radius: 6px;
                text-align: center;
                color: #FFFFFF;
                background-color: rgba(30, 30, 30, 0.7);
                padding: 1px;
                height: 24px;
                font-weight: bold;
                font-size: 10pt;
            }
            
            QProgressBar::chunk {
                background-color: #00BCD4;
                border-radius: 5px;
                margin: 1px;
            }
            
            /* Özel Butonlar */
            #generate_button, #analyze_button, #browse_button, #search_memory_button,
            #save_settings_button, #load_settings_button, #export_settings_button, #import_settings_button {
                background-color: #00BCD4;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                min-height: 30px;
                font-weight: bold;
            }
            
            #generate_button {
                font-size: 12pt;
                min-height: 40px;
                min-width: 250px;
            }
            
            /* Buton Hover ve Press Efektleri */
            QPushButton:hover {
                background-color: #26C6DA;
            }
            
            QPushButton:pressed {
                background-color: #0097A7;
            }
            
            QPushButton:disabled {
                background-color: rgba(0, 188, 212, 0.3);
                color: rgba(255, 255, 255, 0.5);
            }
            
            /* Kaydırma Çubukları */
            QScrollBar:vertical, QScrollBar:horizontal {
                border: none;
                background: rgba(30, 30, 30, 0.5);
                width: 10px;
                height: 10px;
                margin: 0;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: rgba(80, 80, 80, 0.7);
                min-height: 30px;
                min-width: 30px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: rgba(100, 100, 100, 0.8);
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                height: 0;
                width: 0;
                background: none;
            }
            
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            
            /* Ayırıcı */
            QSplitter::handle {
                background-color: #555555;
                height: 2px;
                width: 2px;
            }
            
            QSplitter::handle:horizontal {
                width: 4px;
            }
            
            QSplitter::handle:vertical {
                height: 4px;
            }
            
            QSplitter::handle:hover {
                background-color: #00BCD4;
            }
        """)
    
    def setup_midi_analysis_tab(self):
        """Set up the MIDI analysis tab."""
        self.analysis_tab = QWidget()
        layout = QVBoxLayout(self.analysis_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create group box for analysis controls
        control_group = QGroupBox("Analiz Kontrolleri")
        control_group.setObjectName("midi_file_group")
        control_layout = QHBoxLayout(control_group)
        
        # Add analysis widgets
        self.analysis_label = QLabel("Dosya Seçin:")
        self.analysis_label.setObjectName("param_label")
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("MIDI dosyası seçin...")
        self.file_path_edit.setReadOnly(True)
        
        self.browse_button = QPushButton("Gözat...")
        self.browse_button.setObjectName("browse_button")
        
        self.analyze_button = QPushButton("Analiz Et")
        self.analyze_button.setObjectName("analyze_button")
        
        # Add widgets to control layout
        control_layout.addWidget(self.analysis_label)
        control_layout.addWidget(self.file_path_edit, 1)  # 1 = stretch factor
        control_layout.addWidget(self.browse_button)
        control_layout.addWidget(self.analyze_button)
        
        # Create group box for results
        result_group = QGroupBox("Analiz Sonuçları")
        result_group.setObjectName("analysis_group")
        result_layout = QVBoxLayout(result_group)
        
        # Add piano roll display
        self.piano_roll_display = QLabel()
        self.piano_roll_display.setObjectName("piano_roll_display")
        self.piano_roll_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.piano_roll_display.setText("Piyano Rulosu Burada Görüntülenecek")
        result_layout.addWidget(self.piano_roll_display, 1)  # 1 = stretch factor
        
        # Add progress bar
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        result_layout.addWidget(self.analysis_progress)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setObjectName("result_text")
        
        result_layout.addWidget(self.result_text)
        
        # Add groups to main layout
        layout.addWidget(control_group)
        layout.addWidget(result_group, 1)  # 1 = stretch factor
        
        # Add the tab
        self.tab_widget.addTab(self.analysis_tab, "MIDI Analiz")
    
    def setup_midi_generation_tab(self):
        """Set up the MIDI generation tab."""
        self.generation_tab = QWidget()
        layout = QVBoxLayout(self.generation_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create group box for generation parameters
        param_group = QGroupBox("Üretim Parametreleri")
        param_group.setObjectName("midi_uretim_group")
        param_layout = QGridLayout(param_group)
        
        # Add parameter widgets
        self.bar_count_label = QLabel("Ölçü Sayısı:")
        self.bar_count_label.setObjectName("param_label")
        self.bar_count_spin = QSpinBox()
        self.bar_count_spin.setRange(1, 64)
        self.bar_count_spin.setValue(8)
        
        self.temp_label = QLabel("Sıcaklık:")
        self.temp_label.setObjectName("param_label")
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 100)
        self.temp_slider.setValue(70)
        
        self.temp_value = QLabel("0.7")
        self.temp_value.setObjectName("param_label")
        
        # Add widgets to parameter layout
        param_layout.addWidget(self.bar_count_label, 0, 0)
        param_layout.addWidget(self.bar_count_spin, 0, 1)
        param_layout.addWidget(self.temp_label, 1, 0)
        param_layout.addWidget(self.temp_slider, 1, 1)
        param_layout.addWidget(self.temp_value, 1, 2)
        
        # Create group box for generation controls and results
        result_group = QGroupBox("Üretim Sonuçları")
        result_group.setObjectName("result_group")
        result_layout = QVBoxLayout(result_group)
        
        # Add generated piano roll display
        self.generated_piano_roll = QLabel()
        self.generated_piano_roll.setObjectName("generated_piano_roll")
        self.generated_piano_roll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.generated_piano_roll.setText("Üretilen MIDI Görüntüsü Burada Görüntülenecek")
        result_layout.addWidget(self.generated_piano_roll, 1)  # 1 = stretch factor
        
        # Add progress bar
        self.generation_progress = QProgressBar()
        self.generation_progress.setRange(0, 100)
        self.generation_progress.setValue(0)
        result_layout.addWidget(self.generation_progress)
        
        # Create group box for generation controls
        control_group = QGroupBox("Kontroller")
        control_layout = QHBoxLayout(control_group)
        
        # Add controls layout
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(control_group)
        controls_layout.addWidget(result_group, 1)  # 1 = stretch factor
        
        # Add to main layout
        layout.addLayout(controls_layout, 1)  # 1 = stretch factor
        
        self.generate_button = QPushButton("MIDI Üret")
        self.generate_button.setObjectName("generate_button")
        
        self.save_button = QPushButton("Kaydet")
        self.save_button.setObjectName("save_button")
        
        control_layout.addWidget(self.generate_button)
        control_layout.addWidget(self.save_button)
        
        # Add groups to main layout
        layout.addWidget(param_group)
        layout.addWidget(control_group)
        
        # Add the tab
        self.tab_widget.addTab(self.generation_tab, "MIDI Üretimi")
    
    def setup_memory_tab(self):
        """Set up the memory tab."""
        self.memory_tab = QWidget()
        layout = QVBoxLayout(self.memory_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create splitter for memory and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Memory list
        memory_group = QGroupBox("Hafıza Öğeleri")
        memory_layout = QVBoxLayout(memory_group)
        
        self.memory_list = QListWidget()
        self.memory_list.setObjectName("memory_list")
        
        # Add sample items (for testing)
        self.memory_list.addItems(["Örnek Parça 1", "Örnek Parça 2", "Örnek Parça 3"])
        
        memory_layout.addWidget(self.memory_list)
        
        # Details panel
        details_group = QGroupBox("Detaylar")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setObjectName("details_text")
        
        details_layout.addWidget(self.details_text)
        
        # Add groups to splitter
        splitter.addWidget(memory_group)
        splitter.addWidget(details_group)
        
        # Set initial sizes
        splitter.setSizes([300, 500])
        
        # Add splitter to main layout
        layout.addWidget(splitter, 1)  # 1 = stretch factor
        
        # Add the tab
        self.tab_widget.addTab(self.memory_tab, "Hafıza")
    
    def setup_settings_tab(self):
        """Set up the settings tab."""
        self.settings_tab = QWidget()
        layout = QVBoxLayout(self.settings_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create widget that contains the settings
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(5, 5, 5, 5)
        settings_layout.setSpacing(15)
        
        # General settings group
        general_group = QGroupBox("Genel Ayarlar")
        general_layout = QFormLayout(general_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Koyu Tema", "Açık Tema"])
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Türkçe", "English"])
        
        general_layout.addRow("Tema:", self.theme_combo)
        general_layout.addRow("Dil:", self.language_combo)
        
        # Path settings group
        path_group = QGroupBox("Dosya Yolları")
        path_layout = QFormLayout(path_group)
        
        self.midi_path_edit = QLineEdit()
        self.midi_path_button = QPushButton("Gözat...")
        
        self.model_path_edit = QLineEdit()
        self.model_path_button = QPushButton("Gözat...")
        
        path_layout.addRow("MIDI Klasörü:", self.midi_path_edit)
        path_layout.addRow("", self.midi_path_button)
        path_layout.addRow("Model Klasörü:", self.model_path_edit)
        path_layout.addRow("", self.model_path_button)
        
        # Add groups to settings layout
        settings_layout.addWidget(general_group)
        settings_layout.addWidget(path_group)
        settings_layout.addStretch(1)  # Push everything to the top
        
        # Set the scroll area widget
        scroll.setWidget(settings_widget)
        
        # Add scroll area to main layout
        layout.addWidget(scroll)
        
        # Add save button
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        
        self.save_settings_button = QPushButton("Ayarları Kaydet")
        self.save_settings_button.setObjectName("save_settings_button")
        
        button_layout.addWidget(self.save_settings_button)
        layout.addLayout(button_layout)
        
        # Add the tab
        self.tab_widget.addTab(self.settings_tab, "Ayarlar")
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("Dosya")
        
        # Open action
        open_action = QAction("Aç...", self)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        
        # Save action
        save_action = QAction("Kaydet", self)
        save_action.triggered.connect(self._save_midi)
        file_menu.addAction(save_action)
        
        # Exit action
        exit_action = QAction("Çıkış", self)
        exit_action.triggered.connect(self.close)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Yardım")
        
        # About action
        about_action = QAction("Hakkında", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def generate_piano_roll(self, midi_file, display_label):
        """Generate and display a piano roll for the MIDI file with theme support."""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            # Generate piano roll figure with updated styling
            fig = self.plot_piano_roll(midi_data)
            
            # Save figure to buffer with higher DPI for better quality
            buf = io.BytesIO()
            fig.savefig(
                buf, 
                format='png', 
                dpi=150,  # Higher DPI for better quality
                bbox_inches='tight',
                facecolor=fig.get_facecolor(),  # Maintain transparency
                transparent=True
            )
            buf.seek(0)
            
            # Convert buffer to QPixmap
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            
            # Display pixmap with high quality scaling
            display_label.setPixmap(pixmap.scaled(
                display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Clean up
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generating piano roll: {e}")
            display_label.clear()
            display_label.setStyleSheet("color: #ff6b6b; font-size: 12px;")
            display_label.setText(f"Piano roll oluşturulamadı: {str(e)}")
            
    def plot_piano_roll(self, midi_data, fs=100, start_pitch=21, end_pitch=108):
        """
        Plots the piano roll of the MIDI data with measure/bar lines and labels.
        
        Args:
            midi_data: pretty_midi.PrettyMIDI object
            fs: Sampling frequency for the piano roll
            start_pitch: Starting MIDI pitch (21 = A0)
            end_pitch: Ending MIDI pitch (108 = C8)
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot
        """
        if midi_data is None:
            logger.warning("No MIDI data provided for piano roll plotting.")
            return None
            
        # Get the piano roll
        piano_roll = midi_data.get_piano_roll(fs=fs)
        
        # Trim to the specified pitch range
        piano_roll = piano_roll[start_pitch:end_pitch]
        
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Set transparent background for the figure and axes
        fig.patch.set_facecolor('transparent')
        ax.set_facecolor('transparent')
        
        # Plot the piano roll with a theme-appropriate colormap
        im = ax.imshow(
            piano_roll,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='Blues'  # Using Blues colormap for better visibility
        )
        
        # Set labels and tick colors for theme consistency
        ax.set_ylabel('Pitch', color='#e0e0e0')
        ax.set_xlabel('Time (Bars)', color='#e0e0e0')
        ax.tick_params(axis='x', colors='#e0e0e0')
        ax.tick_params(axis='y', colors='#e0e0e0')
        
        # Set y-axis ticks to show note names
        y_tick_positions = []
        y_tick_labels = []
        
        # Label every octave (C notes) for clarity
        for i, pitch in enumerate(range(start_pitch, end_pitch + 1)):
            note_name = pretty_midi.note_number_to_name(pitch)
            if note_name.startswith('C') or i == 0 or i == (end_pitch - start_pitch):
                y_tick_positions.append(i)
                y_tick_labels.append(note_name)
        
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)
        
        # Add grid
        ax.grid(alpha=0.3, color='#808080')
        
        # Add bar/measure lines
        if midi_data.time_signature_changes:
            time_sig = midi_data.time_signature_changes[0]
            numerator = time_sig.numerator
            denominator = time_sig.denominator
        else:
            numerator = 4
            denominator = 4
            
        # Get downbeat times (start of each measure) in seconds
        downbeat_times = midi_data.get_downbeats()
        
        # Calculate the total number of samples (width of the piano roll)
        total_samples = piano_roll.shape[1]
        
        bar_x_positions = []
        bar_labels = []
        
        # Convert downbeat times to x-positions in the plot
        for i, beat_time in enumerate(downbeat_times):
            x_pos_samples = int(beat_time * fs)  # Convert seconds to samples
            if x_pos_samples < total_samples:  # Only plot if within the visible range
                bar_x_positions.append(x_pos_samples)
                bar_labels.append(str(i + 1))  # Bar numbers start from 1
        
        # Plot vertical lines for each bar
        for x_pos in bar_x_positions:
            ax.axvline(
                x=x_pos, 
                color='#00BCD4',  # Accent color
                linestyle='--', 
                linewidth=0.8, 
                alpha=0.7
            )
        
        # Set x-axis ticks at bar positions with bar numbers as labels
        if bar_x_positions:  # Only if we have valid bar positions
            ax.set_xticks(bar_x_positions)
            ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        
        # Customize spine colors to match theme
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
        
        # Add colorbar with matching style
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')
        cbar.outline.set_edgecolor('#555555')
        
        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()
        
        return fig

    def paintEvent(self, event):
        """Handle paint events to draw the background image and overlay."""
        super().paintEvent(event)
        
        if not hasattr(self, 'background_image') or self.background_image.isNull():
            return
            
        painter = QPainter(self)
        
        # Resize the image to fit the window while maintaining aspect ratio
        scaled_pixmap = self.background_image.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image
        x = (scaled_pixmap.width() - self.width()) // 2
        y = (scaled_pixmap.height() - self.height()) // 2
        
        # Draw the background image
        painter.drawPixmap(0, 0, scaled_pixmap, x, y, self.width(), self.height())
        
        # Add a semi-transparent overlay for better text readability
        painter.setOpacity(0.3)  # 30% opacity
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setOpacity(1.0)  # Reset opacity

    def _init_ui_elements(self):
        """Initialize UI elements."""
        try:
            # Create MIDI production section
            midi_prod_group = QGroupBox("MIDI Üretimi")
            midi_prod_group.setObjectName("midi_prod_group")
            midi_prod_group.setMinimumWidth(500)
            
            # Set up grid layout for parameters
            midi_prod_layout = QGridLayout()
            midi_prod_layout.setHorizontalSpacing(20)
            midi_prod_layout.setVerticalSpacing(15)
            
            # Add MIDI production parameters
            row = 0
            
            # Bar count
            self.bar_count_label = QLabel("Ölçü Sayısı:")
            self.bar_count_label.setObjectName("param_label")
            self.bar_count_label.setMinimumWidth(120)
            self.bar_count_spin = QSpinBox()
            self.bar_count_spin.setObjectName("param_input")
            self.bar_count_spin.setRange(1, 32)
            self.bar_count_spin.setValue(8)
            self.bar_count_spin.setMinimumWidth(150)
            midi_prod_layout.addWidget(self.bar_count_label, row, 0)
            midi_prod_layout.addWidget(self.bar_count_spin, row, 1)
            
            row += 1
            
            # Tempo
            self.tempo_label = QLabel("Tempo (BPM):")
            self.tempo_label.setObjectName("param_label")
            self.tempo_spin = QSpinBox()
            self.tempo_spin.setObjectName("param_input")
            self.tempo_spin.setRange(60, 200)
            self.tempo_spin.setValue(120)
            self.tempo_spin.setMinimumWidth(150)
            midi_prod_layout.addWidget(self.tempo_label, row, 0)
            midi_prod_layout.addWidget(self.tempo_spin, row, 1)
            
            row += 1
            
            # Temperature
            self.temp_label = QLabel("Yaratıcılık (Sıcaklık):")
            self.temp_label.setObjectName("param_label")
            self.temp_slider = QSlider(Qt.Orientation.Horizontal)
            self.temp_slider.setObjectName("param_input")
            self.temp_slider.setRange(0, 200)
            self.temp_slider.setValue(100)
            self.temp_slider.setMinimumWidth(150)
            self.temp_value = QLabel("1.0")
            self.temp_value.setObjectName("param_value")
            midi_prod_layout.addWidget(self.temp_label, row, 0)
            midi_prod_layout.addWidget(self.temp_slider, row, 1)
            midi_prod_layout.addWidget(self.temp_value, row, 2)
            
            row += 1
            
            # Style
            self.style_label = QLabel("Müzik Stili:")
            self.style_label.setObjectName("param_label")
            self.style_combo = QComboBox()
            self.style_combo.setObjectName("param_input")
            self.style_combo.addItems(["Pop", "Rock", "Klasik", "Jazz", "Hip-Hop", "Elektronik"])
            self.style_combo.setMinimumWidth(150)
            midi_prod_layout.addWidget(self.style_label, row, 0)
            midi_prod_layout.addWidget(self.style_combo, row, 1)
            
            # Set column stretches
            midi_prod_layout.setColumnStretch(0, 0)
            midi_prod_layout.setColumnStretch(1, 1)
            
            midi_prod_group.setLayout(midi_prod_layout)
            self.main_layout.addWidget(midi_prod_group)
            
            # Create piano roll section
            piano_roll_group = QGroupBox("Piyano Rulosu")
            piano_roll_group.setObjectName("piano_roll_group")
            piano_roll_group.setMinimumWidth(600)
            
            # Set up vertical layout for piano roll
            piano_roll_layout = QVBoxLayout()
            piano_roll_layout.setSpacing(10)
            
            # Create piano roll label with minimum size
            self.piano_roll_label = QLabel()
            self.piano_roll_label.setObjectName("piano_roll_label")
            self.piano_roll_label.setMinimumSize(600, 300)
            self.piano_roll_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Create detail label
            self.detail_label = QLabel("Detaylar: 0 not")
            self.detail_label.setObjectName("detail_label")
            self.detail_label.setMinimumHeight(30)
            
            piano_roll_layout.addWidget(self.piano_roll_label)
            piano_roll_layout.addWidget(self.detail_label)
            
            piano_roll_group.setLayout(piano_roll_layout)
            self.main_layout.addWidget(piano_roll_group)
            
            # Create buttons section
            buttons_layout = QHBoxLayout()
            buttons_layout.setSpacing(15)
            
            # Generate button
            self.generate_button = QPushButton("MIDI Üret")
            self.generate_button.setObjectName("generate_button")
            self.generate_button.setMinimumSize(250, 40)
            
            buttons_layout.addWidget(self.generate_button)
            
            buttons_group = QGroupBox()
            buttons_group.setLayout(buttons_layout)
            self.main_layout.addWidget(buttons_group)
            
            logger.info("UI elements initialized successfully")
            
        except Exception as e:
            self.logger_manager.log_error(f"UI element initialization failed: {e}")
            logger.error(f"Error initializing core components: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Başlatma Hatası",
                f"Uygulama bileşenleri başlatılırken hata oluştu:\n\n{e}\n\nBazı özellikler çalışmayabilir."
            )
            self._core_components_initialized = False
            return
            
        
    def _setup_ui_styles(self):
        """Set up the styles for UI elements."""
        # Set application style
        style_sheet = """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #f5f5f5;
            }
            QLabel#param_label {
                font-size: 14px;
                color: #333;
                padding: 8px;
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 2px;
            }
            QPushButton#generate_button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 24px;
                border: none;
                border-radius: 6px;
                min-width: 250px;
                min-height: 44px;
            }
            QPushButton#generate_button:hover {
                background-color: #45a049;
            }
            QPushButton#generate_button:pressed {
                background-color: #3e8e41;
            }
            QPushButton#save_button {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 24px;
                border: none;
                border-radius: 6px;
                min-width: 200px;
                min-height: 44px;
            }
            QPushButton#save_button:hover {
                background-color: #1976D2;
            }
            QPushButton#save_button:pressed {
                background-color: #1565C0;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                font-size: 14px;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-width: 150px;
                min-height: 36px;
                background-color: white;
            }
            QSpinBox::up-button, QSpinBox::down-button, 
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 24px;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 2px;
                margin: 2px;
            }
            QComboBox::drop-down {
                border: 1px solid #ddd;
                background-color: #f0f0f0;
                width: 24px;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
                background: white;
            }
            QTabBar::tab {
                background: #e0e0e0;
                border: 1px solid #ccc;
                padding: 8px 16px;
                margin: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
                margin-bottom: -1px;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            QStatusBar {
                background: #f0f0f0;
                color: #333;
                padding: 4px;
                border-top: 1px solid #ccc;
            }
        """
        self.setStyleSheet(style_sheet)

        # Piano roll and other elements are styled in _setup_ui_styles

    def generate_piano_roll(self, midi_file, display_label):
        """Generate and display a piano roll for the MIDI file with theme support."""
        # ... (rest of the code remains the same)
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Plot piano roll
            plot_piano_roll(midi_data, ax=ax)
            
            # Save plot to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            
            # Create QImage from buffer
            buf.seek(0)
            image = QImage()
            image.loadFromData(buf.getvalue())
            
            # Create and set pixmap
            pixmap = QPixmap.fromImage(image)
            display_label.setPixmap(pixmap)
            
        except Exception as e:
            logger.error(f"Piano roll oluşturma hatası: {e}", exc_info=True)
            QMessageBox.critical(None, "Hata", f"Piano roll oluşturulamadı: {str(e)}")

    def on_analysis_finished(self, analysis_result):
        """Handle analysis results."""
        if analysis_result is None:
            self.status_bar.showMessage("Analiz başarısız")
            return
            
        try:
            # Format results
            results_text = self.format_analysis_results(analysis_result)
            
            # Update UI
            self.midi_analysis_panel.update_analysis_results(results_text)
            self.status_bar.showMessage("Analiz tamamlandı")
            
            # Generate button
            self.generate_button = QPushButton("MIDI Üret")
            self.generate_button.setObjectName("generate_button")
            self.generate_button.setMinimumSize(250, 40)
            self.generate_button.clicked.connect(self._generate_midi)
            
            # Save button
            self.save_button = QPushButton("Kaydet")
            self.save_button.setObjectName("save_button")
            self.save_button.setMinimumSize(150, 40)
            self.save_button.clicked.connect(self._save_midi)
            
            buttons_layout.addWidget(self.generate_button)
            buttons_layout.addWidget(self.save_button)
            
            buttons_group = QGroupBox()
            buttons_group.setLayout(buttons_layout)
            self.main_layout.addWidget(buttons_group)
            
            logger.info("UI elements initialized successfully")
            
        except Exception as e:
            self.logger_manager.log_error(f"UI element initialization failed: {e}")
            logger.error(f"Error initializing core components: {e}", exc_info=True)
            QMessageBox.critical(self, "Hata", 
                f"Kritik bileşenler başlatılamadı: {str(e)}")
            raise
            self._core_components_initialized = False
            
            # Show error message
            QMessageBox.critical(
                self,
                "Başlatma Hatası",
                f"Uygulama bileşenleri başlatılırken hata oluştu:\n\n{e}\n\nBazı özellikler çalışmayabilir."
            )
            return
        
    def _setup_ui_elements_style(self):
        """Set up the style for UI elements."""
        # This method applies specific styles that can't be set via the main style sheet
        
        # Set object names for styling
        if hasattr(self, 'piano_roll_label'):
            self.piano_roll_label.setObjectName("piano_roll_label")
            
        if hasattr(self, 'detail_label'):
            self.detail_label.setObjectName("detail_label")
            
        # Set minimum sizes
        if hasattr(self, 'generate_button'):
            self.generate_button.setObjectName("generate_button")
            self.generate_button.setMinimumSize(250, 40)
            
        if hasattr(self, 'save_button'):
            self.save_button.setObjectName("save_button")
            self.save_button.setMinimumSize(200, 40)
            
        # Apply the main style sheet
        self._setup_ui_styles()
        
        # Configure epoch tooltip
        epoch_tooltip = """
        <html>
        <body>
        <p style="color: #b0b0b0; font-size: 10px;">
        Eğitim Adımı (Epoch):<br>
        - Modelin kaç kez tüm veri setini geçeceğinin sayısı<br>
        - Daha fazla epoch: Daha iyi öğrenme ama daha uzun süre<br>
        - Önerilen değer: 10-50<br>
        </p>
        </body>
        </html>
        """
        
        # Memory panel signals
        if hasattr(self, 'memory_panel'):
            self.memory_panel.pattern_selected.connect(self.display_pattern_details)
            self.memory_panel.memory_search_requested.connect(self.search_memory)
        
        # Settings panel signals
        if hasattr(self, 'settings_panel'):
            self.settings_panel.settings_saved.connect(self.save_settings)
            self.settings_panel.settings_loaded.connect(self.load_settings)
        
        # Model training signals
        self.train_button.clicked.connect(self.start_training)
        
        # Set initial tab
        self.tab_widget.setCurrentIndex(0)  # Set first tab as default
        
        # Initialize settings panel with current settings
        self.settings_panel.set_settings(vars(self.settings))

    def start_training(self):
        """Start model training."""
        try:
            # Get training parameters
            epochs = self.epoch_spin.value()
            batch_size = self.batch_spin.value()
            
            # Update progress bar
            self.train_progress.setValue(0)
            self.train_progress.setFormat("%p% - Başlatılıyor...")
            self.statusBar().showMessage("Model eğitimi başlatılıyor...")
            
            # Create worker thread for training
            self.worker_thread = Worker(
                task="train_model",
                data={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "patterns": self.midi_memory.get_all_patterns()
                },
                midi_model=self.midi_model,
                total_epochs=epochs
            )
            
            # Connect signals
            self.worker_thread.training_progress.connect(self.update_training_progress)
            self.worker_thread.training_finished.connect(self.on_training_finished)
            
            # Start worker thread
            self.worker_thread.start()
            
            # Disable train button during training
            self.train_button.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Model eğitimi başlatılırken hata: {e}")
            QMessageBox.critical(self, "Hata", 
                f"Model eğitimi başlatılamadı:\n\n{str(e)}")

    def update_training_progress(self, epoch, total_epochs, loss):
        """Update training progress."""
        progress = (epoch / total_epochs) * 100
        self.train_progress.setValue(int(progress))
        self.train_progress.setFormat(f"%p% - Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
        self.statusBar().showMessage(f"Model eğitimi: Epoch {epoch}/{total_epochs}")

    def on_training_finished(self, trained_model):
        """Handle training completion."""
        if trained_model:
            logger.info("Model eğitimi tamamlandı")
            QMessageBox.information(self, "Başarılı", "Model eğitimi tamamlandı!")
            self.statusBar().showMessage("Model eğitimi tamamlandı")
        else:
            logger.error("Model eğitimi başarısız oldu")
            QMessageBox.warning(self, "Uyarı", "Model eğitimi başarısız oldu")
            self.statusBar().showMessage("Model eğitimi başarısız")
            
        # Re-enable train button
        self.train_button.setEnabled(True)
        
    def analyze_midi_file(self, midi_file_path):
        """Analyze the selected MIDI file."""
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
        
        if analysis_result:
            logger.info("MIDI analysis completed successfully")
            
            # Format analysis results as text
            analysis_text = self.format_analysis_results(analysis_result)
            
            # Display analysis results
            self.analysis_panel.display_analysis_results(analysis_text)
            
            # Generate and display piano roll
            self.generate_piano_roll(analysis_result.file_path, self.analysis_panel.piano_roll_display)
            
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
            
        text = f"Dosya: {os.path.basename(analysis_result.file_path)}\n"
        text += f"Uzunluk: {analysis_result.duration:.2f} saniye\n"
        text += f"Tempo: {analysis_result.tempo:.2f} BPM\n"
        text += f"Zaman İmzası: {analysis_result.time_signature}\n"
        text += f"Ton: {analysis_result.key}\n\n"
        
        # Display instrument information
        if hasattr(analysis_result, 'instrument_names') and analysis_result.instrument_names:
            text += "Enstrümanlar:\n"
            for i, (name, count) in enumerate(analysis_result.instrument_names.items(), 1):
                text += f"  {i}. {name} - {count} nota\n"
        
        # Display program information
        if hasattr(analysis_result, 'instrument_programs') and analysis_result.instrument_programs:
            text += "\nProgram Numaraları:\n"
            for prog, count in analysis_result.instrument_programs.items():
                text += f"  Program {prog}: {count} nota\n"
            
        text += "\nNot İstatistikleri:\n"
        text += f"  Toplam Nota Sayısı: {analysis_result.note_count}\n"
        text += f"  Ortalama Hız: {analysis_result.average_velocity:.2f}\n"
        text += f"  Nota Aralığı: {analysis_result.pitch_range[0]}-{analysis_result.pitch_range[1]} (MIDI notası)\n"
        text += f"  Ritim Karmaşıklığı: {analysis_result.rhythm_complexity:.4f}\n"
        
        # Display polyphony information if available
        if hasattr(analysis_result, 'polyphony_profile') and analysis_result.polyphony_profile:
            max_poly = max(analysis_result.polyphony_profile.values())
            avg_poly = sum(analysis_result.polyphony_profile.values()) / len(analysis_result.polyphony_profile)
            text += f"  Maksimum Eşzamanlı Not: {max_poly}\n"
            text += f"  Ortalama Eşzamanlı Not: {avg_poly:.2f}\n"
                
        return text

    def on_generation_finished(self, generated_sequence):
        """Handle generated MIDI sequence."""
        if generated_sequence is None:
            self.status_bar.showMessage("MIDI üretimi başarısız oldu!")
            self.generation_panel.update_progress(100, "Hata")
            QMessageBox.critical(self, "Hata", "MIDI üretimi başarısız oldu!")
            return
            
        try:
            # Get parameters from UI
            bar_count = self.bar_count_spin.value()
            tempo = self.tempo_spin.value()
            style = self.style_combo.currentText()
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.settings.output_dir,
                f"midi_{timestamp}_bar{bar_count}_tempo{tempo}_{style.lower()}.mid"
            )
            
            # Save MIDI file
            generated_sequence.write(output_file)
            
            # Update status
            self.status_bar.showMessage(f"MIDI dosyası kaydedildi: {output_file}")
            self.generation_panel.update_progress(100, "Başarılı")
            
            # Show success message
            QMessageBox.information(
                self,
                "Başarılı",
                f"MIDI üretimi tamamlandı!\nDosya kaydedildi: {output_file}"
            )
            
        except Exception as e:
            logger.error(f"Error saving MIDI file: {e}")
            self.status_bar.showMessage("MIDI dosyası kaydedilemedi!")
            self.generation_panel.update_progress(100, "Hata")

    def generate_piano_roll(self, midi_file, display_label):
        """Generate and display a piano roll for the MIDI file."""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            # Create piano roll widget
            piano_roll_widget = create_piano_roll_widget(midi_data)
            
            if piano_roll_widget is None:
                logger.error("Failed to create piano roll widget")
                return
                
            # Clear existing layout if any
            if display_label.layout():
                layout = display_label.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            
            # Create a layout for the widget
            layout = QVBoxLayout()
            layout.addWidget(piano_roll_widget)
            
            # Set the layout to the display widget
            display_label.setLayout(layout)
            
            # Update status
            self.status_bar.showMessage(f"Piyano rulosu oluşturuldu: {os.path.basename(midi_file)}")
            
            logger.info(f"Piano roll generated for: {midi_file}")
            
        except Exception as e:
            logger.error(f"Error generating piano roll: {e}", exc_info=True)
            QMessageBox.critical(self, "Hata", f"Piyano rulosu oluşturulamadı: {e}")
            # Clear display in case of error
            if display_label.layout():
                layout = display_label.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            display_label.setLayout(None)
        except Exception as e:
            logger.error(f"Error generating piano roll: {e}")
            display_label.clear()
        if generated_sequence is None:
            self.status_bar.showMessage("MIDI üretimi başarısız oldu!")
            self.generation_panel.update_progress(100, "Hata")
            return

        try:
            # Update progress
            self.generation_panel.update_progress(50, "MIDI dosyası oluşturuluyor...")
            
            # Convert sequence to MIDI
            midi_data = self.processor.sequence_to_midi(
                generated_sequence,
                tempo=self.generation_panel.tempo_spin.value(),
                style=self.generation_panel.style_combo.currentText()
            )
            
            # Save MIDI file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            style = self.generation_panel.style_combo.currentText()
            tempo = self.generation_panel.tempo_spin.value()
            bar_count = self.generation_panel.bar_count_spin.value()
            
            # Create filename with style, tempo, and bar count
            filename = f"generated_{style.lower()}_{tempo}bpm_{bar_count}bars_{timestamp}.mid"
            output_path = os.path.join(self.settings.output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(self.settings.output_dir, exist_ok=True)
            
            # Save MIDI file
            midi_data.write(output_path)
            
            # Update status bar
            self.status_bar.showMessage(f"MIDI dosyası başarıyla oluşturuldu: {filename}")
            
            # Update piano roll
            self.update_piano_roll(midi_data)
            
            # Update progress
            self.generation_panel.update_progress(100, "Tamamlandı")
            
        except Exception as e:
            logger.error(f"MIDI oluşturma sonrası işlem hatası: {e}", exc_info=True)
            self.status_bar.showMessage("MIDI oluşturma sonrası işlem hatası!")
            self.generation_panel.update_progress(100, "Hata")
            QMessageBox.critical(self, "Hata", f"MIDI oluşturma sonrası işlem hatası: {str(e)}")

    def search_memory(self, category):
        """Search memory patterns by category."""
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
        self.worker_thread.start()

    def on_memory_search_finished(self, patterns):
        """Handle memory search results."""
        
        # Update pattern list
        self.memory_panel.update_pattern_list(patterns)
        
        # Update status
        pattern_count = len(patterns)
        self.status_bar.showMessage(f"Hafıza araması tamamlandı: {pattern_count} desen bulundu")
        
        logger.info(f"Memory search completed: {pattern_count} patterns found")
        
    def display_pattern_details(self, pattern_id):
        """Display details of the selected pattern."""
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
                    
                    # Create piano roll widget
                    piano_roll_widget = create_piano_roll_widget(midi_data)
                    
                    if piano_roll_widget is not None:
                        pattern_info['piano_roll_widget'] = piano_roll_widget
                    else:
                        logger.error("Failed to create piano roll widget")
                        pattern_info['piano_roll_widget'] = None
                        
                except Exception as e:
                    logger.error(f"Error generating piano roll: {e}")
                    pattern_info['piano_roll_widget'] = None
            
            # Display pattern details in memory panel
            self.memory_panel.display_pattern_details(pattern_info)
            
        else:
            logger.error(f"Pattern not found: {pattern_id}")
            self.memory_panel.display_pattern_details(None)

    def update_piano_roll(self, midi_data):
        """Update the piano roll display with new MIDI data."""
        try:
            # Clear the previous plot if it exists
            if self.piano_roll_layout:
                for i in reversed(range(self.piano_roll_layout.count())):
                    widget = self.piano_roll_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.deleteLater()
            
            # Generate the piano roll plot with the updated styling
            fig = plot_piano_roll(midi_data)
            
            # Create a new canvas with the figure
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            
            # Configure size policies for proper expansion
            canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            
            # Set minimum size to ensure visibility
            canvas.setMinimumSize(600, 350)
            
            # Add the canvas to the layout with stretch factor
            if self.piano_roll_layout:
                self.piano_roll_layout.addWidget(canvas, stretch=1)
                
                # Adjust layout margins and spacing for better appearance
                self.piano_roll_layout.setContentsMargins(10, 10, 10, 10)
                self.piano_roll_layout.setSpacing(10)
            
            # Force an immediate update of the layout
            canvas.draw()
            
            # Store the figure reference to prevent garbage collection
            self._current_figure = fig
            
            # Schedule a resize event to ensure proper rendering
            canvas.resizeEvent = lambda event: self._on_canvas_resize(canvas, event)
            
            # Update status bar
            self.status_bar.showMessage("Piano roll güncellendi")
            
        except Exception as e:
            logger.error(f"Error updating piano roll: {e}", exc_info=True)
            # Show a more informative error message
            error_label = QLabel(f"Piano roll görüntülenirken hata: {str(e)}")
            error_label.setStyleSheet("""
                color: #ff6b6b; 
                font-size: 12px;
                font-weight: bold;
                padding: 10px;
                background-color: rgba(40, 40, 40, 0.7);
                border-radius: 5px;
            """)
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if self.piano_roll_layout:
                self.piano_roll_layout.addWidget(error_label)
            
            # Update status bar with error
            self.status_bar.showMessage(f"Piano roll güncellenirken hata: {str(e)[:50]}...")
    
    def _on_canvas_resize(self, canvas, event):
        """Handle canvas resize events to maintain aspect ratio."""
        # Get the current figure size
        fig = canvas.figure
        fig.set_size_inches(canvas.width()/fig.dpi, canvas.height()/fig.dpi)
        canvas.draw_idle()
            
    def save_settings(self, settings_dict):
        """Save settings to the application."""
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
            
            # Draw background image
            background_path = r"C:\Users\lombi\midi_composer_test\resources\images\background.jpg"
            if os.path.exists(background_path):
                background = QImage(background_path)
                if not background.isNull():
                    painter.drawImage(self.rect(), background)
            
            # Draw semi-transparent overlay
            painter.fillRect(self.rect(), QColor(0, 0, 0, 128))
            
        except Exception as e:
            logger.error(f"Error in paintEvent: {e}")
            QMessageBox.warning(self, "Uyarı", 
                f"Arka plan fotoğrafı yüklenirken hata oluştu:\n\n{str(e)}")
            
        finally:
            # Ensure painter is properly destroyed
            if painter.isActive():
                painter.end()
            
        # Call base class implementation
        super().paintEvent(event)


def create_piano_roll_widget(midi_data, fs=100, start_pitch=21, end_pitch=108):
    """
    Creates a piano roll widget for the MIDI data.
    
    Args:
        midi_data: pretty_midi.PrettyMIDI object
        fs: Sampling frequency for the piano roll
        start_pitch: Starting MIDI pitch (21 = A0)
        end_pitch: Ending MIDI pitch (108 = C8)
        
    Returns:
        QWidget: The widget containing the piano roll plot
    """
    if midi_data is None:
        logger.warning("No MIDI data provided for piano roll plotting.")
        return None

    try:
        # Get the piano roll
        piano_roll = midi_data.get_piano_roll(fs=fs)
        
        # Trim to the specified pitch range
        piano_roll = piano_roll[start_pitch:end_pitch]
        
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Set transparent background for the figure and axes
        fig.patch.set_facecolor('transparent')
        ax.set_facecolor('transparent')

        # Plot the piano roll with a theme-appropriate colormap
        im = ax.imshow(
            piano_roll,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='Blues'  # Using Blues colormap for better visibility
        )

        # Set labels and tick colors for theme consistency
        ax.set_ylabel('Pitch', color='#e0e0e0')
        ax.set_xlabel('Time (Bars)', color='#e0e0e0')
        ax.tick_params(axis='x', colors='#e0e0e0')
        ax.tick_params(axis='y', colors='#e0e0e0')
        
        # Set y-axis ticks to show note names
        y_tick_positions = []
        y_tick_labels = []
        
        # Label every octave (C notes) for clarity
        for i, pitch in enumerate(range(start_pitch, end_pitch + 1)):
            note_name = pretty_midi.note_number_to_name(pitch)
            if note_name.startswith('C') or i == 0 or i == (end_pitch - start_pitch):
                y_tick_positions.append(i)
                y_tick_labels.append(note_name)
        
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)

        # Add grid
        ax.grid(alpha=0.3, color='#808080')
        
        # Add color bar for better visualization
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Velocity', color='#e0e0e0')
        cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')
        
        # Create a canvas for the figure
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: transparent;")
        
        # Configure size policies
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Set minimum size
        canvas.setMinimumSize(600, 350)
        
        # Return the canvas as the widget
        return canvas
        
    except Exception as e:
        logger.error(f"Error creating piano roll widget: {e}", exc_info=True)
        return None

        # --- Bar/Measure Line Calculation and Plotting ---
        # Get time signature (default to 4/4 if not specified)
        if midi_data.time_signature_changes:
            # Use the first time signature (assume it applies from the beginning)
            time_sig = midi_data.time_signature_changes[0]
            numerator = time_sig.numerator
            denominator = time_sig.denominator
        else:
            # Default to 4/4 if no time signature is found
            numerator = 4
            denominator = 4
            logger.info("No time signature found, defaulting to 4/4.")
        
        # Get downbeat times (start of each measure) in seconds
        downbeat_times = midi_data.get_downbeats()
        
        # Calculate the total number of samples (width of the piano roll)
        total_samples = piano_roll.shape[1]
        
        bar_x_positions = []
        bar_labels = []
        
        # Convert downbeat times to x-positions in the plot
        for i, beat_time in enumerate(downbeat_times):
            x_pos_samples = int(beat_time * fs)  # Convert seconds to samples
            if x_pos_samples < total_samples:  # Only plot if within the visible range
                bar_x_positions.append(x_pos_samples)
                bar_labels.append(str(i + 1))  # Bar numbers start from 1

        # Plot vertical lines for each bar
        for x_pos in bar_x_positions:
            ax.axvline(
                x=x_pos, 
                color='#00BCD4',  # Accent color
                linestyle='--', 
                linewidth=0.8, 
                alpha=0.7
            )

        # Set x-axis ticks at bar positions with bar numbers as labels
        if bar_x_positions:  # Only if we have valid bar positions
            ax.set_xticks(bar_x_positions)
            ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        # Customize spine colors to match theme
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
        
        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()
        
        # Create a canvas for the figure
        canvas = FigureCanvas(fig)
        canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        canvas.setStyleSheet("background-color: transparent;")
        
        # Configure size policies for proper expansion
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Set minimum size
        return None

# Application entry point is now in main.py

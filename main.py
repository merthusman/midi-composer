# src/main.py
# 1. Aşama: Sadece en temel kütüphaneleri yükle
import os
import sys
import time
import logging
import traceback
import atexit
import json
import platform
import datetime
from pathlib import Path

# 2. Aşama: Sadece Qt'nin temel modüllerini yükle
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel, QProgressBar, QMessageBox, QStyle, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer, QRect, QSize, QFile, QTextStream
from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont, QBrush, QLinearGradient, QIcon, QFontDatabase

# Logger'ı yapılandır
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('midi_composer.log')
    ]
)
logger = logging.getLogger(__name__)

# Settings sınıfını içe aktar
try:
    from src.core.settings import Settings
except ImportError as e:
    print(f"Settings modülü yüklenirken hata oluştu: {e}")
    # Settings modülü yüklenemezse, basit bir sınıf oluştur
    class Settings:
        def __init__(self):
            self.model_settings = type('ModelSettings', (), {})()
            self.general_settings = type('GeneralSettings', (), {})()
            self.memory_settings = type('MemorySettings', (), {})()
            
        def load(self, *args, **kwargs):
            pass
            
        def save(self, *args, **kwargs):
            return "{}"

# 3. Aşama: Hemen bir QApplication oluştur
app = QApplication(sys.argv)

# 4. Aşama: Basit bir splash ekranı oluştur
class SimpleSplashScreen(QSplashScreen):
    """Basit bir splash ekranı sınıfı.
    
    Uygulama başlatılırken gösterilecek basit bir splash ekranı sağlar.
    """
    
    def __init__(self, app, pixmap=None):
        # Eğer pixmap verilmediyse, varsayılan bir arkaplan oluştur
        if pixmap is None or not isinstance(pixmap, QPixmap) or pixmap.isNull():
            pixmap = self._create_default_pixmap()
        
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)
        
        # Pencere özelliklerini ayarla
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # Uygulama referansını sakla
        self.app = app
        
        # Pencere boyutunu ayarla
        self.setFixedSize(pixmap.size())
        
        # Pencereyi ekranın ortasına yerleştir
        screen = app.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(100, 250, 600, 25)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 12px;
                background-color: rgba(0, 0, 0, 0.4);
                text-align: center;
                color: white;
                font-weight: bold;
                padding: 2px;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0.5, x2:1, y2:0.5,
                    stop:0 #00b4ff,
                    stop:0.25 #00ff88,
                    stop:0.5 #00ffea,
                    stop:0.75 #b700ff,
                    stop:1 #ff00c8
                );
                border-radius: 10px;
                margin: 1px;
            }
        """)
        
        # Durum etiketi
        self.status_label = QLabel("Başlatılıyor...", self)
        self.status_label.setGeometry(100, 290, 600, 30)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 13px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 4px 8px;
            }
        """)
        
        # Görünürlük bayraklarını ayarla
        self._is_visible = False
        
        # İlk gösterim
        self.show()
        app.processEvents()
        self._is_visible = True
    
    def showEvent(self, event):
        """Pencere gösterildiğinde çağrılır"""
        super().showEvent(event)
        self._is_visible = True
    
    def hideEvent(self, event):
        """Pencere gizlendiğinde çağrılır"""
        super().hideEvent(event)
        self._is_visible = False
    
    def closeEvent(self, event):
        """Pencere kapatıldığında çağrılır"""
        self._is_visible = False
        super().closeEvent(event)
    
    def is_visible(self):
        """Pencerenin görünür olup olmadığını döndürür"""
        return self._is_visible and self.isVisible()
    
    def _create_default_pixmap(self):
        """Varsayılan arkaplan resmini oluşturur."""
        # 800x500 boyutunda pixmap oluştur
        pixmap = QPixmap(800, 500)
        
        # Arkaplan rengi için gradient oluştur
        painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, 0, 500)
        gradient.setColorAt(0.0, QColor(26, 35, 126))  # Koyu mavi
        gradient.setColorAt(0.5, QColor(13, 71, 161))   # Orta mavi
        gradient.setColorAt(1.0, QColor(26, 35, 126))   # Koyu mavi
        painter.fillRect(pixmap.rect(), QBrush(gradient))
        
        # Başlık ekle
        title_font = QFont('Arial', 36, QFont.Weight.Bold)
        painter.setFont(title_font)
        
        # Başlık arka planı (yarı saydam siyah yuvarlak köşeli dikdörtgen)
        title_rect = QRect(50, 80, 700, 80)
        title_bg = QRect(50, 80, 700, 80)
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(title_bg, 10, 10)
        
        # Başlık metni (beyaz renkte)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "MIDI COMPOSER")
        
        # Versiyon bilgisi (sağ alt köşede)
        version_font = QFont('Arial', 9)
        painter.setFont(version_font)
        version_rect = QRect(500, 450, 250, 20)
        painter.drawText(version_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, "v1.0.0")
        
        # Durum metni (ortada, daha yukarıda)
        status_font = QFont('Arial', 12, QFont.Weight.Medium)
        painter.setFont(status_font)
        status_rect = QRect(50, 350, 700, 40)
        painter.drawText(status_rect, Qt.AlignmentFlag.AlignCenter, "Başlatılıyor...")
        
        painter.end()
        return pixmap
    
    def setPixmap(self, pixmap):
        """Splash ekranının arkaplan resmini ayarlar."""
        super().setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        
        # Pencereyi ekranın ortasına yerleştir
        screen = self.app.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def set_progress(self, value, message=None):
        """İlerleme durumunu günceller.
        
        Args:
            value (int): 0-100 arasında ilerleme değeri
            message (str, optional): Güncellenecek durum mesajı
            
        Returns:
            bool: İşlem başarılı olduysa True, aksi takdirde False
        """
        try:
            # Eğer pencere kapatıldıysa veya geçersizse işlemi iptal et
            if not hasattr(self, 'progress_bar') or not self.is_visible():
                return False
                
            # İlerleme değerini sınırla (0-100 arası)
            progress = max(0, min(100, int(value)))
            
            # İlerleme çubuğunu güncelle
            if hasattr(self, 'progress_bar') and self.progress_bar is not None:
                self.progress_bar.setValue(progress)
            
            # Durum mesajını güncelle
            if message is not None and hasattr(self, 'status_label') and self.status_label is not None:
                self.status_label.setText(str(message))
                
            # Arayüzü güncelle
            QApplication.processEvents()
            return True
            
        except RuntimeError as e:
            # C++ nesnesi silinmiş olabilir
            if 'wrapped C/C++ object' in str(e):
                self._is_visible = False
                return False
            logger.error(f"Splash ekranı güncellenirken çalışma zamanı hatası: {e}", exc_info=True)
            return False
            
        except Exception as e:
            logger.error(f"Splash ekranı güncellenirken hata: {e}", exc_info=True)
            return False

# Uygulama başlatma işlemleri buraya gelecek
# ...

def check_single_instance():
    """Uygulamanın sadece tek örnek olarak çalışmasını sağlar.
    
    Returns:
        tuple: (is_running, lock_file)
            - is_running (bool): Uygulama zaten çalışıyorsa True
            - lock_file: Kilit dosyası veya None
    """
    import os
    import sys
    import ctypes
    import tempfile
    
    # Windows için özel bir mutex kullanıyoruz
    mutex_name = 'Global\\MIDIComposerSingleInstanceMutex'
    
    try:
        # Mutex oluşturmayı dene
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
        last_error = ctypes.windll.kernel32.GetLastError()
        
        # Eğer mutex zaten varsa (ERROR_ALREADY_EXISTS), uygulama zaten çalışıyordur
        if last_error == 183:  # ERROR_ALREADY_EXISTS
            return True, None
            
        # Diğer hatalar için kontrol
        if mutex is None or mutex == 0:
            logger.warning(f"Mutex oluşturulamadı. Hata kodu: {last_error}")
            return False, None
            
        # Mutex başarıyla oluşturuldu, uygulama çalışmıyor
        # Uygulama kapatıldığında mutex'i serbest bırakmak için bir dosya oluşturalım
        lock_file = os.path.join(tempfile.gettempdir(), 'midi_composer.lock')
        try:
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            return False, lock_file
        except Exception as e:
            logger.warning(f"Kilit dosyası oluşturulamadı: {e}")
            return False, None
            
    except Exception as e:
        logger.warning(f"Tek örnek kontrolü sırasında hata: {e}")
        return False, None

def cleanup_lock_file(lock_file):
    """Uygulama kapatılırken kilit dosyasını siler."""
    if lock_file:
        try:
            if os.name == 'nt':
                # Windows'ta kilidi serbest bırak
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except (IOError, OSError):
                    pass
            else:
                # Unix/Linux'da kilidi serbest bırak
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (IOError, OSError):
                    pass
            
            # Dosyayı kapat
            lock_file.close()
            
            # Kilit dosyasını sil
            if os.path.exists(lock_file.name):
                try:
                    os.unlink(lock_file.name)
                except (IOError, OSError):
                    pass
        except Exception as e:
            logger.warning(f"Kilit dosyası temizlenirken hata: {e}")

# Logger yapılandırması
try:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('midi_composer.log')
        ]
    )
    logger = logging.getLogger(__name__)
except Exception as e:
    print(f"Logger yapılandırılırken hata oluştu: {e}")
    logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Gerekli modülleri yükle
    import msvcrt  # Windows için
    
    # Çift başlamayı önlemek için tek örnek kontrolü yap
    is_running, lock_file = check_single_instance()
    if is_running:
        print("Uygulama zaten çalışıyor!")
        sys.exit(1)
    
    try:
        # Uygulamayı çalıştır
        sys.exit(run_application())
    finally:
        # Kilit dosyasını temizle
        if 'lock_file' in locals() and lock_file:
            cleanup_lock_file(lock_file)

# 5. Aşama: Diğer kütüphaneleri adım adım yükle
try:
    # Temel Python kütüphaneleri
    import json
    import atexit
    import logging
    import logging.handlers
    import traceback
    import tempfile
    import datetime
    splash.set_progress(10, "Python kütüphaneleri yükleniyor...")
    
    # Windows API kütüphaneleri
    import win32gui
    import win32process
    import win32con
    import win32event
    import win32api
    import winerror
    splash.set_progress(20, "Sistem kütüphaneleri yükleniyor...")
    
    # Qt kütüphaneleri
    from PyQt6.QtWidgets import (QMessageBox, QVBoxLayout, QWidget, QFileDialog)
    from PyQt6.QtCore import (QSettings, QTranslator, QLocale, QLibraryInfo, 
                             QSize, QPoint)
    from PyQt6.QtGui import QIcon, QFont, QLinearGradient, QBrush
    splash.set_progress(30, "Arayüz bileşenleri yükleniyor...")
    
    # Uygulama modülleri
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Uygulama modüllerini yükle
    try:
        from src.core.settings import Settings
        from src.midi.instrument_library import InstrumentLibrary
        from src.model.midi_model import MIDIModel
        from src.midi.midi_memory import MIDIMemory
        splash.set_progress(45, "Uygulama modülleri yükleniyor...")
    except ImportError as e:
        splash.set_progress(100, f"Hata: Modül yüklenemedi: {e}")
        time.sleep(3)
        sys.exit(1)
        
except Exception as e:
    splash.set_progress(100, f"Hata: {str(e)}")
    time.sleep(3)
    sys.exit(1)

# Ensure src directory is added to sys.path so inner modules can be imported as packages
# __file__ is src/main.py
# os.path.dirname(__file__) is src/
# os.path.join(os.path.dirname(__file__), '..') is project_root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Settings class
try:
    from src.core.settings import Settings
    _settings_imported_in_main = True
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import Settings module: {e}", file=sys.stderr)
    _settings_imported_in_main = False

# Configure logging
logs_dir = os.path.join(project_root, "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

log_file_path = os.path.join(logs_dir, "app.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Application logging configured.")
logger.info(f"Project root: {project_root}")

# Import GUI modules
try:
    from src.gui.main_window import MainWindow
    from src.gui.panels.settings_panel import SettingsPanel
    _gui_modules_imported = True
except ImportError as e:
    logger.critical(f"Failed to import GUI modules: {e}", exc_info=True)
    _gui_modules_imported = False


def is_process_running(pid):
    """Belirtilen PID'in çalışıp çalışmadığını kontrol eder."""
    if not isinstance(pid, int) or pid <= 0:
        return False
        
    try:
        # Windows'ta process kontrolü
        if os.name == 'nt':
            import ctypes
            from ctypes.wintypes import BOOL, DWORD, HANDLE
            
            # Process handle'ı aç
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            process = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, 
                False,  # bInheritHandle
                pid
            )
            
            if process == 0:
                # Process yok veya erişim reddedildi
                return False
                
            # Process hala çalışıyor mu kontrol et
            exit_code = ctypes.wintypes.DWORD()
            success = ctypes.windll.kernel32.GetExitCodeProcess(
                process, 
                ctypes.byref(exit_code)
            )
            
            # Handle'ı kapat
            ctypes.windll.kernel32.CloseHandle(process)
            
            # GetExitCodeProcess başarısız olduysa veya process hala çalışıyorsa (STILL_ACTIVE = 259)
            if not success or exit_code.value == 259:
                return True
            return False
            
        else:
            # Unix benzeri sistemlerde process kontrolü
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                # Process var ama erişim iznimiz yok
                return True
            except:
                return False
            return True
            
    except Exception as e:
        debug_info(f"Process kontrolü sırasında hata (PID: {pid}): {e}")
        return False

def cleanup_lock_file(lock_file):
    """Uygulama kapatılırken kilit dosyasını ve mutex'i temizler."""
    if not lock_file:
        return
        
    try:
        # Windows'ta mutex'i serbest bırak
        import ctypes
        try:
            ctypes.windll.kernel32.ReleaseMutex(lock_handle)
            ctypes.windll.kernel32.CloseHandle(lock_handle)
        except Exception as e:
            if hasattr(logger, 'warning'):
                logger.warning(f"Mutex serbest bırakılırken hata: {e}")
    except Exception as e:
        if hasattr(logger, 'error'):
            logger.error(f"Mutex temizleme sırasında beklenmeyen hata: {e}", exc_info=True)

def check_single_instance():
    """Uygulamanın sadece tek örnek olarak çalışmasını sağlar.
    
    Windows için mutex kullanarak tek örnek kontrolü yapar.
    
    Returns:
        tuple: (is_running, mutex_handle)
            - is_running (bool): Uygulama zaten çalışıyorsa True
            - mutex_handle: Mutex tutucusu veya None
    """
    import ctypes
    import ctypes.wintypes
    
    # Mutex adı (global olmalı)
    mutex_name = "Global\\MIDIComposerSingleInstanceMutex"
    
    try:
        # Mutex oluştur veya aç
        mutex = ctypes.windll.kernel32.CreateMutexW(
            None,  # Güvenlik özniteliği
            False,  # İlk oluşturan işlem değil
            mutex_name  # Mutex adı
        )
        
        # Hata kontrolü
        last_error = ctypes.windll.kernel32.GetLastError()
        
        # Eğer mutex zaten varsa (ERROR_ALREADY_EXISTS)
        if last_error == 183:  # ERROR_ALREADY_EXISTS
            logger.info("Uygulamanın başka bir örneği zaten çalışıyor.")
            return True, None
            
        # Diğer hatalar
        if mutex is None or mutex == 0:
            error_msg = f"Mutex oluşturulamadı. Hata kodu: {last_error}"
            logger.error(error_msg)
            debug_info(error_msg)
            return False, None
        
        logger.info("Mutex başarıyla oluşturuldu.")
        return False, mutex
        
    except Exception as e:
        error_msg = f"Tek örnek kontrolü sırasında hata: {e}"
        logger.error(error_msg, exc_info=True)
        debug_info(error_msg)
        return False, None

def activate_existing_window():
    """
    Zaten çalışan uygulama penceresini öne getirir.
    
    Returns:
        bool: Pencere bulunup etkinleştirilebilirse True, aksi halde False
    """
    try:
        debug_info("Mevcut uygulama penceresi öne getiriliyor...")
        
        # Gerekli modülleri kontrol et
        try:
            import win32gui
            import win32con
            import win32process
            import win32api
            import win32com.client
            import psutil
        except ImportError as e:
            debug_info(f"Gerekli modüller yüklenemedi: {e}")
            return False
        
        current_pid = os.getpid()
        window_title = "MIDI Composer"  # Pencere başlığı
        
        debug_info(f"Mevcut pencere aranıyor (Başlık: {window_title}, PID: {current_pid})...")
        
        def enum_windows_callback(hwnd, hwnds):
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                    
                # Pencere başlığını kontrol et
                title = win32gui.GetWindowText(hwnd)
                if not title or window_title not in title:
                    return True
                
                # Pencere işlem ID'sini al
                _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                
                # Aynı uygulamanın diğer pencerelerini bul
                if found_pid == current_pid:
                    return True
                
                # İşlem adını kontrol et
                try:
                    process = psutil.Process(found_pid)
                    process_name = process.name().lower()
                    
                    # Sadece Python işlemlerini kontrol et
                    if "python" not in process_name and "pythonw" not in process_name:
                        return True
                        
                    # İşlem komut satırını kontrol et (isteğe bağlı, daha güvenli ama yavaş olabilir)
                    try:
                        cmdline = " ".join(process.cmdline())
                        if "midi_composer" not in cmdline.lower() and "main.py" not in cmdline.lower():
                            return True
                    except (psutil.AccessDenied, AttributeError):
                        pass
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return True
                
                debug_info(f"Eşleşen pencere bulundu: HWND={hwnd}, PID={found_pid}, Başlık='{title}'")
                hwnds.append((hwnd, found_pid))
                
            except Exception as e:
                debug_info(f"Pencere arama sırasında hata: {e}")
                
            return True
        
        hwnds = []
        win32gui.EnumWindows(enum_windows_callback, hwnds)
        
        if not hwnds:
            debug_info("Eşleşen pencere bulunamadı")
            return False
        
        # Pencere sıralı olarak işle (en son kullanılan pencere önce)
        hwnds.sort(key=lambda x: win32gui.GetWindowLong(x[0], win32con.GWL_STYLE), reverse=True)
        
        # Tüm eşleşen pencereleri işle
        for hwnd, pid in hwnds:
            try:
                debug_info(f"Pencere etkinleştiriliyor: HWND={hwnd}, PID={pid}")
                
                # Pencere durumunu kontrol et ve gerekirse restore et
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                
                # Pencereyi öne getir
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)
                
                # Pencere boyutunu ve konumunu güncelle
                win32gui.SetWindowPos(
                    hwnd,
                    win32con.HWND_TOPMOST,  # En üste getir
                    0, 0, 0, 0,  # Konum ve boyut değişmeyecek
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
                )
                
                # Pencereyi tekrar normal seviyeye getir
                win32gui.SetWindowPos(
                    hwnd,
                    win32con.HWND_NOTOPMOST,  # En üstte olma özelliğini kaldır
                    0, 0, 0, 0,  # Konum ve boyut değişmeyecek
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
                )
                
                # Pencereye odaklan ve aktifleştir
                win32gui.SetForegroundWindow(hwnd)
                win32gui.SetActiveWindow(hwnd)
                win32gui.SetFocus(hwnd)
                
                # Pencereyi kullanıcının karşısına getir
                try:
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shell.SendKeys('%')
                except:
                    pass
                
                debug_info(f"Pencere başarıyla öne getirildi: HWND={hwnd}")
                return True
                
            except Exception as e:
                debug_info(f"Pencere etkinleştirilirken hata (HWND={hwnd}): {e}")
        
        debug_info("Hiçbir pencere etkinleştirilemedi")
        return False
        
    except Exception as e:
        debug_info(f"Pencere etkinleştirme sırasında beklenmeyen hata: {e}")
        import traceback
        debug_info(f"Hata detayı: {traceback.format_exc()}")
        return False

def check_window_visibility(window):
    """
    Pencere görünürlüğünü kontrol eder ve gerekirse düzeltir.
    
    Args:
        window: Kontrol edilecek QMainWindow örneği
    """
    try:
        debug_info("Pencere görünürlüğü kontrol ediliyor...")
        
        # Pencere görünür değilse veya simge durumundaysa
        if not window.isVisible() or window.isMinimized():
            debug_info("Pencere görünür değil veya simge durumunda, düzeltiliyor...")
            window.show()
            window.activateWindow()
            window.raise_()
            
            # Pencere boyutunu ve konumunu kontrol et
            screen = QApplication.primaryScreen().availableGeometry()
            window_size = window.size()
            
            # Pencere çok büyükse boyutlandır
            if window_size.width() > screen.width() or window_size.height() > screen.height():
                debug_info("Pencere boyutu çok büyük, yeniden boyutlandırılıyor...")
                window.resize(min(1200, screen.width() * 0.9), min(800, screen.height() * 0.9))
            
            # Pencere ekran dışındaysa konumlandır
            if not screen.contains(window.geometry()):
                debug_info("Pencere ekran dışında, konumlandırılıyor...")
                window.move(
                    (screen.width() - window.width()) // 2,
                    (screen.height() - window.height()) // 2
                )
            
            debug_info("Pencere başarıyla düzeltildi")
        else:
            debug_info("Pencere zaten görünür durumda")
            
    except Exception as e:
        debug_info(f"Pencere görünürlüğü kontrol edilirken hata: {e}")
        import traceback
        debug_info(f"Hata detayı: {traceback.format_exc()}")

def load_stylesheet(app, qss_path=None):
    """
    Uygulama stil dosyasını yükler ve uygular.
    
    Args:
        app: QApplication örneği
        qss_path: Stil dosyasının yolu. None ise varsayılan stil dosyası kullanılır.
        
    Returns:
        bool: Stil dosyası başarıyla yüklendiyse True, aksi halde False
    """
    try:
        # Eğer stil dosyası yolu belirtilmemişse, varsayılan yolu kullan
        if qss_path is None:
            # Uygulama kök dizinini bul
            if getattr(sys, 'frozen', False):
                # PyInstaller ile paketlenmiş uygulama
                project_root = Path(sys._MEIPASS)
            else:
                # Geliştirme ortamı
                project_root = Path(__file__).parent.parent
                
            qss_path = project_root / 'resources' / 'styles' / 'app.qss'
        
        # Yolu string'e çevir (Path objesi olabilir)
        qss_path = str(qss_path)
        
        if not os.path.exists(qss_path):
            logger.warning(f"Stil dosyası bulunamadı: {qss_path}")
            return False
            
        with open(qss_path, 'r', encoding='utf-8') as f:
            style_sheet = f.read()
            
        # Stil dosyası boş mu kontrol et
        if not style_sheet.strip():
            logger.warning(f"Stil dosyası boş: {qss_path}")
            return False
            
        app.setStyleSheet(style_sheet)
        logger.info(f"Stil dosyası başarıyla yüklendi: {qss_path}")
        return True
        
    except Exception as e:
        error_msg = f"Stil dosyası yüklenirken hata oluştu: {e}"
        logger.error(error_msg, exc_info=True)
        
        # Hata mesajını göster (splash ekranı varsa)
        if 'splash' in globals() and splash:
            splash.showMessage(
                f"Stil hatası: {str(e)}\nVarsayılan stiller kullanılacak.",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                Qt.GlobalColor.white
            )
        
        return False

def set_application_icon(app, icon_path):
    """Set the application icon."""
    try:
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            app.setWindowIcon(app_icon)
            logger.info(f"Application icon set from: {icon_path}")
            return True
    except Exception as e:
        logger.error(f"Error setting application icon: {e}")
    return False

def debug_info(message):
    """Print debug information with timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DEBUG][{timestamp}] {message}")

class SplashScreen(QSplashScreen):
    """Özelleştirilmiş bir splash screen sınıfı.
    
    Bu sınıf, uygulama yüklenirken gösterilen başlangıç ekranını yönetir.
    İlerleme çubuğu, durum mesajı ve başlık içerir.
    """
    
    def __init__(self, pixmap=None, flags=Qt.WindowType.WindowStaysOnTopHint):
        """SplashScreen başlatıcı metodu.
        
        Args:
            pixmap (QPixmap, optional): Arka plan resmi. Varsayılan: None
            flags (Qt.WindowType, optional): Pencere bayrakları. Varsayılan: WindowStaysOnTopHint
        """
        try:
            # Eğer pixmap verilmediyse, varsayılan bir boyut oluştur
            if pixmap is None:
                pixmap = QPixmap(1000, 700)
                pixmap.fill(Qt.GlobalColor.darkBlue)
            
            # Pixmap'i QSplashScreen'e ver
            super().__init__(pixmap, flags)
            
            # Pencere boyutunu ayarla
            self.setFixedSize(1000, 700)
            
            # Pencereyi ekranın ortasına yerleştir
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
            
            # İlerleme değeri ve mesajı için değişkenler
            self._progress = 0
            self._message = ""
            
            # Stil ayarları
            self.setStyleSheet("""
                QSplashScreen {
                    background-color: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1a237e, stop:1 #0d47a1
                    );
                    color: white;
                    font-size: 14px;
                }
                
                QLabel#titleLabel {
                    font-size: 42px;
                    font-weight: bold;
                    color: white;
                    margin: 20px 0 30px 0;
                    /* text-shadow özelliği Qt'de desteklenmiyor */
                    letter-spacing: 3px;
                    padding: 10px 20px;
                    background-color: rgba(0, 0, 0, 0.2);
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                
                QLabel#statusLabel {
                    font-size: 14px;
                    color: white;
                    margin: 10px 0;
                    padding: 5px 10px;
                    background-color: rgba(0, 0, 0, 0.2);
                    border-radius: 5px;
                }
                
                QProgressBar {
                    border: 1px solid #448aff;
                    border-radius: 5px;
                    text-align: center;
                    height: 20px;
                    margin: 10px 0;
                }
                
                QProgressBar::chunk {
                    background-color: #448aff;
                    width: 10px;
                }
            """)
            
            # Başlık etiketi
            self.title_label = QLabel("MIDI COMPOSER", self)
            self.title_label.setObjectName("titleLabel")
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Durum etiketi
            self.status_label = QLabel("Başlatılıyor...", self)
            self.status_label.setObjectName("statusLabel")
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # İlerleme çubuğu
            self.progress_bar = QProgressBar(self)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            
            # Widget'ları konumlandır
            self.title_label.setGeometry(50, 100, 900, 100)
            self.status_label.setGeometry(100, 350, 800, 40)
            self.progress_bar.setGeometry(100, 400, 800, 20)
            
            # Versiyon bilgisi
            self.version_label = QLabel("v1.0.0", self)
            self.version_label.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 12px;")
            self.version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            self.version_label.setGeometry(700, 650, 250, 20)
            
            # Tüm widget'ları göster
            self.title_label.show()
            self.status_label.show()
            self.progress_bar.show()
            self.version_label.show()
            
        except Exception as e:
            logger.error(f"SplashScreen oluşturulurken hata: {e}", exc_info=True)
            raise
    
    def set_progress(self, value):
        """İlerleme çubuğunun değerini günceller."""
        self._progress = value
        self.progress_bar.setValue(value)
        self.repaint()
    
    def set_status(self, message):
        """Durum mesajını günceller."""
        self._message = message
        self.status_label.setText(message)
        self.repaint()
    
    def add_title(self):
        """Başlık etiketini oluşturur ve ekler."""
        try:
            title_label = QLabel("MIDI COMPOSER")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("""
                font-size: 42px;
                font-weight: bold;
                color: white;
                margin: 20px 0 30px 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
                letter-spacing: 3px;
                padding: 10px 20px;
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            """)
            self.layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignTop)
            
        except Exception as e:
            logger.error(f"Başlık eklenirken hata: {e}", exc_info=True)
            raise
    
    def add_status_label(self):
        """Durum etiketini oluşturur ve ekler."""
        try:
            self.status_label = QLabel("Uygulama başlatılıyor...")
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.status_label.setStyleSheet("""
                color: white;
                font-size: 16px;
                font-weight: bold;
                margin: 10px 0 20px 0;
                padding: 8px 15px;
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 5px;
                min-width: 300px;
            """)
            self.layout.addWidget(self.status_label)
            
        except Exception as e:
            logger.error(f"Durum etiketi eklenirken hata: {e}", exc_info=True)
            raise
    
    def add_progress_bar(self):
        """İlerleme çubuğunu oluşturur ve ekler."""
        try:
            # İlerleme çubuğu için container
            progress_container = QWidget()
            progress_container.setStyleSheet("background-color: transparent;")
            progress_layout = QVBoxLayout(progress_container)
            progress_layout.setContentsMargins(0, 0, 0, 0)
            progress_layout.setSpacing(5)
            
            # Yüzde etiketi
            self.percent_label = QLabel("0%")
            self.percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.percent_label.setStyleSheet("""
                color: white;
                font-size: 14px;
                font-weight: bold;
                margin: 0;
            """)
            
            # İlerleme çubuğu
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFixedHeight(20)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #448aff;
                    border-radius: 10px;
                    background-color: rgba(0, 0, 0, 0.3);
                    text-align: center;
                    color: white;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    border-radius: 8px;
                    background: qlineargradient(
                        spread:pad, 
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #00bcd4,
                        stop:0.5 #3f51b5,
                        stop:1 #9c27b0
                    );
                    background-size: 200% 100%;
                    animation: progressAnimation 3s ease infinite;
                }
                
                @keyframes progressAnimation {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
            """)
            
            # Widget'ları layout'a ekle
            progress_layout.addWidget(self.percent_label, alignment=Qt.AlignmentFlag.AlignCenter)
            progress_layout.addWidget(self.progress_bar)
            
            # Ana layout'a ekle
            self.layout.addWidget(progress_container)
            
        except Exception as e:
            logger.error(f"İlerleme çubuğu eklenirken hata: {e}", exc_info=True)
            raise
    
    def add_version_info(self):
        """Versiyon bilgisini ekler."""
        try:
            version_label = QLabel("v1.0.0")
            version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            version_label.setStyleSheet("""
                color: rgba(255, 255, 255, 0.7);
                font-size: 12px;
                margin-top: 30px;
                font-style: italic;
            """)
            self.layout.addWidget(version_label, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            
        except Exception as e:
            logger.error(f"Versiyon bilgisi eklenirken hata: {e}", exc_info=True)
            # Bu hata kritik değil, devam et
    
    def set_progress(self, value):
        """İlerleme çubuğunun değerini günceller.
        
        Args:
            value (int): 0-100 arasında bir değer
        """
        try:
            if self.progress_bar:
                # Değeri sınırla
                progress = max(0, min(100, int(value)))
                self.progress_bar.setValue(progress)
                
                # Yüzde etiketini güncelle
                if hasattr(self, 'percent_label'):
                    self.percent_label.setText(f"%{progress}")
                    
        except Exception as e:
            logger.error(f"İlerleme değeri güncellenirken hata: {e}", exc_info=True)
    
    def set_status(self, message):
        """Durum mesajını günceller.
        
        Args:
            message (str): Gösterilecek durum mesajı
        """
        try:
            if self.status_label and message:
                self.status_label.setText(str(message))
                # UI güncellemelerinin hemen yansıması için
                QApplication.processEvents()
                
        except Exception as e:
            logger.error(f"Durum mesajı güncellenirken hata: {e}", exc_info=True)

def show_splash_screen(app, image_path=None):
    """Uygulama yüklenirken gösterilecek splash ekranını oluşturur ve döndürür.
    
    Args:
        app: QApplication örneği
        image_path (str, optional): Arka plan resmi yolu. Varsayılan: None
        
    Returns:
        SimpleSplashScreen: Oluşturulan splash ekranı örneği
    """
    debug_info("Splash ekranı oluşturuluyor...")
    debug_info(f"Resim yolu: {image_path}")
    
    try:
        # Eğer resim yolu verilmişse ve dosya varsa, resmi yükle
        if image_path and os.path.exists(image_path):
            debug_info(f"Resim dosyası bulundu: {image_path}")
            try:
                # Resmi yükle
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    # Resmi splash ekranı boyutuna ölçekle
                    pixmap = pixmap.scaled(
                        800, 500,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    debug_info(f"Resim boyutlandırıldı: {pixmap.width()}x{pixmap.height()}")
                    
                    # Resimli splash ekranı oluştur
                    splash = SimpleSplashScreen(app)
                    splash.setPixmap(pixmap)
                    splash.set_progress(0, "Uygulama başlatılıyor...")
                    return splash
                
            except Exception as img_error:
                logger.warning(f"Resim yüklenirken hata oluştu: {img_error}")
        
        # Resim yüklenemediyse veya belirtilmediyse varsayılan splash ekranını kullan
        debug_info("Varsayılan splash ekranı kullanılıyor...")
        splash = SimpleSplashScreen(app)
        splash.set_progress(0, "Uygulama başlatılıyor...")
        return splash
        
    except Exception as e:
        logger.error(f"Splash ekranı oluşturulurken hata: {e}", exc_info=True)
        
        # Hata durumunda basit bir hata mesajı göster
        try:
            error_splash = QSplashScreen()
            error_splash.showMessage(
                "Splash ekranı yüklenirken bir hata oluştu.\nUygulama başlatılıyor...",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                Qt.GlobalColor.white
            )
            error_splash.show()
            app.processEvents()
            return error_splash
        except Exception as inner_e:
            logger.critical(f"Hata ekranı oluşturulamadı: {inner_e}", exc_info=True)
            return None

def update_splash_status(splash, value, message):
    """Splash ekranının ilerleme durumunu ve durum mesajını günceller.
    
    Args:
        splash: Güncellenecek SplashScreen örneği
        value (int): 0-100 arasında ilerleme değeri
        message (str): Gösterilecek durum mesajı
    """
    if not splash:
        debug_info("Hata: Geçersiz splash ekranı örneği")
        return
        
    try:
        # İlerleme değerini güncelle
        if hasattr(splash, 'set_progress') and callable(getattr(splash, 'set_progress')):
            splash.set_progress(value)
        
        # Durum mesajını güncelle
        if hasattr(splash, 'set_status') and callable(getattr(splash, 'set_status')):
            splash.set_status(message)
        
        # Pencere başlığını güncelle (varsa)
        if hasattr(splash, 'setWindowTitle') and message:
            splash.setWindowTitle(f"MIDI Composer - {message}")
        
        # UI güncellemelerini işle
        QApplication.processEvents()
        
    except Exception as e:
        error_msg = f"Splash ekranı güncellenirken hata: {e}"
        logger.error(error_msg, exc_info=True)
        debug_info(error_msg)

def main():
    """
    Uygulamanın ana giriş noktası.
    
    Uygulama başlatma sürecini yönetir, gerekli bileşenleri yükler
    ve ana pencereyi gösterir.
    
    Not: Bu fonksiyon artık doğrudan çalıştırılmıyor, run_application() kullanılıyor.
    Bu fonksiyon sadece geriye dönük uyumluluk için tutuluyor.
    """
    logger.warning("main() fonksiyonu artık kullanılmıyor. Lütfen run_application() kullanın.")
    return run_application()

def run_application():
    """Uygulamayı çalıştıran ana fonksiyon"""
    # Global değişkenleri tanımla
    global app
    splash = None
    lock_handle = None
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Uygulama bilgilerini ayarla
        app.setApplicationName("MIDI Composer")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("MIDI Composer Team")
        
        # Splash ekranını oluştur
        splash = SimpleSplashScreen(app)
        splash.set_progress(10, "Sistem kontrolleri yapılıyor...")
        
        # Sistem ve ortam bilgilerini logla
        debug_info("=" * 80)
        debug_info(f"MIDI Composer Başlatılıyor - {datetime.datetime.now()}")
        debug_info("=" * 80)
        debug_info(f"Sistem: {platform.system()} {platform.release()} {platform.version()}")
        debug_info(f"Python: {platform.python_version()}")
        debug_info(f"Çalışma Dizini: {os.getcwd()}")
        debug_info(f"Python Yolu: {sys.executable}")
        debug_info(f"Proje Kök Dizini: {project_root}")
        
        # Settings yükleme
        splash.set_progress(20, "Ayarlar yükleniyor...")
        try:
            # Settings sınıfını doğrudan kullan
            settings = Settings()
            debug_info("Varsayılan ayarlar kullanılıyor.")
        except Exception as e:
            error_msg = f"Ayarlar yüklenirken hata oluştu: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Başlatma Hatası")
            msg_box.setText("Ayarlar Yüklenemedi")
            msg_box.setInformativeText("Varsayılan ayarlar kullanılacak.")
            msg_box.setDetailedText(str(traceback.format_exc()))
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            
            # Varsayılan ayarlarla devam et
            settings = Settings()
        
        splash.set_progress(30, "Uygulama kontrolleri yapılıyor...")
        try:
            is_running, lock_file = check_single_instance()
            if is_running:
                logger.info("Uygulama zaten çalışıyor, mevcut pencere öne getiriliyor...")
                if activate_existing_window():
                    return 0
                else:
                    logger.warning("Mevcut pencere bulunamadı, yeni bir örnek başlatılıyor...")

            # Uygulama kapatılırken kilit dosyasını temizle
            import atexit
            atexit.register(cleanup_lock_file, lock_handle)
            
            # Stil dosyasını yükle
            splash.set_progress(60, "Stiller yükleniyor...")
            try:
                if load_stylesheet(app):
                    debug_info("Stil dosyası başarıyla yüklendi.")
                else:
                    debug_info("Uyarı: Stil dosyası yüklenemedi. Varsayılan stiller kullanılacak.")
            except Exception as e:
                error_msg = f"Stil dosyası yüklenirken hata oluştu: {e}"
                debug_info(error_msg)
                logger.warning(error_msg, exc_info=True)
            
            # Uygulama ikonunu ayarla
            splash.set_progress(70, "Uygulama ikonu yükleniyor...")
            try:
                icon_path = os.path.join(project_root, "resources", "images", "app_icon.ico")
                set_application_icon(app, icon_path)
                debug_info("Uygulama ikonu başarıyla yüklendi.")
            except Exception as e:
                error_msg = f"Uygulama ikonu yüklenirken hata oluştu: {e}"
                debug_info(error_msg)
                logger.warning(error_msg, exc_info=True)
            
            # Ana pencereyi yükle
            splash.set_progress(80, "Ana pencere yükleniyor...")
            try:
                from src.gui.main_window import MainWindow
                
                # Ana pencereyi oluştur ve göster
                splash.set_progress(90, "Pencere oluşturuluyor...")
                main_window = MainWindow()
                main_window.show()
            except ImportError as e:
                error_msg = f"Ana pencere modülü yüklenemedi: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(f"Uygulama başlatılamadı: {error_msg}")
            
            # Splash ekranını kapat
            splash.finish(main_window)
            
            # Uygulama çalıştır
            return app.exec()
                
        except Exception as e:
            error_msg = f"Uygulama kontrolü sırasında hata oluştu: {str(e)}"
            logger.error(error_msg, exc_info=True)
            splash.set_progress(100, "Başlatma hatası oluştu")
            
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Başlatma Hatası")
            msg_box.setText("Uygulama Kontrolü Başarısız")
            msg_box.setInformativeText("Uygulama kontrolü sırasında bir hata oluştu.")
            msg_box.setDetailedText(f"{str(e)}\n\n{traceback.format_exc()}")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            
            return 1  # Hata kodu ile çık
        
    except Exception as e:
        # Herhangi bir hata durumunda
        error_msg = f"Uygulama başlatılırken beklenmeyen bir hata oluştu: {e}"
        debug_info(error_msg)
        logger.critical(error_msg, exc_info=True)
        
        # Splash ekranını kapat
        if 'splash' in locals() and splash:
            splash.close()
        
        # Hata mesajını göster
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Kritik Hata")
        msg_box.setText("Uygulama Başlatılamadı")
        msg_box.setInformativeText("Uygulama başlatılırken beklenmeyen bir hata oluştu.\n\nLütfen uygulamayı yeniden başlatmayı deneyin.")
        msg_box.setDetailedText(f"{str(e)}\n\n{traceback.format_exc()}")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Mesaj kutusunu göster ve uygulamayı sonlandır
        msg_box.exec()
        return 1  # Hata kodu ile çık
            
    finally:
        # Kaynakları temizle
        if lock_handle:
            cleanup_lock_file(lock_handle)
        
        # Splash ekranını kapat
        if splash and hasattr(splash, 'isVisible') and splash.isVisible():
            splash.close()



if __name__ == "__main__":
    # run_application fonksiyonunu tanımla
    def run_application():
        """Uygulamayı çalıştıran ana fonksiyon"""
        # Global değişkenleri tanımla
        global app
        splash = None
        lock_handle = None
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            # Uygulama bilgilerini ayarla
            app.setApplicationName("MIDI Composer")
            app.setApplicationVersion("1.0.0")
            app.setOrganizationName("MIDI Composer Team")
            
            # Splash ekranını oluştur
            splash = SimpleSplashScreen(app)
            splash.set_progress(10, "Sistem kontrolleri yapılıyor...")
            
            # Sistem ve ortam bilgilerini logla
            debug_info("=" * 80)
            debug_info(f"MIDI Composer Başlatılıyor - {datetime.datetime.now()}")
            debug_info("=" * 80)
            debug_info(f"Sistem: {platform.system()} {platform.release()} {platform.version()}")
            debug_info(f"Python: {platform.python_version()}")
            debug_info(f"Çalışma Dizini: {os.getcwd()}")
            debug_info(f"Python Yolu: {sys.executable}")
            debug_info(f"Proje Kök Dizini: {project_root}")
            
            # Settings yükleme
            splash.set_progress(20, "Ayarlar yükleniyor...")
            try:
                # Settings sınıfını doğrudan kullan
                settings = Settings()
                debug_info("Varsayılan ayarlar kullanılıyor.")
            except Exception as e:
                error_msg = f"Ayarlar yüklenirken hata oluştu: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle("Başlatma Hatası")
                msg_box.setText("Ayarlar Yüklenemedi")
                msg_box.setInformativeText("Varsayılan ayarlar kullanılacak.")
                msg_box.setDetailedText(str(traceback.format_exc()))
                msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg_box.exec()
                
                # Varsayılan ayarlarla devam et
                settings = Settings()
            
            # Tek örnek kontrolü
            splash.set_progress(30, "Uygulama kontrolleri yapılıyor...")
            try:
                is_running, lock_handle = check_single_instance()
                if is_running:
                    logger.info("Uygulama zaten çalışıyor, mevcut pencere öne getiriliyor...")
                    if activate_existing_window():
                        return 0
                    else:
                        logger.warning("Mevcut pencere bulunamadı, yeni bir örnek başlatılıyor...")

                # Uygulama kapatılırken kilit dosyasını temizle
                import atexit
                atexit.register(cleanup_lock_file, lock_handle)
                
                # Stil dosyasını yükle
                splash.set_progress(60, "Stiller yükleniyor...")
                try:
                    if load_stylesheet(app):
                        debug_info("Stil dosyası başarıyla yüklendi.")
                    else:
                        debug_info("Uyarı: Stil dosyası yüklenemedi. Varsayılan stiller kullanılacak.")
                except Exception as e:
                    error_msg = f"Stil dosyası yüklenirken hata oluştu: {e}"
                    debug_info(error_msg)
                    logger.warning(error_msg, exc_info=True)
                
                # Uygulama ikonunu ayarla
                splash.set_progress(70, "Uygulama ikonu yükleniyor...")
                try:
                    icon_path = os.path.join(project_root, "resources", "images", "app_icon.ico")
                    set_application_icon(app, icon_path)
                    debug_info("Uygulama ikonu başarıyla yüklendi.")
                except Exception as e:
                    error_msg = f"Uygulama ikonu yüklenirken hata oluştu: {e}"
                    debug_info(error_msg)
                    logger.warning(error_msg, exc_info=True)
                
                # Ana pencereyi yükle
                splash.set_progress(80, "Ana pencere yükleniyor...")
                try:
                    from src.gui.main_window import MainWindow
                    
                    # Ana pencereyi oluştur ve göster
                    splash.set_progress(90, "Pencere oluşturuluyor...")
                    main_window = MainWindow()
                    main_window.show()
                except ImportError as e:
                    error_msg = f"Ana pencere modülü yüklenemedi: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(f"Uygulama başlatılamadı: {error_msg}")
                
                # Splash ekranını kapat
                splash.finish(main_window)
                
                # Uygulama çalıştır
                return app.exec()
                    
            except Exception as e:
                error_msg = f"Uygulama kontrolü sırasında hata oluştu: {str(e)}"
                logger.error(error_msg, exc_info=True)
                if splash:
                    splash.set_progress(100, "Başlatma hatası oluştu")
                
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle("Başlatma Hatası")
                msg_box.setText("Uygulama Kontrolü Başarısız")
                msg_box.setInformativeText("Uygulama kontrolü sırasında bir hata oluştu.")
                msg_box.setDetailedText(f"{str(e)}\n\n{traceback.format_exc()}")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg_box.exec()
                
                return 1  # Hata kodu ile çık
            
        except Exception as e:
            # Herhangi bir hata durumunda
            error_msg = f"Uygulama başlatılırken beklenmeyen bir hata oluştu: {e}"
            debug_info(error_msg)
            logger.critical(error_msg, exc_info=True)
            
            # Splash ekranını kapat
            if 'splash' in locals() and splash and hasattr(splash, 'close'):
                splash.close()
            
            # Hata mesajını göster
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Kritik Hata")
            msg_box.setText("Uygulama Başlatılamadı")
            msg_box.setInformativeText("Uygulama başlatılırken beklenmeyen bir hata oluştu.\n\nLütfen uygulamayı yeniden başlatmayı deneyin.")
            msg_box.setDetailedText(f"{str(e)}\n\n{traceback.format_exc()}")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            
            # Mesaj kutusunu göster ve uygulamayı sonlandır
            msg_box.exec()
            return 1  # Hata kodu ile çık
                
        finally:
            # Kaynakları temizle
            if 'lock_handle' in locals() and lock_handle:
                cleanup_lock_file(lock_handle)
            
            # Splash ekranını kapat
            if 'splash' in locals() and splash and hasattr(splash, 'isVisible') and splash.isVisible():
                splash.close()
    
    # Uygulamayı başlat
    sys.exit(run_application())
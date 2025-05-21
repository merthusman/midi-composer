import logging
import os
from logging.handlers import RotatingFileHandler

class LoggerManager:
    """Logger sınıfı, uygulama genelinde loglama işlemlerini yönetir."""
    
    def __init__(self, log_file='midi_composer.log', log_level=logging.INFO):
        """Logger'ı başlatır.
        
        Args:
            log_file (str): Log dosyasının adı
            log_level: Log seviyesi (default: logging.INFO)
        """
        # Log klasörünü oluştur
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, log_file)
        self.logger = logging.getLogger('MidiComposer')
        self.logger.setLevel(log_level)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Dosyaya loglama
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # Konsola loglama
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Handler'ları ekle
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_info(self, message):
        """Bilgi mesajı loglar."""
        self.logger.info(message)
    
    def log_warning(self, message):
        """Uyarı mesajı loglar."""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Hata mesajı loglar."""
        self.logger.error(message)
    
    def log_debug(self, message):
        """Debug mesajı loglar."""
        self.logger.debug(message)
    
    def log_critical(self, message):
        """Kritik hata mesajı loglar."""
        self.logger.critical(message)

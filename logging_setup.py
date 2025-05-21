"""
Loglama yapılandırması.
Bu modül, uygulama genelinde tutarlı loglama sağlamak için kullanılır.
"""

import os
import sys
import logging
from src.utils.paths import get_logs_dir

def configure_logging(log_file_name="app.log", console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Uygulama genelinde loglama yapılandırması.
    Bu fonksiyon, main.py'de bir kez çağrılmalıdır.
    
    Args:
        log_file_name (str): Log dosyasının adı
        console_level (int): Konsol log seviyesi
        file_level (int): Dosya log seviyesi
        
    Returns:
        logging.Logger: Kök logger
    """
    # Log dizinini oluştur
    logs_dir = get_logs_dir()
    
    # Log dosyası yolu
    log_file_path = os.path.join(logs_dir, log_file_name)
    
    # Kök logger'ı yapılandır
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # En düşük seviye
    
    # Formatı tanımla
    log_format = "%(asctime)s %(levelname)s %(name)s %(funcName)s: %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Dosya handler'ı
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    # Konsol handler'ı
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # Handler'ları ekle
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_logger(name):
    """
    Belirtilen isimle bir logger döndürür.
    Bu fonksiyon, her modülde logger almak için kullanılmalıdır.
    
    Args:
        name (str): Logger adı
        
    Returns:
        logging.Logger: Logger
    """
    return logging.getLogger(name)

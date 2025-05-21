"""
Merkezi hata işleme mekanizmaları.
Bu modül, uygulamada tutarlı hata işleme sağlamak için kullanılır.
"""

import logging
from PyQt6.QtWidgets import QMessageBox

# Logger oluştur
logger = logging.getLogger(__name__)

def handle_error(error, context="", critical=False, parent=None):
    """
    Merkezi hata işleme fonksiyonu.
    
    Args:
        error (Exception): Hata nesnesi
        context (str): Hatanın oluştuğu bağlam
        critical (bool): Kritik hata mı?
        parent (QWidget): Mesaj kutusunun üst widget'ı
        
    Returns:
        None
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    
    if critical:
        logger.critical(error_msg, exc_info=True)
        # Kritik hata mesaj kutusu göster
        QMessageBox.critical(parent, "Kritik Hata", error_msg)
    else:
        logger.error(error_msg, exc_info=True)
        # Normal hata mesaj kutusu göster
        QMessageBox.warning(parent, "Hata", error_msg)

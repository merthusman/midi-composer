"""
Merkezi olay işleme mekanizmaları.
Bu modül, uygulamada tutarlı olay işleme sağlamak için kullanılır.
"""

import logging
from PyQt6.QtWidgets import QApplication

# Logger oluştur
logger = logging.getLogger(__name__)

class MIDIEventHandler:
    """MIDI olaylarını işlemek için merkezi sınıf"""
    
    def __init__(self, main_window):
        """
        Args:
            main_window: Ana pencere referansı
        """
        self.main_window = main_window
    
    def handle_midi_event(self, event_type, data=None):
        """
        Farklı MIDI olaylarını işlemek için merkezi fonksiyon
        
        Args:
            event_type (str): Olay tipi
            data (dict, optional): Olay verileri
            
        Returns:
            bool: İşlem başarılı mı?
        """
        if data is None:
            data = {}
            
        try:
            if event_type == "analyze":
                # Analiz işlemleri
                return self._handle_analyze_event(data)
            elif event_type == "generate":
                # Üretim işlemleri
                return self._handle_generate_event(data)
            elif event_type == "memory_add":
                # Hafızaya ekleme işlemleri
                return self._handle_memory_add_event(data)
            elif event_type == "memory_search":
                # Hafızada arama işlemleri
                return self._handle_memory_search_event(data)
            else:
                logger.warning(f"Bilinmeyen olay tipi: {event_type}")
                return False
        except Exception as e:
            logger.error(f"Olay işlenirken hata oluştu ({event_type}): {e}", exc_info=True)
            return False
    
    def _handle_analyze_event(self, data):
        """Analiz olayını işler"""
        midi_path = data.get("midi_path")
        if not midi_path:
            logger.error("Analiz için MIDI dosya yolu belirtilmedi")
            return False
            
        # İşlem başladığını göster
        self.main_window.statusBar().showMessage("MIDI dosyası analiz ediliyor...")
        QApplication.processEvents()
        
        # Ana penceredeki analiz metodunu çağır
        if hasattr(self.main_window, "start_analysis_worker"):
            self.main_window.start_analysis_worker(midi_path)
            return True
        else:
            logger.error("Ana pencerede start_analysis_worker metodu bulunamadı")
            return False
    
    def _handle_generate_event(self, data):
        """Üretim olayını işler"""
        # Gerekli parametreleri al
        seed_sequence = data.get("seed_sequence")
        bar_count = data.get("bar_count", 4)
        tempo = data.get("tempo", 120)
        temperature = data.get("temperature", 1.0)
        style = data.get("style", "Otomatik")
        
        # İşlem başladığını göster
        self.main_window.statusBar().showMessage("MIDI üretiliyor...")
        QApplication.processEvents()
        
        # Ana penceredeki üretim metodunu çağır
        if hasattr(self.main_window, "start_generation_worker"):
            self.main_window.start_generation_worker(
                seed_sequence=seed_sequence,
                bar_count=bar_count,
                tempo=tempo,
                temperature=temperature,
                style=style
            )
            return True
        else:
            logger.error("Ana pencerede start_generation_worker metodu bulunamadı")
            return False
    
    def _handle_memory_add_event(self, data):
        """Hafızaya ekleme olayını işler"""
        midi_path = data.get("midi_path")
        analysis = data.get("analysis")
        
        if not midi_path:
            logger.error("Hafızaya eklemek için MIDI dosya yolu belirtilmedi")
            return False
            
        # Ana penceredeki hafıza ekleme metodunu çağır
        if hasattr(self.main_window, "add_to_memory_worker"):
            self.main_window.add_to_memory_worker(midi_path, analysis)
            return True
        else:
            logger.error("Ana pencerede add_to_memory_worker metodu bulunamadı")
            return False
    
    def _handle_memory_search_event(self, data):
        """Hafızada arama olayını işler"""
        query = data.get("query")
        
        if not query:
            logger.error("Hafızada arama için sorgu belirtilmedi")
            return False
            
        # Ana penceredeki hafıza arama metodunu çağır
        if hasattr(self.main_window, "search_memory_worker"):
            self.main_window.search_memory_worker(query)
            return True
        else:
            logger.error("Ana pencerede search_memory_worker metodu bulunamadı")
            return False

"""
Proje yolları için yardımcı fonksiyonlar.
Bu modül, proje içindeki çeşitli dizinlere erişim sağlar.
"""

import os
import logging

# Logger oluştur
logger = logging.getLogger(__name__)

def get_project_root():
    """
    Proje kök dizinini döndürür.
    Bu fonksiyon, çağrıldığı dosyanın konumuna bakılmaksızın
    proje kök dizinini doğru şekilde belirler.
    """
    # __file__ is src/utils/paths.py
    # os.path.dirname(__file__) is src/utils
    # os.path.dirname(os.path.dirname(__file__)) is src
    # os.path.dirname(os.path.dirname(os.path.dirname(__file__))) is project_root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_logs_dir():
    """Log dizinini döndürür"""
    logs_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def get_config_dir():
    """Yapılandırma dizinini döndürür"""
    config_dir = os.path.join(get_project_root(), "config")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir

def get_resources_dir():
    """Kaynaklar dizinini döndürür"""
    resources_dir = os.path.join(get_project_root(), "resources")
    os.makedirs(resources_dir, exist_ok=True)
    return resources_dir

def get_temp_dir():
    """Geçici dosyalar dizinini döndürür"""
    temp_dir = os.path.join(get_project_root(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def get_memory_dir():
    """Hafıza dosyaları dizinini döndürür"""
    memory_dir = os.path.join(get_project_root(), "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return memory_dir

def get_model_dir():
    """Model dosyaları dizinini döndürür"""
    model_dir = os.path.join(get_project_root(), "trained_model")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

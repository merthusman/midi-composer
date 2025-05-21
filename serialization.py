"""
Serileştirme işlemleri için yardımcı sınıflar.
Bu modül, nesneleri JSON formatına dönüştürmek için kullanılır.
"""

import logging
import json
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, TypeVar, Type, Optional, List, Union

# Logger oluştur
logger = logging.getLogger(__name__)

T = TypeVar('T')

def object_to_dict(obj) -> Dict[str, Any]:
    """
    Herhangi bir nesneyi JSON serileştirilebilir bir sözlüğe dönüştürür.
    Karmaşık nesneleri, dataclass'ları ve iç içe yapıları işleyebilir.
    
    Args:
        obj: Dönüştürülecek nesne
        
    Returns:
        Dict[str, Any]: JSON serileştirilebilir sözlük
    """
    if obj is None:
        return None
        
    if isinstance(obj, (str, int, float, bool)):
        return obj
        
    if isinstance(obj, (list, tuple)):
        return [object_to_dict(item) for item in obj]
        
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
        
    if isinstance(obj, set):
        return list(obj)
        
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
        
    if is_dataclass(obj):
        return {k: object_to_dict(v) for k, v in asdict(obj).items()}
        
    if hasattr(obj, '__dict__'):
        return {k: object_to_dict(v) for k, v in vars(obj).items() 
                if not k.startswith('_')}
                
    # Son çare: str dönüşümü
    try:
        return str(obj)
    except:
        return f"<Serileştirilemeyen nesne: {type(obj).__name__}>"

def to_json(obj, indent=4, sort_keys=True, ensure_ascii=False) -> str:
    """
    Herhangi bir nesneyi JSON formatına dönüştürür.
    
    Args:
        obj: Dönüştürülecek nesne
        indent: JSON girintisi
        sort_keys: Anahtarları sırala
        ensure_ascii: ASCII karakterleri kullan
        
    Returns:
        str: JSON formatında string
    """
    try:
        dict_obj = object_to_dict(obj)
        return json.dumps(dict_obj, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    except Exception as e:
        logger.error(f"JSON dönüşümü sırasında hata: {e}")
        return f"{{\"error\": \"JSON dönüşümü başarısız: {str(e)}\"}}"

class SerializableMixin:
    """Serileştirme işlevleri için mixin sınıfı"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Nesneyi sözlüğe dönüştürür"""
        if is_dataclass(self):
            data = asdict(self)
        else:
            data = vars(self)
            
        # Karmaşık tipleri dönüştür
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = list(value)
            elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                data[key] = value.to_dict()
            elif is_dataclass(value):
                data[key] = asdict(value)
            elif hasattr(value, '__dict__'):
                data[key] = object_to_dict(value)
                
        return data
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> Optional[T]:
        """Sözlükten nesne oluşturur"""
        if not data:
            return None
            
        # Sınıfa özgü dönüşüm işlemleri burada yapılabilir
        # Bu temel implementasyon, alt sınıflar tarafından override edilebilir
        try:
            return cls(**data)
        except Exception as e:
            logger.error(f"Error creating {cls.__name__} from dict: {e}")
            return None

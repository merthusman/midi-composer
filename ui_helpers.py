"""
UI bileşenleri oluşturmak için yardımcı fonksiyonlar.
Bu modül, tekrarlanan UI bileşeni oluşturma kodlarını azaltmak için kullanılır.
Duyarlı tasarım için geliştirilmiş fonksiyonlar içerir.
"""

import logging
from PyQt6.QtWidgets import (
    QLabel, QPushButton, QSizePolicy, QSpinBox, QDoubleSpinBox, 
    QComboBox, QLineEdit, QTextEdit, QFormLayout, QGridLayout,
    QHBoxLayout, QVBoxLayout, QFrame, QGroupBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor

# Logger oluştur
logger = logging.getLogger(__name__)

def create_styled_label(text, object_name=None, alignment=None, min_width=None, min_height=None, font=None, size_policy=None):
    """
    Ortak stil ve özelliklere sahip QLabel oluşturur.
    
    Args:
        text (str): Etiket metni
        object_name (str, optional): CSS için objectName
        alignment (Qt.AlignmentFlag, optional): Hizalama
        min_width (int, optional): Minimum genişlik
        min_height (int, optional): Minimum yükseklik
        font (QFont, optional): Yazı tipi
        size_policy (tuple, optional): Boyut politikası (yatay, dikey)
        
    Returns:
        QLabel: Oluşturulan etiket
    """
    try:
        label = QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        if alignment:
            label.setAlignment(alignment)
        if min_width:
            label.setMinimumWidth(min_width)
        if min_height:
            label.setMinimumHeight(min_height)
        if font:
            label.setFont(font)
        if size_policy:
            label.setSizePolicy(size_policy[0], size_policy[1])
        return label
    except Exception as e:
        logger.error(f"Etiket oluşturulurken hata: {e}", exc_info=True)
        return QLabel(text)  # Basit bir etiket döndür

def create_styled_button(text, object_name=None, min_width=None, min_height=None, fixed_width=None, 
                         fixed_height=None, fixed_size=None, font=None, connect_to=None, size_policy=None):
    """
    Ortak stil ve özelliklere sahip QPushButton oluşturur.
    
    Args:
        text (str): Buton metni
        object_name (str, optional): CSS için objectName
        min_width (int, optional): Minimum genişlik
        min_height (int, optional): Minimum yükseklik
        fixed_width (int, optional): Sabit genişlik
        fixed_height (int, optional): Sabit yükseklik
        fixed_size (tuple, optional): Sabit boyut (genişlik, yükseklik)
        font (QFont, optional): Yazı tipi
        connect_to (function, optional): Bağlanacak fonksiyon
        size_policy (tuple, optional): Boyut politikası (yatay, dikey)
        
    Returns:
        QPushButton: Oluşturulan buton
    """
    try:
        button = QPushButton(text)
        if object_name:
            button.setObjectName(object_name)
        if min_width:
            button.setMinimumWidth(min_width)
        if min_height:
            button.setMinimumHeight(min_height)
        if fixed_width:
            button.setFixedWidth(fixed_width)
        if fixed_height:
            button.setFixedHeight(fixed_height)
        if fixed_size:
            button.setFixedSize(fixed_size[0], fixed_size[1])
        if font:
            button.setFont(font)
        if connect_to:
            button.clicked.connect(connect_to)
        if size_policy:
            button.setSizePolicy(size_policy[0], size_policy[1])
        return button
    except Exception as e:
        logger.error(f"Buton oluşturulurken hata: {e}", exc_info=True)
        return QPushButton(text)  # Basit bir buton döndür

def create_responsive_grid_layout(parent=None, margins=(10, 10, 10, 10), spacing=10):
    """
    Duyarlı ızgara düzeni oluşturur.
    
    Args:
        parent: Ebeveyn widget
        margins (tuple): Kenar boşlukları (sol, üst, sağ, alt)
        spacing (int): Öğeler arası boşluk
        
    Returns:
        QGridLayout: Oluşturulan ızgara düzeni
    """
    try:
        layout = QGridLayout(parent)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        layout.setColumnStretch(0, 0)  # Etiket sütunu
        layout.setColumnStretch(1, 0)  # Giriş sütunu
        layout.setColumnStretch(2, 1)  # Genişleyen sütun
        return layout
    except Exception as e:
        logger.error(f"Duyarlı ızgara düzeni oluşturulurken hata: {e}", exc_info=True)
        return QGridLayout(parent)

def create_form_layout(parent=None, margins=(10, 10, 10, 10), spacing=10, label_alignment=Qt.AlignmentFlag.AlignRight):
    """
    Form düzeni oluşturur.
    
    Args:
        parent: Ebeveyn widget
        margins (tuple): Kenar boşlukları (sol, üst, sağ, alt)
        spacing (int): Öğeler arası boşluk
        label_alignment: Etiket hizalaması
        
    Returns:
        QFormLayout: Oluşturulan form düzeni
    """
    try:
        layout = QFormLayout(parent)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        layout.setLabelAlignment(label_alignment)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        return layout
    except Exception as e:
        logger.error(f"Form düzeni oluşturulurken hata: {e}", exc_info=True)
        return QFormLayout(parent)

def create_styled_spin_box(min_value, max_value, default_value=None, step=1, object_name=None, 
                          min_width=150, min_height=30, size_policy=None):
    """
    Stillendirilmiş QSpinBox oluşturur.
    
    Args:
        min_value (int): Minimum değer
        max_value (int): Maksimum değer
        default_value (int, optional): Varsayılan değer
        step (int, optional): Adım değeri
        object_name (str, optional): CSS için objectName
        min_width (int, optional): Minimum genişlik
        min_height (int, optional): Minimum yükseklik
        size_policy (tuple, optional): Boyut politikası (yatay, dikey)
        
    Returns:
        QSpinBox: Oluşturulan spin box
    """
    try:
        spin_box = QSpinBox()
        spin_box.setRange(min_value, max_value)
        if default_value is not None:
            spin_box.setValue(default_value)
        spin_box.setSingleStep(step)
        
        if object_name:
            spin_box.setObjectName(object_name)
        if min_width:
            spin_box.setMinimumWidth(min_width)
        if min_height:
            spin_box.setMinimumHeight(min_height)
        if size_policy:
            spin_box.setSizePolicy(size_policy[0], size_policy[1])
        else:
            spin_box.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            
        return spin_box
    except Exception as e:
        logger.error(f"Spin box oluşturulurken hata: {e}", exc_info=True)
        return QSpinBox()

def create_styled_double_spin_box(min_value, max_value, default_value=None, step=0.1, decimals=1, 
                                 object_name=None, min_width=150, min_height=30, size_policy=None):
    """
    Stillendirilmiş QDoubleSpinBox oluşturur.
    
    Args:
        min_value (float): Minimum değer
        max_value (float): Maksimum değer
        default_value (float, optional): Varsayılan değer
        step (float, optional): Adım değeri
        decimals (int, optional): Ondalık basamak sayısı
        object_name (str, optional): CSS için objectName
        min_width (int, optional): Minimum genişlik
        min_height (int, optional): Minimum yükseklik
        size_policy (tuple, optional): Boyut politikası (yatay, dikey)
        
    Returns:
        QDoubleSpinBox: Oluşturulan double spin box
    """
    try:
        spin_box = QDoubleSpinBox()
        spin_box.setRange(min_value, max_value)
        if default_value is not None:
            spin_box.setValue(default_value)
        spin_box.setSingleStep(step)
        spin_box.setDecimals(decimals)
        
        if object_name:
            spin_box.setObjectName(object_name)
        if min_width:
            spin_box.setMinimumWidth(min_width)
        if min_height:
            spin_box.setMinimumHeight(min_height)
        if size_policy:
            spin_box.setSizePolicy(size_policy[0], size_policy[1])
        else:
            spin_box.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            
        return spin_box
    except Exception as e:
        logger.error(f"Double spin box oluşturulurken hata: {e}", exc_info=True)
        return QDoubleSpinBox()

def create_styled_combo_box(items=None, object_name=None, min_width=150, min_height=30, size_policy=None):
    """
    Stillendirilmiş QComboBox oluşturur.
    
    Args:
        items (list, optional): Öğe listesi
        object_name (str, optional): CSS için objectName
        min_width (int, optional): Minimum genişlik
        min_height (int, optional): Minimum yükseklik
        size_policy (tuple, optional): Boyut politikası (yatay, dikey)
        
    Returns:
        QComboBox: Oluşturulan combo box
    """
    try:
        combo_box = QComboBox()
        if items:
            combo_box.addItems(items)
        
        if object_name:
            combo_box.setObjectName(object_name)
        if min_width:
            combo_box.setMinimumWidth(min_width)
        if min_height:
            combo_box.setMinimumHeight(min_height)
        if size_policy:
            combo_box.setSizePolicy(size_policy[0], size_policy[1])
        else:
            combo_box.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            
        return combo_box
    except Exception as e:
        logger.error(f"Combo box oluşturulurken hata: {e}", exc_info=True)
        return QComboBox()

def create_styled_group_box(title, object_name=None, min_width=None, min_height=None, size_policy=None):
    """
    Stillendirilmiş QGroupBox oluşturur.
    
    Args:
        title (str): Başlık
        object_name (str, optional): CSS için objectName
        min_width (int, optional): Minimum genişlik
        min_height (int, optional): Minimum yükseklik
        size_policy (tuple, optional): Boyut politikası (yatay, dikey)
        
    Returns:
        QGroupBox: Oluşturulan grup kutusu
    """
    try:
        group_box = QGroupBox(title)
        
        if object_name:
            group_box.setObjectName(object_name)
        if min_width:
            group_box.setMinimumWidth(min_width)
        if min_height:
            group_box.setMinimumHeight(min_height)
        if size_policy:
            group_box.setSizePolicy(size_policy[0], size_policy[1])
        else:
            group_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            
        return group_box
    except Exception as e:
        logger.error(f"Grup kutusu oluşturulurken hata: {e}", exc_info=True)
        return QGroupBox(title)

class UIStyles:
    """UI bileşenleri için merkezi stil tanımlamaları"""
    
    @staticmethod
    def get_label_style():
        return """
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 11pt;
                background-color: rgba(30, 30, 30, 0.7);
                border-radius: 5px;
                padding: 5px;
                min-height: 20px;
                min-width: 120px;
            }
        """
    
    @staticmethod
    def get_button_style(primary=False):
        if primary:
            return """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ECC71, stop:1 #27AE60);
                    color: white;
                    border-radius: 12px;
                    padding: 15px;
                    font-weight: bold;
                    letter-spacing: 1px;
                    min-width: 250px;
                    min-height: 40px;
                    font-size: 12pt;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27AE60, stop:1 #219653);
                }
                QPushButton:pressed {
                    background: #1E8449;
                    padding: 17px 13px 13px 17px;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                    min-width: 130px;
                    min-height: 30px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QPushButton:pressed {
                    background-color: #1B4F72;
                }
            """
    
    @staticmethod
    def get_input_style():
        return """
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: rgba(40, 40, 40, 0.8);
                color: white;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px 10px;
                min-height: 30px;
                min-width: 150px;
                font-size: 10pt;
                font-weight: bold;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #2196F3;
                background-color: rgba(33, 150, 243, 0.2);
            }
        """
    
    @staticmethod
    def get_param_label_style():
        return """
            QLabel[objectName="param_label"] {
                color: white;
                font-size: 11pt;
                background-color: rgba(33, 150, 243, 0.3);
                border-radius: 5px;
                padding: 5px 10px;
                min-width: 120px;
                min-height: 25px;
            }
        """

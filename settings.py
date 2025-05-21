# src/core/settings.py
import os
import json # JSON işlemleri için import eklendi
from dataclasses import dataclass, field, asdict # asdict eklendi
from typing import List, Dict, Optional, Any, Set, Tuple # Optional, Any, Set eklendi/kontrol edildi
import logging # Loglama modülünü import et
logger = logging.getLogger(__name__) # Bu modül için logger objesini al
# Define data classes for different settings categories
@dataclass
class ModelSettings:
    """Settings related to the AI model."""
    sequence_length: int = 32
    note_range: Tuple[int, int] = (21, 108) # MIDI note numbers (A0 to C8)
    # note_range_size artık burada hesaplanmıyor, MIDIModel içinde hesaplanmalı
    input_features: int = 1 # Number of features per note (e.g., pitch activation, velocity)
    output_features: int = 1 # Number of features predicted (e.g., note on/off, velocity)
    resolution: float = 0.125 # Time steps per quarter note (e.g., 8th notes)
    # steps_per_bar artık burada hesaplanmıyor, MIDIModel içinde hesaplanmalı
    lstm_units: int = 128 # Model architecture setting
    dense_units: int = 128 # Model architecture setting
    dropout_rate: float = 0.3 # Model architecture setting
    batch_size: int = 32 # Training setting
    learning_rate: float = 0.001 # Training setting

@dataclass
class GeneralSettings:
    """General application settings."""
    output_dir: str = "generated_midi" # Directory for generated MIDI files (relative to project root)
    model_dir: str = "trained_model" # Directory for saving/loading the model (relative to project root)
    memory_dir: str = "memory" # Directory for memory file (relative to project root)
    memory_file: str = "midi_memory.json" # File name for memory file

@dataclass
class PatternSimilaritySettings:
     """Settings for pattern similarity calculations."""
     tempo_tolerance: float = 0.1 # e.g., 0.1 for 10% tolerance of the reference tempo
     note_density_tolerance: float = 0.5 # e.g., absolute difference in notes per second
     require_exact_key_match: bool = True
     require_exact_time_signature: bool = True
     chord_match_threshold: float = 0.6 # e.g., Jaccard similarity or similar metric (0-1)
     rhythm_complexity_tolerance: float = 0.2 # e.g., absolute difference
     velocity_tolerance: float = 0.2 # e.g., tolerance for average velocity (0-127 scale)
     polyphony_tolerance: float = 0.3 # e.g., tolerance for average polyphony

@dataclass
class MemorySettings:
    """Settings related to MIDI memory and pattern similarity."""
    # memory_file: str = "midi_memory.json" # File name moved to GeneralSettings for path grouping
    similarity_settings: PatternSimilaritySettings = field(default_factory=PatternSimilaritySettings)
    # Add other memory settings here

@dataclass
class Settings:
    """Main settings class combining all configurations."""
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    general_settings: GeneralSettings = field(default_factory=GeneralSettings)
    memory_settings: MemorySettings = field(default_factory=MemorySettings)

    # Helper to get project root path relative to this settings file
    def _get_project_root(self):
         # __file__ is src/core/settings.py
         # os.path.dirname(__file__) is src/core
         # os.path.dirname(os.path.dirname(__file__)) is src
         # os.path.dirname(os.path.dirname(os.path.dirname(__file__))) is project_root
         return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    def load(self, file_path: str = "config/settings.json"):
        """Loads settings from a JSON file."""
        full_file_path = os.path.join(self._get_project_root(), file_path)
        config_dir = os.path.dirname(full_file_path)

        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)

        if not os.path.exists(full_file_path):
            print(f"Settings file not found: {full_file_path}. Saving default settings.")
            self.save(file_path)
            return # Use defaults if file didn't exist

        try:
            with open(full_file_path, 'r', encoding='utf-8') as f:
                 data = json.load(f)

                 # Update dataclass fields from loaded data
                 if 'model_settings' in data:
                      # Use a temporary dict to filter keys to match dataclass fields
                      model_data = {k: v for k, v in data['model_settings'].items() if k in ModelSettings.__dataclass_fields__}
                      self.model_settings = ModelSettings(**model_data)

                 if 'general_settings' in data:
                      general_data = {k: v for k, v in data['general_settings'].items() if k in GeneralSettings.__dataclass_fields__}
                      self.general_settings = GeneralSettings(**general_data)

                 if 'memory_settings' in data:
                      memory_data = data['memory_settings']
                      if 'similarity_settings' in memory_data:
                           # Need custom loading for nested dataclass
                           similarity_data = {k: v for k, v in memory_data['similarity_settings'].items() if k in PatternSimilaritySettings.__dataclass_fields__}
                           self.memory_settings.similarity_settings = PatternSimilaritySettings(**similarity_data)
                      # Update other fields in memory_settings if any
                      # For now, only similarity_settings is nested and explicitly handled
                      # If MemorySettings gets more simple fields, update like model_settings/general_settings


            print(f"Settings loaded successfully from {full_file_path}")

        except json.JSONDecodeError as e:
             print(f"Error decoding settings JSON from {full_file_path}: {e}. Using default settings.")
             # Depending on severity, you might want to show a critical error message box in the GUI
        except Exception as e:
             print(f"An unexpected error occurred while loading settings from {full_file_path}: {e}. Using default settings.")


   # src/core/settings.py dosyasında, Settings sınıfının içinde
# save metodunun TAMAMI aşağıdaki kod ile değiştirilecek

    def save(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        Saves current settings to a file.
        If file_path is None, returns the JSON string representation instead.
        """
        # Eğer file_path None ise, sadece string temsilini döndür
        if file_path is None:
            logger.debug("Settings.save called with file_path=None. Returning string representation.")
            try:
                # Dataclass'ı JSON serileştirme için dict'e dönüştür
                settings_dict = asdict(self)

                # İç içe geçmiş set'leri list'e dönüştürme gibi JSON uyumluluk düzeltmeleri gerekebilir.
                # asdict çoğu durumda iyi çalışsa da, karmaşık yapılar veya setler için manuel düzenleme gerekebilir.
                # Şimdilik asdict'in yeterli olduğunu varsayıyoruz. Eğer serileştirme hatası alırsak burayı düzeltiriz.

                return json.dumps(settings_dict, indent=4) # JSON string'i döndür

            except Exception as e:
                logger.error(f"Error serializing settings for string representation: {e}", exc_info=True)
                return f"Error serializing settings: {e}" # Hata mesajını string olarak döndür


        # Eğer file_path belirtilmişse, dosyaya kaydetme işlemine devam et
        logger.info(f"Saving settings to {file_path}...")
        # Dizinin var olduğundan emin ol
        config_dir = os.path.dirname(file_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
            logger.info(f"Created config directory: {config_dir}")

        try:
            # Dataclass'ı JSON serileştirme için dict'e dönüştür
            settings_dict = asdict(self)

            # İç içe geçmiş set'leri list'e dönüştürme gibi JSON uyumluluk düzeltmeleri gerekebilir.
            # Eğer burada JSON serileştirme hatası alırsak (örn: TypeError: Object of type set is not JSON serializable),
            # Settings objesi için daha kapsamlı bir to_dict metodu yazmamız gerekebilir.

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=4)

            logger.info("Settings saved successfully.")
            return None # Dosyaya kaydedince None döndür (konvansiyon)

        except Exception as e:
            logger.error(f"Error saving settings to {file_path}: {e}", exc_info=True)
            return f"Error saving settings: {e}" # Hata mesajını string olarak döndür
    # Add properties for frequently accessed settings if needed
    @property
    def min_midi_note(self):
         return self.model_settings.note_range[0]

    @property
    def max_midi_note(self):
         return self.model_settings.note_range[1]

    @property
    def resolution(self):
         return self.model_settings.resolution

    @property
    def output_dir_path(self):
         # Return absolute path relative to project root for generated MIDI files
         return os.path.join(self._get_project_root(), self.general_settings.output_dir)

    @property
    def model_dir_path(self):
         # Return absolute path relative to project root for model files
         return os.path.join(self._get_project_root(), self.general_settings.model_dir)

    @property
    def memory_file_full_path(self):
         # Return absolute path for the memory file
         memory_dir = os.path.join(self._get_project_root(), self.general_settings.memory_dir)
         return os.path.join(memory_dir, self.general_settings.memory_file)


# Example of how to use it (in main.py or elsewhere)
if __name__ == '__main__':
    # This block demonstrates loading/saving and accessing settings
    print("Testing settings.py:")
    settings = Settings()
    # Try loading from default path config/settings.json
    settings.load()
    print(f"Model Sequence Length: {settings.model_settings.sequence_length}")
    print(f"Generated MIDI Output Directory: {settings.output_dir_path}")
    print(f"Memory File Path: {settings.memory_file_full_path}")

    # Example of modifying a setting and saving (this save is currently a placeholder print)
    # settings.model_settings.sequence_length = 64
    # settings.general_settings.output_dir = "my_midi_outputs"
    # settings.memory_settings.similarity_settings.tempo_tolerance = 0.2
    # settings.save()

    # Example of creating a settings file with defaults if it doesn't exist
    # If you delete config/settings.json and run this block, it will print that it's saving defaults.
    # settings_new = Settings()
    # settings_new.load() # This will trigger saving if the file is missing
    # print(f"Loaded sequence length (new instance): {settings_new.model_settings.sequence_length}")
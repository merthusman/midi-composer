# src/midi/midi_memory.py
import json
import os
import logging
from dataclasses import dataclass, asdict, field # asdict added for serialization
from typing import List, Dict, Optional, Set, Tuple, Any # Added Any
from datetime import datetime
import numpy as np # Added numpy import
import uuid # Used for pattern_id

try:
    from src.midi.processor import MIDIAnalysis # MIDIAnalysis is needed for PatternData dataclass
    from src.core.settings import PatternSimilaritySettings, MemorySettings # Import settings related dataclasses
    from src.midi.instrument_library import InstrumentLibrary # Added InstrumentLibrary import

    _midi_analysis_imported_in_memory = True
    _settings_imported_in_memory = True # Check if settings imported
    _instrument_library_imported_in_memory = True # Check if InstrumentLibrary imported

except ImportError as e:
    logging.error(f"Failed to import core dependencies in midi_memory: {e}. Memory functionality may be limited.", exc_info=True)
    _midi_analysis_imported_in_memory = False
    _settings_imported_in_memory = False
    _instrument_library_imported_in_memory = False

@dataclass
class PatternSimilaritySettings:
    pass

class InstrumentLibrary:
    def __init__(self, *args, **kwargs): pass
    def get_instrument(self, name): return None


logger = logging.getLogger(__name__) # Get logger for this module


@dataclass
class MIDIPattern:
    """Stores data for a single MIDI pattern."""
    pattern_id: str # Unique identifier for the pattern
    file_path: str # Original file path (for reference)
    analysis: Optional[MIDIAnalysis] # Analysis results for this pattern
    category: Optional[str] = None # User-defined category (e.g., 'Chorus', 'Verse')
    tags: Set[str] = field(default_factory=set) # User-defined tags
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat()) # ISO format string

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a dictionary for JSON serialization."""
        data = asdict(self)
        data['tags'] = list(data['tags']) # Convert set to list for JSON serialization
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['MIDIPattern']:
         if not data.get("pattern_id"):
              logger.warning("Cannot create MIDIPattern from dict: missing pattern_id.")
              return None

         analysis_data = data.get("analysis")
         analysis_obj = MIDIAnalysis.from_dict(analysis_data) if analysis_data and _midi_analysis_imported_in_memory else None

         return cls(
            pattern_id=data["pattern_id"],
            file_path=data.get("file_path", ""),
            analysis=analysis_obj,
            category=data.get("category"),
            tags=set(data.get("tags", [])),
            creation_date=data.get("creation_date", datetime.now().isoformat())
        )


@dataclass
class PatternCategory:
    """Stores information about a user-defined category."""
    name: str # Category name
    description: Optional[str] = None
    patterns: Set[str] = field(default_factory=set) # Set of pattern_ids belonging to this category

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a dictionary for JSON serialization."""
        data = asdict(self)
        data['patterns'] = list(data['patterns']) # Convert set to list for JSON
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['PatternCategory']:
         if not data.get("name"):
              logger.warning("Cannot create PatternCategory from dict: missing name.")
              return None

         return cls(
             name=data["name"],
             description=data.get("description"),
             patterns=set(data.get("patterns", []))
         )


@dataclass
class MemoryData:
    """The main data structure for the MIDI memory."""
    patterns: Dict[str, MIDIPattern] = field(default_factory=dict) # pattern_id -> MIDIPattern
    categories: Dict[str, PatternCategory] = field(default_factory=dict) # category_name -> PatternCategory

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif hasattr(obj, 'to_dict'):
            return self._convert_numpy_types(obj.to_dict())
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory data to a dictionary, handling numpy types."""
        data_dict = {
            'patterns': {
                pattern_id: {
                    'file_path': pattern.file_path,
                    'pattern_id': pattern.pattern_id,
                    'category': pattern.category,
                    'tags': list(pattern.tags) if pattern.tags else [],
                    'creation_date': pattern.creation_date,
                    'analysis': self._convert_numpy_types(pattern.analysis.to_dict()) if pattern.analysis else None
                }
                for pattern_id, pattern in self.patterns.items()
            },
            'categories': {
                category_name: category.to_dict()
                for category_name, category in self.categories.items()
            }
        }
        return data_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryData':
         patterns_data = data.get("patterns", {})
         categories_data = data.get("categories", {})

         patterns = {pid: MIDIPattern.from_dict(pdata) for pid, pdata in patterns_data.items() if MIDIPattern.from_dict(pdata) is not None}
         categories = {name: PatternCategory.from_dict(cdata) for name, cdata in categories_data.items() if PatternCategory.from_dict(cdata) is not None}

         return cls(
            patterns=patterns,
            categories=categories
        )


class MIDIMemory:
    def __init__(self, settings, instrument_library: Optional[InstrumentLibrary] = None):
        if not _settings_imported_in_memory:
            logger.error("Settings dataclass not imported. Cannot initialize MIDIMemory.")
            raise ImportError("Settings dataclass is required for MIDIMemory initialization.")
        if not _midi_analysis_imported_in_memory:
             logger.warning("MIDIAnalysis not imported. MIDIMemory may not be able to load/process pattern analysis data.")

        self.settings = settings
        self.instrument_library = instrument_library

        self.storage_file = self.settings.memory_file_full_path

        self.data: MemoryData = MemoryData()
        self._load_memory()

        self.similarity_settings = self.settings.memory_settings.similarity_settings

        logger.info(f"MIDIMemory initialized. Storage file: {self.storage_file}")


    def _load_memory(self):
        """Loads memory data from the storage file."""
        if not os.path.exists(self.storage_file):
            logger.info(f"Memory file not found at {self.storage_file}. Starting with empty memory.")
            self.data = MemoryData()
            return

        logger.info(f"Attempting to load memory from {self.storage_file}")
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                self.data = MemoryData.from_dict(loaded_data)
            logger.info(f"Memory loaded successfully from {self.storage_file}. Loaded {len(self.data.patterns)} patterns and {len(self.data.categories)} categories.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding memory JSON from {self.storage_file}: {e}. Starting with empty memory.", exc_info=True)
            self.data = MemoryData()
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading memory from {self.storage_file}: {e}. Starting with empty memory.", exc_info=True)
            self.data = MemoryData()


    def _save_memory(self):
        """Save memory data to file."""
        logger.info(f"Attempting to save memory to {self.storage_file}")
        try:
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            
            data_dict = self.data.to_dict()
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=4)
            
            logger.info("Memory saved successfully.")
        except Exception as e:
            logger.error(f"Error saving memory to {self.storage_file}: {e}", exc_info=True)

    def save(self):
        """Public method to save memory data. Calls the internal _save_memory."""
        self._save_memory()

    def add_pattern(self, midi_path: str, analysis: Optional[MIDIAnalysis], category: Optional[str] = None, tags: Optional[Set[str]] = None) -> Optional[str]:
        """Adds a new pattern to the memory."""
        if not os.path.exists(midi_path):
            logger.warning(f"Cannot add pattern: MIDI file not found at {midi_path}.")
            return None

        pattern_id = str(uuid.uuid4())
        new_pattern = MIDIPattern(
            pattern_id=pattern_id,
            file_path=midi_path,
            analysis=analysis,
            category=category,
            tags=tags if tags is not None else set(),
            creation_date=datetime.now().isoformat()
        )
        self.data.patterns[pattern_id] = new_pattern
        logger.info(f"Added pattern '{os.path.basename(midi_path)}' with ID {pattern_id}")

        if category:
            self.add_pattern_to_category(pattern_id, category)
        else:
            self._save_memory()

        return pattern_id


    def get_pattern(self, pattern_id: str) -> Optional[MIDIPattern]:
        """Retrieves a pattern by its ID."""
        return self.data.patterns.get(pattern_id)

    def get_all_patterns(self) -> List[MIDIPattern]:
        """Returns a list of all patterns in the memory."""
        return list(self.data.patterns.values())


    def remove_pattern(self, pattern_id: str) -> bool:
        """Removes a pattern from the memory."""
        if pattern_id not in self.data.patterns:
            logger.warning(f"Pattern with ID {pattern_id} not found for removal.")
            return False

        removed_pattern = self.data.patterns.pop(pattern_id)
        logger.info(f"Removed pattern with ID {pattern_id} ('{os.path.basename(removed_pattern.file_path)}')")

        if removed_pattern.category and removed_pattern.category in self.data.categories:
            self.data.categories[removed_pattern.category].patterns.discard(pattern_id)
            logger.debug(f"Removed pattern ID {pattern_id} from category '{removed_pattern.category}'.")

        self._save_memory()
        return True


    def add_category(self, name: str, description: Optional[str] = None) -> bool:
        """Adds a new category to the memory."""
        if name in self.data.categories:
            logger.warning(f"Category '{name}' already exists.")
            return False

        new_category = PatternCategory(name=name, description=description)
        self.data.categories[name] = new_category
        self._save_memory()
        logger.info(f"Added category: '{name}'")
        return True


    def get_category(self, name: str) -> Optional[PatternCategory]:
        """Retrieves a category by its name."""
        return self.data.categories.get(name)

    def get_all_categories(self) -> List[PatternCategory]:
        """Returns a list of all categories."""
        return list(self.data.categories.values())


    def remove_category(self, name: str) -> bool:
        """Removes a category and unlinks patterns from it."""
        if name not in self.data.categories:
             logger.warning(f"Category '{name}' not found for deletion.")
             return False

        removed_category = self.data.categories.pop(name)
        logger.info(f"Removed category: '{name}'.")

        for pattern in list(self.data.patterns.values()):
            if pattern.category == name:
                 pattern.category = None
                 logger.debug(f"Unlinked pattern ID {pattern.pattern_id} from category '{name}'.")

        self._save_memory()
        return True


    def add_pattern_to_category(self, pattern_id: str, category_name: str) -> bool:
        """Adds a pattern to a category."""
        pattern = self.get_pattern(pattern_id)
        category = self.get_category(category_name)

        if not pattern:
            logger.warning(f"Pattern with ID {pattern_id} not found.")
            return False
        if not category:
            logger.warning(f"Category '{category_name}' not found. Cannot add pattern to category.")
            return False

        pattern.category = category_name
        category.patterns.add(pattern_id)
        self._save_memory()
        logger.info(f"Added pattern ID {pattern_id} to category '{category_name}'.")
        return True


    def remove_pattern_from_category(self, pattern_id: str, category_name: str) -> bool:
        """Removes a pattern from a specific category."""
        pattern = self.get_pattern(pattern_id)
        category = self.get_category(category_name)

        if not pattern:
            logger.warning(f"Pattern with ID {pattern_id} not found.")
            return False
        if not category:
            logger.warning(f"Category '{category_name}' not found.")
            return False

        if pattern.category != category_name:
             logger.warning(f"Pattern with ID {pattern_id} is not in category '{category_name}'. No action needed.")
             return False

        pattern.category = None
        category.patterns.discard(pattern_id)
        self._save_memory()
        logger.info(f"Removed pattern ID {pattern_id} from category '{category_name}'.")
        return True


    def find_similar_patterns(self, reference_analysis: MIDIAnalysis) -> List[MIDIPattern]:
        """
        Finds patterns in memory similar to the reference analysis based on similarity settings.
        Returns a list of MIDIPattern objects sorted by similarity score.
        Uses the similarity_settings stored in this instance.
        """
        if not _midi_analysis_imported_in_memory or not _settings_imported_in_memory:
             logger.error("Cannot perform similarity search: MIDIAnalysis or Settings not imported.")
             return []

        logger.info(f"Searching for similar patterns to reference (Tempo: {reference_analysis.tempo:.2f}, Key: {reference_analysis.key})...")
        similar_patterns_with_score: List[Tuple[MIDIPattern, float]] = []

        for pattern in self.data.patterns.values():
            if pattern.analysis:
                 score = self._calculate_similarity_score(reference_analysis, pattern.analysis, self.similarity_settings)
                 if score > 0:
                     similar_patterns_with_score.append((pattern, score))

        similar_patterns_with_score.sort(key=lambda item: item[1], reverse=True)

        sorted_similar_patterns = [item[0] for item in similar_patterns_with_score]
        logger.info(f"Finished searching. Found {len(sorted_similar_patterns)} patterns with similarity score > 0.")
        return sorted_similar_patterns


    def _calculate_similarity_score(self, analysis1: MIDIAnalysis, analysis2: MIDIAnalysis, settings: PatternSimilaritySettings) -> float:
        """Calculates a similarity score between two MIDIAnalysis objects based on settings."""
        score_components: Dict[str, float] = {}

        tempo_diff = abs(analysis1.tempo - analysis2.tempo)
        if analysis1.tempo > 0:
             tempo_score = 1.0 if tempo_diff / analysis1.tempo <= settings.tempo_tolerance else 0.0
        else:
             tempo_score = 1.0 if tempo_diff == 0 else 0.0

        score_components['tempo'] = tempo_score

        if settings.require_exact_key_match:
             key_score = 1.0 if analysis1.key == analysis2.key else 0.0
        else:
             key_score = 0.0
             if analysis1.key != "Unknown" and analysis2.key != "Unknown":
                 if analysis1.key == analysis2.key:
                      key_score = 1.0

             score_components['key'] = key_score

        time_signature_score = 1.0 if analysis1.time_signature == analysis2.time_signature else 0.0
        score_components['time_signature'] = time_signature_score

        set1_programs = set(analysis1.instrument_programs.keys())
        set2_programs = set(analysis2.instrument_programs.keys())
        intersection_programs = set1_programs.intersection(set2_programs)
        union_programs = set1_programs.union(set2_programs)
        instrument_program_jaccard = len(intersection_programs) / len(union_programs) if union_programs else 0.0

        instrument_category_score = 0.0
        if self.instrument_library and _instrument_library_imported_in_memory:
             categories1 = {self.instrument_library.get_instrument(name).category for name in analysis1.instrument_names.keys() if self.instrument_library.get_instrument(name)}
             categories2 = {self.instrument_library.get_instrument(name).category for name in analysis2.instrument_names.keys() if self.instrument_library.get_instrument(name)}
             intersection_categories = categories1.intersection(categories2)
             union_categories = categories1.union(categories2)
             instrument_category_jaccard = len(intersection_categories) / len(union_categories) if union_categories else 0.0
             instrument_category_score = instrument_category_jaccard

        score_components['instrument'] = instrument_program_jaccard

        avg_polyphony1 = sum(analysis1.polyphony_profile.values()) / max(1, len(analysis1.polyphony_profile))
        avg_polyphony2 = sum(analysis2.polyphony_profile.values()) / max(1, len(analysis2.polyphony_profile))
        polyphony_diff = abs(avg_polyphony1 - avg_polyphony2)
        polyphony_score = 1.0 if polyphony_diff <= settings.polyphony_tolerance else 0.0

        score_components['polyphony'] = polyphony_score

        rhythm_diff = abs(analysis1.rhythm_complexity - analysis2.rhythm_complexity)
        rhythm_score = 1.0 if rhythm_diff <= settings.rhythm_complexity_tolerance else 0.0
        score_components['rhythm_complexity'] = rhythm_score

        weights = {
            'tempo': 0.2,
            'key': 0.1,
            'time_signature': 0.05,
            'instrument': 0.2,
            'polyphony': 0.15,
            'rhythm_complexity': 0.1,
        }

        total_weight = sum(weights.values())
        if total_weight == 0:
             logger.warning("Total similarity score weights are zero. Score will always be 0.")
             return 0.0

        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        final_score = 0.0
        for component, score in score_components.items():
            final_score += score * normalized_weights.get(component, 0)

        final_score = max(0.0, min(1.0, final_score))

        return final_score

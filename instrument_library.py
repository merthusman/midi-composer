# src/midi/instrument_library.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple # Added Tuple
from enum import Enum
import logging # Import logging
# Remove unused import if dataclasses_json is not used for serialization here
# from dataclasses_json import dataclass_json

logger = logging.getLogger(__name__) # Get logger for this module


class InstrumentCategory(Enum):
    LEAD = "Lead"
    BASS = "Bass"
    RHYTHM = "Rhythm/Guitar/Keys"
    PADS = "Pads"
    PLUCK = "Pluck"
    DRUMS = "Drums" # Added Drums category
    PERCUSSION = "Percussion" # Added Percussion category
    FX = "FX" # Added FX category
    OTHER = "Other" # Added a generic category

    @staticmethod
    def list_categories():
        """Returns a list of all InstrumentCategory names."""
        return [category.value for category in InstrumentCategory]


class MusicStyle(Enum):
    SYNTHWAVE = "Synthwave"
    ROCK = "Rock"
    ELECTRONIC = "Electronic"
    POP = "Pop"
    JAZZ = "Jazz" # Added Jazz style
    CLASSICAL = "Classical" # Added Classical style
    HIPHOP = "HipHop" # Added HipHop style
    AMBIENT = "Ambient" # Added Ambient style
    # Add more styles

    @staticmethod
    def list_styles():
        """Returns a list of all MusicStyle names."""
        return [style.value for style in MusicStyle]


@dataclass
class Articulation:
    name: str
    velocity_min: int # Minimum velocity for this articulation (0-127)
    velocity_max: int # Maximum velocity for this articulation (0-127)
    description: str = "" # Optional description

    def to_dict(self):
        return {
            "name": self.name,
            "velocity_min": self.velocity_min,
            "velocity_max": self.velocity_max,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates an Articulation object from a dictionary."""
        return cls(
            name=data.get("name", "Unknown Articulation"),
            velocity_min=data.get("velocity_min", 0),
            velocity_max=data.get("velocity_max", 127),
            description=data.get("description", "")
        )


@dataclass
class Scale:
    name: str
    notes: List[int] # List of semitone intervals from the root (0-11, 0=root)
    description: str = "" # Optional description

    def to_dict(self):
        return {
            "name": self.name,
            "notes": self.notes,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
         """Creates a Scale object from a dictionary."""
         return cls(
            name=data.get("name", "Unknown Scale"),
            notes=data.get("notes", []),
            description=data.get("description", "")
        )


@dataclass
class Instrument:
    name: str
    category: InstrumentCategory # Use the Enum type directly
    midi_program: int # General MIDI program number (0-127)
    velocity_range: Tuple[int, int] # Typical/effective velocity range (min, max 1-127)
    octave_range: Tuple[int, int] # Typical/effective octave range (min, max)
    description: str = ""
    preferred_scales: List[Scale] = field(default_factory=list) # Scales where this instrument sounds good
    articulations: List[Articulation] = field(default_factory=list) # Supported articulations/velocity layers
    # Add other properties like typical sustain, release, attack characteristics?

    def to_dict(self):
        """Converts the Instrument object to a dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category.value if isinstance(self.category, Enum) else self.category, # Save Enum as its value (string)
            "midi_program": self.midi_program,
            "velocity_range": self.velocity_range,
            "octave_range": self.octave_range,
            "preferred_scales": [scale.to_dict() for scale in self.preferred_scales],
            "articulations": [articulation.to_dict() for articulation in self.articulations],
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
         """Creates an Instrument object from a dictionary."""
         # Safely get values, handle Enum conversion
         category_value = data.get("category")
         category = InstrumentCategory.OTHER # Default category if value is invalid or missing
         if category_value:
             try:
                 # Find the enum member by value, robust against case or slight variations if needed
                 category = InstrumentCategory(category_value)
             except ValueError:
                  logger.warning(f"Unknown or invalid instrument category value '{category_value}' for instrument '{data.get('name', 'Unknown')}'. Using default category '{category.value}'.")


         return cls(
            name=data.get("name", "Unknown Instrument"),
            category=category,
            midi_program=data.get("midi_program", 0), # Default to Acoustic Grand Piano (GM program 0)
            velocity_range=tuple(data.get("velocity_range", (0, 127))), # Default full range
            octave_range=tuple(data.get("octave_range", (0, 10))), # Default full range
            preferred_scales=[Scale.from_dict(scale_data) for scale_data in data.get("preferred_scales", [])],
            articulations=[Articulation.from_dict(articulation_data) for articulation_data in data.get("articulations", [])],
            description=data.get("description", "")
        )

    def is_compatible_with(self, other: 'Instrument') -> bool:
        """
        Basic compatibility check between two instruments.
        Needs more sophisticated logic for real-world use (e.g., considering style,
        pitch range overlap, role in the mix).
        """
        # Simple example: Avoid multiple instruments from the same category unless it's Drums/Percussion/FX
        if self.category == other.category and self.category not in [InstrumentCategory.DRUMS, InstrumentCategory.PERCUSSION, InstrumentCategory.FX, InstrumentCategory.OTHER]:
             logger.debug(f"Compatibility check: Multiple instruments in category {self.category.value} may conflict.")
             # This basic check might be too restrictive. More advanced logic is needed.
             # For a simple check, we could return False here if strict category uniqueness is desired.
             pass # For now, allow same categories but the debug log gives a hint.


        # Add other compatibility checks if needed (e.g., pitch range overlap, preferred styles)

        return True # Default to compatible for basic check


class InstrumentLibrary:
    def __init__(self, settings=None):
        """
        Initialize the instrument library.
        
        Args:
            settings: Optional settings object that may contain configuration for the library
        """
        self.settings = settings
        self.instruments: Dict[str, Instrument] = {}
        self.style_presets: Dict[MusicStyle, List[str]] = {} # Store list of instrument names for each style
        
        # If settings are provided, use them to configure the library
        if settings:
            self._load_from_settings(settings)
        else:
            # Otherwise, initialize with default instruments
            self._initialize_default_instruments()
            self._initialize_style_presets()
        
        logger.info(f"InstrumentLibrary initialized with {len(self.instruments)} instruments and {len(self.style_presets)} style presets.")
    
    def _load_from_settings(self, settings):
        """Load instruments from settings if available."""
        try:
            if hasattr(settings, 'instrument_settings'):
                # Load instruments from settings
                for inst_data in settings.instrument_settings:
                    try:
                        instrument = Instrument.from_dict(inst_data)
                        self.add_instrument(instrument)
                    except Exception as e:
                        logger.warning(f"Failed to load instrument from settings: {e}")
        except Exception as e:
            logger.warning(f"Failed to load instruments from settings: {e}")
            # Fall back to default initialization
            self._initialize_default_instruments()
            self._initialize_style_presets()


    def _initialize_default_instruments(self):
        """Initializes the library with a set of default instruments."""
        # Define common scales
        major_scale = Scale(name="Major", notes=[0, 2, 4, 5, 7, 9, 11], description="Major scale relative to root")
        minor_scale = Scale(name="Minor", notes=[0, 2, 3, 5, 7, 8, 10], description="Minor scale relative to root")
        pentatonic_minor = Scale(name="Pentatonic Minor", notes=[0, 3, 5, 7, 10], description="Minor pentatonic scale")
        blues_scale = Scale(name="Blues", notes=[0, 3, 5, 6, 7, 10], description="Blues scale")
        chromatic_scale = Scale(name="Chromatic", notes=list(range(12)), description="All 12 notes")


        # Add default instruments based on GM MIDI Program numbers where applicable
        # Program 0: Acoustic Grand Piano
        self.add_instrument(Instrument(name="Acoustic Grand Piano", category=InstrumentCategory.RHYTHM, midi_program=0, velocity_range=(10, 127), octave_range=(0, 10), preferred_scales=[major_scale, minor_scale, chromatic_scale], description="Standard acoustic grand piano."))
        # Program 4: Electric Piano 1
        self.add_instrument(Instrument(name="Electric Piano 1", category=InstrumentCategory.RHYTHM, midi_program=4, velocity_range=(10, 120), octave_range=(1, 8), preferred_scales=[major_scale, minor_scale, blues_scale], description="Electric piano sound."))
        # Program 33: Electric Bass (Finger)
        self.add_instrument(Instrument(name="Finger Electric Bass", category=InstrumentCategory.BASS, midi_program=33, velocity_range=(30, 110), octave_range=(0, 3), preferred_scales=[minor_scale, pentatonic_minor, blues_scale], description="Fingered electric bass sound."))
        # Program 81: Lead 1 (Square)
        self.add_instrument(Instrument(name="Synth Lead Square", category=InstrumentCategory.LEAD, midi_program=81, velocity_range=(40, 100), octave_range=(3, 6), preferred_scales=[minor_scale, blues_scale, pentatonic_minor], description="A typical square wave synth lead."))
        # Program 89: Pad 1 (New Age)
        self.add_instrument(Instrument(name="Synth Pad New Age", category=InstrumentCategory.PADS, midi_program=89, velocity_range=(20, 90), octave_range=(2, 7), preferred_scales=[major_scale, minor_scale, chromatic_scale], description="A warm, evolving synth pad."))
        # Program 105: Sitar (Example of a plucked instrument)
        self.add_instrument(Instrument(name="Sitar", category=InstrumentCategory.PLUCK, midi_program=105, velocity_range=(30, 100), octave_range=(3, 6), preferred_scales=[Scale(name="Indian Thaat Kalyan", notes=[0, 2, 4, 6, 7, 9, 11], description="A scale suitable for Sitar")], description="Indian Sitar."))
        # Program 1: Bright Acoustic Piano
        self.add_instrument(Instrument(name="Bright Acoustic Piano", category=InstrumentCategory.RHYTHM, midi_program=1, velocity_range=(20, 127), octave_range=(0, 10), preferred_scales=[major_scale], description="Brighter acoustic piano sound."))
        # Program 25: Acoustic Guitar (Steel)
        self.add_instrument(Instrument(name="Steel Acoustic Guitar", category=InstrumentCategory.RHYTHM, midi_program=25, velocity_range=(10, 120), octave_range=(1, 9), preferred_scales=[major_scale, minor_scale, pentatonic_minor], description="Steel string acoustic guitar."))
        # Program 34: Electric Bass (Pick)
        self.add_instrument(Instrument(name="Picked Electric Bass", category=InstrumentCategory.BASS, midi_program=34, velocity_range=(40, 120), octave_range=(0, 3), preferred_scales=[minor_scale, pentatonic_minor, blues_scale], description="Picked electric bass sound."))
        # Program 41: Violin
        self.add_instrument(Instrument(name="Violin", category=InstrumentCategory.LEAD, midi_program=40, velocity_range=(30, 110), octave_range=(3, 7), preferred_scales=[major_scale, minor_scale], description="Violin sound.")) # GM program 40 is Violin
        # Program 49: String Ensemble 1
        self.add_instrument(Instrument(name="String Ensemble 1", category=InstrumentCategory.PADS, midi_program=48, velocity_range=(20, 100), octave_range=(1, 8), preferred_scales=[major_scale, minor_scale], description="String ensemble pad.")) # GM program 48 is String Ensemble 1
        # Program 113: Melodic Tom (Example Drum)
        self.add_instrument(Instrument(name="Melodic Tom", category=InstrumentCategory.DRUMS, midi_program=117, velocity_range=(1, 127), octave_range=(0, 10), preferred_scales=[chromatic_scale], description="General MIDI Melodic Tom.")) # GM program 117 is Melodic Tom
        # Program 114: Synth Drum (Example Percussion)
        self.add_instrument(Instrument(name="Synth Drum", category=InstrumentCategory.PERCUSSION, midi_program=118, velocity_range=(1, 127), octave_range=(0, 10), preferred_scales=[chromatic_scale], description="General MIDI Synth Drum.")) # GM program 118 is Synth Drum


    def _initialize_style_presets(self):
        """Defines instrument combinations that work well for different styles."""
        self.style_presets[MusicStyle.POP] = ["Acoustic Grand Piano", "Finger Electric Bass", "Synth Lead Square", "Synth Pad New Age"]
        self.style_presets[MusicStyle.ELECTRONIC] = ["Electric Piano 1", "Synth Lead Square", "Synth Pad New Age", "Finger Electric Bass"]
        self.style_presets[MusicStyle.SYNTHWAVE] = ["Synth Lead Square", "Synth Pad New Age", "Finger Electric Bass", "Electric Piano 1"]
        self.style_presets[MusicStyle.ROCK] = ["Steel Acoustic Guitar", "Picked Electric Bass", "Synth Lead Square", "Melodic Tom"] # Example rock setup
        self.style_presets[MusicStyle.JAZZ] = ["Electric Piano 1", "Finger Electric Bass", "Violin"] # Simple jazz example
        self.style_presets[MusicStyle.CLASSICAL] = ["Acoustic Grand Piano", "Violin", "String Ensemble 1"] # Simple classical example
        self.style_presets[MusicStyle.HIPHOP] = ["Finger Electric Bass", "Electric Piano 1", "Synth Drum"] # Simple HipHop example
        self.style_presets[MusicStyle.AMBIENT] = ["Synth Pad New Age", "String Ensemble 1"] # Simple Ambient example


    def add_instrument(self, instrument: Instrument):
        """Adds an Instrument object to the library."""
        if not isinstance(instrument, Instrument):
             logger.warning(f"Attempted to add non-Instrument object: {type(instrument)}. Skipping.")
             return
        self.instruments[instrument.name] = instrument
        # logger.debug(f"Added instrument: {instrument.name}") # Keep debug level for chatty logs

    def get_instrument(self, name: str) -> Optional[Instrument]:
        """Retrieves an Instrument object by its name."""
        return self.instruments.get(name)

    def get_instruments_by_category(self, category: InstrumentCategory) -> List[Instrument]:
        """Retrieves a list of instruments belonging to a specific category."""
        if not isinstance(category, InstrumentCategory):
             logger.warning(f"Invalid category type for get_instruments_by_category: {type(category)}. Returning empty list.")
             return []
        return [inst for inst in self.instruments.values() if inst.category == category]

    def get_instruments_by_midi_program(self, midi_program: int) -> List[Instrument]:
        """Retrieves a list of instruments with a specific MIDI program number."""
        if not 0 <= midi_program <= 127:
             logger.warning(f"Invalid MIDI program number for get_instruments_by_midi_program: {midi_program}. Returning empty list.")
             return []
        return [inst for inst in self.instruments.values() if inst.midi_program == midi_program]


    def get_style_instruments(self, style: MusicStyle) -> List[Instrument]:
        """Retrieves a list of Instrument objects recommended for a specific music style."""
        if not isinstance(style, MusicStyle):
            logger.warning(f"Invalid style type for get_style_instruments: {type(style)}. Returning empty list.")
            return []

        instrument_names = self.style_presets.get(style, [])
        # Return a list of Instrument objects, skipping any names not found in the library
        instruments_list = [self.get_instrument(name) for name in instrument_names if name in self.instruments]
        # Log if some instruments for the style were not found
        if len(instruments_list) != len(instrument_names):
             missing_instruments = set(instrument_names) - {inst.name for inst in instruments_list}
             logger.warning(f"For style '{style.value}', some instruments were not found in the library: {missing_instruments}")

        return instruments_list


    def check_compatibility(self, instruments: List[str]) -> bool:
        """
        Checks if a list of instrument names are compatible with each other
        based on the basic `is_compatible_with` method of each instrument.
        """
        # Get instrument objects, filter out None for names not found
        instrument_objects = [self.get_instrument(name) for name in instruments if name in self.instruments]

        if len(instrument_objects) != len(instruments):
             # Log a warning if some requested instruments were not found in the library
             missing_names = set(instruments) - {inst.name for inst in instrument_objects}
             logger.warning(f"Compatibility check: Some requested instrument names were not found in the library: {missing_names}")
             # Decide if missing instruments should automatically make the set incompatible.
             # For now, proceed with checking compatibility of the instruments that WERE found.


        if len(instrument_objects) < 2:
             logger.debug("Compatibility check: Need at least two valid instruments to check compatibility.")
             return True # A single instrument or empty set is considered compatible with itself/nothing

        # Check pairwise compatibility using the is_compatible_with method
        for i in range(len(instrument_objects)):
            for j in range(i + 1, len(instrument_objects)): # Check each unique pair
                 inst1 = instrument_objects[i]
                 inst2 = instrument_objects[j]
                 if not inst1.is_compatible_with(inst2):
                      # is_compatible_with method already logs the specific failure
                      # logger.info(f"Compatibility check failed between '{inst1.name}' and '{inst2.name}'.") # is_compatible_with logs this
                      return False # Found an incompatible pair

        logger.debug("Compatibility check passed for the provided instruments.")
        return True # All pairs are compatible


# Example usage (if run directly) - Should ideally not configure logging here
if __name__ == "__main__":
    # Configure logging only if running THIS file directly for testing
    # In a real application, logging is configured at the main entry point
    if not logging.getLogger('').handlers: # Check if root logger has handlers
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s: %(message)s")

    logger.info("Running instrument_library.py test.")

    library = InstrumentLibrary()

    # Test retrieving an instrument
    piano = library.get_instrument("Acoustic Grand Piano")
    if piano:
        logger.info(f"Found instrument: {piano.name}, Category: {piano.category.value}")
        # Test to_dict and from_dict (simple serialization test)
        piano_dict = piano.to_dict()
        loaded_piano = Instrument.from_dict(piano_dict)
        logger.info(f"Serialized/Deserialized instrument: {loaded_piano.name}")

    # Test getting instruments by category
    bass_instruments = library.get_instruments_by_category(InstrumentCategory.BASS)
    logger.info(f"Bass instruments: {[inst.name for inst in bass_instruments]}")

    # Test getting instruments by style
    pop_instruments = library.get_style_instruments(MusicStyle.POP)
    logger.info(f"Instruments for Pop style: {[inst.name for inst in pop_instruments]}")

    # Test compatibility check
    compatible_set = ["Acoustic Grand Piano", "Finger Electric Bass", "Synth Lead Square"]
    incompatible_set = ["Synth Lead Square", "Synth Lead Square"] # Example: two leads might be incompatible in some logic
    logger.info(f"Is {compatible_set} compatible? {library.check_compatibility(compatible_set)}")
    logger.info(f"Is {incompatible_set} compatible? {library.check_compatibility(incompatible_set)}")

    logger.info("Instrument library test finished.")
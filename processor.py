# src/midi/processor.py
# Standard library imports
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Set, Tuple

# Third-party imports
try:
    import numpy as np
    import pretty_midi
    import music21
    _midi_libs_imported = True
except ImportError as e:
    logging.error(f"Failed to import MIDI libraries: {e}")
    _midi_libs_imported = False

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Data Classes ---

@dataclass
class MIDIAnalysis:
    """Represents the analysis results of a MIDI file."""
    file_path: str
    duration: float  # seconds
    tempo: float     # BPM
    key: str         # e.g., "C major", "a minor"
    time_signature: str  # e.g., "4/4"
    note_count: int
    average_velocity: float  # 0-127
    pitch_range: Tuple[int, int]  # (min_pitch, max_pitch)
    polyphony_profile: Dict[float, int]  # time_in_seconds -> simultaneous_notes count at that time
    rhythm_complexity: float  # e.g., standard deviation of note durations
    instrument_programs: Dict[int, int]  # program_number -> count of notes for that program
    instrument_names: Dict[str, int]  # GM_name -> count of notes for that name

    def to_dict(self) -> Dict[str, Any]:
        """Converts the MIDIAnalysis object to a dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'duration': float(self.duration),
            'tempo': float(self.tempo),
            'key': str(self.key),
            'time_signature': self.time_signature,
            'note_count': int(self.note_count),
            'average_velocity': float(self.average_velocity),
            'pitch_range': self.pitch_range,
            'polyphony_profile': {str(k): v for k, v in self.polyphony_profile.items()},
            'rhythm_complexity': float(self.rhythm_complexity),
            'instrument_programs': self.instrument_programs,
            'instrument_names': self.instrument_names
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['MIDIAnalysis']:
        try:
            # Convert tuple strings back to actual tuples for pitch_range
            if isinstance(data.get('pitch_range'), str):
                pitch_range_str = data['pitch_range'].strip('()').split(',')
                data['pitch_range'] = (int(pitch_range_str[0]), int(pitch_range_str[1]))
                
            # Convert polyphony profile string keys back to float
            if isinstance(data.get('polyphony_profile'), dict):
                data['polyphony_profile'] = {float(k): v for k, v in data['polyphony_profile'].items()}
                
            return cls(**data)
        except Exception as e:
            logger.error(f"Error creating MIDIAnalysis from dict: {e}")
            return None

@dataclass
class MIDIMemoryPattern:
    """Represents a pattern stored in MIDI memory."""
    sequence: np.ndarray  # The sequence data for the pattern
    source_midi: str      # Path to the original MIDI file
    start_time: float     # Start time in seconds within the source MIDI
    end_time: float       # End time in seconds within the source MIDI
    original_key: str     # Original key of the source MIDI
    original_tempo: float # Original tempo of the source MIDI

@dataclass
class ProcessorSettings:
    """Configuration settings for the MIDI processor."""
    sequence_length: int = 256  # Length of sequences for model
    input_features: int = 2     # Number of features per step (e.g., pitch, velocity)
    output_dir_path: str = "generated_midi"  # Directory to save generated MIDI files
    note_range: Tuple[int, int] = (21, 108)  # MIDI note range (A0-C8)
    resolution: float = 0.125    # Time resolution (1/8th note)
    default_tempo: float = 120  # Default tempo in BPM
    default_velocity: int = 100 # Default note velocity (0-127)

# --- Processor Class ---

class MIDIProcessor:
    """
    Handles MIDI file analysis, sequence processing for models, and sequence to MIDI conversion.
    
    Args:
        settings: Configuration settings for the processor
    """

    def _get_note_pitch_stats(self, midi_data):
        """Get note pitch statistics from MIDI data."""
        all_notes = []
        pitch_counts = {}
        duration_weighted_ratios = {}
        
        # Get all notes from MIDI data
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Get pitch number
                pitch = note.pitch
                
                # Add to all notes
                all_notes.append(pitch)
                
                # Update pitch counts
                pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1
                
                # Calculate duration weighted ratio
                duration = note.end - note.start
                duration_weighted_ratios[pitch] = duration_weighted_ratios.get(pitch, 0) + duration
        
        # Normalize duration weighted ratios
        total_duration = sum(duration_weighted_ratios.values())
        if total_duration > 0:
            for pitch in duration_weighted_ratios:
                duration_weighted_ratios[pitch] /= total_duration
        
        return all_notes, pitch_counts, duration_weighted_ratios
    
    def __init__(self, settings: ProcessorSettings):
        """Initialize the MIDI processor with settings."""
        self.settings = settings
        self._midi_libs_imported = _midi_libs_imported
        
    def analyze_file(self, file_path: str) -> Optional[MIDIAnalysis]:
        """
        Analyze a MIDI file and return a MIDIAnalysis object with musical characteristics.
        
        Args:
            file_path: Path to the MIDI file to analyze
            
        Returns:
            MIDIAnalysis object if successful, None if analysis fails
        """
        if not self._midi_libs_imported:
            logger.error("MIDI libraries not imported")
            return None
            
        try:
            # Load MIDI file using pretty_midi for basic analysis
            midi_data = pretty_midi.PrettyMIDI(file_path)
            
            # Get basic information
            duration = midi_data.get_end_time()
            
            # Get tempo (use first tempo change or default to settings)
            tempo_changes = midi_data.get_tempo_changes()
            tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else self.settings.default_tempo
            
            # Get time signature (use first time signature or default to 4/4)
            time_sig = "4/4"
            if len(midi_data.time_signature_changes) > 0:
                ts = midi_data.time_signature_changes[0]
                time_sig = f"{ts.numerator}/{ts.denominator}"
            
            # Analyze key using our advanced method
            key = self._analyze_key_advanced(midi_data)
            
            # Get all notes and calculate statistics
            all_notes = []
            velocities = []
            pitches = []
            polyphony = {}
            
            # Filter notes within the valid range
            valid_notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    if self.settings.note_range[0] <= note.pitch <= self.settings.note_range[1]:
                        valid_notes.append(note)
                        velocities.append(note.velocity)
                        pitches.append(note.pitch)
                        
                        # Track polyphony (number of simultaneous notes)
                        start = int(note.start * 10)  # 10Hz resolution for polyphony tracking
                        end = int(note.end * 10)
                        for t in range(start, end + 1):
                            polyphony[t/10] = polyphony.get(t/10, 0) + 1
            
            # Calculate note statistics
            note_count = len(valid_notes)
            avg_velocity = sum(velocities) / len(velocities) if velocities else self.settings.default_velocity
            min_pitch = min(pitches) if pitches else self.settings.note_range[0]
            max_pitch = max(pitches) if pitches else self.settings.note_range[1]
            
            # Calculate rhythm complexity (standard deviation of note durations)
            durations = [note.end - note.start for note in valid_notes]
            if durations:
                duration_array = np.array(durations)
                rhythm_complexity = float(np.std(duration_array))
            else:
                rhythm_complexity = 0.0
            
            # Get instrument information
            instrument_programs = {}
            instrument_names = {}
            for instrument in midi_data.instruments:
                program = instrument.program
                name = pretty_midi.program_to_instrument_name(program)
                instrument_programs[program] = instrument_programs.get(program, 0) + len(instrument.notes)
                instrument_names[name] = instrument_names.get(name, 0) + len(instrument.notes)
            
            # Create and return analysis result
            return MIDIAnalysis(
                file_path=file_path,
                duration=duration,
                tempo=tempo,
                key=key,
                time_signature=time_sig,
                note_count=note_count,
                average_velocity=avg_velocity,
                pitch_range=(min_pitch, max_pitch),
                polyphony_profile=polyphony,
                rhythm_complexity=rhythm_complexity,
                instrument_programs=instrument_programs,
                instrument_names=instrument_names
            )
            
        except Exception as e:
            logger.error(f"Error analyzing MIDI file {file_path}: {e}", exc_info=True)
            return None
    
    def _analyze_key_advanced(self, midi_data) -> str:
        """
        Gelişmiş anahtar analizi algoritması:
        1. Nota dağılımını analiz eder
        2. Akor yapısını inceler
        3. Karakteristik aralıkları kontrol eder
        4. Başlangıç ve bitiş notalarına önem verir
        """
        try:
            # Tüm notaları ve akorları al
            all_notes = []
            chords = []
            
            # Nota istatistiklerini al
            all_notes, pitch_counts, duration_weighted_ratios = self._get_note_pitch_stats(midi_data)
            
            # Nota dağılımını analiz et
            note_counts = pitch_counts
            
            # En sık kullanılan nota
            most_common_note = max(note_counts, key=note_counts.get)
            
            # Akor yapısını analiz et
            chord_structure = {}
            for instrument in midi_data.instruments:
                # Her enstrüman için akorları bul
                for note in instrument.notes:
                    # Aynı zaman dilimindeki notaları bul
                    same_time_notes = []
                    for other_note in instrument.notes:
                        if abs(other_note.start - note.start) < 0.1:  # Yaklaşık aynı başlangıç zamanı
                            same_time_notes.append(other_note.pitch)
                    
                    if len(same_time_notes) > 1:  # En az iki nota
                        same_time_notes.sort()
                        chord = tuple(same_time_notes)
                        chord_structure[chord] = chord_structure.get(chord, 0) + 1
            
            # Anahtar adaylarını bul
            key_candidates = []
            
            # En sık kullanılan nota etrafında anahtar adayları
            for offset in [0, 7]:  # Major/minor
                candidate = (most_common_note + offset) % 12
                key_candidates.append(candidate)
            
            # Akor yapısına göre anahtar adayları
            for chord, count in chord_structure.items():
                if len(chord) >= 3:  # En az üç nota olan akorlar
                    # Akorun kökü
                    root = chord[0]
                    if root not in key_candidates:
                        key_candidates[root] = key_candidates.get(root, 0) + count
            
            # Anahtar adaylarını MIDI numarasına göre grupla
            grouped_keys = {}
            for candidate, count in key_candidates.items():
                if 21 <= candidate <= 108:
                    # 12'lik sistemdeki yerini bul
                    mod = candidate % 12
                    grouped_keys[mod] = grouped_keys.get(mod, 0) + count
            
            # Anahtar adaylarını değerlendir ve en iyi sonucu seç
            if grouped_keys:
                most_common_key = max(grouped_keys, key=grouped_keys.get)
                
                # Anahtar isimlerini döndür
                key_names = {
                    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
                    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
                }
                
                # Major/minor kararını ver
                if most_common_note % 12 == most_common_key:
                    return f"{key_names[most_common_key]} major"
                else:
                    return f"{key_names[most_common_key]} minor"
            
            logger.debug("Anahtar adayları bulunamadı")
            return "Unknown"
        
        except Exception as e:
            logger.error(f"Anahtar analizi hatası: {e}", exc_info=True)
            return "Unknown"

    def analyze_midi_file(self, midi_file_path: str) -> Optional[MIDIAnalysis]:
        """Analyzes a MIDI file and returns a MIDIAnalysis object."""
        try:
            logger.info(f"Starting analysis of MIDI file: {midi_file_path}")
            
            # Load with pretty_midi for tempo and basic info
            midi_data_pm = pretty_midi.PrettyMIDI(midi_file_path)
            
            # Geliştirilmiş tempo tespiti algoritması
            tempos, tempo_times = midi_data_pm.get_tempo_changes()
            tempo = None
            
            # 1. Önce dosya adından BPM değerini çıkarmayı dene (genellikle en güvenilir)
            filename = os.path.basename(midi_file_path)
            bpm_match = re.search(r'(\d+)\s*(?:BPM|bpm)', filename)
            if bpm_match:
                tempo = float(bpm_match.group(1))
                logger.info(f"Dosya adından tempo tespit edildi: {tempo} BPM")
            
            # 2. Dosya adından tespit edilemezse, MIDI meta verilerini kontrol et
            if not tempo or tempo <= 0:
                if tempos.size > 0 and tempos[0] > 0:  # Geçerli tempo varsa
                    if tempo_times.size > 1:
                        # Birden fazla tempo değişimi - ağırlıklı ortalama kullan
                        tempo_durations = np.diff(np.append(tempo_times, midi_data_pm.get_end_time()))
                        weighted_tempo = np.average(tempos, weights=tempo_durations)
                        tempo = float(weighted_tempo)
                        logger.info(f"{len(tempos)} tempo değişiminden ağırlıklı ortalama: {tempo:.2f} BPM")
                    else:
                        tempo = float(tempos[0])
                        logger.info(f"MIDI meta verisinden tekil tempo: {tempo} BPM")
            
            # 3. Hala tespit edilemezse, nota zamanlamalarından tahmin et
            if not tempo or tempo <= 0:
                # Nota aralıklarını analiz ederek daha doğru tahmin yap
                notes = [note for instrument in midi_data_pm.instruments for note in instrument.notes]
                if notes:
                    # Nota başlangıç zamanlarını al ve sırala
                    note_starts = sorted([note.start for note in notes])
                    # Ardışık notalar arasındaki zaman farklarını hesapla
                    note_intervals = np.diff(note_starts)
                    # Çok kısa aralıkları filtrele (akorlar için)
                    filtered_intervals = note_intervals[note_intervals > 0.05]
                    
                    if len(filtered_intervals) > 0:
                        # En sık görülen aralıkları bul (vuruş aralıkları olabilir)
                        from scipy import stats
                        if hasattr(stats, 'mode'):
                            mode_result = stats.mode(np.round(filtered_intervals, decimals=2))
                            if hasattr(mode_result, 'mode'):
                                common_interval = mode_result.mode[0]
                            else:  # Newer scipy versions
                                common_interval = mode_result[0]
                        
                        # Vuruş aralığından BPM hesapla (60 saniye / vuruş aralığı)
                        if common_interval > 0:
                            estimated_tempo = 60.0 / common_interval
                            # Makul BPM aralığına düşürme (genelde 60-200 BPM arası)
                            while estimated_tempo > 200:
                                estimated_tempo /= 2
                            while estimated_tempo < 60:
                                estimated_tempo *= 2
                            
                            tempo = float(estimated_tempo)
                            logger.info(f"Nota aralıklarından tahmin edilen tempo: {tempo:.2f} BPM")
            
            # 4. Yine de bulunamazsa, pretty_midi'nin tahmin fonksiyonunu kullan
            if not tempo or tempo <= 0:
                tempo = midi_data_pm.estimate_tempo()
                if tempo > 0:
                    logger.info(f"pretty_midi ile tahmin edilen tempo: {tempo} BPM")
        
            # 5. Son çare olarak varsayılan değeri kullan
            if not tempo or tempo <= 0:
                tempo = 120.0  # Varsayılan tempo
                logger.warning(f"Tempo tespit edilemedi, varsayılan değer kullanılıyor: {tempo} BPM")

            # Load with music21 for advanced analysis
            midi_data_m21 = music21.converter.parse(midi_file_path)
            logger.debug("MIDI file loaded successfully with music21.")
            
            # Get time signature (most common one)
            time_signatures = midi_data_m21.getTimeSignatures()
            if time_signatures:
                ts_counts = {}
                for ts in time_signatures:
                    ts_str = f"{ts.numerator}/{ts.denominator}"
                    ts_counts[ts_str] = ts_counts.get(ts_str, 0) + 1
                time_signature = max(ts_counts.items(), key=lambda x: x[1])[0]
            else:
                time_signature = "4/4"  # Default
            logger.debug(f"Selected time signature: {time_signature}")
            
            # Try to get key from filename first if it's explicitly mentioned
            key = None
            filename = os.path.basename(midi_file_path)
            key_match = re.search(r'([A-G][#b]?\s*(?:major|minor|maj|min))', filename, re.IGNORECASE)
            if key_match:
                key_name = key_match.group(1).lower()
                # Standardize key name
                key_name = key_name.replace('maj', 'major').replace('min', 'minor')
                # Remove any duplicate "or" that might appear from string operations
                key_name = key_name.replace('oror', 'or')
                key = key_name
                logger.debug(f"Using key from filename: {key}")
            
            # If no key in filename or want to verify, use advanced analysis
            if not key:
                key = self._analyze_key_advanced(midi_data_m21)
            logger.debug(f"Final key analysis result: {key}")

            # Get duration and note count
            duration = midi_data_pm.get_end_time()
            notes = list(midi_data_pm.instruments[0].notes) if midi_data_pm.instruments else []
            note_count = len(notes)
            
            # Calculate average velocity
            velocities = [note.velocity for note in notes]
            average_velocity = np.mean(velocities) if velocities else 0.0
            
            # Get pitch range
            if notes:
                pitches = [note.pitch for note in notes]
                pitch_range = (min(pitches), max(pitches))
            else:
                pitch_range = (60, 72)  # Default pitch range if no notes found
            
            # Calculate polyphony profile
            polyphony_profile = {}
            time_step = 0.1  # Resolution for polyphony analysis
            current_time = 0
            while current_time <= duration:
                active_notes = sum(1 for note in notes if note.start <= current_time <= note.end)
                if active_notes > 0:  # Only store non-zero values to save space
                    polyphony_profile[float(current_time)] = active_notes
                current_time += time_step
            
            # Calculate rhythm complexity (using note duration variance)
            note_durations = [note.end - note.start for note in notes]
            rhythm_complexity = float(np.std(note_durations)) if note_durations else 0.0
            
            # Get instrument information
            instrument_programs = {}
            instrument_names = {}
            for instrument in midi_data_pm.instruments:
                if instrument.program not in instrument_programs:
                    instrument_programs[instrument.program] = 0
                    name = pretty_midi.program_to_instrument_name(instrument.program)
                    instrument_names[name] = 0
                
                note_count = len(instrument.notes)
                instrument_programs[instrument.program] += note_count
                instrument_names[pretty_midi.program_to_instrument_name(instrument.program)] += note_count

            # Create and return analysis object
            analysis = MIDIAnalysis(
                file_path=midi_file_path,
                duration=duration,
                tempo=tempo,
                key=key,
                time_signature=time_signature,
                note_count=note_count,
                average_velocity=average_velocity,
                pitch_range=pitch_range,
                polyphony_profile=polyphony_profile,
                rhythm_complexity=rhythm_complexity,
                instrument_programs=instrument_programs,
                instrument_names=instrument_names
            )
            
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing MIDI file: {e}", exc_info=True)
            return None

    def midi_to_sequence(self, midi_file_path: str) -> Optional[np.ndarray]:
        """Converts a MIDI file to a sequence format suitable for the model.
        Returns array shape: (sequence_length, note_range_size, input_features)"""
        try:
            if not os.path.exists(midi_file_path):
                logger.error(f"MIDI file not found: {midi_file_path}")
                return None

            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            end_time = midi_data.get_end_time()

            # Get settings from model_settings if available
            sequence_length = getattr(self.settings, 'sequence_length', 256)
            note_range_start = getattr(self.settings, 'note_range_start', 21)  # MIDI note A0
            note_range_end = getattr(self.settings, 'note_range_end', 108)     # MIDI note C8
            note_range_size = note_range_end - note_range_start + 1
            input_features = getattr(self.settings, 'input_features', 2)  # Default: pitch and velocity

            # Create an empty sequence array
            sequence = np.zeros((sequence_length, note_range_size, input_features), dtype=np.float32)

            # Calculate time step duration based on sequence length and MIDI duration
            time_per_step = end_time / sequence_length

            # For each time step
            for step in range(sequence_length):
                time = step * time_per_step

                # Find all notes that are active at this time
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        if note.start <= time <= note.end:
                            # Convert MIDI pitch to note index
                            note_idx = min(max(note.pitch - note_range_start, 0), note_range_size - 1)
                            sequence[step, note_idx, 0] = 1.0  # Note is active
                            sequence[step, note_idx, 1] = note.velocity / 127.0  # Normalize velocity to [0, 1]

            logger.debug(f"Converted MIDI to sequence with shape {sequence.shape}")
            return sequence

        except Exception as e:
            logger.error(f"Error converting MIDI to sequence: {e}", exc_info=True)
            return None
            
    def sequence_to_midi(self, sequence: np.ndarray, output_path: str, tempo: float = 120.0) -> bool:
        """Converts a sequence back to a MIDI file.
        
        Args:
            sequence: The sequence to convert, shape (sequence_length, note_range_size, input_features)
            output_path: Path to save the MIDI file
            tempo: Tempo in BPM for the MIDI file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if sequence is None or sequence.size == 0:
                logger.error("Cannot convert empty sequence to MIDI")
                return False
                
            # Create a new PrettyMIDI object
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            
            # Create a piano instrument
            piano_program = 0  # Acoustic Grand Piano
            piano = pretty_midi.Instrument(program=piano_program)
            
            # Get settings from model_settings if available
            note_range_start = getattr(self.settings, 'note_range_start', 21)  # MIDI note A0
            sequence_length = sequence.shape[0]
            
            # Define time step duration (4 beats per bar, 4 bars total by default)
            total_duration = 60.0 / tempo * 4 * 4  # 4 beats per bar, 4 bars total
            time_per_step = total_duration / sequence_length
            
            # Track active notes to determine their end times
            active_notes = {}  # {pitch: (start_time, velocity)}
            
            # Process each time step
            for step in range(sequence_length):
                current_time = step * time_per_step
                
                # Check each note in the current step
                for note_idx in range(sequence.shape[1]):
                    # If the note is active (above activation threshold)
                    is_active = sequence[step, note_idx, 0] > 0.5
                    
                    # Get the actual MIDI pitch
                    pitch = note_idx + note_range_start
                    
                    # If note is active and not already being tracked
                    if is_active and pitch not in active_notes:
                        # Get velocity (normalized back to MIDI range 0-127)
                        velocity = int(min(max(sequence[step, note_idx, 1] * 127, 1), 127))
                        active_notes[pitch] = (current_time, velocity)
                    
                    # If note was active but is now inactive, end the note
                    elif not is_active and pitch in active_notes:
                        start_time, velocity = active_notes[pitch]
                        # Create a new note
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=current_time
                        )
                        piano.notes.append(note)
                        del active_notes[pitch]
            
            # End any notes that are still active at the end of the sequence
            final_time = sequence_length * time_per_step
            for pitch, (start_time, velocity) in active_notes.items():
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=final_time
                )
                piano.notes.append(note)
            
            # Add the instrument to the MIDI file
            midi.instruments.append(piano)
            
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write the MIDI file
            midi.write(output_path)
            logger.info(f"Successfully wrote MIDI file to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting sequence to MIDI: {e}", exc_info=True)
            return False
"""
melody_manager.py
===================

This module defines classes and functions for loading, generating and managing
melodies for the ear training application.  Melodies are represented as
collections of simple note events, each carrying MIDI pitch, duration and
velocity values.  The manager can load existing melodies from a directory
containing MIDI or MusicXML files via the music21 library, generate
random melodies within a specified key/scale, transpose melodies and
assemble mixed datasets consisting of actual and random melodies.

Dependencies:

* music21 — for parsing MIDI/MusicXML and working with musical constructs
* random — for randomness in melody generation
* dataclasses — to define simple data containers
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

try:
    import music21 as m21
except ImportError:
    m21 = None  # music21 may not be available in some environments


@dataclass
class NoteEvent:
    """A simple representation of a musical note.

    Attributes
    ----------
    pitch : int
        MIDI note number (e.g. 60 for middle C).
    duration : float
        Duration of the note in quarter lengths (e.g. 1.0 for a quarter note).
    velocity : int
        MIDI velocity (0–127) indicating how loud the note should be. Default
        is 64 (mezzo forte).
    """

    pitch: int
    duration: float
    velocity: int = 64


@dataclass
class Melody:
    """A sequence of NoteEvents with associated metadata.

    Parameters
    ----------
    events : list of NoteEvent
        The ordered sequence of note events making up the melody.
    key : str
        The musical key for the melody (e.g. 'C', 'G# minor').
    tempo : int
        Tempo in beats per minute (BPM).
    name : str
        Optional name or identifier for the melody.
    complexity : float, optional
        Normalised complexity score in the range [0, 1].  A higher score
        indicates a more complex melody (larger intervals, wider range,
        dense rhythms, etc.).
    avg_interval : float, optional
        Average absolute interval between consecutive notes in semitones.
    span : int, optional
        The span (in semitones) between the highest and lowest notes.
    note_density : float, optional
        Average inter‑onset interval (IOI) normalised to [0, 1].  A lower
        value means notes are farther apart in time, while a higher value
        means notes are closer together (faster melody).
    register : str, optional
        Human‑readable indication of the register (e.g. 'bass', 'middle',
        'treble').  This may be inferred from the pitch range.
    """

    events: List[NoteEvent]
    key: str
    tempo: int
    name: str = ""
    complexity: float = 0.0
    avg_interval: float = 0.0
    span: int = 0
    note_density: float = 0.0
    register: str = ""
    # Additional metadata fields (from H5 files or other sources).  This
    # dictionary is optional and will be empty for randomly generated
    # melodies or melodies loaded from sources without additional info.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Path to the original MIDI file, if known.  For random melodies this
    # will be None.
    file_path: Optional[str] = None
    # Chord information extracted from the original file.  Each element
    # represents a chord event with its MIDI pitches and duration (in
    # quarter lengths).  This is optional and will be empty for random
    # melodies or if chords were not extracted.
    chords: List[Dict[str, Any]] = field(default_factory=list)


class MelodyManager:
    """Manage actual and randomly generated melodies.

    This class encapsulates functionality for loading existing melodies from a
    directory of MIDI/MusicXML files, generating new random melodies and
    assembling mixed datasets that combine both actual and randomly generated
    melodies in a specified ratio.
    """

    def __init__(self, melody_dir: Optional[str] = None) -> None:
        self.melody_dir = melody_dir
        # List of Melody objects parsed from files
        self.actual_melodies: List[Melody] = []
        if melody_dir and m21 is not None:
            self.load_melodies(melody_dir)

    def load_from_json(self, json_path: str) -> None:
        """Load preprocessed melodies from a JSON dataset.

        The JSON file must contain a list of dictionaries, each with keys
        ``events`` (a list of event dictionaries with ``pitch``, ``duration`` and
        optionally ``velocity``), ``key``, ``tempo`` and ``name``.  Additional
        metadata fields (e.g. complexity, avg_interval, span, note_density,
        register) will be copied into the Melody objects.  The loaded melodies
        are appended to ``self.actual_melodies``.

        Parameters
        ----------
        json_path : str
            Path to the JSON dataset produced by prepare_dataset.py.
        """
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            events = [NoteEvent(int(ev['pitch']), float(ev['duration']), int(ev.get('velocity', 64))) for ev in entry['events']]
            mel = Melody(
                events=events,
                key=entry.get('key', 'C major'),
                tempo=int(entry.get('tempo', 120)),
                name=entry.get('name', ''),
                complexity=float(entry.get('complexity', 0.0)),
                avg_interval=float(entry.get('avg_interval', 0.0)),
                span=int(entry.get('span', 0)),
                note_density=float(entry.get('note_density', 0.0)),
                register=str(entry.get('register', '')),
                metadata=entry.get('metadata', {}),
                file_path=entry.get('file_path'),
                chords=entry.get('chords', []),
            )
            self.actual_melodies.append(mel)

    def load_melodies(self, melody_dir: str) -> None:
        """Load melodies from MIDI or MusicXML files in the given directory.

        Parameters
        ----------
        melody_dir : str
            Path to a directory containing MIDI (.mid, .midi) or MusicXML
            (.xml, .musicxml) files.  Files outside these extensions are
            ignored.  Parsed melodies are appended to ``self.actual_melodies``.
        """
        if m21 is None:
            raise RuntimeError(
                "music21 is required to load melodies. Please install music21."
            )
        for fname in os.listdir(melody_dir):
            lower = fname.lower()
            if not (lower.endswith(".mid") or lower.endswith(".midi") or lower.endswith(".xml") or lower.endswith(".musicxml")):
                continue
            path = os.path.join(melody_dir, fname)
            try:
                score = m21.converter.parse(path)
            except Exception as exc:
                print(f"Could not parse {fname}: {exc}")
                continue
            # Flatten score into a single stream of notes/rests to simplify
            part = score.flatten().notes.stream()
            events = []
            for element in part:
                if isinstance(element, m21.note.Note):
                    midi_num = element.pitch.midi
                    duration = float(element.quarterLength)
                    velocity = int(element.volume.velocity or 64)
                    events.append(NoteEvent(midi_num, duration, velocity))
            if not events:
                continue
            # Attempt to determine key; default to C major
            try:
                # Use music21 to determine the key of the piece.  If analysis fails,
                # default to C major.  ``name`` returns the pitch class name (e.g., 'C').
                # ``mode`` gives 'major' or 'minor'.
                key_obj = score.analyze('key')
                tonic_name = key_obj.tonic.name
                key_name = f"{tonic_name} {key_obj.mode}"
            except Exception:
                key_name = "C major"
            # Use score's tempo if present
            tempo = 120
            tempos = score.metronomeMarkBoundaries()
            if tempos:
                # returns list of (offset, MetronomeMark)
                tempo = int(tempos[0][1].number)
            self.actual_melodies.append(Melody(events, key_name, tempo, name=fname, file_path=path))
        # Compute metadata (complexity, ranges) for all loaded melodies
        for mel in self.actual_melodies:
            self._compute_metadata(mel)

    def generate_random_melody(
        self,
        key: str = 'C',
        length: int = 8,
        tempo: int = 120,
        scale_type: str = 'major',
        velocity_range: Tuple[int, int] = (50, 100),
        max_interval: int = 12,
        max_span: int = 24,
        register: str = 'any',
        note_density_level: float = 0.5,
    ) -> Melody:
        """Generate a random melody within the given key and scale.

        Parameters
        ----------
        key : str, optional
            The tonic of the key (e.g. 'C', 'G#').  If the key contains a
            quality (e.g. 'A minor'), the tonic and quality will be separated.
        length : int, optional
            Number of notes in the melody.
        tempo : int, optional
            Tempo in beats per minute.
        scale_type : str, optional
            Scale type: 'major', 'minor', 'dorian', etc.  Refer to music21
            documentation for supported scales.
        velocity_range : tuple of int, optional
            Minimum and maximum velocities to randomise dynamics.
        Returns
        -------
        Melody
            A randomly generated Melody object.
        """
        if m21 is None:
            # Without music21 we cannot compute scales; fallback to C major
            tonic = 'C'
            mode = 'major'
        else:
            # Split key into tonic and mode if provided (e.g. 'C minor')
            parts = key.split()
            tonic = parts[0]
            mode = parts[1] if len(parts) > 1 else scale_type
        # Build a scale using music21
        pitches: List[int] = []
        scale_obj = None
        if m21 is not None:
            try:
                # Choose random scale type if requested
                mode_lower = mode.lower()
                # Map friendly names to music21 class prefixes
                scale_map = {
                    'major': 'Major',
                    'minor': 'Minor',
                    'harmonic minor': 'HarmonicMinor',
                    'melodic minor': 'MelodicMinor',
                    'dorian': 'Dorian',
                    'mixolydian': 'Mixolydian',
                    'lydian': 'Lydian',
                    'phrygian': 'Phrygian',
                    'locrian': 'Locrian',
                    'pentatonic': 'Pentatonic',
                    'blues': 'Blues',
                    'altered': 'Altered',
                    'diminished': 'Diminished',
                    'bebop major': 'Bebop',
                    'bebop minor': 'MinorBebop',
                }
                # If mode is 'any', pick randomly from major and minor
                if mode_lower == 'any':
                    mode_lower = random.choice(['major', 'minor'])
                cls_prefix = scale_map.get(mode_lower, mode.capitalize())
                class_name = cls_prefix + 'Scale'
                # Some scale classes have different names (e.g. BebopScale)
                try:
                    scale_class = getattr(m21.scale, class_name)
                except Exception:
                    # Try without 'Scale' suffix
                    scale_class = getattr(m21.scale, cls_prefix)
                scale_obj = scale_class(m21.pitch.Pitch(tonic))
                pitches = [p.midi for p in scale_obj.getPitches(tonic + '2', tonic + '5')]
            except Exception:
                # Could not construct the requested scale; fall back to C major
                scale_obj = None
                pitches = [60, 62, 64, 65, 67, 69, 71]
        else:
            # fallback to C major scale (MIDI numbers)
            pitches = [60, 62, 64, 65, 67, 69, 71]
        # Define some rhythmic values in quarter lengths
        # Filter pitches according to the requested register
        # Define approximate MIDI boundaries for different registers
        register_lower = None
        register_upper = None
        reg = register.lower() if isinstance(register, str) else 'any'
        if reg in ('bass', 'low'):
            register_lower, register_upper = 36, 60  # C2–B3
        elif reg in ('middle', 'mid'):
            register_lower, register_upper = 60, 72  # C4–B4
        elif reg in ('treble', 'high'):
            register_lower, register_upper = 72, 84  # C5–B5
        # Filter pitch list if register constraints are provided
        if register_lower is not None and register_upper is not None:
            pitches = [p for p in pitches if register_lower <= p <= register_upper]
            # If no pitches in this range, fall back to original scale if available
            if not pitches:
                if m21 is not None and scale_obj is not None:
                    try:
                        pitches = [p.midi for p in scale_obj.getPitches(tonic + '2', tonic + '5')]
                    except Exception:
                        pitches = [60, 62, 64, 65, 67, 69, 71]
                else:
                    pitches = [60, 62, 64, 65, 67, 69, 71]
        # Candidate durations (quarter lengths) for rhythmic variety
        durations = [0.25, 0.5, 0.75, 1.0, 1.5]
        # Determine allowed durations based on desired note density.
        # A higher note_density_level selects shorter durations (faster, denser melodies).
        # Sort durations ascending (shortest to longest)
        durations_sorted = sorted(durations)
        # Compute how many durations to include from the smallest upwards.
        # At density=1.0 we include only the smallest duration; at density=0 we include all durations.
        if note_density_level < 0:
            note_density_level = 0.0
        if note_density_level > 1:
            note_density_level = 1.0
        allowed_count = int((1.0 - note_density_level) * (len(durations_sorted) - 1)) + 1
        allowed_durations = durations_sorted[:allowed_count]
        # Generate events respecting max_interval and max_span constraints
        events: List[NoteEvent] = []
        if not pitches:
            return Melody(events, key, tempo)  # empty melody
        # Start with a random pitch
        current_pitch = random.choice(pitches)
        min_pitch = current_pitch
        max_pitch_seen = current_pitch
        for i in range(length):
            if i == 0:
                pitch = current_pitch
            else:
                # Build list of candidate pitches within max_interval and span limits
                candidates = []
                for p in pitches:
                    # Interval constraint
                    if abs(p - current_pitch) > max_interval:
                        continue
                    # Span constraint
                    new_min = min(min_pitch, p)
                    new_max = max(max_pitch_seen, p)
                    if new_max - new_min > max_span:
                        continue
                    candidates.append(p)
                # If no candidates satisfy constraints, fall back to entire scale
                if not candidates:
                    candidates = pitches
                pitch = random.choice(candidates)
                # Update min/max pitch tracking
                min_pitch = min(min_pitch, pitch)
                max_pitch_seen = max(max_pitch_seen, pitch)
                current_pitch = pitch
            duration = random.choice(allowed_durations)
            velocity = random.randint(*velocity_range)
            events.append(NoteEvent(pitch, duration, velocity))
        # Compose a full key string including tonic and mode.  This ensures
        # that random melodies carry both tonic and scale information (e.g.,
        # 'C major').
        full_key = f"{tonic} {mode}".strip()
        melody = Melody(events, full_key, tempo)
        # Compute metadata (complexity, range etc.) for the generated melody
        self._compute_metadata(melody)
        return melody

    @staticmethod
    def transpose_melody(melody: Melody, semitones: int) -> Melody:
        """Return a new Melody transposed by a number of semitones.

        Parameters
        ----------
        melody : Melody
            The original melody to transpose.
        semitones : int
            Number of semitones to shift (positive to transpose up, negative
            to transpose down).
        Returns
        -------
        Melody
            A new melody with all pitches shifted by the given number of
            semitones.  Key is adjusted only if the transposition is a
            multiple of 12; otherwise the key name is left unchanged.
        """
        transposed_events = [
            NoteEvent(event.pitch + semitones, event.duration, event.velocity)
            for event in melody.events
        ]
        # Attempt to adjust the key by semitones; this is simplistic
        new_key = melody.key
        if m21 is not None:
            try:
                # Create a Pitch object, transpose and report the name
                tonic = m21.pitch.Pitch(melody.key.split()[0])
                tonic.transpose(semitones, inPlace=True)
                mode = melody.key.split()[1] if len(melody.key.split()) > 1 else ''
                new_key = f"{tonic.name} {mode}".strip()
            except Exception:
                pass
        return Melody(transposed_events, new_key, melody.tempo, melody.name + f"_transposed_{semitones}")

    def build_dataset(
        self,
        num_melodies: int = 100,
        random_ratio: float = 0.8,
        key: str = 'C',
        tempo: int = 120,
        length: Optional[int] = 8,
        max_interval: int = 12,
        max_span: int = 24,
        register: str = 'any',
        complexity_level: float = 1.0,
        note_density_level: float = 0.5,
        metadata_filter: Optional[str] = None,
        metadata_conditions: Optional[List[Tuple[str, str, List[str]]]] = None,
        scale_type: Optional[str] = None,
        # New filtering parameters for melodic content.  If provided, the
        # manager will search within each actual melody for sub‑sequences
        # matching the criteria and return those truncated melodies.  The
        # parameters apply only to actual melodies; random melodies are
        # unaffected.
        start_note: Optional[str] = None,
        end_note: Optional[str] = None,
        min_between: int = 0,
        max_between: int = 32,
        contains_notes: Optional[str] = None,
        # Variable segmentation options.  When enabled, length is ignored and
        # instead a random length between min_length and max_length is
        # selected for each melody.  For random melodies this determines
        # the number of notes generated; for actual melodies a contiguous
        # subsequence of this length is selected.  align_to_chords is
        # reserved for future use (currently unused).
        variable_length: bool = False,
        min_length: int = 4,
        max_length: int = 16,
        align_to_chords: bool = False,
    ) -> List[Melody]:
        """Create a mixed dataset of actual and random melodies.

        Parameters
        ----------
        num_melodies : int, optional
            Total number of melodies to include in the dataset.
        random_ratio : float, optional
            Fraction of melodies that should be randomly generated.  For
            example, 0.8 means 80% random melodies and 20% actual melodies.
        key : str, optional
            Key to use for random melodies.
        tempo : int, optional
            Tempo for random melodies.
        length : int, optional
            Number of notes in each random melody.
        Returns
        -------
        list of Melody
            A shuffled list of Melody objects.
        """
        dataset: List[Melody] = []
        # Determine how many random and actual melodies to include.  If
        # sequence filters are provided (start/end/contains), we disable
        # random melodies entirely because searching for subsequences in
        # randomly generated melodies is undefined.  Additionally, if
        # random_ratio is 0, we do not generate any random melodies.  We
        # also avoid filling the dataset beyond the available actual
        # melodies.
        if start_note or end_note or contains_notes:
            # ignore random ratio when sequence filters are active
            num_random = 0
            num_actual = num_melodies
        else:
            num_random = int(num_melodies * random_ratio)
            num_actual = num_melodies - num_random
        # Select actual melodies that meet the constraints
        actual_candidates: List[Melody] = []
        for mel in self.actual_melodies:
            # Ensure metadata is computed
            self._compute_metadata(mel)
            # Filter by key and scale.  Determine the requested tonic and
            # mode.  ``key`` may contain both tonic and scale (e.g. 'C major').
            # ``scale_type`` overrides the scale part if provided.  'Any'
            # indicates that any key or scale is acceptable.  When a
            # specific scale_type is provided with 'any' key, the melody must
            # consist entirely of notes from that scale type for at least
            # one tonic.
            mel_tonic = mel.key.split()[0] if mel.key else 'C'
            mel_mode = mel.key.split()[1] if len(mel.key.split()) > 1 else 'major'
            req_tonic = None
            req_mode = None
            if isinstance(key, str) and key.strip():
                parts_in = key.strip().split()
                req_tonic = parts_in[0]
                # If the key string includes a mode (e.g. 'C minor'), use it as the required mode
                if len(parts_in) > 1:
                    req_mode = parts_in[1]
            # Override mode if scale_type is provided
            if scale_type:
                req_mode = scale_type
            # Convert to lower-case for comparison
            tonic_ok = True
            mode_ok = True
            # Tonic filtering: if req_tonic exists and is not 'any', require match
            if req_tonic and req_tonic.lower() not in ('any', ''):
                if mel_tonic.lower() != req_tonic.lower():
                    tonic_ok = False
            # Mode/scale filtering
            if req_mode and req_mode.lower() not in ('any', ''):
                # If the melody's mode matches, accept; else check note subset
                if mel_mode.lower() != req_mode.lower():
                    # If a specific tonic is requested, check notes fit scale
                    if tonic_ok:
                        if not self._notes_fit_scale(mel.events, req_tonic, req_mode):
                            mode_ok = False
                    else:
                        # If key is 'any' but scale specified, test all tonics
                        found = False
                        for cand_pc in ['C','C#','Db','D','D#','Eb','E','F','F#','Gb','G','G#','Ab','A','A#','Bb','B']:
                            if self._notes_fit_scale(mel.events, cand_pc, req_mode):
                                found = True
                                break
                        if not found:
                            mode_ok = False
            if not (tonic_ok and mode_ok):
                continue
            # Filter by length if a fixed length is requested.  When
            # variable_length is enabled, we allow any length and will
            # truncate a subsequence later.
            if length and not variable_length:
                if not (0.5 * length <= len(mel.events) <= 1.5 * length):
                    continue
            # Filter by interval and span
            if mel.avg_interval > max_interval:
                continue
            if mel.span > max_span:
                continue
            # Filter by register
            reg_lower = register.lower() if isinstance(register, str) else 'any'
            if reg_lower not in ('any', '') and mel.register != reg_lower:
                continue
            # Filter by complexity and note density
            if mel.complexity > complexity_level:
                continue
            if mel.note_density > note_density_level:
                continue
            # Metadata filter: if provided, only include melodies whose
            # metadata values contain the filter string (case‑insensitive)
            # Apply simple metadata substring filter
            if metadata_filter:
                found = False
                filt = metadata_filter.lower()
                for k, v in (mel.metadata or {}).items():
                    try:
                        if filt in str(v).lower():
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    continue
            # Apply advanced metadata conditions
            if metadata_conditions:
                passed = True
                for field, op, values in metadata_conditions:
                    # Retrieve value from metadata or built‑in fields
                    if field in mel.metadata:
                        meta_val = mel.metadata[field]
                    else:
                        # Check built‑in fields on Melody
                        try:
                            if field == 'name':
                                meta_val = mel.name
                            elif field == 'key':
                                meta_val = mel.key
                            elif field == 'tempo':
                                meta_val = mel.tempo
                            elif field == 'complexity':
                                meta_val = mel.complexity
                            elif field == 'avg_interval':
                                meta_val = mel.avg_interval
                            elif field == 'span':
                                meta_val = mel.span
                            elif field == 'note_density':
                                meta_val = mel.note_density
                            elif field == 'register':
                                meta_val = mel.register
                            elif field == 'note_count':
                                meta_val = len(mel.events)
                            else:
                                # Field missing
                                passed = False
                                break
                        except Exception:
                            passed = False
                            break
                    try:
                        # Convert arrays or lists to scalar if needed
                        if isinstance(meta_val, list) and meta_val:
                            meta_val = meta_val[0]
                        if op in ('contains', 'equals'):
                            comp_val = str(meta_val).lower() if meta_val is not None else ''
                            target = str(values[0]).lower() if values else ''
                            if op == 'contains' and target not in comp_val:
                                passed = False
                                break
                            if op == 'equals' and comp_val != target:
                                passed = False
                                break
                        else:
                            # Numeric comparisons (attempt to convert to float)
                            num_val = float(meta_val)
                            if op in ('>=', '≥'):
                                if num_val < float(values[0]):
                                    passed = False
                                    break
                            elif op in ('<=', '≤'):
                                if num_val > float(values[0]):
                                    passed = False
                                    break
                            elif op == 'range':
                                low = float(values[0]) if values else float('-inf')
                                high = float(values[1]) if len(values) > 1 else float('inf')
                                if not (low <= num_val <= high):
                                    passed = False
                                    break
                    except Exception:
                        # Unable to compare; treat as fail
                        passed = False
                        break
                if not passed:
                    continue
            actual_candidates.append(mel)
        # If melodic content filters are provided, extract sub‑melodies
        # matching the specified start and end notes (inclusive) and
        # containing additional pitches if required.  Each sub‑melody
        # becomes a new Melody instance with inherited metadata.
        # Note names may be given as MIDI numbers (e.g. '60') or note
        # names (e.g. 'C4', 'G#3') or scale degrees ('1', '5').  Scale
        # degrees are interpreted relative to the melody key.  Values
        # that cannot be parsed are ignored.
        if (start_note or end_note or contains_notes) and actual_candidates:
            sub_candidates: List[Melody] = []
            # Helper to convert note representation to a function that
            # accepts a melody and returns a set of acceptable MIDI
            # pitches.  If a scale degree is specified (1–7), it is
            # mapped to MIDI pitch classes relative to the key (major
            # or minor).  If a note name with octave or plain MIDI
            # number is given, it is converted directly.
            def parse_note(note_str: str) -> Optional[int]:
                note_str = note_str.strip()
                # numeric midi
                if note_str.isdigit():
                    try:
                        val = int(note_str)
                        if 0 <= val <= 127:
                            return val
                    except Exception:
                        pass
                    return None
                # scale degree
                if note_str in {'1','2','3','4','5','6','7'}:
                    return int(note_str)  # mark as degree with sentinel below
                # note name
                # Accept forms like C4, C#4, Db3, etc.
                # Extract pitch class and octave if present
                import re
                m = re.match(r'^([A-Ga-g][#b]?)(\d*)$', note_str)
                if m:
                    pc, oct_str = m.groups()
                    pcs = {
                        'C':0,'C#':1,'DB':1,'D':2,'D#':3,'EB':3,'E':4,'F':5,'F#':6,'GB':6,
                        'G':7,'G#':8,'AB':8,'A':9,'A#':10,'BB':10,'B':11,
                    }
                    pc_val = pcs.get(pc.upper())
                    if pc_val is None:
                        return None
                    if oct_str:
                        try:
                            octave = int(oct_str)
                        except Exception:
                            return None
                        return pc_val + (octave + 1) * 12
                    # If no octave, treat as pitch class sentinel (value < 0)
                    return -(pc_val + 1)
                return None
            # Parse start/end and contains into lists
            start_parsed = parse_note(start_note) if start_note else None
            end_parsed = parse_note(end_note) if end_note else None
            contains_list: List[Optional[int]] = []
            if contains_notes:
                for tok in contains_notes.replace(',', ' ').split():
                    val = parse_note(tok)
                    if val is not None:
                        contains_list.append(val)
            for mel in actual_candidates:
                events = mel.events
                # Precompute pitch list for contains filter
                event_pitches = [ev.pitch for ev in events]
                # Filter by contains (simple case): ensure all specified
                # pitches (exact MIDI numbers) or pitch classes are present
                contains_ok = True
                if contains_list:
                    for cval in contains_list:
                        if cval is None:
                            continue
                        if cval >= 0:
                            # numeric MIDI or degree interpreted as number
                            # For degrees (1–7) we require at least one pitch
                            # matching that degree in this melody
                            if 1 <= cval <= 7:
                                # compute degree relative to key
                                key_parts = mel.key.split()
                                tonic = key_parts[0] if key_parts else 'C'
                                mode = key_parts[1] if len(key_parts) > 1 else 'major'
                                def pitch_to_degree(pitch: int) -> str:
                                    pc = pitch % 12
                                    major_degrees = {0:'1',2:'2',4:'3',5:'4',7:'5',9:'6',11:'7'}
                                    minor_degrees = {0:'1',2:'2',3:'3',5:'4',7:'5',8:'6',10:'7'}
                                    pc_map = {
                                        'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,
                                        'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,
                                        'A':9,'A#':10,'Bb':10,'B':11,'Cb':11
                                    }
                                    tonic_pc = pc_map.get(tonic.split('/')[0],0)
                                    diff = (pc - tonic_pc) % 12
                                    if mode.lower().startswith('minor'):
                                        return minor_degrees.get(diff, '?')
                                    return major_degrees.get(diff, '?')
                                if not any(pitch_to_degree(p)==str(cval) for p in event_pitches):
                                    contains_ok = False
                                    break
                            else:
                                # exact MIDI
                                if cval not in event_pitches:
                                    contains_ok = False
                                    break
                        else:
                            # negative sentinel indicates pitch class (no octave)
                            pc = -cval - 1
                            if not any((p % 12) == pc for p in event_pitches):
                                contains_ok = False
                                break
                if not contains_ok:
                    continue
                # If start/end not provided, treat the whole melody as candidate
                if start_parsed is None and end_parsed is None:
                    sub_candidates.append(mel)
                    continue
                # Build lists of acceptable start and end MIDI values for this melody
                def acceptable_pitches(parsed: int) -> List[int]:
                    # parsed >=0 and <=127: exact MIDI or degree
                    if parsed is None:
                        return []
                    acc: List[int] = []
                    if parsed >= 0:
                        if 1 <= parsed <= 7:
                            # scale degree: compute all possible pitch classes of this degree
                            key_parts = mel.key.split()
                            tonic = key_parts[0] if key_parts else 'C'
                            mode = key_parts[1] if len(key_parts) > 1 else 'major'
                            # compute pc of degree diff
                            # Use mapping similar to _pitch_to_degree but invert mapping
                            major_mapping = {'1':0,'2':2,'3':4,'4':5,'5':7,'6':9,'7':11}
                            minor_mapping = {'1':0,'2':2,'3':3,'4':5,'5':7,'6':8,'7':10}
                            pc_map = {
                                'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,
                                'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,
                                'A':9,'A#':10,'Bb':10,'B':11,'Cb':11
                            }
                            tonic_pc = pc_map.get(tonic.split('/')[0],0)
                            if mode.lower().startswith('minor'):
                                pc = (tonic_pc + minor_mapping.get(str(parsed),'?')) % 12
                            else:
                                pc = (tonic_pc + major_mapping.get(str(parsed),'?')) % 12
                            # Accept any pitch with this pitch class
                            acc = [p for p in range(0,128) if (p % 12)==pc]
                        else:
                            # treat as exact MIDI
                            acc = [parsed]
                    else:
                        # negative sentinel: pitch class
                        pc = -parsed - 1
                        acc = [p for p in range(0,128) if (p % 12)==pc]
                    return acc
                start_acc = acceptable_pitches(start_parsed) if start_parsed is not None else []
                end_acc = acceptable_pitches(end_parsed) if end_parsed is not None else []
                # Now search for sub‑sequences within events
                n = len(events)
                for i in range(n):
                    # If start_acc specified, require starting pitch match
                    if start_acc and events[i].pitch not in start_acc:
                        continue
                    # Determine maximum j index
                    for j in range(i + min_between, min(n, i + max_between + 1)):
                        if end_acc and events[j].pitch not in end_acc:
                            continue
                        # found candidate sub‑melody
                        sub_events = events[i:j+1]
                        if not sub_events:
                            continue
                        # copy metadata
                        new_mel = Melody(
                            events=[NoteEvent(ev.pitch, ev.duration, ev.velocity) for ev in sub_events],
                            key=mel.key,
                            tempo=mel.tempo,
                            name=f"{mel.name}_sub_{i}_{j}",
                            complexity=mel.complexity,
                            avg_interval=mel.avg_interval,
                            span=mel.span,
                            note_density=mel.note_density,
                            register=mel.register,
                            metadata=mel.metadata,
                            file_path=mel.file_path,
                        )
                        # Recompute metadata for the sub‑melody
                        self._compute_metadata(new_mel)
                        sub_candidates.append(new_mel)
                # If no subsequences were added and start/end specified, we simply ignore this melody
            # Replace actual_candidates with the extracted sub‑melodies
            actual_candidates = sub_candidates
        # Randomly pick actual melodies up to requested number
        # Select actual melodies up to requested number
        if actual_candidates and num_actual > 0:
            sample_size = min(num_actual, len(actual_candidates))
            dataset.extend(random.sample(actual_candidates, sample_size))
        # Generate random melodies only if requested and there are slots available
        if num_random > 0:
            # Determine tonic and mode for random generator.  Use the provided
            # key and scale_type.  When key is 'any' or blank, select a
            # random tonic from the 12 semitones.  When scale_type is
            # 'any' or None, choose a random mode ('major' or 'minor').
            for _ in range(num_random):
                # Determine tonic
                if key and isinstance(key, str) and key.strip().lower() not in ('any', ''):
                    tonic_choice = key.strip().split()[0]
                else:
                    # Randomly pick one of the 12 chromatic roots
                    tonalities = ['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B']
                    tonic_choice = random.choice(tonalities)
                # Determine scale/mode
                if scale_type and isinstance(scale_type, str) and scale_type.strip().lower() not in ('any', ''):
                    mode_choice = scale_type.strip()
                else:
                    # Randomly choose between major and minor
                    mode_choice = random.choice(['major','minor'])
                # Choose a random length when variable_length is enabled
                rand_len = length
                if variable_length:
                    try:
                        rand_len = random.randint(min_length, max_length)
                    except Exception:
                        rand_len = length or 8
                dataset.append(self.generate_random_melody(
                    key=f"{tonic_choice}",
                    length=rand_len or 8,
                    tempo=tempo or random.randint(60, 160),
                    scale_type=mode_choice,
                    velocity_range=(50, 100),
                    max_interval=max_interval,
                    max_span=max_span,
                    register=register,
                    note_density_level=note_density_level,
                ))
        # Truncate melodies and optionally align to chord boundaries.  This
        # applies to all melodies in the dataset when they exceed the
        # desired length.  If ``variable_length`` is True, choose a
        # random length between ``min_length`` and ``max_length`` for each
        # melody; otherwise use the fixed ``length``.  When
        # ``align_to_chords`` is True and chord information is present,
        # start indices are chosen from chord boundaries; otherwise any
        # starting index that allows the segment to fit is used.  If a
        # melody is shorter than the desired length it is kept
        # unmodified.  Metadata is recomputed for truncated melodies.
        if dataset:
            new_dataset: List[Melody] = []
            for mel in dataset:
                # Determine desired length L for this melody
                if variable_length:
                    try:
                        L = random.randint(min_length, max_length)
                    except Exception:
                        L = length or len(mel.events)
                else:
                    L = length or len(mel.events)
                # Keep melodies shorter than or equal to L
                if len(mel.events) <= L:
                    new_dataset.append(mel)
                    continue
                # Gather candidate start indices
                start_indices: List[int] = []
                if align_to_chords and mel.chords:
                    # Compute cumulative onset times for notes
                    note_times: List[float] = [0.0]
                    for ev in mel.events[:-1]:
                        note_times.append(note_times[-1] + ev.duration)
                    # Compute cumulative start times for chords
                    chord_times: List[float] = []
                    acc = 0.0
                    for ch in mel.chords:
                        chord_times.append(acc)
                        acc += ch.get('duration', 0.0)
                    # Map chord start times to earliest note indices
                    for ct in chord_times:
                        for idx, t in enumerate(note_times):
                            if t >= ct and idx + L <= len(mel.events):
                                start_indices.append(idx)
                                break
                    if start_indices:
                        start_indices = sorted(set(start_indices))
                # Default to all positions if no chord-aligned indices
                if not start_indices:
                    start_indices = list(range(0, len(mel.events) - L + 1))
                start = random.choice(start_indices)
                end = start + L
                sub_events = [
                    NoteEvent(ev.pitch, ev.duration, ev.velocity)
                    for ev in mel.events[start:end]
                ]
                new_mel = Melody(
                    events=sub_events,
                    key=mel.key,
                    tempo=mel.tempo,
                    name=mel.name,
                    complexity=mel.complexity,
                    avg_interval=mel.avg_interval,
                    span=mel.span,
                    note_density=mel.note_density,
                    register=mel.register,
                    metadata=mel.metadata,
                    file_path=mel.file_path,
                    chords=mel.chords,
                )
                # Recompute metadata for truncated melody
                self._compute_metadata(new_mel)
                new_dataset.append(new_mel)
            dataset = new_dataset
        # We intentionally do NOT fill with additional random melodies if
        # there are not enough actual candidates; dataset length may be less
        # than requested.  Shuffle to mix random and actual melodies
        random.shuffle(dataset)
        return dataset

    # ------------------------------------------------------------------
    # Helper functions for scale and key filtering
    def _scale_pitch_classes(self, tonic: str, mode: str) -> Optional[set[int]]:
        """Return the set of pitch classes for the given tonic and mode.

        If music21 is available, use its scale classes to compute the pitch
        classes from the third octave (C3) up to C5 to capture a complete
        pattern.  If music21 is unavailable or the scale cannot be
        constructed, return a default major scale on C.  Returns None on
        failure.
        """
        try:
            if m21 is None:
                raise RuntimeError
            # Map friendly mode names to music21 scale class names
            mode_lower = mode.lower()
            scale_map = {
                'major': 'Major',
                'minor': 'Minor',
                'harmonic minor': 'HarmonicMinor',
                'melodic minor': 'MelodicMinor',
                'dorian': 'Dorian',
                'mixolydian': 'Mixolydian',
                'lydian': 'Lydian',
                'phrygian': 'Phrygian',
                'locrian': 'Locrian',
                'pentatonic': 'Pentatonic',
                'blues': 'Blues',
                'altered': 'Altered',
                'diminished': 'Diminished',
                'bebop major': 'Bebop',
                'bebop minor': 'MinorBebop',
            }
            # Handle 'any' mode by selecting major as default
            if mode_lower == 'any':
                mode_lower = 'major'
            cls_prefix = scale_map.get(mode_lower, mode.capitalize())
            try:
                scale_class = getattr(m21.scale, cls_prefix + 'Scale')
            except Exception:
                scale_class = getattr(m21.scale, cls_prefix, None)
            if scale_class is None:
                raise RuntimeError
            scale_obj = scale_class(m21.pitch.Pitch(tonic))
            pcs = set([p.midi % 12 for p in scale_obj.getPitches(tonic + '2', tonic + '5')])
            return pcs
        except Exception:
            # Fallback to C major pitch classes
            return set([0,2,4,5,7,9,11])

    def _notes_fit_scale(self, events: List[NoteEvent], tonic: str, mode: str) -> bool:
        """Return True if all pitch classes in events are contained in the given scale.

        Parameters
        ----------
        events : list of NoteEvent
            The melody events whose pitches should be tested.
        tonic : str
            The tonic of the scale (e.g. 'C', 'G#').  Ignored if None or empty.
        mode : str
            The scale/mode name (e.g. 'major', 'minor', etc.).  'Any' means
            accept the scale regardless of mode.
        Returns
        -------
        bool
            True if every pitch class in events is part of the scale, False otherwise.
        """
        if not events:
            return True
        # Compute pitch classes present in the melody
        pcs = set([ev.pitch % 12 for ev in events])
        # If mode is 'any', we accept both major and minor scales on this tonic
        if mode.lower() == 'any':
            for candidate_mode in ['major', 'minor']:
                sc = self._scale_pitch_classes(tonic, candidate_mode)
                if pcs.issubset(sc):
                    return True
            return False
        scale_pcs = self._scale_pitch_classes(tonic, mode)
        return pcs.issubset(scale_pcs)

    def _compute_metadata(self, melody: Melody) -> None:
        """Compute and update statistical metadata for a melody.

        This populates the melody's avg_interval, span, note_density,
        register and complexity fields.  The complexity is calculated as
        a simple combination of normalised interval, span and density metrics.

        Parameters
        ----------
        melody : Melody
            The Melody object to analyse and update.
        """
        events = melody.events
        if not events:
            melody.avg_interval = 0.0
            melody.span = 0
            melody.note_density = 0.0
            melody.register = ''
            melody.complexity = 0.0
            return
        # Average absolute interval between consecutive notes
        if len(events) > 1:
            intervals = [abs(events[i].pitch - events[i - 1].pitch) for i in range(1, len(events))]
            avg_interval = sum(intervals) / (len(events) - 1)
        else:
            avg_interval = 0.0
        melody.avg_interval = avg_interval
        # Span (range of pitches)
        pitches = [ev.pitch for ev in events]
        span = max(pitches) - min(pitches)
        melody.span = span
        # Average duration (quarter lengths)
        avg_duration = sum(ev.duration for ev in events) / len(events)
        # Normalise duration to compute density.  Longer durations yield lower density.
        # We treat 1.0 quarter length as the base; durations longer than 1.0 are capped.
        capped_dur = min(avg_duration, 1.0)
        density = 1.0 - capped_dur / 1.0
        melody.note_density = max(0.0, min(density, 1.0))
        # Register classification based on average pitch
        avg_pitch = sum(pitches) / len(pitches)
        if avg_pitch < 55:
            reg = 'bass'
        elif avg_pitch < 67:
            reg = 'middle'
        else:
            reg = 'treble'
        melody.register = reg
        # Normalise interval and span for complexity calculation
        max_interval_possible = 12.0  # semitones; typical maximum leap considered
        norm_interval = min(avg_interval / max_interval_possible, 1.0)
        max_span_possible = 24.0  # semitones; two octaves
        norm_span = min(span / max_span_possible, 1.0)
        # Combine metrics into a composite complexity score
        melody.complexity = (norm_interval + norm_span + melody.note_density) / 3.0

    def count_filtered(
        self,
        metadata_conditions: Optional[List[Tuple[str, str, List[str]]]] = None,
        metadata_filter: Optional[str] = None,
        length: Optional[int] = None,
        max_interval: Optional[int] = None,
        max_span: Optional[int] = None,
        register: Optional[str] = None,
        complexity_level: Optional[float] = None,
        note_density_level: Optional[float] = None,
    ) -> int:
        """Count how many actual melodies satisfy the given metadata and basic filters.

        This helper allows the GUI to display the number of melodies matching
        the metadata conditions without rebuilding the dataset entirely.  Only
        actual (non-random) melodies are considered.
        """
        count = 0
        for mel in self.actual_melodies:
            self._compute_metadata(mel)
            # Basic filters
            if length is not None:
                if not (0.5 * length <= len(mel.events) <= 1.5 * length):
                    continue
            if max_interval is not None and mel.avg_interval > max_interval:
                continue
            if max_span is not None and mel.span > max_span:
                continue
            if register and register.lower() not in ('any', '') and mel.register != register.lower():
                continue
            if complexity_level is not None and mel.complexity > complexity_level:
                continue
            if note_density_level is not None and mel.note_density > note_density_level:
                continue
            # Simple substring filter
            if metadata_filter:
                found = False
                filt = metadata_filter.lower()
                for k, v in (mel.metadata or {}).items():
                    try:
                        if filt in str(v).lower():
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    continue
            # Advanced conditions
            if metadata_conditions:
                passed = True
                for field, op, values in metadata_conditions:
                    # Retrieve value from metadata or built‑in fields
                    if field in mel.metadata:
                        meta_val = mel.metadata[field]
                    else:
                        # Built‑in fields
                        try:
                            if field == 'name':
                                meta_val = mel.name
                            elif field == 'key':
                                meta_val = mel.key
                            elif field == 'tempo':
                                meta_val = mel.tempo
                            elif field == 'complexity':
                                meta_val = mel.complexity
                            elif field == 'avg_interval':
                                meta_val = mel.avg_interval
                            elif field == 'span':
                                meta_val = mel.span
                            elif field == 'note_density':
                                meta_val = mel.note_density
                            elif field == 'register':
                                meta_val = mel.register
                            elif field == 'note_count':
                                meta_val = len(mel.events)
                            else:
                                passed = False
                                break
                        except Exception:
                            passed = False
                            break
                    try:
                        if isinstance(meta_val, list) and meta_val:
                            meta_val = meta_val[0]
                        if op in ('contains', 'equals'):
                            comp_val = str(meta_val).lower() if meta_val is not None else ''
                            target = str(values[0]).lower() if values else ''
                            if op == 'contains' and target not in comp_val:
                                passed = False
                                break
                            if op == 'equals' and comp_val != target:
                                passed = False
                                break
                        else:
                            num_val = float(meta_val)
                            if op in ('>=', '≥') and num_val < float(values[0]):
                                passed = False
                                break
                            if op in ('<=', '≤') and num_val > float(values[0]):
                                passed = False
                                break
                            if op == 'range':
                                low = float(values[0]) if values else float('-inf')
                                high = float(values[1]) if len(values) > 1 else float('inf')
                                if not (low <= num_val <= high):
                                    passed = False
                                    break
                    except Exception:
                        passed = False
                        break
                if not passed:
                    continue
            count += 1
        return count

    def metadata_fields(self) -> List[str]:
        """Return a sorted list of metadata keys present in the actual melodies.

        This helper is used by the GUI to populate the advanced filter dialog
        with all available metadata fields extracted from the dataset.
        """
        keys: set[str] = set()
        # Always provide these built‑in fields for filtering
        builtin = {
            'name', 'key', 'tempo', 'complexity', 'avg_interval', 'span',
            'note_density', 'register', 'note_count'
        }
        keys.update(builtin)
        for mel in self.actual_melodies:
            # Include any metadata keys extracted from H5
            if mel.metadata:
                keys.update(mel.metadata.keys())
        return sorted(keys)
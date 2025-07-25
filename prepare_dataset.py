"""
prepare_dataset.py
===================

This script processes a directory of MIDI or MusicXML files and
extracts melodies into individual segments that can be used by the
ear‑training application.  It reads every file in an input directory,
parses it using music21 and extracts a monophonic melody line.  By
default the highest note at each time position is chosen to represent
the melodic voice.  The script can optionally segment long melodies into
fixed‑length chunks (by number of notes) or groups of measures, computes
statistical metadata (average interval, span, note density, register
and a composite complexity score) for each segment, and writes the
results to a JSON file.  The resulting JSON can then be loaded by
``MelodyManager`` to avoid reparsing the MIDI files each time the app
starts.

Usage
-----

Run this script from the project root with Python.  For example:

.. code-block:: sh

    python -m ear_trainer.prepare_dataset \
        --input-dir /path/to/midi_files \
        --output-json processed_melodies.json \
        --segment-length 16

Parameters
----------

``--input-dir``
    Directory containing MIDI (.mid/.midi) or MusicXML (.xml/.musicxml) files.

``--output-json``
    Path to the JSON file to write.  The file will contain a list of
    dictionaries; each dictionary represents a melody segment with its
    note events and metadata.

``--segment-length`` (optional)
    Maximum number of notes per segment.  If omitted, each file will
    produce a single melody containing all its notes.

``--segment-measures`` (optional)
    When provided, split melodies by groups of measures instead of by
    note count.  The value specifies how many measures should be included
    in each segment (e.g. ``4`` for four-bar phrases).  If omitted or
    zero, segmentation by note count is used.

``--max-files`` (optional)
    Limit the number of files to process.  Useful for testing.

``--h5-dir`` (optional)
    Path to the root of a directory mirroring the MIDI directory but
    containing corresponding `.h5` files (such as the LMD `lmd_matched_h5` set).
    If provided, basic metadata (title, artist, year, artist terms, key and
    mode) will be extracted from the H5 files and attached to each segment
    in the JSON under the ``metadata`` field.  The segment also includes
    a ``file_path`` key with the full path to the original MIDI file.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any, Optional

try:
    import music21 as m21
except ImportError as exc:
    raise ImportError(
        "music21 is required to process MIDI files. Please install music21 before running this script."
    ) from exc


def compute_metadata(events: List[Dict[str, Any]]) -> Dict[str, float | str]:
    """Compute statistical metadata for a melody represented by a list of events.

    Each event should be a mapping with keys ``pitch`` (int) and ``duration`` (float).

    Returns a dict with keys ``avg_interval``, ``span``, ``note_density``, ``register`` and ``complexity``.
    """
    if not events:
        return {
            "avg_interval": 0.0,
            "span": 0,
            "note_density": 0.0,
            "register": "",
            "complexity": 0.0,
        }
    # Compute average absolute interval between consecutive pitches
    if len(events) > 1:
        intervals = [abs(events[i]["pitch"] - events[i - 1]["pitch"]) for i in range(1, len(events))]
        avg_interval = sum(intervals) / (len(events) - 1)
    else:
        avg_interval = 0.0
    # Compute span (range) across all pitches
    pitches = [e["pitch"] for e in events]
    span = max(pitches) - min(pitches)
    # Compute average duration (quarter lengths) and derive a density metric
    avg_duration = sum(e["duration"] for e in events) / len(events)
    capped_dur = min(avg_duration, 1.0)
    density = 1.0 - capped_dur / 1.0
    density = max(0.0, min(density, 1.0))
    # Classify register based on average pitch
    avg_pitch = sum(pitches) / len(pitches)
    if avg_pitch < 55:
        register = "bass"
    elif avg_pitch < 67:
        register = "middle"
    else:
        register = "treble"
    # Normalise interval and span to compute complexity
    max_interval_possible = 12.0
    norm_interval = min(avg_interval / max_interval_possible, 1.0)
    max_span_possible = 24.0
    norm_span = min(span / max_span_possible, 1.0)
    complexity = (norm_interval + norm_span + density) / 3.0
    return {
        "avg_interval": avg_interval,
        "span": span,
        "note_density": density,
        "register": register,
        "complexity": complexity,
    }


def segment_events(events: List[Dict[str, Any]], length: int) -> List[List[Dict[str, Any]]]:
    """Split a sequence of note events into segments of a given maximum length.

    If ``length`` is 0 or None, returns a single segment containing all events.
    """
    if not length or length <= 0:
        return [events]
    segments = []
    for i in range(0, len(events), length):
        seg = events[i : i + length]
        if seg:
            segments.append(seg)
    return segments


def segment_events_by_measures(events: List[Dict[str, Any]], bars: int, bar_qlen: float) -> List[List[Dict[str, Any]]]:
    """Split events into segments based on a number of measures.

    Parameters
    ----------
    events : list of dict
        Note events in chronological order.
    bars : int
        Number of measures per segment.  If 0, a single segment is returned.
    bar_qlen : float
        Length of one measure in quarter lengths.
    """
    if not bars or bars <= 0:
        return [events]
    target = bars * bar_qlen
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    acc = 0.0
    for ev in events:
        current.append(ev)
        acc += float(ev.get("duration", 0.0))
        if acc >= target:
            segments.append(current)
            current = []
            acc = 0.0
    if current:
        segments.append(current)
    return segments


def extract_melody_line(score: "m21.stream.Stream") -> List[Dict[str, Any]]:
    """Return a monophonic melody line extracted from ``score``.

    The highest sounding pitch at each time offset is chosen to represent the
    melody.  Durations are taken from the chordified representation.
    """
    events: List[Dict[str, Any]] = []
    try:
        chordified = score.chordify()
    except Exception:
        chordified = score.flat
    for element in chordified.flat.notes:
        if isinstance(element, m21.chord.Chord):
            pitch = max(p.midi for p in element.pitches)
        elif isinstance(element, m21.note.Note):
            pitch = element.pitch.midi
        else:
            continue
        dur = float(element.quarterLength)
        vel = int(getattr(element.volume, "velocity", 64) or 64)
        events.append({"pitch": pitch, "duration": dur, "velocity": vel})
    return events


def process_file(path: str, segment_length: int | None, segment_measures: int | None = 0) -> List[Dict[str, Any]]:
    """Parse a single MIDI or MusicXML file and return a list of processed melody segments.

    Parameters
    ----------
    path : str
        Path to the MIDI/MusicXML file.
    segment_length : int or None
        Number of notes per segment when ``segment_measures`` is 0.
    segment_measures : int or None
        Number of measures per segment.  When greater than zero this
        overrides ``segment_length``.

    Returns
    -------
    list of dict
        Each element is a dictionary with keys ``events``, ``key``, ``tempo``
        and computed metadata.
    """
    try:
        score = m21.converter.parse(path)
    except Exception as exc:
        print(f"Could not parse {os.path.basename(path)}: {exc}")
        return []
    # Extract a monophonic melody line using the highest sounding note at each
    # time point.
    events = extract_melody_line(score)
    if not events:
        return []
    # Key
    try:
        key_obj = score.analyze('key')
        key_name = f"{key_obj.tonic.name} {key_obj.mode}"
    except Exception:
        key_name = "C major"
    # Tempo (use first metronome mark if any).  Default to 120.
    tempo = 120
    try:
        tempos = score.metronomeMarkBoundaries()
        if tempos:
            # metronomeMarkBoundaries returns a list of tuples (offset, MetronomeMark)
            # The MetronomeMark.number attribute holds the BPM.  Use the first mark.
            tempo = int(float(tempos[0][1].number))
    except Exception:
        pass
    # Extract chord information from the score.  We chordify the score to
    # collapse simultaneous notes into chords and record each chord's
    # constituent pitches and duration.  Use try/except in case
    # chordification fails (e.g. music21 unavailable).
    chord_events: List[Dict[str, Any]] = []
    try:
        chord_score = score.chordify()
        for elem in chord_score.flat.notes:
            # Only include actual chords (more than one pitch) and treat single notes as chords for completeness
            pitches = [p.midi for p in getattr(elem, 'pitches', [])] if hasattr(elem, 'pitches') else []
            # Fallback: for Note objects, use its single pitch
            if not pitches and hasattr(elem, 'pitch'):
                pitches = [elem.pitch.midi]
            dur = float(elem.quarterLength)
            if pitches:
                chord_events.append({"pitches": pitches, "duration": dur})
    except Exception:
        chord_events = []
    # Determine bar length from the first time signature (default 4/4)
    bar_qlen = 4.0
    try:
        ts = score.recurse().getTimeSignatures()[0]
        bar_qlen = float(ts.barDuration.quarterLength)
    except Exception:
        pass
    if segment_measures and segment_measures > 0:
        segments = segment_events_by_measures(events, segment_measures, bar_qlen)
    else:
        segments = segment_events(events, segment_length)
    processed: List[Dict[str, Any]] = []
    for seg_idx, seg_events in enumerate(segments):
        meta = compute_metadata(seg_events)
        entry: Dict[str, Any] = {
            "events": seg_events,
            "key": key_name,
            "tempo": tempo,
            "name": f"{os.path.basename(path)}_seg{seg_idx}",
        }
        # Attach chord list to each segment; we do not currently align chords
        # to segments.  The full chord sequence is provided for later use in
        # segmentation or chord display.
        entry["chords"] = chord_events
        entry.update(meta)
        processed.append(entry)
    return processed


# -- CSV processing ---------------------------------------------------------

def process_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Parse a CSV file containing note sequences and metadata.

    The CSV is expected to have at least one column called 'notes' (or a
    similarly named column) containing a sequence of notes separated by
    spaces or commas.  Notes may be given as MIDI numbers (0–127) or
    letter names with optional octave (e.g. 'C4', 'G#3').  If a
    corresponding 'durations' column is present (or 'duration'), it
    should contain a sequence of numbers of equal length specifying the
    duration (in quarter lengths) of each note.  If absent, a default
    duration of 1.0 quarter length is applied to all notes.  Additional
    columns are treated as metadata and attached to each melody entry.

    Returns a list of entries, each with keys ``events``, ``key``, ``tempo``,
    ``name`` and ``metadata``.  Durations are floats.  Tempo is taken
    from a column called 'tempo' (if present) or defaults to 120.  Key
    is taken from a column called 'key' (if present) or defaults to
    'C major'.  The entry 'name' is taken from a column called 'title'
    or 'name' (if present) or the CSV row index.
    """
    import csv as _csv
    entries: List[Dict[str, Any]] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        # Determine the note and duration columns heuristically
        note_col = None
        dur_col = None
        # Standard column names we try for notes and durations
        note_candidates = ['notes', 'note_sequence', 'notes_extracted', 'pitch', 'pitches']
        duration_candidates = ['durations', 'duration', 'duration_sequence']
        # Determine columns on first row
        headers = reader.fieldnames or []
        for c in note_candidates:
            if c in headers:
                note_col = c
                break
        for c in duration_candidates:
            if c in headers:
                dur_col = c
                break
        # Helper to parse note string to MIDI integer
        def parse_note_str(s: str) -> Optional[int]:
            s = s.strip()
            if not s:
                return None
            # try integer
            try:
                val = int(s)
                if 0 <= val <= 127:
                    return val
            except Exception:
                pass
            # parse note name with optional octave
            import re
            m = re.match(r'^([A-Ga-g][#b]?)([-]?\d*)$', s)
            if m:
                pc, oct_str = m.groups()
                pc_map = {
                    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4,
                    'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8, 'AB': 8, 'A': 9,
                    'A#': 10, 'BB': 10, 'B': 11, 'CB': 11
                }
                pc_val = pc_map.get(pc.upper())
                if pc_val is None:
                    return None
                if oct_str:
                    try:
                        octv = int(oct_str)
                    except Exception:
                        return None
                    return pc_val + (octv + 1) * 12
                else:
                    # If octave missing, default to C4 range (use octave 4)
                    return pc_val + (4 + 1) * 12
            return None
        row_index = 0
        for row in reader:
            row_index += 1
            # Extract notes
            if note_col and row.get(note_col):
                raw = row[note_col]
            else:
                # fallback: search for first non-empty field containing numbers or letters separated by spaces/commas
                raw = None
                for field, val in row.items():
                    if isinstance(val, str) and (',' in val or ' ' in val):
                        raw = val
                        break
            if not raw:
                continue
            # Split by comma or space
            tokens = [tok for tok in raw.replace(',', ' ').split() if tok]
            pitches: List[int] = []
            for tok in tokens:
                midi = parse_note_str(tok)
                if midi is not None:
                    pitches.append(midi)
            if not pitches:
                continue
            # Durations
            durations: List[float] = []
            if dur_col and row.get(dur_col):
                rawd = row[dur_col]
                toks = [tok for tok in rawd.replace(',', ' ').split() if tok]
                for tok in toks:
                    try:
                        val = float(tok)
                    except Exception:
                        val = 1.0
                    durations.append(val)
            # Ensure durations list matches pitches; if not, pad or trim
            if durations and len(durations) != len(pitches):
                # Pad with last duration or truncate
                if len(durations) < len(pitches):
                    last = durations[-1]
                    durations.extend([last] * (len(pitches) - len(durations)))
                else:
                    durations = durations[:len(pitches)]
            elif not durations:
                durations = [1.0] * len(pitches)
            # Build events
            events = []
            for p, d in zip(pitches, durations):
                events.append({"pitch": p, "duration": float(d), "velocity": 64})
            # Determine key and tempo from columns
            key = row.get('key') or row.get('Key') or 'C major'
            tempo_str = row.get('tempo') or row.get('Tempo') or row.get('bpm')
            try:
                tempo = int(float(tempo_str)) if tempo_str else 120
            except Exception:
                tempo = 120
            # Name/title
            name = row.get('title') or row.get('Title') or row.get('name') or f"Row{row_index}"
            # Metadata: include all fields except notes and durations
            metadata = {k: v for k, v in row.items() if k not in {note_col, dur_col}}
            # Compute statistics
            meta_stats = compute_metadata(events)
            entry: Dict[str, Any] = {
                "events": events,
                "key": key,
                "tempo": tempo,
                "name": name,
            }
            entry.update(meta_stats)
            if metadata:
                entry["metadata"] = metadata
            entries.append(entry)
    return entries


# -- CSV + MIDI processing ----------------------------------------------------
def process_csv_with_midi(csv_path: str, midi_dir: str, segment_length: int | None, segment_measures: int | None = 0) -> List[Dict[str, Any]]:
    """Parse a CSV file referencing MIDI files and combine the metadata.

    This helper is designed to support datasets where a CSV file lists
    metadata (e.g. file name, unique notes, sequence length) and the
    corresponding MIDI files reside in a separate directory.  Each row
    must contain a column called ``Name`` (or ``name``) identifying the
    MIDI file (with or without extension).  The function will attempt
    to locate the MIDI file in ``midi_dir`` and parse it via
    ``music21`` to extract note events, tempo and key.  Additional
    columns (such as ``Unique_notes`` or ``len_Uni_Notes``) are added as
    metadata on the resulting entry.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing metadata.
    midi_dir : str
        Directory containing MIDI files referenced by the CSV.
    segment_length : int or None
        Maximum number of notes to include in each entry when
        ``segment_measures`` is not used.  If 0 or None, the entire MIDI
        file is used.
    segment_measures : int or None
        When greater than zero, split the MIDI file into segments of the
        specified number of measures.  Only the first resulting segment is
        used for each file in the CSV.

    Returns
    -------
    list of dict
        A list of dictionaries compatible with ``MelodyManager.load_from_json``.
    """
    import csv as _csv
    entries: List[Dict[str, Any]] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        row_index = 0
        for row in reader:
            row_index += 1
            # Determine MIDI file name; try 'Name' then 'name'
            fname = row.get('Name') or row.get('name') or ''
            fname = fname.strip() if isinstance(fname, str) else ''
            if not fname:
                # Skip rows without a valid file name
                continue
            # Some file names may include extension; if not, try both .mid and .midi
            possible_paths = []
            base = fname
            # If extension present, use as‑is
            if os.path.splitext(base)[1]:
                possible_paths.append(os.path.join(midi_dir, base))
            else:
                possible_paths.append(os.path.join(midi_dir, base + '.mid'))
                possible_paths.append(os.path.join(midi_dir, base + '.midi'))
                possible_paths.append(os.path.join(midi_dir, base + '.MID'))
            midi_path = None
            for p in possible_paths:
                if os.path.isfile(p):
                    midi_path = p
                    break
            if not midi_path:
                print(f"Warning: MIDI file for row {row_index} ('{fname}') not found in {midi_dir}.")
                continue
            # Parse the MIDI file using process_file helper.  We request no
            # segmentation here so that the entire file is loaded and then
            # optionally truncated or segmented below.
            file_entries = process_file(midi_path, 0, segment_measures)
            if not file_entries:
                continue
            # Use the first entry (only one, because segment_length=0)
            entry = file_entries[0]
            # Optionally truncate events to segment_length if provided and >0
            # only when measure-based segmentation is not active.
            if segment_length and segment_length > 0 and not segment_measures:
                entry['events'] = entry['events'][:segment_length]
                # Recompute metadata for truncated events
                meta_stats = compute_metadata(entry['events'])
                entry.update(meta_stats)
            # Name: use the row's Name or fallback to file name
            entry['name'] = fname
            # Attach original file path
            entry['file_path'] = midi_path
            # Create a metadata dictionary from all CSV columns except
            # the file name and note sequence columns (we don't know
            # durations; we rely on the MIDI file).  Include the raw
            # values as strings.
            meta_dict: Dict[str, Any] = {}
            for k, v in row.items():
                if k in ('Name', 'name'):
                    continue
                meta_dict[k] = v
            if meta_dict:
                entry.setdefault('metadata', {}).update(meta_dict)
            # Append this entry
            entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a directory of MIDI/MusicXML files into a melody JSON dataset.")
    parser.add_argument("--input-dir", required=False, default=None, help="Directory containing MIDI/MusicXML files to process. Not required when --csv-file is used.")
    parser.add_argument("--output-json", required=True, help="Path to write the resulting JSON dataset.")
    parser.add_argument("--segment-length", type=int, default=0, help="Maximum number of notes per segment (0 for no segmentation).")
    parser.add_argument("--segment-measures", type=int, default=0, help="Number of measures per segment when splitting by bars (0 to disable).")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum number of files to process (0 for all).")
    parser.add_argument("--h5-dir", default=None, help="Directory containing corresponding .h5 metadata files (optional). If provided, metadata from matching h5 files will be attached to each segment.")
    # Optional CSV file input.  When provided, the script processes the CSV
    # instead of scanning a directory of MIDI files.  The CSV must contain
    # at least a 'notes' column.  Other columns are added as metadata.
    parser.add_argument("--csv-file", default=None, help="Path to a CSV file with note sequences and metadata.")
    parser.add_argument("--midi-dir", default=None, help="Directory containing MIDI files referenced in the CSV. Only used when --csv-file is provided.")
    args = parser.parse_args()
    in_dir: Optional[str] = args.input_dir
    out_path: str = args.output_json
    seg_len: int = args.segment_length
    seg_measures: int = args.segment_measures
    max_files: int = args.max_files
    h5_dir: Optional[str] = args.h5_dir
    csv_file: Optional[str] = args.csv_file
    all_entries: List[Dict[str, Any]] = []
    # If a CSV file is provided, process it.  If a MIDI directory is also
    # provided, combine the CSV metadata with the actual MIDI contents using
    # process_csv_with_midi.  Otherwise, fall back to reading note sequences
    # directly from the CSV via process_csv.
    if csv_file:
        try:
            if args.midi_dir:
                all_entries = process_csv_with_midi(csv_file, args.midi_dir, seg_len, seg_measures)
            else:
                all_entries = process_csv(csv_file)
        except Exception as exc:
            print(f"Failed to process CSV {csv_file}: {exc}")
            all_entries = []
    else:
        # No CSV provided; process files in a directory if provided
        processed_count = 0
        # Only scan the input directory if it is provided
        if in_dir:
            # Recursively walk through the input directory so that nested directories
            # containing MIDI files are processed.  The original version only
            # listed the top‑level directory, which meant collections like the
            # Lakh MIDI Dataset (which have deep folder structures) were ignored.
            for root, dirs, files in os.walk(in_dir):
                for fname in files:
                    lower = fname.lower()
                    if not (lower.endswith(".mid") or lower.endswith(".midi") or lower.endswith(".xml") or lower.endswith(".musicxml")):
                        continue
                    if max_files and processed_count >= max_files:
                        break
                    path = os.path.join(root, fname)
                    entries = process_file(path, seg_len, seg_measures)
                    # Attach file_path and metadata to each segment
                    if entries:
                        for seg in entries:
                            # Store the absolute file path for later reference
                            seg["file_path"] = path
                            # Attempt to attach metadata from matching H5 file.  Only do
                            # this if an H5 directory was provided and the H5 file exists.
                            if h5_dir:
                                # Compute h5 path by replacing the input directory prefix and extension
                                rel = os.path.relpath(path, in_dir)
                                base = os.path.splitext(rel)[0]
                                h5_path = os.path.join(h5_dir, base + ".h5")
                                if os.path.isfile(h5_path):
                                    # Attempt to read metadata from the corresponding HDF5 file.  We wrap
                                    # the entire block in a try/except so that errors during HDF5
                                    # reading do not abort processing of other files.
                                    try:
                                        import h5py  # deferred import, may not be available
                                        with h5py.File(h5_path, 'r') as hf:
                                            meta_dict: Dict[str, Any] = {}
                                            # Helper to safely convert arbitrary HDF5 data types to
                                            # Python scalars or lists.  Handles bytes, numpy scalars
                                            # and arrays.  If a structured array is provided, its
                                            # fields are flattened into a dict (field -> value).
                                            def to_python(value: Any) -> Any:
                                                import numpy as _np  # local import
                                                # Bytes to string
                                                if isinstance(value, (bytes, bytearray)):
                                                    try:
                                                        return value.decode('utf-8', 'ignore')
                                                    except Exception:
                                                        return value
                                                # Numpy scalar
                                                if hasattr(value, 'item') and not isinstance(value, (list, tuple)):
                                                    try:
                                                        return value.item()
                                                    except Exception:
                                                        pass
                                                # Numpy array
                                                if isinstance(value, _np.ndarray):
                                                    if value.dtype.names:
                                                        # Structured array: return dict of field values
                                                        rec = value[0] if value.shape[0] > 0 else None
                                                        if rec is not None:
                                                            d = {}
                                                            for name in value.dtype.names:
                                                                try:
                                                                    d[name] = to_python(rec[name])
                                                                except Exception:
                                                                    pass
                                                            return d
                                                    # Convert to list of scalars
                                                    try:
                                                        return value.tolist()
                                                    except Exception:
                                                        pass
                                                # Python list or tuple: convert elements
                                                if isinstance(value, (list, tuple)):
                                                    return [to_python(v) for v in value]
                                                return value
                                            # Recursively walk through all groups and datasets under
                                            # the root of the H5 file.  We accumulate metadata
                                            # entries keyed by their dataset name.  For structured
                                            # datasets we flatten the fields.
                                            def visit_item(name: str, obj: Any) -> None:
                                                # Skip the root group itself
                                                if name == '':
                                                    return
                                                try:
                                                    if isinstance(obj, h5py.Dataset):
                                                        data = obj[()]
                                                        py = to_python(data)
                                                        # If the conversion yields a dict (from a
                                                        # structured array), merge its fields with a
                                                        # prefix based on the dataset name
                                                        if isinstance(py, dict):
                                                            for k2, v2 in py.items():
                                                                # Build key as datasetname_field
                                                                meta_dict[f"{name}:{k2}"] = v2
                                                        else:
                                                            meta_dict[name] = py
                                                except Exception:
                                                    pass
                                            # Visit all items
                                            try:
                                                hf.visititems(visit_item)
                                            except Exception:
                                                pass
                                            # Remove None values
                                            meta_dict = {k: v for k, v in meta_dict.items() if v is not None}
                                            if meta_dict:
                                                seg["metadata"] = meta_dict
                                                # Override segment tempo using the best available field.  Try
                                                # several possible keys.  Flattened keys may include
                                                # names like 'analysis:tempo' or 'analysis/tempo'.  Also
                                                # consider top‑level 'tempo'.
                                                tempo_val: Any = None
                                                for tkey in [
                                                    'analysis:tempo', 'analysis/tempo',
                                                    'analysis:songs_tempo', 'analysis:songs:tempo', 'analysis:songs/tempo',
                                                    'analysis:songs:analysis_tempo', 'analysis:tempo_confidence',
                                                    'analysis:bars_start_tempo'
                                                ]:
                                                    if tkey in meta_dict:
                                                        tempo_val = meta_dict[tkey]
                                                        break
                                                if tempo_val is None and 'tempo' in meta_dict:
                                                    tempo_val = meta_dict['tempo']
                                                # Convert tempo to integer BPM if possible
                                                if tempo_val is not None:
                                                    try:
                                                        if isinstance(tempo_val, list) and tempo_val:
                                                            tempo_val = tempo_val[0]
                                                        if isinstance(tempo_val, dict) and 'tempo' in tempo_val:
                                                            tempo_val = tempo_val['tempo']
                                                        if isinstance(tempo_val, (int, float)):
                                                            seg['tempo'] = int(float(tempo_val))
                                                        elif isinstance(tempo_val, str) and tempo_val.replace('.', '', 1).isdigit():
                                                            seg['tempo'] = int(float(tempo_val))
                                                    except Exception:
                                                        pass
                                    except Exception:
                                        # If H5 parsing fails, ignore silently
                                        pass
                    if entries:
                        all_entries.extend(entries)
                    processed_count += 1
                    print(f"Processed {os.path.relpath(path, in_dir)}: {len(entries)} segments")
                # Break out of outer loop if max_files reached
                if max_files and processed_count >= max_files:
                    break
    # Write JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2)
    print(f"Wrote {len(all_entries)} melody segments to {out_path}")


if __name__ == "__main__":
    main()
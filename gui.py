"""
gui.py
======

Defines the graphical user interface for the ear training application.
The GUI is built using the PySide6 toolkit and ties together the melody
management, audio playback and pitch detection components.  Users can
configure the mix of random and actual melodies, select keys and scales,
choose audio input and output devices, and enable automatic note detection
via MIDI or audio.  Playback controls allow manual navigation through the
dataset, while visual feedback indicates the current melody and the
outcome of automatic detection.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt

try:
    import mido
except ImportError:
    mido = None
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QComboBox,
    QSlider,
    QSpinBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QTextEdit,
    QGroupBox,
    QLineEdit,
    QDialog,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox as QtSpinBox,
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen, QBrush

try:
    import sounddevice as sd
except ImportError:
    sd = None

from melody_manager import MelodyManager, Melody
from audio_player import AudioPlayer
from pitch_detector import MidiDetector, AudioDetector


class EarTrainerGUI(QMainWindow):
    """Main window for the ear training app."""

    def __init__(
        self,
        manager: MelodyManager,
        player: AudioPlayer,
        midi_detector: Optional[MidiDetector] = None,
        audio_detector: Optional[AudioDetector] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ear Trainer")
        self.manager = manager
        self.player = player
        self.midi_detector = midi_detector
        self.audio_detector = audio_detector
        # Dataset and playback state
        self.dataset: List[Melody] = []
        self.current_index: int = 0
        # Advanced metadata conditions
        self.metadata_conditions: List[tuple] = []
        # Octave shift state (in semitone increments of 12)
        self.octave_shift: int = 0
        # Set of currently highlighted pitches (MIDI numbers) for keyboard animation
        self.active_pitches: set[int] = set()
        # Build the UI
        self._init_ui()
        # Build initial dataset
        self._rebuild_dataset()

    def _init_ui(self) -> None:
        # Central widget and main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # Data settings group
        data_group = QGroupBox("Dataset Settings")
        data_layout = QHBoxLayout()
        data_group.setLayout(data_layout)

        # Random ratio slider (0–100)
        self.random_slider = QSlider(Qt.Orientation.Horizontal)
        self.random_slider.setRange(0, 100)
        self.random_slider.setValue(80)
        self.random_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.random_slider.setTickInterval(10)
        data_layout.addWidget(QLabel("Random %:"))
        data_layout.addWidget(self.random_slider)

        # Dataset size spin box
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 1000)
        self.size_spin.setValue(100)
        data_layout.addWidget(QLabel("Size:"))
        data_layout.addWidget(self.size_spin)

        # Key selection
        self.key_combo = QComboBox()
        # A simple list of tonic names; include 'Any' to disable key constraint for random melodies
        keys = ["Any", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.key_combo.addItems(keys)
        data_layout.addWidget(QLabel("Key:"))
        data_layout.addWidget(self.key_combo)

        # Scale selection; include 'Any' to allow random scales and additional modes
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "Any", "major", "minor", "dorian", "mixolydian", "lydian",
            "phrygian", "locrian", "harmonic minor", "melodic minor",
            "blues", "pentatonic", "altered", "diminished", "bebop major", "bebop minor"
        ])
        data_layout.addWidget(QLabel("Scale:"))
        data_layout.addWidget(self.scale_combo)

        # Metadata filter (applies only to actual melodies).  If a keyword
        # is entered, actual melodies whose metadata contain the keyword
        # (case‑insensitive) will be included.  Leave blank to include all.
        self.metadata_edit = QLineEdit()
        self.metadata_edit.setPlaceholderText("filter metadata…")
        data_layout.addWidget(QLabel("Metadata filter:"))
        data_layout.addWidget(self.metadata_edit)

        # Button to open advanced metadata filter dialog.  This will
        # construct a detailed filter across metadata fields and display
        # the count of matching melodies.
        self.advanced_filter_btn = QPushButton("Advanced Filters…")
        data_layout.addWidget(self.advanced_filter_btn)

        # Sequence filter group: allows specifying start/end notes and
        # note containment constraints.  This group is hidden by default
        # to avoid clutter; click the toggle to show/hide.  The filters
        # apply only to actual melodies.
        self.content_group = QGroupBox("Sequence Filter")
        self.content_group.setCheckable(True)
        self.content_group.setChecked(False)
        content_layout = QHBoxLayout()
        self.content_group.setLayout(content_layout)
        # Start note
        self.start_note_edit = QLineEdit()
        self.start_note_edit.setPlaceholderText("Start note (e.g. C4 or 1)")
        content_layout.addWidget(QLabel("Start:"))
        content_layout.addWidget(self.start_note_edit)
        # End note
        self.end_note_edit = QLineEdit()
        self.end_note_edit.setPlaceholderText("End note (e.g. G4 or 5)")
        content_layout.addWidget(QLabel("End:"))
        content_layout.addWidget(self.end_note_edit)
        # Min notes between
        self.min_between_spin = QSpinBox()
        self.min_between_spin.setRange(0, 32)
        self.min_between_spin.setValue(0)
        content_layout.addWidget(QLabel("Min between:"))
        content_layout.addWidget(self.min_between_spin)
        # Max notes between
        self.max_between_spin = QSpinBox()
        self.max_between_spin.setRange(0, 64)
        self.max_between_spin.setValue(32)
        content_layout.addWidget(QLabel("Max between:"))
        content_layout.addWidget(self.max_between_spin)
        # Contains notes
        self.contains_edit = QLineEdit()
        self.contains_edit.setPlaceholderText("Contains (e.g. C4,G4 or 3,6)")
        content_layout.addWidget(QLabel("Contains:"))
        content_layout.addWidget(self.contains_edit)
        main_layout.addWidget(self.content_group)

        main_layout.addWidget(data_group)

        # Random melody parameter group
        param_group = QGroupBox("Melody Parameters")
        param_layout = QHBoxLayout()
        param_group.setLayout(param_layout)

        # Complexity slider (0–100)
        self.complexity_slider = QSlider(Qt.Orientation.Horizontal)
        self.complexity_slider.setRange(0, 100)
        self.complexity_slider.setValue(50)
        self.complexity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.complexity_slider.setTickInterval(10)
        param_layout.addWidget(QLabel("Complexity:"))
        param_layout.addWidget(self.complexity_slider)

        # Length spin box (# of notes)
        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 64)
        self.length_spin.setValue(8)
        param_layout.addWidget(QLabel("Length:"))
        param_layout.addWidget(self.length_spin)

        # Tempo spin box (BPM)
        self.tempo_spin = QSpinBox()
        # Allow 'Any' tempo by permitting 0 as the special value.  When 0 is selected,
        # the dataset builder will interpret it as a request for random tempo.
        self.tempo_spin.setRange(0, 240)
        self.tempo_spin.setSpecialValueText("Any")
        self.tempo_spin.setValue(0)
        param_layout.addWidget(QLabel("Tempo:"))
        param_layout.addWidget(self.tempo_spin)

        # Note density slider
        self.density_slider = QSlider(Qt.Orientation.Horizontal)
        self.density_slider.setRange(0, 100)
        self.density_slider.setValue(50)
        self.density_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.density_slider.setTickInterval(10)
        param_layout.addWidget(QLabel("Density:"))
        param_layout.addWidget(self.density_slider)

        # Maximum interval (semitones)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 24)
        self.interval_spin.setValue(12)
        param_layout.addWidget(QLabel("Max interval:"))
        param_layout.addWidget(self.interval_spin)

        # Maximum span (octaves) spin box
        self.span_spin = QSpinBox()
        self.span_spin.setRange(1, 6)
        self.span_spin.setValue(2)
        param_layout.addWidget(QLabel("Max span (octaves):"))
        param_layout.addWidget(self.span_spin)

        # Register selection
        self.register_combo = QComboBox()
        self.register_combo.addItems(["Any", "Bass", "Middle", "Treble"])
        param_layout.addWidget(QLabel("Register:"))
        param_layout.addWidget(self.register_combo)

        main_layout.addWidget(param_group)

        # Segment options group: control segmentation behaviour.  When
        # variable length is enabled, melodies are truncated to a random
        # length between the specified minimum and maximum.  Otherwise a
        # fixed length is used from the Length spin box.
        seg_group = QGroupBox("Segment Options")
        seg_layout = QHBoxLayout()
        seg_group.setLayout(seg_layout)
        self.variable_length_check = QCheckBox("Variable length")
        seg_layout.addWidget(self.variable_length_check)
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 64)
        self.min_length_spin.setValue(4)
        seg_layout.addWidget(QLabel("Min len:"))
        seg_layout.addWidget(self.min_length_spin)
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(1, 64)
        self.max_length_spin.setValue(16)
        seg_layout.addWidget(QLabel("Max len:"))
        seg_layout.addWidget(self.max_length_spin)
        # Alignment to chords reserved for future use
        self.align_chords_check = QCheckBox("Align to chords")
        self.align_chords_check.setToolTip("Start segments at chord boundaries (not yet implemented)")
        seg_layout.addWidget(self.align_chords_check)
        main_layout.addWidget(seg_group)

        # Detection settings group
        detect_group = QGroupBox("Detection Settings")
        detect_layout = QHBoxLayout()
        detect_group.setLayout(detect_layout)

        self.auto_check = QCheckBox("Enable auto detection")
        detect_layout.addWidget(self.auto_check)

        # Detection mode selection
        self.detect_mode = QComboBox()
        self.detect_mode.addItems(["Manual", "MIDI", "Audio"])
        detect_layout.addWidget(QLabel("Mode:"))
        detect_layout.addWidget(self.detect_mode)

        # Input device combobox (for MIDI or Audio depending on mode)
        self.input_combo = QComboBox()
        detect_layout.addWidget(QLabel("Input device:"))
        detect_layout.addWidget(self.input_combo)

        main_layout.addWidget(detect_group)

        # Transposition and Tempo adjustment group
        transpose_group = QGroupBox("Force Transpose/Tempo")
        transpose_layout = QHBoxLayout()
        transpose_group.setLayout(transpose_layout)
        # Force key checkbox and selectors
        self.force_key_check = QCheckBox("Force key")
        transpose_layout.addWidget(self.force_key_check)
        self.force_key_combo = QComboBox()
        # Include an 'Any' option to allow forcing only the scale or only the tonic
        key_list = ["Any", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.force_key_combo.addItems(key_list)
        self.force_scale_combo = QComboBox()
        # Include 'Any' for scale so the user can choose to only force the key
        self.force_scale_combo.addItems([
            "Any", "major", "minor", "dorian", "mixolydian", "lydian",
            "phrygian", "locrian", "harmonic minor", "melodic minor",
            "blues", "pentatonic", "altered", "diminished", "bebop major", "bebop minor"
        ])
        transpose_layout.addWidget(self.force_key_combo)
        transpose_layout.addWidget(self.force_scale_combo)
        # Force tempo checkbox and spin
        self.force_tempo_check = QCheckBox("Force tempo")
        transpose_layout.addWidget(self.force_tempo_check)
        self.force_tempo_spin = QSpinBox()
        # Allow forcing tempo to any BPM between 40 and 240.  A value of 0 is unused
        # for forcing tempo, so keep the minimum at 40.
        self.force_tempo_spin.setRange(40, 240)
        self.force_tempo_spin.setValue(120)
        transpose_layout.addWidget(self.force_tempo_spin)
        main_layout.addWidget(transpose_group)

        # Playback controls group
        controls_group = QGroupBox("Playback Controls")
        controls_layout = QHBoxLayout()
        controls_group.setLayout(controls_layout)

        self.play_button = QPushButton("Play")
        controls_layout.addWidget(self.play_button)
        self.prev_button = QPushButton("Prev")
        controls_layout.addWidget(self.prev_button)
        self.next_button = QPushButton("Next")
        controls_layout.addWidget(self.next_button)

        # Instrument selection
        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems(["Sine", "Grand Piano", "Electric Piano", "Trumpet", "Saxophone"])
        controls_layout.addWidget(QLabel("Instrument:"))
        controls_layout.addWidget(self.instrument_combo)

        # MIDI output selection
        self.midi_out_combo = QComboBox()
        self._update_midi_outputs()
        self._on_midi_out_changed()
        controls_layout.addWidget(QLabel("MIDI Out:"))
        controls_layout.addWidget(self.midi_out_combo)

        # Octave shift buttons
        self.octave_down_btn = QPushButton("Octave -")
        controls_layout.addWidget(self.octave_down_btn)
        self.octave_up_btn = QPushButton("Octave +")
        controls_layout.addWidget(self.octave_up_btn)

        main_layout.addWidget(controls_group)

        # Display group: show note names and messages
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()
        display_group.setLayout(display_layout)

        # Display control checkboxes: note names, key info, staff and keyboard
        self.show_notes_check = QCheckBox("Show note names")
        self.show_notes_check.setChecked(True)
        display_layout.addWidget(self.show_notes_check)
        self.show_key_check = QCheckBox("Show key/tempo info")
        self.show_key_check.setChecked(True)
        display_layout.addWidget(self.show_key_check)
        self.show_staff_check = QCheckBox("Show staff")
        self.show_staff_check.setChecked(True)
        display_layout.addWidget(self.show_staff_check)
        self.show_keyboard_check = QCheckBox("Show keyboard")
        self.show_keyboard_check.setChecked(False)
        display_layout.addWidget(self.show_keyboard_check)

        # Font size selector for the display text.  Users can adjust
        # this spin box to make the note names and metadata larger or
        # smaller.  A range of 8–24 points is provided.
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(12)
        display_layout.addWidget(QLabel("Font size:"))
        display_layout.addWidget(self.font_size_spin)

        # Text area to display notes and messages
        self.display_area = QTextEdit()
        self.display_area.setReadOnly(True)
        # Limit height to prevent overlapping the staff; enable scroll bars
        self.display_area.setMaximumHeight(200)
        display_layout.addWidget(self.display_area)
        # Staff notation display: this label will show a drawn staff with note
        # heads corresponding to the current melody.  It is updated
        # dynamically in _update_display().
        self.staff_label = QLabel()
        # Ensure the label expands horizontally but has sufficient width and height
        # to display notes and ledger lines without clipping.  Reserve a
        # minimum width of 600px and height of 250px.
        self.staff_label.setMinimumSize(600, 250)
        display_layout.addWidget(self.staff_label)

        # Keyboard display: show a simple keyboard highlighting notes.  This
        # is updated in _update_display() when enabled.  Increase the
        # minimum width and height so that two octaves can be displayed
        # without clipping.  The width will scale with the window.
        self.keyboard_label = QLabel()
        # Reserve space for roughly 24 keys (each 25px wide) and a height of 150
        self.keyboard_label.setMinimumSize(600, 150)
        display_layout.addWidget(self.keyboard_label)

        main_layout.addWidget(display_group)

        # Status label
        self.status_label = QLabel()
        main_layout.addWidget(self.status_label)

        # Connect signals
        self.random_slider.valueChanged.connect(self._rebuild_dataset)
        self.size_spin.valueChanged.connect(self._rebuild_dataset)
        self.key_combo.currentIndexChanged.connect(self._rebuild_dataset)
        self.scale_combo.currentIndexChanged.connect(self._rebuild_dataset)
        self.play_button.clicked.connect(self._on_play_clicked)
        self.next_button.clicked.connect(self._on_next_clicked)
        self.prev_button.clicked.connect(self._on_prev_clicked)
        self.auto_check.stateChanged.connect(self._on_auto_changed)
        self.detect_mode.currentIndexChanged.connect(self._on_mode_changed)
        # Melody parameter controls update dataset when changed
        self.complexity_slider.valueChanged.connect(self._rebuild_dataset)
        self.length_spin.valueChanged.connect(self._rebuild_dataset)
        self.tempo_spin.valueChanged.connect(self._rebuild_dataset)
        self.density_slider.valueChanged.connect(self._rebuild_dataset)
        self.interval_spin.valueChanged.connect(self._rebuild_dataset)
        self.span_spin.valueChanged.connect(self._rebuild_dataset)
        self.register_combo.currentIndexChanged.connect(self._rebuild_dataset)
        # Metadata filter
        self.metadata_edit.textChanged.connect(self._rebuild_dataset)
        # Instrument selection
        self.instrument_combo.currentIndexChanged.connect(self._on_instrument_changed)
        self.midi_out_combo.currentIndexChanged.connect(self._on_midi_out_changed)
        # Octave shift buttons
        self.octave_down_btn.clicked.connect(self._on_octave_down)
        self.octave_up_btn.clicked.connect(self._on_octave_up)
        # Populate input devices based on default mode (Manual)
        self._update_input_devices()

        # Advanced metadata filter dialog
        self.advanced_filter_btn.clicked.connect(self._open_filter_dialog)
        # Show/hide display elements
        self.show_notes_check.stateChanged.connect(lambda _: self._update_display())
        self.show_key_check.stateChanged.connect(lambda _: self._update_display())
        self.show_staff_check.stateChanged.connect(lambda _: self._update_display())
        self.show_keyboard_check.stateChanged.connect(lambda _: self._update_display())
        # Font size change updates text display immediately
        self.font_size_spin.valueChanged.connect(lambda _: self._update_display())
        # Content filter interactions
        self.content_group.toggled.connect(lambda _: self._rebuild_dataset())
        self.start_note_edit.textChanged.connect(lambda _: self._rebuild_dataset())
        self.end_note_edit.textChanged.connect(lambda _: self._rebuild_dataset())
        self.min_between_spin.valueChanged.connect(lambda _: self._rebuild_dataset())
        self.max_between_spin.valueChanged.connect(lambda _: self._rebuild_dataset())
        self.contains_edit.textChanged.connect(lambda _: self._rebuild_dataset())
        # Force transpose/tempo interactions
        # Do not rebuild the dataset when these controls change.  Instead
        # update the display so that the current melody is transposed or
        # retimed on the fly.  Dataset rebuild is unnecessary and would
        # disrupt melody order.
        self.force_key_check.stateChanged.connect(lambda _: self._update_display())
        self.force_key_combo.currentIndexChanged.connect(lambda _: self._update_display())
        self.force_scale_combo.currentIndexChanged.connect(lambda _: self._update_display())
        self.force_tempo_check.stateChanged.connect(lambda _: self._update_display())
        self.force_tempo_spin.valueChanged.connect(lambda _: self._update_display())

        # Segment options interactions
        self.variable_length_check.stateChanged.connect(lambda _: self._rebuild_dataset())
        self.min_length_spin.valueChanged.connect(lambda _: self._rebuild_dataset())
        self.max_length_spin.valueChanged.connect(lambda _: self._rebuild_dataset())
        self.align_chords_check.stateChanged.connect(lambda _: self._rebuild_dataset())

    # --- Dataset management -------------------------------------------------
    def _rebuild_dataset(self) -> None:
        """Recreate the dataset whenever settings change."""
        ratio = self.random_slider.value() / 100.0
        size = self.size_spin.value()
        key = self.key_combo.currentText()
        scale = self.scale_combo.currentText()
        # Build dataset using the manager
        # Compose key string including mode for random melodies
        # Compose key string.  If the tonic is 'Any', do not enforce a key;
        # random melodies will use the scale only with a default tonic.  We
        # still include the scale name as part of the string to inform the
        # generator.  Otherwise, join tonic and scale.
        # Determine key (tonic) and scale (mode).  When key is 'Any', we
        # signal to the manager that any tonic is acceptable by passing
        # 'any'.  Scale type is always passed separately in lowercase.  For
        # backwards compatibility, do not include the mode in the key
        # string here; instead use the scale_type argument in build_dataset.
        if key == "Any":
            key_str = 'any'
        else:
            key_str = key
        # Compute additional parameters from sliders and spins
        tempo = self.tempo_spin.value()
        length = self.length_spin.value()
        complexity = self.complexity_slider.value() / 100.0
        density = self.density_slider.value() / 100.0
        max_interval = self.interval_spin.value()
        max_span_octaves = self.span_spin.value()
        max_span = max_span_octaves * 12  # convert octaves to semitones
        register = self.register_combo.currentText()
        metadata_filter = self.metadata_edit.text().strip()
        # Sequence filter parameters.  The sequence filter is considered
        # active if the group box is checked OR if any of the fields
        # contain text.  This allows users to type criteria without
        # explicitly checking the group box.  When active, the
        # corresponding values are passed to build_dataset; otherwise no
        # sequence filtering is applied.
        seq_inputs = [
            self.start_note_edit.text().strip(),
            self.end_note_edit.text().strip(),
            self.contains_edit.text().strip(),
        ]
        seq_active = self.content_group.isChecked() or any(seq_inputs)
        if seq_active:
            start_note = self.start_note_edit.text().strip() or None
            end_note = self.end_note_edit.text().strip() or None
            min_bt = self.min_between_spin.value()
            max_bt = self.max_between_spin.value()
            contains = self.contains_edit.text().strip() or None
        else:
            start_note = end_note = contains = None
            min_bt = 0
            max_bt = 32
        # Segment options
        variable_length = self.variable_length_check.isChecked()
        min_len = self.min_length_spin.value()
        max_len = self.max_length_spin.value()
        align_to_chords = self.align_chords_check.isChecked()
        self.dataset = self.manager.build_dataset(
            num_melodies=size,
            random_ratio=ratio,
            key=key_str,
            tempo=tempo if tempo != 0 else None,
            length=length if not variable_length else None,
            max_interval=max_interval,
            max_span=max_span,
            register=register,
            complexity_level=complexity,
            note_density_level=density,
            metadata_filter=metadata_filter if metadata_filter else None,
            metadata_conditions=self.metadata_conditions if self.metadata_conditions else None,
            start_note=start_note,
            end_note=end_note,
            min_between=min_bt,
            max_between=max_bt,
            contains_notes=contains,
            variable_length=variable_length,
            min_length=min_len,
            max_length=max_len,
            align_to_chords=align_to_chords,
            scale_type=scale.lower(),
        )
        self.current_index = 0
        # Force transpose and tempo are applied on-the-fly in _get_playback_melody.
        # We no longer modify the dataset when toggling these options, as
        # doing so would cause the selected melody to change unexpectedly.
        # Instead we simply refresh the display after rebuilding the dataset.
        self._update_display()

    def _current_melody(self) -> Optional[Melody]:
        return self.dataset[self.current_index] if self.dataset else None

    def _update_display(self, message: str = "") -> None:
        """Update the text and staff displays with the current melody and status."""
        # Apply selected font size to the display area
        try:
            font = self.display_area.font()
            font.setPointSize(self.font_size_spin.value())
            self.display_area.setFont(font)
        except Exception:
            pass
        if not self.dataset:
            self.display_area.setPlainText("No melodies available.")
            self.staff_label.clear()
            self.keyboard_label.clear()
            return
        # Use the playback melody (with force transpose/tempo and octave shift) for display
        melody = self._get_playback_melody()
        if not melody:
            return
        lines: List[str] = []
        # Header: show index and name (if available) or 'Random'
        lines.append(f"Melody {self.current_index + 1}/{len(self.dataset)}: {melody.name or 'Random'}")
        # Show note names with scale degrees if enabled
        if self.show_notes_check.isChecked():
            note_strs: List[str] = []
            shift = self.octave_shift
            key_parts = melody.key.split()
            tonic = key_parts[0] if key_parts else 'C'
            mode = key_parts[1] if len(key_parts) > 1 else 'major'
            for event in melody.events:
                pitch = event.pitch + 12 * shift
                name = self._midi_to_name(pitch)
                degree = self._pitch_to_degree(pitch, tonic, mode)
                note_strs.append(f"{name}({degree})")
            lines.append("Notes: " + ", ".join(note_strs))
        # Key/tempo/metrics
        if self.show_key_check.isChecked():
            lines.append(f"Key: {melody.key}, Tempo: {melody.tempo}, Complexity: {melody.complexity:.2f}")
            lines.append(f"Avg interval: {melody.avg_interval:.2f}, Span: {melody.span}, Density: {melody.note_density:.2f}, Register: {melody.register}")
            # Show additional metadata
            if melody.metadata:
                meta_parts = []
                for meta_key, meta_val in melody.metadata.items():
                    val_str = str(meta_val)
                    if len(val_str) > 40:
                        val_str = val_str[:37] + '…'
                    meta_parts.append(f"{meta_key}: {val_str}")
                lines.append("; ".join(meta_parts))
        # Append any message (e.g. success/failure from detectors)
        if message:
            lines.append("")
            lines.append(message)
        self.display_area.setPlainText("\n".join(lines))
        # Update staff and keyboard
        if self.show_staff_check.isChecked():
            self._update_staff(melody)
        else:
            self.staff_label.clear()
        if self.show_keyboard_check.isChecked():
            self._update_keyboard(melody)
        else:
            self.keyboard_label.clear()

    @staticmethod
    def _midi_to_name(pitch: int) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (pitch // 12) - 1
        note_name = names[pitch % 12]
        return f"{note_name}{octave}"

    @staticmethod
    def _pitch_to_degree(pitch: int, tonic: str, mode: str) -> str:
        """Return the scale degree (1–7) of a given MIDI pitch relative to a key.

        If the pitch does not fall within the diatonic scale of the key
        (major or natural minor), a question mark is returned.  Mode names
        other than 'major' and 'minor' are treated like major.
        """
        # Map note names to pitch classes
        pc_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
            'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11
        }
        tonic_pc = pc_map.get(tonic.split('/')[0], 0)
        pitch_pc = pitch % 12
        diff = (pitch_pc - tonic_pc) % 12
        # Define diatonic degrees for major and natural minor scales
        major_degrees = {0: '1', 2: '2', 4: '3', 5: '4', 7: '5', 9: '6', 11: '7'}
        minor_degrees = {0: '1', 2: '2', 3: '3', 5: '4', 7: '5', 8: '6', 10: '7'}
        if mode.lower().startswith('minor'):
            return minor_degrees.get(diff, '?')
        else:
            return major_degrees.get(diff, '?')

    def _update_staff(self, melody: Melody) -> None:
        """Draw a simple staff notation for the given melody and display it on the staff_label."""
        # Dynamically size the staff based on the melody's pitch range.  This
        # prevents high or low notes from being cropped.  The width is
        # proportional to the number of events; the height expands to
        # accommodate ledger lines.
        num_events = max(len(melody.events), 1)
        # Increase width to allow more space for note names and accidentals
        width = max(500, num_events * 35)
        # Compute pitch range (with octave shift) to adjust height
        shift = self.octave_shift
        pitches = [ev.pitch + shift * 12 for ev in melody.events]
        if not pitches:
            pitches = [60]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        # Line spacing and margins
        margin_top = 20
        line_spacing = 10
        base_y = margin_top + 2.5 * line_spacing  # position for MIDI 60 (C4)
        semitone_step = line_spacing / 2.0
        # Extra space above and below based on pitch range
        extra_up = max(0, (max_pitch - 60) * semitone_step)
        extra_down = max(0, (60 - min_pitch) * semitone_step)
        # Increase base height to provide more room for ledger lines
        height = int(margin_top + 4 * line_spacing + extra_up + extra_down + 80)
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(Qt.GlobalColor.black)
        pen.setWidth(1)
        painter.setPen(pen)
        # Adjust base_y within new height: base_y + extra_up moves down ledger lines
        base_y = margin_top + 2.5 * line_spacing + extra_up
        # Draw five horizontal staff lines
        for i in range(5):
            y = base_y - 2 * line_spacing + i * line_spacing
            painter.drawLine(10, int(y), width - 10, int(y))
        # Draw notes and ledger lines
        for idx, event in enumerate(melody.events):
            x = 20 + idx * (width - 40) / max(num_events - 1, 1)
            pitch = event.pitch + shift * 12
            y = base_y - (pitch - 60) * semitone_step
            radius = 4
            painter.setBrush(QBrush(Qt.GlobalColor.black))
            painter.drawEllipse(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
            # Determine accidental to draw ('#' or '♭') based on note name
            try:
                name = self._midi_to_name(pitch)
                accidental = ''
                if '#' in name:
                    accidental = '#'
                elif 'B' in name and not name.startswith('B'):  # e.g. DB, EB, AB
                    accidental = '♭'
                if accidental:
                    painter.setPen(QPen(Qt.GlobalColor.black))
                    painter.setFont(painter.font())
                    painter.drawText(int(x - radius - 8), int(y + radius), accidental)
            except Exception:
                pass
            # Draw ledger lines for notes outside the staff
            top_line_y = base_y - 2 * line_spacing
            bottom_line_y = base_y + 2 * line_spacing
            if y < top_line_y:
                n = int((top_line_y - y) / (line_spacing)) + 1
                for j in range(1, n + 1):
                    ledger_y = top_line_y - j * line_spacing
                    if ledger_y < 5:
                        break
                    painter.drawLine(int(x - 8), int(ledger_y), int(x + 8), int(ledger_y))
            elif y > bottom_line_y:
                n = int((y - bottom_line_y) / (line_spacing)) + 1
                for j in range(1, n + 1):
                    ledger_y = bottom_line_y + j * line_spacing
                    if ledger_y > height - 5:
                        break
                    painter.drawLine(int(x - 8), int(ledger_y), int(x + 8), int(ledger_y))
        painter.end()
        self.staff_label.setPixmap(pixmap)

    def _update_keyboard(self, melody: Melody) -> None:
        """Draw a two‑octave keyboard indicating which pitches are present and active.

        The keyboard spans 24 semitones starting from a computed base pitch that
        ensures the melody's range fits within two octaves.  White keys are
        drawn longer and black keys shorter.  Keys that appear in the melody
        are highlighted in light blue, while currently sounding keys are
        highlighted in dark blue.
        """
        # Determine the set of pitches (with octave shift) in the melody and active notes
        shift = self.octave_shift
        mel_pitches = [ev.pitch + shift * 12 for ev in melody.events]
        if not mel_pitches:
            mel_pitches = [60]
        active_pitches = [p + shift * 12 for p in getattr(self, 'active_pitches', set())]
        # Compute a start pitch for the 2‑octave keyboard.  We attempt to
        # centre the range so that both low and high notes are visible.  If
        # the range exceeds two octaves, we anchor the keyboard to include
        # the highest note.
        min_pitch = min(mel_pitches)
        max_pitch = max(mel_pitches)
        # Start at the octave below the highest note if range >24
        if max_pitch - min_pitch + 1 > 24:
            start_pitch = (max_pitch // 12) * 12 - 12
        else:
            # Centre around the median pitch
            mid = (min_pitch + max_pitch) // 2
            start_pitch = (mid // 12) * 12 - 12
            # Ensure start_pitch does not exceed min_pitch
            if start_pitch > min_pitch:
                start_pitch = (min_pitch // 12) * 12
        # Two octaves covers 24 semitones
        total_keys = 24
        # Dimensions
        # Use wider keys and taller keyboard for better clarity
        # Increase key width and overall keyboard height for better visibility
        white_key_width = 60
        height = 120
        # There are 7 white keys per octave => 14 white keys total
        total_white_keys = 14
        width = total_white_keys * white_key_width
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Map pitch classes to white key order within an octave (0..6)
        white_order = [0, 2, 4, 5, 7, 9, 11]
        black_pcs = {1, 3, 6, 8, 10}
        # Precompute positions
        # Draw white keys
        for i in range(total_keys):
            pitch = start_pitch + i
            pc = pitch % 12
            octave_index = i // 12
            if pc in white_order:
                local_idx = white_order.index(pc)
                global_white_idx = octave_index * 7 + local_idx
                x = global_white_idx * white_key_width
                # Determine fill colour
                col = None
                if pitch in active_pitches:
                    col = QColor(65, 105, 225)  # dark blue for active
                elif pitch in mel_pitches:
                    col = QColor(173, 216, 230)  # light blue for contained
                if col:
                    painter.fillRect(int(x), 0, int(white_key_width), int(height), col)
                # Draw outline
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawRect(int(x), 0, int(white_key_width), int(height))
        # Draw black keys
        black_height = height * 0.65
        for i in range(total_keys):
            pitch = start_pitch + i
            pc = pitch % 12
            octave_index = i // 12
            if pc in black_pcs:
                # Determine x offset relative to white keys
                # Within an octave, map black pcs to fractional offsets between white keys
                local_map = {1: 0.5, 3: 1.5, 6: 3.5, 8: 4.5, 10: 5.5}
                offset = local_map.get(pc, None)
                if offset is None:
                    continue
                x = (octave_index * 7 + offset) * white_key_width - (white_key_width * 0.5)
                w = white_key_width * 0.8
                # Determine colour
                if pitch in active_pitches:
                    painter.fillRect(int(x), 0, int(w), int(black_height), QColor(30, 30, 200))
                elif pitch in mel_pitches:
                    painter.fillRect(int(x), 0, int(w), int(black_height), QColor(65, 105, 225))
                else:
                    painter.fillRect(int(x), 0, int(w), int(black_height), QColor(0, 0, 0))
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawRect(int(x), 0, int(w), int(black_height))
        painter.end()
        self.keyboard_label.setPixmap(pixmap)

    # --- Playback control handlers -----------------------------------------
    def _on_play_clicked(self) -> None:
        if not self.dataset:
            return
        if self.player.is_playing():
            # Stop playback
            self.player.stop()
            self.play_button.setText("Play")
            # Stop detectors
            self._stop_detectors()
        else:
            # Play current melody
            melody = self._get_playback_melody()
            if melody:
                self.player.play_melody(melody)
                # Animate keyboard with current melody
                self._animate_keyboard(melody)
                self.play_button.setText("Stop")
                # Start automatic detection if enabled
                if self.auto_check.isChecked():
                    self._start_detection(melody)

    def _on_next_clicked(self) -> None:
        if not self.dataset:
            return
        self.player.stop()
        self._stop_detectors()
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self._update_display()
        # Automatically start playback if currently in playing state
        if self.play_button.text() == "Stop":
            mel = self._get_playback_melody()
            if mel:
                self.player.play_melody(mel)
                if self.auto_check.isChecked():
                    self._start_detection(mel)

    def _on_prev_clicked(self) -> None:
        if not self.dataset:
            return
        self.player.stop()
        self._stop_detectors()
        self.current_index = (self.current_index - 1) % len(self.dataset)
        self._update_display()
        if self.play_button.text() == "Stop":
            mel = self._get_playback_melody()
            if mel:
                self.player.play_melody(mel)
                if self.auto_check.isChecked():
                    self._start_detection(mel)

    # --- Detection handling -------------------------------------------------
    def _on_auto_changed(self) -> None:
        """Enable or disable automatic detection."""
        enabled = self.auto_check.isChecked()
        if not enabled:
            self._stop_detectors()
        else:
            # If currently playing, start detection immediately
            if self.player.is_playing():
                melody = self._current_melody()
                if melody:
                    self._start_detection(melody)

    def _on_mode_changed(self) -> None:
        """Update available input devices when the detection mode changes."""
        self._update_input_devices()

    def _update_input_devices(self) -> None:
        """Populate the input device combobox based on the selected detection mode."""
        mode = self.detect_mode.currentText()
        self.input_combo.clear()
        if mode == "MIDI" and self.midi_detector:
            names = MidiDetector.list_inputs()
            self.input_combo.addItems(names)
        elif mode == "Audio" and self.audio_detector:
            names = AudioDetector.list_audio_inputs() if hasattr(AudioDetector, 'list_audio_inputs') else []
            self.input_combo.addItems(names)
        else:
            # Manual mode: no input devices needed
            self.input_combo.addItem("(none)")

    def _update_midi_outputs(self) -> None:
        """Populate the MIDI output combobox."""
        self.midi_out_combo.clear()
        names = self.player.list_midi_outputs() if hasattr(self.player, 'list_midi_outputs') else []
        self.midi_out_combo.addItems(names)

    def _start_detection(self, melody: Melody) -> None:
        """Start the appropriate detector based on mode."""
        mode = self.detect_mode.currentText()
        # Stop any existing detectors
        self._stop_detectors()
        if mode == "MIDI" and self.midi_detector:
            port = self.input_combo.currentText()
            if not port:
                return
            def cb(success: bool, notes: List[int]) -> None:
                msg = "Correct! Moving to next." if success else "Incorrect. Try again."
                self._update_display(msg)
                if success:
                    self._on_next_clicked()
            self.midi_detector.start(melody, port, cb)
        elif mode == "Audio" and self.audio_detector:
            idx = self.input_combo.currentIndex()
            self.audio_detector.set_input_device(idx if idx >= 0 else None)
            def cb(success: bool, notes: List[int]) -> None:
                msg = "Correct! Moving to next." if success else "Incorrect. Try again."
                self._update_display(msg)
                if success:
                    self._on_next_clicked()
            self.audio_detector.start(melody, cb)

    def _stop_detectors(self) -> None:
        """Stop both MIDI and audio detectors if running."""
        if self.midi_detector:
            self.midi_detector.stop()
        if self.audio_detector:
            self.audio_detector.stop()

    # --- New features ------------------------------------------------------
    def _on_instrument_changed(self) -> None:
        """Handle changes to the instrument selection."""
        name = self.instrument_combo.currentText()
        # Update player instrument
        self.player.set_instrument(name)

    def _on_midi_out_changed(self) -> None:
        """Handle selection of a MIDI output port."""
        name = self.midi_out_combo.currentText()
        self.player.set_midi_output(name if name else None)

    def _on_octave_up(self) -> None:
        """Shift the current melody up one octave."""
        self.octave_shift += 1
        self._update_display()
        # If currently playing, restart playback with transposed melody
        if self.play_button.text() == "Stop":
            self.player.stop()
            mel = self._get_playback_melody()
            if mel:
                self.player.play_melody(mel)
                if self.auto_check.isChecked():
                    self._start_detection(mel)

    def _on_octave_down(self) -> None:
        """Shift the current melody down one octave."""
        self.octave_shift -= 1
        self._update_display()
        if self.play_button.text() == "Stop":
            self.player.stop()
            mel = self._get_playback_melody()
            if mel:
                self.player.play_melody(mel)
                if self.auto_check.isChecked():
                    self._start_detection(mel)

    def _get_playback_melody(self) -> Optional[Melody]:
        """Return a melody ready for playback/detection with force transpose,
        force tempo and octave shift applied on the fly.

        A copy of the current melody is created to avoid modifying the
        underlying dataset.  If the force transpose/tempo options are not
        enabled, the original melody is returned (with octave shift applied
        if needed).
        """
        base = self._current_melody()
        if not base:
            return None
        # Start with the base melody (no modification yet)
        mel = base
        # Apply force transpose if requested
        if self.force_key_check.isChecked():
            target_tonic = self.force_key_combo.currentText().strip()
            target_scale = self.force_scale_combo.currentText().strip()
            # Only proceed if at least one of tonic or scale is specified (not Any)
            if target_tonic.lower() not in ('any', '') or target_scale.lower() not in ('any', ''):
                # Determine original tonic and mode
                parts = mel.key.split()
                orig_tonic = parts[0] if parts else 'C'
                orig_mode = parts[1] if len(parts) > 1 else 'major'
                new_mel = mel
                # Transpose tonic
                if target_tonic.lower() not in ('any', ''):
                    pc_map = {
                        'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3,
                        'e': 4, 'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8,
                        'ab': 8, 'a': 9, 'a#': 10, 'bb': 10, 'b': 11, 'cb': 11
                    }
                    orig_pc = pc_map.get(orig_tonic.lower(), 0)
                    target_pc = pc_map.get(target_tonic.lower(), orig_pc)
                    semitones = (target_pc - orig_pc) % 12
                    new_mel = self.manager.transpose_melody(mel, semitones)
                    # Build key string; update tonic
                    new_key_parts = [target_tonic]
                else:
                    new_key_parts = [orig_tonic]
                # Determine scale/mode
                if target_scale.lower() not in ('any', ''):
                    new_mode = target_scale
                else:
                    new_mode = orig_mode
                # Update key string
                new_mel.key = f"{new_key_parts[0]} {new_mode}".strip()
                mel = new_mel
        # Apply force tempo if requested
        if self.force_tempo_check.isChecked():
            # Create a copy with new tempo (if not already a new object)
            if mel is base:
                mel = Melody(
                    events=[NoteEvent(ev.pitch, ev.duration, ev.velocity) for ev in base.events],
                    key=base.key,
                    tempo=self.force_tempo_spin.value(),
                    name=base.name,
                    complexity=base.complexity,
                    avg_interval=base.avg_interval,
                    span=base.span,
                    note_density=base.note_density,
                    register=base.register,
                    metadata=base.metadata,
                    file_path=base.file_path,
                    chords=base.chords,
                )
                self.manager._compute_metadata(mel)
            else:
                mel.tempo = self.force_tempo_spin.value()
        # Apply octave shift
        if self.octave_shift != 0:
            mel = self.manager.transpose_melody(mel, self.octave_shift * 12)
        return mel

    # Keyboard animation helpers
    def _highlight_key(self, pitch: int) -> None:
        """Add a pitch to the active set and update keyboard display."""
        self.active_pitches.add(pitch)
        if self.show_keyboard_check.isChecked():
            mel = self._current_melody()
            if mel:
                self._update_keyboard(mel)

    def _unhighlight_key(self, pitch: int) -> None:
        """Remove a pitch from the active set and update keyboard display."""
        self.active_pitches.discard(pitch)
        if self.show_keyboard_check.isChecked():
            mel = self._current_melody()
            if mel:
                self._update_keyboard(mel)

    def _animate_keyboard(self, melody: Melody) -> None:
        """Animate keyboard highlights in sync with melody playback."""
        if not self.show_keyboard_check.isChecked():
            return
        # Clear any previous highlights
        self.active_pitches.clear()
        # Base time accumulator
        t_ms = 0
        beats_per_second = melody.tempo / 60.0
        for event in melody.events:
            duration_ms = int((event.duration / beats_per_second) * 1000)
            # Capture pitch for closures
            pitch = event.pitch + self.octave_shift * 12
            QTimer.singleShot(t_ms, lambda p=pitch: self._highlight_key(p))
            QTimer.singleShot(t_ms + duration_ms, lambda p=pitch: self._unhighlight_key(p))
            t_ms += duration_ms

    # --- Advanced metadata filter dialog ---------------------------------
    def _open_filter_dialog(self) -> None:
        """Open a dialog to build complex metadata filter conditions."""
        dlg = MetadataFilterDialog(self.manager, self.metadata_conditions, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Retrieve conditions from dialog
            self.metadata_conditions = dlg.conditions
            # Rebuild dataset with new conditions
            self._rebuild_dataset()


class MetadataFilterDialog(QDialog):
    """Dialog for constructing advanced metadata filter conditions.

    Users can add multiple conditions on metadata fields with a variety
    of operators (contains, equals, >=, <=, range).  The dialog shows
    the list of conditions, allows removal, and displays the number of
    actual melodies matching the current set of conditions.
    """
    def __init__(self, manager: MelodyManager, existing: Optional[List[tuple]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Metadata Filters")
        self.manager = manager
        self.conditions: List[Tuple[str, str, List[str]]] = list(existing or [])
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        # Table to display conditions
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Field", "Operator", "Value(s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        # Form to add a new condition
        form = QFormLayout()
        # Field selection
        self.field_combo = QComboBox()
        # Populate with available metadata fields
        for fld in self.manager.metadata_fields():
            self.field_combo.addItem(fld)
        form.addRow("Field:", self.field_combo)
        # Operator selection
        self.op_combo = QComboBox()
        self.op_combo.addItems(["contains", "equals", ">=", "<=", "range"])
        form.addRow("Operator:", self.op_combo)
        # Value 1 and Value 2 edits
        self.value1_edit = QLineEdit()
        form.addRow("Value 1:", self.value1_edit)
        self.value2_edit = QLineEdit()
        form.addRow("Value 2 (for range):", self.value2_edit)
        # Add button
        self.add_btn = QPushButton("Add Condition")
        self.add_btn.clicked.connect(self._on_add)
        form.addRow(self.add_btn)
        layout.addLayout(form)
        # Remove button
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._on_remove)
        layout.addWidget(self.remove_btn)
        # Count label
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        # Populate table with existing conditions
        for cond in self.conditions:
            self._append_row(cond)
        # Update count initially
        self._update_count()

    def _append_row(self, cond: Tuple[str, str, List[str]]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(cond[0]))
        self.table.setItem(row, 1, QTableWidgetItem(cond[1]))
        self.table.setItem(row, 2, QTableWidgetItem(", ".join(cond[2])))

    def _on_add(self) -> None:
        field = self.field_combo.currentText().strip()
        operator = self.op_combo.currentText().strip()
        v1 = self.value1_edit.text().strip()
        v2 = self.value2_edit.text().strip()
        if not field or not operator or not v1:
            return
        values: List[str] = [v1]
        if operator == "range" and v2:
            values.append(v2)
        cond = (field, operator, values)
        self.conditions.append(cond)
        self._append_row(cond)
        # Clear input fields
        self.value1_edit.clear()
        self.value2_edit.clear()
        # Update count
        self._update_count()

    def _on_remove(self) -> None:
        selected = self.table.selectedIndexes()
        if not selected:
            return
        # Remove rows in reverse order to preserve indices
        rows = sorted(set(idx.row() for idx in selected), reverse=True)
        for r in rows:
            if 0 <= r < len(self.conditions):
                self.conditions.pop(r)
            self.table.removeRow(r)
        self._update_count()

    def _update_count(self) -> None:
        # Compute number of actual melodies matching current conditions.
        parent = self.parent()
        if isinstance(parent, EarTrainerGUI):
            # Use parent's current filter settings for length and other constraints
            # Include sequence filter and other dataset parameters when counting
            seq_inputs = [
                parent.start_note_edit.text().strip(),
                parent.end_note_edit.text().strip(),
                parent.contains_edit.text().strip(),
            ]
            seq_active = parent.content_group.isChecked() or any(seq_inputs)
            start_note = parent.start_note_edit.text().strip() if seq_active else None
            end_note = parent.end_note_edit.text().strip() if seq_active else None
            min_bt = parent.min_between_spin.value() if seq_active else None
            max_bt = parent.max_between_spin.value() if seq_active else None
            contains_notes = parent.contains_edit.text().strip() if seq_active else None
            count = parent.manager.count_filtered(
                metadata_conditions=self.conditions or None,
                metadata_filter=parent.metadata_edit.text().strip() or None,
                length=parent.length_spin.value(),
                max_interval=parent.interval_spin.value(),
                max_span=parent.span_spin.value() * 12,
                register=parent.register_combo.currentText(),
                complexity_level=parent.complexity_slider.value() / 100.0,
                note_density_level=parent.density_slider.value() / 100.0,
            )
        else:
            count = self.manager.count_filtered(metadata_conditions=self.conditions or None)
        total = len(self.manager.actual_melodies)
        self.count_label.setText(f"Matches: {count}/{total}")
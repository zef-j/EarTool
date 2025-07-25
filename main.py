"""
main.py
=======

Entry point for the ear training application.  This script creates the
necessary components—melody manager, audio player and detectors—and
initialises the Qt application.  If optional dependencies such as mido or
aubio are unavailable the corresponding detectors will be disabled, but
basic playback and manual mode will still function.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from melody_manager import MelodyManager
from audio_player import AudioPlayer
from pitch_detector import MidiDetector, AudioDetector
from gui import EarTrainerGUI


def main() -> None:
    # Create the Qt application
    app = QApplication(sys.argv)
    # Determine the directory for actual melodies (optional)
    melody_dir = None
    default_dir = os.path.join(os.path.dirname(__file__), 'sample_melodies')
    if os.path.isdir(default_dir):
        melody_dir = default_dir
    # Instantiate core components
    manager = MelodyManager(melody_dir)
    # If a processed dataset JSON exists, load it to augment the library.
    processed_path = os.path.join(os.path.dirname(__file__), 'processed_melodies.json')
    if os.path.isfile(processed_path):
        try:
            manager.load_from_json(processed_path)
            print(f"Loaded preprocessed melodies from {processed_path}")
        except Exception as exc:
            print(f"Could not load preprocessed dataset: {exc}")
    player = AudioPlayer()
    # Try to initialise detectors; failures are logged to console
    try:
        midi_detector = MidiDetector()
    except Exception as exc:
        print(f"MIDI detection unavailable: {exc}")
        midi_detector = None
    try:
        audio_detector = AudioDetector()
    except Exception as exc:
        print(f"Audio detection unavailable: {exc}")
        audio_detector = None
    # Create and show the main window
    window = EarTrainerGUI(manager, player, midi_detector, audio_detector)
    window.show()
    # Run the event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
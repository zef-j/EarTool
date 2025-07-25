"""
pitch_detector.py
=================

This module provides classes for real‑time note detection using either
MIDI input devices or audio (microphone) input.  For MIDI detection the
mido library is used to receive note_on events.  For audio detection the
aubio library performs pitch tracking on a continuous audio stream.  The
detector classes compare the incoming notes against a target Melody and
invoke a callback whenever the user finishes playing all notes or a mistake
occurs.  Both detectors run on background threads to avoid blocking the
GUI.

Note: These detectors are intended as illustrative implementations and may
require tuning of latency and pitch detection parameters for reliable
performance.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

try:
    import mido
except ImportError:
    mido = None

try:
    import aubio
except ImportError:
    aubio = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

from melody_manager import Melody, NoteEvent


class MidiDetector:
    """Detect notes played on a MIDI input device and compare to a melody."""

    def __init__(self) -> None:
        if mido is None:
            raise RuntimeError("mido is required for MIDI detection")
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.expected: Optional[Melody] = None
        self.callback: Optional[Callable[[bool, List[int]], None]] = None
        self.port_name: Optional[str] = None

    @staticmethod
    def list_inputs() -> List[str]:
        """Return a list of available MIDI input port names."""
        return mido.get_input_names() if mido is not None else []

    def start(self, expected: Melody, port_name: str, callback: Callable[[bool, List[int]], None]) -> None:
        """Start listening on the given MIDI port and evaluating played notes.

        Parameters
        ----------
        expected : Melody
            The target melody the user should reproduce.
        port_name : str
            Name of the MIDI input port to listen on.
        callback : callable
            Function invoked when the user finishes or makes a mistake.  It
            receives two arguments: a boolean indicating success and the
            sequence of played MIDI notes.
        """
        self.stop()
        self.expected = expected
        self.callback = callback
        self.port_name = port_name

        def worker():
            expected_pitches = [ev.pitch for ev in expected.events]
            played: List[int] = []
            try:
                with mido.open_input(port_name) as port:
                    for msg in port:
                        if self._stop_event.is_set():
                            return
                        if msg.type == 'note_on' and msg.velocity > 0:
                            played.append(msg.note)
                            # Evaluate note immediately
                            idx = len(played) - 1
                            if idx >= len(expected_pitches) or played[idx] != expected_pitches[idx]:
                                # Wrong note
                                if self.callback:
                                    self.callback(False, played)
                                return
                            if len(played) == len(expected_pitches):
                                # Completed correctly
                                if self.callback:
                                    self.callback(True, played)
                                return
            except Exception as exc:
                print(f"MIDI detection error: {exc}")

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listening for MIDI input."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            self._stop_event.clear()
        self._thread = None


class AudioDetector:
    """Detect notes sung or played through a microphone using aubio."""

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 2048, hop_size: int = 512) -> None:
        if aubio is None or sd is None:
            raise RuntimeError("aubio and sounddevice are required for audio detection")
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.hop_size = hop_size
        self.pitch_o = aubio.pitch(method="yin", buf_size=buffer_size, hop_size=hop_size, samplerate=sample_rate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(0.8)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.expected: Optional[Melody] = None
        self.callback: Optional[Callable[[bool, List[int]], None]] = None
        self.input_device: Optional[int] = None

    @staticmethod
    def list_audio_inputs() -> List[str]:
        """Return a list of available audio input device names."""
        if sd is None:
            return []
        devices = sd.query_devices()
        return [d['name'] for d in devices if d['max_input_channels'] > 0]

    def set_input_device(self, device_index: Optional[int]) -> None:
        """Set the index of the audio input device used for detection."""
        self.input_device = device_index

    def start(self, expected: Melody, callback: Callable[[bool, List[int]], None]) -> None:
        """Begin detecting pitches from audio and comparing to the target melody."""
        self.stop()
        self.expected = expected
        self.callback = callback

        def worker():
            expected_pitches = [ev.pitch for ev in expected.events]
            played: List[int] = []
            # Buffer to accumulate frames
            def audio_callback(indata, frames, time, status):
                if self._stop_event.is_set():
                    raise sd.CallbackStop()
                # Convert to mono
                mono = indata[:, 0]
                pitch = self.pitch_o(mono)[0]
                confidence = self.pitch_o.get_confidence()
                if confidence > 0.8 and pitch > 0:
                    midi_note = int(round(pitch))
                    # Only record note if changed from previous to avoid repeats
                    if not played or midi_note != played[-1]:
                        played.append(midi_note)
                        idx = len(played) - 1
                        if idx >= len(expected_pitches) or played[idx] != expected_pitches[idx]:
                            if self.callback:
                                self.callback(False, played)
                            raise sd.CallbackStop()
                        if len(played) == len(expected_pitches):
                            if self.callback:
                                self.callback(True, played)
                            raise sd.CallbackStop()

            stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.hop_size,
                channels=1,
                callback=audio_callback,
                device=self.input_device,
            )
            with stream:
                try:
                    self._stop_event.wait()  # Wait until stop_event is set or callback stops
                except Exception:
                    pass

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop audio pitch detection."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            self._stop_event.clear()
        self._thread = None
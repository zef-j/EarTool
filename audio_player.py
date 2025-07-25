"""
audio_player.py
================

This module defines the AudioPlayer class responsible for converting
Melody objects into audible waveforms and playing them through the selected
audio output device.  The default implementation uses NumPy to generate
simple sine waves for each note and relies on the sounddevice library to
output audio.  For more realistic sounds you could substitute this with
pyfluidsynth or another synthesizer.

Note that playback runs on a background thread so that the GUI remains
responsive.  The class exposes methods to start, stop and query playback
state, as well as to change the output device.
"""

from __future__ import annotations

import threading
import numpy as np
import sounddevice as sd
from typing import Optional
import os

from melody_manager import Melody, NoteEvent

try:
    import fluidsynth
except ImportError:
    fluidsynth = None


def midi_to_freq(midi_note: int) -> float:
    """Convert a MIDI note number to its fundamental frequency in Hertz.

    Uses the standard formula f = 440 * 2^((m-69)/12).
    """
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


class AudioPlayer:
    """Simple audio playback engine for melodies.

    The player can synthesise simple sine‑wave tones or, when FluidSynth
    and a suitable SoundFont are available, render General MIDI instruments.
    Instrument names are normalised (case‑insensitive and leading/trailing
    whitespace removed) before being looked up.  See ``set_instrument``.

    Parameters
    ----------
    sample_rate : int, optional
        Number of audio samples per second.  CD quality uses 44100 Hz.
    """

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.current_melody: Optional[Melody] = None
        self.output_device: Optional[int] = None  # Device index in sounddevice
        # Instrument and synthesizer support
        # Selected instrument.  The default is a pure sine wave.  The
        # instrument name is normalised in ``set_instrument``.
        self.instrument: str = 'sine'
        self.synth: Optional[fluidsynth.Synth] = None if fluidsynth is None else None
        self.soundfont_id: Optional[int] = None
        # Attempt to locate a default SoundFont file.  Users can place
        # 'soundfont.sf2' in the package directory or the project root to
        # enable FluidSynth.  Also recognise the commonly used
        # 'FluidR3_GM.sf2'.  We search multiple locations and pick the
        # first that exists.
        candidates = []
        # Package directory
        pkg_dir = os.path.dirname(__file__)
        candidates.append(os.path.join(pkg_dir, 'soundfont.sf2'))
        candidates.append(os.path.join(pkg_dir, 'FluidR3_GM.sf2'))
        # Project root (two levels up)
        root_dir = os.path.abspath(os.path.join(pkg_dir, os.pardir))
        candidates.append(os.path.join(root_dir, 'soundfont.sf2'))
        candidates.append(os.path.join(root_dir, 'FluidR3_GM.sf2'))
        # Choose the first existing path
        self.soundfont_path = None
        for c in candidates:
            if os.path.isfile(c):
                self.soundfont_path = c
                break
        # If FluidSynth is available and a SoundFont was found, initialise
        # the synthesiser immediately.  This ensures that instrument
        # selection works even before playback or explicit set_instrument
        # calls.  Otherwise the synthesiser will be initialised lazily.
        if fluidsynth is not None and self.soundfont_path:
            try:
                self.synth = fluidsynth.Synth(samplerate=self.sample_rate)
                self.soundfont_id = self.synth.sfload(self.soundfont_path)
            except Exception:
                self.synth = None
                self.soundfont_id = None

    def set_output_device(self, device_index: Optional[int]) -> None:
        """Set the audio output device by index.

        The index corresponds to the values returned by ``sounddevice.query_devices()``.
        If ``device_index`` is None, the system default will be used.
        """
        self.output_device = device_index

    def set_instrument(self, name: str) -> None:
        """Change the playback instrument.

        Parameters
        ----------
        name : str
            Human‑friendly name of the instrument.  Supported values include
            'sine', 'grand piano', 'electric piano', 'trumpet' and
            'saxophone'.  Case and surrounding whitespace are ignored.

        When a SoundFont and FluidSynth are available, the selected program
        number will be applied at the start of playback.  If the
        synthesiser cannot be initialised or no SoundFont is present, the
        player falls back to sine‑wave synthesis.
        """
        # Normalise the instrument name
        name = (name or 'sine').strip().lower()
        self.instrument = name
        # Reset synthesiser program when instrument changes
        if fluidsynth is None:
            return
        # Lazily initialise the synthesiser if needed
        if self.synth is None and name != 'sine':
            self._init_synth()
        # If the synthesiser and SoundFont are loaded, select the program
        if self.synth is not None and self.soundfont_id is not None:
            # Map instrument names to General MIDI program numbers
            program_map = {
                'grand piano': 0,
                'piano': 0,
                'electric piano': 4,
                'trumpet': 56,
                'saxophone': 64,
                'sax': 64,
                'sine': 0,
            }
            instr_key = self.instrument.strip().lower()
            prog = program_map.get(instr_key, 0)
            try:
                self.synth.program_select(0, self.soundfont_id, 0, prog)
            except Exception:
                pass

    def is_playing(self) -> bool:
        """Return True if a melody is currently playing."""
        return self._play_thread is not None and self._play_thread.is_alive()

    def stop(self) -> None:
        """Stop playback if something is playing."""
        if self.is_playing():
            self._stop_event.set()
            self._play_thread.join()
            self._stop_event.clear()
            self._play_thread = None

    def _synthesize_melody(self, melody: Melody) -> np.ndarray:
        """Generate a NumPy array containing the audio for a given melody.

        Depending on the selected instrument, this method will either
        synthesise sine waves directly (for the default 'Sine' instrument)
        or use FluidSynth to render MIDI notes with a SoundFont.  In the
        latter case the method sequentially triggers note‑on and note‑off
        events and collects the resulting audio samples.
        """
        # Fallback to sine wave synthesis if no instrument specified or fluidsynth unavailable
        # Determine whether to use sine synthesis.  We use sine if the
        # selected instrument is 'sine', FluidSynth is unavailable or
        # the synthesiser failed to initialise.
        use_sine = (self.instrument == 'sine' or fluidsynth is None or self.synth is None)
        beats_per_second = melody.tempo / 60.0
        if use_sine:
            samples = np.array([], dtype=np.float32)
            # Add a small release time (silence) between notes to avoid continuous tone
            release_sec = 0.05  # 50ms silence between notes
            release_samples = int(self.sample_rate * release_sec)
            for event in melody.events:
                freq = midi_to_freq(event.pitch)
                duration_seconds = event.duration / beats_per_second
                num_samples = int(self.sample_rate * duration_seconds)
                t = np.linspace(0, duration_seconds, num_samples, False)
                wave = np.sin(2 * np.pi * freq * t) * (event.velocity / 127.0)
                # Apply simple fade in/out to reduce clicks
                fade_len = min(int(0.01 * self.sample_rate), len(wave) // 2)
                if fade_len > 0:
                    fade = np.linspace(0, 1, fade_len)
                    wave[:fade_len] *= fade
                    wave[-fade_len:] *= fade[::-1]
                # Append the note and a brief silence to create separation
                if samples.size == 0:
                    samples = wave
                else:
                    samples = np.concatenate((samples, wave))
                if release_samples > 0:
                    samples = np.concatenate((samples, np.zeros(release_samples, dtype=np.float32)))
            # Normalize to avoid clipping
            if samples.size > 0:
                max_val = np.max(np.abs(samples))
                if max_val > 0:
                    samples = samples / max_val
            return samples.astype(np.float32)
        else:
            # Use FluidSynth for realistic instrument playback
            samples = np.array([], dtype=np.float32)
            # Ensure synthesiser is initialised
            if self.synth is None:
                self._init_synth()
            # Map instrument names to General MIDI program numbers.  The
            # mapping uses lowercase keys without surrounding whitespace.
            program_map = {
                'grand piano': 0,
                'piano': 0,
                'electric piano': 4,
                'trumpet': 56,
                'saxophone': 64,
                'sax': 64,
                'sine': 0,
            }
            instr_key = self.instrument.strip().lower()
            prog = program_map.get(instr_key, 0)
            # Select program on channel 0 if a SoundFont is loaded
            if self.soundfont_id is not None:
                try:
                    self.synth.program_select(0, self.soundfont_id, 0, prog)
                except Exception:
                    pass
            # Synthesize each note sequentially
            # Add a small release time (silence) between notes when using FluidSynth
            release_sec = 0.05  # 50ms release
            release_samples = int(self.sample_rate * release_sec)
            for event in melody.events:
                duration_seconds = event.duration / beats_per_second
                num_frames = int(self.sample_rate * duration_seconds)
                # Trigger note on
                try:
                    self.synth.noteon(0, event.pitch, event.velocity)
                except Exception:
                    pass
                # Collect stereo samples (FluidSynth outputs stereo).  Each call
                # returns a list of floats of length 2 * num_frames.
                if num_frames > 0:
                    try:
                        stereo = self.synth.get_samples(num_frames)
                    except Exception:
                        stereo = [0.0] * (num_frames * 2)
                    # Convert to mono by averaging left and right channels
                    left = np.array(stereo[::2], dtype=np.float32)
                    right = np.array(stereo[1::2], dtype=np.float32)
                    mono = (left + right) / 2.0
                else:
                    mono = np.array([], dtype=np.float32)
                # Trigger note off
                try:
                    self.synth.noteoff(0, event.pitch)
                except Exception:
                    pass
                # Append to full samples
                if samples.size == 0:
                    samples = mono
                else:
                    samples = np.concatenate((samples, mono))
                # Append release silence
                if release_samples > 0:
                    samples = np.concatenate((samples, np.zeros(release_samples, dtype=np.float32)))
            # Normalize
            if samples.size > 0:
                max_val = np.max(np.abs(samples))
                if max_val > 0:
                    samples = samples / max_val
            return samples.astype(np.float32)

    def _init_synth(self) -> None:
        """Initialise the FluidSynth synthesizer and load a SoundFont if available."""
        if fluidsynth is None:
            return
        # Create the synthesizer if not already created
        try:
            self.synth = fluidsynth.Synth(samplerate=self.sample_rate)
            # Start the synthesizer with the default audio driver.  Without
            # calling start(), program selection may fail and no sound will
            # be produced on some systems.  If this raises an exception we
            # fall back to offline rendering.
            try:
                self.synth.start()
            except Exception:
                # If start fails, we proceed without starting a driver; get_samples
                # will still work for offline rendering.
                pass
            # Load SoundFont if present
            if os.path.isfile(self.soundfont_path):
                self.soundfont_id = self.synth.sfload(self.soundfont_path)
            else:
                self.soundfont_id = None
        except Exception:
            self.synth = None
            self.soundfont_id = None

    def play_melody(self, melody: Melody) -> None:
        """Start playing a melody asynchronously.

        If a melody is already playing, it will be stopped before the new
        playback begins.  The audio is generated on the fly and played
        through the selected output device.  Use ``stop()`` to halt playback.
        """
        # Stop any existing playback
        self.stop()
        self.current_melody = melody

        def playback_thread():
            data = self._synthesize_melody(melody)
            # Use a callback to allow stopping mid‑playback
            def callback(outdata, frames, time, status):
                if self._stop_event.is_set():
                    raise sd.CallbackStop()
                # Determine how many samples to copy for this block
                start = callback.current_pos
                end = start + frames
                out_chunk = data[start:end]
                if len(out_chunk) < frames:
                    # Pad remainder with zeros
                    outdata[:len(out_chunk), 0] = out_chunk
                    outdata[len(out_chunk):, 0] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = out_chunk
                callback.current_pos = end
            callback.current_pos = 0
            # Stream parameters: mono audio, chosen device
            device = self.output_device
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=callback,
                device=device,
            ):
                self._stop_event.wait()  # Wait until callback stops or stop_event is set

        self._play_thread = threading.Thread(target=playback_thread, daemon=True)
        self._play_thread.start()
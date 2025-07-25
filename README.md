# Ear Trainer Application

This project implements a simple ear‑training desktop application in Python.
Users can practice melodic dictation by listening to randomly generated
melodies or melodies loaded from a library of MIDI/MusicXML files and then
reproducing them either manually or via automatic note detection.  The
application is built with PySide6 and supports both audio and MIDI input for
real‑time evaluation.

## Features

* Generate random melodies in a specified key, scale and length with
  configurable complexity and rhythmic density.
* Load existing melodies from MIDI or MusicXML files (place files in
  `ear_trainer/sample_melodies`).
* Mix random and actual melodies using an adjustable slider and control
  dataset size.
* Choose the key, scale and additional generation parameters such as
  complexity, melody length (number of notes), tempo (BPM), maximum
  interval size, maximum pitch span (in octaves), note density (how
  close notes are in time) and register (bass, middle or treble).
* Play melodies using a configurable instrument.  Available options
  include simple sine waves and General MIDI sounds (grand piano,
  electric piano, trumpet, saxophone) via FluidSynth【422143262296804†L95-L101】.  To
  enable realistic instrument playback, place a SoundFont file named
  `soundfont.sf2` (or `FluidR3_GM.sf2`) in the `ear_trainer` directory.  If
  no SoundFont is found or `pyfluidsynth` is unavailable, the app
  falls back to sine‑wave synthesis.
* Output melodies as MIDI events to any available port for playback in
  external software such as a DAW.
* Navigate through melodies manually (play/stop, next, previous) and
  shift the current melody up or down by octaves on the fly.
* Optional automatic note detection via MIDI input or audio (microphone) using
  `mido` and `aubio`.
* Display note names and success/failure feedback.

## Installation

Create and activate a virtual environment, then install dependencies:

```sh
python -m venv venv
source venv/bin/activate  # on Windows use venv\Scripts\activate
pip install -r requirements.txt
```

Some dependencies have native components:

* **PySide6** — Qt bindings for the GUI.  On macOS you can install it via
  `pip3 install pyside6`【965863941627289†L73-L87】.
* **sounddevice** — cross‑platform audio I/O bindings based on PortAudio;
  automatically installs PortAudio on macOS【932382819316275†L10-L15】【590407697250024†L29-L32】.
* **mido[ports‑rtmidi]** — MIDI messaging and ports【864807409754461†L92-L96】.
* **aubio** — real‑time pitch detection【99446480439423†L19-L23】.  On some systems you
  may need to install `libaubio` via a package manager.
* **pyfluidsynth** (optional) — to use a SoundFont synthesizer instead of
  sine‑waves【422143262296804†L95-L101】.

FluidSynth requires a SoundFont (`.sf2`) file to produce realistic instrument
sounds.  You can download a General MIDI SoundFont (such as
`FluidR3_GM.sf2`) and place it in the `ear_trainer` directory as
`soundfont.sf2` or `FluidR3_GM.sf2`.  If no SoundFont is found or
`pyfluidsynth` is not installed, the application will automatically
fall back to the basic sine‑wave synthesizer.

If certain packages cannot be installed in your environment, the application
will still run in manual mode (without automatic detection).

## Running the Application

Run the main script from the repository root:

```sh
python -m ear_trainer.main
```

The GUI will open with dataset settings, detection settings and playback
controls.  Adjust the random ratio and dataset size, select a key and scale,
then press **Play** to begin.  Use **Next** and **Prev** to navigate through
the dataset.  Enable **auto detection** and select a detection mode and
appropriate input device to have your performance evaluated in real time.

## Preparing a Melody Dataset

By default the application only loads a handful of example melodies from
the `ear_trainer/sample_melodies` directory.  To practise with a larger and
more diverse library, you can download a collection of MIDI files and
process them into a JSON dataset.  We recommend using the
**Lakh MIDI Dataset** or **GigaMIDI** collection, which contain
tens of thousands of MIDI files【307508166316718†L17-L31】【685118451599416†L83-L97】.
After unpacking the files into a folder, run the dataset preparation
script to extract melody segments and compute metadata:

```sh
# Example: process the first 1,000 MIDI files and split each into
# segments of 16 notes.  You can also segment by bars using
# --segment-measures.  Adjust --segment-length, --segment-measures and
# --max-files as needed.
    python -m ear_trainer.prepare_dataset \
        --input-dir /path/to/unpacked_midi \
        --output-json processed_melodies.json \
        --segment-length 16 \
        # --segment-measures 4 \
        --max-files 1000
```

The script extracts a monophonic melody line from each MIDI file by
selecting the highest note at each time position.  It can split the
melody into fixed‑length note chunks or into groups of measures and then
computes statistical metadata (average interval, pitch span, rhythmic
density, register and a composite complexity score).
It writes a list of segments to the specified JSON file.  You can then
load this dataset in the application by calling
`MelodyManager.load_from_json('processed_melodies.json')` before
building a practice set, or modify `main.py` accordingly.  See the
docstring in `prepare_dataset.py` for more details.

## Extending the Application

* **Improved synthesis** — Replace the simple sine‑wave generator in
  `audio_player.py` with a call to a MIDI synthesizer such as FluidSynth for
  better sound quality.
* **Rhythmic variety** — Expand the random generator in `melody_manager.py`
  to include more complex rhythms, rests and articulations.
* **Scoring and statistics** — Record user performance over time and
  provide feedback or progress charts.
* **Score display** — Use the music21 rendering facilities or a music
  engraving library to display staff notation instead of note names.

Contributions are welcome.  This project is intended as a learning tool and
starting point for more advanced ear‑training software.
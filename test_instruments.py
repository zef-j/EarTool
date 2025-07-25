import time
import fluidsynth
import os

SOUNDFONT_PATH = "soundfont.sf2"  # Change if needed

def test_play_note():
    if not os.path.exists(SOUNDFONT_PATH):
        print("ERROR: SoundFont file not found at", SOUNDFONT_PATH)
        return

    fs = fluidsynth.Synth()
    fs.start(driver="coreaudio")
    sfid = fs.sfload(SOUNDFONT_PATH)
    if sfid == -1:
        print("Error loading soundfont!")
        return

    print("SoundFont loaded. Testing General MIDI Acoustic Grand Piano...")

    fs.program_select(0, sfid, 0, 0)  # Channel 0, bank 0, preset 0
    print("Playing C4 (note 60)")
    fs.noteon(0, 60, 120)
    time.sleep(1)
    fs.noteoff(0, 60)
    time.sleep(0.2)

    print("Playing E4 (note 64)")
    fs.noteon(0, 64, 120)
    time.sleep(1)
    fs.noteoff(0, 64)
    time.sleep(0.2)

    print("Switching to Church Organ (preset 19)...")
    fs.program_select(0, sfid, 0, 19)
    print("Playing G4 (note 67)")
    fs.noteon(0, 67, 120)
    time.sleep(1)
    fs.noteoff(0, 67)

    print("Test finished.")
    fs.delete()

if __name__ == "__main__":
    test_play_note()

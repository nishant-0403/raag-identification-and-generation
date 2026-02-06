import librosa
import soundfile as sf
import os

INPUT_DIR = "chunks_45s"
OUTPUT_DIR = "harmonic_only"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, fname)

    # Load audio
    y, sr = librosa.load(path, sr=22050, mono=True)

    # Harmonic–Percussive Source Separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Save harmonic-only audio
    out_path = os.path.join(OUTPUT_DIR, fname)
    sf.write(out_path, y_harmonic, sr)

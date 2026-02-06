import librosa
import numpy as np
import os

def detect_sa(wav_path, sr=22050):
    # Load audio
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    # Pitch extraction using pYIN
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        frame_length=2048,
        hop_length=256
    )

    # Compute RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]

    # Keep frames that are either voiced OR energetic
    mask = (voiced_prob > 0.5) | (rms > np.percentile(rms, 60))
    f0_clean = f0[mask]

    # Drop NaNs
    f0_clean = f0_clean[~np.isnan(f0_clean)]

    if len(f0_clean) < 100:
        return None

    # Convert to cents (reference doesn't matter here)
    cents = 1200 * np.log2(f0_clean / np.min(f0_clean))

    # Fold into one octave
    cents_mod = np.mod(cents, 1200)

    # Histogram
    hist, bins = np.histogram(cents_mod, bins=360, range=(0, 1200))

    # Smooth histogram
    kernel = np.hanning(15)
    kernel /= kernel.sum()
    hist_smooth = np.convolve(hist, kernel, mode="same")

    # Peak detection
    peak_idx = np.argmax(hist_smooth)
    sa_cents = bins[peak_idx]

    # Convert back to Hz
    sa_hz = np.min(f0_clean) * (2 ** (sa_cents / 1200))

    return sa_hz


# ---------------- RUN TEST ----------------
if __name__ == "__main__":
    TEST_FILE = "harmonic_only"

    for fname in os.listdir(TEST_FILE):
        if fname.endswith(".wav"):
            path = os.path.join(TEST_FILE, fname)
            sa = detect_sa(path)

            print(f"{fname} → ", end="")
            if sa is None:
                print("Could not detect Sa")
            else:
                print(f"Sa ≈ {sa:.2f} Hz")

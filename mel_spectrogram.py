import os
import glob
import numpy as np
import librosa
import librosa.display

# =========================
# CONFIGURATION
# =========================
CHUNK_DIR = "all_chunks"           # Directory where chunk_*.wav files live
OUTPUT_DIR = "mel_specs"  # Where to save .npy files
SR = 22050                # Must match chunkify.py sr
N_MELS = 128              # Mel filterbank bins (height of spectrogram)
N_FFT = 2048              # FFT window size
HOP_LENGTH = 512          # Hop length in samples
FMIN = 20                 # Minimum frequency (Hz)
FMAX = 8000               # Maximum frequency (Hz)
FIXED_LENGTH = 128        # Fixed time dimension (frames) — pad/trim to this

# =========================
# SETUP
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")))

if not chunk_files:
    raise FileNotFoundError(
        f"No chunk_*.wav files found in '{CHUNK_DIR}'. "
        "Run chunkify.py first to generate chunks."
    )

print(f"Found {len(chunk_files)} chunk(s). Processing...\n")

# =========================
# PROCESS EACH CHUNK
# =========================
specs = []

for path in chunk_files:
    name = os.path.splitext(os.path.basename(path))[0]

    # Load audio (resample to target SR if needed)
    y, sr = librosa.load(path, sr=SR)

    # Compute mel spectrogram (power) — saved as-is for vocoder compatibility
    # The encoder applies log compression internally before the CNN
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # Pad or trim to fixed time dimension → shape: (N_MELS, FIXED_LENGTH)
    if mel.shape[1] < FIXED_LENGTH:
        pad_width = FIXED_LENGTH - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel = mel[:, :FIXED_LENGTH]

    # Add channel dimension for CNN → shape: (1, N_MELS, FIXED_LENGTH)
    # dtype float32 keeps file sizes reasonable
    mel_cnn = mel[np.newaxis, :, :].astype(np.float32)

    # Save individual .npy
    out_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(out_path, mel_cnn)

    specs.append(mel_cnn)

    print(f"  {name}: shape={mel_cnn.shape}  saved → {out_path}")

# =========================
# STACK INTO SINGLE DATASET
# =========================
# Final shape: (N_CHUNKS, 1, N_MELS, FIXED_LENGTH)
dataset = np.stack(specs, axis=0)
dataset_path = os.path.join(OUTPUT_DIR, "mel_dataset.npy")
np.save(dataset_path, dataset)

print(f"\nDataset shape : {dataset.shape}")
print(f"  Axis 0      : {dataset.shape[0]} chunks")
print(f"  Axis 1      : {dataset.shape[1]} channel(s)")
print(f"  Axis 2      : {dataset.shape[2]} mel bins  (frequency)")
print(f"  Axis 3      : {dataset.shape[3]} frames    (time)")
print(f"\nFull dataset saved → {dataset_path}")


# =========================
# QUICK USAGE EXAMPLE
# =========================
# Load the dataset in your CNN training script:
#
#   import numpy as np, torch
#   data = np.load("mel_specs/mel_dataset.npy")   # float32 power mel
#   # data.shape → (N, 1, 128, 128)
#
#   tensor = torch.from_numpy(data).float()
#   # The encoder applies log1p compression internally before the CNN —
#   # do NOT pre-convert to dB; pass the raw power tensor directly.

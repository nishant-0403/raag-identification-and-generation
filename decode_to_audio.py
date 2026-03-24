import torch
import numpy as np
import librosa
import soundfile as sf
import os


# =========================
# CONFIGURATION
# =========================
# Must match mel_spectrogram.py and encoder.py
SR          = 22050
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
FMIN        = 20
FMAX        = 8000
OUTPUT_DIR  = "reconstructed"


# =========================
# TENSOR → AUDIO
# =========================
def mel_tensor_to_audio(mel_hat, n_iter=64):
    """
    Converts a single decoder output tensor to a numpy audio waveform.

    Args:
        mel_hat : torch.Tensor of shape (1, N_MELS, T) or (N_MELS, T)
                  — power mel spectrogram, strictly positive (Softplus output)
        n_iter  : Griffin-Lim iterations. More = cleaner audio, slower.
                  64 is a good balance. Use 128 for final output.

    Returns:
        audio   : np.ndarray of shape (samples,) — waveform at SR
    """
    # Strip channel dim if present
    if mel_hat.dim() == 3:
        mel_hat = mel_hat.squeeze(0)       # (N_MELS, T)

    # Tensor → numpy, move to CPU if on GPU
    mel_power = mel_hat.detach().cpu().numpy().astype(np.float32)

    # Clip to strictly positive (Softplus guarantees this but just in case)
    mel_power = np.clip(mel_power, a_min=1e-10, a_max=None)

    # Power mel → audio via Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=n_iter,
        fmin=FMIN,
        fmax=FMAX,
    )

    return audio


# =========================
# BATCH → WAV FILES
# =========================
def save_decoder_output(mel_hat_batch, output_dir=OUTPUT_DIR, prefix="recon", n_iter=64):
    """
    Takes a full decoder output batch and saves each item as a .wav file.

    Args:
        mel_hat_batch : torch.Tensor of shape (B, 1, N_MELS, T)
                        — direct output of RagaDecoder.forward()
        output_dir    : folder to save .wav files into
        prefix        : filename prefix, e.g. "recon" → recon_0.wav, recon_1.wav
        n_iter        : Griffin-Lim iterations (64 default, 128 for best quality)

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for i in range(mel_hat_batch.size(0)):
        audio = mel_tensor_to_audio(mel_hat_batch[i], n_iter=n_iter)
        path  = os.path.join(output_dir, f"{prefix}_{i}.wav")
        sf.write(path, audio, SR)
        saved.append(path)
        print(f"  Saved: {path}  ({len(audio)/SR:.2f}s)")

    return saved


# =========================
# COMPARE ORIGINAL VS RECONSTRUCTION
# =========================
def save_comparison(mel_original, mel_hat, output_dir=OUTPUT_DIR, idx=0, n_iter=64):
    """
    Saves both the original and reconstructed mel as audio side by side.
    Useful during training to hear how reconstruction quality improves.

    Args:
        mel_original : (1, N_MELS, T) or (N_MELS, T) — ground truth power mel
        mel_hat      : (1, N_MELS, T) or (N_MELS, T) — decoder output
        idx          : sample index label for the filename
    """
    os.makedirs(output_dir, exist_ok=True)

    orig_audio  = mel_tensor_to_audio(mel_original, n_iter=n_iter)
    recon_audio = mel_tensor_to_audio(mel_hat,      n_iter=n_iter)

    orig_path  = os.path.join(output_dir, f"original_{idx}.wav")
    recon_path = os.path.join(output_dir, f"reconstructed_{idx}.wav")

    sf.write(orig_path,  orig_audio,  SR)
    sf.write(recon_path, recon_audio, SR)

    print(f"  Original      → {orig_path}")
    print(f"  Reconstructed → {recon_path}")

    return orig_path, recon_path


# =========================
# LOAD .NPY AND CONVERT
# =========================
def npy_to_audio(npy_path, output_dir=OUTPUT_DIR, n_iter=64):
    """
    Convenience function: loads a saved .npy mel file (from mel_spectrogram.py)
    and converts it directly to audio. Useful for verifying mel_spectrogram.py
    output before any model is involved.

    Args:
        npy_path   : path to a .npy file of shape (1, N_MELS, T)
    """
    mel = np.load(npy_path).astype(np.float32)     # (1, N_MELS, T)
    mel_tensor = torch.from_numpy(mel)

    os.makedirs(output_dir, exist_ok=True)
    audio = mel_tensor_to_audio(mel_tensor, n_iter=n_iter)

    name = os.path.splitext(os.path.basename(npy_path))[0]
    out_path = os.path.join(output_dir, f"{name}_audio.wav")
    sf.write(out_path, audio, SR)
    print(f"  {npy_path} → {out_path}  ({len(audio)/SR:.2f}s)")
    return out_path


# =========================
# SANITY CHECK
# =========================
if __name__ == "__main__":
    from encoder import RagaEncoder, N_MELS, FIXED_LENGTH
    from decoder import RagaDecoder

    print("=== decode_to_audio sanity check ===\n")

    # Simulate a batch of power mel specs (as if loaded from mel_spectrogram.py)
    B   = 2
    mel = torch.rand(B, 1, N_MELS, FIXED_LENGTH).pow(2) * 10.0

    # Run through encoder → decoder
    encoder = RagaEncoder()
    decoder = RagaDecoder()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        z_seq, z_global = encoder(mel)
        mel_hat         = decoder(z_seq)

    print(f"Decoder output shape : {tuple(mel_hat.shape)}")
    print(f"Value range          : [{mel_hat.min():.4f}, {mel_hat.max():.4f}]")
    print(f"All positive         : {(mel_hat > 0).all().item()}")
    print()

    # Save reconstructed audio
    print("Saving reconstructed audio...")
    saved = save_decoder_output(mel_hat, output_dir=OUTPUT_DIR, prefix="recon")

    # Save original vs reconstruction comparison for sample 0
    print("\nSaving comparison (original vs reconstructed)...")
    save_comparison(mel[0], mel_hat[0], output_dir=OUTPUT_DIR, idx=0)

    print(f"\nAll files saved to '{OUTPUT_DIR}/'")
    print("\nNOTE: Audio from untrained weights will sound like noise.")
    print("      Run this again after training to hear real reconstruction quality.")

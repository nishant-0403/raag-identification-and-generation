import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================
# CONFIGURATION
# =========================
# These must match mel_spectrogram.py and be consistent with decoder.py
N_MELS       = 128    # Frequency bins (height of mel spec)
FIXED_LENGTH = 128    # Time frames  (width of mel spec)
D            = 256    # Latent dimension for z_seq and z_global
T_PRIME      = 16     # Number of microchunks in z_seq (time resolution of latent)
CNN_CHANNELS = [32, 64, 128, 256]  # Progressive channel sizes
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 2
N_HEADS      = 8      # Transformer attention heads (D must be divisible by N_HEADS)
N_LAYERS     = 4      # Transformer encoder layers
DROPOUT      = 0.1


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =========================
# CNN FEATURE EXTRACTOR
# =========================
class CNNFeatureExtractor(nn.Module):
    """
    Input:  (B, 1, N_MELS, FIXED_LENGTH)   e.g. (B, 1, 128, 128)
    Output: (B, C_out, H_out, W_out)        where C_out = CNN_CHANNELS[-1]

    Each block: Conv2d → BatchNorm → GELU → MaxPool(2×2)
    After 4 blocks with 128×128 input:
      H_out = 128 / 2^4 = 8
      W_out = 128 / 2^4 = 8  (will be projected to T_PRIME along time axis)
    """
    def __init__(self, in_channels=1, channel_list=CNN_CHANNELS, dropout=DROPOUT):
        super().__init__()

        layers = []
        c_in = in_channels
        for c_out in channel_list:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
                nn.Dropout2d(dropout / 2),
                nn.MaxPool2d(2, 2),
            ]
            c_in = c_out

        self.net = nn.Sequential(*layers)
        self.out_channels = channel_list[-1]

    def forward(self, x):
        return self.net(x)   # (B, C_out, H_out, W_out)


# =========================
# ENCODER
# =========================
class RagaEncoder(nn.Module):
    """
    Full pipeline:
        mel (B,1,N_MELS,T) → CNN → flatten freq → LSTM → Transformer → z_seq, z_global

    z_seq   : (B, T_PRIME, D)   — time-dependent latent sequence
    z_global: (B, D)             — raga-level summary (mean-pooled z_seq)
    """
    def __init__(
        self,
        n_mels=N_MELS,
        fixed_length=FIXED_LENGTH,
        d_model=D,
        t_prime=T_PRIME,
        cnn_channels=CNN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.d_model  = d_model
        self.t_prime  = t_prime

        # --- CNN ---
        self.cnn = CNNFeatureExtractor(in_channels=1, channel_list=cnn_channels, dropout=dropout)

        # After CNN: (B, C_out, H_out, W_out)
        # H_out = n_mels  / 2^len(cnn_channels)
        # W_out = fixed_length / 2^len(cnn_channels)
        n_pool = len(cnn_channels)
        self._h_out = n_mels       // (2 ** n_pool)   # 8 with defaults
        self._w_out = fixed_length // (2 ** n_pool)   # 8 with defaults
        cnn_flat_dim = cnn_channels[-1] * self._h_out # collapse freq; keep time

        # --- Project CNN output to LSTM input size ---
        self.cnn_proj = nn.Linear(cnn_flat_dim, lstm_hidden)

        # --- Bidirectional LSTM over time axis ---
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Bidirectional doubles hidden dim → project back to d_model
        self.lstm_proj = nn.Linear(lstm_hidden * 2, d_model)

        # --- Temporal downsampling to T_PRIME ---
        # Uses adaptive average pool along the time axis
        # Input after LSTM: (B, W_out, d_model) → pool to (B, T_PRIME, d_model)

        # --- Transformer Encoder ---
        self.pos_enc = PositionalEncoding(d_model, max_len=t_prime + 4, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Layer norm on output ---
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel):
        """
        Args:
            mel: (B, 1, N_MELS, T)   — mel spectrogram batch

        Returns:
            z_seq   : (B, T_PRIME, D)
            z_global: (B, D)
        """
        B = mel.size(0)

        # Log-compress power mel before CNN.
        # Power values span many orders of magnitude — log1p compresses them
        # into a range the conv filters can handle uniformly.
        # log1p(x) = log(1+x), safe for x=0 (silence), no epsilon needed.
        x = torch.log1p(mel)

        # CNN: (B, 1, N_MELS, T) → (B, C_out, H_out, W_out)
        x = self.cnn(x)

        # Reshape: treat W_out as time steps, flatten (C_out × H_out) as features
        # (B, C_out, H_out, W_out) → (B, W_out, C_out * H_out)
        x = x.permute(0, 3, 1, 2).reshape(B, self._w_out, -1)

        # Project to LSTM input dim: (B, W_out, lstm_hidden)
        x = F.gelu(self.cnn_proj(x))

        # LSTM: (B, W_out, lstm_hidden) → (B, W_out, lstm_hidden*2)
        x, _ = self.lstm(x)
        x = self.lstm_proj(x)   # → (B, W_out, d_model)

        # Downsample time axis to T_PRIME via adaptive avg pool
        # pool expects (B, C, L) format
        x = x.permute(0, 2, 1)                                  # (B, d_model, W_out)
        x = F.adaptive_avg_pool1d(x, self.t_prime)              # (B, d_model, T_PRIME)
        x = x.permute(0, 2, 1)                                  # (B, T_PRIME, d_model)

        # Positional encoding + Transformer
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)

        # z_seq and z_global
        z_seq    = x                          # (B, T_PRIME, D)
        z_global = x.mean(dim=1)             # (B, D)  — mean pool over time

        return z_seq, z_global

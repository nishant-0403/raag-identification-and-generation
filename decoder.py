import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from encoder import (
    D, T_PRIME, CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS,
    N_HEADS, N_LAYERS, DROPOUT, N_MELS, FIXED_LENGTH,
    PositionalEncoding,
)


# =========================
# TRANSPOSED CNN UPSAMPLER
# =========================
class CNNUpsampler(nn.Module):
    """
    Mirrors CNNFeatureExtractor with transposed convolutions.

    Input:  (B, C_in, H_in, W_in)    e.g. (B, 256, 8, 8)
    Output: (B, 1,    N_MELS, T)     e.g. (B, 1,   128, 128)

    Each block: ConvTranspose2d(stride=2) → BatchNorm → GELU
    Final block uses Softplus (keeps output strictly positive — required for
    power mel spectrograms passed directly to a vocoder).
    """
    def __init__(self, channel_list=CNN_CHANNELS, dropout=DROPOUT):
        super().__init__()

        # Reverse the channel list for upsampling
        reversed_ch = list(reversed(channel_list))   # [256, 128, 64, 32]

        layers = []
        for i, (c_in, c_out) in enumerate(zip(reversed_ch, reversed_ch[1:] + [1])):
            is_last = (i == len(reversed_ch) - 1)
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            ]
            if not is_last:
                layers += [
                    nn.BatchNorm2d(c_out),
                    nn.GELU(),
                    nn.Dropout2d(dropout / 2),
                ]
            else:
                layers += [nn.Softplus()]  # Output strictly positive — power mel scale

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)   # (B, 1, N_MELS, T)


# =========================
# DECODER
# =========================
class RagaDecoder(nn.Module):
    """
    Inverts the encoder:
        z_seq (B, T_PRIME, D) → Transformer → LSTM → reshape → CNN upsample → mel̂

    Output mel̂ : (B, 1, N_MELS, FIXED_LENGTH)  — power mel spectrogram (strictly positive),
                  ready to pass directly to Griffin-Lim or HiFi-GAN vocoder.
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

        self.d_model      = d_model
        self.t_prime      = t_prime
        self.lstm_hidden  = lstm_hidden

        n_pool   = len(cnn_channels)
        self._h_out = n_mels       // (2 ** n_pool)   # 8
        self._w_out = fixed_length // (2 ** n_pool)   # 8

        # --- Transformer Decoder (using encoder-style, no cross-attention needed) ---
        self.pos_enc = PositionalEncoding(d_model, max_len=t_prime + 4, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # --- Upsample T_PRIME → W_out (reverse of adaptive_avg_pool) ---
        # We use a learned linear interpolation via Linear projection
        self.time_upsample = nn.Linear(t_prime, self._w_out)

        # --- Project d_model → lstm_hidden for LSTM ---
        self.lstm_in_proj = nn.Linear(d_model, lstm_hidden)

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_out_proj = nn.Linear(lstm_hidden * 2, cnn_channels[-1] * self._h_out)

        # --- CNN Upsampler ---
        self.cnn_up = CNNUpsampler(channel_list=cnn_channels, dropout=dropout)

    def forward(self, z_seq):
        """
        Args:
            z_seq: (B, T_PRIME, D)

        Returns:
            mel_hat: (B, 1, N_MELS, FIXED_LENGTH)  — values in [0, 1]
        """
        B = z_seq.size(0)

        # Transformer over latent sequence
        x = self.pos_enc(z_seq)
        x = self.transformer(x)
        x = self.norm(x)                        # (B, T_PRIME, D)

        # Upsample time: T_PRIME → W_out
        # permute to (B, D, T_PRIME) → linear → (B, D, W_out) → permute back
        x = x.permute(0, 2, 1)                  # (B, D, T_PRIME)
        x = self.time_upsample(x)               # (B, D, W_out)
        x = x.permute(0, 2, 1)                  # (B, W_out, D)

        # Project and run through LSTM
        x = F.gelu(self.lstm_in_proj(x))        # (B, W_out, lstm_hidden)
        x, _ = self.lstm(x)                     # (B, W_out, lstm_hidden*2)
        x = self.lstm_out_proj(x)               # (B, W_out, C_out * H_out)
        x = F.gelu(x)

        # Reshape into CNN feature map
        x = x.reshape(B, self._w_out, CNN_CHANNELS[-1], self._h_out)
        x = x.permute(0, 2, 3, 1)              # (B, C_out, H_out, W_out)

        # CNN upsample to full mel resolution
        mel_hat = self.cnn_up(x)               # (B, 1, N_MELS, FIXED_LENGTH) — power mel

        return mel_hat


# =========================
# RECONSTRUCTION LOSS
# =========================
class ReconstructionLoss(nn.Module):
    """
    Loss computed in log-power space so that quiet frames (low power)
    are weighted fairly against loud harmonic peaks.

    L_recon = α * L1(log1p(mel), log1p(mel̂)) + (1-α) * L2(log1p(mel), log1p(mel̂))

    Both ground truth and prediction are log-compressed before computing
    the loss — this matches the encoder's internal compression and means
    the loss surface is well-conditioned across the full dynamic range.
    """
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, mel, mel_hat):
        """
        Args:
            mel    : (B, 1, N_MELS, T)  — ground truth power mel
            mel_hat: (B, 1, N_MELS, T)  — reconstructed power mel

        Returns:
            loss (scalar), l1 (scalar), l2 (scalar)
        """
        log_mel     = torch.log1p(mel)
        log_mel_hat = torch.log1p(mel_hat)

        l1   = self.l1(log_mel_hat, log_mel)
        l2   = self.l2(log_mel_hat, log_mel)
        loss = self.alpha * l1 + (1 - self.alpha) * l2
        return loss, l1, l2


# =========================
# QUICK SANITY CHECK
# =========================
if __name__ == "__main__":
    from encoder import RagaEncoder

    B = 4
    # Simulate realistic power mel values (non-negative, sparse like real audio)
    mel = torch.rand(B, 1, N_MELS, FIXED_LENGTH).pow(2) * 10.0

    encoder = RagaEncoder()
    decoder = RagaDecoder()
    loss_fn = ReconstructionLoss()

    z_seq, z_global = encoder(mel)
    mel_hat         = decoder(z_seq)
    loss, l1, l2    = loss_fn(mel, mel_hat)

    print("=== Shape check ===")
    print(f"mel      : {tuple(mel.shape)}")
    print(f"z_seq    : {tuple(z_seq.shape)}   ← (B, T_PRIME={T_PRIME}, D={D})")
    print(f"z_global : {tuple(z_global.shape)} ← (B, D={D})")
    print(f"mel_hat  : {tuple(mel_hat.shape)}  ← power mel, strictly positive (Softplus output)")
    print()
    print("=== Reconstruction loss ===")
    print(f"L_recon  : {loss.item():.4f}")
    print(f"  L1     : {l1.item():.4f}")
    print(f"  L2     : {l2.item():.4f}")
    print()

    total_params = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
    enc_params   = sum(p.numel() for p in encoder.parameters())
    dec_params   = sum(p.numel() for p in decoder.parameters())
    print("=== Parameter counts ===")
    print(f"Encoder  : {enc_params:,}")
    print(f"Decoder  : {dec_params:,}")
    print(f"Total    : {total_params:,}")

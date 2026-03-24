import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from encoder import (
    D, T_PRIME, N_HEADS, N_LAYERS, DROPOUT,
    PositionalEncoding,
)


# =========================
# CONFIGURATION
# =========================
GEN_LAYERS   = 4      # Transformer layers in generator (can differ from encoder)
GEN_HEADS    = 8      # Attention heads
GEN_FF_DIM   = 1024   # Feedforward dim inside transformer
MAX_HISTORY  = 8      # Maximum number of past z_seq chunks to condition on


# =========================
# GENERATOR
# =========================
class RagaGenerator(nn.Module):
    """
    Autoregressive Transformer that predicts the next phrase's z_seq
    given a history of past z_seq vectors and the raga-level z_global.

    Input:
        z_seq_history : (B, K, T_PRIME, D)  — K past phrase encodings
        z_global      : (B, D)              — raga identity vector

    Output:
        z_seq_next    : (B, T_PRIME, D)     — predicted next phrase encoding

    Architecture:
        1. Flatten history: (B, K, T_PRIME, D) → (B, K*T_PRIME, D)
        2. Prepend z_global as a conditioning token: (B, 1 + K*T_PRIME, D)
        3. Causal Transformer (masked self-attention) over the sequence
        4. Extract last T_PRIME tokens → z_seq_next

    The causal mask ensures each position can only attend to past positions,
    which is required for autoregressive generation at inference time.
    """
    def __init__(
        self,
        d_model    = D,
        t_prime    = T_PRIME,
        max_history= MAX_HISTORY,
        n_heads    = GEN_HEADS,
        n_layers   = GEN_LAYERS,
        ff_dim     = GEN_FF_DIM,
        dropout    = DROPOUT,
    ):
        super().__init__()

        self.d_model     = d_model
        self.t_prime     = t_prime
        self.max_history = max_history
        max_seq_len      = 1 + max_history * t_prime  # global token + history tokens

        # --- Project z_global → d_model (already D but explicit for clarity) ---
        self.global_proj = nn.Linear(d_model, d_model)

        # --- Positional encoding over the full flattened sequence ---
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 4, dropout=dropout)

        # --- Causal Transformer ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

        # --- Output projection → predict z_seq_next ---
        self.output_proj = nn.Linear(d_model, d_model)

    def _causal_mask(self, seq_len, device):
        """Upper-triangular mask — prevents attending to future positions."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, z_seq_history, z_global):
        """
        Args:
            z_seq_history : (B, K, T_PRIME, D)  — K <= MAX_HISTORY past phrases
            z_global      : (B, D)

        Returns:
            z_seq_next    : (B, T_PRIME, D)
        """
        B, K, T, D = z_seq_history.shape
        device      = z_seq_history.device

        # Flatten history: (B, K, T, D) → (B, K*T, D)
        history_flat = z_seq_history.reshape(B, K * T, D)

        # Project z_global and prepend as conditioning token: (B, 1, D)
        global_token = F.gelu(self.global_proj(z_global)).unsqueeze(1)

        # Full sequence: [z_global | z_seq_1 | z_seq_2 | ... | z_seq_K]
        # Shape: (B, 1 + K*T, D)
        seq = torch.cat([global_token, history_flat], dim=1)

        # Positional encoding
        seq = self.pos_enc(seq)

        # Causal mask over sequence
        seq_len   = seq.size(1)
        causal    = self._causal_mask(seq_len, device)

        # TransformerDecoder needs memory — we use seq as both tgt and memory
        # (self-conditioned generation, no separate encoder memory needed here)
        out = self.transformer(tgt=seq, memory=seq, tgt_mask=causal)
        out = self.norm(out)

        # Extract the last T_PRIME positions → predicted next z_seq
        z_seq_next = self.output_proj(out[:, -T:, :])   # (B, T_PRIME, D)

        return z_seq_next


# =========================
# TRANSITION LOSS
# =========================
class TransitionLoss(nn.Module):
    """
    L_transition = MSE(z_seq_pred, z_seq_actual)

    Trains the generator to predict the actual next phrase encoding.
    MSE in latent space is appropriate here since z_seq values are
    unbounded real numbers (unlike the mel which needed log compression).
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, z_seq_pred, z_seq_actual):
        """
        Args:
            z_seq_pred  : (B, T_PRIME, D) — generator output
            z_seq_actual: (B, T_PRIME, D) — encoder output for the actual next phrase

        Returns:
            loss (scalar)
        """
        return self.mse(z_seq_pred, z_seq_actual)


# =========================
# SANITY CHECK
# =========================
if __name__ == "__main__":
    from encoder import RagaEncoder, FIXED_LENGTH, N_MELS

    B = 4
    K = 3   # 3 past phrases as history

    # Simulate encoder outputs for K+1 phrases
    encoder   = RagaEncoder()
    generator = RagaGenerator()
    loss_fn   = TransitionLoss()

    # Fake mel batch: (B*(K+1), 1, N_MELS, T)
    mel_all = torch.rand(B * (K + 1), 1, N_MELS, FIXED_LENGTH).pow(2) * 10.0

    with torch.no_grad():
        z_seq_all, z_global_all = encoder(mel_all)

    # Reshape into history and target
    z_seq_all    = z_seq_all.reshape(B, K + 1, T_PRIME, D)
    z_global_all = z_global_all.reshape(B, K + 1, D)

    z_seq_history = z_seq_all[:, :K, :, :]    # (B, K, T_PRIME, D) — past K phrases
    z_seq_target  = z_seq_all[:, K,  :, :]    # (B, T_PRIME, D)    — actual next phrase
    z_global      = z_global_all[:, 0, :]     # (B, D) — use first phrase's global

    # Generate next phrase
    z_seq_pred = generator(z_seq_history, z_global)
    loss       = loss_fn(z_seq_pred, z_seq_target)

    print("=== Generator shape check ===")
    print(f"z_seq_history : {tuple(z_seq_history.shape)}")
    print(f"z_global      : {tuple(z_global.shape)}")
    print(f"z_seq_pred    : {tuple(z_seq_pred.shape)}")
    print(f"z_seq_target  : {tuple(z_seq_target.shape)}")
    print()
    print(f"Transition loss (untrained): {loss.item():.4f}")
    print()

    gen_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator params: {gen_params:,}")

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CONFIGURATION
# =========================
TEMPERATURE  = 0.07    # Softmax temperature — lower = sharper contrast
                       # 0.07 is standard for NT-Xent (SimCLR paper value)
EMBED_DIM    = 256     # Must match D in encoder.py


# =========================
# PROJECTION HEAD
# =========================
class ProjectionHead(nn.Module):
    """
    Small MLP applied to z_global before contrastive loss.

    This is standard practice (SimCLR, MoCo): the encoder learns a rich
    general-purpose z_global, while the projection head absorbs the
    distortion caused by the contrastive objective — so the contrastive
    loss doesn't overfit z_global itself to just raga identity.

    At inference / identification time, use z_global directly (before
    this head), not the projected output.

    Architecture: Linear → BN → GELU → Linear → L2-normalize
    Input/output dim both = EMBED_DIM so z_global shape is preserved.
    """
    def __init__(self, in_dim=EMBED_DIM, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        # z: (B, in_dim)
        out = self.net(z)
        return F.normalize(out, dim=-1)   # L2-normalize → unit hypersphere


# =========================
# NT-XENT LOSS  (Normalized Temperature-scaled Cross Entropy)
# =========================
class NTXentLoss(nn.Module):
    """
    Contrastive loss applied on z_global embeddings.

    Expects a batch where samples are grouped by raga — each raga has
    exactly `n_views` phrases in the batch (default 2, i.e. pairs).

    Batch layout for n_views=2, B total samples:
        [phrase_A_raga1, phrase_A_raga2, ..., phrase_A_ragaN,
         phrase_B_raga1, phrase_B_raga2, ..., phrase_B_ragaN]

    For each anchor, its positive is the other view of the same raga.
    All other samples in the batch are negatives.

    Loss drives:
        same raga  → embeddings close  (high cosine similarity)
        diff raga  → embeddings far    (low cosine similarity)
    """
    def __init__(self, temperature=TEMPERATURE, n_views=2):
        super().__init__()
        self.temperature = temperature
        self.n_views     = n_views

    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, out_dim) — L2-normalized projected z_globals.
                        B must be divisible by n_views.
                        First B//n_views rows = view 1, next B//n_views = view 2, etc.

        Returns:
            loss (scalar)
        """
        B      = embeddings.size(0)
        N      = B // self.n_views
        device = embeddings.device

        # Cosine similarity matrix (B, B) — embeddings are L2-normalized
        # so dot product = cosine similarity
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Suppress self-similarity by subtracting a large constant on the diagonal.
        # Avoids -inf which causes NaN in log_softmax when gradients flow back.
        sim = sim - torch.eye(B, device=device) * 1e9

        # Positive pair labels: for n_views=2 with layout [view0 | view1],
        # row i (view0) has its positive at column i+N, and vice versa.
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),  # view0 rows → view1 cols
            torch.arange(0, N,     device=device),  # view1 rows → view0 cols
        ])

        # Cross-entropy: for each row, classify which column is the positive.
        # This is numerically stable and equivalent to NT-Xent.
        return F.cross_entropy(sim, labels)


# =========================
# CONVENIENCE WRAPPER
# =========================
class ContrastiveLoss(nn.Module):
    """
    Full contrastive module: ProjectionHead + NTXentLoss.

    Usage during training:
        contrastive = ContrastiveLoss()
        ...
        _, z_global = encoder(mel_batch)       # (B, D)
        loss = contrastive(z_global)

    Batch must be built so same-raga phrases are paired — see
    RagaContrastiveDataset in this file for a ready-made DataLoader.
    """
    def __init__(self, embed_dim=EMBED_DIM, proj_hidden=512, proj_out=128,
                 temperature=TEMPERATURE, n_views=2):
        super().__init__()
        self.proj_head = ProjectionHead(embed_dim, proj_hidden, proj_out)
        self.criterion = NTXentLoss(temperature, n_views)

    def forward(self, z_global):
        """
        Args:
            z_global: (B, D) — raw z_global from encoder, NOT yet projected

        Returns:
            loss (scalar)
        """
        projected = self.proj_head(z_global)   # (B, proj_out), L2-normalized
        return self.criterion(projected)


# =========================
# DATASET HELPER
# =========================
class RagaContrastiveDataset(torch.utils.data.Dataset):
    """
    Wraps a mel spectrogram dataset for contrastive learning.

    Expects:
        mels  : np.ndarray of shape (N, 1, N_MELS, T)  — power mel specs
        labels: np.ndarray of shape (N,)                — integer raga IDs

    Each __getitem__ returns TWO randomly selected phrases from the SAME
    raga (a positive pair). The DataLoader then stacks these into:
        view1: (batch, 1, N_MELS, T)
        view2: (batch, 1, N_MELS, T)

    Call build_batch(view1, view2) to interleave into the layout
    NTXentLoss expects: [all_view1 | all_view2].
    """
    def __init__(self, mels, labels):
        import numpy as np
        self.mels   = torch.from_numpy(mels).float()
        self.labels = labels

        # Build index: raga_id → list of sample indices
        self.raga_to_idx = {}
        for i, lbl in enumerate(labels):
            self.raga_to_idx.setdefault(int(lbl), []).append(i)

        # Only keep ragas that have at least 2 phrases
        self.valid_ragas = [r for r, idxs in self.raga_to_idx.items() if len(idxs) >= 2]

    def __len__(self):
        return len(self.valid_ragas)

    def __getitem__(self, idx):
        import random
        raga  = self.valid_ragas[idx]
        i, j  = random.sample(self.raga_to_idx[raga], 2)
        return self.mels[i], self.mels[j], raga

    @staticmethod
    def build_batch(view1, view2):
        """
        Interleave two views into the layout NTXentLoss expects.

        Args:
            view1, view2: (B, 1, N_MELS, T) tensors from DataLoader

        Returns:
            (2B, 1, N_MELS, T) — [view1_batch | view2_batch]
        """
        return torch.cat([view1, view2], dim=0)


# =========================
# SANITY CHECK
# =========================
if __name__ == "__main__":
    import numpy as np

    B_per_view = 8   # 8 raga instances × 2 views = 16 total
    D          = EMBED_DIM

    # Simulate z_global from encoder — build so same-raga pairs are similar
    # First 8 = view1, next 8 = view2
    base    = torch.randn(B_per_view, D)
    view1   = F.normalize(base + 0.1 * torch.randn(B_per_view, D), dim=-1)
    view2   = F.normalize(base + 0.1 * torch.randn(B_per_view, D), dim=-1)
    z_global = torch.cat([view1, view2], dim=0)  # (16, D)

    contrastive = ContrastiveLoss()
    loss = contrastive(z_global)

    print("=== Contrastive loss sanity check ===")
    print(f"z_global shape : {tuple(z_global.shape)}")
    print(f"Projected shape: {tuple(contrastive.proj_head(z_global).shape)}")
    print(f"Loss (similar pairs, should be LOW)  : {loss.item():.4f}")

    # Now test with random (dissimilar) pairs — loss should be higher
    z_random = F.normalize(torch.randn(16, D), dim=-1)
    loss_rand = contrastive(z_random)
    print(f"Loss (random pairs, should be HIGHER): {loss_rand.item():.4f}")

    print()
    proj_params = sum(p.numel() for p in contrastive.parameters())
    print(f"Projection head params: {proj_params:,}")

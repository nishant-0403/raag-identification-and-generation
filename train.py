import os
import glob
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from encoder     import RagaEncoder,    N_MELS, FIXED_LENGTH, D, T_PRIME
from decoder     import RagaDecoder,    ReconstructionLoss
from contrastive import ContrastiveLoss
from generator   import RagaGenerator,  TransitionLoss, MAX_HISTORY


# =========================
# TRAINING CONFIGURATION
# =========================
MEL_DIR        = "mel_specs"        # Folder containing chunk_*.npy files
LABEL_FILE     = "labels.json"      # Maps npy filename → raga integer ID (see below)
CHECKPOINT_DIR = "checkpoints"
AUDIO_DIR      = "reconstructed"    # Periodic audio reconstructions during training

EPOCHS         = 200
BATCH_SIZE     = 8                  # Smaller batch = more gradient steps per epoch
LR             = 1e-4               # Slightly lower but decays much more slowly now
WEIGHT_DECAY   = 1e-4

# Loss weights from your pipeline spec: L = λ1*L_recon + λ2*L_contrastive + λ3*L_transition
LAMBDA_RECON       = 1.0
LAMBDA_CONTRASTIVE = 0.5
LAMBDA_TRANSITION  = 0.5

SAVE_EVERY         = 25    # Save checkpoint every N epochs
AUDIO_EVERY        = 50    # Save reconstructed audio every N epochs
K_HISTORY          = 3     # Number of past phrases to feed the generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATASET
# =========================
class RagaDataset(Dataset):
    """
    Loads mel spectrogram .npy files and their raga labels.

    Expected folder structure:
        mel_specs/
            chunk_0.npy
            chunk_1.npy
            ...

    Expected labels.json format:
        {
            "chunk_0": 0,
            "chunk_1": 0,
            "chunk_2": 1,
            ...
        }
    where the integer is a raga ID (0-indexed).

    To create labels.json for a single raga (Desh):
        python3 -c "
        import glob, json
        files = sorted(glob.glob('mel_specs/chunk_*.npy'))
        labels = {os.path.splitext(os.path.basename(f))[0]: 0 for f in files}
        json.dump(labels, open('labels.json','w'), indent=2)
        "

    When you add more ragas, assign each a new integer ID.
    """
    def __init__(self, mel_dir=MEL_DIR, label_file=LABEL_FILE):
        with open(label_file) as f:
            label_map = json.load(f)

        self.samples = []
        #for npy_path in sorted(glob.glob(os.path.join(mel_dir, "chunk_*.npy"))):
        for npy_path in sorted(glob.glob(os.path.join(mel_dir, "*chunk*.npy"))):
            name = os.path.splitext(os.path.basename(npy_path))[0]
            if name in label_map:
                mel = np.load(npy_path).astype(np.float32)   # (1, N_MELS, T)
                self.samples.append((mel, label_map[name]))

        if not self.samples:
            raise RuntimeError(
                f"No samples found. Check that '{mel_dir}' contains chunk_*.npy "
                f"files and '{label_file}' has matching entries."
            )

        # Group indices by raga for contrastive + transition sampling
        self.raga_to_idx = {}
        for i, (_, lbl) in enumerate(self.samples):
            self.raga_to_idx.setdefault(lbl, []).append(i)

        self.n_ragas = len(self.raga_to_idx)
        print(f"Dataset: {len(self.samples)} chunks across {self.n_ragas} raga(s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, label = self.samples[idx]
        return torch.from_numpy(mel), label


# =========================
# BATCH BUILDERS
# =========================
def build_contrastive_batch(dataset, batch_size, device):
    """
    Samples batch_size raga pairs (2 phrases per raga) for contrastive loss.
    Returns two mel batches of shape (batch_size, 1, N_MELS, T).
    """
    import random
    valid_ragas = [r for r, idxs in dataset.raga_to_idx.items() if len(idxs) >= 2]

    if len(valid_ragas) < 2:
        return None, None   # Need at least 2 ragas for meaningful contrastive loss

    ragas   = random.choices(valid_ragas, k=batch_size)
    view1   = []
    view2   = []

    for raga in ragas:
        i, j = np.random.choice(dataset.raga_to_idx[raga], size=2, replace=False)
        view1.append(dataset.samples[i][0])
        view2.append(dataset.samples[j][0])

    view1 = torch.from_numpy(np.stack(view1)).to(device)
    view2 = torch.from_numpy(np.stack(view2)).to(device)
    return view1, view2


def build_transition_batch(dataset, batch_size, k_history, device):
    """
    Samples sequences of (k_history + 1) consecutive phrases from the same raga.
    Returns:
        history_mels : (batch_size, k_history, 1, N_MELS, T)
        target_mel   : (batch_size, 1, N_MELS, T)
    """
    import random
    valid_ragas = [r for r, idxs in dataset.raga_to_idx.items()
                   if len(idxs) >= k_history + 1]

    if not valid_ragas:
        return None, None

    history_list = []
    target_list  = []

    for _ in range(batch_size):
        raga   = random.choice(valid_ragas)
        idxs   = dataset.raga_to_idx[raga]
        start  = np.random.randint(0, len(idxs) - k_history)
        seq    = [dataset.samples[idxs[start + j]][0] for j in range(k_history + 1)]
        history_list.append(np.stack(seq[:k_history]))   # (k_history, 1, N_MELS, T)
        target_list.append(seq[k_history])               # (1, N_MELS, T)

    history_mels = torch.from_numpy(np.stack(history_list)).to(device)
    target_mel   = torch.from_numpy(np.stack(target_list)).to(device)
    return history_mels, target_mel


# =========================
# TRAINING LOOP
# =========================
def train(resume=False, resume_path="checkpoints/best.pt"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR,      exist_ok=True)

    # --- Dataset ---
    dataset    = RagaDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=0)

    # --- Models ---
    encoder     = RagaEncoder().to(DEVICE)
    decoder     = RagaDecoder().to(DEVICE)
    contrastive = ContrastiveLoss().to(DEVICE)
    generator   = RagaGenerator().to(DEVICE)

    # --- Losses ---
    recon_loss      = ReconstructionLoss().to(DEVICE)
    transition_loss = TransitionLoss().to(DEVICE)

    # --- Optimizer (single optimizer over all parameters) ---
    all_params = (
        list(encoder.parameters())     +
        list(decoder.parameters())     +
        list(contrastive.parameters()) +
        list(generator.parameters())
    )
    optimizer = AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 100)

    total_params = sum(p.numel() for p in all_params)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Training on: {DEVICE}")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LR}\n")

    start_epoch = 1
    best_loss   = float('inf')

    if resume and os.path.exists(resume_path):
        start_epoch = load_checkpoint(
            resume_path, encoder, decoder, contrastive, generator,
            optimizer, scheduler
        ) + 1
        best_loss = torch.load(resume_path, map_location=DEVICE)["loss"]
        print(f"Resuming from epoch {start_epoch}, best loss so far: {best_loss:.4f}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        encoder.train()
        decoder.train()
        contrastive.train()
        generator.train()

        epoch_recon  = 0.0
        epoch_contra = 0.0
        epoch_trans  = 0.0
        epoch_total  = 0.0
        n_batches    = 0

        for mel_batch, labels in dataloader:
            mel_batch = mel_batch.to(DEVICE)   # (B, 1, N_MELS, T)
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=DEVICE)

            # ── 1. RECONSTRUCTION LOSS ──────────────────────────────────────
            z_seq, z_global = encoder(mel_batch)
            mel_hat         = decoder(z_seq)
            l_recon, _, _   = recon_loss(mel_batch, mel_hat)
            total_loss      = total_loss + LAMBDA_RECON * l_recon

            # ── 2. CONTRASTIVE LOSS ─────────────────────────────────────────
            view1, view2 = build_contrastive_batch(dataset, BATCH_SIZE, DEVICE)
            if view1 is not None:
                _, zg1 = encoder(view1)
                _, zg2 = encoder(view2)
                z_global_pairs = torch.cat([zg1, zg2], dim=0)   # (2B, D)
                l_contra        = contrastive(z_global_pairs)
                total_loss      = total_loss + LAMBDA_CONTRASTIVE * l_contra
                epoch_contra   += l_contra.item()

            # ── 3. TRANSITION LOSS ──────────────────────────────────────────
            history_mels, target_mel = build_transition_batch(
                dataset, BATCH_SIZE, K_HISTORY, DEVICE
            )
            if history_mels is not None:
                B, K, C, H, W = history_mels.shape

                # Encode each phrase in history
                history_flat = history_mels.reshape(B * K, C, H, W)
                z_seq_hist, zg_hist = encoder(history_flat)
                z_seq_hist   = z_seq_hist.reshape(B, K, T_PRIME, D)
                z_global_hist = zg_hist.reshape(B, K, D).mean(dim=1)   # mean over history

                # Encode target phrase
                z_seq_target, _ = encoder(target_mel)

                # Generate and compute loss
                z_seq_pred = generator(z_seq_hist, z_global_hist)
                l_trans    = transition_loss(z_seq_pred, z_seq_target.detach())
                total_loss = total_loss + LAMBDA_TRANSITION * l_trans
                epoch_trans += l_trans.item()

            total_loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)   # Prevent exploding gradients
            optimizer.step()

            epoch_recon += l_recon.item()
            epoch_total += total_loss.item()
            n_batches   += 1

        scheduler.step()

        # ── LOGGING ─────────────────────────────────────────────────────────
        avg_total  = epoch_total  / n_batches
        avg_recon  = epoch_recon  / n_batches
        avg_contra = epoch_contra / n_batches if epoch_contra > 0 else 0.0
        avg_trans  = epoch_trans  / n_batches if epoch_trans  > 0 else 0.0

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Total: {avg_total:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"Contrastive: {avg_contra:.4f} | "
            f"Transition: {avg_trans:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── CHECKPOINTING ───────────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0 or avg_total < best_loss:
            if avg_total < best_loss:
                best_loss = avg_total
                ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")
            else:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")

            torch.save({
                "epoch"      : epoch,
                "encoder"    : encoder.state_dict(),
                "decoder"    : decoder.state_dict(),
                "contrastive": contrastive.state_dict(),
                "generator"  : generator.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "loss"       : avg_total,
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

        # ── AUDIO RECONSTRUCTION CHECK ───────────────────────────────────────
        if epoch % AUDIO_EVERY == 0:
            try:
                from decode_to_audio import save_comparison
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    sample_mel = dataset[0][0].unsqueeze(0).to(DEVICE)
                    z_seq_s, _ = encoder(sample_mel)
                    mel_hat_s  = decoder(z_seq_s)
                save_comparison(
                    sample_mel[0].cpu(), mel_hat_s[0].cpu(),
                    output_dir=AUDIO_DIR, idx=epoch
                )
                print(f"  → Audio saved to '{AUDIO_DIR}/'")
                encoder.train()
                decoder.train()
            except Exception as e:
                print(f"  Audio save failed: {e}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to '{CHECKPOINT_DIR}/'")


# =========================
# RESUME FROM CHECKPOINT
# =========================
def load_checkpoint(path, encoder, decoder, contrastive, generator, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    contrastive.load_state_dict(ckpt["contrastive"])
    generator.load_state_dict(ckpt["generator"])
    if optimizer  and "optimizer"  in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  and "scheduler"  in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Loaded checkpoint '{path}' (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return ckpt["epoch"]


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # Before running, create labels.json:
    #
    #   python3 -c "
    #   import glob, json, os
    #   files = sorted(glob.glob('mel_specs/chunk_*.npy'))
    #   labels = {os.path.splitext(os.path.basename(f))[0]: 0 for f in files}
    #   json.dump(labels, open('labels.json', 'w'), indent=2)
    #   print('labels.json created for', len(labels), 'chunks')
    #   "
    #
    # All 34 Desh chunks get label 0.
    # When you add a second raga, assign its chunks label 1, and so on.

    # To resume from your existing checkpoint instead of starting fresh:
    #   set RESUME = True below
    RESUME          = True
    RESUME_CHECKPOINT = "checkpoints/best.pt"

    train(resume=RESUME, resume_path=RESUME_CHECKPOINT)
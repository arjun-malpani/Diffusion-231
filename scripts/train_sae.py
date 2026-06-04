"""
Train the SAE on cached activations.

Pre-req: run `python scripts/collect_activations.py` first so the activation
cache exists at DATA_DIR.

Run (from the repo root):
    python scripts/train_sae.py

Hyperparameters mirror SAeUron's setup for the style block (up.1.2):
    expansion_factor = 16  -> num_latents = 1280 * 16 = 20480
    k                = 32
    optimizer        = Adam, lr = 4e-4
    batch_size       = 4096
    epochs           = 10
    dead_threshold   = 10_000_000 samples (from config)

Checkpoint cadence is ~every 5k optimizer steps; with ~7500 steps/epoch at batch
4096 over the 4000-prompt cache, that's ~15 checkpoints across 10 epochs.
Mid-training checkpoints contain weights + Adam state + history (resumable).
The final artifact is a state_dict-only `sae_final.pt` for inference.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from diffusion_sae.sae   import SAE
from diffusion_sae.train import train, plot_history
from diffusion_sae.data  import ActivationShardDataset


# ============================ CONFIG ============================
DATA_DIR    = "data/activations/style_v1"     # set to "data/activations/smoke" if testing locally

D_IN        = 1280                            # up.1.2 channel dim
NUM_LATENTS = 20480                           # 16x expansion (SAeUron)
BATCH_SIZE  = 4096                            # SAeUron
EPOCHS      = 10                              # SAeUron style block
LR          = 4e-4                            # SAeUron
CKPT_EVERY  = 5000
CKPT_DIR    = "checkpoints/sae_v1"
PLOT_PATH   = "output_img/sae_v1_train_curves.png"
# ================================================================


# ---- dataset ----
ds = ActivationShardDataset(DATA_DIR)
print(f"dataset: {ds.info()}")

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,            # MPS prefers 0; on CUDA you can bump to 4 if I/O-bound
    pin_memory=False,         # MPS doesn't support pinned memory
    drop_last=True,
)
print(f"batches/epoch: {len(loader):,} | samples: {len(ds):,}")

# ---- model ----
sae = SAE(d_in=D_IN, num_latents=NUM_LATENTS)
print(f"SAE: d_in={D_IN}  num_latents={NUM_LATENTS}  k=32")

# ---- train ----
sae, history = train(
    sae, loader,
    epochs=EPOCHS,
    learning_rate=LR,
    log_every=100,
    ckpt_dir=CKPT_DIR,
    checkpoint_every=CKPT_EVERY,
)

# ---- save final plot ----
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plot_history(history, save=PLOT_PATH, show=False)
print(f"final weights: {CKPT_DIR}/sae_final.pt")
print(f"curves:        {PLOT_PATH}")

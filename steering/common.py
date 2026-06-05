"""Shared helpers + config for the style-injection steering experiment.

Pipeline (see README at bottom of this file):
  identify_directions.py -> features.pt   (per-style: feature idxs, mu levels, 16x16 styled maps)
  run_experiment.py      -> output/<subgroup>/<method>/...png   (3 subgroups x 2 methods x sweep)
  visualize.py           -> output/grids/*.png
"""

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import torch
from diffusion_sae.sae    import SAE
from diffusion_sae.config import DEVICE, DTYPE


#CONFIG STUFF
HOOKPOINT   = "unet.up_blocks.1.attentions.2"  
D_IN        = 1280
HW          = 16                                 # 16x16, 256 spatial patches at up_blocks.1.attentions2
SAE_CKPT    = os.path.join(REPO, "checkpoints/sae_v1/sae_final.pt")

# feature selection (neutral-vs-styled "what the style adds")
PERCENTILE  = 99.9                               # keep features above this pct of the (styled-neutral) score

# steering
STRENGTHS   = [0.5, 1.0, 2.0]                    # multipliers x each feature
COND_ONLY   = True                               # inject into the conditional CFG half only (the sweet spot)

# steering demo: 2 NEUTRAL prompts, shared across all styles so the grid compares cleanly
DEMO_PROMPTS = ["a cat", "a house"]

# three seed subgroups (see the user's spec)
TRAIN_SEED  = 42                                 # the noise used during activation collection
NEW_SEED    = 1234                               # a fixed seed != training, for subgroup C
B_META_SEED = 7                                  # meta-seed deriving B's (decoupled) random seeds, reproducibly
OUT_DIR     = os.path.join(HERE, "output")
FEATURES_PT = os.path.join(HERE, "features.pt")
# ================================================================


def load_sae(device=DEVICE):
    """Load the trained SAE in eval mode (deterministic per-row top-k encode)."""
    sae = SAE(d_in=D_IN)
    state = torch.load(SAE_CKPT, map_location="cpu", weights_only=True)
    sae.load_state_dict(state)
    sae.eval()
    return sae.to(device)

"""
Collect SAE training activations from SD-1.4 at the style hookpoint (up.1.2).

Run from the repo root:
    python scripts/collect_activations.py

Workflow:
  - Build 10 styles x 400 anchor prompts = 4000 styled prompts (stratified sample).
  - For each prompt: run SD's denoise loop with the VAE decode SKIPPED
    (decode_frames=False on the generate function) -- much faster, since we
    only need the UNet forwards to fire the activation hook.
  - Drop the unconditional CFG half (paper §5.1: only text-conditioned activations).
  - Reshape [1, C, H, W] -> [H*W, C] per timestep -> ~7,680 training rows per prompt.
  - Stream to disk every PROMPTS_PER_SHARD prompts (no need to hold all in RAM).
  - Sanity-check at the end by loading the shards via ActivationShardDataset.

The smoke test (SMOKE_TEST = True) verifies the whole pipeline end-to-end with
2 prompts in ~2 min on Mac MPS. Set SMOKE_TEST = False for the real AWS run.

Expected output on AWS (g6.xlarge, ~3-5s/prompt):
    4000 prompts x ~7680 samples per prompt = ~30.7M samples
    @ fp16, d_in=1280: ~78 GB on disk, 40 shards of ~2 GB each
    wall time: ~3-5 hours on a g6.xlarge
"""

import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from diffusion_sae.model  import import_model, get_activation
from diffusion_sae.config import MODEL_ID
from diffusion_sae.data   import ActivationShardDataset


# ============================ CONFIG ============================
STYLE_HOOKPOINT = "unet.up_blocks.1.attentions.2"        # up.1.2 -> 1280-ch / 16x16

# 10 styles deliberately spread for visual distinctness
STYLES = [
    "Van_Gogh",   "Picasso",         "Monet",     "Cubism",       "Watercolor",
    "Pop_Art",    "Cartoon",         "Ukiyoe",    "Pencil_Drawing","Impressionism",
]

ANCHORS_PER_STYLE  = 400              # 400 * 10 = 4000 prompts total
PROMPTS_PER_SHARD  = 100              # -> ~40 shards on the real run, ~2 GB each fp16

ANCHOR_FILE = "data/prompts/anchor_prompts_all.txt"
OUT_DIR     = "data/activations/style_v1"

SEED = 0

# ---- smoke test (~2 min, 2 generations on Mac MPS, verifies the whole pipeline)
SMOKE_TEST = False                    # FLIP TO True for a 2-prompt local smoke test
if SMOKE_TEST:
    STYLES            = STYLES[:1]    # Van_Gogh only
    ANCHORS_PER_STYLE = 2
    PROMPTS_PER_SHARD = 2
    OUT_DIR           = "data/activations/smoke"
# ================================================================


os.makedirs(OUT_DIR, exist_ok=True)

# ---- build prompt list ----
with open(ANCHOR_FILE) as f:
    anchors = [l.strip() for l in f if l.strip()]
print(f"loaded {len(anchors)} anchor prompts from {ANCHOR_FILE}")

rng = random.Random(SEED)
prompts = []
for style in STYLES:
    style_str = style.replace("_", " ")
    sampled = rng.sample(anchors, min(ANCHORS_PER_STYLE, len(anchors)))
    for a in sampled:
        prompts.append(f"{a} in {style_str} style.")
rng.shuffle(prompts)
print(f"built {len(prompts)} prompts ({len(STYLES)} styles x {ANCHORS_PER_STYLE} anchors)")
print(f"output dir: {OUT_DIR}")

# ---- load SD ----
print(f"loading SD: {MODEL_ID} ...")
pipe = import_model(MODEL_ID)
pipe.set_progress_bar_config(disable=True)


def encode_prompt_to_samples(pipe, prompt):
    """Run SD on `prompt`, capture activations at the style block.
    Returns [N_steps*256, d_in] fp16 -- ready to concatenate into a shard."""
    acts = get_activation(pipe, [STYLE_HOOKPOINT], prompt, decode_frames=False)
    chunks = []
    for t_act in acts[STYLE_HOOKPOINT]:                  # one per UNet forward (timestep)
        cond = t_act.chunk(2, dim=0)[1]                  # [1, 1280, 16, 16]  -- text-cond half only
        # [1, C, H, W] -> [H*W, C] (each spatial position becomes one training row)
        chunk = cond.permute(0, 2, 3, 1).reshape(-1, cond.shape[1])
        chunks.append(chunk.to(torch.float16))           # fp16 on disk
    return torch.cat(chunks, dim=0)                      # [steps * H*W, d_in]


# ---- collect ----
shard_buf = []
shard_idx = 0
t0 = time.time()

def flush_shard():
    global shard_idx, shard_buf
    if not shard_buf:
        return
    tensor = torch.cat(shard_buf, dim=0)
    path = os.path.join(OUT_DIR, f"shard_{shard_idx:05d}.pt")
    torch.save(tensor, path)
    mb = tensor.element_size() * tensor.numel() / 1e6
    elapsed = (time.time() - t0) / 60
    print(f"  saved {os.path.basename(path)}  shape={tuple(tensor.shape)} ({mb:.0f} MB)   "
          f"elapsed: {elapsed:.1f} min")
    shard_idx += 1
    shard_buf = []

print()
print(f"=== collecting (decode_frames=False) ===")
for i, prompt in enumerate(prompts):
    samples = encode_prompt_to_samples(pipe, prompt)
    shard_buf.append(samples)

    if (i + 1) % 10 == 0 and not SMOKE_TEST:
        rate = (i + 1) / max(time.time() - t0, 1e-9)
        eta_min = (len(prompts) - i - 1) / max(rate, 1e-9) / 60
        print(f"  [{i+1:4d}/{len(prompts)}]  {rate:.2f} prompts/s  eta ~{eta_min:.0f} min")

    if (i + 1) % PROMPTS_PER_SHARD == 0:
        flush_shard()

flush_shard()    # drain remainder

print()
print(f"=== collection complete in {(time.time()-t0)/60:.1f} min ===")

# ---- sanity check via the Dataset class ----
ds = ActivationShardDataset(OUT_DIR)
print(ds.info())
print(f"first row: shape={tuple(ds[0].shape)}  dtype={ds[0].dtype}")
print(f"last  row: shape={tuple(ds[-1].shape)}  dtype={ds[-1].dtype}")
print()
print(f"ready to train: `python scripts/train_sae.py` will load from {OUT_DIR}/")

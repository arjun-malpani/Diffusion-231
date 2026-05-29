#!/usr/bin/env python
"""
run_style_steering.py
=====================================================================
End-to-end SAeUron *style steering* reproducer.

What it does, start to finish:
  1. Loads vanilla Stable Diffusion 1.4 + the SAeUron *style* SAE (up.1.2).
  2. Identifies the "Van Gogh" SAE feature by scoring activations (Part 1).
  3. Injects that feature's decoder direction into the cross-attention
     feature map (additive steering, paper Eq. 7) and renders a neutral
     "a photo of a cat" prompt that NEVER mentions Van Gogh.
  4. Saves every figure into ./output_img/.

This is the script form of notebooks/style_steering_test.ipynb. It reproduces
the exact images explored interactively:
    output_img/vangogh_cat_sweep.png     (Fig 1: naive both-halves sweep)
    output_img/vangogh_cat_strong.png    (Fig 2: both-halves vs conditional-only)
    output_img/vangogh_cat_condonly.png  (Fig 3: conditional-only sweet spot)
    output_img/vangogh_cat_topk.png      (Fig 4: Top-tau, the clearest result)
    output_img/cat_baseline.png, output_img/cat_steered_strong.png

Run it (conda activate is broken on this machine, so call the env python directly):
    /opt/anaconda3/envs/diffusion231/bin/python run_style_steering.py

Tips:
  * Scoring (Part 1) is the slow part on MPS. Set RUN_SCORING = False to skip it
    and reuse FALLBACK_TOP5 while you iterate on the figures.
  * No CUDA here -> runs on Apple MPS in float32. On a CUDA GPU it uses float16.
=====================================================================
"""

import os          # filesystem paths
import sys         # to put the SAeUron clone on the import path
import time        # simple elapsed-time logging
import torch       # tensors / the diffusion model / the SAE

import matplotlib
matplotlib.use("Agg")              # headless backend: write PNGs, never open a window
import matplotlib.pyplot as plt    # used only to lay out the comparison grids

# ------------------------------------------------------------------ #
# A tiny timestamped logger so you can see progress in a long run.
# ------------------------------------------------------------------ #
_T0 = time.time()                                  # record the start time once
def log(*a):                                       # call log(...) like print(...)
    print(f"[{time.time() - _T0:6.1f}s]", *a, flush=True)   # prepend elapsed seconds; flush so it streams

# ================================================================== #
# 0. PATHS  (everything is anchored to this file's location)
# ================================================================== #
REPO = os.path.dirname(os.path.abspath(__file__))          # repo root = folder containing this script
SAEURON_PATH = os.path.join(REPO, "external", "SAeUron")   # the cloned SAeUron repo (a git submodule)
OUTPUT_DIR = os.path.join(REPO, "output_img")              # all images are written here
os.makedirs(OUTPUT_DIR, exist_ok=True)                     # create output_img/ if it does not exist
def out(name):                                             # helper: build an "output_img/<name>" path
    return os.path.join(OUTPUT_DIR, name)

# ================================================================== #
# 1. MAKE SAeUron IMPORTABLE
# SAeUron uses bare top-level imports (`from SAE...`, `import utils...`).
# This repo has NO top-level utils/ package, so prepending the clone to
# sys.path is safe and nothing gets shadowed (this is "Branch A").
# ================================================================== #
if SAEURON_PATH not in sys.path:                           # avoid adding it twice
    sys.path.insert(0, SAEURON_PATH)                       # prepend so `import SAE...` resolves
from SAE.sae import Sae                                    # the sparse autoencoder (encoder + W_dec decoder)
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline  # SD wrapper that supports forward hooks
from SAE.unlearning_utils import compute_feature_importance              # scores how concept-specific each feature is

# ================================================================== #
# 2. DEVICE / DTYPE
# CUDA -> float16 (fast). Apple MPS or CPU -> float32 (fp16 is flaky for SD on MPS).
# ================================================================== #
if torch.cuda.is_available():                              # a real NVIDIA GPU?
    DEVICE, DTYPE = "cuda", torch.float16
elif torch.backends.mps.is_available():                    # Apple Silicon GPU?
    DEVICE, DTYPE = "mps", torch.float32
else:                                                      # fall back to CPU
    DEVICE, DTYPE = "cpu", torch.float32
log("device:", DEVICE, "| dtype:", DTYPE)                  # report what we picked

# ================================================================== #
# 3. WHICH CROSS-ATTENTION BLOCK
# Paper §5.1: up.1.2 specializes in STYLE, up.1.1 in objects. We want style.
# ================================================================== #
STYLE_HOOKPOINT = "unet.up_blocks.1.attentions.2"          # the block whose output we read/steer

# ================================================================== #
# 4. LOAD SD-1.4 + THE STYLE SAE
# ================================================================== #
log("loading SD-1.4 (from the local HF cache) ...")
pipe = HookedStableDiffusionPipeline.from_pretrained(      # wraps a normal SD pipeline
    "CompVis/stable-diffusion-v1-4",                       # vanilla SD-1.4 weights
    torch_dtype=DTYPE,                                     # match our device dtype
    safety_checker=None,                                   # disable the NSFW filter (research use)
).to(DEVICE)                                               # move UNet/VAE/text-encoder onto the device
pipe.set_progress_bar_config(disable=True)                 # hide the per-step tqdm bars

log("loading the style SAE from the Hub ...")
sae = Sae.load_from_hub(                                   # downloads (or reuses cached) safetensors
    "bcywinski/SAeUron", hookpoint=STYLE_HOOKPOINT, device=DEVICE
).to(DTYPE)                                                # cast SAE weights to our dtype
# W_dec has shape [num_latents, d_in]; ROW i is feature i's decoder direction d_i (unit-norm here).
log("SAE  W_dec:", tuple(sae.W_dec.shape), "| d_in:", sae.d_in,
    "| num_latents:", sae.num_latents, "| k:", sae.cfg.k,
    "| input_unit_norm:", sae.cfg.input_unit_norm)

# ================================================================== #
# 5. ADDITIVE STEERING HOOK  (paper Appendix H, Eq. 7)
# ------------------------------------------------------------------ #
# A PyTorch forward hook fires on the OUTPUT of the cross-attn block.
# Under classifier-free guidance (CFG) that output is shaped [2B, C, H, W]:
#     rows 0..B-1  = UNCONDITIONAL pass (empty prompt)
#     rows B..2B-1 = CONDITIONAL  pass (our text prompt)
# Here C == d_in (1280) and H*W = 256 spatial positions (a 16x16 grid).
#
# Eq. 7:   F_t  <-  F_t  +  sum_i  gamma+ * mu(i) * d_i
# We fold gamma+ * mu into a single `strength` scalar and add the (summed,
# optionally renormalized) decoder direction(s) into the feature map.
# ================================================================== #
class SAEStyleSteeringHook:
    def __init__(self, sae, feature_idxs, strength, cond_only=True, normalize=True):
        idx = torch.as_tensor(feature_idxs, device=sae.W_dec.device).long()  # feature indices -> tensor
        direction = sae.W_dec[idx].float().sum(0)          # sum the chosen decoder rows -> [C] (Top-tau)
        if normalize:                                      # make `strength` comparable across any tau
            direction = direction / (direction.norm() + 1e-8)   # renormalize the summed vector to unit norm
        self.direction = direction                         # cache the [C] steering vector
        self.strength = float(strength)                    # how hard to push
        self.cond_only = cond_only                         # inject into conditional half only? (usually yes)

    @torch.no_grad()                                       # never build autograd graph inside a hook
    def __call__(self, module, inp, output):               # signature required by register_forward_hook
        hidden = output[0]                                 # the block output tensor [2B, C, H, W]
        add = self.strength * self.direction.view(1, -1, 1, 1).to(hidden.dtype)  # broadcast [C] onto channel axis
        if self.cond_only:                                 # CFG amplifies a conditional-only push by ~guidance_scale
            hidden = hidden.clone()                        # copy so we don't edit the original in place
            half = hidden.shape[0] // 2                    # = B
            hidden[half:] = hidden[half:] + add            # add only to the conditional rows (second half)
        else:
            hidden = hidden + add                          # add to BOTH halves (literal Eq. 7; weaker under CFG)
        return (hidden,)                                   # a forward hook must return the (modified) output tuple

# ================================================================== #
# 6. SMALL HELPERS:  generate one image  +  save a labelled grid
# ================================================================== #
def generate(prompt, hook=None, seed=40, steps=30, guidance=7.5):
    gen = torch.Generator(device="cpu").manual_seed(seed)  # cpu generator => reproducible across devices
    hooks = {STYLE_HOOKPOINT: hook} if hook is not None else {}   # empty dict => clean baseline
    img = pipe.run_with_hooks(                             # runs the full DDIM denoising loop with our hook attached
        prompt=prompt, generator=gen,
        num_inference_steps=steps, guidance_scale=guidance,
        device=torch.device(DEVICE),                       # must pass: this method defaults to "cuda"
        position_hook_dict=hooks,                          # where + which hook to register
    )[0]                                                   # run_with_hooks returns a list of PIL images
    if DEVICE == "mps":                                    # free MPS memory between generations
        torch.mps.empty_cache()
    return img

def save_grid(images, titles, suptitle, filename, ncols=None):
    n = len(images)                                        # number of panels
    ncols = ncols or n                                     # default: one row
    nrows = (n + ncols - 1) // ncols                       # enough rows to fit everything
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.5 * nrows))
    axes = [axes] if n == 1 else axes.flatten()            # normalise to a flat list of axes
    for ax in axes:                                        # hide all axes first (blank cells stay clean)
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):            # draw each image with its title
        ax.imshow(im); ax.set_title(t, fontsize=9)
    fig.suptitle(suptitle, fontsize=12)                    # overall caption
    plt.tight_layout()
    path = out(filename)                                   # -> output_img/<filename>
    plt.savefig(path, dpi=110, bbox_inches="tight")        # write the PNG
    plt.close(fig)                                         # release the figure
    log("saved", path)

# ================================================================== #
# PART 1 - IDENTIFY the Van Gogh style feature
# ------------------------------------------------------------------ #
# No precomputed index ships with the repo, so we derive one cheaply,
# replicating scripts/gather_sae_acts_ca_prompts.py:
#   * generate a few content prompts per style,
#   * cache the up.1.2 output activations [num_prompts, steps, H*W, d_in],
#   * SAE-encode each spatial position, scatter the top-k into a dense
#     latent vector, average over positions -> [steps, num_latents] per prompt,
#   * stack -> {style: [num_prompts, steps, num_latents]},
#   * compute_feature_importance scores Van-Gogh-specificity per feature,
#   * argmax = the style feature; keep the top-5 for the Top-tau figure.
# ================================================================== #
RUN_SCORING = True                                         # False => skip scoring, use FALLBACK_TOP5
FALLBACK_TOP5 = [10765, 17145, 14323, 10224, 6618]         # result of a previous scoring run

CONTENT = ["a house", "a cat", "a mountain", "a bicycle", "a tree", "a boat"]   # neutral content words
STYLES = {                                                 # target style first, then "other" styles to contrast
    "Van_Gogh": "in Van Gogh style",
    "Cubism": "in Cubism style",
    "Watercolor": "in Watercolor style",
    "Pop_Art": "in Pop Art style",
}
SCORING_STEPS = 15                                         # fewer than the 50 used in the paper => cheap smoke test

def encode_to_latents(acts, sae, steps):
    """acts [num_prompts, steps, H*W, d_in] (one style) -> [num_prompts, steps, num_latents]."""
    out_list = []                                          # collect one [steps, num_latents] per prompt
    with torch.no_grad():
        for i in range(acts.shape[0]):                     # loop over prompts
            x = acts[i].reshape(steps, -1, sae.d_in).to(sae.device).to(sae.dtype)  # [steps, H*W, d_in]
            top_acts, top_idx = sae.encode(x)              # TopK encode -> values/indices [steps*H*W, k]
            buf = torch.zeros((top_acts.shape[0], sae.num_latents),                # dense latent buffer
                              device=sae.device, dtype=top_acts.dtype)
            dense = buf.scatter(-1, top_idx, top_acts)     # place the k activations -> [steps*H*W, num_latents]
            dense = dense.reshape(steps, -1, sae.num_latents)  # [steps, H*W, num_latents]
            out_list.append(dense.mean(1).float().cpu())   # average over H*W positions -> [steps, num_latents]
    return torch.stack(out_list)                           # [num_prompts, steps, num_latents]

if RUN_SCORING:
    style_latents = {}                                     # {style_name: [num_prompts, steps, num_latents]}
    for name, suffix in STYLES.items():
        per_prompt = []                                    # accumulate per-prompt latents for this style
        for c in CONTENT:                                  # ONE prompt per call: batch>1 OOMs MPS attention
            _, cache = pipe.run_with_cache(                # run the denoise loop and cache the block output
                prompt=[f"{c} {suffix}"],                  # e.g. "a cat in Van Gogh style"
                generator=torch.Generator(device="cpu").manual_seed(188),  # fixed seed for repeatability
                num_inference_steps=SCORING_STEPS, guidance_scale=9.0,      # 9.0 matches the gather script
                positions_to_cache=[STYLE_HOOKPOINT], save_output=True,     # cache up.1.2's output
                output_type="latent", device=torch.device(DEVICE))         # "latent" => skip VAE decode (faster)
            acts = cache["output"][STYLE_HOOKPOINT]        # [1, steps, H*W, d_in]  (conditional half only)
            per_prompt.append(encode_to_latents(acts, sae, SCORING_STEPS))  # -> [1, steps, num_latents]
            del cache, acts                                # drop big tensors promptly
            if DEVICE == "mps":
                torch.mps.empty_cache()                    # and free MPS memory
        style_latents[name] = torch.cat(per_prompt, 0)     # -> [num_prompts, steps, num_latents]
        log(name, "->", tuple(style_latents[name].shape))
    # score each feature per timestep, then average the scores over timesteps
    scores = torch.stack([compute_feature_importance(style_latents, "Van_Gogh", t)
                          for t in range(SCORING_STEPS)]).mean(0)   # -> [num_latents]
    top_vals, top_idx = scores.topk(5)                     # the 5 most Van-Gogh-specific features
    TOP5 = top_idx.tolist()                                # python list of indices
    mu = style_latents["Van_Gogh"][:, :, TOP5[0]].mean().item()    # avg activation of the top feature (Eq.7 mu)
    log("Van Gogh top-5 (idx, score):",
        list(zip(TOP5, [round(v, 4) for v in top_vals.tolist()])), "| mu:", round(mu, 4))
else:
    TOP5 = FALLBACK_TOP5                                    # reuse a known result
    log("RUN_SCORING=False -> using fallback top-5:", TOP5)

VANGOGH_IDX = TOP5[0]                                       # the single best feature (tau=1)

# ================================================================== #
# Common settings for every generation figure below
# ================================================================== #
PROMPT, SEED = "a photo of a cat", 40                      # neutral prompt (no style word) + fixed seed

# ================================================================== #
# PART 2 - FIGURE 1: naive strength sweep, adding to BOTH CFG halves.
# This is the literal Eq. 7 wiring; it turns out too weak to see clearly.
# ================================================================== #
imgs, titles = [], []
for s in [0, 5, 10, 20]:                                   # strength 0 == baseline (no hook)
    hook = None if s == 0 else SAEStyleSteeringHook(sae, [VANGOGH_IDX], float(s),
                                                    cond_only=False, normalize=True)
    imgs.append(generate(PROMPT, hook=hook, seed=SEED)); titles.append(f"strength={s}")
    log("fig1", titles[-1])
imgs[0].save(out("cat_baseline.png"))                      # also save the clean baseline alone
imgs[-1].save(out("cat_steered_strong.png"))               # and the strongest steered single image
save_grid(imgs, titles, f"Fig 1: both-halves sweep on '{PROMPT}' (subtle)", "vangogh_cat_sweep.png")

# ================================================================== #
# PART 3 - FIGURE 2: stronger sweep, BOTH-halves vs CONDITIONAL-only.
# both-halves stays weak even at 200; cond-only at 120+ OVERSHOOTS and the
# image collapses, because CFG amplifies the conditional-only push ~guidance_scale.
# ================================================================== #
panels = [(0, False, "baseline"),
          (60, False, "both s=60"), (120, False, "both s=120"), (200, False, "both s=200"),
          (120, True, "cond s=120"), (200, True, "cond s=200")]
imgs, titles = [], []
for s, co, label in panels:
    hook = None if s == 0 else SAEStyleSteeringHook(sae, [VANGOGH_IDX], float(s),
                                                    cond_only=co, normalize=True)
    imgs.append(generate(PROMPT, hook=hook, seed=SEED)); titles.append(label)
    log("fig2", label)
save_grid(imgs, titles, f"Fig 2: both vs conditional-only on '{PROMPT}'", "vangogh_cat_strong.png", ncols=6)

# ================================================================== #
# PART 4 - FIGURE 3: conditional-only LOW sweep -> the sweet spot (~40-60),
# where a visible, structure-preserving painterly shift appears.
# ================================================================== #
imgs, titles = [], []
for s in [0, 10, 20, 30, 40, 60]:
    hook = None if s == 0 else SAEStyleSteeringHook(sae, [VANGOGH_IDX], float(s),
                                                    cond_only=True, normalize=True)
    imgs.append(generate(PROMPT, hook=hook, seed=SEED)); titles.append(f"cond s={s}")
    log("fig3", titles[-1])
save_grid(imgs, titles, f"Fig 3: conditional-only sweet spot on '{PROMPT}'", "vangogh_cat_condonly.png", ncols=6)

# ================================================================== #
# PART 5 - FIGURE 4: Top-tau. Sum the top-1/3/5 Van Gogh directions and inject
# them conditional-only. tau>=3 at strength ~70 gives the clearest, structure-
# preserving Van-Gogh look -- the best result on vanilla SD-1.4.
# ================================================================== #
specs = [(1, 0, "baseline")]                               # one baseline panel
for tau in (1, 3, 5):                                      # how many top features to sum
    for s in (40, 70):                                     # two strengths in the sweet-spot range
        specs.append((tau, s, f"tau={tau} s={s}"))
imgs, titles = [], []
for tau, s, label in specs:
    hook = None if s == 0 else SAEStyleSteeringHook(sae, TOP5[:tau], float(s),
                                                    cond_only=True, normalize=True)
    imgs.append(generate(PROMPT, hook=hook, seed=SEED)); titles.append(label)
    log("fig4", label)
save_grid(imgs, titles, f"Fig 4: Top-tau conditional-only on '{PROMPT}'", "vangogh_cat_topk.png", ncols=4)

log("ALL DONE -> images are in output_img/")

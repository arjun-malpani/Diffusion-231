"""
Identify the SAE directions each style ADDS to a neutral image (injection, not erasure).

Method (per style):
  * Take the 20 per-class anchors (row 0 of each by_class file).
  * For each anchor, generate it WITHOUT the style word (neutral) and WITH it (styled),
    using the SAME seed for the pair so the only difference is the style.
  * Cache up.1.2 (conditional half), SAE-encode -> [steps, HW, num_latents].
  * Accumulate, per feature f:
        mean_styled[f], mean_neutral[f]   (over anchors, steps, patches)
        styled spatial map M[f, 16, 16]   (over anchors, steps; keeps spatial structure)
  * score[f] = mean_styled[f] - mean_neutral[f]   ("what the style turns on that was off")
  * keep features above the PERCENTILE-th percentile of score (and score>0) -> the style's
    "new" directions. Save their decoder-injection level mu (=mean_styled) and spatial map.

Output: steering/features.pt  { style: {idx[F], mu[F], maps[F,16,16]} }  (+ features.json summary)

Run from repo root:  python steering/identify_directions.py
"""

import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import torch

from diffusion_sae.model  import import_model, get_activation
from diffusion_sae.config import MODEL_ID, STYLES, BY_CLASS_DIR
from steering.common      import (load_sae, HOOKPOINT, HW, D_IN, PERCENTILE,
                                  FEATURES_PT, HERE)

ID_SEED_BASE = 0   # anchor i uses seed (ID_SEED_BASE + i) for BOTH its neutral and styled gen


def clean(a):
    a = a.strip()
    return a[:-1] if a.endswith(".") else a


def load_anchors(n_per_class_row=0):
    """20 anchors -- row `n_per_class_row` of each sd_prompt_*.txt, sorted by class."""
    paths = sorted(glob.glob(os.path.join(BY_CLASS_DIR, "sd_prompt_*.txt")))
    anchors = []
    for p in paths:
        with open(p) as f:
            lines = [l.strip() for l in f if l.strip()]
        anchors.append(clean(lines[n_per_class_row]))
    return anchors


def cond_acts(pipe, prompt, seed):
    """Run SD on `prompt`; return up.1.2 conditional-half activations [steps, HW, d_in] (fp32, on GPU)."""
    acts = get_activation(pipe, [HOOKPOINT], prompt, decode_frames=False, seed=seed)
    chunks = []
    for t_act in acts[HOOKPOINT]:                 # one [2, C, 16, 16] per denoise step
        cond = t_act.chunk(2, dim=0)[1]            # [1, C, 16, 16] -- text-conditioned half
        chunks.append(cond.permute(0, 2, 3, 1).reshape(-1, cond.shape[1]))   # [HW, C]
    return torch.stack(chunks, dim=0).float()      # [steps, HW, d_in]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = load_sae(device)
    pipe = import_model(MODEL_ID)
    pipe.set_progress_bar_config(disable=True)

    anchors = load_anchors()
    print(f"{len(anchors)} anchors | {len(STYLES)} styles | percentile={PERCENTILE}")

    features = {}
    summary  = {}
    t0 = time.time()

    for s_i, style in enumerate(STYLES):
        style_str = style.replace("_", " ")
        L, P = sae.num_latents, HW * HW                           # P = 256 patches
        styled_spatial = torch.zeros(L, P, device=device)         # sum over (anchors, steps), keep patches
        neutral_sum    = torch.zeros(L, device=device)            # sum over (anchors, steps, patches)
        styled_steps = neutral_steps = 0                          # # step-vectors accumulated for each

        for a_i, anchor in enumerate(anchors):
            seed = ID_SEED_BASE + a_i                             # SAME seed for the neutral/styled pair
            z_n = sae.encode(cond_acts(pipe, anchor, seed))                            # [steps, P, L]
            neutral_sum    += z_n.sum(dim=(0, 1)); neutral_steps += z_n.shape[0]
            z_s = sae.encode(cond_acts(pipe, f"{anchor} in {style_str} style.", seed)) # [steps, P, L]
            styled_spatial += z_s.sum(dim=0).transpose(0, 1);      styled_steps += z_s.shape[0]

        # per-feature mean activation = total sum / (#step-vectors * #patches)
        mean_styled  = styled_spatial.sum(dim=1) / (styled_steps * P)    # [L]
        mean_neutral = neutral_sum / (neutral_steps * P)                 # [L]
        score = mean_styled - mean_neutral                              # what the style ADDS

        thr = torch.quantile(score, PERCENTILE / 100.0)
        sel = torch.nonzero((score > thr) & (score > 0), as_tuple=False).squeeze(-1)
        sel = sel[torch.argsort(score[sel], descending=True)]      # strongest first

        maps = (styled_spatial[sel] / styled_steps).reshape(-1, HW, HW)  # [F,16,16] styled per-patch level
        features[style] = {"idx": sel.cpu(),
                           "mu":  mean_styled[sel].cpu(),
                           "maps": maps.cpu()}
        summary[style] = {"n_selected": int(sel.numel()),
                          "top_idx":   sel[:5].tolist(),
                          "top_score": [round(score[i].item(), 4) for i in sel[:5]],
                          "top_mu":    [round(mean_styled[i].item(), 4) for i in sel[:5]]}
        print(f"[{s_i+1:2d}/{len(STYLES)}] {style:15s} selected {sel.numel():4d} feats | "
              f"top idx {sel[:3].tolist()} | top mu {summary[style]['top_mu'][:3]} | "
              f"elapsed {(time.time()-t0)/60:.1f} min")

    torch.save(features, FEATURES_PT)
    with open(os.path.join(HERE, "features.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {FEATURES_PT}")
    print(f"saved {os.path.join(HERE, 'features.json')}")


if __name__ == "__main__":
    main()

"""
Run the full style-injection experiment:  3 seed subgroups x 2 methods x 10 styles x 2 prompts x 3 strengths.

Seed subgroups (per the spec):
  A : training noise (seed 42) -- SAME seed for the neutral baseline and the steered image.
  B : fully random noise, with a DIFFERENT random seed for the baseline vs. each steered image
      (decoupled). Draws are reproducible via B_META_SEED so re-runs match.
  C : a fixed NEW seed (1234, != training) -- SAME seed for baseline and steered.

Baselines are the neutral prompt with no injection (one per subgroup/prompt; in A & C they are shared
across style rows since they don't depend on style). Outputs are written so visualize.py can grid them.
Existing files are skipped, so the run is resumable.

Run from repo root (ideally in tmux):  python steering/run_experiment.py
"""

import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import torch

from diffusion_sae.model  import import_model
from diffusion_sae.config import MODEL_ID, STYLES
from steering.common import (load_sae, OUT_DIR, FEATURES_PT, DEMO_PROMPTS, STRENGTHS,
                             TRAIN_SEED, NEW_SEED, B_META_SEED)
from steering.steer  import set_pipe, generate_baseline, generate_steered

METHODS = ["uniform", "patch"]


def slug(p):
    return p.replace(" ", "_")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = load_sae(device)
    features = torch.load(FEATURES_PT, map_location=device, weights_only=False)
    pipe = import_model(MODEL_ID); pipe.set_progress_bar_config(disable=True)
    set_pipe(pipe)                                   # registers pipe + upcasts VAE to fp32

    brng = random.Random(B_META_SEED)                # reproducible "random" seeds for subgroup B

    # seed plan: for each subgroup, a baseline seed per prompt + a steered-seed generator
    def seeds_for(subgroup, prompt_i):
        if subgroup == "A":
            return TRAIN_SEED, (lambda: TRAIN_SEED)          # baseline & steered share 42
        if subgroup == "C":
            return NEW_SEED, (lambda: NEW_SEED)              # baseline & steered share 1234
        # B: baseline random; each steered a fresh, different random draw
        return brng.randint(0, 2**31 - 1), (lambda: brng.randint(0, 2**31 - 1))

    total = len(["A", "B", "C"]) * len(METHODS) * len(STYLES) * len(DEMO_PROMPTS) * len(STRENGTHS)
    done = 0; t0 = time.time()
    manifest = {}

    for subgroup in ["A", "B", "C"]:
        for p_i, prompt in enumerate(DEMO_PROMPTS):
            base_seed, steered_seed = seeds_for(subgroup, p_i)

            # ---- baseline (no injection), one per subgroup/prompt ----
            bdir = os.path.join(OUT_DIR, subgroup, "baseline"); os.makedirs(bdir, exist_ok=True)
            bpath = os.path.join(bdir, f"{slug(prompt)}.png")
            if not os.path.exists(bpath):
                generate_baseline(pipe, prompt, base_seed).save(bpath)
            manifest[f"{subgroup}/baseline/{slug(prompt)}"] = base_seed

            # ---- steered ----
            for method in METHODS:
                mdir = os.path.join(OUT_DIR, subgroup, method)
                for style in STYLES:
                    sdir = os.path.join(mdir, style); os.makedirs(sdir, exist_ok=True)
                    for strength in STRENGTHS:
                        spath = os.path.join(sdir, f"{slug(prompt)}__s{strength}.png")
                        seed = steered_seed()
                        if not os.path.exists(spath):
                            img = generate_steered(pipe, sae, prompt, features[style],
                                                   method, strength, seed)
                            img.save(spath)
                        manifest[f"{subgroup}/{method}/{style}/{slug(prompt)}__s{strength}"] = seed
                        done += 1
                        if done % 20 == 0:
                            rate = done / max(time.time() - t0, 1e-9)
                            print(f"  [{done:4d}/{total}] {subgroup}/{method}/{style}/{slug(prompt)} "
                                  f"s={strength} | {rate:.2f} img/s | eta ~{(total-done)/max(rate,1e-9)/60:.0f} min")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\ndone: {done} steered + baselines in {(time.time()-t0)/60:.1f} min")
    print(f"images under {OUT_DIR}/  -> now run: python steering/visualize.py")


if __name__ == "__main__":
    main()

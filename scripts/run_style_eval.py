#!/usr/bin/env python
"""
run_style_eval.py
=============================================================
Full SAE style-steering evaluation sweep.

For every style in data/prompts/inference/<style>.txt, for every prompt in that
file, generate at a fixed seed (default 42):

  unstyled  : the prompt as-is
  prompted  : "<prompt> in <style> style."        (text-conditioned reference)
  injected  : SAE 'uniform' injection  @ each weight   (one vector per patch)
  entangled : SAE 'patch'   injection  @ each weight   (one embedding per patch)

Score each image with 4 metrics (prompt_clip, style_clip, uc_style, content_clip),
write results/<...>/scores.csv, and render the requested figures (F1-F5).

This script does NOT run unless you invoke it. A full run is
  n_styles × n_prompts × (2 + n_weights × n_methods)  diffusion generations
(e.g. 10 × 20 × 6 = 1200) -- heavy on MPS/CPU. Use --scope to subsample.

Examples
--------
  # full sweep, all figures (needs the SAE ckpt + UC style50.pth)
  python scripts/run_style_eval.py

  # quick proof-of-concept: 5 prompts/style, just the headline figures
  python scripts/run_style_eval.py --scope limited-prompts --figures F1 F2

  # one style, smoke test
  python scripts/run_style_eval.py --styles Van_Gogh --scope smoke

  # re-plot from an existing scores.csv without loading any models
  python scripts/run_style_eval.py --figures-only --figures F2 F4
=============================================================
"""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion_sae.config import STYLES, INFERENCE_DIR
from style_eval_lib import (
    Generator, Scorer, build_conditions, read_prompts, run_pipeline, load_scores,
    detect_device, log, DEFAULT_WEIGHTS, DEFAULT_METHODS, DEFAULT_SEED,
    DEFAULT_UC_CKPT, DEFAULT_UC_REPO, DEFAULT_RESULTS,
)
from style_eval_figures import make_figures, ALL_FIGURES

SCOPE_LIMIT = {"all-prompts": None, "limited-prompts": 5, "smoke": 1}


def parse_args():
    p = argparse.ArgumentParser(description="SAE style-steering evaluation sweep",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--styles", nargs="+", default=STYLES,
                   help="styles to evaluate (must have inference prompts + SAE features)")
    p.add_argument("--scope", choices=list(SCOPE_LIMIT), default="all-prompts",
                   help="how many prompts/style: all-prompts | limited-prompts (5) | smoke (1)")
    p.add_argument("--limit", type=int, default=None, help="override prompts/style count")
    p.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS,
                   help="SAE steering weightages (the hyperparam perturbing SAE activations)")
    p.add_argument("--methods", nargs="+", choices=["uniform", "patch"], default=DEFAULT_METHODS,
                   help="uniform=one vector/patch (injected), patch=one embedding/patch (entangled)")
    p.add_argument("--skip-prompted", action="store_true",
                   help="do not generate the '<prompt> in <style> style.' reference image")
    p.add_argument("--figures", nargs="+", default=["all"], choices=ALL_FIGURES + ["all"],
                   help="figures to render (subset of: grid, lines, tradeoff, per_style, bars)")
    p.add_argument("--grid-prompts", type=int, default=6,
                   help="max prompts per style shown in the F1 grid")
    p.add_argument("--save-images", action="store_true",
                   help="also persist each condition image to results/images/ (default: grid only)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="generation seed (42 = training seed)")
    p.add_argument("--inference-dir", default=os.path.join(REPO, INFERENCE_DIR))
    p.add_argument("--results-dir", default=DEFAULT_RESULTS)
    p.add_argument("--uc-checkpoint", default=DEFAULT_UC_CKPT)
    p.add_argument("--uc-repo", default=DEFAULT_UC_REPO)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    p.add_argument("--batch-size", type=int, default=16, help="scoring batch size")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="regenerate/re-score even if image+row already exist")
    p.add_argument("--no-generate", action="store_true",
                   help="score pre-existing images only (no diffusion, no SD/SAE load)")
    p.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    p.add_argument("--figures-only", action="store_true",
                   help="render figures from existing scores.csv and exit (no models loaded)")
    return p.parse_args()


def main():
    args = parse_args()
    figures_dir = os.path.join(args.results_dir, "figures")

    # ── fast path: just (re)render figures from an existing CSV ───────── #
    if args.figures_only:
        csv_path = os.path.join(args.results_dir, "scores.csv")
        if not os.path.isfile(csv_path):
            sys.exit(f"--figures-only: no scores.csv at {csv_path}")
        rows = load_scores(csv_path)
        log(f"loaded {len(rows)} rows from {csv_path}")
        make_figures(rows, args.figures, figures_dir, grid_max_prompts=args.grid_prompts)
        return

    device = detect_device(args.device)
    limit = args.limit if args.limit is not None else SCOPE_LIMIT[args.scope]

    # ── collect prompts per style ────────────────────────────────────── #
    prompts_by_style = {}
    for style in args.styles:
        fp = os.path.join(args.inference_dir, f"{style}.txt")
        if not os.path.isfile(fp):
            log(f"WARNING: no prompt file for '{style}' at {fp} -- skipping")
            continue
        prompts_by_style[style] = read_prompts(fp, limit)
    if not prompts_by_style:
        sys.exit(f"no usable styles under {args.inference_dir} for {args.styles}")

    conditions = build_conditions(args.weights, args.methods, include_prompted=not args.skip_prompted)
    n_prompts = sum(len(v) for v in prompts_by_style.values())
    log(f"device={device} | styles={list(prompts_by_style)} | prompts/style limit={limit}")
    log(f"conditions ({len(conditions)}): {[c.name for c in conditions]}")
    log(f"planned generations: {n_prompts} prompts × {len(conditions)} = {n_prompts * len(conditions)}")

    # ── models ───────────────────────────────────────────────────────── #
    scorer = Scorer(device=device, uc_checkpoint=args.uc_checkpoint,
                    uc_repo=args.uc_repo, batch_size=args.batch_size)
    if not scorer.has_uc():
        log("WARNING: UnlearnCanvas classifier unavailable -> uc_style will be BLANK.")
        log(f"  fix: pip install timm  &&  place style50.pth at {args.uc_checkpoint}")
        log("  (Google Drive folder 18dhkXyZQWjdMvlAlxZx3fZhdCZvlj2Hw -> subdir 'classifiers')")
    generator = None
    if not args.no_generate:
        generator = Generator(device=device, seed=args.seed)
        avail = set(generator.available_styles())
        for s in [s for s in prompts_by_style if s not in avail]:
            log(f"WARNING: no SAE features for '{s}' -- skipping")
            del prompts_by_style[s]
        if not prompts_by_style:
            sys.exit("no styles left after intersecting with available SAE features")

    # ── generate + score ─────────────────────────────────────────────── #
    rows, csv_path, images = run_pipeline(prompts_by_style, conditions, scorer, args.results_dir,
                                          generator=generator, seed=args.seed,
                                          skip_existing=args.skip_existing,
                                          save_images=args.save_images)

    # ── figures ──────────────────────────────────────────────────────── #
    if not args.no_figures:
        make_figures(rows, args.figures, figures_dir, images=images,
                     grid_max_prompts=args.grid_prompts)
    log(f"done. scores -> {csv_path} | images -> {os.path.join(args.results_dir, 'images')} "
        f"| figures -> {figures_dir}")


if __name__ == "__main__":
    main()

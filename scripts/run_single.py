#!/usr/bin/env python
"""
run_single.py
=============================================================
Run the SAE style-steering evaluation on ONE user-specified prompt + style,
and render user-specified figures. A thin, fully-configurable front-end over
style_eval_lib / style_eval_figures (same conditions, scores, and seed as the
full sweep in run_style_eval.py).

Generates: unstyled, prompted, and injected/entangled at each weight; scores all
four metrics; prints a per-image table; writes scores.csv + the chosen figures.

Examples
--------
  python scripts/run_single.py --prompt "a cat" --style Van_Gogh
  python scripts/run_single.py --prompt "a house on a hill" --style Monet \\
      --weights 0.5 1 2 --methods uniform patch_selective --figures F4 F5
  python scripts/run_single.py --prompt "a bear" --style Cubism --methods uniform
=============================================================
"""

import os
import re
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion_sae.config import STYLES
from eval.style_eval_lib import (
    Generator, Scorer, build_conditions, run_pipeline,
    detect_device, log, DEFAULT_WEIGHTS, DEFAULT_METHODS, DEFAULT_SEED,
    DEFAULT_UC_CKPT, DEFAULT_UC_REPO,
)
from eval.style_eval_figures import make_figures, ALL_FIGURES


def slug(text, n=40):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "prompt"


def parse_args():
    p = argparse.ArgumentParser(description="Single-prompt SAE style-steering eval",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompt", required=True, help="the (neutral) prompt to steer")
    p.add_argument("--style", required=True, help=f"style to inject (e.g. {', '.join(STYLES[:4])}, ...)")
    p.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    p.add_argument("--methods", nargs="+", choices=["uniform", "patch_selective"], default=DEFAULT_METHODS)
    p.add_argument("--skip-prompted", action="store_true",
                   help="do not generate the '<prompt> in <style> style.' reference")
    p.add_argument("--figures", nargs="+", default=["all"], choices=ALL_FIGURES + ["all"],
                   help="figures to render (subset of: grid, lines, tradeoff, per_style, bars)")
    p.add_argument("--no-save-images", dest="save_images", action="store_false",
                   help="do NOT persist per-condition images (default: all images saved)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--results-dir", default=None, help="default: results/single/<style>_<slug>")
    p.add_argument("--uc-checkpoint", default=DEFAULT_UC_CKPT)
    p.add_argument("--uc-repo", default=DEFAULT_UC_REPO)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grid-prompts", type=int, default=1, help="max prompts in the F1 grid")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def print_table(rows):
    hdr = f"{'condition':<16}{'prompt_clip':>12}{'style_clip':>12}{'uc_style':>10}{'content_clip':>14}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        def f(x):
            return "  n/a" if x is None else f"{x:.4f}"
        print(f"{r['condition']:<16}{f(r['prompt_clip']):>12}{f(r['style_clip']):>12}"
              f"{f(r['uc_style']):>10}{f(r['content_clip']):>14}")
    print()


def main():
    args = parse_args()
    device = detect_device(args.device)
    results_dir = args.results_dir or os.path.join(REPO, "results/single",
                                                   f"{args.style}_{slug(args.prompt)}")
    figures_dir = os.path.join(results_dir, "figures")

    conditions = build_conditions(args.weights, args.methods, include_prompted=not args.skip_prompted)
    log(f"device={device} | style={args.style} | prompt={args.prompt!r}")
    log(f"conditions ({len(conditions)}): {[c.name for c in conditions]} -> {results_dir}")

    scorer = Scorer(device=device, uc_checkpoint=args.uc_checkpoint,
                    uc_repo=args.uc_repo, batch_size=args.batch_size)
    if not scorer.has_uc():
        log("WARNING: UnlearnCanvas classifier unavailable -> uc_style will be BLANK.")
        log(f"  fix: pip install timm  &&  place style50.pth at {args.uc_checkpoint}")
        log("  (Google Drive folder 18dhkXyZQWjdMvlAlxZx3fZhdCZvlj2Hw -> subdir 'classifiers')")
    generator = Generator(device=device, seed=args.seed)
    if args.style not in generator.available_styles():
        sys.exit(f"no SAE features for style '{args.style}'. available: {generator.available_styles()}")

    rows, csv_path, images = run_pipeline({args.style: [args.prompt]}, conditions, scorer, results_dir,
                                          generator=generator, seed=args.seed,
                                          skip_existing=args.skip_existing, save_images=args.save_images)
    print_table(rows)
    if not args.no_figures:
        make_figures(rows, args.figures, figures_dir, images=images, grid_max_prompts=args.grid_prompts)
    log(f"done. scores -> {csv_path} | images -> {os.path.join(results_dir, 'images')} "
        f"| figures -> {figures_dir}")


if __name__ == "__main__":
    main()

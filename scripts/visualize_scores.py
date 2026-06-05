#!/usr/bin/env python
"""
visualize_scores.py
=============================================================
Large-scale trend figures over a whole scores.csv cache (as written by
run_style_eval.py / run_single.py). Use this after a batch run across all the
prompt files to see how the four conditions behave across styles and prompts.

Conditions are grouped into 4 colored CATEGORIES:
  unstyled          -> gold     (the prompt as-is)
  prompted          -> red      ("<prompt> in <style> style.")
  perturbed-normal  -> green    (SAE 'uniform' injection -- one vector per patch)
  perturbed-patch   -> blue     (SAE 'patch_selective' injection -- per-patch feature set)
(perturbed categories aggregate over all steering weights unless --by-weight.)

Figures (choose with --figures, default all):
  scatter      A1  per-image scatter: x=content_clip, y=style_clip, colored by category
  style        A2  style_clip vs style (x = each style), one line per category (±95% CI)
  content      A3  content_clip vs style, one line per category (±95% CI)
  uc           A4  uc_style vs style, one line per category (skipped if no UC scores)
  dist         A5  box-plots of style_clip & content_clip per category (distribution view)

Examples
--------
  python scripts/visualize_scores.py
  python scripts/visualize_scores.py --scores results/style_eval/scores.csv --figures scatter style
  python scripts/visualize_scores.py --by-weight          # split perturbed lines per weight
=============================================================
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from style_eval_lib import load_scores, log, DEFAULT_RESULTS
from style_eval_figures import mean_ci
try:
    from diffusion_sae.config import STYLES as CONFIG_STYLES
except Exception:
    CONFIG_STYLES = []

# ---- categories (the 4 colored groups the user asked for) ---- #
CATS = ["unstyled", "prompted", "perturbed-normal", "perturbed-patch"]
CAT_COLOR = {"unstyled": "gold", "prompted": "red",
             "perturbed-normal": "green", "perturbed-patch": "blue"}
ALL_FIGURES = ["scatter", "style", "content", "uc", "dist"]


def category(r):
    if r["kind"] == "unstyled":
        return "unstyled"
    if r["kind"] == "prompted":
        return "prompted"
    if r["method"] == "uniform":
        return "perturbed-normal"
    if r["method"] == "patch_selective":
        return "perturbed-patch"
    return None


def ordered_styles(rows):
    """Styles present, in config order first, then any extras alphabetically."""
    present = {r["style"] for r in rows}
    head = [s for s in CONFIG_STYLES if s in present]
    tail = sorted(present - set(head))
    return head + tail


def disp(style):
    return style.replace("_", " ")


def has_uc(rows):
    return any(r["uc_style"] is not None for r in rows)


def line_series(rows, by_weight):
    """Yield (label, color, linestyle, predicate) for each line in the per-style plots."""
    weights = sorted({r["weight"] for r in rows if r["kind"] == "sae"})
    styles = ["-", "--", ":", "-."]
    series = []
    for cat in CATS:
        if cat in ("unstyled", "prompted") or not by_weight:
            series.append((cat, CAT_COLOR[cat], "-", (lambda r, c=cat: category(r) == c)))
        else:
            for i, w in enumerate(weights):
                series.append((f"{cat} w{w:g}", CAT_COLOR[cat], styles[i % 4],
                               (lambda r, c=cat, w=w: category(r) == c and r["weight"] == w)))
    return series


# ------------------------------- A1 --------------------------------- #
def fig_scatter(rows, out_path, by_weight=False):
    fig, ax = plt.subplots(figsize=(8, 7))
    for cat in CATS:
        xs = [r["content_clip"] for r in rows if category(r) == cat and r["content_clip"] is not None]
        ys = [r["style_clip"] for r in rows if category(r) == cat and r["style_clip"] is not None]
        if by_weight and cat.startswith("perturbed"):
            sizes = [12 + 22 * r["weight"] for r in rows
                     if category(r) == cat and r["content_clip"] is not None]
        else:
            sizes = 18
        ax.scatter(xs, ys, s=sizes, c=CAT_COLOR[cat], alpha=0.45,
                   edgecolors="black", linewidths=0.2, label=cat)
    ax.set_xlabel("similarity to unstyled  →   CLIP(img, unstyled)")
    ax.set_ylabel("style similarity  →   CLIP(img, \"<style> style\")")
    ax.set_title("Per-image style vs similarity-to-unstyled  (each point = one generated image)",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


# --------------------------- A2 / A3 / A4 --------------------------- #
def fig_by_style(rows, out_path, metric, ylabel, title, by_weight=False):
    styles = ordered_styles(rows)
    x = np.arange(len(styles))
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(styles) + 3), 6))
    for label, color, ls, pred in line_series(rows, by_weight):
        ys, los, his = [], [], []
        for s in styles:
            m, ci = mean_ci([r[metric] for r in rows if pred(r) and r["style"] == s])
            ys.append(m); los.append(m - ci); his.append(m + ci)
        ys = np.array(ys, float)
        if np.all(np.isnan(ys)):
            continue
        ax.plot(x, ys, ls, marker="o", ms=4, color=color, label=label)
        ax.fill_between(x, np.array(los), np.array(his), color=color, alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels([disp(s) for s in styles], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


# ------------------------------- A5 --------------------------------- #
def fig_distributions(rows, out_path):
    metrics = [("style_clip", "CLIP style similarity"), ("content_clip", "CLIP similarity to unstyled")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (metric, label) in zip(axes, metrics):
        data, colors, ticks = [], [], []
        for cat in CATS:
            vals = [r[metric] for r in rows if category(r) == cat and r[metric] is not None]
            if vals:
                data.append(vals); colors.append(CAT_COLOR[cat]); ticks.append(cat)
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticklabels(ticks, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(label); ax.set_title(f"{label} by category", fontsize=11)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Score distributions per category", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


# ------------------------------ main -------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Aggregate trend figures over a scores.csv cache",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scores", default=os.path.join(DEFAULT_RESULTS, "scores.csv"),
                   help="path to the scores.csv cache")
    p.add_argument("--out", default=None, help="output dir (default: <scores_dir>/aggregate_figures)")
    p.add_argument("--figures", nargs="+", default=["all"], choices=ALL_FIGURES + ["all"])
    p.add_argument("--by-weight", action="store_true",
                   help="split perturbed categories into one line/series per steering weight")
    p.add_argument("--styles", nargs="+", default=None, help="restrict to these styles")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.scores):
        sys.exit(f"no scores cache at {args.scores} (run run_style_eval.py first)")
    rows = load_scores(args.scores)
    if args.styles:
        rows = [r for r in rows if r["style"] in set(args.styles)]
    if not rows:
        sys.exit("no rows to plot after filtering")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores)), "aggregate_figures")
    os.makedirs(out_dir, exist_ok=True)

    n_img = len(rows)
    n_style = len({r["style"] for r in rows})
    n_prompt = len({(r["style"], r["prompt_id"]) for r in rows})
    log(f"{n_img} scored images | {n_style} styles | {n_prompt} (style,prompt) pairs -> {out_dir}")
    counts = defaultdict(int)
    for r in rows:
        counts[category(r)] += 1
    log("per-category counts: " + ", ".join(f"{c}={counts[c]}" for c in CATS))

    figs = ALL_FIGURES if "all" in args.figures else args.figures
    saved = []
    if "scatter" in figs:
        saved.append(fig_scatter(rows, os.path.join(out_dir, "A1_scatter_style_vs_content.png"),
                                 by_weight=args.by_weight))
    if "style" in figs:
        saved.append(fig_by_style(rows, os.path.join(out_dir, "A2_style_clip_by_style.png"),
                                  "style_clip", "CLIP style similarity",
                                  "Style similarity by style", by_weight=args.by_weight))
    if "content" in figs:
        saved.append(fig_by_style(rows, os.path.join(out_dir, "A3_content_clip_by_style.png"),
                                  "content_clip", "CLIP similarity to unstyled",
                                  "Similarity to unstyled (CLIP) by style", by_weight=args.by_weight))
    if "uc" in figs:
        if has_uc(rows):
            saved.append(fig_by_style(rows, os.path.join(out_dir, "A4_uc_style_by_style.png"),
                                      "uc_style", "UnlearnCanvas P(style)",
                                      "Classifier style score by style", by_weight=args.by_weight))
        else:
            log("uc figure skipped -- no uc_style values in cache (download style50.pth)")
    if "dist" in figs:
        saved.append(fig_distributions(rows, os.path.join(out_dir, "A5_distributions.png")))

    for s in saved:
        log("figure: " + s)
    log(f"done: {len(saved)} figures in {out_dir}")


if __name__ == "__main__":
    main()

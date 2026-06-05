"""
style_eval_figures.py
=============================================================
Figures for the SAE style-steering evaluation, built from scores.csv rows
(see style_eval_lib.load_scores). Pure numpy + matplotlib -- no pandas.

Figure keys (CLI tokens) -> output filename:
  grid       -> F1_grid               qualitative image grid (<=4 images/row, 3-score captions)
  lines      -> F2_line_graph_quad    2x2: each score vs weightage, injected_normal vs injected_patches
  tradeoff   -> F3_tradeoff           style-strength vs content-preservation Pareto scatter
  per_style  -> F4_per_style          small-multiples: style strength vs weightage per style
  bars       -> F5_bars               grouped mean-per-condition bars across the 4 metrics

Filenames are intentionally terse (the results live in a prompt-specific subfolder,
so the figure number + kind is enough). make_figures(...) is the dispatcher.

The grid pulls images from the in-memory `images` dict returned by run_pipeline
(falling back to img_path on disk when --save-images was used), so individual
images need not be persisted.
=============================================================
"""

import os
import sys
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from eval.style_eval_lib import METRICS, METRIC_LABELS, METHOD_LABEL, style_display

# CLI tokens, in render order
ALL_FIGURES = ["grid", "lines", "tradeoff", "per_style", "bars"]
METHOD_COLOR = {"uniform": "#1f77b4", "patch_selective": "#ff7f0e"}


# ----------------------------- helpers ------------------------------ #
def mean_ci(vals):
    """mean and 95% CI half-width over non-null values; (nan, nan) if empty."""
    a = np.array([v for v in vals if v is not None], dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, np.nan
    m = float(a.mean())
    ci = float(1.96 * a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
    return m, ci


def _vals(rows, metric, **eq):
    return [r[metric] for r in rows if all(r.get(k) == v for k, v in eq.items())]


def sae_weights(rows):
    return sorted({r["weight"] for r in rows if r["kind"] == "sae"})


def sae_methods(rows):
    return [m for m in ("uniform", "patch_selective") if any(r["kind"] == "sae" and r["method"] == m for r in rows)]


def condition_order(rows):
    return list(dict.fromkeys(r["condition"] for r in rows))


def styles_in(rows):
    return list(dict.fromkeys(r["style"] for r in rows))


def has_uc(rows):
    return any(r["uc_style"] is not None for r in rows)


# =============================== F1 grid ============================= #
def _caption3(r):
    """The image's three style/content scores (prompt_clip omitted)."""
    def f(x):
        return "n/a" if x is None else f"{x:.2f}"
    return f"sty {f(r['style_clip'])} | uc {f(r['uc_style'])} | cnt {f(r['content_clip'])}"


def _get_image(style, pid, cond, r, images):
    if images and (style, pid, cond) in images:
        return images[(style, pid, cond)]
    p = r.get("img_path")
    if p and os.path.exists(p):
        return Image.open(p).convert("RGB")
    return None


def fig_grids(rows, figures_dir, images=None, max_per_row=4, max_prompts=None):
    """One image grid per style. Each prompt's conditions are laid out left-to-right,
    wrapping at `max_per_row` images per row; every cell is captioned with its 3 scores.
    Returns the list of saved paths (F1_grid.png, or F1_grid_<style>.png if >1 style)."""
    by_style = defaultdict(list)
    for r in rows:
        by_style[r["style"]].append(r)
    multi_style = len(by_style) > 1
    saved = []

    for style, srows in by_style.items():
        index = {(r["prompt_id"], r["condition"]): r for r in srows}
        conds = condition_order(srows)
        pids = sorted({r["prompt_id"] for r in srows})
        if max_prompts:
            pids = pids[:max_prompts]
        multi_prompt = len(pids) > 1

        # rows of <=max_per_row cells; never mix prompts within a row
        grid = []
        for pid in pids:
            line = []
            for cond in conds:
                line.append((pid, cond))
                if len(line) == max_per_row:
                    grid.append(line); line = []
            if line:
                grid.append(line)
        if not grid:
            continue

        nrows, ncols = len(grid), max_per_row
        # generous hspace so the 2-line caption never overlaps the image in the row above
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3.5 * nrows + 0.5),
                                 squeeze=False, gridspec_kw={"hspace": 0.6, "wspace": 0.08})
        for ax in axes.flat:
            ax.axis("off")
        for ri, line in enumerate(grid):
            for ci, (pid, cond) in enumerate(line):
                ax = axes[ri][ci]
                r = index.get((pid, cond))
                if r is None:
                    continue
                im = _get_image(style, pid, cond, r, images)
                if im is not None:
                    ax.imshow(im)
                title = (f"#{pid} " if multi_prompt else "") + cond.replace("_", " ")
                # caption (all 3 scores) folded into the title -- set_xlabel is hidden by axis("off")
                ax.set_title(f"{title}\n{_caption3(r)}", fontsize=8)
        # no tight_layout -- it would clobber the explicit hspace above
        fig.suptitle(f"{style_display(style)}: unstyled vs injected_normal vs injected_patches\n"
                     f"captions  sty = CLIP style sim · uc = UnlearnCanvas P(style) · cnt = CLIP sim to unstyled",
                     fontsize=11)
        name = "F1_grid" + (f"_{style}" if multi_style else "")
        out = os.path.join(figures_dir, name + ".png")
        plt.savefig(out, dpi=95, bbox_inches="tight"); plt.close(fig)
        saved.append(out)
    return saved


# ============================ F2 line quad =========================== #
def fig_metric_vs_weight(rows, out_path):
    weights = sae_weights(rows)
    methods = sae_methods(rows)
    xs = [0.0] + weights
    uc_ok = has_uc(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.flat, METRICS):
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.set_xlabel("steering weightage"); ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        if metric == "uc_style" and not uc_ok:
            ax.text(0.5, 0.5, "no UC checkpoint\n(uc_style unavailable)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            continue
        m0, c0 = mean_ci(_vals(rows, metric, kind="unstyled"))      # weight 0 anchor (shared)
        for method in methods:
            ys, es = [m0], [c0]
            for w in weights:
                mm, cc = mean_ci([r[metric] for r in rows
                                  if r["kind"] == "sae" and r["method"] == method and r["weight"] == w])
                ys.append(mm); es.append(cc)
            ys, es = np.array(ys), np.array(es)
            ax.plot(xs, ys, "-o", color=METHOD_COLOR.get(method), label=METHOD_LABEL.get(method, method))
            ax.fill_between(xs, ys - es, ys + es, color=METHOD_COLOR.get(method), alpha=0.15)
        pm, _ = mean_ci(_vals(rows, metric, kind="prompted"))
        if not np.isnan(pm):
            ax.axhline(pm, ls="--", lw=1.2, color="gray", label="prompted ref")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle("Metric vs steering weightage  (mean ± 95% CI over styles × prompts)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


# ============================= F3 tradeoff =========================== #
def fig_tradeoff(rows, out_path):
    use_uc = has_uc(rows)
    xkey = "uc_style" if use_uc else "style_clip"
    xlabel = "UnlearnCanvas P(style)" if use_uc else "CLIP style similarity"
    methods, weights, styles = sae_methods(rows), sae_weights(rows), styles_in(rows)

    fig, ax = plt.subplots(figsize=(8, 7))
    for method in methods:
        for w in weights:
            for style in styles:
                sub = [r for r in rows if r["kind"] == "sae" and r["method"] == method
                       and r["weight"] == w and r["style"] == style]
                x, _ = mean_ci([r[xkey] for r in sub])
                y, _ = mean_ci([r["content_clip"] for r in sub])
                if np.isnan(x) or np.isnan(y):
                    continue
                ax.scatter(x, y, s=25 + 45 * w, color=METHOD_COLOR.get(method),
                           alpha=0.7, edgecolors="none")
    for kind, marker, color in [("unstyled", "s", "black"), ("prompted", "*", "dimgray")]:
        x, _ = mean_ci(_vals(rows, xkey, kind=kind))
        y, _ = mean_ci(_vals(rows, "content_clip", kind=kind))
        if not (np.isnan(x) or np.isnan(y)):
            ax.scatter(x, y, marker=marker, s=180, color=color, zorder=5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=METHOD_COLOR.get(m),
                          label=f"{METHOD_LABEL.get(m, m)} ({m})") for m in methods]
    handles += [plt.Line2D([], [], marker="o", ls="", color="gray",
                           markersize=math.sqrt(25 + 45 * w), label=f"weight {w:g}") for w in weights]
    handles += [plt.Line2D([], [], marker="s", ls="", color="black", label="unstyled"),
                plt.Line2D([], [], marker="*", ls="", color="dimgray", label="prompted")]
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.set_xlabel(f"style strength  →   {xlabel}")
    ax.set_ylabel("similarity to unstyled  →   CLIP(img, unstyled)")
    ax.set_title("Style strength vs similarity-to-unstyled  (up-left = more style, less drift)\n"
                 "each point = one style × method × weight (mean over prompts)", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


# ============================ F4 per-style =========================== #
def fig_per_style(rows, out_path):
    metric = "uc_style" if has_uc(rows) else "style_clip"
    ylabel = "UC P(style)" if metric == "uc_style" else "CLIP style sim"
    styles, methods, weights = styles_in(rows), sae_methods(rows), sae_weights(rows)
    xs = [0.0] + weights

    ncols = min(5, max(1, len(styles)))
    nrows = math.ceil(len(styles) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, style in enumerate(styles):
        ax = axes[i // ncols][i % ncols]; ax.axis("on")
        m0, _ = mean_ci([r[metric] for r in rows if r["kind"] == "unstyled" and r["style"] == style])
        for method in methods:
            ys = [m0]
            for w in weights:
                mm, _ = mean_ci([r[metric] for r in rows if r["kind"] == "sae"
                                 and r["method"] == method and r["weight"] == w and r["style"] == style])
                ys.append(mm)
            ax.plot(xs, ys, "-o", ms=4, color=METHOD_COLOR.get(method),
                    label=METHOD_LABEL.get(method, method))
        ax.set_title(style_display(style), fontsize=9)
        ax.set_xticks(xs); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Per-style style strength vs weightage  ({ylabel})", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


# ============================== F5 bars ============================== #
def fig_summary_bars(rows, out_path):
    conds = condition_order(rows)
    x = np.arange(len(METRICS))
    width = 0.8 / max(len(conds), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.get_cmap("tab10")
    for ci, cond in enumerate(conds):
        means = []
        for metric in METRICS:
            m, _ = mean_ci([r[metric] for r in rows if r["condition"] == cond])
            means.append(0.0 if np.isnan(m) else m)
        ax.bar(x + ci * width, means, width, label=cond.replace("_", " "), color=cmap(ci % 10))
    ax.set_xticks(x + width * (len(conds) - 1) / 2)
    ax.set_xticklabels([METRIC_LABELS[m].split("\n")[0] for m in METRICS], fontsize=9)
    ax.set_ylabel("mean score")
    ax.set_title("Mean score per condition", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


# ---------------------------- dispatcher ---------------------------- #
def make_figures(rows, figures, figures_dir, images=None, grid_max_prompts=None, max_per_row=4):
    """Render the requested figures (subset of ALL_FIGURES or 'all'). Returns saved paths.
    `images` is the in-memory {(style,pid,cond): PIL} map from run_pipeline (for the grid)."""
    os.makedirs(figures_dir, exist_ok=True)
    figures = ALL_FIGURES if (not figures or "all" in figures) else figures
    saved = []
    if not rows:
        print("make_figures: no rows -- nothing to plot")
        return saved
    if "grid" in figures:
        g = fig_grids(rows, figures_dir, images=images, max_per_row=max_per_row,
                      max_prompts=grid_max_prompts)
        if not g:
            print("  grid: no images available (use --save-images, or run with a generator)")
        saved.extend(g)
    if "lines" in figures:
        saved.append(fig_metric_vs_weight(rows, os.path.join(figures_dir, "F2_line_graph_quad.png")))
    if "tradeoff" in figures:
        saved.append(fig_tradeoff(rows, os.path.join(figures_dir, "F3_tradeoff.png")))
    if "per_style" in figures:
        saved.append(fig_per_style(rows, os.path.join(figures_dir, "F4_per_style.png")))
    if "bars" in figures:
        saved.append(fig_summary_bars(rows, os.path.join(figures_dir, "F5_bars.png")))
    for p in saved:
        print("  figure:", p)
    return saved

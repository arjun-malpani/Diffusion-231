#!/usr/bin/env python
"""
make_results_figures.py
=============================================================
Publication-quality figures for the SAE style-steering results, built ENTIRELY
from a scores.csv cache -- no diffusion, no model loading, no image files needed.
Run it as many times as you like to iterate on the visuals.

    python eval/make_results_figures.py
    python eval/make_results_figures.py --scores results/style_eval/scores.csv --out figs/
    python eval/make_results_figures.py --figures tradeoff heatmaps

Self-contained on purpose (only numpy + matplotlib + csv) so it keeps working
regardless of where the rest of the pipeline modules live.

Condition groups & colors (red / purple / blue theme):
    unstyled            gray    (prompt as-is)
    prompted            red     ("<prompt> in <style> style.")
    injected_normal     purple  (SAE 'uniform' injection -- one vector per patch)
    injected_patches    blue    (SAE 'patch'   injection -- one embedding per patch)

Metrics (columns in scores.csv):
    style_clip    CLIP cos(image, "<style> style")     -- style alignment (CLIP)
    uc_style      UnlearnCanvas ViT-L P(style)          -- style alignment (classifier)
    content_clip  CLIP cos(image, unstyled image)       -- "CLIP similarity to unstyled"
    prompt_clip   CLIP cos(image, original prompt)      -- prompt fidelity

Figures produced:
    R1  tradeoff        style vs similarity-to-unstyled, with weight trajectories  (the hero)
    R2  weight          each metric vs steering weight, normal vs patches (±95% CI)
    R3  heatmaps        style x condition heatmaps for style_clip and content_clip
    R4  gaincost        per-condition style gain (→) vs content cost (←) diverging bars
    R5  distributions   violins of each metric by condition group
    R6  perstyle        per-style style-vs-weight small multiples
    R7  agreement       CLIP style sim vs classifier P(style) -- do the metrics agree?
    R8  summary         grouped mean bars at max steering vs unstyled/prompted
=============================================================
"""

import os
import csv
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ----------------------------- palette ------------------------------ #
COL = {
    "unstyled":         "#9AA0A6",   # neutral gray
    "prompted":         "#E63946",   # red
    "injected_normal":  "#7B2CBF",   # purple  (uniform)
    "injected_patches": "#2A6FDB",   # blue    (patch)
}
CAT_ORDER = ["unstyled", "prompted", "injected_normal", "injected_patches"]
CAT_LABEL = {
    "unstyled": "unstyled",
    "prompted": "prompted",
    "injected_normal": "injected_normal (uniform)",
    "injected_patches": "injected_patches (patch)",
}
# on-theme sequential / diverging colormaps
CMAP_STYLE = LinearSegmentedColormap.from_list("style_seq", ["#F7F4FB", "#B388EB", "#7B2CBF", "#4A0F7A"])
CMAP_CONT  = LinearSegmentedColormap.from_list("cont_seq",  ["#F2F7FF", "#90B8F0", "#2A6FDB", "#143C82"])
CMAP_DIV   = LinearSegmentedColormap.from_list("rpb_div",   ["#2A6FDB", "#FFFFFF", "#E63946"])

METRICS = {
    "style_clip":   "CLIP style similarity",
    "uc_style":     "UnlearnCanvas P(style)",
    "content_clip": "CLIP similarity to unstyled",
    "prompt_clip":  "CLIP prompt fidelity",
}
ALL_FIGURES = ["tradeoff", "weight", "heatmaps", "gaincost", "distributions",
               "perstyle", "agreement", "summary"]


def set_theme():
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#555555", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": "#E2E2E8", "grid.linewidth": 0.9, "grid.alpha": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.labelsize": 11, "axes.labelcolor": "#222222", "axes.labelweight": "bold",
        "font.size": 11, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "xtick.color": "#333333", "ytick.color": "#333333",
        "legend.fontsize": 9, "legend.frameon": False,
        "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


# ------------------------------ data -------------------------------- #
def load_scores(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("prompt_clip", "style_clip", "uc_style", "content_clip", "weight"):
                r[k] = float(r[k]) if r.get(k) not in ("", None) else None
            r["prompt_id"] = int(r["prompt_id"])
            rows.append(r)
    return rows


def category(r):
    if r["kind"] == "unstyled":
        return "unstyled"
    if r["kind"] == "prompted":
        return "prompted"
    return "injected_normal" if r["method"] == "uniform" else "injected_patches"


def sae_weights(rows):
    return sorted({r["weight"] for r in rows if r["kind"] == "sae"})


def conditions_ordered(rows):
    present = {r["condition"] for r in rows}
    order = [c for c in ("unstyled", "prompted") if c in present]
    for w in sae_weights(rows):
        for lab in ("injected_normal", "injected_patches"):
            name = f"{lab}_w{w:g}"
            if name in present:
                order.append(name)
    return order


def styles_by_response(rows):
    """Styles ordered by how strongly injected_normal at max weight moves style_clip."""
    if not sae_weights(rows):
        return sorted({r["style"] for r in rows})
    wmax = max(sae_weights(rows))
    score = {}
    for s in {r["style"] for r in rows}:
        sub = [r["style_clip"] for r in rows
               if r["style"] == s and r["method"] == "uniform" and r["weight"] == wmax]
        score[s] = np.mean(sub) if sub else 0.0
    return sorted(score, key=lambda s: -score[s])


def _arr(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    return a[~np.isnan(a)]


def mean_ci(vals):
    a = _arr(vals)
    if a.size == 0:
        return np.nan, np.nan
    m = float(a.mean())
    ci = float(1.96 * a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return m, ci


def _save(fig, out_path, suptitle=None):
    # titles intentionally omitted -- figures are captioned externally. `suptitle`
    # is accepted (and ignored) so callers stay unchanged.
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _panel(ax, letter):
    """Bold sub-caption (a)/(b)/... just above the panel's top-left corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 8), textcoords="offset points",
                fontsize=13, fontweight="bold", va="bottom", ha="left")


# =============================== R1 ================================= #
def fig_tradeoff(rows, out_path):
    weights = sae_weights(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), layout="constrained")
    for i, (ax, ycol) in enumerate(zip(axes, ("style_clip", "uc_style"))):
        # SAE weight trajectories
        for cat in ("injected_normal", "injected_patches"):
            xs, ys = [], []
            for w in weights:
                sub = [r for r in rows if r["kind"] == "sae" and category(r) == cat and r["weight"] == w]
                xs.append(mean_ci([r["content_clip"] for r in sub])[0])
                ys.append(mean_ci([r[ycol] for r in sub])[0])
            ax.plot(xs, ys, "-", color=COL[cat], lw=2.4, alpha=0.85, zorder=3)
            wn = [(w - min(weights)) / (max(weights) - min(weights) + 1e-9) for w in weights]
            ax.scatter(xs, ys, s=[70 + 200 * t for t in wn], color=COL[cat],
                       edgecolor="white", linewidth=1.3, zorder=4)
            ax.annotate(f"w{weights[0]:g}", (xs[0], ys[0]), fontsize=7.5, color=COL[cat],
                        xytext=(4, 6), textcoords="offset points")
            ax.annotate(f"w{weights[-1]:g}", (xs[-1], ys[-1]), fontsize=7.5, color=COL[cat],
                        fontweight="bold", xytext=(4, 6), textcoords="offset points")
        # reference points
        for cat, marker in (("unstyled", "s"), ("prompted", "*")):
            x = mean_ci([r["content_clip"] for r in rows if category(r) == cat])[0]
            y = mean_ci([r[ycol] for r in rows if category(r) == cat])[0]
            ax.scatter(x, y, marker=marker, s=320 if marker == "*" else 170, color=COL[cat],
                       edgecolor="white", linewidth=1.4, zorder=5)
        ax.set_xlabel("CLIP similarity to unstyled  →  (content preserved)")
        ax.set_ylabel(METRICS[ycol] + "  →  (more style)")
        _panel(ax, "ab"[i])
    handles = [Line2D([], [], marker="o", ls="", color=COL[c], label=CAT_LABEL[c]) for c in CAT_ORDER]
    handles.append(Line2D([], [], marker="", ls="", label="marker size grows with weight (1->2)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5)
    return _save(fig, out_path, "Style ↔ content trade-off   (up-left is better: more style, less drift)")


# =============================== R2 ================================= #
def fig_weight(rows, out_path):
    weights = sae_weights(rows)
    xs = [0.0] + weights
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for i, (ax, metric) in enumerate(zip(axes.flat, ("style_clip", "uc_style", "content_clip", "prompt_clip"))):
        m0, c0 = mean_ci([r[metric] for r in rows if r["kind"] == "unstyled"])
        for cat in ("injected_normal", "injected_patches"):
            ys, es = [m0], [c0]
            for w in weights:
                m, c = mean_ci([r[metric] for r in rows
                                if r["kind"] == "sae" and category(r) == cat and r["weight"] == w])
                ys.append(m); es.append(c)
            ys, es = np.array(ys), np.array(es)
            ax.fill_between(xs, ys - es, ys + es, color=COL[cat], alpha=0.16, lw=0)
            ax.plot(xs, ys, "-o", color=COL[cat], lw=2.4, ms=6, mec="white", mew=1.2,
                    label=CAT_LABEL[cat])
        pm, _ = mean_ci([r[metric] for r in rows if r["kind"] == "prompted"])
        if not np.isnan(pm):
            ax.axhline(pm, ls=(0, (5, 3)), lw=1.6, color=COL["prompted"], label="prompted ref")
        ax.scatter([0], [m0], s=80, color=COL["unstyled"], zorder=5, ec="white")
        ax.set_xticks(xs); ax.set_xlabel("steering weight")
        ax.set_ylabel(METRICS[metric])
        _panel(ax, "abcd"[i])
    axes.flat[0].legend(loc="best")
    return _save(fig, out_path, "Each metric vs steering weight   (mean ± 95% CI over 10 styles × 20 prompts)")


# =============================== R3 ================================= #
def _heatmap(ax, rows, styles, conds, metric, cmap, title, show_x=True):
    M = np.full((len(styles), len(conds)), np.nan)
    for i, s in enumerate(styles):
        for j, c in enumerate(conds):
            M[i, j] = mean_ci([r[metric] for r in rows if r["style"] == s and r["condition"] == c])[0]
    im = ax.imshow(M, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(conds)))
    if show_x:
        ax.set_xticklabels([c.replace("injected_", "").replace("_", " ") for c in conds],
                           rotation=45, ha="right", fontsize=9)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(len(styles)))
    ax.set_yticklabels([s.replace("_", " ") for s in styles], fontsize=9)
    ax.grid(False)
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    for i in range(len(styles)):
        for j in range(len(conds)):
            v = M[i, j]
            if np.isnan(v):
                continue
            t = (v - vmin) / (vmax - vmin + 1e-9)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if t > 0.55 else "#333333")
    cbar = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cbar.set_label(title, fontsize=11, fontweight="bold")   # metric identity -> colorbar label


def fig_heatmaps(rows, out_path):
    styles = styles_by_response(rows)
    conds = conditions_ordered(rows)
    # stacked (full-width) instead of side-by-side -> the 12 condition labels get room
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), layout="constrained")
    _heatmap(axes[0], rows, styles, conds, "style_clip", CMAP_STYLE, "CLIP style similarity", show_x=False)
    _heatmap(axes[1], rows, styles, conds, "content_clip", CMAP_CONT, "CLIP similarity to unstyled")
    _panel(axes[0], "a"); _panel(axes[1], "b")
    return _save(fig, out_path)


# =============================== R4 ================================= #
def fig_gaincost(rows, out_path):
    conds = [c for c in conditions_ordered(rows) if c not in ("unstyled",)]
    base_style = mean_ci([r["style_clip"] for r in rows if r["kind"] == "unstyled"])[0]
    base_cont = mean_ci([r["content_clip"] for r in rows if r["kind"] == "unstyled"])[0]  # ~1.0
    gains, costs, colors = [], [], []
    for c in conds:
        gains.append(mean_ci([r["style_clip"] for r in rows if r["condition"] == c])[0] - base_style)
        costs.append(mean_ci([r["content_clip"] for r in rows if r["condition"] == c])[0] - base_cont)  # <=0
        rep = next(r for r in rows if r["condition"] == c)
        colors.append(COL[category(rep)])
    y = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(conds) + 2.5), layout="constrained")
    ax.barh(y, gains, color=colors, alpha=0.95, height=0.62, label="style gain (→)")
    ax.barh(y, costs, color=colors, alpha=0.45, height=0.62, hatch="///", label="content cost (←)")
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([c.replace("_", " ") for c in conds], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ vs unstyled    ◄ content lost    |    style gained ►")
    handles = [Line2D([], [], marker="s", ls="", color="#555", label="style gain (solid)"),
               Line2D([], [], marker="s", ls="", color="#999", label="content cost (hatched)")]
    ax.legend(handles=handles, loc="lower right")
    return _save(fig, out_path, "Style gain vs content cost, per condition   (relative to unstyled)")


# =============================== R5 ================================= #
def fig_distributions(rows, out_path):
    cats = [c for c in CAT_ORDER if any(category(r) == c for r in rows)]
    cols = [("style_clip", "CLIP style sim"), ("uc_style", "UC P(style)"),
            ("content_clip", "CLIP sim to unstyled")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), layout="constrained")
    for i, (ax, (col, lab)) in enumerate(zip(axes, cols)):
        data = [_arr([r[col] for r in rows if category(r) == c]) for c in cats]
        parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.82)
        for b, c in zip(parts["bodies"], cats):
            b.set_facecolor(COL[c]); b.set_alpha(0.7); b.set_edgecolor("white"); b.set_linewidth(1.0)
        if "cmeans" in parts:
            parts["cmeans"].set_color("#222"); parts["cmeans"].set_linewidth(1.6)
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels([CAT_LABEL[c].replace(" (", "\n(") for c in cats], fontsize=8)
        ax.set_ylabel(lab)
        _panel(ax, "abc"[i])
    return _save(fig, out_path, "Score distributions by condition group   (violins, mean marked)")


# =============================== R6 ================================= #
def fig_perstyle(rows, out_path):
    styles = styles_by_response(rows)
    weights = sae_weights(rows)
    xs = [0.0] + weights
    ncols = 5
    nrows = int(np.ceil(len(styles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)
    for i, s in enumerate(styles):
        ax = axes[i // ncols][i % ncols]; ax.set_visible(True)
        m0 = mean_ci([r["style_clip"] for r in rows if r["kind"] == "unstyled" and r["style"] == s])[0]
        for cat in ("injected_normal", "injected_patches"):
            ys = [m0] + [mean_ci([r["style_clip"] for r in rows if r["style"] == s
                                  and category(r) == cat and r["weight"] == w])[0] for w in weights]
            ax.plot(xs, ys, "-o", color=COL[cat], lw=2, ms=4, mec="white", mew=0.8)
        ax.text(0.04, 0.94, s.replace("_", " "), transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top", ha="left")   # panel identifier (label, not a title)
        ax.set_xticks([0, 1, 2]); ax.tick_params(labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel("CLIP style sim", fontsize=9)
    handles = [Line2D([], [], marker="o", color=COL[c], label=CAT_LABEL[c]) for c in
               ("injected_normal", "injected_patches")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    return _save(fig, out_path)


# =============================== R7 ================================= #
def fig_agreement(rows, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 7.5), layout="constrained")
    for c in CAT_ORDER:
        xs = [r["style_clip"] for r in rows if category(r) == c and r["uc_style"] is not None]
        ys = [r["uc_style"] for r in rows if category(r) == c and r["uc_style"] is not None]
        ax.scatter(xs, ys, s=14, color=COL[c], alpha=0.35, edgecolors="none", label=CAT_LABEL[c])
    x = _arr([r["style_clip"] for r in rows]); y = _arr([r["uc_style"] for r in rows])
    if x.size > 2:
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.97, 0.03, f"Pearson r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11)
    ax.set_xlabel("CLIP style similarity"); ax.set_ylabel("UnlearnCanvas P(style)")
    leg = ax.legend(loc="upper left", markerscale=2)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return _save(fig, out_path, "Do the two style metrics agree?")


# =============================== R8 ================================= #
def fig_summary(rows, out_path):
    weights = sae_weights(rows)
    wmax = max(weights) if weights else None
    picks = [("unstyled", "unstyled"), ("prompted", "prompted")]
    if wmax is not None:
        picks += [(f"injected_normal_w{wmax:g}", "injected_normal"),
                  (f"injected_patches_w{wmax:g}", "injected_patches")]
    metrics = ["style_clip", "uc_style", "content_clip", "prompt_clip"]
    x = np.arange(len(metrics)); width = 0.8 / len(picks)
    fig, ax = plt.subplots(figsize=(12, 5.8), layout="constrained")
    for k, (cond, cat) in enumerate(picks):
        vals = [mean_ci([r[m] for r in rows if r["condition"] == cond])[0] for m in metrics]
        errs = [mean_ci([r[m] for r in rows if r["condition"] == cond])[1] for m in metrics]
        label = cond.replace("_", " ") if cond.startswith("injected") else cat
        ax.bar(x + k * width, vals, width, yerr=errs, capsize=3, color=COL[cat], alpha=0.92,
               edgecolor="white", linewidth=0.8, label=label)
    ax.set_xticks(x + width * (len(picks) - 1) / 2)
    ax.set_xticklabels([METRICS[m] for m in metrics], fontsize=9)
    ax.set_ylabel("mean score")
    ax.legend(loc="upper right")
    sub = f"at max steering weight w={wmax:g}" if wmax is not None else ""
    return _save(fig, out_path, f"Condition comparison across metrics   {sub}")


FIGS = {
    "tradeoff":      ("R1_tradeoff.png",       fig_tradeoff),
    "weight":        ("R2_metric_vs_weight.png", fig_weight),
    "heatmaps":      ("R3_heatmaps.png",       fig_heatmaps),
    "gaincost":      ("R4_gain_vs_cost.png",   fig_gaincost),
    "distributions": ("R5_distributions.png",  fig_distributions),
    "perstyle":      ("R6_per_style.png",      fig_perstyle),
    "agreement":     ("R7_clip_vs_classifier.png", fig_agreement),
    "summary":       ("R8_summary_bars.png",   fig_summary),
}


def main():
    p = argparse.ArgumentParser(description="Beautiful results figures from a scores.csv cache",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    p.add_argument("--scores", default=os.path.join(here, "results/style_eval/scores.csv"))
    p.add_argument("--out", default=None, help="output dir (default: <scores_dir>/results_figures)")
    p.add_argument("--figures", nargs="+", default=["all"], choices=ALL_FIGURES + ["all"])
    args = p.parse_args()

    if not os.path.isfile(args.scores):
        sys.exit(f"no scores.csv at {args.scores}")
    rows = load_scores(args.scores)
    if not rows:
        sys.exit("scores.csv is empty")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores)), "results_figures")
    os.makedirs(out, exist_ok=True)
    set_theme()

    wanted = ALL_FIGURES if "all" in args.figures else args.figures
    n_img = len(rows); n_pair = len({(r["style"], r["prompt_id"]) for r in rows})
    print(f"{n_img} rows | {len({r['style'] for r in rows})} styles | {n_pair} (style,prompt) pairs -> {out}")
    for key in wanted:
        fname, fn = FIGS[key]
        try:
            print("  saved", fn(rows, os.path.join(out, fname)))
        except Exception as e:
            print(f"  FAILED {key}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

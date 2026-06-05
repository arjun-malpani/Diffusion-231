"""
Build comparison grids -- one figure per (subgroup x method).

Each figure: rows = styles, columns = for each demo prompt [ baseline | s=0.5 | s=1.0 | s=2.0 ].
So you can read, per style row, the neutral baseline and how injection strength changes it.

Run from repo root (after run_experiment.py):  python steering/visualize.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from diffusion_sae.config import STYLES
from steering.common import OUT_DIR, DEMO_PROMPTS, STRENGTHS

SUBGROUPS = {"A": "A: training seed 42 (shared)",
             "B": "B: random, decoupled noise",
             "C": "C: new seed 1234 (shared)"}
METHODS = ["uniform", "patch_selective"]


def slug(p): return p.replace(" ", "_")

def load(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


def build_grid(subgroup, method):
    ncols = len(DEMO_PROMPTS) * (1 + len(STRENGTHS))      # baseline + strengths, per prompt
    nrows = len(STYLES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.1 * nrows))
    for ax in axes.flat:
        ax.axis("off")

    # column headers
    col = 0
    col_titles = []
    for prompt in DEMO_PROMPTS:
        col_titles.append(f'"{prompt}"\nbaseline')
        for s in STRENGTHS:
            col_titles.append(f'"{prompt}"\ns={s}')
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8)

    for r, style in enumerate(STYLES):
        axes[r, 0].set_ylabel(style.replace("_", " "), rotation=0, ha="right", va="center",
                              fontsize=9, labelpad=38)
        # make the ylabel visible even with axis off
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values():
            sp.set_visible(False)

        c = 0
        for prompt in DEMO_PROMPTS:
            bimg = load(os.path.join(OUT_DIR, subgroup, "baseline", f"{slug(prompt)}.png"))
            if bimg is not None:
                axes[r, c].imshow(bimg)
            c += 1
            for s in STRENGTHS:
                simg = load(os.path.join(OUT_DIR, subgroup, method, style,
                                         f"{slug(prompt)}__s{s}.png"))
                if simg is not None:
                    axes[r, c].imshow(simg)
                c += 1

    fig.suptitle(f"Style injection -- subgroup {SUBGROUPS[subgroup]} -- {method} method",
                 fontsize=14, y=0.997)
    plt.tight_layout()
    gdir = os.path.join(OUT_DIR, "grids"); os.makedirs(gdir, exist_ok=True)
    out = os.path.join(gdir, f"{subgroup}_{method}.png")
    plt.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    for subgroup in SUBGROUPS:
        for method in METHODS:
            build_grid(subgroup, method)
    print(f"\n{len(SUBGROUPS)*len(METHODS)} grids in {OUT_DIR}/grids/")


if __name__ == "__main__":
    main()

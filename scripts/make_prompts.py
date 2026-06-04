"""
Build the two prompt sets used after SAE training, from the styles in
scripts/config.py and the 20 per-class anchor files in data/prompts/by_class/.

Outputs (one file per style in each folder):

  data/prompts/feature_activation/<Style>.txt
      20 STYLED prompts per style -- "<anchor> in <Style> style." -- one per
      object class. Used to IDENTIFY each style's SAE features, the SAeUron way
      (compute_feature_importance: target style vs. all others). Every style
      reuses the SAME anchor row (index 0) so the ONLY thing that varies across
      styles is the style word -- that's what isolates style from content.

  data/prompts/inference/<Style>.txt
      20 NEUTRAL content prompts per style -- "<anchor>", no style word. Used at
      inference to steer the identified feature (style injection). Drawn from
      HELD-OUT anchor rows (index >= 1), disjoint from the feature-activation
      set, so we never evaluate on the prompts we used to pick the features.
      With DISTINCT_INFER_PER_STYLE, each style gets its own held-out row.

Run from the repo root (or anywhere -- paths are resolved against the repo):
    python scripts/make_prompts.py
"""

import os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))          # .../scripts
REPO = os.path.dirname(HERE)                                # repo root
sys.path.insert(0, HERE)                                    # so `import config` finds scripts/config.py
from config import STYLES                                   # the 10 styles


# ============================ CONFIG ============================
BY_CLASS_DIR = os.path.join(REPO, "data/prompts/by_class")
FEATURE_DIR  = os.path.join(REPO, "data/prompts/feature_activation")
INFER_DIR    = os.path.join(REPO, "data/prompts/inference")

FEATURE_ANCHOR_IDX = 0          # anchor row for the (styled) feature-ID prompts -- shared by all styles
INFER_ANCHOR_IDX   = 1          # first anchor row for the (neutral) inference prompts (held out from feature-ID)
DISTINCT_INFER_PER_STYLE = True # True -> style i uses row (INFER_ANCHOR_IDX + i); False -> all styles share row INFER_ANCHOR_IDX
# ================================================================


def clean(anchor: str) -> str:
    """Trim whitespace and a trailing period (SAeUron strips it before adding the style suffix)."""
    a = anchor.strip()
    return a[:-1] if a.endswith(".") else a


def load_class_anchors():
    """Return [(class_name, [anchor, ...]), ...] sorted by class name (deterministic)."""
    paths = sorted(glob.glob(os.path.join(BY_CLASS_DIR, "sd_prompt_*.txt")))
    if not paths:
        raise FileNotFoundError(f"no sd_prompt_*.txt files in {BY_CLASS_DIR}")
    out = []
    for p in paths:
        name = os.path.basename(p)[len("sd_prompt_"):-len(".txt")]
        with open(p) as f:
            anchors = [l.strip() for l in f if l.strip()]
        out.append((name, anchors))
    return out


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    classes = load_class_anchors()
    n_classes = len(classes)
    print(f"loaded {n_classes} object classes from {BY_CLASS_DIR}")
    print(f"styles ({len(STYLES)}): {', '.join(STYLES)}")

    # bounds check: the largest anchor row we'll touch must exist in every class
    max_idx = max(FEATURE_ANCHOR_IDX,
                  INFER_ANCHOR_IDX + (len(STYLES) - 1 if DISTINCT_INFER_PER_STYLE else 0))
    for name, anchors in classes:
        if len(anchors) <= max_idx:
            raise IndexError(f"class {name!r} has only {len(anchors)} anchors but row {max_idx} is needed")

    # ---- feature-activation prompts: styled, shared anchor row, one file per style ----
    for style in STYLES:
        style_str = style.replace("_", " ")
        lines = [f"{clean(anchors[FEATURE_ANCHOR_IDX])} in {style_str} style."
                 for _, anchors in classes]
        write_lines(os.path.join(FEATURE_DIR, f"{style}.txt"), lines)
    print(f"\nwrote {len(STYLES)} files to {FEATURE_DIR}  "
          f"({n_classes} styled prompts each, anchor row {FEATURE_ANCHOR_IDX})")

    # ---- inference prompts: neutral content, held-out anchor row(s), one file per style ----
    for i, style in enumerate(STYLES):
        idx = INFER_ANCHOR_IDX + (i if DISTINCT_INFER_PER_STYLE else 0)
        lines = [clean(anchors[idx]) for _, anchors in classes]    # NO style word
        write_lines(os.path.join(INFER_DIR, f"{style}.txt"), lines)
    span = (f"rows {INFER_ANCHOR_IDX}..{INFER_ANCHOR_IDX + len(STYLES) - 1}"
            if DISTINCT_INFER_PER_STYLE else f"row {INFER_ANCHOR_IDX}")
    print(f"wrote {len(STYLES)} files to {INFER_DIR}  "
          f"({n_classes} neutral prompts each, {span})")

    # ---- show a sample so the format is obvious ----
    s0 = STYLES[0]
    print(f"\nsample -- feature_activation/{s0}.txt (first 2):")
    for l in open(os.path.join(FEATURE_DIR, f"{s0}.txt")).read().splitlines()[:2]:
        print("   ", l)
    print(f"sample -- inference/{s0}.txt (first 2):")
    for l in open(os.path.join(INFER_DIR, f"{s0}.txt")).read().splitlines()[:2]:
        print("   ", l)


if __name__ == "__main__":
    main()

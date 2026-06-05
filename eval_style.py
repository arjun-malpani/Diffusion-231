#!/Users/ananthnamboothiry/Diffusion-231/diffusion231/bin/python
"""
eval_style.py
=============================================================
Van Gogh style scoring for images produced by run_style_steering.py.

Produces TWO independent scores per image (both written every run):

  clip_score : CLIP ViT-B/32 cosine similarity to a Van Gogh reference.
               Reference is --ref_image (CLIP image embedding) if given,
               otherwise the text prompt FALLBACK_TEXT. Range ~[-1, 1].
               Always computed -- needs no checkpoint.

  uc_score   : UnlearnCanvas style classifier P(Van_Gogh).
               This is the SAME model as
                 UnlearnCanvas/diffusion_model_finetuning/evaluation/classification.py
               a ViT-Large (vit_large_patch16_224.augreg_in21k) with a
               len(theme_available)-way head, loaded from style50.pth.
               Score = full-softmax probability of the Van_Gogh class. [0, 1].
               Requires --uc_checkpoint; if absent/unloadable, uc_score is
               left blank (CLIP still runs). We do NOT silently fall back.

Why ViT-Large and not a ResNet50: the official UnlearnCanvas checkpoint is a
ViT-Large state_dict (key "model_state_dict", no class list). There is no
ResNet50 classifier checkpoint, and the classifier expects Resize((224,224))
+ Normalize([0.5],[0.5]) -- not ImageNet crop/normalize.

Checkpoint:
  Download style50.pth from the UnlearnCanvas Google Drive (classifier ckpts)
  and pass --uc_checkpoint checkpoints/unlearncanvas_classifier/style50.pth (this repo's default).

Demo mode (runs automatically when the 3 key images exist in --image_dir):
  Scores the baseline / SAE-steered / prompt-conditioned cat images and saves
  a 3-panel comparison figure to --image_dir/style_eval_demo.png.

Sweep mode (runs when files matching {pid:03d}_{method}_a{alpha}.png exist):
  Scores all sweep images and writes a CSV to --output with both scores.

Expected sweep filename pattern:
  {prompt_id:03d}_{method}_a{alpha}.png   (e.g. 042_sae_a1.5.png)

Usage:
  python eval_style.py --image_dir output_img/ --output results/style_scores.csv
  python eval_style.py --image_dir output_img/ --output results/style_scores.csv \\
      --uc_checkpoint checkpoints/unlearncanvas_classifier/style50.pth
  python eval_style.py --image_dir output_img/ --output results/style_scores.csv \\
      --uc_checkpoint checkpoints/unlearncanvas_classifier/style50.pth --ref_image starry_night.jpg
=============================================================
"""

import os
import re
import csv
import sys
import time
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------ #
_T0 = time.time()
def log(*a):
    print(f"[{time.time() - _T0:6.1f}s]", *a, flush=True)

STYLED_RE = re.compile(r'^(\d{3})_(sae|prompt)_a([\d.]+)\.png$')

DEMO_IMAGES = {
    "baseline":  "cat_baseline.png",
    "sae":       "cat_steered_tau3_s40.png",
    "prompted":  "vangogh_cat_prompted.png",
}
DEMO_LABELS = {
    "baseline": "Neutral baseline\n(no style)",
    "sae":      "SAE-steered\n(tau=3, s=40)",
    "prompted": "Prompt-conditioned\n(cat in Van Gogh style)",
}

FALLBACK_TEXT = "a painting in Van Gogh style, post-impressionist, swirling brushstrokes"

# Canonical UnlearnCanvas style ordering (constants/const.py in the repo).
# The classifier head is sized to len(THEME_AVAILABLE) and the Van_Gogh column
# is its index here -- both MUST match the checkpoint, so do not reorder.
# We prefer importing this from the cloned repo (single source of truth) and
# fall back to this embedded copy if the repo isn't present.
THEME_AVAILABLE_FALLBACK = [
    "Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
    "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
    "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
    "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
    "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
    "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
    "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
    "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter",
]

# ImageNet stats for CLIP path are handled by open_clip's own preprocess; the
# UC classifier uses the [0.5] normalization defined in load_uc_classifier.

# ------------------------------------------------------------------ #

def detect_device(override):
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_theme_available(uc_repo):
    """theme_available from the cloned repo's constants/const.py, else fallback."""
    if uc_repo and os.path.isdir(uc_repo):
        sys.path.insert(0, uc_repo)
        try:
            from constants.const import theme_available
            log(f"theme_available imported from {uc_repo} ({len(theme_available)} classes)")
            return list(theme_available)
        except Exception as e:
            log(f"could not import constants.const from {uc_repo} ({e}); using embedded list")
        finally:
            sys.path.pop(0)
    return list(THEME_AVAILABLE_FALLBACK)


def load_uc_classifier(ckpt_path, theme_available, device):
    """Load the UnlearnCanvas ViT-Large style classifier.

    Mirrors UnlearnCanvas/.../evaluation/classification.py exactly:
      vit_large_patch16_224.augreg_in21k, head = Linear(1024, num_classes),
      weights from ckpt["model_state_dict"].
    Returns (model, vg_idx, transform) or None on any failure.
    """
    try:
        import timm
        import torchvision.transforms as T

        if "Van_Gogh" not in theme_available:
            log("UC classifier: 'Van_Gogh' not in theme_available -- cannot score")
            return None
        vg_idx = theme_available.index("Van_Gogh")
        num_classes = len(theme_available)

        model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
        model.head = torch.nn.Linear(1024, num_classes)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)  # strict: fail loud on architecture mismatch
        model.eval().to(device)

        # Match classification.py: square resize (no aspect-preserving crop) + [0.5] norm.
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        log(f"UC ViT-Large loaded: Van_Gogh class index = {vg_idx} / {num_classes}")
        return model, vg_idx, transform

    except Exception as e:
        log(f"UC classifier load FAILED ({type(e).__name__}: {e}) -- uc_score will be blank")
        return None


def load_clip_model(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.eval().to(device), preprocess, tokenizer


def get_clip_reference(model, preprocess, tokenizer, ref_image_path, device):
    """Returns unit-norm reference embedding [1, D] from image or text."""
    with torch.no_grad():
        if ref_image_path and os.path.isfile(ref_image_path):
            img = preprocess(Image.open(ref_image_path).convert("RGB")).unsqueeze(0).to(device)
            ref = model.encode_image(img)
            log(f"CLIP reference: image '{ref_image_path}'")
        else:
            tokens = tokenizer([FALLBACK_TEXT]).to(device)
            ref = model.encode_text(tokens)
            log(f"CLIP reference: text '{FALLBACK_TEXT}'")
    return (ref / ref.norm(dim=-1, keepdim=True)).float()


def score_clip(paths, ref_emb, model, preprocess, device, batch_size):
    scores = []
    for i in range(0, len(paths), batch_size):
        imgs = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]
        ]).to(device)
        with torch.no_grad():
            embs = model.encode_image(imgs)
        embs = (embs / embs.norm(dim=-1, keepdim=True)).float()
        sims = (embs @ ref_emb.T).squeeze(-1)
        scores.extend(sims.reshape(-1).tolist())
    return scores


def score_uc(paths, model, vg_idx, transform, device, batch_size):
    """Full-softmax P(Van_Gogh) for each image."""
    scores = []
    for i in range(0, len(paths), batch_size):
        imgs = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]
        ]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(imgs), dim=-1)
        scores.extend(probs[:, vg_idx].tolist())
    return scores


def parse_image_dir(image_dir):
    records = []
    for fname in sorted(os.listdir(image_dir)):
        m = STYLED_RE.match(fname)
        if m:
            records.append({
                "path":      os.path.join(image_dir, fname),
                "prompt_id": int(m.group(1)),
                "method":    m.group(2),
                "alpha":     float(m.group(3)),
            })
    return records


def fmt(x):
    return "" if x is None else f"{x:.3f}"


def save_demo_figure(image_dir, images, labels, clip_scores, uc_scores):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax in axes:
        ax.axis("off")
    for ax, img, label, cs, us in zip(axes, images, labels, clip_scores, uc_scores):
        ax.imshow(img)
        uc_txt = f"{us:.3f}" if us is not None else "n/a"
        ax.set_title(f"{label}\nCLIP: {cs:.3f}   |   UC P(VG): {uc_txt}", fontsize=9)
    fig.suptitle("Van Gogh Style: CLIP similarity vs UnlearnCanvas classifier", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(image_dir, "style_eval_demo.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log("saved", path)


def main():
    parser = argparse.ArgumentParser(description="Van Gogh style scorer (CLIP + UnlearnCanvas)")
    parser.add_argument("--image_dir",     required=True,  help="Directory containing generated images")
    parser.add_argument("--output",        required=True,  help="Path for output CSV")
    parser.add_argument("--uc_checkpoint", default="checkpoints/unlearncanvas_classifier/style50.pth",
                        help="UnlearnCanvas ViT-Large checkpoint (style50.pth)")
    parser.add_argument("--uc_repo",       default="UnlearnCanvas/diffusion_model_finetuning",
                        help="Cloned repo dir providing constants/const.py (for theme_available)")
    parser.add_argument("--ref_image",     default=None,   help="Reference Van Gogh image for CLIP")
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--device",        default=None,   help="cuda / mps / cpu (auto if omitted)")
    args = parser.parse_args()

    device = detect_device(args.device)
    log("device:", device)

    # ── CLIP scorer (always available) ─────────────────────────────── #
    log("loading CLIP ViT-B/32 ...")
    clip_model, clip_pre, clip_tok = load_clip_model(device)
    ref_emb = get_clip_reference(clip_model, clip_pre, clip_tok, args.ref_image, device)

    # ── UnlearnCanvas ViT-Large scorer (needs checkpoint) ──────────── #
    uc = None
    if args.uc_checkpoint and os.path.isfile(args.uc_checkpoint):
        theme_available = load_theme_available(args.uc_repo)
        log(f"loading UnlearnCanvas ViT-Large from {args.uc_checkpoint} ...")
        uc = load_uc_classifier(args.uc_checkpoint, theme_available, device)
    else:
        log(f"UC checkpoint not found at '{args.uc_checkpoint}' -- uc_score will be blank "
            f"(download style50.pth and pass --uc_checkpoint)")

    def score_both(paths):
        clip_scores = score_clip(paths, ref_emb, clip_model, clip_pre, device, args.batch_size)
        if uc is not None:
            model_uc, vg_idx, transform = uc
            uc_scores = score_uc(paths, model_uc, vg_idx, transform, device, args.batch_size)
        else:
            uc_scores = [None] * len(paths)
        return clip_scores, uc_scores

    # ── Demo figure ────────────────────────────────────────────────── #
    demo_paths = {k: os.path.join(args.image_dir, v) for k, v in DEMO_IMAGES.items()}
    if all(os.path.isfile(p) for p in demo_paths.values()):
        log("running demo figure ...")
        keys = list(DEMO_IMAGES)
        demo_pil = [Image.open(demo_paths[k]).convert("RGB") for k in keys]
        paths_list = [demo_paths[k] for k in keys]
        clip_scores, uc_scores = score_both(paths_list)
        labels = [DEMO_LABELS[k] for k in keys]
        for label, cs, us in zip(labels, clip_scores, uc_scores):
            log(f"  {label.split(chr(10))[0]}: CLIP={cs:.4f}  UC={fmt(us) or 'n/a'}")
        save_demo_figure(args.image_dir, demo_pil, labels, clip_scores, uc_scores)
    else:
        missing = [v for k, v in DEMO_IMAGES.items() if not os.path.isfile(demo_paths[k])]
        log(f"demo skipped -- missing: {missing} (run run_style_steering.py first)")

    # ── Sweep CSV ──────────────────────────────────────────────────── #
    records = parse_image_dir(args.image_dir)
    if records:
        log(f"scoring {len(records)} sweep images ...")
        paths_list = [r["path"] for r in records]
        clip_scores, uc_scores = score_both(paths_list)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt_id", "method", "alpha", "clip_score", "uc_score"]
            )
            writer.writeheader()
            for record, cs, us in zip(records, clip_scores, uc_scores):
                writer.writerow({
                    "prompt_id":  record["prompt_id"],
                    "method":     record["method"],
                    "alpha":      record["alpha"],
                    "clip_score": f"{cs:.6f}",
                    "uc_score":   "" if us is None else f"{us:.6f}",
                })
        log(f"wrote {len(records)} rows to {args.output} "
            f"(uc_score {'populated' if uc is not None else 'BLANK -- no checkpoint'})")
    else:
        log("no sweep images found (expected pattern: 042_sae_a1.5.png)")

    if not all(os.path.isfile(p) for p in demo_paths.values()) and not records:
        log("nothing to score -- place images in --image_dir and re-run")


if __name__ == "__main__":
    main()

#!/Users/ananthnamboothiry/Diffusion-231/diffusion231/bin/python
"""
eval_style.py
=============================================================
Van Gogh style score for images produced by run_style_steering.py.

Scoring strategy (tried in order):
  1. UnlearnCanvas style classifier (ResNet50).  Requires a checkpoint:
       --uc_checkpoint /path/to/uc_style_classifier.pth
     Download instructions:
       * Clone the UnlearnCanvas repo:
           git clone https://github.com/OPTML-Group/UnlearnCanvas
       * Download the style-classifier checkpoint from the project page
         or Hugging Face (model card: OPTML-Group/UnlearnCanvas).
     The checkpoint must contain a class list that includes a class
     containing "van" or "gogh" (case-insensitive).

  2. CLIP ViT-B/32 cosine similarity (fallback -- no checkpoint needed).
     If --ref_image is provided, embeds that image via the CLIP image
     encoder.  Otherwise embeds the text:
       "a painting in Van Gogh style, post-impressionist, swirling brushstrokes"

Demo mode (runs automatically when the 3 key images exist in --image_dir):
  Scores cat_baseline.png, cat_steered_strong.png, vangogh_cat_prompted.png
  and saves a 3-panel comparison figure to --image_dir/style_eval_demo.png.
  Run run_style_steering.py first to generate these images.

Sweep mode (runs when files matching {pid:03d}_{method}_a{alpha}.png exist):
  Scores all sweep images and writes a CSV to --output.

Expected sweep filename pattern:
  {prompt_id:03d}_{method}_a{alpha}.png   (e.g. 042_sae_a1.5.png)

Usage:
  python eval_style.py --image_dir output_img/ --output results/style_scores.csv
  python eval_style.py --image_dir output_img/ --output results/style_scores.csv \\
      --uc_checkpoint /path/to/uc_style_classifier.pth
  python eval_style.py --image_dir output_img/ --output results/style_scores.csv \\
      --ref_image /path/to/starry_night.jpg --batch_size 16
=============================================================
"""

import os
import re
import csv
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

# ------------------------------------------------------------------ #

def detect_device(override):
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_uc_classifier(ckpt_path, device):
    """Load UnlearnCanvas ResNet50 style classifier. Returns (model, vg_idx, transform) or None."""
    try:
        import torchvision.models as tvm
        import torchvision.transforms as T

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        class_names = None
        for key in ("class_names", "classes", "labels"):
            if key in ckpt:
                class_names = ckpt[key]
                break
        if class_names is None:
            log(f"UC classifier: no class list found in checkpoint (tried class_names/classes/labels)")
            return None

        vg_idx = None
        for i, c in enumerate(class_names):
            if "van" in c.lower() or "gogh" in c.lower():
                vg_idx = i
                break
        if vg_idx is None:
            log(f"UC classifier: Van Gogh not found in class list: {class_names[:10]}")
            return None

        model = tvm.resnet50(weights=None)
        model.fc = torch.nn.Linear(2048, len(class_names))

        state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval().to(device)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)
        ])
        log(f"UC classifier loaded: Van Gogh class index = {vg_idx} ('{class_names[vg_idx]}')")
        return model, vg_idx, transform

    except Exception as e:
        log(f"UC classifier load failed ({e}), falling back to CLIP")
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
        else:
            tokens = tokenizer([FALLBACK_TEXT]).to(device)
            ref = model.encode_text(tokens)
    return (ref / ref.norm(dim=-1, keepdim=True)).float()


def score_with_uc(paths, model, vg_idx, transform, device, batch_size):
    scores = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(imgs), dim=-1)
        scores.extend(probs[:, vg_idx].tolist())
    return scores


def score_with_clip(paths, ref_emb, model, preprocess, device, batch_size):
    scores = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            embs = model.encode_image(imgs)
        embs = (embs / embs.norm(dim=-1, keepdim=True)).float()
        sims = (embs @ ref_emb.T).squeeze(-1)
        scores.extend(sims.tolist())
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


def save_demo_figure(image_dir, images, labels, scores):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax in axes:
        ax.axis("off")
    for ax, img, label, score in zip(axes, images, labels, scores):
        ax.imshow(img)
        ax.set_title(f"{label}\nStyle score: {score:.3f}", fontsize=9)
    fig.suptitle("Van Gogh Style Score: Prompt vs SAE Steering", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(image_dir, "style_eval_demo.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log("saved", path)


def main():
    parser = argparse.ArgumentParser(description="Van Gogh style scorer")
    parser.add_argument("--image_dir",     required=True,       help="Directory containing generated images")
    parser.add_argument("--output",        required=True,       help="Path for output CSV")
    parser.add_argument("--uc_checkpoint", default=None,        help="UnlearnCanvas ResNet50 checkpoint path")
    parser.add_argument("--ref_image",     default=None,        help="Reference Van Gogh image for CLIP fallback")
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--device",        default=None,        help="cuda / mps / cpu (auto-detects if omitted)")
    args = parser.parse_args()

    device = detect_device(args.device)
    log("device:", device)

    uc_result = load_uc_classifier(args.uc_checkpoint, device) if args.uc_checkpoint else None
    scorer_tag = "uc" if uc_result else "clip"

    clip_model = clip_pre = clip_tok = ref_emb = None
    if not uc_result:
        log("loading CLIP ViT-B/32 ...")
        clip_model, clip_pre, clip_tok = load_clip_model(device)
        ref_emb = get_clip_reference(clip_model, clip_pre, clip_tok, args.ref_image, device)
        log("CLIP reference embedding ready")

    # ── Demo figure ────────────────────────────────────────────────── #
    demo_paths = {k: os.path.join(args.image_dir, v) for k, v in DEMO_IMAGES.items()}
    if all(os.path.isfile(p) for p in demo_paths.values()):
        log("running demo figure ...")
        demo_pil = [Image.open(demo_paths[k]).convert("RGB") for k in DEMO_IMAGES]
        paths_list = [demo_paths[k] for k in DEMO_IMAGES]
        if uc_result:
            model_uc, vg_idx, transform = uc_result
            demo_scores = score_with_uc(paths_list, model_uc, vg_idx, transform, device, args.batch_size)
        else:
            demo_scores = score_with_clip(paths_list, ref_emb, clip_model, clip_pre, device, args.batch_size)
        labels = [DEMO_LABELS[k] for k in DEMO_IMAGES]
        for label, score in zip(labels, demo_scores):
            log(f"  {label.split(chr(10))[0]}: {score:.4f}")
        save_demo_figure(args.image_dir, demo_pil, labels, demo_scores)
    else:
        missing = [v for k, v in DEMO_IMAGES.items() if not os.path.isfile(demo_paths[k])]
        log(f"demo skipped — missing: {missing} (run run_style_steering.py first)")

    # ── Sweep CSV ──────────────────────────────────────────────────── #
    records = parse_image_dir(args.image_dir)
    if records:
        log(f"scoring {len(records)} sweep images ...")
        paths_list = [r["path"] for r in records]
        if uc_result:
            model_uc, vg_idx, transform = uc_result
            raw_scores = score_with_uc(paths_list, model_uc, vg_idx, transform, device, args.batch_size)
        else:
            raw_scores = score_with_clip(paths_list, ref_emb, clip_model, clip_pre, device, args.batch_size)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt_id", "method", "alpha", "score", "scorer"])
            writer.writeheader()
            for record, score in zip(records, raw_scores):
                writer.writerow({
                    "prompt_id": record["prompt_id"],
                    "method":    record["method"],
                    "alpha":     record["alpha"],
                    "score":     f"{score:.6f}",
                    "scorer":    scorer_tag,
                })
        log(f"wrote {len(records)} rows to {args.output}")
    else:
        log("no sweep images found (expected pattern: 042_sae_a1.5.png)")

    if not any(os.path.isfile(p) for p in demo_paths.values()) and not records:
        log("nothing to score — place images in --image_dir and re-run")


if __name__ == "__main__":
    main()

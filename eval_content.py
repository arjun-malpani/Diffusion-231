#!/Users/ananthnamboothiry/Diffusion-231/diffusion231/bin/python
"""
eval_content.py
=============================================================
CLIP ViT-B/32 content preservation score for images produced by the
style steering pipeline.

Measures cosine similarity between each styled image and the neutral
baseline (same prompt, no style applied).  Higher = more content preserved.

Demo mode (runs automatically when the 3 key images exist in --image_dir):
  Scores cat_baseline.png, cat_steered_strong.png, vangogh_cat_prompted.png
  against cat_baseline.png as the content reference, then saves a 3-panel
  comparison figure to --image_dir/content_eval_demo.png.
  Run run_style_steering.py first to generate these images.

Sweep mode (runs when files matching {pid:03d}_{method}_a{alpha}.png AND
{pid:03d}_baseline.png exist):
  Pre-embeds all baselines once, then scores all styled images against their
  per-prompt baseline. Writes CSV to --output.

Expected sweep filename patterns:
  Styled:   {prompt_id:03d}_{method}_a{alpha}.png  (e.g. 042_sae_a1.5.png)
  Baseline: {prompt_id:03d}_baseline.png            (e.g. 042_baseline.png)

Usage:
  python eval_content.py --image_dir output_img/ --output results/content_scores.csv
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

STYLED_RE   = re.compile(r'^(\d{3})_(sae|prompt)_a([\d.]+)\.png$')
BASELINE_RE = re.compile(r'^(\d{3})_baseline\.png$')

DEMO_IMAGES = {
    "baseline": "cat_baseline.png",
    "sae":      "cat_steered_tau3_s40.png",
    "prompted": "vangogh_cat_prompted.png",
}
DEMO_LABELS = {
    "baseline": "Neutral baseline\n(reference)",
    "sae":      "SAE-steered\n(tau=3, s=40)",
    "prompted": "Prompt-conditioned\n(cat in Van Gogh style)",
}

# ------------------------------------------------------------------ #

def detect_device(override):
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_model(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model.eval().to(device), preprocess


def embed_images(paths, model, preprocess, device, batch_size):
    """Embed a list of image paths -> unit-norm tensors on CPU, one per path."""
    embs = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]
        ]).to(device)
        with torch.no_grad():
            e = model.encode_image(batch)
        e = (e / e.norm(dim=-1, keepdim=True)).float().cpu()
        embs.append(e)
    return torch.cat(embs, dim=0)   # [N, D]


def parse_image_dir(image_dir):
    styled_records, baseline_map = [], {}
    for fname in sorted(os.listdir(image_dir)):
        m = STYLED_RE.match(fname)
        if m:
            styled_records.append({
                "path":      os.path.join(image_dir, fname),
                "prompt_id": int(m.group(1)),
                "method":    m.group(2),
                "alpha":     float(m.group(3)),
            })
            continue
        m = BASELINE_RE.match(fname)
        if m:
            baseline_map[int(m.group(1))] = os.path.join(image_dir, fname)
    return styled_records, baseline_map


def save_demo_figure(image_dir, images, labels, scores):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax in axes:
        ax.axis("off")
    for ax, img, label, score in zip(axes, images, labels, scores):
        ax.imshow(img)
        ax.set_title(f"{label}\nContent sim: {score:.3f}", fontsize=9)
    fig.suptitle("Content Preservation: Prompt vs SAE Steering", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(image_dir, "content_eval_demo.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log("saved", path)


def main():
    parser = argparse.ArgumentParser(description="CLIP content preservation scorer")
    parser.add_argument("--image_dir",  required=True,       help="Directory containing generated images")
    parser.add_argument("--output",     required=True,       help="Path for output CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     default=None,        help="cuda / mps / cpu (auto-detects if omitted)")
    args = parser.parse_args()

    device = detect_device(args.device)
    log("device:", device)
    log("loading CLIP ViT-B/32 ...")
    clip_model, clip_pre = load_clip_model(device)
    log("CLIP ready")

    # ── Demo figure ────────────────────────────────────────────────── #
    demo_paths = {k: os.path.join(args.image_dir, v) for k, v in DEMO_IMAGES.items()}
    if all(os.path.isfile(p) for p in demo_paths.values()):
        log("running demo figure ...")
        demo_pil  = [Image.open(demo_paths[k]).convert("RGB") for k in DEMO_IMAGES]
        paths_list = [demo_paths[k] for k in DEMO_IMAGES]
        embs = embed_images(paths_list, clip_model, clip_pre, device, args.batch_size)
        baseline_emb = embs[0:1]          # [1, D] — the neutral reference
        sims = (embs @ baseline_emb.T).squeeze(-1).tolist()   # [3]
        labels = [DEMO_LABELS[k] for k in DEMO_IMAGES]
        for label, sim in zip(labels, sims):
            log(f"  {label.split(chr(10))[0]}: {sim:.4f}")
        save_demo_figure(args.image_dir, demo_pil, labels, sims)
    else:
        missing = [v for k, v in DEMO_IMAGES.items() if not os.path.isfile(demo_paths[k])]
        log(f"demo skipped — missing: {missing} (run run_style_steering.py first)")

    # ── Sweep CSV ──────────────────────────────────────────────────── #
    styled_records, baseline_map = parse_image_dir(args.image_dir)
    if not styled_records:
        log("no sweep images found (expected pattern: 042_sae_a1.5.png)")
        return
    if not baseline_map:
        log("ERROR: sweep images found but no baseline files (expected pattern: 042_baseline.png)")
        raise SystemExit(1)

    needed_pids = {r["prompt_id"] for r in styled_records}
    log(f"embedding {len(baseline_map)} baselines ...")
    baseline_paths = [baseline_map[pid] for pid in sorted(baseline_map) if pid in needed_pids]
    baseline_pids  = [pid for pid in sorted(baseline_map) if pid in needed_pids]
    baseline_embs_tensor = embed_images(baseline_paths, clip_model, clip_pre, device, args.batch_size)
    baseline_embs = {pid: baseline_embs_tensor[i] for i, pid in enumerate(baseline_pids)}

    log(f"scoring {len(styled_records)} sweep images ...")
    results = []
    skipped_pids = set()
    for i in range(0, len(styled_records), args.batch_size):
        batch = styled_records[i:i + args.batch_size]
        valid = [r for r in batch if r["prompt_id"] in baseline_embs]
        for r in batch:
            if r["prompt_id"] not in baseline_embs and r["prompt_id"] not in skipped_pids:
                log(f"  WARNING: no baseline for prompt_id={r['prompt_id']}, skipping")
                skipped_pids.add(r["prompt_id"])
        if not valid:
            continue
        paths_list = [r["path"] for r in valid]
        embs = embed_images(paths_list, clip_model, clip_pre, device, args.batch_size)
        for r, emb in zip(valid, embs):
            ref = baseline_embs[r["prompt_id"]].to(emb.device)
            sim = float(torch.dot(emb, ref))
            results.append({**r, "score": sim})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "method", "alpha", "score"])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "prompt_id": row["prompt_id"],
                "method":    row["method"],
                "alpha":     row["alpha"],
                "score":     f"{row['score']:.6f}",
            })
    log(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()

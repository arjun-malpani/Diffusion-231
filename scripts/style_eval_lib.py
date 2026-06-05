"""
style_eval_lib.py
=============================================================
Shared library for the SAE style-steering evaluation pipeline.

For each (style, prompt) at a fixed seed (default 42 = the training seed) the
pipeline generates a set of CONDITIONS and scores each generated image with
FOUR metrics.

Conditions (see build_conditions):
  unstyled  : generate(prompt)                          -- the "original image"
                                                          (weight 0 anchor / content ref)
  prompted  : generate(f"{prompt} in {style} style.")   -- text-conditioned style reference
  injected  : SAE 'uniform'         injection (one vector / patch)       at weight w   (steering.steer)
  entangled : SAE 'patch_selective' injection (per-patch feature set)    at weight w

  -> "injected" vs "entangled" vs "unstyled" is exactly the grid the user asked for;
     'uniform' adds the same style vector to every patch, 'patch_selective' picks a
     different top-K feature set per patch (see steering/steer.py:compute_delta).

Scores (see Scorer.score_images):
  prompt_clip   : CLIP cos(image, original prompt text)      -- prompt fidelity
  style_clip    : CLIP cos(image, "<style> style" text)      -- style alignment (CLIP)
  uc_style      : UnlearnCanvas ViT-L  P(style)              -- style alignment (classifier)
  content_clip  : CLIP cos(image, unstyled baseline image)   -- similarity to the unstyled
                  baseline (drift from the un-steered image; NOT prompt fidelity -- that is
                  prompt_clip = CLIP cos(image, prompt). unstyled scores 1.0 against itself.)

The steering math is reused verbatim from steering/steer.py. The model loaders
mirror eval_style.py (CLIP ViT-B/32 via open_clip + UnlearnCanvas ViT-Large from
style50.pth); theme_available is imported from eval_style so there is a single
source of truth for the 51-way class ordering.

This module is import-only (no side effects beyond matplotlib's Agg backend via
eval_style). Both run_style_eval.py and run_single.py drive it.
=============================================================
"""

import os
import sys
import csv
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from PIL import Image

# repo root on path so diffusion_sae / steering / eval_style import cleanly
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ----------------------------- defaults ----------------------------- #
DEFAULT_WEIGHTS    = [1.0, 2.0]
DEFAULT_METHODS    = ["uniform", "patch_selective"]
DEFAULT_SEED       = 42                       # matches the activation-collection / training seed
DEFAULT_UC_CKPT    = os.path.join(REPO, "checkpoints/unlearncanvas_classifier/style50.pth")
DEFAULT_UC_REPO    = os.path.join(REPO, "UnlearnCanvas/diffusion_model_finetuning")
DEFAULT_RESULTS    = os.path.join(REPO, "results/style_eval")
METHOD_LABEL       = {"uniform": "injected_normal", "patch_selective": "injected_patches"}   # display names

METRICS = ["prompt_clip", "style_clip", "uc_style", "content_clip"]
METRIC_LABELS = {
    "prompt_clip":  "Prompt fidelity\nCLIP(img, prompt)",
    "style_clip":   "Style align (CLIP)\nCLIP(img, \"<style> style\")",
    "uc_style":     "Style align (classifier)\nUnlearnCanvas P(style)",
    "content_clip": "Similarity to unstyled (CLIP)\nCLIP(img, unstyled)",
}
# +1 = higher is "more style", -1 = higher is "more content/fidelity" (for reading trade-offs)
METRIC_IS_STYLE = {"prompt_clip": False, "style_clip": True, "uc_style": True, "content_clip": False}

CSV_FIELDS = ["style", "prompt_id", "prompt", "condition", "kind", "method", "weight",
              "prompt_clip", "style_clip", "uc_style", "content_clip", "img_path"]

_T0 = time.time()
def log(*a):
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


def detect_device(override=None):
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------- text helpers --------------------------- #
def style_display(style: str) -> str:
    """'Van_Gogh' -> 'Van Gogh' for natural-language prompts/labels."""
    return style.replace("_", " ")


def stylized_prompt(prompt: str, style: str) -> str:
    """The text-conditioned style reference: original prompt + ' in <style> style.'"""
    return f"{prompt} in {style_display(style)} style."


def style_clip_text(style: str) -> str:
    """Reference text for the style_clip score (user choice: terse '<style> style')."""
    return f"{style_display(style)} style"


# --------------------------- conditions ----------------------------- #
@dataclass(frozen=True)
class Condition:
    name: str                 # e.g. "unstyled", "prompted", "injected_w1", "entangled_w2"
    kind: str                 # "unstyled" | "prompted" | "sae"
    method: Optional[str]     # "uniform" | "patch_selective" | None
    weight: float             # 0.0 for unstyled/prompted


def build_conditions(weights=None, methods=None, include_prompted=True):
    """Ordered list of conditions. 'unstyled' is always first (it is the content baseline)."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    methods = DEFAULT_METHODS if methods is None else methods
    conds = [Condition("unstyled", "unstyled", None, 0.0)]
    if include_prompted:
        conds.append(Condition("prompted", "prompted", None, 0.0))
    for w in weights:
        for m in methods:
            lab = METHOD_LABEL.get(m, m)
            conds.append(Condition(f"{lab}_w{w:g}", "sae", m, float(w)))
    return conds


# --------------------------- generation ----------------------------- #
class Generator:
    """Loads SD-1.4 + the trained SAE + per-style features once, then generates any condition.

    Thin wrapper over steering/steer.py so the injection math is identical to run_experiment.py.
    """

    def __init__(self, device=None, seed=DEFAULT_SEED, features_pt=None, sae_ckpt=None):
        from diffusion_sae.model import import_model
        from diffusion_sae.config import MODEL_ID
        from steering.common import load_sae, FEATURES_PT, SAE_CKPT
        from steering.steer import set_pipe

        self.device = detect_device(device)
        self.seed = seed
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        sae_ckpt = sae_ckpt or SAE_CKPT
        if not os.path.isfile(sae_ckpt):
            raise FileNotFoundError(f"SAE checkpoint not found: {sae_ckpt}")
        feats_path = features_pt or FEATURES_PT
        if not os.path.isfile(feats_path):
            raise FileNotFoundError(f"features.pt not found: {feats_path} (run steering/identify_directions.py)")

        log(f"loading SD-1.4 ({MODEL_ID}) on {self.device} / {dtype} ...")
        self.pipe = import_model(MODEL_ID, dtype=dtype, device=self.device)
        self.pipe.set_progress_bar_config(disable=True)
        set_pipe(self.pipe)                                  # fp32 VAE -> avoids fp16 NaNs

        log("loading SAE + per-style features ...")
        self.sae = load_sae(self.device)
        self.features = torch.load(feats_path, map_location=self.device, weights_only=False)
        log(f"generator ready. styles with features: {list(self.features.keys())}")

    def available_styles(self):
        return list(self.features.keys())

    def generate(self, prompt, style, cond: Condition, seed=None):
        """Return a PIL image for one condition."""
        from steering.steer import generate_baseline, generate_steered
        seed = self.seed if seed is None else seed
        if cond.kind == "unstyled":
            return generate_baseline(self.pipe, prompt, seed)
        if cond.kind == "prompted":
            return generate_baseline(self.pipe, stylized_prompt(prompt, style), seed)
        if cond.kind == "sae":
            if style not in self.features:
                raise KeyError(f"no SAE features for style '{style}' (have {self.available_styles()})")
            return generate_steered(self.pipe, self.sae, prompt, self.features[style],
                                    cond.method, cond.weight, seed)
        raise ValueError(f"unknown condition kind: {cond.kind!r}")


# ----------------------------- scoring ------------------------------ #
class Scorer:
    """CLIP ViT-B/32 (always) + optional UnlearnCanvas ViT-Large classifier.

    Mirrors eval_style.py's loaders. UC is multi-style here: we keep the full
    softmax and index the requested style's column (not just Van Gogh).
    """

    def __init__(self, device=None, uc_checkpoint=DEFAULT_UC_CKPT, uc_repo=DEFAULT_UC_REPO,
                 batch_size=16):
        self.device = detect_device(device)
        self.batch_size = batch_size

        import open_clip
        log("loading CLIP ViT-B/32 ...")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.clip = model.eval().to(self.device)
        self.clip_pre = preprocess
        self.clip_tok = open_clip.get_tokenizer("ViT-B-32")

        self.uc = None
        self.theme = None
        self.uc_tf = None
        if uc_checkpoint and os.path.isfile(uc_checkpoint):
            self._load_uc(uc_checkpoint, uc_repo)
        else:
            log(f"UC checkpoint not found at '{uc_checkpoint}' -- uc_style will be blank "
                f"(download style50.pth to enable the classifier metric)")

    def _load_uc(self, ckpt_path, uc_repo):
        try:
            import timm
            import torchvision.transforms as T
            from eval_style import load_theme_available     # single source of truth for the 51 labels

            theme = load_theme_available(uc_repo)
            model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
            model.head = torch.nn.Linear(1024, len(theme))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state)                    # strict: fail loud on mismatch
            self.uc = model.eval().to(self.device)
            self.theme = theme
            self.uc_tf = T.Compose([
                T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5], [0.5]),
            ])
            log(f"UC ViT-Large loaded ({len(theme)}-way head)")
        except Exception as e:
            log(f"UC load FAILED ({type(e).__name__}: {e}) -- uc_style will be blank")
            self.uc = None

    def has_uc(self):
        return self.uc is not None

    # ---- embedding helpers ---- #
    def _clip_image_emb(self, pil_images):
        embs = []
        for i in range(0, len(pil_images), self.batch_size):
            batch = torch.stack([
                self.clip_pre(im.convert("RGB")) for im in pil_images[i:i + self.batch_size]
            ]).to(self.device)
            with torch.no_grad():
                e = self.clip.encode_image(batch)
            embs.append((e / e.norm(dim=-1, keepdim=True)).float().cpu())
        return torch.cat(embs)                              # [N, D] unit-norm

    def _clip_text_emb(self, texts):
        with torch.no_grad():
            tok = self.clip_tok(texts).to(self.device)
            e = self.clip.encode_text(tok)
        return (e / e.norm(dim=-1, keepdim=True)).float().cpu()

    def _uc_probs(self, pil_images):
        out = []
        for i in range(0, len(pil_images), self.batch_size):
            batch = torch.stack([
                self.uc_tf(im.convert("RGB")) for im in pil_images[i:i + self.batch_size]
            ]).to(self.device)
            with torch.no_grad():
                p = torch.softmax(self.uc(batch), dim=-1)
            out.append(p.float().cpu())
        return torch.cat(out)                               # [N, C]

    def score_images(self, images, prompt, style):
        """Score a list of PIL images that share the SAME prompt/style.

        images[0] is assumed to be the unstyled baseline (content reference); content_clip
        is cosine similarity to it (so the unstyled image scores 1.0 against itself).
        Returns four lists aligned to `images`: prompt_clip, style_clip, uc_style, content_clip.
        uc_style entries are None when the classifier is unavailable.
        """
        img_emb = self._clip_image_emb(images)              # [N, D]
        base_emb = img_emb[0]                               # unstyled = images[0]
        prompt_emb = self._clip_text_emb([prompt])[0]
        style_emb = self._clip_text_emb([style_clip_text(style)])[0]

        prompt_clip = (img_emb @ prompt_emb).tolist()
        style_clip = (img_emb @ style_emb).tolist()
        content_clip = (img_emb @ base_emb).tolist()

        if self.uc is not None and style in (self.theme or []):
            idx = self.theme.index(style)
            uc_style = self._uc_probs(images)[:, idx].tolist()
        else:
            uc_style = [None] * len(images)
        return prompt_clip, style_clip, uc_style, content_clip


# ------------------------------- IO --------------------------------- #
def read_prompts(path, limit=None):
    """Non-empty lines from a prompt file, optionally truncated to `limit`."""
    with open(path) as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    return prompts[:limit] if limit else prompts


def load_scores(csv_path):
    """Parse scores.csv -> list of row dicts with numeric fields cast (uc_style may be None)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["prompt_id"] = int(r["prompt_id"])
            r["weight"] = float(r["weight"])
            for k in ("prompt_clip", "style_clip", "content_clip"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            r["uc_style"] = float(r["uc_style"]) if r["uc_style"] not in ("", None) else None
            rows.append(r)
    return rows


def write_scores(csv_path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in CSV_FIELDS}
            for k in ("prompt_clip", "style_clip", "uc_style", "content_clip"):
                out[k] = "" if out[k] in ("", None) else f"{float(out[k]):.6f}"
            w.writerow(out)


def merge_scores(csv_path, rows):
    """Write `rows` to csv_path, MERGING with any existing cache by (style, prompt_id, condition).

    New rows overwrite matching cached ones; rows from prior runs that aren't in this batch are
    preserved. This makes scores.csv an accumulating cache across separate invocations into the
    same results dir, so large-scale trends can be visualized over everything ever run there.
    Returns the merged row list.
    """
    merged = {}
    if os.path.isfile(csv_path):
        for r in load_scores(csv_path):
            merged[(r["style"], r["prompt_id"], r["condition"])] = r
    for r in rows:
        merged[(r["style"], int(r["prompt_id"]), r["condition"])] = r
    write_scores(csv_path, list(merged.values()))
    return list(merged.values())


def existing_keys(csv_path):
    """(style, prompt_id, condition) keys already scored -- for resumable runs."""
    if not os.path.isfile(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            keys.add((r["style"], int(r["prompt_id"]), r["condition"]))
    return keys


# --------------------------- orchestration -------------------------- #
def run_pipeline(prompts_by_style, conditions, scorer, results_dir,
                 generator=None, seed=DEFAULT_SEED, skip_existing=True, save_images=False):
    """Generate (or reuse) all condition images per prompt, score them, write scores.csv.

    prompts_by_style : {style: [prompt, ...]}
    generator        : a Generator, or None to score pre-existing images only (needs save_images
                       artifacts on disk).
    save_images      : if True, persist each condition image under results/images/<style>/<pid>/.
                       Default False -- images are kept in memory only long enough to score and
                       compose the grid figure (the grid is the deliverable, not the loose files).
                       Saving also enables resume (skip_existing) and re-plotting via --figures-only.

    Returns (rows, csv_path, images) where images[(style, prompt_id, condition_name)] = PIL.Image
    for every image generated/loaded this run (used to build the grid without round-tripping disk).
    """
    images_dir = os.path.join(results_dir, "images")
    csv_path = os.path.join(results_dir, "scores.csv")
    os.makedirs(results_dir, exist_ok=True)
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    reuse = skip_existing and save_images       # can only resume from images we actually saved

    rows = []
    images = {}
    n_styles = len(prompts_by_style)

    for si, (style, prompts) in enumerate(prompts_by_style.items(), 1):
        log(f"[style {si}/{n_styles}] {style}: {len(prompts)} prompts x {len(conditions)} conditions")
        for pid, prompt in enumerate(prompts):
            imgs = []
            for cond in conditions:
                ipath = os.path.join(images_dir, style, f"{pid:03d}", f"{cond.name}.png")
                if reuse and os.path.exists(ipath):
                    img = Image.open(ipath).convert("RGB")
                else:
                    if generator is None:
                        raise FileNotFoundError(
                            f"no image and no generator (scores-only mode needs --save-images): {ipath}")
                    img = generator.generate(prompt, style, cond, seed)
                    if save_images:
                        os.makedirs(os.path.dirname(ipath), exist_ok=True)
                        img.save(ipath)
                imgs.append((cond, img, ipath if save_images else ""))
                images[(style, pid, cond.name)] = img       # in-memory store for the grid

            images_list = [t[1] for t in imgs]              # imgs[0] is 'unstyled' (conditions[0])
            pc, sc, uc, cc = scorer.score_images(images_list, prompt, style)
            for (cond, _img, ipath), a, b, c, d in zip(imgs, pc, sc, uc, cc):
                rows.append({
                    "style": style, "prompt_id": pid, "prompt": prompt,
                    "condition": cond.name, "kind": cond.kind,
                    "method": cond.method or "", "weight": cond.weight,
                    "prompt_clip": a, "style_clip": b, "uc_style": c, "content_clip": d,
                    "img_path": ipath,
                })

        merge_scores(csv_path, rows)                        # accumulating cache, flushed per style
        log(f"  cached {len(rows)} rows this run -> {csv_path}")

    log(f"pipeline done: {len(rows)} rows for {n_styles} styles -> {csv_path}")
    return rows, csv_path, images

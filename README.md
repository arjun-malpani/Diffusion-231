# Diffusion-231


## Setup

- Activate the env from requirements.txt 
- Run everything from the repo root.
- Styles, paths, hyperparameters in `diffusion_sae/config.py`.

## Pipeline

Run in order, ~few hours on GPU 

- Collect activations — `python scripts/collect_activations.py`
  Runs SD over the styled prompts and caches `up.1.2` activations to `data/activations/style_v1/`.

- Train SAE — `python scripts/train_sae.py`
  Trains on the cached activations. Writes `checkpoints/sae_v1/sae_final.pt`.

- Build prompt sets (optional, set of promtps included in the repo) 
 `python scripts/make_prompts.py`
  Writes `data/prompts/feature_activation/` (styled, for finding features) and
  `data/prompts/inference/` (neutral, for steering).

- Identify style-specific directions — `python steering/identify_directions.py`
  Finds the SAE features each style adds (neutral vs. styled). Writes `steering/features.pt`.

- Steering demo — `python steering/run_experiment.py` then `python steering/visualize.py`
  Injects style directions into neutral prompts (uniform and patch-selective). Images and comparison in `steering/output/`.

- Eval — `python scripts/run_style_eval.py`
  Gen the eval conditions per style/prompt, scores with CLIP (plus the UnlearnCanvas
  classifier if `checkpoints/unlearncanvas_classifier/style50.pth` is present), and writes
  `scores.csv` + figures to `results/style_eval/`.
  - quick check: add `--scope smoke` (1 prompt/style) or `--scope limited-prompts` (5/style)
  - single example: `python scripts/run_single.py --prompt "a cat" --style Van_Gogh`

## Notes

- Developed on an A10G

import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import math

from tqdm.auto import tqdm

from .config import (DEVICE, EPOCHS, AUX_ALPHA, LEARNING_RATE, DEAD_THRESHOLD,
                     STYLES, FEATURE_ACT_DIR, INFERENCE_DIR)

from .sae import SAE

from .model import generate, get_activation


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #root
_PROMPT_DIRS = {"feature_activation": FEATURE_ACT_DIR, "inference": INFERENCE_DIR}


def load_style_prompts(kind="feature_activation", styles=STYLES):
    #Maps each style to its prompt list. returns like {style: [prompt, ...]} 
    """
      "feature_activation":  styled prompts used to identify each style's features
      "inference": neutral content prompts used for steering
    """
    if kind not in _PROMPT_DIRS:
        raise ValueError(f"kind must be one of {list(_PROMPT_DIRS)} but got {kind!r}")

    base = os.path.join(_REPO, _PROMPT_DIRS[kind])

    prompts = {}
    for style in styles:
        path = os.path.join(base, f"{style}.txt")
        with open(path) as f:
            prompts[style] = [line.strip() for line in f if line.strip()]
    return prompts






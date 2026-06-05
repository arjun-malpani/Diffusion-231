"""Script-level config.

STYLES / SEED / ANCHOR_FILE come from the canonical library config
(diffusion_sae.config) so there is ONE source of truth -- editing the style list
in diffusion_sae/config.py updates collection, prompt-building, and vocab alike.
Only run-specific knobs (the collection output dir) live here.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_sae.config import STYLES, SEED, ANCHOR_FILE   # re-export

OUT_DIR = "data/activations/style_v1"   # where collect_activations.py writes shards

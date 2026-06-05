"""
Inject SAE style directions into SD generation at up.1.2.

Two methods, both built from a style's selected features {idx, mu, maps}:
  uniform : add the SAME vector to every patch  -- delta[:,i,j] = strength * Sum_f mu[f]  * W_dec[f]
  patch   : per-patch via styled spatial maps   -- delta[:,i,j] = strength * Sum_f map[f,i,j]* W_dec[f]

Note mu[f] == map[f].mean(patches), so 'uniform' is exactly the spatially-flattened 'patch' --
both inject the same TOTAL style; they differ only in spatial distribution. W_dec rows are unit-norm,
and mu is the feature's styled activation level, so strength=1.0 injects the feature at ~the level the
real style produces (Eq.7-style calibration). Injection is on the conditional CFG half only, every step.

Purely functional: compute_delta -> make_injection_hook -> style_injection (register/remove).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import contextmanager

import torch
from PIL import Image

from diffusion_sae.model import generate, get_block
from steering.common    import HOOKPOINT, COND_ONLY


def compute_delta(sae, feat, strength, method):
    """Build the additive style delta [d_in, 16, 16] for one style's features at a given strength."""
    dev  = sae.W_dec.device
    W    = sae.W_dec[feat["idx"].to(dev)].detach().float()    # [F, d_in] unit-norm directions
    maps = feat["maps"].to(dev).float()                       # [F, 16, 16] styled per-patch level
    if method == "uniform":
        level = maps.mean(dim=(1, 2), keepdim=True).expand_as(maps)   # flat per-feature level (== mu)
    elif method == "patch":
        level = maps                                          # per-patch level
    else:
        raise ValueError(f"method must be 'uniform' or 'patch', got {method!r}")
    with torch.no_grad():
        return strength * torch.einsum("fc,fij->cij", W, level)       # delta[c,i,j]=sum_f level[f,i,j]*W[f,c]


def make_injection_hook(delta, cond_only=COND_ONLY):
    """Return a forward-hook closure that adds `delta` to the block output (conditional half if cond_only)."""
    def hook(module, inp, out):
        sample = out[0]                                       # [2, d_in, 16, 16]  (uncond ; cond)
        d = delta.to(sample.dtype)
        if cond_only:
            sample[1:2] += d                                  # conditional half only
        else:
            sample += d
        return out
    return hook


@contextmanager
def style_injection(pipe, sae, feat, strength, method, cond_only=COND_ONLY):
    """Context manager: register the style-injection hook on up.1.2 for the duration of the block, then remove it."""
    delta  = compute_delta(sae, feat, strength, method)
    handle = get_block(pipe, HOOKPOINT).register_forward_hook(make_injection_hook(delta, cond_only))
    try:
        yield
    finally:
        handle.remove()


def set_pipe(pipe):
    """Decode in fp32 -> avoids SD-1.4 fp16 VAE NaNs (the UNet stays fp16). Call once after loading."""
    pipe.vae.to(torch.float32)


def vae_decode(pipe, latents):
    """Decode final latents to a PIL image in fp32 (robust against fp16 VAE NaNs)."""
    vae = pipe.vae
    lat = (latents / vae.config.scaling_factor).to(vae.dtype)
    with torch.no_grad():
        img = vae.decode(lat).sample
    img = (img / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).float().numpy()
    return Image.fromarray((img * 255).round().astype("uint8"))


def generate_baseline(pipe, prompt, seed):
    """Neutral image, no steering."""
    latents = generate(pipe, prompt, decode_frames=False, seed=seed)
    return vae_decode(pipe, latents)


def generate_steered(pipe, sae, prompt, feat, method, strength, seed):
    """Neutral prompt + injected style direction(s) at up.1.2."""
    with style_injection(pipe, sae, feat, strength, method):
        latents = generate(pipe, prompt, decode_frames=False, seed=seed)
    return vae_decode(pipe, latents)

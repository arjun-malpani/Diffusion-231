"""
Inject SAE style directions into SD generation at up.1.2.

Two methods:
  uniform         : GLOBAL feature set, SAME vector at every patch
                    delta[:,i,j] = strength * Sum_f mu[f] * W_dec[idx[f]]
  patch_selective : a DIFFERENT feature SET chosen per patch (top-K style-adding feats at each patch)
                    delta[:,i,j] = strength * Sum_k patch_level[p,k] * W_dec[patch_idx[p,k]]   (p = i*16+j)

W_dec rows are unit-norm and the levels (mu / patch_level) are styled activation levels, so strength=1.0
injects features at ~the level the real style produces. Injection is on the conditional CFG half only, every step.

Purely functional: compute_delta -> make_injection_hook -> style_injection (register/remove).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import contextmanager

import torch
from PIL import Image

from diffusion_sae.model import generate, get_block
from steering.common    import HOOKPOINT, COND_ONLY, HW



def compute_delta(sae, feat, strength, method):
    """Build the additive style delta [d_in, 16, 16] for one style at a given strength."""
    dev = sae.W_dec.device
    with torch.no_grad():
        
        if method == "uniform":
            # global feature set, identical vector at every patch: sum_f mu[f] * W_dec[idx[f]]
            W   = sae.W_dec[feat["idx"].to(dev)].float()              # [F, d_in] unit-norm directions
            mu  = feat["mu"].to(dev).float()                          # [F] styled levels
            vec = torch.einsum("f,fc->c", mu, W)                      # [d_in]
            delta = vec[:, None, None].expand(-1, HW, HW)             # broadcast flat -> [d_in,16,16]
        elif method == "patch_selective":
            # per-patch feature set: at patch p add sum_k level[p,k] * W_dec[idx[p,k]]
            pidx = feat["patch_idx"].to(dev)                          # [P, K] feature ids per patch
            plvl = feat["patch_level"].to(dev).float()               # [P, K] level     per patch
            dirs = sae.W_dec[pidx].float()                           # [P, K, d_in]
            perp = (plvl.unsqueeze(-1) * dirs).sum(dim=1)            # [P, d_in]
            delta = perp.reshape(HW, HW, -1).permute(2, 0, 1)        # [d_in,16,16]  (p = i*16+j)
        else:
            raise ValueError(f"method must be 'uniform' or 'patch_selective', got {method!r}")
        return strength * delta


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

'''

load_pipeline: imports diffusion pipeline

generate: generates image given model 

'''


import torch
import matplotlib.pyplot as plt

import math

from diffusers import StableDiffusionPipeline

from PIL import Image 

from .config import DEVICE, DTYPE, HEIGHT, WIDTH, INFERENCE_STEPS, GUIDANCE_SCALE


#get sd 1.4
def import_model(model_id, dtype=DTYPE, device=DEVICE):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None,
    ).to(device)
    return pipe

#use SD CLIP embedding to encode
def embed(pipe, text):
    #Encode a single text string -> [1, 77, 768] CLIP embeddings.
    #Always runs on whatever device the pipe is loaded on (no separate device kwarg).

    tk = pipe.tokenizer(text, padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad():
        return pipe.text_encoder(tk.input_ids.to(pipe.device))[0]

#use SD's VAE decoding (latent --> full scale image, scale up 8x)
def decode(pipe, lat):
    with torch.no_grad(): 
        img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).float().numpy()
        return Image.fromarray((img * 255).round().astype("uint8"))

#generate from SD and prompt
def generate(pipe, prompt="", guidance_scale=GUIDANCE_SCALE, inference_steps=INFERENCE_STEPS,
             height=HEIGHT, width=WIDTH, seed=42, decode_frames=True):
    #CLIP text embeddings for CFG: stack [uncond ; cond] along batch
    text_embeddings = torch.cat([embed(pipe, ""), embed(pipe, prompt)])

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8),
                          generator=generator).to(pipe.device, dtype=pipe.unet.dtype)

    pipe.scheduler.set_timesteps(inference_steps)
    latents = latents * pipe.scheduler.init_noise_sigma

    frames = [("start (pure noise)", decode(pipe, latents))] if decode_frames else None

    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embeddings).sample
        nu, nt = noise_pred.chunk(2)
        noise_pred = nu + guidance_scale * (nt - nu)              #CFG
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        if decode_frames:
            frames.append((f"step {i+1}", decode(pipe, latents)))

    return frames if decode_frames else latents

#plot the frames of the output of the model 
def plot_generate(frames, full_trajectory=False, show=True, save=True, path="denoising_out.png"):
    if full_trajectory:
        #grid of every step in  denoising trajectory
        cols = 8
        rows = math.ceil(len(frames) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (title, im) in zip(axes.flat, frames):
            ax.imshow(im)
            ax.set_title(title, fontsize=8)
    else:
        #just  final image
        title, im = frames[-1]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=110, bbox_inches="tight")
        print(f"saved {path}")
    if show:
        plt.show()

    plt.close(fig)


def get_block(pipe, hookpoint: str): 
    #get block of model based on the hookpoint: ex 'unet.up_blocks.1.attentions.2'
    curr_block = pipe

    for part in hookpoint.split("."): 
        if part.isdigit(): 
            curr_block = curr_block[int(part)]
        else: 
            curr_block = getattr(curr_block, part)

    return curr_block 


def get_activation(pipe, hookpoints: list[str], prompt, **generate_kwargs):
    #hookpoints: list of dotted-path strings (see get_block)
  
    activations = {hp: [] for hp in hookpoints}
    handles = []

    def make_hook(hp):
        def hook_fn(module, inp, out):
            activations[hp].append(out[0].detach().cpu())
        return hook_fn

    for hp in hookpoints:
        block = get_block(pipe, hp)
        handles.append(block.register_forward_hook(make_hook(hp)))

    try:
        generate(pipe, prompt, **generate_kwargs)   
    finally:
        for h in handles:                            
            h.remove()

    return activations


    
    


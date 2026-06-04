import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np 

import math

from diffusers import StableDiffusionPipeline

from PIL import Image 

from .config import NUM_LATENTS


class SAE(nn.Module): 
    def __init__(self, d_in, num_latents = NUM_LATENTS): 
        super().__init__()

        self.d_in = d_in 
        self.num_latents = num_latents

        self.encoder = nn.Linear(d_in, num_latents)
        self.W_dec =  nn.Parameter(torch.randn(num_latents, d_in) * 0.01) 
        
        self.b_dec = nn.Parameter(torch.zeros(d_in)) #we might want to set this to be the mean of the data

        self.normalize_decoder()
    
    @torch.no_grad()
    def normalize_decoder(self): 
        self.W_dec.div_(self.W_dec.norm(dim =1, keepdim = True) + 1e-8)

    def forward(self, x, k=32):
        # center on the way in
        x = x - self.b_dec

        hidden = F.relu(self.encoder(x))

        if self.training:
            # BatchTopK: pick the top (k*B) latents across the WHOLE batch's pre-acts.
            B = hidden.shape[0]
            flat = hidden.flatten()                                                 # [B * num_latents]
            top = torch.topk(flat, k * B, sorted=False)
            z = torch.zeros_like(flat).scatter_(-1, top.indices, top.values).reshape(hidden.shape)
        else:
            #PatchTopK: each sample independently keeps its top-k. Deterministic per-row sparsity = k.
            topk = torch.topk(hidden, k, dim=-1)
            z = torch.zeros_like(hidden).scatter_(-1, topk.indices, topk.values)

        recon = z @ self.W_dec + self.b_dec
        return recon, z, hidden          # hidden = post-ReLU pre-topk acts, reused by AuxK

        



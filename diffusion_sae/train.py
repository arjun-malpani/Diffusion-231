import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import math

from tqdm.auto import tqdm

from .config import DEVICE, EPOCHS, AUX_ALPHA, LEARNING_RATE, DEAD_THRESHOLD

from .sae import SAE


def compute_auxk_loss(sae, X, recon, pre_acts, dead_mask, k_aux):
    #reconstructs dead features against the residual of the main pass.
    #no loss if no dead latents
    
    if dead_mask.sum() == 0:
        return torch.tensor(0.0, device=X.device)

    neg_inf = torch.full_like(pre_acts, float("-inf"))
    pool = torch.where(dead_mask, pre_acts, neg_inf)  #mask out live features

    #topk of dead latents based on k_aux
    newk = min(k_aux, int(dead_mask.sum().item()))   #in case fewer dead than k_aux
    top_of_aux = torch.topk(pool, newk, dim=-1)
    z_aux    = torch.zeros_like(pre_acts).scatter_(-1, top_of_aux.indices, top_of_aux.values)

    #no b_dec on aux head -- residual already lives in centered space
    aux_recon = z_aux @ sae.W_dec
    residual  = (X - recon).detach()
    return (aux_recon - residual).pow(2).mean()


def maybe_resume(path, sae, optimizer):
    #load a checkpoint into sae + optimizer in-place.
    #returns (step, epoch, dead_latents, history), last two may be None w fresh start
    if path is None or not os.path.exists(path):
        return 0, 0, None, None
    ckpt = torch.load(path, map_location="cpu")
    sae.load_state_dict(ckpt["sae"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[resume] loaded {path}: step={ckpt['step']}, epoch={ckpt['epoch']}")
    return ckpt["step"], ckpt["epoch"], ckpt["dead_latents"], ckpt.get("history")


def save_ckpt(path, sae, optimizer, step, epoch, dead_latents, history):
    #full mid-training checkpoint: weights + Adam state + step/epoch + dead-feature counter + history
    torch.save({
        "sae":          sae.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "step":         step,
        "epoch":        epoch,
        "dead_latents": dead_latents.cpu(),
        "history":      history,
    }, path)


def empty_history():
    # schema for the history dict, one list per metric, all parallel to "step".
    return {"step": [], "mse": [], "aux": [], "total": [], "l0": [], "dead": [], "fvu": []}


def plot_history(history, save=None, show=True):
    '''2x2 grid: losses, L0, dead-feature count, FVU. Pass the history dict returned by train()'''
    if not history["step"]:
        print("history is empty"); return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    steps = history["step"]

    # (0,0) loss curves -- log scale so MSE and AuxK can share axes
    axes[0, 0].plot(steps, history["mse"],   label="mse")
    axes[0, 0].plot(steps, history["aux"],   label="aux")
    axes[0, 0].plot(steps, history["total"], label="total", linestyle="--")
    axes[0, 0].set_yscale("log"); axes[0, 0].set_title("loss")
    axes[0, 0].set_xlabel("step"); axes[0, 0].legend()

    # (0,1) L0 mean -- should hover around k (32 by default)
    axes[0, 1].plot(steps, history["l0"])
    axes[0, 1].set_title("L0 mean (target = k)")
    axes[0, 1].set_xlabel("step")

    # (1,0) dead feature count -- should plateau, not climb without bound
    axes[1, 0].plot(steps, history["dead"])
    axes[1, 0].set_title("dead features")
    axes[1, 0].set_xlabel("step")

    # (1,1) FVU -- normalized reconstruction quality; SAeUron paper-final ~0.18-0.2
    axes[1, 1].plot(steps, history["fvu"])
    axes[1, 1].set_title("FVU (fraction variance unexplained)")
    axes[1, 1].set_xlabel("step")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=110, bbox_inches="tight")
        print(f"saved {save}")
    if show:
        plt.show()
    plt.close(fig)


def train(sae, data_loader, learning_rate=LEARNING_RATE,
          epochs=EPOCHS, aux_alpha=AUX_ALPHA, device=DEVICE,
          dead_threshold=DEAD_THRESHOLD, log_every=100,
          ckpt_dir="checkpoints", checkpoint_every=5000, resume_from=None,
          final_path=None):

    os.makedirs(ckpt_dir, exist_ok=True)
    if final_path is None:
        final_path = os.path.join(ckpt_dir, "sae_final.pt")     #defaults inside ckpt_dir

    #model
    sae = sae.to(device)
    sae.train()
    sae.normalize_decoder()

    dead_latents = torch.zeros(sae.num_latents, device=device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    k_aux = sae.d_in // 2   #NOTE: SAeUron heuristic, half of d_in (modifiable)
    history = empty_history()

    #resume if a checkpoint path was given (and exists) otherwise fresh start
    start_step, start_epoch, saved_dead, saved_hist = maybe_resume(resume_from, sae, optimizer)
    if saved_dead is not None:
        dead_latents = saved_dead.to(device)
    if saved_hist is not None:
        history = saved_hist
    step = start_step

    for epoch in range(start_epoch, epochs):
        for batch in data_loader:

            X = batch.to(device)
            recon, z, pre_acts = sae(X)   #pre_acts reused by AuxK

            #losses
            mse  = (recon - X).pow(2).mean()
            dead_mask = dead_latents >= dead_threshold
            aux  = compute_auxk_loss(sae, X, recon, pre_acts, dead_mask, k_aux)
            loss = mse + aux_alpha * aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()  #unit-norm rows after every step

            with torch.no_grad():
                #counting dead latents (samples since last fire)
                fired = (z > 0).any(dim=0)
                dead_latents += X.shape[0]
                dead_latents[fired] = 0

                if step % log_every == 0:
                    dead = (dead_latents >= dead_threshold).sum().item()
                    l0   = (z > 0).sum(dim=-1).float().mean().item()
                    fvu  = (mse / X.var()).item()        #normalized reconstruction quality
                    print(f"Epoch {epoch} | Step {step} | mse {mse.item():.5f} | aux {aux.item():.5f} "
                          f"| L0 {l0:.1f} | dead {dead}/{sae.num_latents} | fvu {fvu:.4f}")
                    
                    #append to history 
                    history["step"].append(step)
                    history["mse"].append(mse.item())
                    history["aux"].append(aux.item())
                    history["total"].append(loss.item())
                    history["l0"].append(l0)
                    history["dead"].append(dead)
                    history["fvu"].append(fvu)

                #periodic checkpoint
                if step > 0 and step % checkpoint_every == 0:
                    save_ckpt(os.path.join(ckpt_dir, f"sae_step{step}.pt"),
                              sae, optimizer, step, epoch, dead_latents, history)

            step += 1

    #final artifact, weights only, the file you'd actually load for inference / sharing
    torch.save(sae.state_dict(), final_path)
    print(f"saved final weights to {final_path}")
    return sae, history







            


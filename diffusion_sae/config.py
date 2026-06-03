import torch


def _pick_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16       
    if torch.backends.mps.is_available():
        return "mps", torch.float32        
    return "cpu", torch.float32 

DEVICE, DTYPE = _pick_device_dtype()

MODEL_ID = "CompVis/stable-diffusion-v1-4"

#paths 
ACT_CACHE_DIR = "data/activations"
SAE_CKPT_DIR  = "checkpoints"
OUTPUT_DIR    = "output_img"

#generate params
HEIGHT = 512
WIDTH = 512
INFERENCE_STEPS = 30 
GUIDANCE_SCALE = 7.5

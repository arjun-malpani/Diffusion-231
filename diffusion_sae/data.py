"""
ActivationShardDataset: lazy, memory-mapped, multi-shard activation dataset
for SAE training. Each shard is a single torch tensor file of shape [N, d_in]
saved as fp16 (compact on disk). On read, samples are returned as fp32 by
default to match the SAE's training dtype.

Shards are loaded with torch.load(..., mmap=True), so the dataset can be much
larger than RAM -- only the rows accessed in a batch get pulled from disk into
the page cache.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ActivationShardDataset(Dataset):
    def __init__(self, shard_dir, dtype=torch.float32, pattern="shard_*.pt"):
        self.shard_dir   = shard_dir
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, pattern)))
        if not self.shard_paths:
            raise FileNotFoundError(f"no shards matching {pattern!r} in {shard_dir!r}")

        # memmap each shard. They never get fully read into RAM unless accessed.
        # weights_only=True is safe for raw tensors and silences the deprecation warning
        self.shards = [torch.load(p, map_location="cpu", mmap=True, weights_only=True)
                       for p in self.shard_paths]

        self.lens = [s.shape[0] for s in self.shards]
        self.cum  = np.cumsum(self.lens)            # cumulative row counts
        self.total = int(self.cum[-1])
        self.d_in  = int(self.shards[0].shape[-1])
        self.dtype = dtype

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # locate which shard, then row within shard
        s = int(np.searchsorted(self.cum, idx, side="right"))
        local = idx - (int(self.cum[s - 1]) if s > 0 else 0)
        return self.shards[s][local].to(self.dtype)

    def info(self):
        return (f"{len(self.shards)} shards | {self.total:,} samples | "
                f"d_in={self.d_in} | on-disk dtype={self.shards[0].dtype} | "
                f"returns {self.dtype}")

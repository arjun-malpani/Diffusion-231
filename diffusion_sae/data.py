"""
ActivationShardDataset: lazy, memory-mapped, multi-shard activation dataset
for SAE training. Each shard is a single torch tensor file of shape [N, d_in]
saved as fp16 (compact on disk). On read, samples are returned as fp32 by
default to match the SAE's training dtype.

Shards are loaded with torch.load(..., mmap=True), so the dataset can be much
larger than RAM -- only the rows accessed in a batch get pulled from disk into
the page cache.

NOTE: ActivationShardDataset + DataLoader(shuffle=True) is only fast when the
data fits (mostly) in the page cache. When it doesn't (e.g. 78 GB data, 15 GB
RAM), random access thrashes the disk and the GPU starves. For that case use
StreamingShardDataset (below), which reads whole shards sequentially and
shuffles in RAM.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


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


class StreamingShardDataset(IterableDataset):
    """Shard-streaming dataset for datasets far larger than RAM.

    The random-access ActivationShardDataset above thrashes when the data
    (e.g. 78 GB) dwarfs the page cache (e.g. 10 GB): with shuffle=True every
    batch pulls scattered rows, so almost every read misses cache and hits disk
    -- the GPU starves at ~0% util.

    This class instead reads a few WHOLE shards into RAM at a time (a fast
    *sequential* ~2 GB/shard read), shuffles rows *within* that in-RAM buffer,
    and yields ready-made [batch_size, d_in] batches. Disk access becomes
    sequential (~one pass per epoch) and the GPU stays fed.

    Shuffle quality stays high because collection already shuffled the prompts
    before sharding, so each shard is a random mix of styles; we additionally
    (a) shuffle shard order every epoch and (b) mix `shard_buffer` shards
    together before shuffling rows. Yields full batches, so use it directly as
    the loader (no DataLoader needed):  for batch in ds: ...

    Args:
        shard_dir:    folder of shard_*.pt tensors ([N, d_in], fp16 on disk)
        batch_size:   rows per yielded batch
        dtype:        dtype of yielded batches (fp32 to match SAE training)
        shuffle:      shuffle shard order + rows each epoch
        shard_buffer: how many shards to hold/mix in RAM at once (RAM ~= 2 GB * this)
        seed:         base RNG seed; epoch index is added so each epoch differs
        drop_last:    drop the <batch_size remainder of each buffer group
    """

    def __init__(self, shard_dir, batch_size, dtype=torch.float32, shuffle=True,
                 shard_buffer=2, seed=0, pattern="shard_*.pt", drop_last=True):
        super().__init__()
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, pattern)))
        if not self.shard_paths:
            raise FileNotFoundError(f"no shards matching {pattern!r} in {shard_dir!r}")

        self.batch_size   = batch_size
        self.dtype        = dtype
        self.shuffle      = shuffle
        self.shard_buffer = max(1, shard_buffer)
        self.seed         = seed
        self.drop_last    = drop_last
        self._epoch       = 0

        # read shard shapes via mmap (no data pulled) to get totals + d_in
        lens = []
        for p in self.shard_paths:
            t = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
            lens.append(t.shape[0])
            self.d_in = int(t.shape[-1])
        self.total = int(sum(lens))

    def __len__(self):
        # batches per epoch (approximate: drop_last is applied per buffer group)
        return self.total // self.batch_size

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self._epoch)
        order = list(range(len(self.shard_paths)))
        if self.shuffle:
            perm = torch.randperm(len(order), generator=rng).tolist()
            order = [order[i] for i in perm]
        self._epoch += 1

        B = self.batch_size
        for start in range(0, len(order), self.shard_buffer):
            group = order[start:start + self.shard_buffer]
            # FULL load (sequential read) of each shard in the group -> RAM
            tensors = [torch.load(self.shard_paths[i], map_location="cpu",
                                  weights_only=True) for i in group]
            data = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
            del tensors

            n = data.shape[0]
            idx = (torch.randperm(n, generator=rng) if self.shuffle
                   else torch.arange(n))
            limit = (n - (n % B)) if self.drop_last else n
            for s in range(0, limit, B):
                yield data[idx[s:s + B]].to(self.dtype)
            del data, idx

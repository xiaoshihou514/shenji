"""
dataset.py – memory-mapped loader for sharded NumPy archives.

Each shard is produced by ``shenji/preprocess.py`` and contains:
    x : uint8  shape (N, 71)   – board-state tokens
    y : int16  shape (N,)      – AlphaZero move index  (0 … 4671)

NPZShard memory-maps a single file to avoid large RAM spikes.
MultiShard concatenates multiple shards with O(log S) index routing.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["NPZShard", "MultiShard", "shard_paths"]


def shard_paths(data_dir: Path, pattern: str) -> list[Path]:
    """Return shard paths sorted deterministically."""
    return sorted(data_dir.glob(pattern))


class NPZShard(Dataset):
    """Memory-maps a single ``.npz`` shard."""

    def __init__(self, path: Path) -> None:
        npz = np.load(path, mmap_mode="r")
        self.x: np.ndarray = npz["x"]  # (N, 71) uint8
        self.y: np.ndarray = npz["y"]  # (N,)    int16

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        # Cast on the fly to avoid an upfront int64 copy of the entire shard.
        x = torch.from_numpy(self.x[idx].astype(np.int64, copy=False))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


class MultiShard(Dataset):
    """
    Concatenates multiple shards into one virtual dataset.

    Uses a prefix-sum table to route each global index to the correct
    shard with O(log S) binary search, where S = number of shards.
    """

    def __init__(self, data_dir: Path, pattern: str) -> None:
        paths = shard_paths(data_dir, pattern)
        if not paths:
            raise FileNotFoundError(
                f"No shards matching {pattern!r} found in {data_dir}"
            )
        self.shards = [NPZShard(p) for p in paths]
        lengths = [len(s) for s in self.shards]
        self.cum_lens = np.add.accumulate(lengths)

    def __len__(self) -> int:
        return int(self.cum_lens[-1])

    def __getitem__(self, idx: int):
        shard_idx = int(np.searchsorted(self.cum_lens, idx, side="right"))
        inner_idx = idx - (self.cum_lens[shard_idx - 1] if shard_idx else 0)
        return self.shards[shard_idx][inner_idx]

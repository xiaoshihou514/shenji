from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .encoder import BoardEncoder
from .move_vocab import MoveVocab
from tqdm import tqdm

class NPZShard(Dataset):
    """
    One memory-mapped shard produced by preprocess_pgn.py.
    Each shard holds two arrays:
        'x' - uint8  shape (N, 65)
        'y' - int32  shape (N,)
    """
    def __init__(self, path: Path):
        self._npz = np.load(path, mmap_mode='r')
        self.x = self._npz['x']
        self.y = self._npz['y']

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx].astype(np.int64)),
            torch.tensor(self.y[idx], dtype=torch.long),
        )

def shard_paths(data_dir: Path, pattern: str):
    return sorted(data_dir.glob(pattern))

class MultiShard(Dataset):
    def __init__(self, data_dir: Path, pattern: str):
        self.shards = [NPZShard(p) for p in shard_paths(data_dir, pattern)]
        self.cum_lens = np.cumsum([len(s) for s in self.shards])

    def __len__(self):                # noqa: D401
        return int(self.cum_lens[-1])

    def __getitem__(self, idx: int):
        shard_idx = np.searchsorted(self.cum_lens, idx, side='right')
        inner_idx = idx - (self.cum_lens[shard_idx - 1] if shard_idx else 0)
        return self.shards[shard_idx][inner_idx]

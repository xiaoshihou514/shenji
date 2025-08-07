import torch
from pathlib import Path
from datetime import datetime

def save_checkpoint(state: dict, folder: Path, epoch: int):
    folder.mkdir(parents=True, exist_ok=True)
    fname = folder / f"epoch_{epoch:03d}.pt"
    torch.save(state, fname)

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

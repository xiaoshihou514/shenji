from __future__ import annotations
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import MultiShard
from .model import ChessTransformer
from .config import TrainConfig
from .move_vocab import MoveVocab

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MultiShard(Path(args.data), cfg.shard_pattern)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = ChessTransformer(
        cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dim_feedforward, cfg.dropout
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    correct_top1 = correct_topk = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += y.size(0)
            pred = logits.topk(args.topk, dim=1).indices
            correct_top1 += (pred[:, 0] == y).sum().item()
            correct_topk += (pred == y.unsqueeze(1)).any(dim=1).sum().item()

    print(json.dumps({
        "top1": correct_top1 / total,
        f"top{args.topk}": correct_topk / total
    }, indent=2))

if __name__ == "__main__":
    main()

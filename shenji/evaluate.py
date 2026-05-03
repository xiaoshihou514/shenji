"""
evaluate.py – compute top-1 / top-5 accuracy for a trained checkpoint.

Usage:
    python -m shenji.evaluate \\
        --config shenji/config.yaml \\
        --checkpoint checkpoints/epoch_020_step_0123456.pt \\
        --data ./data \\
        [--topk 5]
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shenji.config import TrainConfig
from shenji.dataset import MultiShard
from shenji.model import ChessTransformer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a ChessTransformer checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluation at this many positions (default: full dataset)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MultiShard(args.data, cfg.shard_pattern)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── load model ────────────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("cfg", {})
    model = ChessTransformer(
        d_model=saved_cfg.get("d_model", cfg.d_model),
        nhead=saved_cfg.get("nhead", cfg.nhead),
        num_layers=saved_cfg.get("num_layers", cfg.num_layers),
        dim_feedforward=saved_cfg.get("dim_feedforward", cfg.dim_feedforward),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ── evaluation loop ───────────────────────────────────────────────────────
    top1_correct = topk_correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            if args.max_samples and total >= args.max_samples:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)

            topk_preds = logits.topk(args.topk, dim=1).indices  # (B, k)
            top1_correct += (topk_preds[:, 0] == y).sum().item()
            topk_correct += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

    results = {
        "checkpoint": str(args.checkpoint),
        "positions_evaluated": total,
        "top1_accuracy": round(top1_correct / total, 6),
        f"top{args.topk}_accuracy": round(topk_correct / total, 6),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

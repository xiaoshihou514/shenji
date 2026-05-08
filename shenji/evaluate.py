"""
evaluate.py – compute top-1 / top-k accuracy for a trained checkpoint.

Usage:
    uv run shenji/evaluate.py \
        --config shenji/config.yaml \
        --checkpoint checkpoints/epoch_020_step_0123456.pt \
        --data ./data \
        [--shard-split val] \
        [--topk 5]
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shenji.config import TrainConfig
from shenji.dataset import MultiShard, shard_paths
from shenji.model import ChessTransformer


def _select_paths(cfg: TrainConfig, data_dir: Path, shard_split: str, shards: str | None) -> list[Path]:
    paths = shard_paths(data_dir, cfg.shard_pattern, cfg.max_shards)
    if not paths:
        raise FileNotFoundError(f"No shards matching {cfg.shard_pattern!r} found in {data_dir}")

    if shards:
        selected: list[Path] = []
        seen: set[int] = set()
        for part in shards.split(","):
            idx = int(part.strip())
            if idx < 0 or idx >= len(paths):
                raise ValueError(
                    f"Shard index {idx} out of range for {len(paths)} available shards"
                )
            if idx not in seen:
                selected.append(paths[idx])
                seen.add(idx)
        return selected

    if shard_split == "all":
        return paths
    if shard_split == "train":
        if len(paths) <= cfg.val_shards:
            return paths
        return paths[:-cfg.val_shards]

    # default: same validation holdout rule as train.py
    if len(paths) <= cfg.val_shards:
        return paths[-1:]
    return paths[-cfg.val_shards:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ChessTransformer checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--shard-split",
        choices=["train", "val", "all"],
        default="val",
        help="Which shard split to evaluate (default: val, matching train.py holdout)",
    )
    parser.add_argument(
        "--shards",
        default=None,
        help="Comma-separated zero-based shard indices to evaluate; overrides --shard-split",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluation at this many positions (default: full selected shards)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_paths = _select_paths(cfg, args.data, args.shard_split, args.shards)
    ds = MultiShard.from_paths(selected_paths)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

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

    top1_correct = topk_correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            if args.max_samples and total >= args.max_samples:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)

            topk_preds = logits.topk(args.topk, dim=1).indices
            top1_correct += (topk_preds[:, 0] == y).sum().item()
            topk_correct += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

    results = {
        "checkpoint": str(args.checkpoint),
        "shard_split": args.shard_split if args.shards is None else "explicit",
        "shards": [path.name for path in selected_paths],
        "positions_evaluated": total,
        "top1_accuracy": round(top1_correct / total, 6),
        f"top{args.topk}_accuracy": round(topk_correct / total, 6),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

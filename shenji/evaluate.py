"""
evaluate.py – compute top-1 / top-k accuracy on held-out test shards.

Usage:
    uv run shenji/evaluate.py \
        --config shenji/config.yaml \
        --checkpoint checkpoints/epoch_020_step_0123456.pt \
        --data ./data \
        [--shards 1] \
        [--topk 5]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from shenji.config import TrainConfig
from shenji.dataset import MultiShard, shard_paths
from shenji.model import ChessTransformer


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _select_test_paths(cfg: TrainConfig, data_dir: Path, shard_count: int | None) -> list[Path]:
    paths = shard_paths(data_dir, cfg.shard_pattern, cfg.max_shards)
    if not paths:
        raise FileNotFoundError(f"No shards matching {cfg.shard_pattern!r} found in {data_dir}")
    if len(paths) <= cfg.val_shards + cfg.test_shards:
        raise ValueError(
            f"Need more than val_shards + test_shards = {cfg.val_shards + cfg.test_shards} shards "
            f"to keep train/val/test separate, got {len(paths)}."
        )

    test_paths = paths[-cfg.test_shards :]
    use_count = shard_count if shard_count is not None else len(test_paths)
    if use_count <= 0:
        raise ValueError("--shards must be a positive integer")
    if use_count > len(test_paths):
        raise ValueError(
            f"--shards={use_count} exceeds available held-out test shards ({len(test_paths)})"
        )
    return test_paths[:use_count]


def _log_path(save_dir: Path, checkpoint: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = save_dir / "eval_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stamp}_{checkpoint.stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ChessTransformer checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--shards",
        type=int,
        default=None,
        help="How many held-out test shards to use (default: all configured test shards)",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluation at this many positions (default: full selected test shards)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(device)

    selected_paths = _select_test_paths(cfg, args.data, args.shards)
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
    target_total = min(len(ds), args.max_samples) if args.max_samples else len(ds)
    progress = tqdm(total=target_total, desc="evaluate", unit="pos", dynamic_ncols=True)

    with torch.no_grad():
        for x, y in loader:
            if args.max_samples and total >= args.max_samples:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(x)

            topk_preds = logits.topk(args.topk, dim=1).indices
            top1_correct += (topk_preds[:, 0] == y).sum().item()
            topk_correct += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
            progress.update(min(y.size(0), max(target_total - progress.n, 0)))

    progress.close()

    results = {
        "checkpoint": str(args.checkpoint),
        "shards_used": len(selected_paths),
        "shard_files": [path.name for path in selected_paths],
        "positions_evaluated": total,
        "top1_accuracy": round(top1_correct / total, 6),
        f"top{args.topk}_accuracy": round(topk_correct / total, 6),
    }
    log_path = _log_path(cfg.save_dir, args.checkpoint)
    log_path.write_text(json.dumps(results, indent=2) + "\n")
    results["log_path"] = str(log_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

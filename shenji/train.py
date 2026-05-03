"""
train.py – supervised training of ChessTransformer on pre-processed shards.

Usage:
    python -m shenji.train --config shenji/config.yaml

Features:
    • fp16 mixed-precision via torch.amp
    • AdamW + cosine LR schedule with linear warm-up
    • Gradient accumulation for large effective batch sizes
    • Checkpoint save / resume  (model + optimiser + scaler + step counter)
    • TensorBoard logging
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from shenji.config import TrainConfig
from shenji.dataset import MultiShard
from shenji.model import ChessTransformer


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)  # near-atomic rename


def _load_checkpoint(
    path: Path,
    model: ChessTransformer,
    opt: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    """Load checkpoint and return (start_epoch, global_step)."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    opt.load_state_dict(state["opt_state"])
    scaler.load_state_dict(state["scaler_state"])
    print(f"Resumed from {path}  (epoch {state['epoch']}, step {state['global_step']})")
    return state["epoch"], state["global_step"]


def _build_scheduler(
    opt: AdamW, warmup_steps: int, total_steps: int, min_lr: float
) -> SequentialLR:
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        opt, T_max=max(total_steps - warmup_steps, 1), eta_min=min_lr
    )
    return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])


@torch.no_grad()
def _quick_eval(
    model: ChessTransformer,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    max_samples: int = 50_000,
) -> tuple[float, float]:
    """Evaluate on the first *max_samples* examples from *loader*."""
    model.eval()
    loss_sum = acc_sum = n = 0
    for x, y in loader:
        if n >= max_samples:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
        acc_sum += (logits.argmax(1) == y).float().sum().item()
        n += y.size(0)
    model.train()
    return loss_sum / max(n, 1), acc_sum / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChessTransformer")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{_now()} device={device}")

    # ── data ──────────────────────────────────────────────────────────────────
    ds = MultiShard(cfg.data_dir, cfg.shard_pattern)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    steps_per_epoch = len(loader) // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    print(
        f"{_now()} dataset={len(ds):,} positions  "
        f"steps_per_epoch={steps_per_epoch:,}  total_steps={total_steps:,}"
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = ChessTransformer(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)
    print(f"{_now()} parameters={model.num_parameters():,}")

    # ── optimiser / scheduler ─────────────────────────────────────────────────
    opt = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scaler = torch.amp.GradScaler(device=device.type)
    scheduler = _build_scheduler(opt, cfg.warmup_steps, total_steps, cfg.min_lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=cfg.save_dir / "tb")

    # ── optional resume ───────────────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    if cfg.resume:
        start_epoch, global_step = _load_checkpoint(
            cfg.resume, model, opt, scaler, device
        )
        # Fast-forward the scheduler to the resumed step
        for _ in range(global_step):
            scheduler.step()
        start_epoch += 1

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(start_epoch, cfg.epochs + 1):
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        opt.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for local_step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y) / cfg.grad_accum

            scaler.scale(loss).backward()
            batch_loss += loss.item()

            if local_step % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    acc = (logits.argmax(1) == y).float().mean().item()
                    lr_now = opt.param_groups[0]["lr"]
                    writer.add_scalar("train/loss", batch_loss, global_step)
                    writer.add_scalar("train/acc", acc, global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)
                    pbar.set_postfix(
                        step=global_step,
                        loss=f"{batch_loss:.4f}",
                        acc=f"{acc:.3f}",
                        lr=f"{lr_now:.2e}",
                    )

                if global_step % cfg.eval_every == 0:
                    eval_loss, eval_acc = _quick_eval(model, loader, criterion, device)
                    writer.add_scalar("eval/loss", eval_loss, global_step)
                    writer.add_scalar("eval/acc", eval_acc, global_step)
                    print(
                        f"\n{_now()} EVAL  step={global_step}  "
                        f"loss={eval_loss:.4f}  acc={eval_acc:.4f}"
                    )

                batch_loss = 0.0

        # ── save checkpoint at end of every epoch ─────────────────────────────
        ckpt_path = cfg.save_dir / f"epoch_{epoch:03d}_step_{global_step:07d}.pt"
        _save_checkpoint(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "cfg": {
                    "d_model": cfg.d_model,
                    "nhead": cfg.nhead,
                    "num_layers": cfg.num_layers,
                    "dim_feedforward": cfg.dim_feedforward,
                    "dropout": cfg.dropout,
                },
            },
            ckpt_path,
        )
        print(f"{_now()} saved {ckpt_path}")

    writer.close()
    print(f"{_now()} Training complete.")


if __name__ == "__main__":
    main()

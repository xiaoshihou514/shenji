"""
train.py – supervised training of ChessTransformer on pre-processed shards.

Usage:
    uv run shenji/train.py --config shenji/config.yaml

Features:
    • fp16/bf16 mixed-precision via torch.amp (bf16 used automatically if supported)
    • AdamW + cosine LR schedule with linear warm-up
    • Gradient accumulation for large effective batch sizes
    • NaN/Inf loss & gradient guard: corrupted batches are skipped safely
    • Grad norm + scaler scale logged to TensorBoard for early anomaly detection
    • Checkpoint save / resume  (model + optimiser + scaler + step counter)
    • best.pt saved whenever validation accuracy improves
    • TensorBoard logging
"""

import argparse
import warnings
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
from shenji.dataset import MultiShard, shard_paths
from shenji.model import ChessTransformer


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _amp_dtype(device: torch.device) -> torch.dtype:
    """Return bf16 if the device supports it natively, else fp16."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)  # near-atomic rename


def _load_checkpoint(
    path: Path,
    model: ChessTransformer,
    opt: AdamW,
    scheduler: SequentialLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[int, int]:
    """Load checkpoint and return (start_epoch, global_step).

    scaler state is only restored when the checkpoint and current run use the
    same amp dtype. fp16 scaler state is meaningless for a bf16 run (and vice
    versa), so we skip it and let the scaler re-initialise cleanly.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    opt.load_state_dict(state["opt_state"])

    ckpt_dtype_name = state.get("amp_dtype", "float16")  # old checkpoints default to fp16
    cur_dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
    if ckpt_dtype_name == cur_dtype_name:
        scaler.load_state_dict(state["scaler_state"])
    else:
        print(
            f"  ⚠  amp_dtype changed ({ckpt_dtype_name} → {cur_dtype_name}): "
            "scaler state not restored (will re-initialise)."
        )

    if "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(state["global_step"]):
                scheduler.step()

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


def _build_checkpoint_state(model, opt, scheduler, scaler, epoch, global_step, cfg, amp_dtype) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "amp_dtype": "bfloat16" if amp_dtype == torch.bfloat16 else "float16",
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "cfg": {
            "d_model": cfg.d_model,
            "nhead": cfg.nhead,
            "num_layers": cfg.num_layers,
            "dim_feedforward": cfg.dim_feedforward,
            "dropout": cfg.dropout,
        },
    }


@torch.no_grad()
def _quick_eval(
    model: ChessTransformer,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_samples: int = 50_000,
) -> tuple[float, float]:
    """Evaluate on the first *max_samples* examples from *loader*."""
    model.eval()
    loss_sum = acc_sum = n = 0
    for x, y in loader:
        if n >= max_samples:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
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
    amp_dtype = _amp_dtype(device)
    print(f"{_now()} device={device}  amp_dtype={amp_dtype}")

    # ── data split ────────────────────────────────────────────────────────────
    all_paths = shard_paths(cfg.data_dir, cfg.shard_pattern, cfg.max_shards)
    held_out = cfg.val_shards + cfg.test_shards
    if len(all_paths) <= held_out:
        raise ValueError(
            f"Only {len(all_paths)} shard(s) found but val_shards + test_shards = {held_out}. "
            "Reduce held-out shards or provide more data."
        )
    train_cut = len(all_paths) - held_out
    val_cut = len(all_paths) - cfg.val_shards
    train_paths = all_paths[:train_cut]
    test_paths = all_paths[train_cut:val_cut]
    val_paths = all_paths[val_cut:]
    print(
        f"{_now()} shards: {len(train_paths)} train + {len(val_paths)} val"
        f" + {len(test_paths)} test  (pattern={cfg.shard_pattern!r})"
    )

    train_ds = MultiShard.from_paths(train_paths)
    val_ds = MultiShard.from_paths(val_paths)

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )

    steps_per_epoch = len(loader) // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    print(
        f"{_now()} train={len(train_ds):,}  val={len(val_ds):,} positions  "
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
    # GradScaler is only meaningful for fp16; disable it for bf16 (no overflow risk).
    # enabled=False makes all scaler calls no-ops, so the training loop is unchanged.
    scaler = torch.amp.GradScaler(
        device=device.type,
        init_scale=2**14,
        enabled=(amp_dtype == torch.float16),
    )
    scheduler = _build_scheduler(opt, cfg.warmup_steps, total_steps, cfg.min_lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    writer = SummaryWriter(log_dir=cfg.save_dir / "tb")

    # ── optional resume ───────────────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    best_val_acc = 0.0
    if cfg.resume:
        start_epoch, global_step = _load_checkpoint(
            cfg.resume, model, opt, scheduler, scaler, device, amp_dtype
        )
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
        skipped = 0

        for local_step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits, y) / cfg.grad_accum

            # ── NaN/Inf loss guard ────────────────────────────────────────────
            # Also skip batches with finite-but-extreme loss values (bad data).
            loss_val = loss.item()
            if not torch.isfinite(loss) or (
                cfg.loss_spike_threshold is not None
                and loss_val > cfg.loss_spike_threshold / cfg.grad_accum
            ):
                print(f"\n⚠  Bad loss ({loss_val * cfg.grad_accum:.2f}) at epoch "
                      f"{epoch} local_step {local_step}, skipping batch.")
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                skipped += 1
                continue

            scaler.scale(loss).backward()
            batch_loss += loss.item()

            if local_step % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                # ── NaN/Inf gradient guard ────────────────────────────────────
                grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                if not torch.isfinite(grad_norm):
                    print(f"\n⚠  Non-finite grad_norm at epoch {epoch} "
                          f"step {global_step}, skipping update.")
                    opt.zero_grad(set_to_none=True)
                    scaler.update()  # must call to keep scaler state consistent
                    batch_loss = 0.0
                    skipped += 1
                    continue

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    acc = (logits.argmax(1) == y).float().mean().item()
                    lr_now = opt.param_groups[0]["lr"]
                    scale_now = scaler.get_scale()
                    writer.add_scalar("train/loss", batch_loss, global_step)
                    writer.add_scalar("train/acc", acc, global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)
                    writer.add_scalar("train/grad_norm", grad_norm_val, global_step)
                    writer.add_scalar("train/scaler_scale", scale_now, global_step)
                    pbar.set_postfix(
                        step=global_step,
                        loss=f"{batch_loss:.4f}",
                        acc=f"{acc:.3f}",
                        lr=f"{lr_now:.2e}",
                        gnorm=f"{grad_norm_val:.2f}",
                    )

                if global_step % cfg.eval_every == 0:
                    eval_loss, eval_acc = _quick_eval(
                        model, val_loader, criterion, device, amp_dtype
                    )
                    writer.add_scalar("eval/loss", eval_loss, global_step)
                    writer.add_scalar("eval/acc", eval_acc, global_step)
                    print(
                        f"\n{_now()} EVAL  step={global_step}  "
                        f"loss={eval_loss:.4f}  acc={eval_acc:.4f}"
                    )
                    if eval_acc > best_val_acc:
                        best_val_acc = eval_acc
                        best_path = cfg.save_dir / "best.pt"
                        _save_checkpoint(
                            _build_checkpoint_state(
                                model, opt, scheduler, scaler, epoch, global_step, cfg, amp_dtype
                            ),
                            best_path,
                        )
                        print(f"{_now()} ★ new best val_acc={best_val_acc:.4f}  → {best_path}")

                batch_loss = 0.0

        # ── save checkpoint at end of every epoch ─────────────────────────────
        ckpt_path = cfg.save_dir / f"epoch_{epoch:03d}_step_{global_step:07d}.pt"
        _save_checkpoint(
            _build_checkpoint_state(
                model, opt, scheduler, scaler, epoch, global_step, cfg, amp_dtype
            ),
            ckpt_path,
        )
        if skipped:
            print(f"{_now()} epoch {epoch}: skipped {skipped} bad batches/steps")
        print(f"{_now()} saved {ckpt_path}")

    writer.close()
    print(f"{_now()} Training complete.  Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

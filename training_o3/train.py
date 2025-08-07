from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
from .config import TrainConfig
from .dataset import MultiShard
from .model import ChessTransformer
from .utils import save_checkpoint, timestamp
from torch.utils.tensorboard import SummaryWriter

def main():
    import argparse, os, random, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    ds = MultiShard(Path(args.data), cfg.shard_pattern)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model & optim
    model = ChessTransformer(
        cfg.d_model,
        cfg.nhead,
        cfg.num_layers,
        cfg.dim_feedforward,
        cfg.dropout,
    ).to(device)
    opt = AdamW(model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas)
    crit = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=cfg.save_dir / "tb")

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()

            if global_step % cfg.log_every == 0:
                acc = (logits.argmax(dim=1) == y).float().mean()
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/acc", acc.item(), global_step)
                print(f"{timestamp()} E{epoch} S{global_step} "
                      f"loss={loss:.4f} acc={acc:.4f}")

            global_step += 1

            if global_step % cfg.eval_every == 0:
                evaluate(model, loader, crit, device, writer, global_step)

        save_checkpoint(
            {"epoch": epoch,
             "model_state": model.state_dict(),
             "opt_state": opt.state_dict()},
            cfg.save_dir,
            epoch,
        )

def evaluate(model, loader, crit, device, writer, step):
    model.eval()
    loss_tot, acc_tot, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y).item()
            acc = (logits.argmax(dim=1) == y).float().sum().item()
            bs = y.size(0)
            loss_tot += loss * bs
            acc_tot += acc
            n += bs
            if n >= 50_000:        # quick dev eval
                break
    loss_avg, acc_avg = loss_tot / n, acc_tot / n
    writer.add_scalar("eval/loss", loss_avg, step)
    writer.add_scalar("eval/acc", acc_avg, step)
    print(f"Eval @ step {step}: loss={loss_avg:.4f} acc={acc_avg:.4f}")
    model.train()

if __name__ == "__main__":
    main()

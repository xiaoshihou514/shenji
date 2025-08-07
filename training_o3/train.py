"""Train the ChessTransformer on pre-processed Lichess shards."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm  # ← NEW

from .config import TrainConfig
from .dataset import MultiShard
from .model import ChessTransformer
from .utils import save_checkpoint, timestamp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------- data
    ds = MultiShard(Path(args.data), cfg.shard_pattern)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------- model
    model = ChessTransformer(
        cfg.d_model,
        cfg.nhead,
        cfg.num_layers,
        cfg.dim_feedforward,
        cfg.dropout,
    ).to(device)

    opt = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    crit = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=cfg.save_dir / "tb")

    # ------------------------------------------------------------------ train
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(  # ← NEW
            loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()

            if global_step % cfg.log_every == 0:
                acc = (logits.argmax(dim=1) == y).float().mean()
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/acc", acc.item(), global_step)

            # update progress-bar every batch
            pbar.set_postfix(
                step=global_step,
                loss=f"{loss.item():.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

            global_step += 1
            if global_step % cfg.eval_every == 0:
                evaluate(model, loader, crit, device, writer, global_step)

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
            },
            cfg.save_dir,
            epoch,
        )


def evaluate(model, loader, crit, device, writer, step):
    """Quick dev-set evaluation (first 50 k samples)."""
    model.eval()
    loss_tot = acc_tot = n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if n >= 50_000:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss_tot += crit(logits, y).item() * y.size(0)
            acc_tot += (logits.argmax(1) == y).float().sum().item()
            n += y.size(0)

    writer.add_scalar("eval/loss", loss_tot / n, step)
    writer.add_scalar("eval/acc", acc_tot / n, step)
    print(
        f"{timestamp()} EVAL step={step} loss={loss_tot / n:.4f} "
        f"acc={acc_tot / n:.4f}"
    )
    model.train()


if __name__ == "__main__":
    main()

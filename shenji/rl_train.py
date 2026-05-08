"""
rl_train.py – reinforce the supervised checkpoint with self-play and a BC anchor.

Usage:
    uv run shenji/rl_train.py --config shenji/rl_config.yaml

The training loop intentionally keeps the current policy architecture unchanged:
it loads the SL checkpoint weights, starts with a fresh optimiser, and fine-tunes
with a policy-gradient loss on self-play data plus a cross-entropy anchor on
supervised shards.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from shenji.board import MoveCodec
from shenji.dataset import MultiShard, shard_paths
from shenji.model import ChessTransformer
from shenji.rl_config import RLConfig
from shenji.selfplay import (
    ReplayDataset,
    concat_replays,
    evaluate_matches,
    generate_replay,
    save_replay,
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def _build_scheduler(
    opt: AdamW, warmup_steps: int, total_steps: int, min_lr: float
) -> SequentialLR:
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine = CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1), eta_min=min_lr)
    return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[max(warmup_steps, 1)])


def _restore_scheduler_fallback(scheduler: SequentialLR, global_step: int) -> None:
    """Rebuild state for old checkpoints without calling scheduler.step()."""
    state = scheduler.state_dict()
    warmup_steps = state["_milestones"][0]
    warm, cosine = state["_schedulers"]
    base_lr = warm["base_lrs"][0]
    eta_min = cosine["eta_min"]
    t_max = cosine["T_max"]

    if global_step < warmup_steps:
        factor = warm["start_factor"] + (
            (warm["end_factor"] - warm["start_factor"]) * global_step / max(warm["total_iters"], 1)
        )
        lr = base_lr * factor
        warm["last_epoch"] = global_step
        warm["_step_count"] = global_step + 1
        warm["_last_lr"] = [lr]
        cosine["last_epoch"] = -1
        cosine["_step_count"] = 1
        cosine["_last_lr"] = [lr]
    else:
        cosine_epoch = global_step - warmup_steps
        lr = eta_min + (
            (base_lr - eta_min) * (1 + torch.cos(torch.tensor(torch.pi * cosine_epoch / t_max)).item()) / 2
        )
        warm["last_epoch"] = max(warmup_steps - 1, 0)
        warm["_step_count"] = max(warmup_steps, 1)
        warm["_last_lr"] = [base_lr]
        cosine["last_epoch"] = cosine_epoch
        cosine["_step_count"] = cosine_epoch + 1
        cosine["_last_lr"] = [lr]

    state["last_epoch"] = global_step
    state["_last_lr"] = [lr]
    scheduler.load_state_dict(state)
    for group in scheduler.optimizer.param_groups:
        group["lr"] = lr


def _load_model_only(checkpoint: Path, model: ChessTransformer, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])


def _load_rl_checkpoint(
    path: Path,
    model: ChessTransformer,
    opt: AdamW,
    scheduler: SequentialLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[int, int, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    opt.load_state_dict(state["opt_state"])
    ckpt_dtype_name = state.get("amp_dtype", "float16")
    cur_dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
    if ckpt_dtype_name == cur_dtype_name:
        scaler.load_state_dict(state["scaler_state"])
    if "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    else:
        _restore_scheduler_fallback(scheduler, state["global_step"])
    print(
        f"Resumed RL from {path} (iter {state['iteration']}, step {state['global_step']})"
    )
    return state["iteration"], state["global_step"], float(state.get("best_eval_score", -1.0))


def _model_cfg(model: ChessTransformer) -> dict[str, int | float | bool]:
    layer = model.encoder.layers[0]
    return {
        "d_model": model.policy_head.in_features,
        "nhead": layer.self_attn.num_heads,
        "num_layers": len(model.encoder.layers),
        "dim_feedforward": layer.linear1.out_features,
        "dropout": 0.0,
        "rl": True,
    }


def _build_checkpoint_state(
    model: ChessTransformer,
    opt: AdamW,
    scheduler: SequentialLR,
    scaler: torch.amp.GradScaler,
    iteration: int,
    global_step: int,
    best_eval_score: float,
    amp_dtype: torch.dtype,
) -> dict:
    return {
        "iteration": iteration,
        "global_step": global_step,
        "best_eval_score": best_eval_score,
        "amp_dtype": "bfloat16" if amp_dtype == torch.bfloat16 else "float16",
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "cfg": _model_cfg(model),
    }


def _load_model_from_state(path: Path, device: torch.device, fallback: RLConfig) -> ChessTransformer:
    state = torch.load(path, map_location=device, weights_only=False)
    saved_cfg = state.get("cfg", {})
    model = ChessTransformer(
        d_model=saved_cfg.get("d_model", fallback.d_model),
        nhead=saved_cfg.get("nhead", fallback.nhead),
        num_layers=saved_cfg.get("num_layers", fallback.num_layers),
        dim_feedforward=saved_cfg.get("dim_feedforward", fallback.dim_feedforward),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _history_paths(save_dir: Path, limit: int) -> list[Path]:
    return sorted(save_dir.glob("iter_*.pt"))[-limit:]


def _opponent_counts(cfg: RLConfig) -> tuple[int, int, int]:
    total = cfg.selfplay_games_per_iter
    weights = [
        max(cfg.opponent_selfplay_frac, 0.0),
        max(cfg.opponent_history_frac, 0.0),
        max(cfg.opponent_engine_frac, 0.0),
    ]
    if sum(weights) == 0.0:
        return total, 0, 0
    raw = [w / sum(weights) * total for w in weights]
    counts = [int(v) for v in raw]
    while sum(counts) < total:
        idx = max(range(3), key=lambda i: raw[i] - counts[i])
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def _lambda_bc(cfg: RLConfig, global_step: int) -> float:
    if global_step >= cfg.lambda_bc_decay_steps:
        return cfg.lambda_bc_end
    ratio = global_step / max(cfg.lambda_bc_decay_steps, 1)
    return cfg.lambda_bc_start + ratio * (cfg.lambda_bc_end - cfg.lambda_bc_start)


def _legal_masks(fens: list[str], device: torch.device) -> torch.Tensor:
    masks = [MoveCodec.legal_mask(chess.Board(fen)) for fen in fens]
    return torch.stack(masks).to(device)


@torch.no_grad()
def _sl_eval(
    model: ChessTransformer,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_samples: int = 50_000,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = acc_sum = total = 0
    for x, y in loader:
        if total >= max_samples:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(x)
        logits = logits.float()
        loss_sum += criterion(logits, y).item() * y.size(0)
        acc_sum += (logits.argmax(1) == y).float().sum().item()
        total += y.size(0)
    model.train()
    return loss_sum / max(total, 1), acc_sum / max(total, 1)


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="RL fine-tuning for ChessTransformer")
    parser.add_argument("--config", required=True, help="Path to shenji/rl_config.yaml")
    args = parser.parse_args()

    cfg = RLConfig.load(args.config)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(device)

    bootstrap = torch.load(cfg.sl_checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = bootstrap.get("cfg", {})
    model = ChessTransformer(
        d_model=saved_cfg.get("d_model", cfg.d_model),
        nhead=saved_cfg.get("nhead", cfg.nhead),
        num_layers=saved_cfg.get("num_layers", cfg.num_layers),
        dim_feedforward=saved_cfg.get("dim_feedforward", cfg.dim_feedforward),
        dropout=0.0,
    ).to(device)

    opt = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scaler = torch.amp.GradScaler(
        device=device.type,
        init_scale=2**14,
        enabled=(amp_dtype == torch.float16),
    )
    total_steps_est = max(
        cfg.iterations
        * cfg.rl_epochs
        * max(cfg.selfplay_games_per_iter, 1)
        // max(cfg.batch_size * cfg.grad_accum, 1),
        1,
    )
    scheduler = _build_scheduler(opt, cfg.warmup_steps, total_steps_est, cfg.min_lr)

    start_iter = 1
    global_step = 0
    best_eval_score = -1.0
    if cfg.resume:
        start_iter, global_step, best_eval_score = _load_rl_checkpoint(
            cfg.resume, model, opt, scheduler, scaler, device, amp_dtype
        )
        start_iter += 1
    else:
        _load_model_only(cfg.sl_checkpoint, model, device)
        print(f"Loaded SL weights from {cfg.sl_checkpoint} into a fresh RL optimiser.")

    shard_list = shard_paths(cfg.sl_data_dir, cfg.sl_shard_pattern, cfg.sl_max_shards)
    if not shard_list:
        raise FileNotFoundError(f"No SL shards found in {cfg.sl_data_dir} matching {cfg.sl_shard_pattern}")
    if len(shard_list) <= cfg.sl_val_shards:
        train_paths = shard_list
        val_paths = shard_list[-1:]
    else:
        train_paths = shard_list[:-cfg.sl_val_shards]
        val_paths = shard_list[-cfg.sl_val_shards:]
    anchor_paths = train_paths[: max(cfg.sl_anchor_shards, 1)]
    anchor_ds = MultiShard.from_paths(anchor_paths)
    val_ds = MultiShard.from_paths(val_paths)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    anchor_loader = DataLoader(
        anchor_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    anchor_iter = _cycle(anchor_loader)
    bc_criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=cfg.save_dir / "tb")

    model.train()
    for iteration in range(start_iter, cfg.iterations + 1):
        print(f"{_now()} RL iteration {iteration}/{cfg.iterations}")
        n_self, n_history, n_engine = _opponent_counts(cfg)
        replay_parts: list[dict[str, object]] = []

        if n_self > 0:
            replay_self, summary_self = generate_replay(
                model,
                device,
                games=n_self,
                batch_size=cfg.selfplay_batch_size,
                max_moves=cfg.selfplay_max_moves,
                gamma=cfg.gamma,
                temperature_open=cfg.temperature_open,
                temperature_mid=cfg.temperature_mid,
                temperature_switch_ply=cfg.temperature_switch_ply,
                source="self",
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_eps=cfg.dirichlet_eps,
                repetition_penalty=cfg.repetition_penalty,
            )
            replay_parts.append(replay_self)
            print(f"{_now()} self-play summary: {json.dumps(summary_self)}")

        history_pool = _history_paths(cfg.save_dir, cfg.history_pool_size)
        if n_history > 0 and history_pool:
            history_ckpt = random.choice(history_pool)
            history_model = _load_model_from_state(history_ckpt, device, cfg)
            replay_hist, summary_hist = generate_replay(
                model,
                device,
                games=n_history,
                batch_size=cfg.selfplay_batch_size,
                max_moves=cfg.selfplay_max_moves,
                gamma=cfg.gamma,
                temperature_open=cfg.temperature_open,
                temperature_mid=cfg.temperature_mid,
                temperature_switch_ply=cfg.temperature_switch_ply,
                source="history",
                opponent_model=history_model,
                repetition_penalty=cfg.repetition_penalty,
            )
            replay_parts.append(replay_hist)
            print(f"{_now()} history summary ({history_ckpt.name}): {json.dumps(summary_hist)}")
        elif n_history > 0:
            print(f"{_now()} history pool empty; folding history games into self-play.")
            replay_self, _ = generate_replay(
                model,
                device,
                games=n_history,
                batch_size=cfg.selfplay_batch_size,
                max_moves=cfg.selfplay_max_moves,
                gamma=cfg.gamma,
                temperature_open=cfg.temperature_open,
                temperature_mid=cfg.temperature_mid,
                temperature_switch_ply=cfg.temperature_switch_ply,
                source="self",
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_eps=cfg.dirichlet_eps,
                repetition_penalty=cfg.repetition_penalty,
            )
            replay_parts.append(replay_self)

        if n_engine > 0 and cfg.engine_path:
            replay_engine, summary_engine = generate_replay(
                model,
                device,
                games=n_engine,
                batch_size=cfg.selfplay_batch_size,
                max_moves=cfg.selfplay_max_moves,
                gamma=cfg.gamma,
                temperature_open=cfg.temperature_open,
                temperature_mid=cfg.temperature_mid,
                temperature_switch_ply=cfg.temperature_switch_ply,
                source="engine",
                engine_path=cfg.engine_path,
                engine_depth=cfg.engine_depth,
                repetition_penalty=cfg.repetition_penalty,
            )
            replay_parts.append(replay_engine)
            print(f"{_now()} engine summary: {json.dumps(summary_engine)}")
        elif n_engine > 0:
            print(f"{_now()} engine_path not set; folding engine games into self-play.")
            replay_self, _ = generate_replay(
                model,
                device,
                games=n_engine,
                batch_size=cfg.selfplay_batch_size,
                max_moves=cfg.selfplay_max_moves,
                gamma=cfg.gamma,
                temperature_open=cfg.temperature_open,
                temperature_mid=cfg.temperature_mid,
                temperature_switch_ply=cfg.temperature_switch_ply,
                source="self",
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_eps=cfg.dirichlet_eps,
                repetition_penalty=cfg.repetition_penalty,
            )
            replay_parts.append(replay_self)

        replay = concat_replays(replay_parts)
        replay_path = cfg.save_dir / "replays" / f"iter_{iteration:03d}.npz"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        save_replay(replay_path, replay)
        rl_ds = ReplayDataset(replay)
        rl_loader = DataLoader(
            rl_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        for rl_epoch in range(1, cfg.rl_epochs + 1):
            pbar = tqdm(
                rl_loader,
                desc=f"RL iter {iteration}/{cfg.iterations} epoch {rl_epoch}/{cfg.rl_epochs}",
                dynamic_ncols=True,
            )
            opt.zero_grad(set_to_none=True)
            running = {"loss": 0.0, "pg": 0.0, "bc": 0.0, "entropy": 0.0, "return": 0.0}
            for local_step, (x_rl, y_rl, returns_rl, fens_rl) in enumerate(pbar, start=1):
                x_bc, y_bc = next(anchor_iter)
                x_rl = x_rl.to(device, non_blocking=True)
                y_rl = y_rl.to(device, non_blocking=True)
                returns_rl = returns_rl.to(device, non_blocking=True)
                x_bc = x_bc.to(device, non_blocking=True)
                y_bc = y_bc.to(device, non_blocking=True)
                legal_mask = _legal_masks(list(fens_rl), device)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    logits_rl = model(x_rl)
                    logits_bc = model(x_bc)

                logits_rl = logits_rl.float()
                logits_bc = logits_bc.float()
                masked_logits = logits_rl.masked_fill(~legal_mask, -torch.inf)
                log_probs = torch.log_softmax(masked_logits, dim=1)
                probs = torch.softmax(masked_logits, dim=1)
                chosen_logp = log_probs.gather(1, y_rl.unsqueeze(1)).squeeze(1)
                advantages = returns_rl - returns_rl.mean()
                pg_loss = -(chosen_logp * advantages).mean()
                entropy = -(probs * log_probs.masked_fill(~legal_mask, 0.0)).sum(dim=1).mean()
                bc_loss = bc_criterion(logits_bc, y_bc)
                lambda_bc = _lambda_bc(cfg, global_step)
                total_loss = pg_loss + lambda_bc * bc_loss - cfg.entropy_coeff * entropy

                loss_val = float(total_loss.item())
                if not torch.isfinite(total_loss) or (
                    cfg.loss_spike_threshold is not None and loss_val > cfg.loss_spike_threshold
                ):
                    print(
                        f"\n⚠  Bad RL loss ({loss_val:.2f}) at iter {iteration} "
                        f"local_step {local_step}, skipping batch."
                    )
                    opt.zero_grad(set_to_none=True)
                    continue

                loss = total_loss / cfg.grad_accum
                scaler.scale(loss).backward()
                running["loss"] += total_loss.item()
                running["pg"] += pg_loss.item()
                running["bc"] += bc_loss.item()
                running["entropy"] += entropy.item()
                running["return"] += returns_rl.mean().item()

                should_step = local_step % cfg.grad_accum == 0 or local_step == len(rl_loader)
                if not should_step:
                    continue

                scaler.unscale_(opt)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                if not torch.isfinite(grad_norm):
                    print(
                        f"\n⚠  Non-finite RL grad_norm at iter {iteration} "
                        f"step {global_step}, skipping update."
                    )
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    lr_now = opt.param_groups[0]["lr"]
                    writer.add_scalar("rl/loss", running["loss"], global_step)
                    writer.add_scalar("rl/pg_loss", running["pg"], global_step)
                    writer.add_scalar("rl/bc_loss", running["bc"], global_step)
                    writer.add_scalar("rl/entropy", running["entropy"], global_step)
                    writer.add_scalar("rl/return_mean", running["return"], global_step)
                    writer.add_scalar("rl/lambda_bc", lambda_bc, global_step)
                    writer.add_scalar("rl/lr", lr_now, global_step)
                    writer.add_scalar("rl/grad_norm", grad_norm_val, global_step)
                    pbar.set_postfix(
                        step=global_step,
                        loss=f"{running['loss']:.3f}",
                        pg=f"{running['pg']:.3f}",
                        bc=f"{running['bc']:.3f}",
                        ent=f"{running['entropy']:.3f}",
                        ret=f"{running['return']:.3f}",
                        lr=f"{lr_now:.2e}",
                    )
                    running = {"loss": 0.0, "pg": 0.0, "bc": 0.0, "entropy": 0.0, "return": 0.0}

        sl_loss, sl_acc = _sl_eval(model, val_loader, device, amp_dtype)
        writer.add_scalar("eval/sl_loss", sl_loss, iteration)
        writer.add_scalar("eval/sl_acc", sl_acc, iteration)
        print(f"{_now()} SL check: loss={sl_loss:.4f} acc={sl_acc:.4f}")

        eval_score = -1.0
        if cfg.eval_every_iter > 0 and iteration % cfg.eval_every_iter == 0:
            if cfg.engine_path:
                series = evaluate_matches(
                    model,
                    device,
                    games=cfg.eval_games,
                    batch_size=min(cfg.selfplay_batch_size, cfg.eval_games),
                    max_moves=cfg.selfplay_max_moves,
                    engine_path=cfg.engine_path,
                    engine_depth=cfg.engine_depth,
                )
                score = (series["wins"] + 0.5 * series["draws"]) / max(cfg.eval_games, 1)
                eval_score = score
                writer.add_scalar("eval/engine_score", score, iteration)
                writer.add_scalar("eval/engine_wins", series["wins"], iteration)
                writer.add_scalar("eval/engine_draws", series["draws"], iteration)
                writer.add_scalar("eval/engine_losses", series["losses"], iteration)
                print(f"{_now()} engine eval: {json.dumps(series)} score={score:.4f}")
            else:
                eval_score = sl_acc

        ckpt_path = cfg.save_dir / f"iter_{iteration:03d}.pt"
        _save_checkpoint(
            _build_checkpoint_state(
                model, opt, scheduler, scaler, iteration, global_step, best_eval_score, amp_dtype
            ),
            ckpt_path,
        )
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            _save_checkpoint(
                _build_checkpoint_state(
                    model, opt, scheduler, scaler, iteration, global_step, best_eval_score, amp_dtype
                ),
                cfg.save_dir / "best.pt",
            )
            print(f"{_now()} ★ new best RL eval score={best_eval_score:.4f}")

    print(f"{_now()} RL training complete. Best eval score={best_eval_score:.4f}")


if __name__ == "__main__":
    main()

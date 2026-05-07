"""config.py – frozen training configuration loaded from YAML."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["TrainConfig"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # ── data ──────────────────────────────────────────────────────────────────
    data_dir: Path
    shard_pattern: str
    num_workers: int

    # ── model ─────────────────────────────────────────────────────────────────
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float

    # ── optimiser ─────────────────────────────────────────────────────────────
    lr: float
    min_lr: float
    weight_decay: float
    betas: tuple[float, float]
    clip_grad_norm: float
    warmup_steps: int

    # ── training ──────────────────────────────────────────────────────────────
    batch_size: int
    grad_accum: int          # gradient accumulation steps
    epochs: int
    label_smoothing: float   # cross-entropy label smoothing coefficient (0 = off)

    # ── logging / checkpointing ───────────────────────────────────────────────
    log_every: int           # log every N optimiser steps
    eval_every: int          # eval every N optimiser steps
    save_dir: Path

    # ── optional ──────────────────────────────────────────────────────────────
    max_shards: int | None = None  # total shards to consider (None = all)
    val_shards: int = 1            # last N shards held out for validation
    resume: Path | None = None
    # Skip a batch if loss exceeds this value (catches bad-data spikes that are
    # finite but pathologically large; None = disabled).
    loss_spike_threshold: float | None = 20.0

    @staticmethod
    def load(cfg_path: str | Path) -> "TrainConfig":
        import yaml

        raw: dict[str, Any] = yaml.safe_load(Path(cfg_path).read_text())
        raw["data_dir"] = Path(raw["data_dir"]).expanduser()
        raw["save_dir"] = Path(raw["save_dir"]).expanduser()
        raw["betas"] = tuple(raw["betas"])
        if raw.get("resume"):
            raw["resume"] = Path(raw["resume"]).expanduser()
        else:
            raw.pop("resume", None)
        raw.setdefault("max_shards", None)
        raw.setdefault("val_shards", 1)
        raw.setdefault("label_smoothing", 0.1)
        raw.setdefault("loss_spike_threshold", 20.0)
        return TrainConfig(**raw)

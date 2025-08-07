from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # data
    data_dir: Path
    shard_pattern: str
    batch_size: int
    num_workers: int
    max_games: int | None
    # model
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_len: int
    # optim
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    clip_grad_norm: float
    # training
    epochs: int
    log_every: int
    eval_every: int
    save_dir: Path

    @staticmethod
    def load(cfg_path: str | Path) -> "TrainConfig":
        import yaml

        raw = yaml.safe_load(Path(cfg_path).read_text())
        raw["data_dir"] = Path(raw["data_dir"]).expanduser()
        raw["save_dir"] = Path(raw["save_dir"]).expanduser()
        return TrainConfig(**raw)

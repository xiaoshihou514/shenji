"""rl_config.py – configuration loader for reinforcement learning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RLConfig"]


@dataclass(frozen=True, slots=True)
class RLConfig:
    # ── bootstrap / data ─────────────────────────────────────────────────────
    sl_checkpoint: Path
    sl_data_dir: Path
    save_dir: Path
    sl_shard_pattern: str = "shard_*.npz"
    sl_anchor_shards: int = 2
    sl_val_shards: int = 1
    sl_max_shards: int | None = None
    num_workers: int = 4

    # ── model fallback (actual shape is loaded from checkpoint cfg) ──────────
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.0

    # ── self-play ─────────────────────────────────────────────────────────────
    iterations: int = 10
    rl_epochs: int = 1
    selfplay_games_per_iter: int = 500
    selfplay_batch_size: int = 128
    selfplay_max_moves: int = 300
    gamma: float = 0.99
    repetition_penalty: float = 0.0
    temperature_open: float = 1.0
    temperature_mid: float = 1.0
    temperature_switch_ply: int = 10
    dirichlet_alpha: float | None = None
    dirichlet_eps: float = 0.0
    material_reward_scale: float = 0.02
    material_delta_clip: float | None = 3.0
    terminal_reward_scale: float = 1.0

    # ── opponent mix ──────────────────────────────────────────────────────────
    opponent_selfplay_frac: float = 0.8
    opponent_history_frac: float = 0.2
    opponent_engine_frac: float = 0.0
    history_pool_size: int = 5
    engine_path: Path | None = None
    engine_depth: int = 5

    # ── optimisation ──────────────────────────────────────────────────────────
    batch_size: int = 128
    grad_accum: int = 8
    lr: float = 5.0e-6
    min_lr: float = 1.0e-6
    weight_decay: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    clip_grad_norm: float = 1.0
    warmup_steps: int = 500
    lambda_bc_start: float = 0.15
    lambda_bc_end: float = 0.05
    lambda_bc_decay_steps: int = 10_000
    entropy_coeff: float = 0.02
    ppo_clip_epsilon: float = 0.2
    gradient_checkpointing: bool = True
    loss_spike_threshold: float | None = 20.0
    collapse_previous_bc_ratio: float = 2.0
    collapse_baseline_bc_ratio: float = 3.0

    # ── logging / checkpointing ───────────────────────────────────────────────
    log_every: int = 20
    eval_every_iter: int = 1
    eval_games: int = 50
    seed: int = 42
    resume: Path | None = None

    @staticmethod
    def load(cfg_path: str | Path) -> "RLConfig":
        import yaml

        raw: dict[str, Any] = yaml.safe_load(Path(cfg_path).read_text())
        for key in ("sl_checkpoint", "sl_data_dir", "save_dir", "engine_path", "resume"):
            if raw.get(key):
                raw[key] = Path(raw[key]).expanduser()
            else:
                raw.pop(key, None)

        if "sl_checkpoint" not in raw:
            raise ValueError("rl config requires 'sl_checkpoint'")
        if "sl_data_dir" not in raw:
            raise ValueError("rl config requires 'sl_data_dir'")
        if "save_dir" not in raw:
            raise ValueError("rl config requires 'save_dir'")

        raw["betas"] = tuple(raw.get("betas", (0.9, 0.95)))

        raw.setdefault("sl_shard_pattern", "shard_*.npz")
        raw.setdefault("sl_anchor_shards", 2)
        raw.setdefault("sl_val_shards", 1)
        raw.setdefault("sl_max_shards", None)
        raw.setdefault("num_workers", 4)
        raw.setdefault("d_model", 768)
        raw.setdefault("nhead", 12)
        raw.setdefault("num_layers", 12)
        raw.setdefault("dim_feedforward", 3072)
        raw.setdefault("dropout", 0.0)
        raw.setdefault("iterations", 10)
        raw.setdefault("rl_epochs", 1)
        raw.setdefault("selfplay_games_per_iter", 500)
        raw.setdefault("selfplay_batch_size", 128)
        raw.setdefault("selfplay_max_moves", 300)
        raw.setdefault("gamma", 0.99)
        raw.setdefault("repetition_penalty", 0.0)
        raw.setdefault("temperature_open", 1.0)
        raw.setdefault("temperature_mid", 1.0)
        raw.setdefault("temperature_switch_ply", 10)
        raw.setdefault("dirichlet_alpha", None)
        raw.setdefault("dirichlet_eps", 0.0)
        raw.setdefault("material_reward_scale", 0.02)
        raw.setdefault("material_delta_clip", 3.0)
        raw.setdefault("terminal_reward_scale", 1.0)
        raw.setdefault("opponent_selfplay_frac", 0.8)
        raw.setdefault("opponent_history_frac", 0.2)
        raw.setdefault("opponent_engine_frac", 0.0)
        raw.setdefault("history_pool_size", 5)
        raw.setdefault("engine_depth", 5)
        raw.setdefault("batch_size", 128)
        raw.setdefault("grad_accum", 8)
        raw.setdefault("lr", 5.0e-6)
        raw.setdefault("min_lr", 1.0e-6)
        raw.setdefault("weight_decay", 1.0e-4)
        raw.setdefault("clip_grad_norm", 1.0)
        raw.setdefault("warmup_steps", 500)
        raw.setdefault("lambda_bc_start", 0.15)
        raw.setdefault("lambda_bc_end", 0.05)
        raw.setdefault("lambda_bc_decay_steps", 10_000)
        raw.setdefault("entropy_coeff", 0.02)
        raw.setdefault("ppo_clip_epsilon", 0.2)
        raw.setdefault("gradient_checkpointing", True)
        raw.setdefault("loss_spike_threshold", 20.0)
        raw.setdefault("collapse_previous_bc_ratio", 2.0)
        raw.setdefault("collapse_baseline_bc_ratio", 3.0)
        raw.setdefault("log_every", 20)
        raw.setdefault("eval_every_iter", 1)
        raw.setdefault("eval_games", 50)
        raw.setdefault("seed", 42)
        return RLConfig(**raw)

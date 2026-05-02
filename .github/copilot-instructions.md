# Shenji – Copilot Instructions

Shenji is a transformer-based chess AI. The `shenji/` Python package handles preprocessing, training, evaluation, and interactive play.

## Project Layout

```
shenji/             # main Python package (run everything from repo root with -m)
├── board.py        # BoardEncoder + MoveCodec (AlphaZero 4672-class move scheme)
├── model.py        # ChessTransformer (pre-norm encoder + per-square policy head)
├── dataset.py      # NPZShard, MultiShard (memory-mapped .npz shards)
├── config.py       # TrainConfig frozen dataclass + YAML loader
├── config.yaml     # default hyperparameters
├── preprocess.py   # Lichess .pgn.zst → sharded .npz
├── train.py        # training loop (fp16 AMP, AdamW + cosine LR, grad accum)
├── evaluate.py     # top-1/top-5 accuracy
├── play.py         # interactive terminal play (UCI input + SVG rendering)
└── pyproject.toml  # uv-based dependencies
```

## Commands

All commands run from the **repo root**. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```sh
# Install dependencies (PyTorch must be installed separately for the right CUDA version)
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
cd shenji && uv sync

# Preprocess a Lichess .pgn.zst archive into sharded .npz files
python -m shenji.preprocess \
    --pgn-archive /data/lichess_db_standard_rated_2025-05.pgn.zst \
    --out-dir ./data \
    --shard-size 1_000_000 --max-games 5_000_000

# Train
python -m shenji.train --config shenji/config.yaml

# Monitor training
tensorboard --logdir checkpoints/tb

# Evaluate a checkpoint
python -m shenji.evaluate \
    --config shenji/config.yaml \
    --checkpoint checkpoints/epoch_020_step_0123456.pt \
    --data ./data

# Play interactively against the model
python -m shenji.play --checkpoint checkpoints/epoch_020_step_0123456.pt

# Lint / format
black shenji/
isort shenji/
```

## Architecture

The pipeline is: **preprocess → train → evaluate / play**.

### Board representation (`board.py`)
`BoardEncoder.encode(board)` produces a **71-token `LongTensor`** per position:
- Tokens 0–63: piece ID per square (a1=sq0 … h8=sq63)
  — `0`=empty, `1–6`=white {P N B R Q K}, `7–12`=black {p n b r q k}
- Token 64: side to move (0=white, 1=black)
- Tokens 65–68: castling rights [WK, WQ, BK, BQ] (each 0 or 1)
- Token 69: en-passant file (0–7, or 8 = no EP)
- Token 70: halfmove-clock bucket `floor(min(clock, 50) / 5)` ∈ [0, 10]

`BoardEncoder.encode_np()` returns a `uint8` numpy array for bulk preprocessing.

### Move encoding (`board.py`)
`MoveCodec` uses the **AlphaZero 4672-class scheme**: `idx = from_sq * 73 + plane`
- Planes 0–55: queen-style moves (8 directions × 7 distances)
- Planes 56–63: knight moves (8 orientations)
- Planes 64–72: pawn underpromotions (3 files × 3 pieces: R/B/N)
- Queen promotions share the same plane as the corresponding queen-direction move.

Tables are precomputed at import time. Key methods:
- `MoveCodec.to_idx(move)` — encode
- `MoveCodec.to_move(idx, board)` — decode (needs board for queen-promo inference)
- `MoveCodec.legal_mask(board)` → `BoolTensor(4672,)`

### Model (`model.py`)
`ChessTransformer(B, 71) → (B, 4672)`:
- `BoardEmbedding`: separate `nn.Embedding` tables per token type (piece, side-to-move, castling, en-passant, halfmove) + learned positional embedding
- Pre-norm `TransformerEncoder` (GELU, `batch_first=True`, `norm_first=True`)
- Policy head: `Linear(d_model, 73)` applied to the 64 square tokens → reshape to `(B, 4672)`
- Default: d_model=768, nhead=12, num_layers=12, ff=3072 ≈ 85M parameters

### Data loading (`dataset.py`)
`MultiShard` concatenates sharded `.npz` files via prefix-sum + binary search. Shards are memory-mapped (`mmap_mode="r"`). Each shard: `x: uint8 (N, 71)`, `y: int16 (N,)` (move index 0–4671).

### Training (`train.py`)
- fp16 AMP (`torch.amp.autocast`), gradient scaler
- AdamW, cosine LR decay with linear warmup, gradient clipping
- Gradient accumulation (`grad_accum` steps, effective batch = `batch_size × grad_accum`)
- Checkpoints: `epoch_NNN_step_NNNNNNN.pt` — contain `model_state`, `opt_state`, `scaler_state`, `cfg` dict

## Key Conventions

- **Run with `-m`** from the repo root (e.g. `python -m shenji.train`), not by running files directly.
- **ELO filter**: preprocessing keeps games where **both** players have Elo ≥ 2000 (`--min-elo`, default 2000). Lower ELO games are discarded.
- **Inference**: always call `MoveCodec.legal_mask(board)` and set `logits[~mask] = -torch.inf` before sampling — the model is not trained with explicit legality masking.
- Shard naming: `shard_NNNN.npz` (4-digit zero-padded). `config.yaml`'s `shard_pattern` must match.
- Checkpoint `cfg` dict is embedded in every `.pt` file so the model can be reconstructed without a config file (used in `evaluate.py` and `play.py`).
- Board display in `play.py` writes `board.svg` and renders via kitty (`kitty +kitten icat`) or wezterm (`wezterm imgcat`); falls back to ASCII if neither is available.

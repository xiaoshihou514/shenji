"""
preprocess.py – stream-parse Lichess PGN dumps into sharded .npz archives.

Usage:
    python -m shenji.preprocess \\
        --pgn-archive /data/lichess_db_standard_rated_2025-05.pgn.zst \\
        --out-dir ./data \\
        --shard-size 1_000_000 \\
        --max-games 5_000_000

Each output shard ``shard_NNNN.npz`` contains:
    x : uint8  (N, 71)   – encoded board state (see board.py)
    y : int16  (N,)      – AlphaZero move index (0 … 4671)

Only games where **both** players have Elo ≥ 2000 are included, so the
model learns from stronger play.
"""

import argparse
import io
from pathlib import Path

import chess.pgn
import numpy as np
import zstandard as zstd
from tqdm import tqdm

from shenji.board import BoardEncoder, MoveCodec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Lichess PGN → .npz shards")
    p.add_argument(
        "--pgn-archive",
        required=True,
        type=Path,
        help="Path to Lichess .pgn or .pgn.zst archive",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write shard_NNNN.npz files",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=1_000_000,
        help="Positions per shard (default: 1 000 000)",
    )
    p.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after this many qualifying games (default: unlimited)",
    )
    p.add_argument(
        "--min-elo",
        type=int,
        default=2000,
        help="Minimum Elo for both players (default: 2000)",
    )
    return p.parse_args()


def _open_pgn(path: Path):
    """Return a text-mode file-like object, decompressing .zst if needed."""
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        fh = open(path, "rb")
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _flush_shard(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    out_dir: Path,
    shard_no: int,
) -> None:
    path = out_dir / f"shard_{shard_no:04d}.npz"
    np.savez_compressed(path, x=np.stack(xs), y=np.array(ys, dtype=np.int16))
    print(f"  saved {path}  ({len(xs):,} positions)")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    buf = _open_pgn(args.pgn_archive)

    games_seen = 0
    games_kept = 0
    shard_no = 0
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    pbar = tqdm(desc="games", unit="game")
    while True:
        game = chess.pgn.read_game(buf)
        if game is None:
            break
        games_seen += 1
        pbar.update(1)

        # ── ELO filter: keep only strong games ───────────────────────────────
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
        except ValueError:
            continue
        if white_elo < args.min_elo or black_elo < args.min_elo:
            continue

        # ── encode every position in the game ────────────────────────────────
        board = game.board()
        for move in game.mainline_moves():
            try:
                y = MoveCodec.to_idx(move)
            except ValueError:
                # Should never happen for legal moves; skip defensively
                board.push(move)
                continue
            xs.append(BoardEncoder.encode_np(board))
            ys.append(y)
            board.push(move)

            if len(xs) >= args.shard_size:
                _flush_shard(xs, ys, args.out_dir, shard_no)
                shard_no += 1
                xs, ys = [], []

        games_kept += 1
        pbar.set_postfix(
            kept=games_kept, positions=len(xs) + shard_no * args.shard_size
        )

        if args.max_games and games_kept >= args.max_games:
            break

    pbar.close()

    if xs:
        _flush_shard(xs, ys, args.out_dir, shard_no)

    print(
        f"\nDone. {games_seen:,} games scanned, "
        f"{games_kept:,} kept (both Elo ≥ {args.min_elo}). "
        f"{shard_no + (1 if xs else 0)} shards written to {args.out_dir}."
    )


if __name__ == "__main__":
    main()

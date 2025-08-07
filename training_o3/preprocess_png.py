#!/usr/bin/env python
"""
Stream-parse a compressed PGN dump and build NumPy shards.

Typical usage:
$ python scripts/preprocess_pgn.py \
      --pgn-archive /data/lichess_db_2025-07.pgn.zst \
      --out-dir ./data \
      --shard-size 1_000_000 --max-games 5_000_000
"""
from __future__ import annotations
import argparse
import zstandard as zstd
import chess.pgn
import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from training_o3.encoder import BoardEncoder
from training_o3.move_vocab import MoveVocab

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn-archive", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--max-games", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(args.pgn_archive, "rb") as fh, dctx.stream_reader(fh) as reader:
        buf = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        game_no, shard_no = 0, 0
        xs, ys = [], []

        for game in tqdm(iter(lambda: chess.pgn.read_game(buf), None)):
            if game is None or (args.max_games and game_no >= args.max_games):
                break
            board = game.board()
            for move in game.mainline_moves():
                x = BoardEncoder.encode(board)          # (65,)
                y = MoveVocab.to_idx(move)
                xs.append(x.numpy().astype(np.uint8))
                ys.append(np.int32(y))
                board.push(move)

                if len(xs) >= args.shard_size:
                    dump(xs, ys, args.out_dir, shard_no)
                    shard_no += 1
                    xs, ys = [], []

            game_no += 1

        if xs:
            dump(xs, ys, args.out_dir, shard_no)

def dump(xs, ys, out_dir: Path, shard_no: int):
    path = out_dir / f"shard_{shard_no:04d}.npz"
    np.savez_compressed(path, x=np.stack(xs), y=np.array(ys))
    print(f"Saved {path}  ({len(xs):,} samples)")

if __name__ == "__main__":
    main()

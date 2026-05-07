"""
evaluate_rl.py – play a fixed match series for RL checkpoints.

Usage:
    uv run shenji/evaluate_rl.py --checkpoint rl_checkpoints/best.pt --engine-path /usr/bin/stockfish

Supports:
    1. model vs external UCI engine
    2. model vs another Shenji checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from shenji.selfplay import evaluate_matches, load_policy_model


def _performance_rating(score: float, opponent_elo: float | None) -> float | None:
    if opponent_elo is None or score <= 0.0 or score >= 1.0:
        return None
    return opponent_elo - 400.0 * math.log10((1.0 / score) - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RL checkpoint by playing matches")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--engine-depth", type=int, default=5)
    parser.add_argument("--opponent-elo", type=float, default=None)
    args = parser.parse_args()

    if args.opponent_checkpoint and args.engine_path:
        raise SystemExit("Use either --opponent-checkpoint or --engine-path, not both.")
    if not args.opponent_checkpoint and not args.engine_path:
        raise SystemExit("Provide --opponent-checkpoint or --engine-path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy_model(args.checkpoint, device)
    opponent_model = (
        load_policy_model(args.opponent_checkpoint, device)
        if args.opponent_checkpoint
        else None
    )
    summary = evaluate_matches(
        model,
        device,
        games=args.games,
        batch_size=min(args.batch_size, args.games),
        max_moves=args.max_moves,
        opponent_model=opponent_model,
        engine_path=args.engine_path,
        engine_depth=args.engine_depth,
    )
    score = (summary["wins"] + 0.5 * summary["draws"]) / max(args.games, 1)
    result = {
        "checkpoint": str(args.checkpoint),
        "games": args.games,
        "wins": int(summary["wins"]),
        "draws": int(summary["draws"]),
        "losses": int(summary["losses"]),
        "score": round(score, 6),
        "avg_plies": round(summary["avg_plies"], 2),
        "performance_rating": (
            round(_performance_rating(score, args.opponent_elo), 2)
            if _performance_rating(score, args.opponent_elo) is not None
            else None
        ),
    }
    if args.opponent_checkpoint:
        result["opponent_checkpoint"] = str(args.opponent_checkpoint)
    if args.engine_path:
        result["engine_path"] = str(args.engine_path)
        result["engine_depth"] = args.engine_depth
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

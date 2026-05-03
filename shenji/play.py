"""
play.py – interactive terminal chess against a trained ChessTransformer.

Usage:
    python -m shenji.play \\
        --checkpoint checkpoints/epoch_020_step_0123456.pt \\
        [--topk 1]          # 1 = greedy, >1 = sample from top-k

Board rendering:
    Requires either kitty (``TERM=xterm-kitty``) or WezTerm (``wezterm imgcat``).
    The SVG is written to ``board.svg`` in the current directory and displayed
    inline via the terminal's image protocol.

Move input:
    Enter moves in UCI notation, e.g. ``e2e4``, ``g8f6``, ``e7e8q``.
    Type ``quit`` or ``exit`` to end the session.
"""

import argparse
import shutil
import subprocess
import sys
from os import getenv
from pathlib import Path

import chess
import chess.svg
import torch

from shenji.board import BoardEncoder, MoveCodec
from shenji.model import ChessTransformer


def _detect_renderer() -> str | None:
    """Return 'kitty', 'wezterm', or None if no image renderer is available."""
    if getenv("TERM") == "xterm-kitty" and shutil.which("kitty"):
        return "kitty"
    if shutil.which("wezterm"):
        return "wezterm"
    return None


def _render_board(
    board: chess.Board, arrow: chess.svg.Arrow | None, renderer: str | None
) -> None:
    arrows = [arrow] if arrow else []
    svg_text = chess.svg.board(board, arrows=arrows, size=400)
    Path("board.svg").write_text(svg_text)
    if renderer == "kitty":
        subprocess.run(["kitty", "+kitten", "icat", "board.svg"], check=False)
    elif renderer == "wezterm":
        subprocess.run(["wezterm", "imgcat", "board.svg"], check=False)
    else:
        # ASCII fallback
        print(board)


def _pick_move(
    model: ChessTransformer,
    board: chess.Board,
    device: torch.device,
    topk: int,
) -> chess.Move:
    """Return the model's chosen move, masking illegal moves."""
    x = BoardEncoder.encode(board).unsqueeze(0).to(device)  # (1, 71)
    mask = MoveCodec.legal_mask(board).to(device)  # (4672,)

    with torch.no_grad():
        logits = model(x).squeeze(0)  # (4672,)

    logits[~mask] = -torch.inf

    if topk == 1:
        idx = int(logits.argmax())
    else:
        top_vals, top_idxs = logits.topk(min(topk, int(mask.sum())))
        probs = top_vals.softmax(0)
        chosen = int(torch.multinomial(probs, 1))
        idx = int(top_idxs[chosen])

    return MoveCodec.to_move(idx, board)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play chess against ChessTransformer")
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to a .pt checkpoint file"
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="Greedy (1) or top-k sampling (default: 1)"
    )
    parser.add_argument(
        "--play-as",
        choices=["white", "black"],
        default="white",
        help="Human plays as white or black (default: white)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("cfg", {})
    model = ChessTransformer(
        d_model=saved_cfg.get("d_model", 768),
        nhead=saved_cfg.get("nhead", 12),
        num_layers=saved_cfg.get("num_layers", 12),
        dim_feedforward=saved_cfg.get("dim_feedforward", 3072),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(
        f"You play as {'White' if args.play_as == 'white' else 'Black'}.  Type 'quit' to exit.\n"
    )

    renderer = _detect_renderer()
    if renderer is None:
        print(
            "Note: no image renderer detected (kitty / wezterm). Falling back to ASCII.\n"
        )

    human_color = chess.WHITE if args.play_as == "white" else chess.BLACK
    board = chess.Board()
    _render_board(board, None, renderer)

    while not board.is_game_over():
        if board.turn == human_color:
            # ── human move ────────────────────────────────────────────────────
            uci = input("\nYour move (UCI): ").strip().lower()
            if uci in ("quit", "exit", "q"):
                print("Goodbye.")
                sys.exit(0)
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                print("Invalid UCI string. Try again.")
                continue
            if move not in board.legal_moves:
                print("Illegal move. Try again.")
                continue
            board.push(move)
            _render_board(
                board,
                chess.svg.Arrow(move.from_square, move.to_square, color="#3daf2b"),
                renderer,
            )
        else:
            # ── AI move ───────────────────────────────────────────────────────
            move = _pick_move(model, board, device, args.topk)
            board.push(move)
            print(f"AI plays: {move.uci()}")
            _render_board(
                board,
                chess.svg.Arrow(move.from_square, move.to_square, color="#e05c2a"),
                renderer,
            )

    print(f"\nGame over. Result: {board.result()}")
    outcome = board.outcome()
    if outcome:
        if outcome.winner is None:
            print("Draw.")
        elif outcome.winner == human_color:
            print("You win!")
        else:
            print("AI wins.")


if __name__ == "__main__":
    main()

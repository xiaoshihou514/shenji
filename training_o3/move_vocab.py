"""
A static bidirectional mapping between every legal *UCI* move on a blank board and an
integer id in [0, VOCAB_SIZE).

For simplicity we consider all 4672 origin-destination pairs plus 4×8 promotion squares.
Moves that are illegal in a concrete position receive zero probability after masking.
"""
from __future__ import annotations
import json
from pathlib import Path
import chess

__all__ = ["MoveVocab", "VOCAB_SIZE"]

class MoveVocab:
    _move_to_idx: dict[str, int] = {}
    _idx_to_move: list[str] = []

    @classmethod
    def build(cls) -> None:  # run once at import
        board = chess.Board(None)           # empty board for all squares
        idx = 0
        promotes = ["q", "r", "b", "n"]
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                uci = f"{chess.square_name(from_sq)}{chess.square_name(to_sq)}"
                # promotions only if move is a pawn promotion square
                for prom in ([None] if (chess.square_rank(from_sq) not in (1, 6)) else promotes):
                    move_str = uci + (prom or "")
                    cls._move_to_idx[move_str] = idx
                    cls._idx_to_move.append(move_str)
                    idx += 1
        # add castle moves explicitly (they are not captured above)
        for mv in ["e1g1", "e1c1", "e8g8", "e8c8"]:
            if mv not in cls._move_to_idx:
                cls._move_to_idx[mv] = idx
                cls._idx_to_move.append(mv)
                idx += 1

    @classmethod
    def to_idx(cls, move: chess.Move) -> int:
        return cls._move_to_idx[move.uci()]

    @classmethod
    def to_move(cls, idx: int) -> chess.Move:
        return chess.Move.from_uci(cls._idx_to_move[idx])

MoveVocab.build()
VOCAB_SIZE: int = len(MoveVocab._idx_to_move)

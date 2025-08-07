"""
move_vocab.py - universal UCI-move vocabulary.

Maps every syntactically legal UCI move to a unique integer id.
Moves that are illegal in the *current* position are masked at inference.
"""

from __future__ import annotations

import chess

__all__ = ["MoveVocab", "VOCAB_SIZE"]


class MoveVocab:
    _move_to_idx: dict[str, int] = {}
    _idx_to_move: list[str] = []

    @classmethod
    def build(cls) -> None:
        idx = 0
        promotions = ["q", "r", "b", "n"]

        # All 64 × 64 origin-destination pairs
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                uci = f"{chess.square_name(from_sq)}{chess.square_name(to_sq)}"

                # Add plain move
                cls._move_to_idx[uci] = idx
                cls._idx_to_move.append(uci)
                idx += 1

                # Add promotions when moving from 2nd or 7th rank
                if chess.square_rank(from_sq) in (1, 6):  # pawn start ranks
                    for promo in promotions:
                        move_str = uci + promo
                        cls._move_to_idx[move_str] = idx
                        cls._idx_to_move.append(move_str)
                        idx += 1

        # Explicitly add castling moves (not covered above)
        for castle in ("e1g1", "e1c1", "e8g8", "e8c8"):
            if castle not in cls._move_to_idx:
                cls._move_to_idx[castle] = idx
                cls._idx_to_move.append(castle)
                idx += 1

    @classmethod
    def to_idx(cls, move: chess.Move) -> int:
        """UCI move → integer id."""
        return cls._move_to_idx[move.uci()]

    @classmethod
    def to_move(cls, idx: int) -> chess.Move:
        """Integer id → `chess.Move`."""
        return chess.Move.from_uci(cls._idx_to_move[idx])


# Initialise on import
MoveVocab.build()
VOCAB_SIZE: int = len(MoveVocab._idx_to_move)

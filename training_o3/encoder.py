"""
BoardEncoder: 64-token sequence representation.

Each square is encoded as an *integer piece id* ∈ [0, 13]:
  0  = empty
  1-6  = white {pawn, knight, bishop, rook, queen, king}
  7-12 = black {pawn, knight, bishop, rook, queen, king}
Side-to-move is appended as the 65-th CLS token: 0 = white, 1 = black.
"""
from __future__ import annotations
import torch
import chess
from torch import Tensor

__all__ = ["BoardEncoder", "PIECE_IDS"]

PIECE_IDS = {
    None: 0,
    chess.Piece(chess.PAWN, chess.WHITE):   1,
    chess.Piece(chess.KNIGHT, chess.WHITE): 2,
    chess.Piece(chess.BISHOP, chess.WHITE): 3,
    chess.Piece(chess.ROOK, chess.WHITE):   4,
    chess.Piece(chess.QUEEN, chess.WHITE):  5,
    chess.Piece(chess.KING, chess.WHITE):   6,
    chess.Piece(chess.PAWN, chess.BLACK):   7,
    chess.Piece(chess.KNIGHT, chess.BLACK): 8,
    chess.Piece(chess.BISHOP, chess.BLACK): 9,
    chess.Piece(chess.ROOK, chess.BLACK):   10,
    chess.Piece(chess.QUEEN, chess.BLACK):  11,
    chess.Piece(chess.KING, chess.BLACK):   12,
}

class BoardEncoder:
    @staticmethod
    def encode(board: chess.Board) -> Tensor:            # (65,)
        sq = torch.tensor(
            [PIECE_IDS[board.piece_at(i)] for i in chess.SQUARES],
            dtype=torch.long
        )
        cls = torch.tensor([0 if board.turn else 1], dtype=torch.long)
        return torch.cat([cls, sq])      # CLS first

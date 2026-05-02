"""
board.py – board encoding and AlphaZero-style move codec.

Board encoding: 71-token integer sequence per position
  Tokens  0–63 : piece ID per square (a1=sq0 … h8=sq63)
                 0=empty, 1–6=white {P N B R Q K}, 7–12=black {p n b r q k}
  Token  64    : side to move  (0=white, 1=black)
  Tokens 65–68 : castling rights [WK, WQ, BK, BQ] (0 or 1 each)
  Token  69    : en-passant file (0–7 = files a–h, 8 = no EP)
  Token  70    : halfmove-clock bucket  floor(min(clock, 50) / 5) ∈ [0, 10]

Move encoding: AlphaZero 4672-class scheme
  idx = from_sq * 73 + plane,  plane ∈ [0, 72]
  Planes  0–55 : queen-style moves  (8 directions × 7 distances)
                 direction order: N NE E SE S SW W NW
  Planes 56–63 : knight moves        (8 orientations)
  Planes 64–72 : pawn underpromotions (3 files × 3 pieces: R B N)
  Queen promotions share the same plane as the corresponding queen-direction move.
"""

import chess
import numpy as np
import torch
from torch import Tensor

__all__ = ["BoardEncoder", "MoveCodec", "VOCAB_SIZE", "SEQ_LEN"]

SEQ_LEN = 71
VOCAB_SIZE = 4672  # 64 * 73

# ── piece-ID lookup ────────────────────────────────────────────────────────────
_PIECE_ID: dict[chess.Piece | None, int] = {None: 0}
for _color, _base in ((chess.WHITE, 1), (chess.BLACK, 7)):
    for _offset, _ptype in enumerate(
        (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
    ):
        _PIECE_ID[chess.Piece(_ptype, _color)] = _base + _offset

# ── AlphaZero move-encoding tables ────────────────────────────────────────────
# Compass directions as (Δfile, Δrank): N NE E SE S SW W NW
_QUEEN_DIRS: list[tuple[int, int]] = [
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)
]
# Knight (Δfile, Δrank) ordered consistently
_KNIGHT_DELTAS: list[tuple[int, int]] = [
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]
_UNDERPROMOTE = (chess.ROOK, chess.BISHOP, chess.KNIGHT)

# (from_sq, to_sq, promotion_piece | None) → global idx
_ENCODE: dict[tuple[int, int, int | None], int] = {}
# global idx → (from_sq, to_sq, promotion_piece | None)
_DECODE: list[tuple[int, int, int | None] | None] = [None] * VOCAB_SIZE


def _build_tables() -> None:
    for from_sq in chess.SQUARES:
        ff = chess.square_file(from_sq)
        fr = chess.square_rank(from_sq)

        # Queen-style moves (planes 0–55)
        for dir_idx, (df, dr) in enumerate(_QUEEN_DIRS):
            for dist in range(1, 8):
                tf = ff + df * dist
                tr = fr + dr * dist
                if not (0 <= tf <= 7 and 0 <= tr <= 7):
                    break
                to_sq = chess.square(tf, tr)
                plane = dir_idx * 7 + (dist - 1)
                idx = from_sq * 73 + plane
                # Regular move (covers queen promotions encoded with promotion=None too)
                _ENCODE[(from_sq, to_sq, None)] = idx
                _DECODE[idx] = (from_sq, to_sq, None)
                # Also register the explicit queen-promotion key for pawn promotion moves
                if (fr == 6 and tr == 7) or (fr == 1 and tr == 0):
                    _ENCODE[(from_sq, to_sq, chess.QUEEN)] = idx

        # Knight moves (planes 56–63)
        for kn_idx, (df, dr) in enumerate(_KNIGHT_DELTAS):
            tf = ff + df
            tr = fr + dr
            if not (0 <= tf <= 7 and 0 <= tr <= 7):
                continue
            to_sq = chess.square(tf, tr)
            plane = 56 + kn_idx
            idx = from_sq * 73 + plane
            _ENCODE[(from_sq, to_sq, None)] = idx
            _DECODE[idx] = (from_sq, to_sq, None)

        # Underpromotions (planes 64–72): only from promotion-adjacent ranks
        if fr not in (1, 6):
            continue
        tr = 7 if fr == 6 else 0
        for fd_idx, fd in enumerate((-1, 0, 1)):
            tf = ff + fd
            if not (0 <= tf <= 7):
                continue
            to_sq = chess.square(tf, tr)
            for piece_idx, promo in enumerate(_UNDERPROMOTE):
                plane = 64 + fd_idx * 3 + piece_idx
                idx = from_sq * 73 + plane
                _ENCODE[(from_sq, to_sq, promo)] = idx
                _DECODE[idx] = (from_sq, to_sq, promo)


_build_tables()


# ── Public API ─────────────────────────────────────────────────────────────────

class BoardEncoder:
    """Encodes a ``chess.Board`` into a 71-token integer tensor."""

    @staticmethod
    def encode(board: chess.Board) -> Tensor:
        """Return ``LongTensor(71,)`` representing the board state."""
        arr = np.empty(SEQ_LEN, dtype=np.int64)
        for sq in chess.SQUARES:
            arr[sq] = _PIECE_ID[board.piece_at(sq)]
        arr[64] = 0 if board.turn == chess.WHITE else 1
        arr[65] = int(board.has_kingside_castling_rights(chess.WHITE))
        arr[66] = int(board.has_queenside_castling_rights(chess.WHITE))
        arr[67] = int(board.has_kingside_castling_rights(chess.BLACK))
        arr[68] = int(board.has_queenside_castling_rights(chess.BLACK))
        arr[69] = chess.square_file(board.ep_square) if board.ep_square is not None else 8
        arr[70] = min(board.halfmove_clock // 5, 10)
        return torch.from_numpy(arr)

    @staticmethod
    def encode_np(board: chess.Board) -> np.ndarray:
        """Return ``uint8`` numpy array (71,) for bulk preprocessing."""
        arr = np.empty(SEQ_LEN, dtype=np.uint8)
        for sq in chess.SQUARES:
            arr[sq] = _PIECE_ID[board.piece_at(sq)]
        arr[64] = 0 if board.turn == chess.WHITE else 1
        arr[65] = int(board.has_kingside_castling_rights(chess.WHITE))
        arr[66] = int(board.has_queenside_castling_rights(chess.WHITE))
        arr[67] = int(board.has_kingside_castling_rights(chess.BLACK))
        arr[68] = int(board.has_queenside_castling_rights(chess.BLACK))
        arr[69] = chess.square_file(board.ep_square) if board.ep_square is not None else 8
        arr[70] = min(board.halfmove_clock // 5, 10)
        return arr


class MoveCodec:
    """Encodes/decodes moves using the AlphaZero 4672-class scheme."""

    @staticmethod
    def to_idx(move: chess.Move) -> int:
        """``chess.Move`` → global index in ``[0, 4671]``."""
        key = (move.from_square, move.to_square, move.promotion)
        try:
            return _ENCODE[key]
        except KeyError:
            raise ValueError(f"Cannot encode move {move.uci()!r} (key={key})") from None

    @staticmethod
    def to_move(idx: int, board: chess.Board) -> chess.Move:
        """Global index → ``chess.Move`` valid in *board*."""
        entry = _DECODE[idx]
        if entry is None:
            raise ValueError(f"Invalid move index {idx}")
        from_sq, to_sq, promo = entry
        # For queen-direction planes, the promotion is stored as None in the table;
        # infer queen promotion if a pawn is present and the target is a back rank.
        if promo is None:
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(to_sq)
                if to_rank in (0, 7):
                    promo = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promo)

    @staticmethod
    def legal_mask(board: chess.Board) -> Tensor:
        """Return a ``BoolTensor(4672,)`` where ``True`` marks legal moves."""
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        for move in board.legal_moves:
            key = (move.from_square, move.to_square, move.promotion)
            if key in _ENCODE:
                mask[_ENCODE[key]] = True
        return mask

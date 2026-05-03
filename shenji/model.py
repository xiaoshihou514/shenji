"""
model.py – Chess Transformer policy network.

Architecture:
    Input  : (B, 71) int64 token sequence
    Embed  : separate embedding tables per token type + learnable position
    Encode : pre-norm Transformer encoder (GELU activations, batch_first)
    Head   : Linear(d_model, 73) applied to the 64 square tokens
             → reshape to (B, 4672) policy logits
"""

import torch
import torch.nn as nn
from torch import Tensor

from shenji.board import VOCAB_SIZE

__all__ = ["ChessTransformer", "BoardEmbedding"]


class BoardEmbedding(nn.Module):
    """
    Embeds the 71-token board sequence into ``(B, 71, d_model)`` floats.

    Separate embedding tables are used for each distinct token type so that
    the model does not conflate, e.g., piece IDs with castling-right flags.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.piece_embed = nn.Embedding(13, d_model)  # tokens 0–63  (0–12)
        self.stm_embed = nn.Embedding(2, d_model)  # token  64     (0–1)
        self.castling_embed = nn.Embedding(2, d_model)  # tokens 65–68  (0–1)
        self.ep_embed = nn.Embedding(9, d_model)  # token  69     (0–8)
        self.hm_embed = nn.Embedding(11, d_model)  # token  70     (0–10)
        self.pos_embed = nn.Embedding(71, d_model)  # learnable positional

    def forward(self, x: Tensor) -> Tensor:  # x: (B, 71)
        pieces = self.piece_embed(x[:, :64])  # (B, 64, d)
        stm = self.stm_embed(x[:, 64:65])  # (B,  1, d)
        castling = self.castling_embed(x[:, 65:69])  # (B,  4, d)
        ep = self.ep_embed(x[:, 69:70])  # (B,  1, d)
        hm = self.hm_embed(x[:, 70:71])  # (B,  1, d)

        tok_emb = torch.cat([pieces, stm, castling, ep, hm], dim=1)  # (B, 71, d)

        positions = torch.arange(71, device=x.device)
        return tok_emb + self.pos_embed(positions)  # (B, 71, d)


class ChessTransformer(nn.Module):
    """
    Transformer encoder that predicts a policy distribution over chess moves.

    Forward input : ``(B, 71)`` int64 board tokens
    Forward output: ``(B, 4672)`` raw policy logits (not softmax-normalised)

    During inference, mask illegal moves to ``-inf`` before taking softmax:
        from shenji.board import MoveCodec
        mask = MoveCodec.legal_mask(board)
        logits = model(x)
        logits[~mask] = -torch.inf
        probs = logits.softmax(-1)
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = BoardEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.policy_head = nn.Linear(d_model, 73)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, 71)
        emb = self.embedding(x)  # (B, 71, d)
        enc = self.encoder(emb)  # (B, 71, d)
        sq = enc[:, :64]  # (B, 64, d)  – square representations
        logits = self.policy_head(sq)  # (B, 64, 73)
        return logits.reshape(x.size(0), -1)  # (B, 4672)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

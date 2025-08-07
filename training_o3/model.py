from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .move_vocab import VOCAB_SIZE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 65):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(pos * div)
        self.pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", self.pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)]


class ChessTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.embed = nn.Embedding(13, d_model)  # 13 piece ids incl. empty
        self.cls_embed = nn.Embedding(2, d_model)  # side-to-move token
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, 65)
        cls, squares = x[:, :1], x[:, 1:]
        emb = torch.cat(
            [self.cls_embed(cls), self.embed(squares)],
            dim=1,
        )
        enc = self.encoder(self.pos_enc(emb))
        logits = self.head(enc[:, 0])  # use CLS token
        return logits

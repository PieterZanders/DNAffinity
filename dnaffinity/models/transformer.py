"""Lightweight Transformer encoder for local MD feature windows."""

import math

import torch
import torch.nn as nn

from dnaffinity.models.torch_base import TorchRegressor


class MLPHead(nn.Module):
    def __init__(self, d_in, dropout=0.1):
        super().__init__()
        d_hidden = max(d_in // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class Transformer(nn.Module):
    def __init__(
        self,
        feat_dim,
        d_model=64,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
        pos_encoding='sinusoidal',
        pooling='cls',
        normalization=True,
    ):
        super().__init__()
        self.pooling = pooling
        self.norm_in = nn.LayerNorm(feat_dim) if normalization else None
        self.input_proj = nn.Linear(feat_dim, d_model)
        if pos_encoding == 'sinusoidal':
            self.pos_enc = SinusoidalPositionalEncoding(d_model)
            self.pos_embed = None
        elif pos_encoding == 'learned':
            self.pos_enc = None
            self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        else:
            raise ValueError("pos_encoding must be 'sinusoidal' or 'learned'")
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = MLPHead(d_model, dropout=dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            return self._forward(x), None
        return self._forward(x)

    def _forward(self, x):
        if self.norm_in is not None:
            x = self.norm_in(x)
        h = self.input_proj(x)
        if self.pos_enc is not None:
            h = self.pos_enc(h)
        else:
            h = h + self.pos_embed[:, : h.size(1), :]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        if self.pooling == 'cls':
            pooled = h[:, 0, :]
        elif self.pooling == 'mean':
            pooled = h[:, 1:, :].mean(dim=1)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")
        return self.head(pooled)


class TransformerRegressor(TorchRegressor):
    architecture = 'Transformer'
    input_format = 'windows'
    input_ndim = 3

    def _build_net(self, input_shape):
        kw = self.model_kwargs
        return Transformer(
            feat_dim=input_shape[-1],
            d_model=kw.get('d_model', 64),
            num_layers=kw.get('num_layers', 2),
            nhead=kw.get('nhead', 4),
            dim_feedforward=kw.get('dim_feedforward', 128),
            dropout=kw.get('dropout', 0.1),
            pos_encoding=kw.get('pos_encoding', 'sinusoidal'),
            pooling=kw.get('pooling', 'cls'),
            normalization=kw.get('normalization', True),
        )

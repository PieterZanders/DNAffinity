"""Lightweight Transformer encoder for local MD feature windows."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dnaffinity.models.base import BaseRegressor


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


class TransformerRegressor(BaseRegressor):
    architecture = 'Transformer'
    input_format = 'windows'

    def __init__(
        self,
        lr=0.001,
        normalization=True,
        num_epochs=100,
        batch_size=32,
        **model_kwargs,
    ):
        self.lr = lr
        self.normalization = normalization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_kwargs = dict(model_kwargs)
        self.model_kwargs.setdefault('normalization', normalization)
        self.net = None
        self.optimizer = None
        self.criterion = None
        self._device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self._device_str)

    def _build_net(self, feat_dim):
        kw = self.model_kwargs
        return Transformer(
            feat_dim=feat_dim,
            d_model=kw.get('d_model', 64),
            num_layers=kw.get('num_layers', 2),
            nhead=kw.get('nhead', 4),
            dim_feedforward=kw.get('dim_feedforward', 128),
            dropout=kw.get('dropout', 0.1),
            pos_encoding=kw.get('pos_encoding', 'sinusoidal'),
            pooling=kw.get('pooling', 'cls'),
            normalization=kw.get('normalization', True),
        )

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"{self.architecture} expects X with shape (n_samples, seq_len, feat_dim), got {X.shape}"
            )
        y = np.asarray(y, dtype=np.float32).ravel()
        feat_dim = X.shape[2]

        self.net = self._build_net(feat_dim)
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        print(f"Using {self.architecture} on device: {self.device}")
        print(f"Input shape: (batch, {X.shape[1]}, {feat_dim})")

        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        if sample_weight is not None:
            w_train = torch.tensor(
                np.asarray(sample_weight, dtype=np.float32), dtype=torch.float32
            ).to(self.device)
        else:
            w_train = None

        self.net.train()
        print("Start training...")
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_train), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_train))
                X_batch = X_train[i:batch_end]
                y_batch = y_train[i:batch_end]

                self.optimizer.zero_grad()
                y_pred = self.net(X_batch)

                if w_train is not None:
                    w_batch = w_train[i:batch_end]
                    loss = torch.mean(w_batch * (y_pred.squeeze() - y_batch) ** 2)
                else:
                    loss = self.criterion(y_pred.squeeze(), y_batch)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if epoch % 20 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / n_batches:.6f}")

        print(f"{self.architecture} training completed")

    def predict(self, X):
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32).to(
            self.device
        )
        self.net.eval()
        with torch.no_grad():
            return self.net(X_tensor).cpu().numpy()

    def state_dict(self):
        return self.net.state_dict() if self.net is not None else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('optimizer', None)
        state.pop('criterion', None)
        state.pop('device', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.device = torch.device(
            self._device_str if torch.cuda.is_available() or self._device_str == 'cpu' else 'cpu'
        )
        self.criterion = nn.MSELoss()
        if self.net is not None:
            self.net.to(self.device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.optimizer = None

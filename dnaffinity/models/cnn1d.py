"""1D-CNN sequence model for local MD feature windows."""

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


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ConvBackbone(nn.Module):
    """Stem + residual 1D conv stack; output (B, L, hidden_dim)."""

    def __init__(self, feat_dim, hidden_dim=64, num_blocks=2, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(hidden_dim, kernel_size) for _ in range(num_blocks)]
        )

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.stem(h)
        h = self.blocks(h)
        return h.transpose(1, 2)


def _global_pool(seq, pool='avg'):
    if pool == 'avg':
        return seq.mean(dim=1)
    if pool == 'max':
        return seq.max(dim=1).values
    raise ValueError(f"pool must be 'avg' or 'max', got {pool!r}")


class CNN1D(nn.Module):
    def __init__(
        self,
        feat_dim,
        hidden_dim=64,
        num_blocks=2,
        kernel_size=3,
        pool='avg',
        dropout=0.1,
        normalization=True,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(feat_dim) if normalization else None
        self.backbone = ConvBackbone(feat_dim, hidden_dim, num_blocks, kernel_size)
        self.pool = pool
        self.head = MLPHead(hidden_dim, dropout=dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            return self._forward(x), None
        return self._forward(x)

    def _forward(self, x):
        if self.norm_in is not None:
            x = self.norm_in(x)
        seq = self.backbone(x)
        pooled = _global_pool(seq, self.pool)
        return self.head(pooled)


class CNN1DRegressor(BaseRegressor):
    architecture = 'CNN1D'
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
        return CNN1D(
            feat_dim=feat_dim,
            hidden_dim=kw.get('hidden_dim', 64),
            num_blocks=kw.get('num_blocks', 2),
            kernel_size=kw.get('kernel_size', 3),
            pool=kw.get('pool', 'avg'),
            dropout=kw.get('dropout', 0.1),
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

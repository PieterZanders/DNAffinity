"""1D-CNN sequence model for local MD feature windows."""

import torch.nn as nn

from dnaffinity.models.torch_base import WindowTorchRegressor


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


class CNN1DRegressor(WindowTorchRegressor):
    """CNN1D wrapper using the shared training loop (see torch_base.TorchRegressor)."""

    architecture = 'CNN1D'

    def _build_net(self, X):
        kw = self.model_kwargs
        return CNN1D(
            feat_dim=X.shape[2],
            hidden_dim=kw.get('hidden_dim', 64),
            num_blocks=kw.get('num_blocks', 2),
            kernel_size=kw.get('kernel_size', 3),
            pool=kw.get('pool', 'avg'),
            dropout=kw.get('dropout', 0.1),
            normalization=kw.get('normalization', True),
        )

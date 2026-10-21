"""1D-CNN + self-attention sequence model for local MD feature windows."""

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


class CNNAttention(nn.Module):
    def __init__(
        self,
        feat_dim,
        hidden_dim=64,
        num_blocks=2,
        kernel_size=3,
        pool='avg',
        num_heads=4,
        num_attn_layers=1,
        dropout=0.1,
        normalization=True,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(feat_dim) if normalization else None
        self.backbone = ConvBackbone(feat_dim, hidden_dim, num_blocks, kernel_size)
        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.attn_norms.append(nn.LayerNorm(hidden_dim))
        self.pool = pool
        self.head = MLPHead(hidden_dim, dropout=dropout)
        self.last_attention = None

    def forward(self, x, return_attention=False):
        if self.norm_in is not None:
            x = self.norm_in(x)
        seq = self.backbone(x)
        attn_weights = []
        h = seq
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            attn_out, weights = attn(h, h, h, need_weights=True, average_attn_weights=True)
            h = norm(h + attn_out)
            attn_weights.append(weights)
        self.last_attention = attn_weights[-1] if attn_weights else None
        pooled = _global_pool(h, self.pool)
        out = self.head(pooled)
        if return_attention:
            return out, attn_weights
        return out


class CNNAttentionRegressor(WindowTorchRegressor):
    """CNNAttention wrapper using the shared training loop (see torch_base.TorchRegressor)."""

    architecture = 'CNNAttention'

    def _build_net(self, input_shape):
        kw = self.model_kwargs
        return CNNAttention(
            feat_dim=input_shape[-1],
            hidden_dim=kw.get('hidden_dim', 64),
            num_blocks=kw.get('num_blocks', 2),
            kernel_size=kw.get('kernel_size', 3),
            pool=kw.get('pool', 'avg'),
            num_heads=kw.get('num_heads', 4),
            num_attn_layers=kw.get('num_attn_layers', 1),
            dropout=kw.get('dropout', 0.1),
            normalization=kw.get('normalization', True),
        )

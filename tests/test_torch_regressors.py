"""Tests for the window-based PyTorch regressors (CNN1D, CNNAttention, Transformer)."""

import pytest

pytest.importorskip('torch')

import torch

from dnaffinity.models import get_model
from dnaffinity.models.cnn_attention import CNNAttention


@pytest.mark.parametrize('name', ['CNN1D', 'CNNAttention', 'Transformer'])
def test_regressor_fit_predict(name, window_batch):
    X, y, _w = window_batch
    model = get_model(name, num_epochs=2, batch_size=16, lr=1e-3)
    model.fit(X, y)
    pred = model.predict(X[:5])
    assert pred.shape == (5, 1)


@pytest.mark.parametrize('name', ['CNN1D', 'CNNAttention', 'Transformer'])
def test_regressor_weighted_loss(name, window_batch):
    X, y, w = window_batch
    model = get_model(name, num_epochs=2, batch_size=16)
    model.fit(X, y, sample_weight=w)
    assert model.predict(X[:1]).shape == (1, 1)


def test_regressor_rejects_flat_input(window_batch):
    X, y, _ = window_batch
    model = get_model('CNN1D', num_epochs=1)
    with pytest.raises(ValueError, match='seq_len, feat_dim'):
        model.fit(X.reshape(X.shape[0], -1), y)


def test_cnn_attention_returns_weights(window_batch):
    X, _, _ = window_batch
    net = CNNAttention(feat_dim=X.shape[2], num_blocks=1, num_attn_layers=1)
    out, weights = net(torch.tensor(X[:4]), return_attention=True)
    assert out.shape == (4, 1)
    assert len(weights) == 1
    assert weights[0].shape == (4, X.shape[1], X.shape[1])
    assert net.last_attention is not None

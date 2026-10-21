"""Tests for epsilon-LRP relevance propagation (DFN and CNN1D)."""

import numpy as np
import pytest

pytest.importorskip('torch')

import torch.nn as nn

from dnaffinity.models import get_model
from dnaffinity.utils.lrp import lrp_relevance


@pytest.fixture
def flat_batch():
    rng = np.random.default_rng(1)
    n, n_features = 64, 20
    X = rng.standard_normal((n, n_features), dtype=np.float32)
    y = rng.standard_normal(n, dtype=np.float32)
    return X, y


def _zero_biases(net):
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm)) and m.bias is not None:
            m.bias.data.zero_()


def test_dfn_relevance_shape_and_conservation(flat_batch):
    X, y = flat_batch
    reg = get_model('DFN', layers=[16, 1], num_epochs=2, batch_size=16)
    reg.fit(X, y)
    # with all biases zeroed the epsilon rule conserves relevance exactly
    _zero_biases(reg.net)

    R = lrp_relevance(reg, X)
    assert R.shape == X.shape
    pred = reg.predict(X).ravel()
    np.testing.assert_allclose(R.sum(axis=1), pred, rtol=1e-3, atol=1e-4)


def test_cnn1d_relevance_shape(window_batch):
    X, y, _w = window_batch
    reg = get_model('CNN1D', num_epochs=2, batch_size=16, num_blocks=2)
    reg.fit(X, y)

    R = lrp_relevance(reg, X)
    assert R.shape == X.shape
    assert np.isfinite(R).all()
    assert np.abs(R).sum() > 0


def test_lrp_rejects_unsupported_models(window_batch):
    with pytest.raises(ValueError, match='implemented for'):
        lrp_relevance(get_model('ridge'), np.zeros((2, 3)))
    with pytest.raises(ValueError, match='implemented for'):
        lrp_relevance(get_model('Transformer'), window_batch[0])


def test_lrp_requires_trained_network(flat_batch):
    reg = get_model('DFN', num_epochs=1)
    with pytest.raises(RuntimeError, match='not been trained'):
        lrp_relevance(reg, flat_batch[0])

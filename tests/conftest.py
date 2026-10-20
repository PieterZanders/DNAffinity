"""Pytest fixtures and path setup."""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATA4_TRAINING = REPO_ROOT / 'data' / 'upbm' / 'Gata4' / 'Gata4_training.txt'


@pytest.fixture
def repo_root():
    return REPO_ROOT


@pytest.fixture
def window_batch():
    """Small random batch (N, L, F) for window-based model smoke tests."""
    rng = np.random.default_rng(0)
    n, seq_len, feat_dim = 32, 6, 12
    X = rng.standard_normal((n, seq_len, feat_dim), dtype=np.float32)
    y = rng.standard_normal(n, dtype=np.float32)
    w = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    return X, y, w

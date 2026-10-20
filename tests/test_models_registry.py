"""Tests for model registry and config helpers."""

import json

import pytest

from dnaffinity.common import load_pytorch_params
from dnaffinity.models import MODEL_NAMES, get_model


def test_model_names_include_torch_regressors():
    for name in ('CNN1D', 'CNNAttention', 'Transformer', 'DFN'):
        assert name in MODEL_NAMES


def test_get_model_sklearn_without_torch():
    model = get_model('ridge')
    assert model.__class__.__name__ == 'Ridge'


def test_get_model_unknown_regressor():
    with pytest.raises(ValueError, match='Unknown regressor'):
        get_model('not_a_model')


def test_load_pytorch_params_merges_defaults(tmp_path):
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps({'lr': 0.01, 'batch_size': 8}))
    params = load_pytorch_params(str(cfg_path), defaults={'lr': 0.001, 'num_epochs': 50})
    assert params['lr'] == 0.01
    assert params['batch_size'] == 8
    assert params['num_epochs'] == 50


def test_load_pytorch_params_nested_key(tmp_path):
    cfg_path = tmp_path / 'nested.json'
    cfg_path.write_text(json.dumps({'pytorch_model_params': {'num_epochs': 3}}))
    params = load_pytorch_params(str(cfg_path), defaults={'num_epochs': 100, 'lr': 0.001})
    assert params['num_epochs'] == 3
    assert params['lr'] == 0.001

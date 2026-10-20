"""Model registry: one module per model, resolved by name via get_model()."""

from dnaffinity.models.base import BaseRegressor
from dnaffinity.models.linear import Linear
from dnaffinity.models.random_forest import RandomForest
from dnaffinity.models.ridge import Ridge
from dnaffinity.models.svr import SVR

_SKLEARN_MODELS = {
    'random_forest': RandomForest,
    'linear': Linear,
    'ridge': Ridge,
    'svr': SVR,
}

# PyTorch-backed models, imported lazily so the sklearn models work without torch.
# name -> (module, class)
_PYTORCH_MODELS = {
    'DFN': ('dnaffinity.models.dfn', 'DFN'),
    'CNN1D': ('dnaffinity.models.cnn1d', 'CNN1DRegressor'),
    'CNNAttention': ('dnaffinity.models.cnn_attention', 'CNNAttentionRegressor'),
    'Transformer': ('dnaffinity.models.transformer', 'TransformerRegressor'),
}

PYTORCH_MODEL_NAMES = tuple(_PYTORCH_MODELS)
MODEL_NAMES = list(_SKLEARN_MODELS) + list(_PYTORCH_MODELS)


def get_model(name, **params):
    """Instantiate a fresh regressor by name. Extra params are passed to the constructor."""
    if name in _SKLEARN_MODELS:
        return _SKLEARN_MODELS[name](**params)
    if name in _PYTORCH_MODELS:
        module_name, class_name = _PYTORCH_MODELS[name]
        try:
            from importlib import import_module
            module = import_module(module_name)
        except ImportError as e:
            raise ImportError(f"PyTorch dependencies not available: {e}. Please install torch.")
        return getattr(module, class_name)(**params)
    raise ValueError(f"Unknown regressor '{name}'. Available: {MODEL_NAMES}")


__all__ = [
    'BaseRegressor', 'RandomForest', 'Linear', 'Ridge', 'SVR',
    'MODEL_NAMES', 'PYTORCH_MODEL_NAMES', 'get_model',
]

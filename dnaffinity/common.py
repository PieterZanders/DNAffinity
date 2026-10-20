"""Shared logic for the regressor CLI scripts: argument parsing, PyTorch config
loading, the train-and-evaluate routine, and model saving."""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

from dnaffinity.dataset import Dataset
from dnaffinity.model import Model
from dnaffinity.models import MODEL_NAMES

# Repository root (parent of the dnaffinity package)
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / 'outputs'

# Default DFN hyperparameters (uPBM / gcPBM)
# Training-loop settings shared by all PyTorch models (early stopping on the
# validation split, AdamW weight decay, ReduceLROnPlateau)
TRAINING_DEFAULTS = {
    'num_epochs': 200, 'batch_size': 64, 'weight_decay': 1e-4, 'patience': 20,
    'scheduler': 'plateau', 'warmup_epochs': 0, 'loss': 'mse',
}

DEFAULT_DFN_PARAMS = {
    'layers': [64, 32, 1],  # [hidden1, hidden2, ..., output]
    'lr': 0.001,
    'normalization': True,
    **TRAINING_DEFAULTS,
}

# SELEX uses a deeper network with a smaller learning rate and batch size
SELEX_DFN_PARAMS = {
    'layers': [256, 128, 64, 32, 1],
    'lr': 1e-4,
    'normalization': True,
    **TRAINING_DEFAULTS,
}

# Default hyperparameters per PyTorch model; mirrored by configs/{model}_config.json
DEFAULT_MODEL_PARAMS = {
    'DFN': DEFAULT_DFN_PARAMS,
    'CNN1D': {
        'lr': 0.001, 'normalization': True, **TRAINING_DEFAULTS,
        'dropout': 0.1, 'hidden_dim': 64, 'num_blocks': 2, 'kernel_size': 3, 'pool': 'avg',
    },
    'CNNAttention': {
        'lr': 0.001, 'normalization': True, **TRAINING_DEFAULTS,
        'dropout': 0.1, 'hidden_dim': 64, 'num_blocks': 2, 'kernel_size': 3, 'pool': 'avg',
        'num_heads': 4, 'num_attn_layers': 1,
    },
    'Transformer': {
        # lower LR + warmup and mean pooling: the original (lr 1e-3, CLS pooling)
        # collapsed to a constant prediction on SELEX and gcPBM
        'lr': 3e-4, 'normalization': True, **TRAINING_DEFAULTS, 'warmup_epochs': 5,
        'dropout': 0.1, 'd_model': 32, 'num_layers': 2, 'nhead': 4, 'dim_feedforward': 64,
        'pos_encoding': 'sinusoidal', 'pooling': 'mean',
    },
}


def default_params(regressor):
    """Default hyperparameters for a regressor (empty dict for sklearn models)."""
    return dict(DEFAULT_MODEL_PARAMS.get(regressor, {}))


def build_parser(description, default_data_dir, extra_args=()):
    """
    Build the argument parser shared by all regressor scripts.

    :param description: str, text shown in --help
    :param default_data_dir: default for --data-dir (scripts read {data_dir}/{protein}/...)
    :param extra_args: iterable of (flag, kwargs) tuples for technique-specific arguments
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--protein', type=str, required=True, help='Protein name')
    for flag, kwargs in extra_args:
        parser.add_argument(flag, **kwargs)
    parser.add_argument('--data-dir', type=str, default=str(default_data_dir),
                        help=f'Data directory (default: {default_data_dir})')
    parser.add_argument('--regressor', type=str, default='DFN', choices=MODEL_NAMES,
                        help='Regressor type (default: DFN)')
    parser.add_argument('--training-set-size', type=float, default=0.9,
                        help='Training set size fraction (default: 0.9)')
    parser.add_argument('--randomize-fce', action='store_true',
                        help='Randomize FCE features')
    parser.add_argument('--score', type=str, default='Median_intensity',
                        help='Score column name (default: Median_intensity)')
    parser.add_argument('--model-target', type=str, default='octamers',
                        choices=['octamers', 'tetramers'],
                        help='Model target (default: octamers)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON file with PyTorch model parameters')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for splits and training')
    parser.add_argument('--cv', type=int, default=0,
                        help='If >1, run k-fold group cross-validation instead of a single split')
    parser.add_argument('--target-transform', type=str, default=None, choices=[None, 'log'],
                        help='Optional target transform (fitted on the training split)')
    parser.add_argument('--rc-average', action='store_true',
                        help='Average predictions over both strands of each sequence')
    parser.add_argument('--onehot', action='store_true',
                        help='Add one-hot sequence features (per-window for CNN/attention/transformer)')
    parser.add_argument('--legacy-preprocessing', action='store_true',
                        help='Use the original (pre-audit) preprocessing for comparison')
    return parser


def add_onehot(chosen_features, regressor):
    """Append the matching one-hot sequence feature for a regressor."""
    from dnaffinity.models import get_model
    fmt = get_model(regressor).input_format
    return list(chosen_features) + (['onehot_window'] if fmt == 'windows' else ['onehot_1mer'])


def load_pytorch_params(config_path, defaults=DEFAULT_DFN_PARAMS):
    """
    Merge PyTorch model parameters from a JSON config file over the defaults.
    The config may have the parameters at the top level or nested under
    "pytorch_model_params".
    """
    params = dict(defaults)
    if config_path is not None:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if 'pytorch_model_params' in cfg:
            cfg = cfg['pytorch_model_params']
        params.update(cfg)
    return params


def train_model(df_train, args, chosen_features, pytorch_params, weighted=False, label='the'):
    """
    Build a Dataset from df_train (ID_REF/VALUE columns), train a Model
    according to the CLI args (group split on canonical sequences, train-only
    scaling), report timing and test metrics, and return the model. With
    --cv k, k-fold group cross-validation is run first and summarized.
    """
    from dnaffinity.model import cross_validate
    if getattr(args, 'onehot', False):
        chosen_features = add_onehot(chosen_features, args.regressor)
    strings = list(df_train['ID_REF'])
    len_aln = len(strings[0])
    selected_tetramers = list(range(len_aln - 3))

    start_time = time.time()
    dset_train = Dataset(args.protein, df_train, args.model_target, args.randomize_fce,
                         chosen_features, args.score, selected_tetramers, weighted=weighted)
    model_kwargs = dict(weighted=weighted, pytorch_model_params=pytorch_params,
                        target_transform=getattr(args, 'target_transform', None),
                        rc_average=getattr(args, 'rc_average', False))
    seed = getattr(args, 'seed', 0)
    if getattr(args, 'cv', 0) and args.cv > 1:
        cv = cross_validate(dset_train, len_aln, args.regressor, n_folds=args.cv, seed=seed, **model_kwargs)
        summ = cv[['pearson', 'spearman', 'r2', 'rmse']].agg(['mean', 'std'])
        print(f'{args.cv}-fold CV summary:\n{summ.round(4)}')
    model = Model(dset_train, len_aln, args.regressor, args.training_set_size, seed=seed, **model_kwargs)
    print(f'{label} model has been trained')
    print("It took %s seconds" % (time.time() - start_time))

    metrics = model.predict()
    print('Test metrics (per unique sequence):', {k: round(v, 4) for k, v in metrics.items()})
    return model


def save_model(model, filename, protocol=None):
    """
    Pickle the full Model object to `filename`; for PyTorch-backed models the
    network state dict is additionally saved next to it with a .pth extension.
    """
    filename = str(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = model.regressor.state_dict()
    if state is not None:
        import torch
        torch.save(state, os.path.splitext(filename)[0] + '.pth')

    with open(filename, 'wb') as f:
        if protocol is None:
            pickle.dump(model, f)
        else:
            pickle.dump(model, f, protocol=protocol)
    print(f'Model saved to {filename}')

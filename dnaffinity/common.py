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
DEFAULT_DFN_PARAMS = {
    'layers': [64, 32, 1],  # [hidden1, hidden2, ..., output]
    'lr': 0.001,
    'normalization': True,
    'num_epochs': 100,
    'batch_size': 32,
}

# SELEX uses a deeper network with a smaller learning rate and batch size
SELEX_DFN_PARAMS = {
    'layers': [256, 128, 64, 32, 1],
    'lr': 1e-4,
    'normalization': True,
    'num_epochs': 100,
    'batch_size': 16,
}

# Default hyperparameters per PyTorch model; mirrored by configs/{model}_config.json
DEFAULT_MODEL_PARAMS = {
    'DFN': DEFAULT_DFN_PARAMS,
    'CNN1D': {
        'lr': 0.001, 'normalization': True, 'num_epochs': 100, 'batch_size': 32,
        'dropout': 0.1, 'hidden_dim': 64, 'num_blocks': 2, 'kernel_size': 3, 'pool': 'avg',
    },
    'CNNAttention': {
        'lr': 0.001, 'normalization': True, 'num_epochs': 100, 'batch_size': 32,
        'dropout': 0.1, 'hidden_dim': 64, 'num_blocks': 2, 'kernel_size': 3, 'pool': 'avg',
        'num_heads': 4, 'num_attn_layers': 1,
    },
    'Transformer': {
        'lr': 0.001, 'normalization': True, 'num_epochs': 100, 'batch_size': 32,
        'dropout': 0.1, 'd_model': 64, 'num_layers': 2, 'nhead': 4, 'dim_feedforward': 128,
        'pos_encoding': 'sinusoidal', 'pooling': 'cls',
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
    return parser


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
    according to the CLI args, report timing and r2, and return the model.
    """
    strings = list(df_train['ID_REF'])
    len_aln = len(strings[0])
    selected_tetramers = list(range(len_aln - 3))

    start_time = time.time()
    dset_train = Dataset(args.protein, df_train, args.model_target, args.randomize_fce,
                         chosen_features, args.score, selected_tetramers, weighted=weighted)
    model = Model(dset_train, len_aln, args.regressor, args.training_set_size,
                  weighted=weighted, pytorch_model_params=pytorch_params)
    print(f'{label} model has been trained')
    print("It took %s seconds" % (time.time() - start_time))

    model.predict()
    print(model.y_test.shape, model.y_pred.shape)
    print('The r2 is ', model.r2)
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

"""Tests for Dataset feature construction."""

from pathlib import Path

import pandas as pd
import pytest

from dnaffinity.dataset import Dataset

GATA4_TRAINING = (
    Path(__file__).resolve().parents[1] / 'data' / 'upbm' / 'Gata4' / 'Gata4_training.txt'
)


@pytest.fixture
def gata4_df():
    if not GATA4_TRAINING.is_file():
        pytest.skip(f'Missing fixture data: {GATA4_TRAINING}')
    return pd.read_csv(GATA4_TRAINING, sep='\t').head(200)


def test_windowed_features_shape_and_positional_blocks(gata4_df):
    selected = list(range(5))
    dset = Dataset(
        'Gata4',
        gata4_df,
        model_target='octamers',
        chosen_features=['presence_tetramer', 'avg', 'diagonal_fce'],
        selected_tetramers=selected,
    )
    seq = dset.windowed_features
    flat = dset.features

    assert seq.ndim == 3
    assert seq.shape[0] == len(gata4_df) * 2  # forward + inverse complement rows
    assert seq.shape[1] == len(selected)
    assert seq.shape[2] == 12  # 6 avg + 6 diagonal_fce per window

    # Flat features hstack blocks by type; windowed_features interleaves per window
    n, seq_len = seq.shape[0], seq.shape[1]
    avg_seq = dset.tetra_avg.reshape(n, seq_len, 6)
    diag_seq = dset.tetra_fce_reduced.reshape(n, seq_len, 6)
    assert (seq[:, :, :6] == avg_seq).all()
    assert (seq[:, :, 6:] == diag_seq).all()
    assert flat.shape[1] == 256 + seq_len * 12


def test_windowed_features_requires_positional_feature(gata4_df):
    dset = Dataset(
        'Gata4',
        gata4_df,
        chosen_features=['presence_tetramer'],
        selected_tetramers=[0, 1, 2],
    )
    with pytest.raises(ValueError, match='No positional features'):
        _ = dset.windowed_features

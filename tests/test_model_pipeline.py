"""Integration tests for Model train/predict with window-based regressors."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('torch')

from dnaffinity.dataset import Dataset
from dnaffinity.model import Model

GATA4_TRAINING = (
    Path(__file__).resolve().parents[1] / 'data' / 'upbm' / 'Gata4' / 'Gata4_training.txt'
)


@pytest.fixture
def small_gata4_dataset():
    if not GATA4_TRAINING.is_file():
        pytest.skip(f'Missing fixture data: {GATA4_TRAINING}')
    df = pd.read_csv(GATA4_TRAINING, sep='\t').head(80)
    len_aln = len(df['ID_REF'].iloc[0])
    selected = list(range(len_aln - 3))
    return Dataset(
        'Gata4',
        df,
        model_target='octamers',
        chosen_features=['avg', 'diagonal_fce'],
        selected_tetramers=selected,
        weighted=False,
    ), len_aln


def test_model_uses_windowed_features_for_cnn1d(small_gata4_dataset):
    dset, len_aln = small_gata4_dataset
    model = Model(
        dset,
        len_aln,
        regressor='CNN1D',
        training_set_size=0.8,
        pytorch_model_params={'num_epochs': 2, 'batch_size': 16},
    )
    assert model.X.ndim == 3
    assert not model.trained
    with pytest.raises(RuntimeError, match='not trained'):
        model.predict()

    model.train()
    model.predict()
    assert model.y_pred is not None
    assert model.r2 is not None
    assert len(model.history['train']) == 2
    assert len(model.history['val']) == 2


@pytest.mark.parametrize('regressor', ['random_forest', 'DFN', 'CNN1D', 'CNNAttention', 'Transformer'])
def test_model_repr_shows_architecture_before_training(small_gata4_dataset, regressor):
    dset, len_aln = small_gata4_dataset
    model = Model(dset, len_aln, regressor=regressor,
                  pytorch_model_params={'num_epochs': 1, 'batch_size': 16})
    text = repr(model)
    assert 'not trained' in text
    if regressor == 'random_forest':
        assert 'RandomForestRegressor' in text
        assert model.history is None
    else:
        assert 'trainable parameters' in text
        assert 'Sequential' in text or 'Linear' in text

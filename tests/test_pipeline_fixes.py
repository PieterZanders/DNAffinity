"""Tests for the leakage-free evaluation pipeline and preprocessing fixes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnaffinity.dataset import Dataset
from dnaffinity.model import FeatureScaler, Model, TargetScaler, cross_validate, group_split
from dnaffinity.preprocess import aggregate_canonical, canonical, load_gcpbm, load_selex, revcomp

ROOT = Path(__file__).resolve().parents[1]
GATA4 = ROOT / 'data' / 'upbm' / 'Gata4' / 'Gata4_training.txt'


def _toy_df(n=120, L=10, seed=0):
    rng = np.random.default_rng(seed)
    seqs = [''.join(rng.choice(list('ACGT'), L)) for _ in range(n)]
    y = np.array([s.count('GATA') + 0.1 * rng.standard_normal() for s in seqs])
    return pd.DataFrame({'ID_REF': seqs, 'VALUE': y})


def test_revcomp_and_canonical():
    assert revcomp('AACG') == 'CGTT'
    assert canonical('TTTT') == 'AAAA'
    df = pd.DataFrame({'ID_REF': ['AACG', 'CGTT', 'GGGG'], 'VALUE': [1.0, 3.0, 5.0]})
    agg = aggregate_canonical(df)
    assert len(agg) == 2
    assert agg.set_index('ID_REF').loc['AACG', 'VALUE'] == 2.0


def test_dataset_groups_pair_strands():
    df = _toy_df(30)
    d = Dataset('x', df, chosen_features=['avg'], selected_tetramers=range(7))
    n = len(df)
    assert len(d.groups) == 2 * n
    assert (d.groups[:n] == d.groups[n:]).all()


def test_electrostatic_is_float():
    d = Dataset('x', _toy_df(10), chosen_features=['avg', 'electrostatic'], selected_tetramers=range(7))
    assert d.features.dtype.kind == 'f'
    assert d.windowed_features.dtype.kind == 'f'


def test_onehot_window_alignment():
    df = pd.DataFrame({'ID_REF': ['ACGTAC'], 'VALUE': [1.0]})
    d = Dataset('x', df, chosen_features=['onehot_window'], selected_tetramers=range(3))
    w = d.windowed_features
    assert w.shape == (2, 3, 16)
    # window 1 of the forward strand is CGTA
    expected = np.eye(4)[[1, 2, 3, 0]].ravel()
    assert (w[0, 1] == expected).all()


def test_group_split_keeps_strands_together():
    groups = np.repeat(np.arange(50), 2)
    tr, te = group_split(groups, 0.2, seed=0)
    assert not set(groups[tr]) & set(groups[te])


@pytest.mark.parametrize('reg', ['ridge', 'DFN'])
def test_model_no_leakage_and_train_only_scaling(reg):
    df = _toy_df(150)
    d = Dataset('x', df, chosen_features=['presence_tetramer', 'avg'], selected_tetramers=range(7))
    m = Model(d, 10, reg, training_set_size=0.8, seed=1,
              pytorch_model_params={'num_epochs': 3, 'batch_size': 32})
    assert not set(d.groups[m.fit_idx]) & set(d.groups[m.test_idx])
    assert not set(d.groups[m.val_idx]) & set(d.groups[m.test_idx])
    ref = FeatureScaler().fit(d.features[m.fit_idx])
    assert np.allclose(m.feature_scaler.mean_, ref.mean_)
    assert np.isclose(m.target_scaler.mean_, d.scores.ravel()[m.fit_idx].mean())
    res = m.predict()
    assert set(res) >= {'pearson', 'spearman', 'r2', 'mse', 'rmse', 'mae'}
    assert np.isclose(res['rmse'] ** 2, res['mse'])


def test_rc_average_gives_equal_strand_predictions():
    d = Dataset('x', _toy_df(80), chosen_features=['avg'], selected_tetramers=range(7))
    m = Model(d, 10, 'CNN1D', seed=0, rc_average=True,
              pytorch_model_params={'num_epochs': 2, 'batch_size': 32})
    m.predict()
    p = pd.Series(m.y_pred.ravel()).groupby(d.groups[m.test_idx]).nunique()
    assert (p == 1).all()


def test_target_scaler_log_roundtrip():
    y = np.array([0.0, 0.5, 2.0, 100.0])
    ts = TargetScaler('log').fit(y)
    assert np.allclose(ts.inverse_transform(ts.transform(y)), y, rtol=1e-5, atol=1e-6)


def test_cross_validate_folds_disjoint():
    d = Dataset('x', _toy_df(100), chosen_features=['presence_tetramer', 'avg'], selected_tetramers=range(7))
    cv = cross_validate(d, 10, 'ridge', n_folds=5, seed=0, verbose=False)
    assert len(cv) == 5
    assert cv['n_test_groups'].sum() == 100


def test_torch_training_is_deterministic_and_early_stops():
    from dnaffinity.models import get_model
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 5, 4)).astype(np.float32)
    y = rng.standard_normal(64).astype(np.float32)
    preds = []
    for _ in range(2):
        m = get_model('CNN1D', num_epochs=50, patience=2, batch_size=16, seed=3)
        m.fit(X[:48], y[:48], eval_set=(X[48:], y[48:]))
        preds.append(m.predict(X[:4]))
        assert m.epochs_trained < 50
    assert np.allclose(*preds)


def test_load_selex_count_selection(tmp_path):
    f = tmp_path / 's.txt'
    pd.DataFrame({'Kmer': ['AACG', 'CGTT', 'GGGG'], 'ObservedCount': [20, 30, 2],
                  'Probability': [0.1, 0.2, 0.3], 'ExpectedCount': [1, 1, 1],
                  'Affinity': [1.0, 0.5, 0.2], 'SE': [0.1, 0.1, 0.1]}).to_csv(f, sep='\t')
    df = load_selex(f, min_count=10)
    assert list(df['ID_REF']) == ['AACG'] and df['VALUE'].iloc[0] == 0.75


def test_load_gcpbm_flank_filter(tmp_path):
    core = 'CACGTG'
    clean = 'A' * 15 + core + 'T' * 9      # 30 bp, only the central site
    flanked = 'CACGTGAAA' + 'A' * 6 + core + 'T' * 9  # exact E-box in the left flank
    pad = 'GGG'
    data = tmp_path / 'p_1.txt'
    pd.DataFrame({'ID': ['a', 'b'], 'Sequence': [pad + clean + 'C' * 27, pad + flanked + 'C' * 27],
                  'Intensity': [10, 20]}).to_csv(data, sep='\t', index=False)
    fm = tmp_path / 'fm.txt'
    fm.write_text('A\tC\tG\tT\n' + '\n'.join(' '.join(map(str, r)) for r in np.eye(4)[[1, 0, 1, 2, 3, 2]]))
    out = load_gcpbm(data, fm, flank_threshold=4.5, core=(15, 21))
    assert len(out) == 1 and out['VALUE'].iloc[0] == 10


@pytest.mark.skipif(not GATA4.is_file(), reason='missing data')
def test_gata4_real_data_split_has_no_shared_sequences():
    from dnaffinity.preprocess import load_upbm
    df = load_upbm(GATA4)
    assert df['ID_REF'].is_unique
    d = Dataset('Gata4', df.head(300), chosen_features=['avg'], selected_tetramers=range(7), weighted=True)
    m = Model(d, 10, 'linear', weighted=True, seed=0)
    seqs = np.array(d.sequences)
    test_seqs = set(seqs[m.test_idx])
    train_all = set(seqs[m.fit_idx])
    assert not test_seqs & train_all
    assert not {revcomp(s) for s in test_seqs} & train_all

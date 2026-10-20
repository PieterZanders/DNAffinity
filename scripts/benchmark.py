#!/usr/bin/env python3
"""5-fold group-CV benchmark for one (dataset, preprocessing, model, variant) job.

Usage: python scripts/benchmark.py DATASET {fixed,legacy} MODEL VARIANT [--folds 5] [--out DIR]
Variants (cumulative):
  base      fixed evaluation, physics features, MSE, no RC averaging
  rcavg     + predictions averaged over both strands
  onehot    + one-hot sequence features (per-window for windowed models, flat otherwise)
  loghuber  + log-transformed targets (+ Huber loss for neural models)
  orig      (Transformer only) original hyperparameters: lr 1e-3, CLS pooling, d_model 64, no warmup
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dnaffinity.common import SELEX_DFN_PARAMS, add_onehot, default_params  # noqa: E402
from dnaffinity.dataset import Dataset  # noqa: E402
from dnaffinity.model import cross_validate  # noqa: E402
from dnaffinity.models import PYTORCH_MODEL_NAMES  # noqa: E402
from dnaffinity import preprocess as pp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('dataset', choices=['upbm', 'selex', 'gcpbm'])
p.add_argument('prep', choices=['fixed', 'legacy'])
p.add_argument('model')
p.add_argument('variant', choices=['base', 'rcavg', 'onehot', 'loghuber', 'orig'])
p.add_argument('--folds', type=int, default=5)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--threads', type=int, default=1)
p.add_argument('--out', default=str(ROOT / 'outputs' / 'cv_runs'))
a = p.parse_args()
torch.set_num_threads(a.threads)
legacy = a.prep == 'legacy'

weighted = False
if a.dataset == 'upbm':
    f = ROOT / 'data/upbm/Gata4/Gata4_training.txt'
    import pandas as pd
    df = pd.read_csv(f, sep='\t') if legacy else pp.load_upbm(f)
    feats, weighted = ['presence_tetramer', 'avg', 'diagonal_fce'], True
elif a.dataset == 'selex':
    df = pp.load_selex(ROOT / 'data/selex/Gata4/4.txt', selection='legacy' if legacy else 'count')
    feats = ['presence_tetramer', 'avg', 'diagonal_fce']
else:
    d = ROOT / 'data/gcpbm/cbf1'
    df = (pp.load_gcpbm(d / 'cbf1_100.txt', d / 'cbf1_freq_matrix_6.txt', flank_threshold=2.5, aggregate=False)
          if legacy else pp.load_gcpbm(d / 'cbf1_100.txt', d / 'cbf1_freq_matrix_6.txt'))
    if legacy:  # original script MinMax-normalised intensities on all probes
        df['VALUE'] = (df['VALUE'] - df['VALUE'].min()) / (df['VALUE'].max() - df['VALUE'].min())
    feats = ['presence_tetramer', 'avg', 'diagonal_fce', 'electrostatic']

v = a.variant
if v in ('onehot', 'loghuber'):
    feats = add_onehot(feats, a.model)
L = len(df['ID_REF'].iloc[0])
dset = Dataset(a.dataset, df, 'octamers', False, feats, 'Median_intensity', list(range(L - 3)), weighted=weighted)

params = {}
if a.model in PYTORCH_MODEL_NAMES:
    params = dict(SELEX_DFN_PARAMS if (a.dataset == 'selex' and a.model == 'DFN') else default_params(a.model))
    if v == 'orig':
        assert a.model == 'Transformer'
        params.update(lr=1e-3, pooling='cls', d_model=64, dim_feedforward=128, warmup_epochs=0)
    if v == 'loghuber':
        params['loss'] = 'huber'
    params['progress'] = False
model_params = {'n_jobs': 1} if a.model == 'random_forest' else None
kw = dict(weighted=weighted, pytorch_model_params=params, model_params=model_params,
          rc_average=v in ('rcavg', 'onehot', 'loghuber'),
          target_transform='log' if v == 'loghuber' else None)
t0 = time.time()
cv = cross_validate(dset, L, a.model, n_folds=a.folds, seed=a.seed, **kw)
el = time.time() - t0
summary = {'dataset': a.dataset, 'prep': a.prep, 'model': a.model, 'variant': v,
           'n_rows': len(dset.groups), 'n_sequences': int(len(np.unique(dset.groups))),
           'features': feats, 'cv_time_s': round(el, 1)}
for m in ['pearson', 'spearman', 'r2', 'rmse', 'r2_train', 'epochs']:
    col = cv[m].astype(float)
    summary[m + '_mean'], summary[m + '_std'] = float(col.mean()), float(col.std())
summary['folds'] = cv.to_dict(orient='records')
Path(a.out).mkdir(parents=True, exist_ok=True)
with open(Path(a.out) / f"{a.dataset}_{a.prep}_{a.model}_{v}.json", 'w') as fh:
    json.dump(summary, fh, indent=1, default=float)
print(json.dumps({k: summary[k] for k in summary if k != 'folds'}, default=float))

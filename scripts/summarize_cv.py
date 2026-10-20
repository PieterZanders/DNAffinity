#!/usr/bin/env python3
"""Collect outputs/cv_runs/*.json into outputs/fixed_results.csv and print markdown tables."""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(glob.glob(str(ROOT / 'outputs/cv_runs/*.json'))):
    d = json.load(open(f))
    d.pop('folds', None)
    d['features'] = '+'.join(d['features'])
    rows.append(d)
df = pd.DataFrame(rows)
order = {'upbm': 0, 'selex': 1, 'gcpbm': 2}
vorder = {'base': 0, 'orig': 1, 'rcavg': 2, 'onehot': 3, 'loghuber': 4}
df = df.sort_values(['dataset', 'prep', 'model', 'variant'],
                    key=lambda s: s.map(order) if s.name == 'dataset' else
                    (s.map(vorder) if s.name == 'variant' else s)).reset_index(drop=True)
df.to_csv(ROOT / 'outputs/fixed_results.csv', index=False)
fmt = lambda r, m: f"{r[m + '_mean']:.3f} ± {r[m + '_std']:.3f}"
df['Pearson'] = df.apply(lambda r: fmt(r, 'pearson'), axis=1)
df['Spearman'] = df.apply(lambda r: fmt(r, 'spearman'), axis=1)
df['R2'] = df.apply(lambda r: fmt(r, 'r2'), axis=1)
cols = ['dataset', 'prep', 'model', 'variant', 'n_sequences', 'Pearson', 'Spearman', 'R2', 'epochs_mean', 'cv_time_s']
print(df[cols].to_markdown(index=False, floatfmt='.0f'))

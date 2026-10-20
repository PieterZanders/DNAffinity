#!/usr/bin/env python3
"""gcPBM regressor for affinity prediction.

Requires '{data_dir}/{protein}/{protein}_{concentration}.txt' (raw intensities)
and '{data_dir}/{protein}/{protein}_freq_matrix_6.txt' (position frequency
matrix). Probes whose flanking regions score too high against the frequency
matrix (secondary binding sites) are discarded before training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnaffinity.common import (OUTPUTS_DIR, ROOT, build_parser, default_params,
                               load_pytorch_params, save_model, train_model)
from dnaffinity.plotting import save_plots

parser = build_parser('gcPBM regressor for affinity prediction',
                      default_data_dir=ROOT / 'data' / 'gcpbm',
                      extra_args=[('--concentration', dict(type=str, required=True,
                                                           help='Concentration label'))])
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Preprocessing: normalize intensities and filter out probes with secondary
# binding sites in the flanks
# ---------------------------------------------------------------------------
raw_data = pd.read_csv(f'{args.data_dir}/{args.protein}/{args.protein}_{args.concentration}.txt',
                       sep='\t')
proc_data = pd.concat([raw_data[["Sequence", "Intensity"]]])
proc_data = proc_data.dropna()
proc_data = proc_data.reset_index()
strings = list(proc_data["Sequence"])
strings = [seq[3:33] for seq in strings]
values = proc_data["Intensity"]

min_val = min(values)
normalization = max(values) - min(values)
values = list((values - min_val) / normalization)

plt.plot(sorted(values))
print(strings[0][12:18])

freq_matrix = pd.read_csv(f'{args.data_dir}/{args.protein}/{args.protein}_freq_matrix_6.txt',
                          sep=r'\s+')
freq_matrix = np.array(freq_matrix)
translate = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# flank positions to scan (outside the central binding site)
l = list(np.arange(0, 12 - 5))
for k in np.arange(18, 30 - 5):
    l.append(k)

discarded = []
for s in np.arange(len(strings)):
    string = strings[s]
    scores = []
    for i in l:
        score = 0
        for j in np.arange(0, len(freq_matrix)):
            a = translate[string[i + j]]
            score = score + freq_matrix[j][a]
        scores.append(score)
    if any(sc > 2.5 for sc in scores):
        discarded.append(s)

discarded = list(discarded)
print('We have discarded', len(discarded) / len(strings) * 100, '% of the data')

df_train = pd.DataFrame({'ID_REF': strings, 'VALUE': values})
df_train = df_train.drop(discarded)
df_train = df_train.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
chosen_features = ['presence_tetramer', 'avg', 'diagonal_fce', 'electrostatic']
pytorch_params = load_pytorch_params(args.config, default_params(args.regressor))

model = train_model(df_train, args, chosen_features, pytorch_params, label='the gcPBM')

out_dir = OUTPUTS_DIR / 'gcpbm' / args.protein / args.regressor
save_model(model, out_dir / 'model.pck')
importances = save_plots(model, out_dir)
if importances is not None:
    print(importances.head(10))
else:
    print("Feature importance analysis not available for this regressor type")

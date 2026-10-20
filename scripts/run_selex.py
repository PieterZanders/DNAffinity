#!/usr/bin/env python3
"""HT-SELEX regressor for affinity prediction.

Requires '{data_dir}/{protein}/{cycle}.txt' (preprocessed with the R pipeline
under notebooks/preprocess_selex/). Undersamples to the top 10% of k-mers by
probability, writes the training files to outputs/selex/{protein}/{regressor}/, then trains.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dnaffinity.common import (OUTPUTS_DIR, ROOT, SELEX_DFN_PARAMS, build_parser,
                               default_params, load_pytorch_params, save_model, train_model)
from dnaffinity.plotting import save_plots

parser = build_parser('SELEX regressor for affinity prediction',
                      default_data_dir=ROOT / 'data' / 'selex',
                      extra_args=[('--cycle', dict(type=str, required=True, help='Cycle number'))])
args = parser.parse_args()

deca = pd.read_csv(f'{args.data_dir}/{args.protein}/{args.cycle}.txt', sep='\t')

strings = list(deca["Kmer"])
values = list(deca["Affinity"])
counts = list(deca["ObservedCount"])
prob = list(deca["Probability"])

# undersampling: keep the top 10% of k-mers by probability
prob_bound = sorted(prob)[int(len(prob) * 0.9)]

mat = []
for i in np.arange(0, len(prob)):
    if prob[i] > prob_bound:
        mat.append([strings[i], values[i]])

out_dir = OUTPUTS_DIR / 'selex' / args.protein / args.regressor
out_dir.mkdir(parents=True, exist_ok=True)

# ordered training file
with open(out_dir / f'{args.protein}_training_ordered.txt', 'w') as file:
    file.write('ID_REF\tVALUE\n')
    for vector in mat:
        file.write("%s\t" % vector[0])
        file.write("%s\n" % vector[1])

# randomized file ready for training
np.random.shuffle(mat)

with open(out_dir / 'SELEX_training.txt', 'w') as file:
    file.write('ID_REF\tVALUE\n')
    for vector in mat:
        file.write("%s\t" % vector[0])
        file.write("%s\n" % vector[1])

df_train = pd.read_csv(out_dir / 'SELEX_training.txt', sep='\t')

chosen_features = ['presence_tetramer', 'avg', 'diagonal_fce']
# SELEX historically uses a deeper DFN; other models use their standard defaults
defaults = SELEX_DFN_PARAMS if args.regressor == 'DFN' else default_params(args.regressor)
pytorch_params = load_pytorch_params(args.config, defaults)

model = train_model(df_train, args, chosen_features, pytorch_params, label='the SELEX')

save_model(model, out_dir / 'model.sav', protocol=4)
importances = save_plots(model, out_dir)
if importances is not None:
    print(importances.head(10))

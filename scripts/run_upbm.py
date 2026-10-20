#!/usr/bin/env python3
"""uPBM regressor for affinity prediction.

Requires '{data_dir}/{protein}/{protein}_training.txt' (preprocessed with
notebooks/uPBM_preprocess.ipynb). Example proteins available in the dataset:
Tbx2, Foxc2, Pou1f1, Zscan10, Sox10, Nr5a2, Rora, Xbp1, Sp140, Sdccag8, Gata4,
Ar, Foxo6, Nfil3, Zfp202, Zfp263, Egr3, Mlx, Tfec, Snai1, Ahcftf1, Atf3, Atf4,
Dbp, Dmrtc2, Esrrb, Esrrg, Klf8, Klf9, Klf12, Mybl2, Mzf1, Nhlh2, Nkx2-9,
Nr2e1, Nr2f1, Nr2f6, Nr4a2, P42pop, Pit1, Prdm11, Rarg, Rfx7, Rorb, Sox3,
Srebf1, Tbx1, Tbx4, Tbx5, Tbx20, Zbtb1, Zfp300, Zfp3, Zfp637, Zfx, Zic5,
Zkscan1, Zkscan5, Znf740, Foxg1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dnaffinity.common import (OUTPUTS_DIR, ROOT, build_parser, default_params,
                               load_pytorch_params, save_model, train_model)
from dnaffinity.plotting import save_plots
from dnaffinity.preprocess import load_upbm

parser = build_parser('uPBM regressor for affinity prediction',
                      default_data_dir=ROOT / 'data' / 'upbm')
args = parser.parse_args()
np.random.seed(20)

train_file = f'{args.data_dir}/{args.protein}/{args.protein}_training.txt'
if args.legacy_preprocessing:
    df_train = pd.read_csv(train_file, sep='\t')
else:
    # collapse duplicate / reverse-complement 10-mers (mean label, max weight)
    df_train = load_upbm(train_file)

chosen_features = ['presence_tetramer', 'avg', 'diagonal_fce']
pytorch_params = load_pytorch_params(args.config, default_params(args.regressor))

model = train_model(df_train, args, chosen_features, pytorch_params,
                    weighted=True, label='the uPBM')

out_dir = OUTPUTS_DIR / 'upbm' / args.protein / args.regressor
save_model(model, out_dir / 'model.sav')
importances = save_plots(model, out_dir)
if importances is not None:
    print(importances.head(10))

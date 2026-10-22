import pandas as pd, numpy as np
from lib import OUT
R = OUT / 'report'; T = pd.read_csv(OUT / 'results_final.csv')
NM = {'linear': 'Linear', 'ridge': 'Ridge', 'svr': 'SVR (RBF)', 'rf': 'Random forest', 'lightgbm': 'LightGBM', 'dfn': 'DFN', 'cnn1d': 'CNN1D',
      'cnnattention': 'CNNAttention', 'transformer': 'Transformer', 'ens_all_nn': 'Ensemble (all NNs)', 'ens_cnn1d+cnnattention': 'Ensemble (CNN1D+CNNAttn)'}
f = lambda v: '--' if pd.isna(v) else str(v).replace('±', '$\\pm$')
L = []
for _, r in T.iterrows():
    L.append(' & '.join([NM[r.model], str(int(r.runs)), f(r.get('train_r2')), f(r.get('val_r2')), f(r.test_r2), f(r.test_pearson), f(r.test_spearman), f(r.get('gap')), f'{r.test_ens_r2:.3f}']) + r' \\')
open(R / 'tab_models.tex', 'w').write('\n'.join(L))
L = []
for _, r in T.dropna(subset=['epochs_mean']).iterrows():
    L.append(f"{NM[r.model]} & {r.best_epoch_mean:.1f} & {r.epochs_mean:.1f} & {r.final_train_mse:.3f} & {r.best_val_mse:.3f} \\\\")
open(R / 'tab_curves.tex', 'w').write('\n'.join(L)); print(open(R / 'tab_models.tex').read()); print(open(R / 'tab_curves.tex').read())

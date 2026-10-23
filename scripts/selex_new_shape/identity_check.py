"""How much sequence identity is encoded in the MD tetramer table (27 values/tetramer)?"""
import numpy as np, json
from lib import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from scipy.spatial.distance import pdist
ks = sorted(TAB[(4, 'avg')]); M = np.array([np.concatenate([TAB[(4, 'avg')][k], TAB[(4, 'fce')][k]]) for k in ks]); Z = (M - M.mean(0)) / M.std(0)
r = dict(n=len(ks), rank=int(np.linalg.matrix_rank(Z)), min_pair_dist=float(pdist(Z).min()), median_pair_dist=float(np.median(pdist(Z))))
g = np.array([canonical(k) for k in ks]); _, g = np.unique(g, return_inverse=True)
for p in range(4):
    y = np.array([NUC.index(k[p]) for k in ks]) if (NUC := 'ACGT') else 0
    r[f'logreg_acc_pos{p+1}'] = float(cross_val_score(LogisticRegression(max_iter=3000, C=10), Z, y, cv=GroupKFold(5), groups=g).mean())
# full tetramer identity via 1-NN under noise: fraction of tetramers still uniquely identified when z-scored features get gaussian noise sd 0.1
rs = np.random.RandomState(0); N = Z + rs.randn(*Z.shape) * 0.1; d = ((N[:, None] - Z[None]) ** 2).sum(2); r['nn_identity_noise0.1'] = float((d.argmin(1) == np.arange(len(ks))).mean())
print(json.dumps(r, indent=1)); json.dump(r, open(OUT / 'md_identity_check.json', 'w'), indent=1)

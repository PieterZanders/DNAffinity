"""Preprocessing ablation. Common evaluation for every config: held-out-fold rows = per-strand measured
labels with count>=10, metrics on log(affinity). Grouped (k-mer + RC) 5-fold CV on the 90% train pool.
Resumable: one json per config in outputs/selex_new_shape/ablation/."""
import sys, json, itertools, numpy as np, pandas as pd, torch
from lib import *
from sklearn.ensemble import RandomForestRegressor
from dnaffinity.models.cnn1d import CNN1DRegressor
torch.set_num_threads(2)
model = sys.argv[1]  # rf | cnn
D = OUT / 'ablation'; D.mkdir(parents=True, exist_ok=True)
full = pd.read_csv(ROOT / 'data/selex/Gata4/4.txt', sep='\t')
LAB = dict(zip(full.Kmer, full.Affinity)); CNT = dict(zip(full.Kmer, full.ObservedCount))
allg = full.Kmer.map(canonical).values; tg = set(allg[split(allg)])
BASE = dict(tf='log', mc=10, mode='perstrand', ks='4', props='avg+fce', onehot=0, shape=1, std=1)
VAR = dict(tf=['raw', 'log'], mc=[1, 3, 10, 30], mode=['merged', 'perstrand'], ks=['4', '6', '46'],
           props=['avg', 'fce', 'avg+fce'], onehot=[0, 1], shape=[0, 1], std=[0, 1])
cfgs = [BASE] + [dict(BASE, **{k: v}) for k, vs in VAR.items() for v in vs if v != BASE[k]]
cfgs += [dict(BASE, onehot=1, ks='46'), dict(BASE, onehot=1, ks='46', mode='merged')]
cfgs += [dict(BASE, shape=0, onehot=1)]
if model == 'cnn': cfgs = [c for c in cfgs if c['mc'] not in (1, 3)]  # mc=1 (126k rows) too costly for CNN on CPU
def run(c):
    name = '_'.join(f'{k}-{v}' for k, v in c.items()); f = D / f'{model}__{name}.json'
    if f.exists(): return
    d = load_labels(min(c['mc'], 10), c['mode']); d = d[~d.group.isin(tg)].reset_index(drop=True)
    keep = np.array([CNT.get(s, 0) >= c['mc'] for s in d.seq]) if c['mc'] > 10 else np.ones(len(d), bool)
    ks = tuple(int(x) for x in c['ks']); props = tuple(c['props'].split('+'))
    if not c['shape'] and not c['onehot']: return
    X = (flat_feats if model == 'rf' else windowed)(list(d.seq), ks, props, bool(c['onehot']), bool(c['shape']))
    y = tf(d.aff.values, c['tf'])
    ev_ok = np.array([CNT.get(s, 0) >= 10 for s in d.seq]); ev_y = np.log([LAB.get(s, np.nan) for s in d.seq])
    res = []
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, d.group)):
        tr = tr[keep[tr]]; Xtr, Xva = X[tr], X[va]
        if c['std']:
            ax = (0,) if X.ndim == 2 else (0, 1); mu, sd = Xtr.mean(ax), Xtr.std(ax) + 1e-8
            Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        ym, ys = y[tr].mean(), y[tr].std()
        if model == 'rf':
            m = RandomForestRegressor(300, min_samples_leaf=3, max_features=0.33, n_jobs=4, random_state=0).fit(Xtr, y[tr]); p = m.predict(Xva)
        else:
            m = CNN1DRegressor(num_epochs=150, patience=15, batch_size=64, lr=1e-3, weight_decay=1e-2, dropout=0.2, seed=0, progress=False, normalization=not c['std'])
            iv = np.random.RandomState(k).rand(len(tr)) < 0.1
            m.fit(Xtr[~iv], (y[tr][~iv] - ym) / ys, eval_set=(Xtr[iv], (y[tr][iv] - ym) / ys)); p = m.predict(Xva).ravel() * ys + ym
        p = np.log(np.clip(inv(p, c['tf']), 1e-4, None)); sel = ev_ok[va]
        res.append(metrics(ev_y[va][sel], p[sel]))
    out = dict(cfg=c, model=model, folds=res, **{f'{m}_mean': float(np.mean([r[m] for r in res])) for m in ['r2', 'pearson', 'spearman']},
               **{f'{m}_sd': float(np.std([r[m] for r in res])) for m in ['r2', 'pearson', 'spearman']})
    json.dump(out, open(f, 'w'), indent=1); print(model, name, round(out['r2_mean'], 3), flush=True)
PART = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i, c in enumerate(cfgs):
    if i % 2 == PART or len(sys.argv) <= 2: run(c)

"""Final RF / DFN / CNN1D with chosen preprocessing. Grouped 5-fold CV on 90% pool (+10% RC-grouped held-out test).
NNs: 3 seeds per fold, internal 10% early-stopping split from the training fold; histories saved.
Resumable: one npz/json per (model, fold, seed). Usage: train_final.py MODEL CFG_JSON"""
import sys, json, numpy as np, pandas as pd, torch
from lib import *
from sklearn.ensemble import RandomForestRegressor
from dnaffinity.models.cnn1d import CNN1DRegressor
from dnaffinity.models.dfn import DFN
torch.set_num_threads(3)
model = sys.argv[1]; c = json.loads(sys.argv[2])
D = OUT / 'final' / model; D.mkdir(parents=True, exist_ok=True); json.dump(c, open(OUT / 'final' / 'config.json', 'w'))
full = pd.read_csv(ROOT / 'data/selex/Gata4/4.txt', sep='\t'); allg = full.Kmer.map(canonical).values; tg = set(allg[split(allg)])
d = load_labels(c['mc'], c['mode']); te = d.group.isin(tg).values
ks = tuple(int(x) for x in c['ks']); props = tuple(c['props'].split('+'))
X = (flat_feats if model in ('rf', 'dfn') else windowed)(list(d.seq), ks, props, bool(c['onehot']), bool(c['shape']))
y = tf(d.aff.values, c['tf'])
P, T = np.where(~te)[0], np.where(te)[0]
for k, (tr, va) in enumerate(GroupKFold(5).split(P, P, d.group.values[P])):
    tr, va = P[tr], P[va]
    ax = (0,) if X.ndim == 2 else (0, 1); mu, sd = X[tr].mean(ax), X[tr].std(ax) + 1e-8
    Z = (X - mu) / sd if c['std'] else X
    ym, ys = y[tr].mean(), y[tr].std()
    for seed in ([0] if model == 'rf' else [0, 1, 2]):
        f = D / f'fold{k}_seed{seed}.npz'
        if f.exists(): continue
        hist = None
        if model == 'rf':
            m = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.33, n_jobs=6, random_state=seed).fit(Z[tr], y[tr])
            pred = m.predict(Z)
        else:
            iv = np.random.RandomState(100 + k).rand(len(tr)) < 0.1; a, b = tr[~iv], tr[iv]
            kw = dict(num_epochs=300, patience=25, batch_size=64, weight_decay=1e-2, seed=seed, progress=False, normalization=not c['std'])
            m = (DFN(layers=[256, 128, 64, 1], dropout=0.2, lr=3e-4, **kw) if model == 'dfn'
                 else CNN1DRegressor(dropout=0.2, lr=1e-3, **kw))
            m.fit(Z[a], (y[a] - ym) / ys, eval_set=(Z[b], (y[b] - ym) / ys))
            pred = m.predict(Z).ravel() * ys + ym; hist = m.history
            np.save(D / f'fold{k}_seed{seed}_hist.npy', np.array([hist['train'], hist['val']]))
        np.savez(f, pred=pred, tr=tr, va=va, te=T, y=y, best=-1 if hist is None else m.best_epoch)
        print(model, k, seed, 'test r2', round(r2_score(y[T], pred[T]), 3), flush=True)

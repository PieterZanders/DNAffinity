"""Final RF / DFN / CNN1D with chosen preprocessing. Grouped 5-fold CV on 90% pool (+10% RC-grouped held-out test).
NNs: 3 seeds per fold, internal 10% early-stopping split from the training fold; histories saved.
Resumable: one npz/json per (model, fold, seed). Usage: train_final.py MODEL CFG_JSON"""
import os, sys, json, time, numpy as np, pandas as pd, torch
from lib import *
from sklearn.ensemble import RandomForestRegressor
from dnaffinity.models.cnn1d import CNN1DRegressor
from dnaffinity.models.dfn import DFN
from dnaffinity.models.cnn_attention import CNNAttentionRegressor
from dnaffinity.models.transformer import TransformerRegressor
from dnaffinity.models import get_model
WIN = ('cnn1d', 'cnnattention', 'transformer'); SK = ('rf', 'ridge', 'linear', 'svr', 'lightgbm')
torch.set_num_threads(int(os.environ.get("NT", 3)))
model = sys.argv[1]; c = json.loads(sys.argv[2])
FD = OUT / (os.environ['OUTDIR'] if 'OUTDIR' in os.environ else ('final' if c['ks'] == '4' else f"final_ks{c['ks']}")); D = FD / model; D.mkdir(parents=True, exist_ok=True); json.dump(c, open(FD / 'config.json', 'w'))
full = pd.read_csv(ROOT / 'data/selex/Gata4/4.txt', sep='\t'); allg = full.Kmer.map(canonical).values; tg = set(allg[split(allg)])
d = load_labels(c['mc'], c['mode']); te = d.group.isin(tg).values
ks = tuple(int(x) for x in c['ks']); props = tuple(c['props'].split('+'))
t0 = time.perf_counter(); X = (flat_feats if model not in WIN else windowed)(list(d.seq), ks, props, bool(c['onehot']), bool(c['shape']))
y = tf(d.aff.values, c['tf']); t_feat = time.perf_counter() - t0
P, T = np.where(~te)[0], np.where(te)[0]
for k, (tr, va) in enumerate(GroupKFold(5).split(P, P, d.group.values[P])):
    tr, va = P[tr], P[va]
    if len(sys.argv) > 3 and str(k) not in sys.argv[3].split(','): continue
    ax = (0,) if X.ndim == 2 else (0, 1); mu, sd = X[tr].mean(ax), X[tr].std(ax) + 1e-8
    Z = (X - mu) / sd if c['std'] else X
    ym, ys = y[tr].mean(), y[tr].std()
    for seed in ([0] if model in SK else [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]):
        f = D / f'fold{k}_seed{seed}.npz'
        if f.exists(): continue
        hist = None; t1 = time.perf_counter()
        if model in SK:
            if model == 'rf': m = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.33, n_jobs=6, random_state=seed)
            elif model == 'lightgbm':
                import lightgbm as lgb; m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=4)
            else: m = get_model(model, **({'alpha': 10.0} if model == 'ridge' else {})).model
            if model == 'lightgbm':
                iv = np.random.RandomState(100 + k).rand(len(tr)) < 0.1; a, b = tr[~iv], tr[iv]
                m.fit(Z[a], (y[a] - ym) / ys, eval_set=[(Z[b], (y[b] - ym) / ys)], callbacks=[lgb.early_stopping(100, verbose=False)])
                pred = m.predict(Z) * ys + ym; hist = None
            else:
                m.fit(Z[tr], (y[tr] - ym) / ys); pred = m.predict(Z) * ys + ym
        else:
            iv = np.random.RandomState(100 + k).rand(len(tr)) < 0.1; a, b = tr[~iv], tr[iv]
            kw = dict(num_epochs=300, patience=25, batch_size=64, weight_decay=1e-2, seed=seed, progress=False, normalization=not c['std'])
            m = {'dfn': lambda: DFN(layers=[256, 128, 64, 1], dropout=0.2, lr=3e-4, **kw),
                 'cnn1d': lambda: CNN1DRegressor(dropout=0.2, lr=1e-3, **kw),
                 'cnnattention': lambda: CNNAttentionRegressor(dropout=0.2, lr=1e-3, hidden_dim=64, num_blocks=2, kernel_size=3, pool='avg', num_heads=4, num_attn_layers=1, **kw),
                 'transformer': lambda: TransformerRegressor(dropout=0.2, lr=3e-4, warmup_epochs=5, d_model=32, num_layers=2, nhead=4, dim_feedforward=64, pos_encoding='sinusoidal', pooling='mean', **kw)}[model]()
            m.fit(Z[a], (y[a] - ym) / ys, eval_set=(Z[b], (y[b] - ym) / ys))
            pred = m.predict(Z).ravel() * ys + ym; hist = m.history
            np.save(D / f'fold{k}_seed{seed}_hist.npy', np.array([hist['train'], hist['val']]))
        t_fit = time.perf_counter() - t1  # includes predicting all rows; inference timed separately below
        t2 = time.perf_counter(); _ = m.predict(Z[T]); t_inf = (time.perf_counter() - t2) / len(T) * 1000
        np.savez(f, t_feat=t_feat, t_fit=t_fit, t_inf1k=t_inf, n_rows=len(d), pred=pred, tr=tr, va=va, te=T, y=y, best=getattr(m, 'best_iteration_', -1) if hist is None else m.best_epoch)
        if os.environ.get('SAVEW') and hist is not None:  # best (early-stopped) weights + standardisation for LRP
            torch.save(dict(state=m.net.state_dict(), mu=mu, sd=sd, ym=ym, ys=ys, input_shape=Z.shape[1:]), D / f'fold{k}_seed{seed}.pt')
        print(model, k, seed, 'test r2', round(r2_score(y[T], pred[T]), 3), flush=True)

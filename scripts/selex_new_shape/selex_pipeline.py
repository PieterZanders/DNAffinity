"""Importable version of the report's training path (same code as train_final.py, same hyper-parameters),
used by notebooks/dnaffinity_selex.ipynb. train_final.py / bench.py / lrp.py remain the CLI scripts that produced the report.
Differences to train_final.py: (i) runs in-process and returns results instead of writing npz files; (ii) epochs/patience are arguments
(report: 300 / 25) so that a QUICK mode is possible. Nothing else is changed (labels, split, folds, standardisation, models)."""
import os, time, contextlib, numpy as np, pandas as pd
from lib import *            # TAB, windowed, flat_feats, load_labels, split, metrics, tf, ROOT, OUT, canonical, revcomp ...
import lib
import torch
from sklearn.ensemble import RandomForestRegressor
from dnaffinity.models import get_model
from dnaffinity.models.cnn1d import CNN1DRegressor
from dnaffinity.models.dfn import DFN
from dnaffinity.models.cnn_attention import CNNAttentionRegressor
from dnaffinity.models.transformer import TransformerRegressor

MODELS = ['linear', 'ridge', 'svr', 'rf', 'dfn', 'cnn1d', 'cnnattention', 'transformer']
NM = dict(linear='Linear', ridge='Ridge', svr='SVR', rf='RF', dfn='DFN', cnn1d='CNN1D', cnnattention='CNNAttention', transformer='Transformer')
WIN = ('cnn1d', 'cnnattention', 'transformer')      # per-position (n, 7, channels) input; the others get a flat vector
SK = ('rf', 'ridge', 'linear', 'svr')
_base = dict(tf='log', mc=10, mode='perstrand', props='avg+fce', std=1)
FEATURE_SETS = {   # letter -> (report name, config); same configs as outputs/selex_new_shape/final*/config.json
    'a': ('tetramer', dict(_base, ks='4', onehot=1, shape=1)),    # tetramer MD shape + one-hot
    'b': ('hexamer', dict(_base, ks='6', onehot=1, shape=1)),     # hexamer MD shape + one-hot
    'c': ('tetra+hexa', dict(_base, ks='46', onehot=1, shape=1)), # both + one-hot
    'd': ('one-hot', dict(_base, ks='4', onehot=1, shape=0)),     # one-hot only
    'e': ('tetramer-MD', dict(_base, ks='4', onehot=0, shape=1)), # tetramer MD only
    'f': ('hexamer-MD', dict(_base, ks='6', onehot=0, shape=1)),  # hexamer MD only
    'g': ('tetra+hexa-MD', dict(_base, ks='46', onehot=0, shape=1)),
}
FS_LABEL = {'a': '(a) tetramer shape + one-hot', 'b': '(b) hexamer shape + one-hot', 'c': '(c) tetra+hexa shape + one-hot', 'd': '(d) one-hot only',
            'e': '(e) tetramer MD only', 'f': '(f) hexamer MD only', 'g': '(g) tetra+hexa MD only'}

def set_threads(n=None):
    torch.set_num_threads(n or os.cpu_count() or 1)

def nn_factory(model, seed, epochs=300, patience=25, normalization=False):
    """Hyper-parameters of the report (Section Models). Report: num_epochs=300, patience=25."""
    kw = dict(num_epochs=epochs, patience=patience, batch_size=64, weight_decay=1e-2, seed=seed, progress=False, normalization=normalization)
    return {'dfn': lambda: DFN(layers=[256, 128, 64, 1], dropout=0.2, lr=3e-4, **kw),
            'cnn1d': lambda: CNN1DRegressor(dropout=0.2, lr=1e-3, **kw),
            'cnnattention': lambda: CNNAttentionRegressor(dropout=0.2, lr=1e-3, hidden_dim=64, num_blocks=2, kernel_size=3, pool='avg', num_heads=4, num_attn_layers=1, **kw),
            'transformer': lambda: TransformerRegressor(dropout=0.2, lr=3e-4, warmup_epochs=5, d_model=32, num_layers=2, nhead=4, dim_feedforward=64, pos_encoding='sinusoidal', pooling='mean', **kw)}[model]()

def sk_factory(model, seed, n_jobs=None):
    if model == 'rf': return RandomForestRegressor(500, min_samples_leaf=3, max_features=0.33, n_jobs=n_jobs or os.cpu_count(), random_state=seed)
    return get_model(model, **({'alpha': 10.0} if model == 'ridge' else {})).model

def labels_and_test(cfg):
    """Labelled rows (count >= mc, per strand) and boolean test mask: 10% of the reverse-complement groups of ALL k-mers of 4.txt (seed 20)."""
    full = pd.read_csv(ROOT / 'data/selex/Gata4/4.txt', sep='\t'); allg = full.Kmer.map(canonical).values
    test_groups = set(allg[split(allg)])
    d = load_labels(cfg['mc'], cfg['mode']); return d, d.group.isin(test_groups).values

def features(d, model, cfg):
    ks = tuple(int(x) for x in cfg['ks']); props = tuple(cfg['props'].split('+'))
    return (flat_feats if model not in WIN else windowed)(list(d.seq), ks, props, bool(cfg['onehot']), bool(cfg['shape']))

def folds(d, te):
    """GroupKFold(5) on the 90% pool, groups = reverse-complement groups. Yields (k, train_idx, val_idx) as indices into d."""
    P = np.where(~te)[0]
    for k, (tr, va) in enumerate(GroupKFold(5).split(P, P, d.group.values[P])): yield k, P[tr], P[va]

def fit_run(model, X, y, tr, T, fold, seed, std=True, epochs=300, patience=25):
    """One run (= one fold, one seed). Returns dict(pred over all rows (original log-affinity units), net/model, standardisation, history, timing)."""
    ax = (0,) if X.ndim == 2 else (0, 1); mu, sd = X[tr].mean(ax), X[tr].std(ax) + 1e-8
    Z = (X - mu) / sd if std else X
    ym, ys = y[tr].mean(), y[tr].std(); hist = None; t1 = time.perf_counter()
    if model in SK:
        m = sk_factory(model, seed); m.fit(Z[tr], (y[tr] - ym) / ys); pred = m.predict(Z) * ys + ym
    else:
        iv = np.random.RandomState(100 + fold).rand(len(tr)) < 0.1; a, b = tr[~iv], tr[iv]   # 10% early-stopping split of the training fold
        m = nn_factory(model, seed, epochs, patience, normalization=not std)
        m.fit(Z[a], (y[a] - ym) / ys, eval_set=(Z[b], (y[b] - ym) / ys)); pred = m.predict(Z).ravel() * ys + ym; hist = m.history
    t_fit = time.perf_counter() - t1; t2 = time.perf_counter(); _ = m.predict(Z[T]); t_inf = (time.perf_counter() - t2) / len(T) * 1000
    return dict(model=m, pred=pred, mu=mu, sd=sd, ym=ym, ys=ys, hist=hist, t_fit=t_fit, t_inf1k=t_inf, best=getattr(m, 'best_epoch', None), fold=fold, seed=seed, tr=tr)

def run_cv(model, cfg, fold_list=(0,), seeds=(0,), epochs=300, patience=25, data=None):
    """All runs of one model on one feature-set config. fold_list=range(5), seeds=(0,1,2) = the report's full protocol (NNs; sklearn models use seed 0 only)."""
    d, te = data if data is not None else labels_and_test(cfg)
    t0 = time.perf_counter(); X = features(d, model, cfg); y = tf(d.aff.values, cfg['tf']); t_feat = time.perf_counter() - t0
    T = np.where(te)[0]; runs = []
    for k, tr, va in folds(d, te):
        if k not in fold_list: continue
        for seed in ((0,) if model in SK else seeds):
            r = fit_run(model, X, y, tr, T, k, seed, bool(cfg['std']), epochs, patience); r.update(va=va, te=T, y=y, X=X, t_feat=t_feat, d=d); runs.append(r)
    return runs

def run_metrics(r):
    y, p = r['y'], r['pred']; a, b, v = metrics(y[r['tr']], p[r['tr']]), metrics(y[r['te']], p[r['te']]), metrics(y[r['va']], p[r['va']])
    return dict(fold=r['fold'], seed=r['seed'], train_r2=a['r2'], val_r2=v['r2'], test_r2=b['r2'], pearson=b['pearson'], spearman=b['spearman'], gap=a['r2'] - b['r2'], t_fit=r['t_fit'], t_inf1k=r['t_inf1k'])

def summarize(runs):
    """Report-style row: mean±sd over runs, gap = train-test, avg_runs_r2 = test R2 of the mean prediction of all runs."""
    M = pd.DataFrame([run_metrics(r) for r in runs]); ddof = 0 if len(M) == 1 else 1; S = {'runs': len(M)}
    for c in ['train_r2', 'val_r2', 'test_r2', 'pearson', 'spearman', 'gap', 't_fit']: S[c] = M[c].mean(); S[c + '_sd'] = M[c].std(ddof=ddof)
    z = runs[0]; ens = np.mean([r['pred'][z['te']] for r in runs], 0); S['avg_runs_r2'] = metrics(z['y'][z['te']], ens)['r2']; S['t_feat'] = z['t_feat']; S['t_inf1k'] = M.t_inf1k.mean()
    return S

@contextlib.contextmanager
def permuted_md_table(seed):
    """Control of the report: reassign the 256 real tetramer MD rows (avg and fce together) to random tetramers (same values, same dimension)."""
    saved = {q: lib.TAB[(4, q)] for q in ('avg', 'fce')}; ks = sorted(saved['avg']); p = np.random.RandomState(seed).permutation(len(ks))
    for q in saved: lib.TAB[(4, q)] = {k: saved[q][ks[j]] for k, j in zip(ks, p)}
    try: yield
    finally:
        for q in saved: lib.TAB[(4, q)] = saved[q]

def lrp_for_run(model, r, cfg):
    """LRP (lrp.py, exact rules of the report: BN folded, eps=0.01 input/linear, gamma=0.25 hidden convs) for one trained run of 'dfn' or 'cnn1d'
    on the held-out test rows. Returns dict(R, R0, G, out, X, seqs, y): composite LRP, LRP-0, gradient x input, f(x), features (z-scored), test sequences, test labels."""
    import lrp
    from dnaffinity.models.cnn1d import CNN1D
    from dnaffinity.models.dfn import RegressionCV, FeedForward
    T = r['te']; Z = torch.tensor((r['X'][T] - r['mu']) / r['sd'], dtype=torch.float32); sd_ = r['model'].net.state_dict()
    if model == 'cnn1d': net = CNN1D(Z.shape[2], 64, 2, 3, 'avg', 0.2, normalization=False); fn = lrp.cnn_lrp
    else: net = RegressionCV(FeedForward([Z.shape[1], 256, 128, 64, 1], dropout=0.2), normalization=False); fn = lrp.dfn_lrp
    net.load_state_dict(sd_); net.eval()
    with torch.no_grad(): ref = net(Z).ravel()
    R, o = fn(net, Z, 'comp'); R0, _ = fn(net, Z, 'zero'); G = lrp.gxi(net, Z)
    assert torch.allclose(o, ref, atol=1e-4), 'LRP forward mismatch (BN folding)'
    d = r['d']; return dict(R=R.numpy(), R0=R0.numpy(), G=G.numpy(), out=o.numpy(), X=Z.numpy(), seqs=np.array(list(d.seq.values[T])), y=r['y'][T])

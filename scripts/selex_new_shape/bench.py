"""Benchmark models x feature sets (tetramer / hexamer / tetra+hexa): metrics, timing, plots, LaTeX tables."""
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lib import *
FS = {'tetramer': 'final', 'hexamer': 'final_ks6', 'tetra+hexa': 'final_ks46'}
TIME = {'tetramer': 'final_retime_ks4', 'hexamer': 'final_ks6', 'tetra+hexa': 'final_ks46'}
MODELS = ['linear', 'ridge', 'svr', 'rf', 'lightgbm', 'dfn', 'cnn1d', 'cnnattention', 'transformer']; NN = MODELS[5:]
NM = dict(linear='Linear', ridge='Ridge', svr='SVR', rf='RF', lightgbm='LightGBM', dfn='DFN', cnn1d='CNN1D', cnnattention='CNNAttention', transformer='Transformer',
          ens_all_nn='Ens. all NNs', ens_cnn='Ens. CNN1D+CNNAttn')
P = OUT / 'plots' / 'bench'; P.mkdir(parents=True, exist_ok=True); rows = []
for fs, d in FS.items():
    ENS = {}
    for m in MODELS:
        fl = sorted(glob.glob(str(OUT / d / m / 'fold*_seed*.npz')))
        if not fl: continue
        R = []
        for f in fl:
            z = np.load(f); p, y = z['pred'], z['y']; a, b = metrics(y[z['tr']], p[z['tr']]), metrics(y[z['te']], p[z['te']])
            R.append(dict(tr=a['r2'], va=metrics(y[z['va']], p[z['va']])['r2'], te=b['r2'], pe=b['pearson'], sp=b['spearman']))
        R = pd.DataFrame(R); z0 = np.load(fl[0]); te = z0['te']; ens = np.mean([np.load(f)['pred'][te] for f in fl], 0); ENS[m] = (z0['y'][te], ens)
        tl = sorted(glob.glob(str(OUT / TIME[fs] / m / 'fold*_seed*.npz'))); T = [np.load(f) for f in tl if 't_fit' in np.load(f)]
        tf_ = np.array([t['t_fit'] for t in T]) if T else np.array([np.nan])
        rows.append(dict(fs=fs, model=m, runs=len(fl), train_r2=R.tr.mean(), val_r2=R.va.mean(), test_r2=R.te.mean(), test_r2_sd=R.te.std(), pearson=R.pe.mean(), spearman=R.sp.mean(),
                         gap=(R.tr - R.te).mean(), avg_runs_r2=metrics(z0['y'][te], ens)['r2'], t_feat=np.mean([t['t_feat'] for t in T]) if T else np.nan,
                         t_fit=tf_.mean(), t_fit_sd=tf_.std(), t_total=tf_.mean() * len(fl), t_inf1k=np.mean([t['t_inf1k'] for t in T]) if T else np.nan, n_timed=len(T)))
        # scatter test fold0 seed0
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
        for a, idx, lab in [(ax[0], z0['tr'], 'train'), (ax[1], te, 'test')]:
            a.scatter(z0['y'][idx], z0['pred'][idx], s=2, alpha=.3); lo, hi = z0['y'].min(), z0['y'].max(); a.plot([lo, hi], [lo, hi], 'k--', lw=1)
            a.set(title=f"{NM[m]} [{fs}] {lab} R²={metrics(z0['y'][idx], z0['pred'][idx])['r2']:.3f}", xlabel='log affinity', ylabel='prediction')
        plt.tight_layout(); plt.savefig(P / f'scatter_{fs.replace("+", "_")}_{m}.png', dpi=100); plt.close()
        hs = sorted(glob.glob(str(OUT / d / m / '*_hist.npy')))
        if hs:
            fig, a = plt.subplots(figsize=(4.5, 3.2))
            for h in hs: H = np.load(h, allow_pickle=True); a.plot(H[0], 'b', alpha=.3, lw=1); a.plot(H[1], 'r', alpha=.3, lw=1)
            a.set(title=f'{NM[m]} [{fs}] {len(hs)} runs', xlabel='epoch', ylabel='MSE'); a.set_yscale('log'); plt.tight_layout(); plt.savefig(P / f'curves_{fs.replace("+", "_")}_{m}.png', dpi=100); plt.close()
    for name, ms in [('ens_all_nn', NN), ('ens_cnn', ['cnn1d', 'cnnattention'])]:
        if all(m in ENS for m in ms):
            yt = ENS[ms[0]][0]; mt = metrics(yt, np.mean([ENS[m][1] for m in ms], 0))
            rows.append(dict(fs=fs, model=name, runs=len(ms), test_r2=mt['r2'], pearson=mt['pearson'], spearman=mt['spearman'], avg_runs_r2=mt['r2']))
B = pd.DataFrame(rows); B.to_csv(OUT / 'benchmark_featuresets.csv', index=False); pd.set_option('display.width', 250); print(B.round(3).to_string())
order = MODELS + ['ens_all_nn', 'ens_cnn']; fss = list(FS)
fig, a = plt.subplots(figsize=(11, 4)); w = .27
for i, fs in enumerate(fss):
    s = B[B.fs == fs].set_index('model').reindex(order); a.bar(np.arange(len(order)) + (i - 1) * w, s.test_r2, w, yerr=s.test_r2_sd.fillna(0), label=fs)
a.set_xticks(range(len(order))); a.set_xticklabels([NM[m] for m in order], rotation=30, ha='right'); a.set_ylabel('test R² (mean±sd over runs)'); a.set_ylim(0.3, 0.8); a.legend(); plt.tight_layout(); plt.savefig(P / 'grouped_bar.png', dpi=120); plt.close()
fig, a = plt.subplots(figsize=(7, 4.5)); mk = {'tetramer': 'o', 'hexamer': 's', 'tetra+hexa': '^'}
for fs in fss:
    s = B[(B.fs == fs) & B.t_fit.notna()]; a.scatter(s.t_fit, s.test_r2, marker=mk[fs], label=fs)
    for _, r in s.iterrows(): a.annotate(NM[r.model], (r.t_fit, r.test_r2), fontsize=6)
a.set_xscale('log'); a.set(xlabel='training time per run [s] (1 thread for NNs/linear/SVR)', ylabel='test R²'); a.legend(); plt.tight_layout(); plt.savefig(P / 'time_vs_r2.png', dpi=120); plt.close()
f = lambda v, n=3: '--' if pd.isna(v) else f'{v:.{n}f}'
L = []
for m in order:
    cells = [NM[m]]
    for fs in fss:
        r = B[(B.fs == fs) & (B.model == m)]
        if len(r) == 0: cells += ['--'] * 4; continue
        r = r.iloc[0]; best = B[B.model == m].test_r2.max()
        t = (f'{r.test_r2:.3f}$\\pm${r.test_r2_sd:.3f}' if not pd.isna(r.get('test_r2_sd')) else f'{r.test_r2:.3f}')
        cells += [('\\textbf{%s}' % t) if r.test_r2 == best else t, f(r.pearson), f(r.gap), f(r.avg_runs_r2)]
    L.append(' & '.join(cells) + r' \\')
open(OUT / 'report' / 'tab_bench.tex', 'w').write('\n'.join(L))
L = []
for m in MODELS:
    cells = [NM[m]]
    for fs in fss:
        r = B[(B.fs == fs) & (B.model == m)]
        cells += ['--'] * 3 if len(r) == 0 else [f(r.iloc[0].t_feat, 2), f'{r.iloc[0].t_fit:.1f}$\\pm${r.iloc[0].t_fit_sd:.1f}', f(r.iloc[0].t_inf1k, 2)]
    L.append(' & '.join(cells) + r' \\')
open(OUT / 'report' / 'tab_time.tex', 'w').write('\n'.join(L))
L = []
for m in MODELS:
    L.append(NM[m] + ' & ' + ' & '.join(f(B[(B.fs == fs) & (B.model == m)].t_total.iloc[0] / 60, 1) if len(B[(B.fs == fs) & (B.model == m)]) else '--' for fs in fss) + r' \\')
open(OUT / 'report' / 'tab_time_total.tex', 'w').write('\n'.join(L))

import json, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lib import *
F = OUT / 'final'; rows = []; fig_dir = OUT / 'plots'; fig_dir.mkdir(exist_ok=True)
MODELS = ['linear', 'ridge', 'svr', 'rf', 'lightgbm', 'dfn', 'cnn1d', 'cnnattention', 'transformer']; NN = ['dfn', 'cnn1d', 'cnnattention', 'transformer']; ENS = {}
for model in MODELS:
    fs = sorted(glob.glob(str(F / model / 'fold*_seed*.npz')))
    if not fs: continue
    R = []
    for f in fs:
        z = np.load(f); p, y = z['pred'], z['y']
        R.append({**{f'train_{k}': v for k, v in metrics(y[z['tr']], p[z['tr']]).items()},
                  **{f'val_{k}': v for k, v in metrics(y[z['va']], p[z['va']]).items()},
                  **{f'test_{k}': v for k, v in metrics(y[z['te']], p[z['te']]).items()}, 'best': int(z['best'])})
    R = pd.DataFrame(R); R['gap'] = R.train_r2 - R.test_r2
    row = {'model': model, 'runs': len(R)}
    for s in ['train', 'val', 'test']:
        for m in ['r2', 'pearson', 'spearman']: row[f'{s}_{m}'] = f"{R[f'{s}_{m}'].mean():.3f}±{R[f'{s}_{m}'].std():.3f}"
    row['gap'] = f"{R.gap.mean():.3f}±{R.gap.std():.3f}"
    # seed/fold ensemble on test
    z0 = np.load(fs[0]); ens = np.mean([np.load(f)['pred'][z0['te']] for f in fs], 0)
    row['test_ens_r2'] = round(metrics(z0['y'][z0['te']], ens)['r2'], 3); rows.append(row); ENS[model] = (z0['y'][z0['te']], ens)
    # scatter (fold0 seed0)
    z = z0; fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, idx, lab in [(ax[0], z['tr'], 'train'), (ax[1], z['te'], 'test')]:
        r2 = metrics(z['y'][idx], z['pred'][idx])['r2']
        a.scatter(z['y'][idx], z['pred'][idx], s=3, alpha=.3); lo, hi = z['y'].min(), z['y'].max(); a.plot([lo, hi], [lo, hi], 'k--', lw=1)
        a.set(xlabel='log affinity (truth)', ylabel='prediction', title=f'{model} {lab} R²={r2:.3f}')
    plt.tight_layout(); plt.savefig(fig_dir / f'{model}_truth_vs_pred.png', dpi=120); plt.close()
    hs = sorted(glob.glob(str(F / model / '*_hist.npy')))
    if hs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for h in hs:
            H = np.load(h, allow_pickle=True); ax.plot(H[0], 'b', alpha=.3, lw=1); ax.plot(H[1], 'r', alpha=.3, lw=1)
        ax.plot([], [], 'b', label='train'); ax.plot([], [], 'r', label='val (early-stop split)'); ax.legend(); ax.set(xlabel='epoch', ylabel='MSE (standardized log aff)', title=f'{model}: {len(hs)} runs'); ax.set_yscale('log')
        plt.tight_layout(); plt.savefig(fig_dir / f'{model}_learning_curves.png', dpi=120); plt.close()
        row['best_epoch_mean'] = float(R.best.mean()); row['epochs_mean'] = float(np.mean([len(np.load(h)[0]) for h in hs]))
        row['final_train_mse'] = float(np.mean([np.load(h)[0][-1] for h in hs])); row['best_val_mse'] = float(np.mean([np.load(h)[1].min() for h in hs]))
for name, ms in [('ens_all_nn', NN), ('ens_cnn1d+cnnattention', ['cnn1d', 'cnnattention'])]:
    if all(m in ENS for m in ms):
        yt = ENS[ms[0]][0]; p = np.mean([ENS[m][1] for m in ms], 0); mt = metrics(yt, p)
        rows.append({'model': name, 'runs': len(ms), 'test_r2': f"{mt['r2']:.3f}", 'test_pearson': f"{mt['pearson']:.3f}", 'test_spearman': f"{mt['spearman']:.3f}", 'test_ens_r2': round(mt['r2'], 3)})
        fig, a = plt.subplots(figsize=(5, 4.5)); a.scatter(yt, p, s=3, alpha=.3); a.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'k--', lw=1)
        a.set(xlabel='log affinity (truth)', ylabel='prediction', title=f'{name} test R²={mt["r2"]:.3f}'); plt.tight_layout(); plt.savefig(fig_dir / f'{name}_truth_vs_pred.png', dpi=120); plt.close()
T = pd.DataFrame(rows)
fig, a = plt.subplots(figsize=(9, 4)); tm = [float(str(v).split('±')[0]) for v in T.test_r2]; tsd = [float(str(v).split('±')[1]) if '±' in str(v) else 0 for v in T.test_r2]
a.bar(np.arange(len(T)) - .2, tm, .4, yerr=tsd, label='test R² (mean±sd runs)'); a.bar(np.arange(len(T)) + .2, T.test_ens_r2, .4, label='test R² averaged runs')
a.set_xticks(range(len(T))); a.set_xticklabels(T.model, rotation=30, ha='right'); a.set_ylabel('R² (log affinity)'); a.legend(); plt.tight_layout(); plt.savefig(fig_dir / 'model_comparison.png', dpi=120); plt.close(); T.to_csv(OUT / 'results_final.csv', index=False); print(T.to_markdown(index=False))
A = [json.load(open(f)) for f in glob.glob(str(OUT / 'ablation/*.json'))]
A = pd.DataFrame([{**a['cfg'], 'model': a['model'], 'r2': f"{a['r2_mean']:.3f}±{a['r2_sd']:.3f}", 'pearson': round(a['pearson_mean'], 3), 'spearman': round(a['spearman_mean'], 3)} for a in A]).sort_values(['model', 'r2'], ascending=[True, False])
A.to_csv(OUT / 'ablation_results.csv', index=False); print(A.to_markdown(index=False))

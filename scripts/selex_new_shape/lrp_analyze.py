"""Aggregate LRP relevance (test rows, mean over all fold x seed models): feature types, windows/positions, base x position maps, logos,
signed vs absolute, conservation, LRP vs gradient x input. Writes report/tab_lrp*.tex and plots/lrp/*.png, lrp_summary.json."""
import json, re, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, logomaker
from lib import *
P = OUT / 'plots' / 'lrp'; P.mkdir(parents=True, exist_ok=True)
PAR = [l.split() for l in open(ROOT / 'data/selex/tetramer/avg_tetramer.dat')][0][1:]  # SHIFT SLIDE RISE TILT ROLL TWIST
FCE = [f'{PAR[i]}-{PAR[j]}' for i, j in zip(*IU)]; NUC = 'ACGT'
S = {}
def load(od, m):
    z = np.load(OUT / od / f'lrp_{m}.npz'); return z
def per_base(R, kind, shape):
    """-> (n,10,4) one-hot relevance per base position & nucleotide channel; plus shape groups."""
    n = len(R); B = np.zeros((n, 10, 4))
    if kind == 'cnn':
        o = 27 if shape else 0
        for i in range(7):
            for k in range(4): B[:, i + k] += R[:, i, o + 4 * k:o + 4 * k + 4]
    else:
        o = 189 if shape else 0; B = R[:, o:o + 40].reshape(n, 10, 4)
    return B
def groups(R, kind, shape):
    """dict group -> (n,) summed relevance; and per-window shape relevance (n,7)."""
    n = len(R); g = {}
    if kind == 'cnn':
        g['one-hot'] = R[:, :, 27 if shape else 0:].sum((1, 2))
        if shape:
            for p in range(6): g['avg ' + PAR[p]] = R[:, :, p].sum(1)
            dg = [6 + j for j, (a, b) in enumerate(zip(*IU)) if a == b]; od = [6 + j for j, (a, b) in enumerate(zip(*IU)) if a != b]
            g['fce diagonal'] = R[:, :, dg].sum((1, 2)); g['fce off-diagonal'] = R[:, :, od].sum((1, 2))
            fce = R[:, :, 6:27].sum(1); win_shape = R[:, :, :27].sum(2)
        else: fce = None; win_shape = None
    else:
        g['one-hot'] = R[:, 189 if shape else 0:][:, :40].sum(1)
        if shape:
            A = R[:, :42].reshape(n, 7, 6); F = R[:, 42:189].reshape(n, 7, 21)
            for p in range(6): g['avg ' + PAR[p]] = A[:, :, p].sum(1)
            dg = [j for j, (a, b) in enumerate(zip(*IU)) if a == b]; od = [j for j in range(21) if j not in dg]
            g['fce diagonal'] = F[:, :, dg].sum((1, 2)); g['fce off-diagonal'] = F[:, :, od].sum((1, 2))
            fce = F.sum(1); win_shape = A.sum(2) + F.sum(2)
        else: fce = None; win_shape = None
    return g, fce, win_shape
def absgroups(R, kind, shape):  # same grouping on |R|
    return groups(np.abs(R), kind, shape)
RUNS = [('DFN (a)', 'lrp_a', 'dfn', 'dfn', True), ('CNN1D (a)', 'lrp_a', 'cnn1d', 'cnn', True), ('CNN1D (d)', 'lrp_d', 'cnn1d', 'cnn', False)]
summ, grows, cons_rows = {}, [], []
for name, od, m, kind, shape in RUNS:
    z = load(od, m); Rall, out, G, R0 = z['R'], z['out'], z['G'], z['R0']; seqs = z['seqs']; y = z['y']; nr = len(Rall)
    # conservation per run
    sr = Rall.reshape(nr, Rall.shape[1], -1).sum(2)
    cs = [np.corrcoef(sr[r], out[r])[0, 1] for r in range(nr)]; sl = [np.polyfit(out[r], sr[r], 1)[0] for r in range(nr)]
    big = np.abs(out) > 0.5; ratio = np.median((sr / out)[big])
    lz = [np.abs(R0[r] - G[r]).max() / np.abs(G[r]).max() for r in range(nr)]
    cg = [np.corrcoef(Rall[r].ravel(), G[r].ravel())[0, 1] for r in range(nr)]
    R = Rall.mean(0)  # mean over models, per test row
    gs, fce, ws = groups(R, kind, shape); ga, fcea, wsa = absgroups(R, kind, shape)
    totA = sum(v.mean() for v in ga.values())
    for k in gs: grows.append(dict(model=name, group=k, abs_share=ga[k].mean() / totA, signed_mean=gs[k].mean(), signed_top=gs[k][np.argsort(-y)[:100]].mean()))
    cons_rows.append(dict(model=name, runs=nr, corr=np.mean(cs), slope=np.mean(sl), ratio=ratio, lrp0_vs_gxi=np.max(lz), corr_lrp_gxi=np.mean(cg)))
    # per-base / position
    B = per_base(R, kind, shape); Ba = per_base(np.abs(R), kind, shape)
    pos_abs = Ba.sum(2).mean(0)
    present = np.array([[NUC.index(ch) for ch in s] for s in seqs])
    Bp = np.take_along_axis(B, present[:, :, None], 2)[:, :, 0]  # relevance on present-base channel
    Btot = B.sum(2)  # all 4 channels of the position (absent channels are z-scored nonzero)
    top = np.argsort(-y)[:50]
    # GATA core check: top k-mers containing GATA (or TATC on this strand)
    core = []
    for i in top:
        s = seqs[i]; mo = re.search('GATA', s) or re.search('TATC', s)
        if mo: a = np.abs(Btot[i]); core.append(a[mo.start():mo.start() + 4].sum() / a.sum())
    summ[name] = dict(pos_abs=pos_abs.tolist(), win_shape_abs=(wsa.mean(0).tolist() if wsa is not None else None),
                      top50_with_core=len(core), core_share_mean=float(np.mean(core)) if core else None,
                      fce_top=(sorted(zip(FCE, (fcea.mean(0) / totA).tolist()), key=lambda t: -t[1])[:5] if fcea is not None else None))
    fn = name.replace(' ', '_').replace('(', '').replace(')', '')
    # heatmap nucleotide x position, mean signed relevance on each channel over top-50 k-mers, and over all test rows
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.6))
    for a, idx, t in [(ax[0], top, 'top-50 affinity k-mers'), (ax[1], slice(None), 'all test rows')]:
        H = B[idx].mean(0).T; v = np.abs(H).max(); im = a.imshow(H, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
        a.set_yticks(range(4)); a.set_yticklabels(list(NUC)); a.set_xticks(range(10)); a.set_xticklabels(range(1, 11)); a.set(xlabel='base position', title=f'{name}: {t}'); plt.colorbar(im, ax=a)
    plt.tight_layout(); plt.savefig(P / f'heat_{fn}.png', dpi=120); plt.close()
    # logos for top 6 k-mers: letter height = signed relevance on present-base channel
    fig, ax = plt.subplots(2, 3, figsize=(11, 4))
    for a, i in zip(ax.ravel(), top[:6]):
        M = pd.DataFrame(0.0, index=range(10), columns=list(NUC))
        for j, ch in enumerate(seqs[i]): M.loc[j, ch] = Btot[i, j]
        logomaker.Logo(M, ax=a, color_scheme='classic'); a.axhline(0, c='k', lw=.5); a.set_title(f'{seqs[i]}  logAff={y[i]:.2f}', fontsize=8); a.set_xticks([])
    fig.suptitle(f'{name}: LRP relevance per base (sum over the 4 channels of the position), top-6 test k-mers'); plt.tight_layout(); plt.savefig(P / f'logo_{fn}.png', dpi=120); plt.close()
    # position profile + window shape
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    ax[0].bar(range(1, 11), Ba.sum(2).mean(0)); ax[0].set(xlabel='base position', ylabel='mean |R| (one-hot)', title=f'{name}: one-hot per position')
    if wsa is not None: ax[1].bar(range(1, 8), wsa.mean(0)); ax[1].set(xlabel='tetramer window', ylabel='mean |R| (shape)', title=f'{name}: shape per window')
    else: ax[1].axis('off')
    plt.tight_layout(); plt.savefig(P / f'pos_{fn}.png', dpi=120); plt.close()
Gd = pd.DataFrame(grows); C = pd.DataFrame(cons_rows); print(Gd.round(4).to_string()); print(C.round(4).to_string()); print(json.dumps(summ, indent=1))
Gd.to_csv(OUT / 'lrp_groups.csv', index=False); C.to_csv(OUT / 'lrp_conservation.csv', index=False); json.dump(summ, open(OUT / 'lrp_summary.json', 'w'), indent=1)
# feature-type bar (abs share) for set (a)
fig, a = plt.subplots(figsize=(9, 3.4)); gs_ = [g for g in Gd.group.unique()]; w = .38
for i, mname in enumerate(['DFN (a)', 'CNN1D (a)']):
    s = Gd[Gd.model == mname].set_index('group').reindex(gs_); a.bar(np.arange(len(gs_)) + (i - .5) * w, s.abs_share, w, label=mname)
a.set_xticks(range(len(gs_))); a.set_xticklabels(gs_, rotation=30, ha='right'); a.set_ylabel('share of total |R|'); a.legend(); plt.tight_layout(); plt.savefig(P / 'groups_abs.png', dpi=120); plt.close()
f3 = lambda v: f'{v:.3f}'
L = []
for g in gs_:
    cells = [g.replace('avg ', '\\texttt{avg} ').replace('fce ', '\\texttt{fce} ')]
    for mname in ['DFN (a)', 'CNN1D (a)', 'CNN1D (d)']:
        r = Gd[(Gd.model == mname) & (Gd.group == g)]
        cells += ['--'] * 3 if len(r) == 0 else [f3(r.abs_share.iloc[0]), f'{r.signed_mean.iloc[0]:+.3f}', f'{r.signed_top.iloc[0]:+.3f}']
    L.append(' & '.join(cells) + r' \\')
open(OUT / 'report' / 'tab_lrp_groups.tex', 'w').write('\n'.join(L))
L = [f"{r.model} & {r.runs} & {r.corr:.3f} & {r.slope:.3f} & {r.ratio:.3f} & {r.lrp0_vs_gxi:.1e} & {r.corr_lrp_gxi:.3f} \\\\" for r in C.itertuples()]
open(OUT / 'report' / 'tab_lrp_cons.tex', 'w').write('\n'.join(L))

"""Aggregate LRP relevance (test rows, mean over all fold x seed models): feature types, windows/positions, base x position maps, logos,
signed vs absolute, conservation, LRP vs gradient x input. Writes report/tab_lrp*.tex and plots/lrp/*.png, lrp_summary.json."""
import json, re, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, logomaker
from lib import *
P = OUT / 'plots' / 'lrp'; P.mkdir(parents=True, exist_ok=True)
from lrp_agg import PAR, FCE, NUC, per_base, groups, absgroups
S = {}
def load(od, m):
    z = np.load(OUT / od / f'lrp_{m}.npz'); return z
RUNS = [('DFN (a)', 'lrp_a', 'dfn', 'dfn', True, True), ('CNN1D (a)', 'lrp_a', 'cnn1d', 'cnn', True, True), ('CNN1D (d)', 'lrp_d', 'cnn1d', 'cnn', False, True),
        ('DFN (e)', 'final_e', 'dfn', 'dfn', True, False), ('CNN1D (e)', 'final_e', 'cnn1d', 'cnn', True, False)]
summ, grows, cons_rows = {}, [], []
for name, od, m, kind, shape, oh in RUNS:
    z = load(od, m); Rall, out, G, R0 = z['R'], z['out'], z['G'], z['R0']; seqs = z['seqs']; y = z['y']; nr = len(Rall)
    # conservation per run
    sr = Rall.reshape(nr, Rall.shape[1], -1).sum(2)
    cs = [np.corrcoef(sr[r], out[r])[0, 1] for r in range(nr)]; sl = [np.polyfit(out[r], sr[r], 1)[0] for r in range(nr)]
    big = np.abs(out) > 0.5; ratio = np.median((sr / out)[big])
    lz = [np.abs(R0[r] - G[r]).max() / np.abs(G[r]).max() for r in range(nr)]
    cg = [np.corrcoef(Rall[r].ravel(), G[r].ravel())[0, 1] for r in range(nr)]
    R = Rall.mean(0)  # mean over models, per test row
    gs, fce, ws = groups(R, kind, shape, oh); ga, fcea, wsa = absgroups(R, kind, shape, oh)
    totA = sum(v.mean() for v in ga.values())
    for k in gs: grows.append(dict(model=name, group=k, abs_share=ga[k].mean() / totA, signed_mean=gs[k].mean(), signed_top=gs[k][np.argsort(-y)[:100]].mean()))
    cons_rows.append(dict(model=name, runs=nr, corr=np.mean(cs), slope=np.mean(sl), ratio=ratio, lrp0_vs_gxi=np.max(lz), corr_lrp_gxi=np.mean(cg)))
    # per-base / position
    B = per_base(R, kind, shape, oh, seqs); Ba = per_base(np.abs(R), kind, shape, oh, seqs)
    pos_abs = Ba.sum(2).mean(0)
    present = np.array([[NUC.index(ch) for ch in s] for s in seqs])
    Bp = np.take_along_axis(B, present[:, :, None], 2)[:, :, 0]  # relevance on present-base channel
    Btot = B.sum(2)  # all 4 channels of the position (absent channels are z-scored nonzero)
    top = np.argsort(-y)[:50]
    # GATA core check: top k-mers containing GATA (or TATC on this strand)
    def core_share(pats, idx):
        out = []
        for i in idx:
            mo = None
            for p in pats:
                mo = mo or re.search(p, seqs[i])
            if mo:
                a_ = np.abs(Btot[i]); sl_ = slice(mo.start(), mo.start() + len(mo.group()))
                out.append((a_[sl_].sum() / a_.sum(), prof[sl_].sum()))
        if not out: return 0, None, None
        o_ = np.array(out); return len(o_), float(o_[:, 0].mean()), float(o_[:, 1].mean())  # n, observed share, positional expectation
    prof = np.abs(Btot).mean(0); prof = prof / prof.sum()
    top100 = np.argsort(-y)[:100]
    motif = {k: core_share(p, top100) for k, p in [('GGAA/TTCC (ETS core)', ['GGAA', 'TTCC']), ('GATA/TATC', ['GATA', 'TATC'])]}
    allm = {k: core_share(p, range(len(y))) for k, p in [('GGAA/TTCC (ETS core)', ['GGAA', 'TTCC']), ('GATA/TATC', ['GATA', 'TATC'])]}
    summ[name] = dict(pos_abs=pos_abs.tolist(), win_shape_abs=(wsa.mean(0).tolist() if wsa is not None else None),
                      motif_top100=motif, motif_all=allm,
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
    ax[0].bar(range(1, 11), Ba.sum(2).mean(0)); ax[0].set(xlabel='base position', ylabel='mean |R| (one-hot)' if oh else 'mean |R| (projected from windows)', title=f'{name}: ' + ('one-hot per position' if oh else 'window relevance projected to bases'))
    if wsa is not None: ax[1].bar(range(1, 8), wsa.mean(0)); ax[1].set(xlabel='tetramer window', ylabel='mean |R| (shape)', title=f'{name}: shape per window')
    else: ax[1].axis('off')
    plt.tight_layout(); plt.savefig(P / f'pos_{fn}.png', dpi=120); plt.close()
Gd = pd.DataFrame(grows); C = pd.DataFrame(cons_rows); print(Gd.round(4).to_string()); print(C.round(4).to_string()); print(json.dumps(summ, indent=1))
Gd.to_csv(OUT / 'lrp_groups.csv', index=False); C.to_csv(OUT / 'lrp_conservation.csv', index=False); json.dump(summ, open(OUT / 'lrp_summary.json', 'w'), indent=1)
MN = ['DFN (a)', 'CNN1D (a)', 'CNN1D (d)', 'DFN (e)', 'CNN1D (e)']
gs_ = list(dict.fromkeys(Gd.group))
fig, a = plt.subplots(figsize=(10, 3.6)); w = .8 / len(MN)
for i, mname in enumerate(MN):
    s_ = Gd[Gd.model == mname].set_index('group').reindex(gs_); a.bar(np.arange(len(gs_)) + (i - 2) * w, s_.abs_share.fillna(0), w, label=mname, hatch='//' if '(e)' in mname else None, edgecolor='k', lw=.3)
a.set_xticks(range(len(gs_))); a.set_xticklabels(gs_, rotation=30, ha='right'); a.set_ylabel('share of total |R|'); a.legend(fontsize=7); plt.tight_layout(); plt.savefig(P / 'groups_abs.png', dpi=120); plt.close()
fig, a = plt.subplots(figsize=(10, 3.6))
for i, mname in enumerate(MN):
    s_ = Gd[Gd.model == mname].set_index('group').reindex(gs_); a.bar(np.arange(len(gs_)) + (i - 2) * w, s_.signed_top.fillna(0), w, label=mname, hatch='//' if '(e)' in mname else None, edgecolor='k', lw=.3)
a.axhline(0, c='k', lw=.5); a.set_xticks(range(len(gs_))); a.set_xticklabels(gs_, rotation=30, ha='right'); a.set_ylabel('mean signed R, top-100 k-mers'); a.legend(fontsize=7); plt.tight_layout(); plt.savefig(P / 'groups_signed.png', dpi=120); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for mname in ['DFN (a)', 'DFN (e)', 'CNN1D (a)', 'CNN1D (e)']:
    v = np.array(summ[mname]['win_shape_abs']); ax[0].plot(range(1, 8), v / v.sum(), 'o-', label=mname)
    p = np.array(summ[mname]['pos_abs']); ax[1].plot(range(1, 11), p / p.sum(), 'o-', label=mname)
p = np.array(summ['CNN1D (d)']['pos_abs']); ax[1].plot(range(1, 11), p / p.sum(), 'o-', label='CNN1D (d)')
ax[0].set(xlabel='tetramer window', ylabel='share of shape |R|', title='shape relevance per window'); ax[1].set(xlabel='base position', ylabel='share', title='per-base relevance (one-hot, or projected for (e))'); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(P / 'window_compare.png', dpi=120); plt.close()
L, L2 = [], []
for g in gs_:
    nm = g.replace('avg ', '\\texttt{avg} ').replace('fce ', '\\texttt{fce} ')
    row = lambda k, fmt: [('--' if len(Gd[(Gd.model == mn) & (Gd.group == g)]) == 0 else fmt(Gd[(Gd.model == mn) & (Gd.group == g)][k].iloc[0])) for mn in MN]
    L.append(' & '.join([nm] + row('abs_share', lambda v: f'{v:.3f}')) + r' \\')
    L2.append(' & '.join([nm] + row('signed_mean', lambda v: f'{v:+.3f}') + row('signed_top', lambda v: f'{v:+.3f}')) + r' \\')
open(OUT / 'report' / 'tab_lrp_groups.tex', 'w').write('\n'.join(L)); open(OUT / 'report' / 'tab_lrp_signed.tex', 'w').write('\n'.join(L2))
L = [f"{r.model} & {r.runs} & {r.corr:.3f} & {r.slope:.3f} & {r.ratio:.3f} & {r.lrp0_vs_gxi:.1e} & {r.corr_lrp_gxi:.3f} \\\\" for r in C.itertuples()]
open(OUT / 'report' / 'tab_lrp_cons.tex', 'w').write('\n'.join(L))

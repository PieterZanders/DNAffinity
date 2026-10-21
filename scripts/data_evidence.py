#!/usr/bin/env python3
"""Evidence plots/tables for the uPBM and SELEX data issues -> outputs/data_evidence/."""
import itertools
import json
from pathlib import Path

import logomaker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr, ks_2samp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'outputs' / 'data_evidence'
O.mkdir(parents=True, exist_ok=True)
DPI = 150
rc = lambda s: s[::-1].translate(str.maketrans('ACGT', 'TGCA'))
canon = lambda s: min(s, rc(s))
K = {}

def save(fig, name):
    fig.tight_layout(); fig.savefig(O / name, dpi=DPI); plt.close(fig)

# ======================= uPBM =======================
up = pd.read_csv(ROOT / 'data/upbm/Gata4/Gata4_training.txt', sep='\t')
up['canon'] = up.ID_REF.map(canon)
pairs = []
for c, g in up.groupby('canon'):
    if len(g) < 2:
        continue
    for (i, a), (j, b) in itertools.combinations(g.iterrows(), 2):
        kind = 'exact' if a.ID_REF == b.ID_REF else 'reverse-complement'
        pairs.append(dict(canon=c, seq_a=a.ID_REF, seq_b=b.ID_REF, kind=kind, y_a=a.VALUE, y_b=b.VALUE,
                          identical_value=bool(np.isclose(a.VALUE, b.VALUE, rtol=0, atol=1e-12))))
P = pd.DataFrame(pairs); P.to_csv(O / 'upbm_duplicate_pairs.csv', index=False)

def corr(df):
    return (len(df), pearsonr(df.y_a, df.y_b)[0], spearmanr(df.y_a, df.y_b)[0]) if len(df) > 2 else (len(df), np.nan, np.nan)
subsets = {'all pairs': P, 'exact duplicates': P[P.kind == 'exact'], 'RC pairs': P[P.kind == 'reverse-complement'],
           'non-identical values': P[~P.identical_value]}
rows = []
for k, d in subsets.items():
    n, r, rho = corr(d)
    rows.append(dict(subset=k, n_pairs=n, pearson=r, spearman=rho,
                     frac_identical_value=float(d.identical_value.mean()) if n else np.nan,
                     median_abs_diff=float((d.y_a - d.y_b).abs().median()) if n else np.nan))
DS = pd.DataFrame(rows)
grp = up.groupby('canon').VALUE
gstats = pd.DataFrame({'n': grp.size(), 'mean': grp.mean(), 'std': grp.std(ddof=0), 'range': grp.max() - grp.min()})
dup = gstats[gstats.n > 1]
summary_tbl = pd.DataFrame([
    ('rows in Gata4_training.txt', len(up)), ('unique sequences (exact)', up.ID_REF.nunique()),
    ('unique canonical sequences', up.canon.nunique()), ('rows that are exact duplicates', int(up.ID_REF.duplicated().sum())),
    ('canonical groups with >1 row', len(dup)), ('rows in duplicated groups', int(dup.n.sum())),
    ('median label (all)', float(up.VALUE.median())), ('median within-group range (dups)', float(dup['range'].median())),
    ('median within-group std (dups)', float(dup['std'].median())),
    ('groups with range > median label', int((dup['range'] > up.VALUE.median()).sum())),
], columns=['statistic', 'value'])
summary_tbl.to_csv(O / 'upbm_duplicate_summary.csv', index=False)
DS.to_csv(O / 'upbm_duplicate_pair_correlations.csv', index=False)
K['upbm_pairs'] = DS.set_index('subset').to_dict(orient='index')

# (1) scatter
fig, ax = plt.subplots(figsize=(5.5, 5))
for kind, col in [('exact', 'tab:blue'), ('reverse-complement', 'tab:orange')]:
    d = P[P.kind == kind]; n, r, rho = corr(d)
    ax.scatter(d.y_a, d.y_b, s=10, alpha=.6, c=col, label=f'{kind}: n={n}, r={r:.2f}, ρ={rho:.2f}')
n, r, rho = corr(P)
ax.plot([0, 1], [0, 1], 'k--', lw=.8)
ax.set_xlabel('label, copy A'); ax.set_ylabel('label, copy B')
ax.set_title(f'uPBM Gata4: same 10-mer measured twice\nall pairs n={n}, Pearson r={r:.2f}, Spearman ρ={rho:.2f}')
ax.legend(fontsize=8); save(fig, 'upbm_1_duplicate_pair_scatter.png')

# (2) spread histogram
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].hist(dup['std'], 50, color='tab:purple'); ax[0].axvline(up.VALUE.median(), c='r', ls='--', label=f'median label {up.VALUE.median():.3f}')
ax[0].set_xlabel('within-sequence label std'); ax[0].set_ylabel('# duplicated sequences'); ax[0].legend()
ax[0].set_title(f'{len(dup)} duplicated canonical 10-mers')
ax[1].scatter(dup['mean'], dup['range'], s=8, alpha=.5); ax[1].set_xlabel('mean label'); ax[1].set_ylabel('max − min label')
ax[1].set_title('spread vs mean (noise grows with signal)'); save(fig, 'upbm_2_duplicate_spread_hist.png')

# (3) label distribution artifacts
vc = up.VALUE.round(12).value_counts()
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].hist(up.VALUE, 200, color='gray'); ax[0].set_yscale('log'); ax[0].set_xlabel('label (min-max normalised)'); ax[0].set_ylabel('count (log)')
ax[0].set_title(f'skewed; {int((up.VALUE == 1).sum())} at max, {int((up.VALUE < 1e-3).sum())} < 1e-3')
ax[1].bar(*np.unique(vc.values, return_counts=True), color='tab:red'); ax[1].set_yscale('log')
ax[1].set_xlabel('times an exact label value occurs'); ax[1].set_ylabel('# distinct values (log)')
ax[1].set_title(f'{len(vc)} distinct values in {len(up)} rows: no quantisation, no copied labels')
srt = np.sort(up.VALUE.values)
ax[2].plot(np.arange(len(srt)) / len(srt), srt); ax[2].set_xlabel('rank quantile'); ax[2].set_ylabel('label')
ax[2].set_title('sorted labels: long upper tail (bin-undersampled)')
save(fig, 'upbm_3_label_distribution_artifacts.png')
K['upbm_labels'] = dict(n_distinct_values=int(len(vc)), n_values_repeated=int((vc > 1).sum()), max_repeat=int(vc.max()),
                        rows_with_repeated_value=int(vc[vc > 1].sum()), skew=float(pd.Series(up.VALUE).skew()))

# (4) noise ceiling
ind = P[~P.identical_value]
r1 = pearsonr(ind.y_a, ind.y_b)[0]; rho1 = spearmanr(ind.y_a, ind.y_b)[0]
kbar = up.groupby('canon').size().mean()
rel_k = kbar * r1 / (1 + (kbar - 1) * r1)          # Spearman-Brown reliability of the aggregated label
ceil_r = np.sqrt(rel_k)                            # max r of a perfect predictor with the noisy label
rel_k_s = kbar * rho1 / (1 + (kbar - 1) * rho1); ceil_s = np.sqrt(rel_k_s)
K['upbm_ceiling'] = dict(single_measurement_r=r1, single_measurement_rho=rho1, mean_copies_per_seq=kbar,
                         spearman_brown_reliability=rel_k, ceiling_pearson=ceil_r, ceiling_spearman=ceil_s, n_pairs=len(ind))
fr = pd.read_csv(ROOT / 'outputs/fixed_results.csv')
u = fr[(fr.dataset == 'upbm')].copy(); u['label'] = u.model + ' · ' + u.variant + np.where(u.prep == 'legacy', ' (legacy)', '')
u = u.sort_values('pearson_mean')
fig, ax = plt.subplots(figsize=(8, 9))
y = np.arange(len(u))
ax.barh(y, u.pearson_mean, xerr=u.pearson_std, color=np.where(u.prep == 'legacy', 'lightgray', 'tab:blue'), label='CV Pearson (mean±std)')
ax.axvline(ceil_r, c='r', lw=2, label=f'noise ceiling r≈{ceil_r:.2f} (dup r={r1:.2f}, Spearman-Brown k={kbar:.2f})')
ax.axvline(r1, c='orange', ls='--', label=f'replicate agreement r={r1:.2f}')
ax.set_yticks(y); ax.set_yticklabels(u.label, fontsize=7); ax.set_xlim(0.3, 0.8); ax.set_xlabel('5-fold group-CV Pearson r')
ax.set_title('uPBM Gata4: every model is at the noise ceiling'); ax.legend(loc='lower right', fontsize=8)
save(fig, 'upbm_4_noise_ceiling_vs_models.png')

# ======================= SELEX =======================
se = pd.read_csv(ROOT / 'data/selex/Gata4/4.txt', sep='\t')
top = se[se.ObservedCount >= 10].sort_values('Affinity', ascending=False)
gata = lambda s: any(m in s for m in ['AGATAA', 'TGATAA', 'AGATAG', 'TGATAG', 'TTATCT', 'TTATCA', 'CTATCT', 'CTATCA'])
gata4 = lambda s: 'GATA' in s or 'TATC' in s
ets = lambda s: 'CCGGAA' in s or 'TTCCGG' in s
gga = lambda s: 'GGAA' in s or 'TTCC' in s
se['WGATAR'] = se.Kmer.map(gata); se['GATA_core'] = se.Kmer.map(gata4); se['ETS_CCGGAA'] = se.Kmer.map(ets); se['GGAA_core'] = se.Kmer.map(gga)
t30 = top.head(30).merge(se[['Kmer', 'WGATAR', 'ETS_CCGGAA', 'GGAA_core']], on='Kmer')
t30.to_csv(O / 'selex_top30_kmers.csv', index=False)
K['selex_top10_kmers'] = t30[['Kmer', 'ObservedCount', 'Affinity', 'ETS_CCGGAA', 'WGATAR']].head(10).to_dict(orient='records')

# enrichment table: fraction of motif-containing k-mers among top-N vs all
enr = []
srt_se = se.sort_values('Affinity', ascending=False)
reliable = srt_se[srt_se.ObservedCount >= 10]
for motif in ['WGATAR', 'GATA_core', 'ETS_CCGGAA', 'GGAA_core']:
    bg = se[motif].mean()
    row = dict(motif=motif, background_frac=bg, mean_affinity_with=se[se[motif]].Affinity.mean(),
               mean_affinity_without=se[~se[motif]].Affinity.mean())
    for n in [100, 1000]:
        f = reliable.head(n)[motif].mean(); row[f'top{n}_frac'] = f; row[f'top{n}_enrichment'] = f / bg
    enr.append(row)
E = pd.DataFrame(enr); E.to_csv(O / 'selex_motif_enrichment.csv', index=False)
K['selex_enrichment'] = E.set_index('motif').to_dict(orient='index')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(E)); w = .27
ax[0].bar(x - w, E.background_frac, w, label='all k-mers'); ax[0].bar(x, E.top1000_frac, w, label='top 1000 (count≥10)')
ax[0].bar(x + w, E.top100_frac, w, label='top 100 (count≥10)')
ax[0].set_xticks(x); ax[0].set_xticklabels(E.motif); ax[0].set_ylabel('fraction of k-mers containing motif'); ax[0].legend()
ax[0].set_title('SELEX "Gata4": ETS core enriched, GATA not')
cats = {'WGATAR (GATA)': se.WGATAR & ~se.ETS_CCGGAA, 'CCGGAA (ETS)': se.ETS_CCGGAA & ~se.WGATAR,
        'GGAA only': se.GGAA_core & ~se.ETS_CCGGAA & ~se.WGATAR, 'neither': ~se.GGAA_core & ~se.WGATAR}
ax[1].boxplot([se.loc[m, 'Affinity'] for m in cats.values()], showfliers=False)
ax[1].set_xticks(range(1, 5)); ax[1].set_xticklabels([f'{k}\nn={int(m.sum())}' for k, m in cats.items()], fontsize=8)
ax[1].set_ylabel('SELEX affinity'); ax[1].set_title('affinity by motif class (all k-mers)')
save(fig, 'selex_3_gata_vs_ets_enrichment.png')

# logos: top-200 reliable k-mers aligned on TTCCGG vs GATA consensus
def align(kmers, core, flank=2):
    out = []
    for s in kmers:
        for q in (s, rc(s)):
            i = q.find(core)
            if i >= 0:
                win = ('N' * flank + q + 'N' * flank)[i:i + len(core) + 2 * flank]
                out.append(win); break
    return out
top200 = reliable.head(200).Kmer.tolist()
al = align(top200, 'TTCCGG')
frac_ets = len(al) / 200
frac_gata_top200 = np.mean([gata(s) for s in top200])
def counts(seqs):
    L = len(seqs[0]); m = np.zeros((L, 4))
    for s in seqs:
        for i, ch in enumerate(s):
            if ch in 'ACGT': m[i, 'ACGT'.index(ch)] += 1
    return pd.DataFrame(m, columns=list('ACGT'))
gata_cons = pd.DataFrame([[.45, .05, .05, .45], [.0, .0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [.6, 0, .4, 0]], columns=list('ACGT'))
# unaligned positional PWM of the top 200 (to show no GATA even without alignment)
fig, ax = plt.subplots(3, 1, figsize=(7, 7))
logomaker.Logo(logomaker.transform_matrix(counts(al), from_type='counts', to_type='information'), ax=ax[0])
ax[0].set_title(f'top-200 SELEX k-mers aligned on TTCCGG ({frac_ets:.0%} contain it): ETS motif (A)CCGGAAGT')
logomaker.Logo(logomaker.transform_matrix(counts(top200), from_type='counts', to_type='information'), ax=ax[1])
ax[1].set_title(f'top-200 k-mers, unaligned positional PWM ({frac_gata_top200:.0%} contain WGATAR)')
logomaker.Logo(logomaker.transform_matrix(gata_cons + 1e-3, from_type='probability', to_type='information'), ax=ax[2])
ax[2].set_title('expected GATA4 consensus WGATAR (reference)')
for a in ax: a.set_ylabel('bits')
save(fig, 'selex_2_logo_top_kmers_vs_gata.png')
counts(al).to_csv(O / 'selex_top200_TTCCGG_aligned_counts.csv', index=False)
K['selex_logo'] = dict(top200_frac_TTCCGG=frac_ets, top200_frac_WGATAR=float(frac_gata_top200))

# (1) top-k-mer table figure
fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
tb = t30.head(20)[['Kmer', 'ObservedCount', 'ExpectedCount', 'Affinity', 'ETS_CCGGAA', 'WGATAR']].copy()
tb['ExpectedCount'] = tb.ExpectedCount.round(2); tb['Affinity'] = tb.Affinity.round(3)
t = ax.table(cellText=tb.values, colLabels=tb.columns, loc='center'); t.scale(1, 1.2); t.auto_set_font_size(False); t.set_fontsize(8)
ax.set_title('SELEX "Gata4" cycle 4: top 20 k-mers by affinity (count ≥ 10)')
save(fig, 'selex_1_top_kmers_table.png')

# (4) SELEX_test.txt vs uPBM
st = pd.read_csv(ROOT / 'data/selex/Gata4/SELEX_test.txt', sep='\t')
ov = set(st.ID_REF) & set(up.ID_REF)
m = st.groupby('ID_REF').VALUE.mean().to_frame('selex_test').join(up.groupby('ID_REF').VALUE.mean().to_frame('upbm'), how='inner')
ks = ks_2samp(st.VALUE, up.VALUE)
sel_kmers = set(se.Kmer)
ident = dict(rows_selex_test=len(st), rows_upbm=len(up), shared_exact_sequences=len(ov),
             frac_selex_test_seqs_in_upbm=len(ov) / st.ID_REF.nunique(),
             frac_selex_test_seqs_in_selex_4txt=len(set(st.ID_REF) & sel_kmers) / st.ID_REF.nunique(),
             pearson_shared=float(pearsonr(m.selex_test, m.upbm)[0]), frac_shared_identical_value=float(np.isclose(m.selex_test, m.upbm, atol=1e-9).mean()),
             same_weight_column_values=sorted(st.WEIGHT.unique().tolist()) == sorted(up.WEIGHT.unique().tolist()),
             median_selex_test=float(st.VALUE.median()), median_upbm=float(up.VALUE.median()),
             ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
pd.Series(ident).to_csv(O / 'selex_test_vs_upbm_identity.csv', header=['value'])
K['selex_test_vs_upbm'] = ident
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(m.upbm, m.selex_test, s=6, alpha=.5); ax[0].plot([0, 1], [0, 1], 'k--', lw=.8)
ax[0].set_xlabel('uPBM Gata4_training.txt label'); ax[0].set_ylabel('SELEX_test.txt label')
ax[0].set_title(f'{len(ov)} shared 10-mers, r={ident["pearson_shared"]:.3f}')
q = np.linspace(0, 1, 101)
ax[1].plot(np.quantile(up.VALUE, q), np.quantile(st.VALUE, q), 'o', ms=3); ax[1].plot([0, 1], [0, 1], 'k--', lw=.8)
ax[1].set_xlabel('uPBM quantiles'); ax[1].set_ylabel('SELEX_test quantiles')
ax[1].set_title(f'Q-Q: KS D={ks.statistic:.3f}; {ident["frac_selex_test_seqs_in_selex_4txt"]:.0%} of seqs in SELEX 4.txt')
save(fig, 'selex_4_selex_test_is_upbm.png')

# combined summary figure
imgs = ['upbm_1_duplicate_pair_scatter.png', 'upbm_4_noise_ceiling_vs_models.png', 'upbm_3_label_distribution_artifacts.png',
        'selex_2_logo_top_kmers_vs_gata.png', 'selex_3_gata_vs_ets_enrichment.png', 'selex_4_selex_test_is_upbm.png']
fig, axs = plt.subplots(2, 3, figsize=(24, 13))
for a, f in zip(axs.ravel(), imgs):
    a.imshow(plt.imread(O / f)); a.axis('off'); a.set_title(f, fontsize=10)
fig.suptitle('DNAffinity data evidence: uPBM label noise and SELEX "Gata4" identity issues', fontsize=16)
save(fig, 'summary_data_evidence.png')
json.dump(K, open(O / 'key_numbers.json', 'w'), indent=1, default=float)
print(json.dumps(K, indent=1, default=float))

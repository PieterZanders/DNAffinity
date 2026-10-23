"""Tables for the MD-aim analysis (verdict, permuted-table control, summary/timing tables) from benchmark_featuresets.csv."""
import numpy as np, pandas as pd
from lib import OUT
B = pd.read_csv(OUT / 'benchmark_featuresets.csv'); R = OUT / 'report'
NM = dict(linear='Linear', ridge='Ridge', svr='SVR', rf='RF', dfn='DFN', cnn1d='CNN1D', cnnattention='CNNAttention', transformer='Transformer')
g = lambda fs, m, k='test_r2': B[(B.fs == fs) & (B.model == m)][k].iloc[0]
f3 = lambda v: f'{v:.3f}'; sg = lambda v: f'{v:+.3f}'
# verdict
L = []
for m in NM:
    d, a, e = g('one-hot', m), g('tetramer', m), g('tetramer-MD', m)
    L.append(' & '.join([NM[m], f3(d), f3(a), f'{e:.3f}$\\pm${g("tetramer-MD", m, "test_r2_sd"):.3f}', sg(e - d), f'{e / d:.2f}', f3(g('tetramer-MD', m, 'pearson')), f3(g('tetramer-MD', m, 'spearman')), f3(g('tetramer-MD', m, 'avg_runs_r2'))]) + r' \\')
open(R / 'tab_md_verdict.tex', 'w').write('\n'.join(L))
# permuted control
L = []
for m in ['rf', 'dfn', 'cnn1d']:
    e, p1, p2 = g('tetramer-MD', m), g('perm1', m), g('perm2', m); pm_ = (p1 + p2) / 2
    ea, pa = g('tetramer-MD', m, 'avg_runs_r2'), (g('perm1', m, 'avg_runs_r2') + g('perm2', m, 'avg_runs_r2')) / 2
    L.append(' & '.join([NM[m], f'{e:.3f}$\\pm${g("tetramer-MD", m, "test_r2_sd"):.3f}', f'{p1:.3f}$\\pm${g("perm1", m, "test_r2_sd"):.3f}', f'{p2:.3f}$\\pm${g("perm2", m, "test_r2_sd"):.3f}', sg(e - pm_),
                         f3(g('tetramer-MD', m, 'gap')), f3((g('perm1', m, 'gap') + g('perm2', m, 'gap')) / 2), f3(ea), f3(pa), sg(ea - pa), f3(g('one-hot', m))]) + r' \\')
open(R / 'tab_perm.tex', 'w').write('\n'.join(L))
# summary: split tab_bench into two tables (test R2 / averaged-runs R2); order a e b f c g d
po = ['tetramer', 'tetramer-MD', 'hexamer', 'hexamer-MD', 'tetra+hexa', 'tetra+hexa-MD', 'one-hot']
for key, fn in [('test_r2', 'tab_sum_test.tex'), ('avg_runs_r2', 'tab_sum_avg.tex')]:
    L = []
    for m in NM:
        v = [g(fs, m, key) for fs in po]; best = max(v)
        L.append(' & '.join([NM[m]] + [('\\textbf{%s}' % f3(x)) if x == best else f3(x) for x in v]) + r' \\')
    open(R / fn, 'w').write('\n'.join(L))
print(open(R / 'tab_md_verdict.tex').read()); print(open(R / 'tab_perm.tex').read())

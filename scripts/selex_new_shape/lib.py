"""SELEX Gata4 + new SELEX shape tables (data/selex/{tetramer,hexamer}). Labels from 4.txt."""
import json, os, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from dnaffinity.preprocess import revcomp, canonical
OUT = ROOT / 'outputs' / 'selex_new_shape'
IU = np.triu_indices(6)

def _tab(p):
    d = {}
    for l in open(p):
        t = l.split()
        if t[0] in ('SEQUENCE',) or t[0].startswith('SHIFT'): continue
        d[t[0]] = np.array(t[1:], float)
    return d
TAB = {}
for k, n in [(4, 'tetramer'), (6, 'hexamer')]:
    TAB[(k, 'avg')] = _tab(ROOT / f'data/selex/{n}/avg_{n}.dat')
    TAB[(k, 'fce')] = {s: v.reshape(6, 6)[IU] for s, v in _tab(ROOT / f'data/selex/{n}/fce_{n}.dat').items()}  # 21 unique of symmetric 6x6
if os.environ.get('PERM_TAB'):  # control: random lookup table = real MD rows reassigned to random tetramers (same values, same dim)
    _ks = sorted(TAB[(4, 'avg')]); _p = np.random.RandomState(int(os.environ['PERM_TAB'])).permutation(len(_ks))
    for _q in ('avg', 'fce'): _o = TAB[(4, _q)]; TAB[(4, _q)] = {k: _o[_ks[j]] for k, j in zip(_ks, _p)}
OH = {b: np.eye(4)[i] for i, b in enumerate('ACGT')}

def windowed(seqs, ks=(4,), props=('avg', 'fce'), onehot=False, shape=True):
    """(n, 7, d): tetramer window i = seq[i:i+4]; hexamer window aligned on its central tetramer
    (hexamer j covers tetramer windows j..j+2, placed at i=j+1; ends zero-padded). One-hot = bases of the tetramer window."""
    L = len(seqs[0]); W = L - 3; out = []
    for s in seqs:
        rows = []
        for i in range(W):
            r = []
            if shape:
                for k in ks:
                    for p in props:
                        if k == 4: r.append(TAB[(4, p)][s[i:i+4]])
                        else:
                            j = i - 1
                            r.append(TAB[(6, p)][s[j:j+6]] if 0 <= j <= L - 6 else np.zeros(len(TAB[(6, p)]['AAAAAA'])))
            if onehot: r.append(np.concatenate([OH[c] for c in s[i:i+4]]))
            rows.append(np.concatenate(r))
        out.append(rows)
    return np.array(out, dtype=np.float32)

def flat_feats(seqs, ks=(4,), props=('avg', 'fce'), onehot=False, shape=True):
    """flat: tetramer windows 0..6 and hexamer windows 0..4 (no padding)."""
    L = len(seqs[0]); X = []
    for s in seqs:
        r = []
        if shape:
            for k in ks:
                for p in props:
                    r += [TAB[(k, p)][s[i:i+k]] for i in range(L - k + 1)]
        if onehot: r += [OH[c] for c in s]
        X.append(np.concatenate(r))
    return np.array(X, dtype=np.float32)

def load_labels(min_count=10, mode='perstrand'):
    """perstrand: each orientation its own row with its own label (RC row present only if measured).
    merged: one label per canonical k-mer (mean affinity of the measured strands), both orientations emitted (RC augmentation)."""
    d = pd.read_csv(ROOT / 'data/selex/Gata4/4.txt', sep='\t')
    d = d[d.ObservedCount >= min_count][['Kmer', 'Affinity', 'ObservedCount']].rename(columns={'Kmer': 'seq', 'Affinity': 'aff', 'ObservedCount': 'n'})
    d['group'] = d.seq.map(canonical)
    if mode == 'merged':
        m = d.groupby('group').agg(aff=('aff', 'mean'), n=('n', 'sum')).reset_index()
        a = m.assign(seq=m.group); b = m.assign(seq=m.group.map(revcomp))
        d = pd.concat([a, b]).drop_duplicates('seq')
    return d.reset_index(drop=True)

def split(groups, seed=20, test_frac=0.1):
    g = np.unique(groups); rng = np.random.RandomState(seed); test_g = set(rng.choice(g, int(len(g) * test_frac), replace=False))
    return np.array([x in test_g for x in groups])

def metrics(y, p):
    y, p = np.ravel(y), np.ravel(p)
    return dict(r2=float(r2_score(y, p)), pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]), n=int(len(y)))

def tf(a, kind):  return np.log(a) if kind == 'log' else (a ** 0.25 if kind == 'quarter' else a)
def inv(y, kind): return np.exp(y) if kind == 'log' else (np.clip(y, 0, None) ** 4 if kind == 'quarter' else y)

"""Dataset loaders with the preprocessing fixes from outputs/data_audit.md.

Every loader returns a DataFrame with columns ID_REF (canonical sequence), VALUE
(raw-scale label; scaling happens on the training split inside Model) and,
for uPBM, WEIGHT. With ``aggregate=True`` reverse-complement and exact
duplicates are collapsed into one row per canonical sequence (mean label), so
a train/test split on rows can never put the same molecule on both sides.
"""

import numpy as np
import pandas as pd

_COMP = str.maketrans('ACGT', 'TGCA')


def revcomp(seq):
    return seq.translate(_COMP)[::-1]


def canonical(seq):
    """Lexicographically smaller of a sequence and its reverse complement."""
    return min(seq, revcomp(seq))


def aggregate_canonical(df, value='VALUE', weight=None):
    """Collapse rows to one per canonical sequence (mean label, max weight)."""
    df = df.copy()
    df['ID_REF'] = df['ID_REF'].map(canonical)
    agg = {value: 'mean'}
    if weight is not None:
        agg[weight] = 'max'
    out = df.groupby('ID_REF', sort=True).agg(agg).reset_index()
    return out


def load_upbm(path, aggregate=True):
    df = pd.read_csv(path, sep='\t')[['ID_REF', 'VALUE', 'WEIGHT']].dropna()
    if aggregate:
        df = aggregate_canonical(df, weight='WEIGHT')
    return df.reset_index(drop=True)


def load_selex(path, selection='count', min_count=10, seed=20):
    """
    :param selection: 'count' keeps k-mers with ObservedCount >= min_count (fixed);
                      'legacy' keeps the top 10% by background Probability (original
                      scripts/run_selex.py behaviour, shuffled with ``seed``).
    """
    deca = pd.read_csv(path, sep='\t')
    if selection == 'legacy':
        bound = sorted(deca['Probability'])[int(len(deca) * 0.9)]
        df = deca[deca['Probability'] > bound][['Kmer', 'Affinity']]
        df = df.sample(frac=1, random_state=seed)
        df.columns = ['ID_REF', 'VALUE']
        return df.reset_index(drop=True)
    if selection != 'count':
        raise ValueError("selection must be 'count' or 'legacy'")
    df = deca[deca['ObservedCount'] >= min_count][['Kmer', 'Affinity']]
    df.columns = ['ID_REF', 'VALUE']
    return aggregate_canonical(df)


def flank_max_score(seq, freq_matrix, positions):
    tr = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    w = len(freq_matrix)
    return max(sum(freq_matrix[j][tr[seq[i + j]]] for j in range(w)) for i in positions)


def load_gcpbm(data_file, freq_matrix_file, flank_threshold=4.5, aggregate=True,
               trim=(3, 33), core=(12, 18)):
    """
    :param flank_threshold: discard probes whose flanks score above this against the
        frequency matrix (legacy value 2.5 discards ~78% of probes, see audit; 4.5 =
        >=5/6 matches to the E-box consensus)
    :param aggregate: collapse the ~4 measurements (2 replicates x 2 orientations)
        of each canonical 30-mer into their mean intensity
    """
    raw = pd.read_csv(data_file, sep='\t')[['Sequence', 'Intensity']].dropna()
    seqs = raw['Sequence'].str[trim[0]:trim[1]]
    df = pd.DataFrame({'ID_REF': seqs.values, 'VALUE': raw['Intensity'].astype(float).values})
    fm = np.array(pd.read_csv(freq_matrix_file, sep=r'\s+'))
    w, L = len(fm), trim[1] - trim[0]
    positions = list(range(0, core[0] - w + 1)) + list(range(core[1], L - w + 1))
    if flank_threshold is not None:
        scores = df['ID_REF'].map(lambda s: flank_max_score(s, fm, positions))
        df = df[scores <= flank_threshold]
    if aggregate:
        df = aggregate_canonical(df)
    return df.reset_index(drop=True)

"""LRP aggregation helpers (feature groups, per-base maps) shared by lrp_analyze.py and the notebook."""
import numpy as np
from lib import *
PAR = [l.split() for l in open(ROOT / 'data/selex/tetramer/avg_tetramer.dat')][0][1:]  # SHIFT SLIDE RISE TILT ROLL TWIST
FCE = [f'{PAR[i]}-{PAR[j]}' for i, j in zip(*IU)]; NUC = 'ACGT'
def per_base(R, kind, shape, oh=True, seqs=None):
    """-> (n,10,4) relevance per base position & nucleotide channel. Without one-hot, the relevance of each tetramer
    window (sum of its shape channels) is projected equally onto its 4 bases, on the channel of the base present."""
    n = len(R); B = np.zeros((n, 10, 4))
    if not oh:
        W = R.sum(2) if kind == 'cnn' else R[:, :42].reshape(n, 7, 6).sum(2) + R[:, 42:189].reshape(n, 7, 21).sum(2)
        pres = np.array([[NUC.index(ch) for ch in s] for s in seqs])
        for i in range(7):
            for k in range(4): B[np.arange(n), i + k, pres[:, i + k]] += W[:, i] / 4
        return B
    if kind == 'cnn':
        o = 27 if shape else 0
        for i in range(7):
            for k in range(4): B[:, i + k] += R[:, i, o + 4 * k:o + 4 * k + 4]
    else:
        o = 189 if shape else 0; B = R[:, o:o + 40].reshape(n, 10, 4)
    return B
def groups(R, kind, shape, oh=True):
    """dict group -> (n,) summed relevance; and per-window shape relevance (n,7)."""
    n = len(R); g = {}
    if kind == 'cnn':
        if oh: g['one-hot'] = R[:, :, 27 if shape else 0:].sum((1, 2))
        if shape:
            for p in range(6): g['avg ' + PAR[p]] = R[:, :, p].sum(1)
            dg = [6 + j for j, (a, b) in enumerate(zip(*IU)) if a == b]; od = [6 + j for j, (a, b) in enumerate(zip(*IU)) if a != b]
            g['fce diagonal'] = R[:, :, dg].sum((1, 2)); g['fce off-diagonal'] = R[:, :, od].sum((1, 2))
            fce = R[:, :, 6:27].sum(1); win_shape = R[:, :, :27].sum(2)
        else: fce = None; win_shape = None
    else:
        if oh: g['one-hot'] = R[:, 189 if shape else 0:][:, :40].sum(1)
        if shape:
            A = R[:, :42].reshape(n, 7, 6); F = R[:, 42:189].reshape(n, 7, 21)
            for p in range(6): g['avg ' + PAR[p]] = A[:, :, p].sum(1)
            dg = [j for j, (a, b) in enumerate(zip(*IU)) if a == b]; od = [j for j in range(21) if j not in dg]
            g['fce diagonal'] = F[:, :, dg].sum((1, 2)); g['fce off-diagonal'] = F[:, :, od].sum((1, 2))
            fce = F.sum(1); win_shape = A.sum(2) + F.sum(2)
        else: fce = None; win_shape = None
    return g, fce, win_shape
def absgroups(R, kind, shape, oh=True):  # same grouping on |R|
    return groups(np.abs(R), kind, shape, oh)

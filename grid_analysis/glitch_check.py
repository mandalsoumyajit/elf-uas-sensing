"""Are the impulsive events physical or acquisition glitches?

Per channel/cell: fraction of samples at the rails (0 or 4095), rate of single-sample spikes (sample deviates
from both neighbours' mean by > 10 sigma while the neighbours agree), rate of repeated-value runs >= 3,
and, for 7-9.5 kHz events, the fraction explained by such spikes. Also: are glitches simultaneous across
channels (same sample index)?
"""
import os, sys
import numpy as np
import scipy.io as sio

EXP = sys.argv[1]
def load(c, ch):
    return sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)

for c in (1, 7, 13, 19, 25):
    X = {ch: load(c, ch) for ch in 'ABCD'}
    n = min(len(v) for v in X.values()); T = n / 20000
    spikes = {}
    line = f'cell {c:2d}:'
    for ch in 'ABCD':
        x = X[ch][:n]
        s = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2)
        nb = 0.5 * (x[:-2] + x[2:]); dev = x[1:-1] - nb
        sp = np.flatnonzero((np.abs(dev) > 10 * s) & (np.abs(x[:-2] - x[2:]) < 3 * s)) + 1
        spikes[ch] = sp
        rail = np.mean((x == 0) | (x == 4095))
        d = np.diff(x); runs = np.flatnonzero((d[:-1] == 0) & (d[1:] == 0))
        vals = np.unique(x[sp], return_counts=True)
        top = vals[0][np.argsort(vals[1])[::-1][:3]] if len(sp) else []
        line += (f' {ch}: spikes {len(sp)/T:6.2f}/s, rail {100*rail:.3f}%, runs>=3 {len(runs)/T:6.1f}/s, '
                 f'common spike values {list(np.int64(top))} |')
    print(line)
    for a, b in (('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')):
        if len(spikes[a]) and len(spikes[b]):
            same = np.isin(spikes[a], np.r_[spikes[b], spikes[b] + 1, spikes[b] - 1]).mean()
            print(f'   {a}{b}: fraction of {a} spikes within +-1 sample of a {b} spike: {same:.2f}')

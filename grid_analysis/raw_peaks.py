import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, 'raw_psd.npz'))
names, psd, f = list(D['names']), D['psd'], D['f']
def P(c, ch): return psd[names.index(f'c{c:02d}_{ch}.mat')]

# 1) mains lines in a far cell: where are the 50 Hz family peaks?
for c, ch in ((25, 'A'), (1, 'C'), (13, 'B')):
    p = P(c, ch)
    m = (f > 20) & (f < 420)
    pk, _ = find_peaks(10 * np.log10(p[m]), prominence=10)
    top = pk[np.argsort(p[m][pk])[::-1][:10]]
    print(f'cell {c} ch {ch}: strongest peaks 20-420 Hz:', np.round(np.sort(f[m][top]), 2))

# 2) drone lines: peaks strongly enhanced near each antenna vs the same channel at a far cell
pairs = {'A': (1, 25), 'B': (5, 21), 'C': (25, 1), 'D': (21, 5)}
for ch, (near, far) in pairs.items():
    r = 10 * np.log10(P(near, ch) / P(far, ch))
    m = (f > 20) & (f < 9900)
    pk, _ = find_peaks(r[m], prominence=15, distance=10)
    top = pk[np.argsort(r[m][pk])[::-1][:15]]
    print(f'ch {ch}: near cell {near} vs far cell {far}; top excess lines (Hz, dB):',
          [(round(float(f[m][i]), 1), round(float(r[m][i]), 1)) for i in sorted(top, key=lambda i: f[m][i])])

fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
for ax, (ch, (near, far)) in zip(axs, pairs.items()):
    for c, col in ((near, 'C3'), (13, 'C2'), (far, 'k')):
        ax.semilogy(f, P(c, ch), color=col, lw=0.6, label=f'cell {c}')
    ax.set_ylabel(f'ch {ch} PSD (counts²/Hz)'); ax.legend(fontsize=7); ax.grid(alpha=.3)
axs[-1].set_xlabel('Hz'); axs[-1].set_xlim(0, 10000)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'raw_psd_overview.png'), dpi=110)

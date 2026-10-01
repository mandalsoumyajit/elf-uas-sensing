"""Per-cell drone band power on each channel, mapped onto the 5x5 grid.

Band power uses the median PSD across the band (robust to the 50 Hz harmonic comb and the 5.93 kHz spike),
expressed in dB above the same channel's lowest-cell value (proxy for the drone-free floor).
Grid numbering (report Fig. 4.5): cell n -> row (n-1)//5 from top, col (n-1)%5 from left.
Antennas: A near cell 1 (top-left), B near 5 (top-right), C near 25 (bottom-right), D near 21 (bottom-left).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, 'raw_psd.npz'))
names, psd, f = list(D['names']), D['psd'], D['f']
BANDS = {'1.0-1.35 kHz lines': (1000, 1350), '5.2-6.8 kHz humps': (5200, 6800)}
CH = 'ABCD'

def bandpow(c, ch, lo, hi):
    p = psd[names.index(f'c{c:02d}_{ch}.mat')]
    m = (f >= lo) & (f <= hi) & (np.abs(f - 5930) > 30)
    return np.median(p[m])

res = {}
for bname, (lo, hi) in BANDS.items():
    M = np.array([[bandpow(c, ch, lo, hi) for c in range(1, 26)] for ch in CH])   # 4 x 25
    res[bname] = 10 * np.log10(M / M.min(axis=1, keepdims=True))
np.savez(os.path.join(HERE, 'raw_maps.npz'), **{k.split()[0]: v for k, v in res.items()})

fig, axs = plt.subplots(2, 4, figsize=(12, 6.2))
for i, (bname, R) in enumerate(res.items()):
    for j, ch in enumerate(CH):
        ax = axs[i, j]
        im = ax.imshow(R[j].reshape(5, 5), cmap='magma', vmin=0, vmax=max(20, R.max()))
        for c in range(25):
            ax.text(c % 5, c // 5, f'{R[j, c]:.0f}', ha='center', va='center', fontsize=7,
                    color='w' if R[j, c] < 0.6 * R.max() else 'k')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'ch {ch} | {bname}', fontsize=8)
fig.colorbar(im, ax=axs, shrink=0.6, label='dB above channel minimum')
fig.savefig(os.path.join(HERE, 'raw_maps.png'), dpi=120, bbox_inches='tight')
for bname, R in res.items():
    print(bname)
    for j, ch in enumerate(CH):
        print(f'  ch {ch}:')
        print(np.array2string(R[j].reshape(5, 5), precision=1, suppress_small=True, prefix='    '))

"""Background coherence between antennas (~5-7 m apart) from the synchronized raw records.

Uses pairs of channels that are both far from the drone for the chosen cell, and frequency bins away
from the drone bands. Reports magnitude-squared coherence at mains-harmonic bins and in between.
"""
import os
import numpy as np
import scipy.io as sio
from scipy.signal import coherence

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 20000.0
def load(c, ch):
    return sio.loadmat(os.path.join(HERE, 'raw_export', f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)

cases = [(1, 'C', 'D', '~5 m apart; drone at cell 1 is ~4.5-6.4 m away'),
         (1, 'B', 'C', '~5 m apart'),
         (25, 'A', 'B', '~5 m apart; drone at cell 25 far'),
         (25, 'A', 'D', '~5 m apart'),
         (5, 'A', 'C', '~7 m diagonal'),
         (21, 'A', 'C', '~7 m diagonal')]
for c, a, b, note in cases:
    xa, xb = load(c, a), load(c, b)
    n = min(len(xa), len(xb))
    f, g = coherence(xa[:n] - xa.mean(), xb[:n] - xb.mean(), fs=FS, nperseg=2 ** 15, noverlap=2 ** 14)
    df = f[1] - f[0]
    harm = np.zeros_like(f, bool)
    for k in range(1, 100):
        harm |= np.abs(f - 50.05 * k) <= 0.6 * df
    drone = ((f > 950) & (f < 1450)) | ((f > 5100) & (f < 7000))
    sel = lambda lo, hi, h: (f > lo) & (f < hi) & (harm == h) & ~drone
    mains = sel(40, 3000, True); between = sel(200, 4800, False); hi = sel(7000, 9500, False)
    print(f'cell {c:2d} {a}-{b} ({note}): gamma^2 mains harmonics median {np.median(g[mains]):.3f} '
          f'(90th pct {np.percentile(g[mains], 90):.3f}); between harmonics 0.2-4.8 kHz {np.median(g[between]):.3f}; '
          f'7-9.5 kHz {np.median(g[hi]):.3f}; at 50 Hz {g[np.argmin(np.abs(f-50.05))]:.3f}')

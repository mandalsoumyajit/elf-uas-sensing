"""Motor-line frequencies per cell (session) and their wander within each 316 s record."""
import os
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks, stft

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, 'raw_psd.npz'))
names, psd, f = list(D['names']), D['psd'], D['f']
FS = 20000.0
band = (f > 1000) & (f < 1400)

def best_channel(c):
    # channel with largest excess power in the line band relative to its own cross-cell median
    best, score = None, -np.inf
    for ch in 'ABCD':
        p = psd[names.index(f'c{c:02d}_{ch}.mat')][band]
        ref = np.median(np.stack([psd[names.index(f'c{k:02d}_{ch}.mat')][band] for k in range(1, 26)]), axis=0)
        s = np.max(10 * np.log10(p / ref))
        if s > score:
            best, score = ch, s
    return best, score

rows = []
for c in range(1, 26):
    ch, score = best_channel(c)
    p = psd[names.index(f'c{c:02d}_{ch}.mat')]
    ref = np.median(np.stack([psd[names.index(f'c{k:02d}_{ch}.mat')] for k in range(1, 26)]), axis=0)
    ex = 10 * np.log10(p[band] / ref[band])
    pk, pr = find_peaks(ex, prominence=6, distance=int(15 / (f[1] - f[0])))
    top = pk[np.argsort(ex[pk])[::-1][:4]]
    lines = np.sort(f[band][top])
    # wander of the strongest line: STFT with 1 s windows, track peak within +-40 Hz
    x = sio.loadmat(os.path.join(HERE, 'raw_export', f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)
    fz, tz, Z = stft(x - x.mean(), fs=FS, nperseg=int(FS), noverlap=0)
    f0 = f[band][top[0]] if len(top) else np.nan
    w = (fz > f0 - 40) & (fz < f0 + 40)
    track = fz[w][np.argmax(np.abs(Z[w]), axis=0)]
    rows.append((c, ch, score, lines, f0, np.std(track), np.ptp(np.percentile(track, [5, 95]))))
    print(f'cell {c:2d} ch {ch} excess {score:5.1f} dB  lines {np.round(lines, 1)}  strongest {f0:7.1f} Hz  '
          f'1-s track sd {np.std(track):5.2f} Hz, 5-95% span {np.ptp(np.percentile(track, [5, 95])):5.1f} Hz')
np.save(os.path.join(HERE, 'linefreq.npy'), np.array([(r[0], r[4], r[5]) for r in rows]))

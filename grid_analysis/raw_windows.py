"""1-s window band-power features for all cells/channels (robust medians over 1 Hz bins)."""
import os
import numpy as np
import scipy.io as sio
from scipy.signal import stft
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 20000.0
BANDS = [(1000, 1400), (1400, 2000), (2000, 3500), (3500, 5000), (5200, 5850), (6300, 6900), (7000, 9000)]

def feats(args):
    c, ch = args
    x = sio.loadmat(os.path.join(HERE, 'raw_export', f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)
    fz, tz, Z = stft(x - x.mean(), fs=FS, nperseg=int(FS), noverlap=0, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    keep = np.abs(fz - 5930) > 30
    out = np.stack([np.median(P[(fz >= lo) & (fz < hi) & keep], axis=0) for lo, hi in BANDS], axis=1)
    return c, ch, np.log10(out)

if __name__ == '__main__':
    jobs = [(c, ch) for c in range(1, 26) for ch in 'ABCD']
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(feats, jobs))
    F = {}
    for c, ch, v in res:
        F.setdefault(c, {})[ch] = v
    X, y, tix = [], [], []
    for c in range(1, 26):
        n = min(F[c][ch].shape[0] for ch in 'ABCD')
        X.append(np.hstack([F[c][ch][:n] for ch in 'ABCD'])); y += [c] * n; tix += list(range(n))
    X = np.vstack(X)
    np.savez(os.path.join(HERE, 'raw_windows.npz'), X=X, y=np.array(y), t=np.array(tix), bands=np.array(BANDS))
    print(X.shape)

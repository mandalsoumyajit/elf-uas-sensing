"""Frequency-wander statistics of the strongest motor line (for the simulation source model):
std, correlation time (1/e of the autocorrelation of f(t) - mean) and rms change over 0.2 s / 1 s.
Nearest-antenna records of the four corner cells plus four edge cells; 0.1 s Hann STFT, 0.025 s hop,
8x zero-padding, peak in fc +- 80 Hz with parabolic interpolation."""
import os, sys
import numpy as np
from scipy.signal import stft
import scipy.io as sio

EXP = sys.argv[1]; FS = 20000.0
HERE = os.path.dirname(os.path.abspath(__file__))
A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()
NEAR = {1: 'A', 5: 'B', 25: 'C', 21: 'D', 2: 'A', 10: 'B', 24: 'C', 16: 'D'}
for c, ch in NEAR.items():
    x = sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float); x -= x.mean()
    fc = float(A[c]['fc'])
    f, t, Z = stft(x, fs=FS, nperseg=2000, noverlap=1500, nfft=16000)
    sel = np.abs(f - fc) < 80; P = np.abs(Z[sel]) ** 2; fs_ = f[sel]
    i = np.clip(P.argmax(0), 1, sel.sum() - 2); j = np.arange(P.shape[1])
    a, b, cc = np.log(P[i - 1, j] + 1e-30), np.log(P[i, j] + 1e-30), np.log(P[i + 1, j] + 1e-30)
    tr = fs_[i] + 0.5 * (a - cc) / (a - 2 * b + cc) * (fs_[1] - fs_[0])
    d = tr - tr.mean(); ac = np.correlate(d, d, 'full')[len(d) - 1:]; ac /= ac[0]
    tau = np.argmax(ac < np.exp(-1)) * 0.025
    dt = lambda T: np.sqrt(np.mean((tr[int(T / 0.025):] - tr[:-int(T / 0.025)]) ** 2))
    print(f'cell {c:2d} {ch}: fc {fc:.0f} Hz, sd {d.std():5.1f} Hz, tau {tau:5.2f} s, rms change over 0.2 s {dt(0.2):5.1f} Hz, 1 s {dt(1.0):5.1f} Hz')

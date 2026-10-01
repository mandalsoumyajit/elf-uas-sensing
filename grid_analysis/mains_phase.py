"""Inter-channel delay calibration from 50 Hz harmonics.

For a quasi-static field from a common current path, the relative phase between two loops is 0 or pi at
every harmonic. A linear trend of relative phase with frequency is therefore an instrument delay (front-end
group-delay difference or ADC sampling skew). Doubling the phase removes the 0/pi sign ambiguity:
2*arg(S_xy(f_n)) = 2*theta + 4*pi*f_n*dtau. Fit per cell, per pair, using coherent harmonics only.
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import csd, welch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1]
FS = 20000.0
NPER = 2 ** 16
def load(c, ch):
    x = sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float); return x - x.mean()

def fit_delay(f, ph2, w):
    """Weighted circular fit of ph2 = a + 4*pi*f*dtau over a dtau grid (seconds)."""
    grid = np.linspace(-80e-6, 80e-6, 3201)
    best = None
    for dt in grid:
        z = np.sum(w * np.exp(1j * (ph2 - 4 * np.pi * f * dt)))
        if best is None or np.abs(z) > best[0]:
            best = (np.abs(z) / w.sum(), dt, np.angle(z))
    return best

rows = []
for c in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]:
    X = {ch: load(c, ch) for ch in 'ABCD'}
    n = min(len(v) for v in X.values())
    f, pa = welch(X['A'][:n], fs=FS, nperseg=NPER)
    harm = np.array([np.argmin(np.abs(f - 50.05 * k)) for k in range(3, 60)])   # 150 Hz - 3 kHz
    for ch in 'BCD':
        _, pb = welch(X[ch][:n], fs=FS, nperseg=NPER)
        _, s = csd(X['A'][:n], X[ch][:n], fs=FS, nperseg=NPER)
        g2 = np.abs(s) ** 2 / (pa * pb)
        sel = harm[g2[harm] > 0.5]
        if len(sel) < 6:
            continue
        R, dt, a = fit_delay(f[sel], 2 * np.angle(s[sel]), g2[sel])
        # residual phase scatter (on the doubled phase, halved back)
        res = np.angle(np.exp(1j * (2 * np.angle(s[sel]) - a - 4 * np.pi * f[sel] * dt))) / 2
        rows.append((c, ch, len(sel), dt, R, np.degrees(np.std(res))))
        print(f'cell {c:2d} A-{ch}: {len(sel):2d} coherent harmonics, delay(A->{ch}) {dt*1e6:7.2f} us, '
              f'fit R {R:.2f}, residual {np.degrees(np.std(res)):5.1f} deg, phase at 1.3 kHz from delay {360*1300*dt:6.1f} deg', flush=True)
rows = np.array(rows, dtype=object)
np.save(os.path.join(HERE, 'mains_delays.npy'), rows, allow_pickle=True)
print()
for ch in 'BCD':
    m = rows[:, 1] == ch
    d = np.array(rows[m, 3], float) * 1e6
    print(f'A-{ch}: median delay {np.median(d):.2f} us, IQR {np.percentile(d, 25):.2f}..{np.percentile(d, 75):.2f} us over {m.sum()} cells '
          f'-> {360*1.3e3*np.median(d)*1e-6:.1f} deg at 1.3 kHz, {360*315*np.median(d)*1e-6:.1f} deg change across 1070-1385 Hz')

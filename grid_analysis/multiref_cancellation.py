"""Best-case reference cancellation at array scale (5-7 m): predict one antenna's background from the other
three with an optimal frequency-domain Wiener filter. Residual power fraction = 1 - multiple coherence.
Cells where the drone sits next to a different antenna; drone bands (0.95-1.45, 5.1-7 kHz) excluded."""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import csd

EXP = sys.argv[1]; FS = 20000.0; NP = 2 ** 14
def load(c, ch):
    x = sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float); return x - x.mean()
CASES = [(25, 'A'), (21, 'B'), (1, 'C'), (5, 'D'), (13, 'A'), (13, 'C')]   # (cell, predicted channel)
for c, tgt in CASES:
    X = {ch: load(c, ch) for ch in 'ABCD'}; n = min(len(v) for v in X.values())
    refs = [ch for ch in 'ABCD' if ch != tgt]; chs = [tgt] + refs
    S = {}
    for a in chs:
        for b in chs:
            f, S[(a, b)] = csd(X[a][:n], X[b][:n], fs=FS, nperseg=NP)
    Srr = np.stack([np.stack([S[(a, b)] for b in refs], -1) for a in refs], -2)      # F x 3 x 3
    Syr = np.stack([S[(tgt, b)] for b in refs], -1)                                  # F x 3
    sol = np.linalg.solve(Srr + 1e-12 * np.eye(3) * np.abs(Srr).max(), Syr[..., None])[..., 0]
    g2 = np.real(np.einsum('fi,fi->f', Syr.conj(), sol)) / np.real(S[(tgt, tgt)])     # multiple coherence
    harm = np.zeros_like(f, bool)
    for k in range(1, 200):
        harm |= np.abs(f - 50.0 * k) <= 1.5 * (f[1] - f[0])
    drone = ((f > 950) & (f < 1450)) | ((f > 5100) & (f < 7000))
    sel = lambda lo, hi, h: (f > lo) & (f < hi) & (harm == h) & ~drone
    canc = lambda m: -10 * np.log10(1 - np.clip(np.median(g2[m]), 0, 0.9999))
    print(f'cell {c:2d}, predict {tgt} from {"".join(refs)}: multiple coherence between harmonics 0.2-4.5 kHz median {np.median(g2[sel(200,4500,False)]):.3f} '
          f'(cancellation {canc(sel(200,4500,False)):.2f} dB), 7-9.5 kHz {np.median(g2[sel(7000,9500,False)]):.3f}; mains harmonics <1 kHz median '
          f'{np.median(g2[sel(40,1000,True)]):.2f} ({canc(sel(40,1000,True)):.1f} dB), 1-4.5 kHz {np.median(g2[sel(1000,4500,True)]):.2f} '
          f'({canc(sel(1000,4500,True)):.1f} dB); 50 Hz {g2[np.argmin(np.abs(f-50))]:.3f}')

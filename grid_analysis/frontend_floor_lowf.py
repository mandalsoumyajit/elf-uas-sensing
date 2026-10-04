"""Calibrated inter-harmonic floor below 1 kHz (where the Phase-2 high-pass corner is uncertain)."""
import os
import numpy as np
from frontend_calibration import H_B, GRID, dist, HERE

P = np.load(os.path.join(HERE, 'raw_psd.npz')); PN, PP, f = list(P['names']), P['psd'], P['f']
harm = np.zeros_like(f, bool)
for k in range(1, 200):
    harm |= np.abs(f - 50.0 * k) <= 5
IH = ~harm
cen = np.array([80, 125, 175, 225, 275, 325, 400, 500, 700, 900])
for ch in 'ABCD':
    far = [c for c in range(1, 26) if dist(c, ch) >= 4.0]
    bg = np.median(np.stack([PP[PN.index(f'c{c:02d}_{ch}.mat')] for c in far]), 0)
    row = []
    for c in cen:
        a = np.sqrt(np.median(bg[IH & (f > c - 25) & (f < c + 25)]))
        h = np.array([np.abs(H_B(c, *p)) for p in GRID])
        row.append(f'{a / h.max() * 1e15:5.0f}-{a / h.min() * 1e15:5.0f}')
    print(f'{ch}: ' + '  '.join(f'{c}Hz {r}' for c, r in zip(cen, row)))

"""Calibrated inter-harmonic floor spectrum (fT/rtHz) per channel with the documented front end, and a power-law
fit for the range model's measured-indoor background scenario."""
import os
import numpy as np
from frontend_calibration import H_B, GRID, dist, HERE

P = np.load(os.path.join(HERE, 'raw_psd.npz')); PN, PP, f = list(P['names']), P['psd'], P['f']
harm = np.zeros_like(f, bool)
for k in range(1, 200):
    harm |= np.abs(f - 50.0 * k) <= 5
IH = ~harm & (np.abs(f - 5930) > 30) & ~((f > 950) & (f < 1450)) & ~((f > 5100) & (f < 7000))
cen = np.arange(375, 9800, 250)
Hm = np.array([np.median([np.abs(H_B(c, *p)) for p in GRID]) for c in cen])
out = {}
for ch in 'ABCD':
    far = [c for c in range(1, 26) if dist(c, ch) >= 4.0]
    bg = np.median(np.stack([PP[PN.index(f'c{c:02d}_{ch}.mat')] for c in far]), 0)
    fl = np.array([np.sqrt(np.median(bg[IH & (f > c - 125) & (f < c + 125)])) for c in cen])
    out[ch] = fl / Hm * 1e15
print('freq (Hz): ' + ' '.join(f'{c:6.0f}' for c in cen[::4]))
for ch in 'ABCD':
    print(f'  {ch} fT  : ' + ' '.join(f'{v:6.0f}' for v in out[ch][::4]))
geo = np.exp(np.mean(np.log(np.stack(list(out.values()))), 0))
print('  geo-mean: ' + ' '.join(f'{v:6.0f}' for v in geo[::4]))
for lo, hi in ((375, 3000), (375, 9800)):
    m = (cen >= lo) & (cen <= hi)
    p = np.polyfit(np.log(cen[m] / 1e3), np.log(geo[m]), 1)
    print(f'power-law fit {lo}-{hi} Hz (geo-mean of channels): {np.exp(p[1]):.0f} fT/rtHz at 1 kHz, exponent {p[0]:.2f}')
for fx in (600, 1100, 1300, 3000, 8000):
    print(f'  at {fx} Hz: geo-mean {np.interp(fx, cen, geo):.0f} fT, channel range {min(np.interp(fx, cen, v) for v in out.values()):.0f}-{max(np.interp(fx, cen, v) for v in out.values()):.0f} fT')

"""Absolute check of the 5-inch source model against the grid, line by line (2026-10-04).

For the four corner cells (drone 0.85 m horizontally from the nearest loop), the excess power of the nearest
antenna over its far-cell median is integrated in a window around each model line (all four motors, wander
included) and converted to field with the documented front end (frontend_calibration.py, median calibration).
The model prediction sums four motors in power: B_rms = sqrt(4 / 2) * g * mu0 m / (4 pi r^3), g = 1 (equatorial)
to 2 (axial), for r = 0.85 m (loop centre level with the drone) and r = 1.27 m (loop centre ~0.95 m below it).
Moments: drone_source_results.json, 5-inch class (nominal; the model spans about x5 either way).
"""
import json, os
import numpy as np
from frontend_calibration import H_B, GRID, dist, HERE

P = np.load(os.path.join(HERE, 'raw_psd.npz')); PN, PP, f = list(P['names']), P['psd'], P['f']; df = f[1] - f[0]
A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()
SRC = json.load(open(os.path.join(HERE, '..', 'source_model', 'drone_source_results.json')))['5-inch FPV (6S)']['lines']
mom = {}
for L in SRC:
    key = {'rotor residual': 'f_m', 'phase leads': None, 'DC bus': '6f_e'}.get(L['source'])
    if L['source'] == 'phase leads':
        key = {1: 'f_e', 5: '5f_e', 7: '7f_e'}.get(int(round(L['f'] / 1074.0)))
    if key and key not in mom:
        mom[key] = L['m'][1]
LINES = {'f_m': (1 / 7, 0.06), 'f_e': (1, 0.08), '5f_e': (5, 0.12), '6f_e': (6, 0.06), '7f_e': (7, 0.10)}   # (multiple of f_e, half-window / f)
print('model nominal moments (A m^2):', {k: f'{v:.2e}' for k, v in mom.items()})
rows = {k: [] for k in LINES}
for c, ch in ((1, 'A'), (5, 'B'), (21, 'D'), (25, 'C')):
    p = PP[PN.index(f'c{c:02d}_{ch}.mat')]
    far = [k for k in range(1, 26) if dist(k, ch) >= 4.0]
    ref = np.median(np.stack([PP[PN.index(f'c{k:02d}_{ch}.mat')] for k in far]), 0)
    fe = float(A[c]['fc'])
    for name, (mult, hw) in LINES.items():
        fc = mult * fe; w = (f > fc * (1 - hw)) & (f < fc * (1 + hw))
        ex = max((p[w] - ref[w]).sum() * df, 0.0)
        h = np.median([np.abs(H_B(fc, *q)) for q in GRID])
        rows[name].append(np.sqrt(ex) / h)                         # rms field of the excess
for name in LINES:
    meas = np.array(rows[name]); m = mom[name]
    pred = lambda r: np.sqrt(2) * 1e-7 * m / r ** 3             # sqrt(4/2) * mu0 m / (4 pi r^3), equatorial
    print(f'{name:5s}: measured rms {", ".join(f"{v*1e12:6.1f}" for v in meas)} pT | model (g=1..2) '
          f'r=0.85 m {pred(0.85)*1e12:6.0f}-{2*pred(0.85)*1e12:6.0f} pT, r=1.27 m {pred(1.27)*1e12:5.0f}-{2*pred(1.27)*1e12:5.0f} pT | '
          f'median meas/pred(0.85, g=1) {np.median(meas)/pred(0.85):.3f}')

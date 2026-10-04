"""Locate the cell whose tracked fundamental is ~1161.5 Hz and quantify the f_e/7 rotor line on every antenna,
plus the calibrated 5th/1st and 7th/1st field ratios on the nearest antenna of the corner cells."""
import os
import numpy as np
from frontend_calibration import H_B, GRID, dist, HERE

P = np.load(os.path.join(HERE, 'raw_psd.npz')); PN, PP, f = list(P['names']), P['psd'], P['f']; df = f[1] - f[0]
A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()
Hm = lambda x: np.median([np.abs(H_B(x, *q)) for q in GRID])
Hs = lambda x: np.array([np.abs(H_B(x, *q)) for q in GRID])

def excess(c, ch, fc, hw):
    p = PP[PN.index(f'c{c:02d}_{ch}.mat')]
    far = [k for k in range(1, 26) if dist(k, ch) >= 4.0]
    ref = np.median(np.stack([PP[PN.index(f'c{k:02d}_{ch}.mat')] for k in far]), 0)
    w = (f > fc - hw) & (f < fc + hw)
    return max((p[w] - ref[w]).sum() * df, 0.0), 10 * np.log10(p[w].sum() / ref[w].sum())

c0 = min(A, key=lambda c: abs(float(A[c]['fc']) - 1161.5)); fe = float(A[c0]['fc'])
print(f'cell with f_e closest to 1161.5 Hz: {c0} (f_e {fe:.1f}); rotor line f_e/7 = {fe/7:.1f} Hz')
for ch in 'ABCD':
    ex, rel = excess(c0, ch, fe / 7, 4.0)
    print(f'  ant {ch} r {dist(c0, ch):4.2f} m: {rel:+5.1f} dB over far cells, rms field {np.sqrt(ex)/Hm(fe/7)*1e12:6.1f} pT')
print('\ncalibrated harmonic field ratios (dB) on the nearest antenna, window +-8%/12%/10% of the harmonic:')
for c, ch in ((1, 'A'), (5, 'B'), (21, 'D'), (25, 'C')):
    fe = float(A[c]['fc']); e1, _ = excess(c, ch, fe, 0.08 * fe); e5, _ = excess(c, ch, 5 * fe, 0.6 * fe); e7, _ = excess(c, ch, 7 * fe, 0.7 * fe)
    r5 = 10 * np.log10(e5 / e1) - 20 * np.log10(Hs(5 * fe) / Hs(fe)); r7 = 10 * np.log10(e7 / e1) - 20 * np.log10(Hs(7 * fe) / Hs(fe))
    old5 = 10 * np.log10(e5 / e1) - 20 * np.log10(5); old7 = 10 * np.log10(e7 / e1) - 20 * np.log10(7)
    print(f'  cell {c:2d} ant {ch}: 5th/1st {r5.min():+5.1f}..{r5.max():+5.1f} (flat-gain {old5:+5.1f}); 7th/1st {r7.min():+5.1f}..{r7.max():+5.1f} (flat-gain {old7:+5.1f})')

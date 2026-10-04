"""Is there a rotor-residual line at f_m = f_e / 7 (7 pole pairs) near the drone?  Compare the spectrum of the
antenna nearest the drone with the same antenna's median far-cell spectrum, at f_e/7 (+- wander), and convert the
excess to field with the documented front end. Upper limits use the excess power in a +-10 Hz window."""
import os
import numpy as np
from frontend_calibration import H_B, GRID, dist, HERE

P = np.load(os.path.join(HERE, 'raw_psd.npz')); PN, PP, f = list(P['names']), P['psd'], P['f']
A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()
df = f[1] - f[0]
print(f'PSD resolution {df:.3f} Hz')
for c, ch in ((1, 'A'), (5, 'B'), (21, 'D'), (25, 'C'), (2, 'A'), (10, 'B'), (16, 'D'), (24, 'C')):
    p = PP[PN.index(f'c{c:02d}_{ch}.mat')]
    far = [k for k in range(1, 26) if dist(k, ch) >= 4.0]
    ref = np.median(np.stack([PP[PN.index(f'c{k:02d}_{ch}.mat')] for k in far]), 0)
    fe = float(A[c]['fc']); fm = fe / 7
    win = (f > fm - 10) & (f < fm + 10)
    ex = (p[win] - ref[win]).sum() * df                                   # excess power (counts^2) in the window
    rel = 10 * np.log10(p[win].sum() / ref[win].sum())
    h = np.median([np.abs(H_B(fm, *q)) for q in GRID])
    amp = np.sqrt(2 * max(ex, 0)) / h                                       # peak field of an equivalent sinusoid
    r = dist(c, ch); Bp = 1e-7 * 3.6e-3 / r ** 3
    # the fundamental for comparison
    w2 = (f > fe - 30) & (f < fe + 30)
    ex2 = (p[w2] - ref[w2]).sum() * df; h2 = np.median([np.abs(H_B(fe, *q)) for q in GRID])
    print(f'cell {c:2d} ant {ch} r {r:4.2f} m: f_e {fe:6.1f}, f_m {fm:5.1f} Hz | window power vs far cells {rel:+5.1f} dB, '
          f'excess-equivalent line {amp*1e12:7.1f} pT (nominal rotor residual {Bp*1e12:6.0f}-{2*Bp*1e12:6.0f} pT) | '
          f'fundamental excess-equivalent {np.sqrt(2*max(ex2,0))/h2*1e12:6.1f} pT')

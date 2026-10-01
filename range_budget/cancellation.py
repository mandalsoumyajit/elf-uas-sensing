"""Range gain from reference-channel background cancellation.

A reference channel that sees the background but (almost) no drone field is used to predict and subtract
the background at the primary channel (optimal Wiener / adaptive cancellation). With magnitude-squared
coherence g2 between the background at the two channels, the residual ambient is n_amb*sqrt(1-g2);
the reference's own instrument noise adds to the primary's (x sqrt(2) for equal channels).
"""
import numpy as np
from range_budget import max_range, wire_loop_noise, M_NOMINAL

f = 665.0
n_inst, _ = wire_loop_noise(f, swg_diam_mm=0.2134)
print(f'instrument floor (primary + reference): {np.sqrt(2)*n_inst*1e15:.0f} fT/rtHz')
print(f"{'ambient':>8} {'coherence g2':>13} {'cancel dB':>9} {'resid fT':>9} {'0.2 s':>6} {'60 s incoh':>10} {'60 s coh':>9}")
for na in (0.3e-12, 3e-12):
    for g2 in (0.0, 0.9, 0.99, 0.999, 0.9999):
        n_res = np.sqrt(na ** 2 * (1 - g2) + 2 * n_inst ** 2) if g2 > 0 else np.hypot(na, n_inst)
        r = [max_range(M_NOMINAL, n_res, 0.2, 0.2), max_range(M_NOMINAL, n_res, 60, 0.2), max_range(M_NOMINAL, n_res, 60, 60)]
        cdb = 20 * np.log10(np.hypot(na, n_inst) / n_res)
        print(f'{na*1e12:6.1f}pT {g2:13.4f} {cdb:9.1f} {n_res*1e15:9.0f} {r[0]:6.1f} {r[1]:10.1f} {r[2]:9.1f}')

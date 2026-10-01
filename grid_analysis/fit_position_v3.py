"""V3: V2 complex physics fit + frequency-dependent channel response, self-calibrated on training cells.

g_i(f) = g_i * (f/F0)^kappa_i * exp(j*2*pi*(f-F0)*tau_i),  tau_A = kappa_A = 0 (reference channel).
Each cell uses its own tracked line frequency (1070-1385 Hz), so tau/kappa are identifiable across cells.
Leave-one-cell-out exactly as in fit_position.py.
"""
import os
import numpy as np
from scipy.optimize import minimize
from fit_position import (A, cells, XY, CORNER, GRID, loop_tensors, model_v, cost_complex, weights, sets)

HERE = os.path.dirname(os.path.abspath(__file__))
F0 = 1250.0
FC = {c: float(A[c]['fc']) for c in cells}

def unpack3(p):
    th = p[0:4]; lg = np.r_[0, p[4:7]]; pg = np.r_[0, p[7:10]]; beta = p[10] + 1j * p[11]; d = p[12]; dh = p[13]
    tau = np.r_[0, p[14:17]] * 1e-4; kap = np.r_[0, p[17:20]]          # tau in units of 100 us
    return th, np.exp(lg) * np.exp(1j * pg), beta, d, dh, tau, kap

def gf(g, tau, kap, f):
    return g * (f / F0) ** kap * np.exp(2j * np.pi * (f - F0) * tau)

def train_cost3(p, tr):
    th, g, beta, d, dh, tau, kap = unpack3(p)
    if not (0 <= d <= 1.0 and -1.5 <= dh <= 1.5) or np.any(np.abs(tau) > 1e-3) or np.any(np.abs(kap) > 3):
        return 1e9
    pos = np.array([XY[c] for c in tr])
    u = loop_tensors(pos, th, d, dh)
    tot = 0.0
    for j, c in enumerate(tr):
        v = model_v(u[j:j + 1], gf(g, tau, kap, FC[c]), beta)            # 1 x K x 4
        tot += cost_complex(A[c]['rec'][None], weights(A[c]['rec'], A[c]['nv_rec'])[None], v, True).min()
    return tot

def fold3(c):
    rng = np.random.default_rng(200 + c)
    tr = [k for k in cells if k != c]; truth = XY[c]
    # warm start from the saved V2 calibration of this fold, plus random restarts
    prev = np.load(os.path.join(HERE, 'fit_position_results.npy'), allow_pickle=True).item()['params'][('V2', c)]
    starts = [np.r_[prev, np.zeros(6)]]
    for s in range(4):
        starts.append(np.r_[prev + rng.normal(0, 0.2, len(prev)), rng.normal(0, 1.0, 3), rng.normal(0, 0.3, 3)])
    best = None
    for p0 in starts:
        r = minimize(train_cost3, p0, args=(tr,), method='Powell', options={'maxiter': 30000, 'xtol': 1e-3, 'ftol': 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    th, g, beta, d, dh, tau, kap = unpack3(best.x)
    v = model_v(loop_tensors(GRID, th, d, dh), gf(g, tau, kap, FC[c]), beta)
    out = {}
    for key, a_all, nv in sets(c):
        w = np.array([weights(a, nv) for a in a_all]); loc = []
        for s in range(0, len(a_all), 4):
            cst = cost_complex(a_all[s:s + 4, None, :], w[s:s + 4, None, :], v[None], True)
            loc.append(GRID[np.argmin(cst.min(2), axis=1)])
        out[key] = list(np.linalg.norm(np.concatenate(loc) - truth, axis=1))
    return c, out, best.x, best.fun

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    res = {'rec': [], 'w10': [], 'w1': []}; prm = {}
    with ProcessPoolExecutor(max_workers=20) as ex:
        for c, out, p, fun in ex.map(fold3, cells):
            for k in out:
                res[k] += [(c, e) for e in out[k]]
            prm[c] = p
            th, g, beta, d, dh, tau, kap = unpack3(p)
            print(f'cell {c:2d}: record error {out["rec"][0]:.2f} m | tau B,C,D = {np.round(tau[1:]*1e6,1)} us, '
                  f'kappa = {np.round(kap[1:],2)}, train cost {fun:.1f}', flush=True)
    inner = {7, 8, 9, 12, 13, 14, 17, 18, 19}
    print('\nV3 summary (leave-one-cell-out):')
    for key in ('rec', 'w10', 'w1'):
        arr = np.array(res[key]); e = arr[:, 1]; ci = np.isin(arr[:, 0], list(inner))
        print(f'  V3 {key:4s}: median {np.median(e):.2f} m, mean {e.mean():.2f} m, 90th {np.percentile(e, 90):.2f} m, '
              f'<0.71 m {np.mean(e < 0.71):.2f} | interior 3x3 median {np.median(e[ci]):.2f} m')
    T = np.array([unpack3(prm[c])[5][1:] for c in cells]) * 1e6
    K = np.array([unpack3(prm[c])[6][1:] for c in cells])
    print('  fold-to-fold tau (us) median', np.round(np.median(T, 0), 1), 'IQR', np.round(np.percentile(T, 75, 0) - np.percentile(T, 25, 0), 1))
    print('  fold-to-fold kappa median', np.round(np.median(K, 0), 2), 'IQR', np.round(np.percentile(K, 75, 0) - np.percentile(K, 25, 0), 2))
    np.save(os.path.join(HERE, 'fit_position_v3_results.npy'), {'results': res, 'params': prm}, allow_pickle=True)

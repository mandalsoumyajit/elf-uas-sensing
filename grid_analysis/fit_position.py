"""Step B: leave-one-cell-out position fits from complex coherent amplitudes.

V1  magnitude, empirical power law: |a_i| = G_i * S * r_i^-k (S per cell/window, solved analytically).
V2m magnitude, physics: rotating horizontal moment + axial term, finite 1 m vertical loops (flux-averaged
    point-dipole field), unknown loop azimuths, channel gains, antenna offset and loop height.
V2  complex, same physics plus channel phases: fits log-magnitude AND relative phase.
Shared parameters are calibrated on the 24 training cells (their true positions, per-cell yaw and spin free);
the held-out cell is located by grid search over (x, y, yaw, spin). Evaluated on the record, 10 s and 1 s windows.
"""
import os
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()
cells = sorted(A)
XY = {c: np.array([(c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5]) for c in cells}
CORNER = np.array([[0, 5], [5, 5], [5, 0], [0, 0]], float)   # A B C D
OUTDIR = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) / np.sqrt(2)
EPS = 0.10                                                   # model-error floor (~10% / 0.1 rad)

# ---------- finite loop quadrature ----------
nr, na = 4, 12
rq = 0.5 * np.sqrt((np.arange(nr) + 0.5) / nr)               # equal-area radii, R = 0.5 m
aq = 2 * np.pi * np.arange(na) / na
RQ, AQ = np.meshgrid(rq, aq); RQ, AQ = RQ.ravel(), AQ.ravel()

def loop_tensors(pos, th, d, dh):
    """u[p, i, :] = flux-averaged n_i^T D(point - r0) for drone positions pos (P x 2) -> P x 4 x 3."""
    P = len(pos); u = np.zeros((P, 4, 3))
    for i in range(4):
        c0 = np.r_[CORNER[i] + d * OUTDIR[i], dh]
        n = np.array([np.cos(th[i]), np.sin(th[i]), 0.0])
        e1 = np.array([-np.sin(th[i]), np.cos(th[i]), 0.0]); e2 = np.array([0, 0, 1.0])
        pts = c0 + RQ[:, None] * (np.cos(AQ)[:, None] * e1 + np.sin(AQ)[:, None] * e2)   # Q x 3
        dv = pts[None, :, :] - np.c_[pos, np.zeros(P)][:, None, :]                          # P x Q x 3
        r = np.linalg.norm(dv, axis=2, keepdims=True); dh_ = dv / r
        ndot = dh_ @ n                                                                       # P x Q
        u[:, i, :] = ((3 * ndot[..., None] * dh_ - n[None, None, :]) / r ** 3).mean(1)       # n^T D
    return u

PSI = np.linspace(0, 2 * np.pi, 36, endpoint=False)
def moments(beta):
    """M[k, :] complex moment phasors over (yaw, spin) candidates -> 72 x 3."""
    out = []
    for s in (1, -1):
        for p in PSI:
            out.append(np.array([np.cos(p), np.sin(p), 0]) + 1j * s * np.array([-np.sin(p), np.cos(p), 0]) + beta * np.array([0, 0, 1]))
    return np.array(out)

def weights(a, nv):
    return 1.0 / (nv / (2 * np.abs(a) ** 2 + 1e-30) + EPS ** 2)

def cost_complex(a, w, v, use_phase):
    """a: (..., 4) measured, v: (..., K, 4) model; returns (..., K) cost after solving per-case complex scale."""
    la = np.log(np.abs(a))[..., None, :]; lv = np.log(np.abs(v) + 1e-30)
    W = w[..., None, :]
    em = la - lv; em = em - (W * em).sum(-1, keepdims=True) / W.sum(-1, keepdims=True)
    c = (W * em ** 2).sum(-1)
    if use_phase:
        dp = np.angle(a[..., None, :] * np.conj(v))
        m = np.angle((W * np.exp(1j * dp)).sum(-1, keepdims=True))
        ep = np.angle(np.exp(1j * (dp - m)))
        c = c + (W * ep ** 2).sum(-1)
    return c

def model_v(u, g, beta):
    M = moments(beta)                          # K x 3
    return np.einsum('pic,kc->pki', u, M) * g[None, None, :]    # P x K x 4

def unpack(p):
    th = p[0:4]; lg = np.r_[0, p[4:7]]; pg = np.r_[0, p[7:10]]; beta = p[10] + 1j * p[11]; d = p[12]; dh = p[13]
    return th, np.exp(lg) * np.exp(1j * pg), beta, d, dh

def train_cost(p, tr, use_phase):
    th, g, beta, d, dh = unpack(p)
    if not (0 <= d <= 1.0 and -1.5 <= dh <= 1.5):
        return 1e9
    pos = np.array([XY[c] for c in tr]); a = np.array([A[c]['rec'] for c in tr])
    w = np.array([weights(A[c]['rec'], A[c]['nv_rec']) for c in tr])
    v = model_v(loop_tensors(pos, th, d, dh), g, beta)
    return cost_complex(a, w, v, use_phase).min(-1).sum()

GX, GY = np.meshgrid(np.linspace(0.05, 4.95, 50), np.linspace(0.05, 4.95, 50))
GRID = np.c_[GX.ravel(), GY.ravel()]

def locate(a, w, ug, g, beta, use_phase):
    v = model_v(ug, g, beta)                                         # G x K x 4
    cst = cost_complex(a[None, :], w[None, :], v, use_phase)        # G x K
    return GRID[np.argmin(cst.min(1))]

# V1 empirical power law on magnitudes
def v1_fit(tr):
    def cost(p):
        lg = np.r_[0, p[0:3]]; k, d, dh = p[3], p[4], p[5]
        tot = 0
        for c in tr:
            r = np.linalg.norm(np.c_[XY[c] - (CORNER + d * OUTDIR), np.full(4, dh)], axis=1)
            w = weights(A[c]['rec'], A[c]['nv_rec'])
            e = np.log(np.abs(A[c]['rec'])) - (lg - k * np.log(r)); e -= (w * e).sum() / w.sum()
            tot += (w * e ** 2).sum()
        return tot
    return minimize(cost, np.r_[0, 0, 0, 2.0, 0.2, 0.3], method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-4, 'fatol': 1e-6}).x
def v1_locate(a, w, p):
    lg = np.r_[0, p[0:3]]; k, d, dh = p[3], p[4], p[5]
    r = np.linalg.norm(np.concatenate([GRID[:, None, :] - (CORNER + d * OUTDIR)[None], np.full((len(GRID), 4, 1), dh)], 2), axis=2)
    e = np.log(np.abs(a))[None] - (lg[None] - k * np.log(r)); e -= (w * e).sum(1, keepdims=True) / w.sum()
    return GRID[np.argmin((w * e ** 2).sum(1))]

def fit_shared(tr, use_phase, rng):
    best = None
    for s in range(6):
        th0 = np.arctan2(*(np.array([2.5, 2.5]) - CORNER).T[::-1]) + (0 if s < 3 else np.pi / 2) + rng.normal(0, 0.3, 4)
        p0 = np.r_[th0, rng.normal(0, 0.5, 3), rng.uniform(-np.pi, np.pi, 3), rng.normal(0, 0.5, 2), 0.2, rng.uniform(-0.5, 0.5)]
        r = minimize(train_cost, p0, args=(tr, use_phase), method='Powell', options={'maxiter': 20000, 'xtol': 1e-3, 'ftol': 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun

def sets(c):
    return (('rec', A[c]['rec'][None], A[c]['nv_rec']), ('w10', A[c]['w10'], A[c]['nv10']), ('w1', A[c]['w1'], A[c]['nv1']))

def fold(c):
    rng = np.random.default_rng(100 + c)
    tr = [k for k in cells if k != c]; truth = XY[c]
    out = {k: {'rec': [], 'w10': [], 'w1': []} for k in ('V1', 'V2m', 'V2')}; prm = {}
    p1 = v1_fit(tr)
    for key, a_all, nv in sets(c):
        for a in a_all:
            out['V1'][key].append(np.linalg.norm(v1_locate(a, weights(a, nv), p1) - truth))
    for name, use_phase in (('V2m', False), ('V2', True)):
        p, f = fit_shared(tr, use_phase, rng); prm[name] = p
        th, g, beta, d, dh = unpack(p)
        v = model_v(loop_tensors(GRID, th, d, dh), g, beta)            # G x K x 4, computed once
        for key, a_all, nv in sets(c):
            w = np.array([weights(a, nv) for a in a_all])
            loc = []
            for s in range(0, len(a_all), 4):
                cst = cost_complex(a_all[s:s + 4, None, :], w[s:s + 4, None, :], v[None], use_phase)   # n x G x K
                loc.append(GRID[np.argmin(cst.min(2), axis=1)])
            loc = np.concatenate(loc)
            out[name][key] = list(np.linalg.norm(loc - truth, axis=1))
    return c, out, prm

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    results = {k: {'rec': [], 'w10': [], 'w1': []} for k in ('V1', 'V2m', 'V2')}; params = {}
    with ProcessPoolExecutor(max_workers=20) as ex:
        for c, out, prm in ex.map(fold, cells):
            for name in out:
                for key in out[name]:
                    results[name][key] += [(c, e) for e in out[name][key]]
            params.update({(n, c): p for n, p in prm.items()})
            th, g, beta, d, dh = unpack(prm['V2'])
            print(f'cell {c:2d}: record errors V1 {out["V1"]["rec"][0]:.2f} m, V2m {out["V2m"]["rec"][0]:.2f} m, '
                  f'V2 {out["V2"]["rec"][0]:.2f} m | V2 d={d:.2f} dh={dh:.2f} |beta|={abs(beta):.2f}', flush=True)
    print('\nSummary (leave-one-cell-out):')
    inner = {7, 8, 9, 12, 13, 14, 17, 18, 19}
    for name in results:
        for key in ('rec', 'w10', 'w1'):
            arr = np.array(results[name][key]); e = arr[:, 1]; ci = np.isin(arr[:, 0], list(inner))
            print(f'  {name:4s} {key:4s}: median {np.median(e):.2f} m, mean {e.mean():.2f} m, 90th {np.percentile(e, 90):.2f} m, '
                  f'<0.71 m {np.mean(e < 0.71):.2f} | interior 3x3 median {np.median(e[ci]):.2f} m')
    np.save(os.path.join(HERE, 'fit_position_results.npy'), {'results': results, 'params': params}, allow_pickle=True)


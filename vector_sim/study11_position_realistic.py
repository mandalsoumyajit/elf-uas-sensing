"""Study 11 (v6): array position estimation with the measurement-informed source and noise models.

Replaces the white-noise study 4/9 sweep.  Geometry follows the measured grid: four nodes 0.1 m outside the
corners of a 5 m square, 1.0 m high; drone over [0.5, 4.5]^2 m (8 x 8 cells, 4 poses each), height
U[0.5, 2.0] m, roll/pitch U[+-30 deg], yaw U[+-180 deg].  Nodes are triaxial, or single loops whose
normal points horizontally at the square centre (as in the experiment).
Per window the four motor phasors are demodulated with their tracked phases (motors failing the measured
tracking threshold are dropped; a window with none is a failure).  Two estimators:
  known  : the paper's estimator - attitude, rotor geometry and moment ratio known; one unknown complex
           gain per motor (variable projection)
  free   : attitude-free - each motor's complex moment vector is unknown (3 complex), all motors share the
           drone-centre position (rotor offsets ignored -> 0.11 m model error); variable projection
CRLB for the attitude-free model (triaxial and scalar) at the true pose.
Output: simulation_results/v6_realistic/position.json
"""
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
import realistic_model as RM
from forward_model import R_body_to_world, dipole_tensor
from simulation_utils import save_json, provenance

OUT = Path('simulation_results/v6_realistic'); OUT.mkdir(parents=True, exist_ok=True)
NODES = np.array([[-0.1, -0.1, 1.0], [5.1, -0.1, 1.0], [5.1, 5.1, 1.0], [-0.1, 5.1, 1.0]])
CTR = np.array([2.5, 2.5, 1.0])
NRM = (CTR - NODES) * np.array([1, 1, 0]); NRM /= np.linalg.norm(NRM, axis=1, keepdims=True)
LEVELS = {'quiet outdoor (0.02 pT)': 0.02e-12, 'after cancellation (0.3 pT)': 0.3e-12,
          'indoor / semi-urban (1 pT)': 1e-12, 'noisy indoor (10 pT)': 10e-12}
TS = (0.2, 1.0)
LO, HI = np.array([0.0, 0.0, 0.25]), np.array([5.0, 5.0, 2.5])
SRC = RM.Src()

def body_phasor(src):
    return np.column_stack([np.full(4, src.m_rot), -1j * src.spin * src.m_rot, -1j * np.full(4, src.m_axial)])

def G_at(r):
    return dipole_tensor(NODES - r)                                # node, 3, 3

def proj(Y, A):
    """residual of the best complex fit Y ~ A c (columns of A), Y: (m,), A: (m, p)."""
    c, *_ = np.linalg.lstsq(A, Y, rcond=None); return Y - A @ c

def res_free(r, Ys, scalar):
    G = G_at(r)
    A = np.einsum('na,nab->nb', NRM, G) if scalar else G.reshape(-1, 3)
    out = np.concatenate([proj(Y, A) for Y in Ys])
    return np.r_[out.real, out.imag]

def res_known(r, Ys, ks, att, scalar):
    R = R_body_to_world(*att); rot = SRC.rotor_body() @ R.T + r
    m = body_phasor(SRC) @ R.T
    out = []
    for Y, k in zip(Ys, ks):
        g = np.einsum('nab,b->na', dipole_tensor(NODES - rot[k]), m[k])
        a = np.einsum('na,na->n', NRM, g) if scalar else g.ravel()
        out.append(proj(Y, a[:, None]))
    out = np.concatenate(out); return np.r_[out.real, out.imag]

def fit(fun, args):
    Ys = args[0]; s = np.sqrt(sum(np.sum(np.abs(y) ** 2) for y in Ys))
    args = ([y / s for y in Ys],) + tuple(args[1:])                    # scale-free (variable projection)
    starts =[LO + (HI - LO) * np.array([x, y, z]) for x in (0.2, 0.5, 0.8) for y in (0.2, 0.5, 0.8) for z in (0.25, 0.75)]
    starts = sorted(starts, key=lambda s: np.sum(fun(s, *args) ** 2))[:3]
    best = min((least_squares(fun, s, args=args, bounds=(LO, HI), max_nfev=300, xtol=1e-10, ftol=1e-12, gtol=1e-12) for s in starts), key=lambda r: r.cost)
    return best.x

def crlb_free(r, Mk, sig2, scalar):
    """Mk: list of true complex moment 3-vectors (world, demod convention) of the tracked motors."""
    def model(th):
        G = G_at(th[:3]); out = []
        for i in range(len(Mk)):
            c = th[3 + 6 * i: 6 + 6 * i] + 1j * th[6 + 6 * i: 9 + 6 * i]
            b = G @ c
            out.append(np.einsum('na,na->n', NRM, b) if scalar else b.ravel())
        return np.concatenate(out)
    th0 = np.r_[r, np.concatenate([np.r_[m.real, m.imag] for m in Mk])]
    J = np.empty((model(th0).size, th0.size), complex)
    for i in range(th0.size):
        h = 1e-5 if i < 3 else 1e-5 * max(1e-9, np.max(np.abs(th0[3:])))
        e = np.zeros(th0.size); e[i] = h; J[:, i] = (model(th0 + e) - model(th0 - e)) / (2 * h)
    sc = np.linalg.norm(J, axis=0); sc[sc == 0] = 1
    I = 2 / sig2 * np.real((J / sc).conj().T @ (J / sc))
    if np.linalg.cond(I) > 1e13:
        return np.inf
    C = np.linalg.inv(I) / np.outer(sc, sc); return float(np.sqrt(np.trace(C[:3, :3])))

def one(args):
    seed, T, poses = args
    rng = np.random.default_rng(seed); out = []
    for pos, att in poses:
        B, psi, f = RM.simulate(pos, att, NODES, T, rng, SRC)
        fm = f.mean(1); Y0 = RM.demod(B, psi)                       # k, node, 3
        R = R_body_to_world(*att); Mtrue = (body_phasor(SRC) @ R.T) * np.exp(1j * 0)   # phase convention irrelevant for CRLB
        row = dict(pos=pos, att=att)
        for name, lv in LEVELS.items():
            x = B + RM.noise(B.shape, lv, rng)
            Y = RM.demod(x, psi)
            sig2 = float(np.mean(RM.phasor_noise_var(lv, fm))) / T
            for scalar in (False, True):
                lab = 'scalar' if scalar else 'triax'
                ok = RM.tracking_ok(Y0, fm, lv, NRM if scalar else None)
                ks = np.flatnonzero(ok)
                if len(ks) == 0:
                    row[(name, lab)] = None; continue
                Ys = [np.einsum('na,na->n', NRM, Y[k]) if scalar else Y[k].ravel() for k in ks]
                pf = fit(res_free, (Ys, scalar)); pk = fit(res_known, (Ys, ks, att, scalar))
                cr = crlb_free(pos, [Mtrue[k] for k in ks], sig2, scalar)
                row[(name, lab)] = dict(free=pf, known=pk, crlb=cr, ntrack=len(ks))
        out.append(row)
    return out

def summarize(rows, name, lab, which):
    e, fails = [], 0
    for r in rows:
        v = r[(name, lab)]
        if v is None:
            fails += 1; e.append(np.linalg.norm(CTR[:2] - r['pos'][:2])); continue      # failure -> array centre
        e.append(np.linalg.norm(v[which][:2] - r['pos'][:2]))
    e = np.array(e)
    return dict(median_m=float(np.median(e)), p90_m=float(np.percentile(e, 90)), rmse_m=float(np.sqrt(np.mean(e ** 2))),
                fail_fraction=fails / len(rows))

def main():
    rng = np.random.default_rng(11)
    grid = np.linspace(0.5, 4.5, 8)
    poses = [(np.array([x, y, rng.uniform(0.5, 2.0)]), np.r_[rng.uniform(-np.pi / 6, np.pi / 6, 2), rng.uniform(-np.pi, np.pi)])
             for x in grid for y in grid for _ in range(4)]
    res = {}
    for T in TS:
        chunks = [(100 + i + int(1000 * T), T, poses[i::22]) for i in range(22)]
        with ProcessPoolExecutor(max_workers=22) as ex:
            rows = [r for part in ex.map(one, chunks) for r in part]
        for name in LEVELS:
            for lab in ('triax', 'scalar'):
                for which in ('known', 'free'):
                    s = summarize(rows, name, lab, which); key = f'T={T}|{name}|{lab}|{which}'; res[key] = s
                    print(key, {k: round(v, 3) for k, v in s.items()}, flush=True)
                cr = [r[(name, lab)]['crlb'] for r in rows if r[(name, lab)] is not None]
                cr = np.array(cr); fin = np.isfinite(cr)
                res[f'T={T}|{name}|{lab}|crlb_free'] = dict(median_m=float(np.median(cr[fin])) if fin.any() else None, finite_fraction=float(fin.mean()) if len(cr) else 0.0)
                print(f'T={T}|{name}|{lab}|crlb_free', res[f'T={T}|{name}|{lab}|crlb_free'], flush=True)
    res['centre_baseline_median_m'] = float(np.median([np.linalg.norm(CTR[:2] - p[0][:2]) for p in poses]))
    save_json(OUT / 'position.json', res)
    provenance(OUT, 'study11', dict(levels={k: v for k, v in LEVELS.items()}, T_s=TS, poses=len(poses), nodes=NODES.tolist(),
                                    estimators='known-attitude gain projection; attitude-free complex-moment projection', crlb='attitude-free'))

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

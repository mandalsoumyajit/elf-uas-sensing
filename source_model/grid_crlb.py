"""CRLB for the measured 4-antenna grid with the rotating-moment model used in the measured fit (V2).

m = M e^{j phi} [ (cos psi, sin psi, 0) + j s (-sin psi, cos psi, 0) + beta z ],  beta complex, s = +-1 (known)
Unknowns: x, y (z known), psi, M, phi, Re beta, Im beta -> 7 real; 4 complex measurements = 8 real.
SNR per antenna scaled to the measured 1 s coherent line SNRs. Compare with the measured 0.35 m median error.
"""
import os
import numpy as np
import array_model as AM

HERE = os.path.dirname(os.path.abspath(__file__))
A = np.load(os.path.join(HERE, '..', 'grid_analysis', 'complex_amps.npy'), allow_pickle=True).item()
corners = np.array([[-0.1, 5.1, 1.0], [5.1, 5.1, 1.0], [5.1, -0.1, 1.0], [-0.1, -0.1, 1.0]])
ctr = np.array([2.5, 2.5, 1.0]); nrm = ctr - corners; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

def amps(p, s):
    x, y, psi, M, phi, br, bi = p
    m = M * np.exp(1j * phi) * (np.array([np.cos(psi), np.sin(psi), 0]) + 1j * s * np.array([-np.sin(psi), np.cos(psi), 0])
                                + (br + 1j * bi) * np.array([0, 0, 1.0]))
    D = AM.dipole_tensor(corners - np.array([x, y, 1.5])[None, :])        # drone 0.5 m above loop centres (cf. V2 fit)
    return np.einsum('ni,nij,j->n', nrm, D, m)

def crlb(p, s, sigma2):
    FREE = (0, 1, 3, 4) if os.environ.get("BETA_KNOWN") == "1" else (0, 1, 3, 4, 5, 6)   # yaw degenerate with phase
    J = np.empty((4, len(FREE)), complex)
    for j, k in enumerate(FREE):
        h = 1e-5 * max(1.0, abs(p[k])); e = np.zeros(7); e[k] = h
        J[:, j] = (amps(p + e, s) - amps(p - e, s)) / (2 * h)
    sc = np.linalg.norm(J, axis=0); sc[sc == 0] = 1.0
    Jn = J / sc                                            # column-normalised (scale-free conditioning)
    In = 2 / sigma2 * np.real(Jn.conj().T @ Jn)
    if np.linalg.cond(In) > 1e12:
        return np.inf
    C = np.linalg.inv(In) / np.outer(sc, sc); return float(np.sqrt(C[0, 0] + C[1, 1]))

res = []
for cell in range(1, 26):
    pos = ((cell - 1) % 5 + 0.5, 4.5 - (cell - 1) // 5)
    meas = np.mean(np.abs(A[cell]['w1']) ** 2, axis=0) / A[cell]['nv1'] - 1          # measured 1 s coherent SNR (linear)
    meas = np.maximum(meas, 1e-3)
    rng = np.random.default_rng(cell); e = []
    for _ in range(40):
        p = np.array([pos[0], pos[1], rng.uniform(0, 2 * np.pi), 1.0, 0.2, rng.normal(0, 1), rng.normal(0, 1)])
        s = rng.choice([-1, 1])
        a = amps(p, s)
        sigma2 = np.exp(np.mean(np.log(np.abs(a) ** 2 / meas)))                       # common noise level matching measured SNRs
        e.append(crlb(p, s, sigma2))
    e = np.array(e); res.append(np.median(e[np.isfinite(e)]) if np.isfinite(e).any() else np.inf)
res = np.array(res); fin = np.isfinite(res)
print(f'Grid CRLB per 1 s window (rotating-moment model, z known): median {np.median(res[fin]):.3f} m, '
      f'90th pct {np.percentile(res[fin], 90):.3f} m, finite {fin.sum()}/25 cells')
print('per cell (m):', np.round(res, 3))
print('Measured inversion error (V2, 1 s windows): median 0.35 m -> ratio measured/CRLB =', round(0.35 / np.median(res[fin]), 1))

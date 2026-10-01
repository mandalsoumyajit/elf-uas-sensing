"""Ground effect as a MODEL ERROR: localisation bias when the processor inverts with a free-space dipole model
but the data contain the ground's secondary field, compared with the noise-limited CRLB (2 s window).

Unknowns (as in array_model_loc): position r (3), real moment vector m (3), common phase phi -> 7.
Data: complex line amplitudes on triaxial nodes within 40 m, noise-free, ground-aware forward model (rho_g).
Fit : weighted least squares (weights 1/sigma per channel), free-space model, started at the truth (best case).
Layouts: street canyon 'mixed' and perimeter 'poles+roofs', B = 16 nodes per 100 m.
"""
import os, sys
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import site_model as SM

def amps(th, nodes, G):
    r, m, ph = th[:3], th[3:6], th[6]
    return np.exp(1j * ph) * np.einsum('nij,j->ni', G(r[None], nodes)[0], m).ravel()

def crlb(th, nodes, G, sig):
    J = np.empty((3 * len(nodes), 7), complex)
    for k in range(7):
        h = 1e-4 if k < 3 or k == 6 else 1e-4 * np.linalg.norm(th[3:6]); e = np.zeros(7); e[k] = h
        J[:, k] = (amps(th + e, nodes, G) - amps(th - e, nodes, G)) / (2 * h)
    Jw = J / sig; I = 2 * np.real(Jw.conj().T @ Jw)
    if np.linalg.cond(I) > 1e14:
        return np.inf
    return float(np.sqrt(np.trace(np.linalg.inv(I)[:3, :3])))

def one(args):
    case, proc, layout, rho_g, zb, seed = args
    m0, f = SM.line(case, proc); f = SM.FUND_F[case] if proc != 'PWM' else f
    n = SM.noise(f); T = 2.0; sig = np.sqrt(2 * n ** 2 / T)
    Gt, Gf = SM.Green(f, rho_g), SM.Green(f, None)
    rng = np.random.default_rng(seed); out = []
    for _ in range(40):
        if layout == 'canyon':
            Bld = SM.buildings(rng, 160.0, (-1, 1)); nodes = SM.canyon_nodes('mixed', 16, Bld, rng)
            r = np.array([rng.uniform(-7, 7), rng.uniform(-10, 10), rng.uniform(*zb)])
        else:
            nodes = SM.perimeter_nodes('poles+roofs', 16, rng)
            r = np.array([rng.uniform(-8, 8), rng.uniform(-10, 10), rng.uniform(*zb)])
        nodes = nodes[np.linalg.norm(nodes - r, axis=1) < 40]
        u = rng.normal(size=3); th = np.r_[r, m0 * u / np.linalg.norm(u), 0.4]
        d = amps(th, nodes, Gt)
        snr = np.sum(np.abs(d) ** 2) / sig ** 2
        if snr < 30:                     # only windows where localisation is meaningful (coherent SNR >= ~15 dB)
            continue
        sc = np.r_[1, 1, 1, [m0] * 3, 1]
        res = lambda p: np.r_[((amps(p * sc, nodes, Gf) - d) / sig).real, ((amps(p * sc, nodes, Gf) - d) / sig).imag]
        fit = least_squares(res, th / sc, x_scale='jac', max_nfev=400)
        bias = np.linalg.norm(fit.x[:3] - r)
        chi2 = 2 * fit.cost                                  # misfit (noise-free) vs expected 2N - 7 with noise
        out.append((bias, crlb(th, nodes, Gt, sig), chi2 / (2 * d.size - 7), 10 * np.log10(snr)))
    return args, np.array(out)

if __name__ == '__main__':
    jobs = []
    for case, proc in (('DJI-like', 'PWM'), ('5-inch FPV', 'PWM'), ('heavy-lift', 'PWM'), ('5-inch FPV', 'P3'), ('heavy-lift', 'P3')):
        for layout, zbs in (('canyon', ((3, 9), (26, 40))), ('perimeter', ((5, 15), (15, 30)))):
            for rho_g in (1000.0, 100.0, 30.0):
                for zb in zbs:
                    jobs.append((case, proc, layout, rho_g, zb, hash((case, proc, layout, zb)) % 10000))
    with ProcessPoolExecutor(max_workers=20) as ex:
        R = list(ex.map(one, jobs))
    lines = ['Free-space inversion of ground-affected data: position bias vs CRLB (2 s window, triaxial, B = 16/100 m), '
             'windows with coherent array SNR >= 15 dB.  misfit = chi2/dof of the noise-free residual (1 = noise level).']
    for (case, proc, layout, rho_g, zb, _), o in R:
        if len(o) == 0:
            lines.append(f'{case:10s} {proc:3s} {layout:9s} alt {zb[0]:2d}-{zb[1]:2d} m rho_g {rho_g:5.0f}: no windows with SNR >= 15 dB'); continue
        lines.append(f'{case:10s} {proc:3s} {layout:9s} alt {zb[0]:2d}-{zb[1]:2d} m rho_g {rho_g:5.0f}: n={len(o):2d} '
                     f'bias median {np.median(o[:,0]):6.3f} m (90% {np.percentile(o[:,0],90):6.3f}) | CRLB median {np.median(o[:,1]):6.3f} m | '
                     f'bias/CRLB median {np.median(o[:,0]/o[:,1]):6.2f} | misfit chi2/dof median {np.median(o[:,2]):8.2f} | SNR median {np.median(o[:,3]):4.1f} dB')
    print('\n'.join(lines))
    open(os.path.join(HERE, 'ground_bias_out.txt'), 'w').write('\n'.join(lines) + '\n')
